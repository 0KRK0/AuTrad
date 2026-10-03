"""Deterministic Risk Gate and Kill Switch. No ML in this module — by design."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLimits:
    max_position: float = 0.05        # per symbol, fraction of equity
    max_sector: float = 0.30
    max_gross: float = 1.00
    daily_loss: float = 0.02          # vs start-of-day equity
    dd_reduce: float = 0.08           # capital-preservation: half size
    dd_paper: float = 0.12            # paper-only
    atr_z_spike: float = 3.0
    max_data_age_s: float = 5.0
    max_orders_per_min: int = 30
    human_max_position: float = 0.10  # manual overrides may go bigger than AI, never unlimited
    largecap_max_position: float = 0.10     # v0.3 policy: NIFTY 50 constituents (liquid large caps)
    diversified_max_position: float = 1.0   # v0.3 policy (needs owner sign-off): broad index / gold ETFs may be a whole sleeve, matching lab NAME_CAP


@dataclass
class PortfolioState:
    equity: float
    sod_equity: float
    peak_equity: float
    weights: dict[str, float] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)

    @property
    def drawdown(self) -> float:
        return max(0.0, 1 - self.equity / self.peak_equity) if self.peak_equity > 0 else 0.0

    @property
    def daily_pnl(self) -> float:
        return self.equity / self.sod_equity - 1 if self.sod_equity > 0 else 0.0


@dataclass
class OrderIntent:
    symbol: str
    target_weight: float
    atr_z: float = 0.0
    data_age_s: float = 0.0
    model_status: str = "ACTIVE"
    source: str = "ai"                   # ai | human (manual override)
    max_position: float | None = None    # tighter per-category cap set by the category agent
    note: str = ""
    instrument_class: str = "default"    # default | largecap | diversified


@dataclass
class RiskDecision:
    approved: bool
    weight: float
    checks: dict[str, bool]
    reasons: list[str]
    mode: str  # NORMAL | PRESERVE | PAPER_ONLY | HALTED | KILLED


class KillSwitch:
    """Process-wide latch. Once tripped, only an explicit (MFA-gated in the API) reset clears it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = False
        self.reason = ""
        self.tripped_at: float | None = None

    def trip(self, reason: str) -> None:
        with self._lock:
            if not self.active:
                self.active, self.reason, self.tripped_at = True, reason, time.time()

    def reset(self) -> None:
        with self._lock:
            self.active, self.reason, self.tripped_at = False, "", None


class RiskGate:
    def __init__(self, limits: RiskLimits | None = None, kill: KillSwitch | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.kill = kill or KillSwitch()
        self._order_times: deque[float] = deque()

    def mode(self, pf: PortfolioState) -> str:
        L = self.limits
        if self.kill.active:
            return "KILLED"
        if pf.drawdown >= L.dd_paper:
            self.kill.trip(f"drawdown {pf.drawdown:.1%} >= {L.dd_paper:.0%}")
            return "KILLED"
        if pf.daily_pnl <= -L.daily_loss:
            return "HALTED"
        if pf.drawdown >= L.dd_reduce:
            return "PRESERVE"
        return "NORMAL"

    def check(self, intent: OrderIntent, pf: PortfolioState, now: float | None = None) -> RiskDecision:
        L = self.limits
        now = time.time() if now is None else now
        reasons: list[str] = []
        checks: dict[str, bool] = {}
        mode = self.mode(pf)
        cur = pf.weights.get(intent.symbol, 0.0)
        w = intent.target_weight
        reducing = abs(w) < abs(cur) and (w * cur >= 0)

        checks["kill_switch"] = mode != "KILLED"
        if mode == "KILLED":
            return RiskDecision(False, 0.0, checks, [f"kill switch: {self.kill.reason}"], mode)

        human = intent.source == "human"
        checks["model_status"] = human or intent.model_status == "ACTIVE"
        if not checks["model_status"] and not reducing:
            reasons.append(f"model {intent.model_status}")
            w = cur if intent.model_status == "WARMUP" else 0.0

        checks["data_fresh"] = intent.data_age_s <= L.max_data_age_s
        if not checks["data_fresh"]:
            return RiskDecision(False, cur, checks, [f"stale data {intent.data_age_s:.1f}s"], mode)

        # Human overrides skip the AI's soft brakes (daily halt, preservation, vol halving, sector)
        # but never the kill switch, stale data, gross exposure, order-rate or the human position cap.
        checks["daily_loss"] = mode != "HALTED"
        if mode == "HALTED" and not reducing and not human:
            reasons.append("daily loss limit: entries halted")
            w = cur

        if mode == "PRESERVE" and not human:
            reasons.append("capital preservation: size halved")
            w *= 0.5

        checks["volatility"] = intent.atr_z <= L.atr_z_spike
        if not checks["volatility"] and not human:
            reasons.append("volatility spike: size halved")
            w *= 0.5

        base = {"largecap": L.largecap_max_position, "diversified": L.diversified_max_position}.get(intent.instrument_class, L.max_position)
        cap = L.human_max_position if human else min(base, intent.max_position or base)
        checks["position_limit"] = abs(w) <= cap + 1e-12
        if not checks["position_limit"]:
            reasons.append(f"resized to {cap:.1%} position cap")
            w = cap if w > 0 else -cap

        sector = pf.sectors.get(intent.symbol, "UNKNOWN")
        sector_exp = sum(abs(v) for s, v in pf.weights.items() if pf.sectors.get(s, "UNKNOWN") == sector and s != intent.symbol)
        checks["sector_limit"] = human or sector_exp + abs(w) <= L.max_sector + 1e-12
        if not checks["sector_limit"]:
            room = max(0.0, L.max_sector - sector_exp)
            reasons.append(f"sector {sector} capped")
            w = room if w > 0 else -room

        gross_other = sum(abs(v) for s, v in pf.weights.items() if s != intent.symbol)
        checks["gross_exposure"] = gross_other + abs(w) <= L.max_gross + 1e-12
        if not checks["gross_exposure"]:
            room = max(0.0, L.max_gross - gross_other)
            reasons.append("gross exposure capped")
            w = room if w > 0 else -room

        # Order-rate circuit breaker (only counts orders that change the book)
        if abs(w - cur) > 1e-9:
            while self._order_times and now - self._order_times[0] > 60:
                self._order_times.popleft()
            self._order_times.append(now)
            if len(self._order_times) > L.max_orders_per_min:
                self.kill.trip("order rate exceeded")
                checks["order_rate"] = False
                return RiskDecision(False, cur, checks, ["order rate breaker tripped kill switch"], "KILLED")
        checks["order_rate"] = True

        return RiskDecision(True, float(w), checks, reasons, mode)
