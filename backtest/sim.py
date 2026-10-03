"""Simulator v2: one event-driven engine for AuTrad and every baseline, on real data.

Per bar t (union calendar across NSE, crypto and FX):
  1. orders placed at t-1 fill at t's open (spread, impact, fees, participation caps)
  2. unfilled remainders expire; idle cash earns `cash_rate` (liquid-fund assumption)
  3. mark to market; kill-switch cooldown handling
  4. strategy sees data <= t and returns target weights
  5. orders sized from the difference, skipped inside the no-trade band
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from argo import build_features
from execution.broker import Order, PaperBroker
from execution.costs import PROFILES, CostProfile
from risk.gate import PortfolioState, RiskGate, RiskLimits

from .ledger import build_ledger
from .strategies import BarContext, Strategy


@dataclass
class SimConfig:
    cash: float = 10_00_000.0          # Rs 10 lakh
    cash_rate: float = 0.06            # idle cash yield p.a. (parked in a liquid/overnight fund)
    min_trade_frac: float = 0.0025     # no-trade band: ignore rebalances < 0.25% of equity
    kill_cooldown_bars: int = 20       # after a kill: flat for N bars, then reset (models a human review)
    limits: RiskLimits = field(default_factory=RiskLimits)
    zero_costs: bool = False


@dataclass
class SimResult:
    name: str
    equity: pd.Series
    costs: pd.Series                   # cumulative fees + slippage, Rs
    fees: float
    slippage: float
    exposure: pd.Series                # gross exposure / equity
    weights: pd.DataFrame
    ledger: pd.DataFrame
    turnover: float                    # sum |traded notional| / average equity
    kill_events: list[str]
    violations: int
    log: list[dict] = field(default_factory=list)

    @property
    def gross_equity(self) -> pd.Series:
        return self.equity + self.costs


class Market:
    """Pre-computed, point-in-time market data shared by every simulation run."""

    def __init__(self, data: dict[str, pd.DataFrame], sectors: dict[str, str], categories: dict[str, str],
                 profiles: dict[str, CostProfile]) -> None:
        self.data, self.sectors, self.categories, self.profiles = data, sectors, categories, profiles
        self.symbols = sorted(data)
        self.dates = sorted(set().union(*[d.index for d in data.values()]))
        feats = {s: build_features(d) for s, d in data.items()}
        self.rows: dict[pd.Timestamp, dict[str, dict]] = {}
        for s in self.symbols:
            f = feats[s].drop(columns=["y_next"]).astype(float)
            for t, rec in zip(f.index, f.to_dict("records"), strict=True):
                self.rows.setdefault(t, {})[s] = rec
        self.bars = {s: d[["open", "high", "low", "close", "volume"]].to_dict("index") for s, d in data.items()}
        self.close_arr = {s: d["close"].to_numpy() for s, d in data.items()}
        self.pos = {s: {t: k for k, t in enumerate(d.index)} for s, d in data.items()}


def simulate(market: Market, strategy: Strategy, cfg: SimConfig | None = None,
             start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> SimResult:
    cfg = cfg or SimConfig()
    profiles = {s: PROFILES["zero"] for s in market.symbols} if cfg.zero_costs else market.profiles
    broker = PaperBroker(cfg.cash, profiles=profiles)
    gate = RiskGate(cfg.limits)
    strategy.bind(gate)
    dates = [t for t in market.dates if (start is None or t >= start) and (end is None or t <= end)]
    last: dict[str, float] = {}
    eq_s, cost_s, expo_s, w_rows = [], [], [], []
    fills: list[tuple[pd.Timestamp, object]] = []
    traded = 0.0
    peak = sod = cfg.cash
    cooldown, kills, violations = 0, [], 0
    prev_t: pd.Timestamp | None = None
    prev_w: dict[str, float] = {}

    for i, t in enumerate(dates):
        bars_t = market.rows.get(t, {})
        # 1) fills at this bar's open for orders placed last bar
        for s in bars_t:
            b = market.bars[s][t]
            vol = bars_t[s].get("vol20")
            for f in broker.on_bar(s, b["open"], b["high"], b["low"], b["volume"], 0.01 if not np.isfinite(vol) else vol):
                fills.append((t, f))
                traded += abs(f.qty * f.price)
            last[s] = b["close"]
        broker.cancel_all()
        # 2) idle cash yield
        if prev_t is not None and broker.cash > 0 and cfg.cash_rate:
            broker.cash *= (1 + cfg.cash_rate) ** ((t - prev_t).days / 365.0)
        prev_t = t
        # 3) mark to market
        pos = broker.get_positions()
        equity = broker.cash + sum(q * last.get(s, 0.0) for s, q in pos.items())
        weights = {s: q * last[s] / equity for s, q in pos.items()} if equity > 0 else {}
        strategy.on_weights(prev_w, weights, last)
        if gate.kill.active:
            cooldown += 1
            if cooldown >= cfg.kill_cooldown_bars and not pos:
                gate.kill.reset()
                peak, cooldown = equity, 0     # human review done: drawdown re-based
        peak = max(peak, equity)
        eq_s.append(equity)
        cost_s.append(broker.fees_paid + broker.slippage_paid)
        expo_s.append(sum(abs(w) for w in weights.values()))
        w_rows.append(weights)
        pf = PortfolioState(equity, sod, peak, dict(weights), market.sectors)
        # 4) strategy decides with data <= t
        if bars_t and not gate.kill.active:
            hist = {s: market.close_arr[s][: market.pos[s][t] + 1] for s in bars_t}
            ctx = BarContext(i, t, bars_t, {s: last[s] for s in bars_t},
                             {s: market.bars[s][t]["volume"] for s in bars_t}, dict(weights), pf, hist, len(market.symbols))
            targets = strategy.targets(ctx)
            was_killed = gate.kill.active
            for s, w in targets.items():
                if s not in bars_t:
                    continue
                cur = weights.get(s, 0.0)
                if strategy.gated and abs(w) > cfg.limits.max_position + 1e-9:
                    violations += 1               # invariant: gated strategies never exceed the cap
                if abs(w - cur) < cfg.min_trade_frac and w != 0.0:
                    continue
                if abs(w - cur) < 1e-9:
                    continue
                broker.place_order(Order(s, (w - cur) * equity / last[s]))
            if gate.kill.active and not was_killed:
                kills.append(f"{t.date()}: {gate.kill.reason}")
        # 5) kill switch: flatten everything next bar
        if gate.kill.active:
            broker.cancel_all()
            for s, q in broker.get_positions().items():
                broker.place_order(Order(s, -q))
        sod = equity
        prev_w = weights

    idx = pd.DatetimeIndex(dates)
    equity = pd.Series(eq_s, idx, name=strategy.name)
    ledger = build_ledger(fills, last, market.categories, dates[-1] if dates else pd.Timestamp.now())
    avg_eq = float(np.mean(eq_s)) if eq_s else 1.0
    return SimResult(strategy.name, equity, pd.Series(cost_s, idx), broker.fees_paid, broker.slippage_paid,
                     pd.Series(expo_s, idx), pd.DataFrame(w_rows, index=idx).fillna(0.0), ledger,
                     traded / avg_eq, kills, violations, getattr(strategy, "log", []))

