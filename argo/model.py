"""ARGO: Adaptive Regime-Gated Online learner — the AuTrad AI model.

predict(x, ctx) -> Decision  (for bar t, using data <= t)
learn(y)                     (after bar t+1 closes; y = realized log return t -> t+1)

horizon h: experts predict the mean daily return over the next h bars. Each prediction is
scored only once all h returns are known, so a 20-bar trend call is judged on 20 bars, not 1.
h = 1 is the original daily behaviour.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .experts import Expert, default_experts
from .features import FEATURES
from .meta import REGIMES, PageHinkley, ThompsonAllocator, classify_regime


@dataclass
class Decision:
    symbol: str
    regime: str
    mu: float                  # blended expected next-bar log return
    sigma: float               # current per-bar volatility
    target_weight: float       # proposed portfolio weight before the Risk Gate
    expert_weights: dict[str, float]
    expert_preds: dict[str, float]
    status: str                # ACTIVE | PAPER_ONLY (after drift) | WARMUP
    reason: str = ""


@dataclass
class ArgoConfig:
    kelly_fraction: float = 0.25
    max_weight: float = 0.05
    cost_bps: float = 5.0            # round-trip cost estimate per unit turnover, bps
    edge_multiple: float = 1.5       # trade only if |mu| > edge_multiple * cost
    warmup_bars: int = 60
    quarantine_bars: int = 20        # paper-only bars after a drift alarm
    drift_delta: float = 0.25
    drift_threshold: float = 40.0
    horizon: int = 1                 # bars each prediction is judged over


@dataclass
class ArgoModel:
    symbol: str
    cfg: ArgoConfig = field(default_factory=ArgoConfig)
    experts: list[Expert] = field(default_factory=default_experts)

    def __post_init__(self) -> None:
        self.alloc = ThompsonAllocator(n_experts=len(self.experts))
        self.drift = PageHinkley(delta=self.cfg.drift_delta, threshold=self.cfg.drift_threshold)
        self._queue: deque[list] = deque()   # [x, regime, preds, sigma, mu, sum_y, n]
        self.bars_seen = 0
        self.quarantine = 0
        self.drift_events = 0

    @property
    def cost(self) -> float:
        return self.cfg.cost_bps / 1e4

    def predict(self, row: dict[str, float]) -> Decision:
        x = np.array([row.get(f, np.nan) for f in FEATURES], dtype=float)
        x = np.nan_to_num(x, nan=0.0)
        sigma = float(row.get("vol20") or 0.0) or 0.01
        r = classify_regime(row.get("ema_slope", np.nan), row.get("vol20", np.nan), row.get("vol_long", np.nan))
        preds = np.array([e.predict(x) for e in self.experts])
        w = self.alloc.weights(r)
        mu = float(w @ preds)
        self._queue.append([x, r, preds, sigma, mu, 0.0, 0])
        self.bars_seen += 1
        h = self.cfg.horizon

        status, reason, tw = "ACTIVE", "", 0.0
        if self.bars_seen <= self.cfg.warmup_bars:
            status, reason = "WARMUP", "learning; no capital"
        elif self.quarantine > 0:
            status, reason = "PAPER_ONLY", f"drift quarantine {self.quarantine} bars left"
        elif abs(mu) * h <= self.cfg.edge_multiple * self.cost:
            reason = "edge below cost threshold"
        else:
            edge = np.sign(mu) * (abs(mu) - self.cost / h)
            tw = float(np.clip(self.cfg.kelly_fraction * edge / sigma**2, -self.cfg.max_weight, self.cfg.max_weight))
        names = [e.name for e in self.experts]
        return Decision(
            symbol=self.symbol,
            regime=REGIMES[r],
            mu=mu,
            sigma=sigma,
            target_weight=tw,
            expert_weights=dict(zip(names, map(float, w), strict=False)),
            expert_preds=dict(zip(names, map(float, preds), strict=False)),
            status=status,
            reason=reason,
        )

    def learn(self, y: float) -> None:
        if not self._queue or not np.isfinite(y):
            return
        if self.quarantine > 0:
            self.quarantine -= 1
        h = self.cfg.horizon
        for item in self._queue:
            item[5] += y
            item[6] += 1
        while self._queue and self._queue[0][6] >= h:
            x, r, preds, sigma, mu, acc, _ = self._queue.popleft()
            y_h = acc / h                                   # realised mean daily return over the horizon
            # Counterfactual reward for every expert: sign-trade PnL over the horizon after cost, vol-normalised
            traded = np.abs(preds) * h > self.cfg.edge_multiple * self.cost
            rewards = np.where(traded, np.sign(preds) * y_h * h - self.cost, 0.0) / max(sigma * np.sqrt(h), 1e-4)
            self.alloc.update(r, rewards)
            for e in self.experts:
                e.update(x, y_h)
            z = (y_h - mu) / max(sigma / np.sqrt(h), 1e-4)
            if self.bars_seen > self.cfg.warmup_bars and self.drift.update(z**2):
                self.drift_events += 1
                self.quarantine = self.cfg.quarantine_bars
