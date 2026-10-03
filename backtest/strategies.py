"""Strategies for simulator v2: AuTrad (ARGO agents) and the baselines it must beat.

A strategy sees only data up to and including the current bar (BarContext) and returns
target portfolio weights. Execution happens at the NEXT bar's open inside the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents.allocator import CategoryAllocator
from agents.system import DecisionAgent
from agents.taxonomy import Instrument
from argo import ArgoConfig
from risk.gate import PortfolioState, RiskGate


@dataclass
class BarContext:
    i: int
    t: pd.Timestamp
    rows: dict[str, dict]              # point-in-time features per symbol with a bar at t
    prices: dict[str, float]           # close at t
    volumes: dict[str, float]
    weights: dict[str, float]          # current portfolio weights
    pf: PortfolioState
    history: dict[str, np.ndarray]     # closes up to and including t, per symbol
    n_universe: int


class Strategy:
    name = "strategy"
    gated = False                      # True: strategy applies the Risk Gate itself

    def bind(self, gate: RiskGate) -> None:
        self.gate = gate

    def targets(self, ctx: BarContext) -> dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def on_weights(self, old: dict[str, float], new: dict[str, float], prices: dict[str, float]) -> None:
        """Called after fills with realised weights (lets agents arm stops at true entry prices)."""


class ArgoAgents(Strategy):
    """The AuTrad system: category agents + ARGO models + allocator + Risk Gate."""

    gated = True

    def __init__(self, instruments: dict[str, Instrument], cfg: ArgoConfig | None = None,
                 cost_bps: dict[str, float] | None = None, use_allocator: bool = True, name: str = "autrad",
                 decide_every: int = 1, band: float = 0.0) -> None:
        """decide_every: rebalance entries/resizes every N bars (exits and stops act any bar).
        band: skip resizes smaller than this fraction of the position (hysteresis against churn)."""
        self.name = name
        self.instruments, self.cfg, self.cost_bps, self.use_allocator = instruments, cfg or ArgoConfig(), cost_bps, use_allocator
        self.decide_every, self.band = max(1, decide_every), band

    def bind(self, gate: RiskGate) -> None:
        self.gate = gate
        self.brain = DecisionAgent(self.instruments, gate, argo_cfg=self.cfg, cost_bps=self.cost_bps,
                                   allocator=CategoryAllocator() if self.use_allocator else None)
        self.log: list[dict] = []

    def targets(self, ctx: BarContext) -> dict[str, float]:
        self.brain.learn({s: float(r["r1"]) for s, r in ctx.rows.items()}, ctx.weights)
        decisions = self.brain.step(float(ctx.i) * 86400, ctx.rows, ctx.prices, ctx.volumes, [], ctx.pf, ts=ctx.t)
        decision_day = ctx.i % self.decide_every == 0
        out = {}
        for d in decisions:
            if d.ok:
                cur, w = ctx.weights.get(d.symbol, 0.0), d.approved
                exiting = abs(w) < 1e-12 or (w * cur < 0)
                if not exiting and not decision_day:
                    continue
                if not exiting and abs(cur) > 1e-12 and abs(w - cur) < self.band * max(abs(w), abs(cur)):
                    continue
                out[d.symbol] = w
            self.log.append({"t": ctx.t, "symbol": d.symbol, "category": d.category, "proposed": d.proposed,
                             "approved": d.approved, "ok": d.ok, "reason": "; ".join(d.reasons)[:160]})
        return out

    def on_weights(self, old, new, prices) -> None:
        self.brain.on_fills(prices, old, new)


class BuyHold(Strategy):
    """Buy each asset once at 1/N when it first trades; never rebalance."""

    name = "buy_hold"

    def __init__(self) -> None:
        self.bought: set[str] = set()

    def targets(self, ctx: BarContext) -> dict[str, float]:
        out = {}
        for s in ctx.rows:
            if s not in self.bought:
                self.bought.add(s)
                out[s] = 1.0 / ctx.n_universe
        return out


class _Periodic(Strategy):
    every = 21

    def __init__(self) -> None:
        self.last: dict[str, int] = {}
        self.seen: dict[str, int] = {}

    def due(self, s: str) -> bool:
        self.seen[s] = self.seen.get(s, 0) + 1
        if self.seen[s] - self.last.get(s, -10**9) >= self.every:
            self.last[s] = self.seen[s]
            return True
        return False


class Momentum(_Periodic):
    """Time-series momentum: long 1/N if the 6-month return (skipping the last week) is positive; monthly."""

    name = "momentum_6m"
    lookback, skip = 126, 5

    def targets(self, ctx: BarContext) -> dict[str, float]:
        out = {}
        for s in ctx.rows:
            if not self.due(s):
                continue
            h = ctx.history[s]
            if len(h) <= self.lookback:
                continue
            r = h[-1 - self.skip] / h[-1 - self.lookback] - 1
            out[s] = 1.0 / ctx.n_universe if r > 0 else 0.0
        return out


class MACross(_Periodic):
    """Long 1/N while the 50-day average is above the 200-day average; checked weekly."""

    name = "ma_50_200"
    every = 5

    def targets(self, ctx: BarContext) -> dict[str, float]:
        out = {}
        for s in ctx.rows:
            if not self.due(s):
                continue
            h = ctx.history[s]
            if len(h) < 200:
                continue
            out[s] = 1.0 / ctx.n_universe if h[-50:].mean() > h[-200:].mean() else 0.0
        return out


@dataclass
class RandomTrader(Strategy):
    """Null model: random entries/exits with the SAME position size, exposure and holding period as AuTrad.

    If AuTrad cannot beat most random traders with identical risk-taking, its signals carry no edge.
    """

    seed: int
    position_size: float
    target_exposure: float
    hold_bars: float
    name: str = "random"
    rng: np.random.Generator = field(init=False)
    held: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def targets(self, ctx: BarContext) -> dict[str, float]:
        frac = min(0.95, self.target_exposure / max(self.position_size * ctx.n_universe, 1e-9))
        q = 1.0 / max(self.hold_bars, 1.0)
        lam = frac * q / max(1 - frac, 1e-9)
        out = {}
        for s in ctx.rows:
            if self.held.get(s):
                if self.rng.random() < q:
                    self.held[s] = False
                    out[s] = 0.0
            elif self.rng.random() < lam:
                self.held[s] = True
                out[s] = self.position_size
        return out
