"""Category agents + the Decision agent that coordinates them.

   evidence / tweets / speeches ─┐
   market bars ── ShockDetector ─┼─> events ─> ImpactGraph ─> category tilts
                                 │
   one CategoryAgent per category (own ARGO models, own policy, own stops)
                                 │ proposals
   DecisionAgent: overrides first, pins, pauses, category budgets, speculative sleeve
                                 │ intents
                              Risk Gate ─> broker
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from argo import ArgoConfig, ArgoModel
from risk.gate import OrderIntent, PortfolioState, RiskGate

from .allocator import CategoryAllocator
from .events import Event, EventType, ImpactGraph, ShockDetector
from .override import OverrideBook
from .taxonomy import POLICIES, SPECULATIVE, SPECULATIVE_SLEEVE, Category, CategoryPolicy, Instrument, classify


@dataclass
class Proposal:
    symbol: str
    category: Category
    weight: float
    reason: str
    status: str = "ACTIVE"
    confidence: float = 0.0


@dataclass
class Holding:
    entry: float
    high: float
    low: float
    bars: int = 0


@dataclass
class CategoryAgent:
    category: Category
    policy: CategoryPolicy
    argo_cfg: ArgoConfig = field(default_factory=ArgoConfig)
    cost_bps: dict[str, float] = field(default_factory=dict)   # per-symbol one-way cost used for the edge test

    reentry_cooldown: int = 5                                   # bars blocked after a stop exit

    def __post_init__(self) -> None:
        self.models: dict[str, ArgoModel] = {}
        self.holdings: dict[str, Holding] = {}
        self.cooldown: dict[str, int] = {}

    def model(self, symbol: str) -> ArgoModel:
        if symbol not in self.models:
            cfg = ArgoConfig(**{**self.argo_cfg.__dict__, "max_weight": self.policy.max_position,
                                "cost_bps": self.cost_bps.get(symbol, self.argo_cfg.cost_bps)})
            self.models[symbol] = ArgoModel(symbol, cfg)
        return self.models[symbol]

    def learn(self, symbol: str, y: float) -> None:
        self.model(symbol).learn(y)

    def _stops(self, symbol: str, price: float, cur: float) -> str | None:
        h = self.holdings.get(symbol)
        if h is None or abs(cur) < 1e-9:
            self.holdings.pop(symbol, None)
            return None
        h.bars += 1
        h.high, h.low = max(h.high, price), min(h.low, price)
        p = self.policy
        if cur > 0:
            if price <= h.entry * (1 - p.stop_loss):
                return f"stop-loss {p.stop_loss:.0%}"
            if price <= h.high * (1 - p.trailing_stop):
                return f"trailing stop {p.trailing_stop:.0%}"
        else:
            if price >= h.entry * (1 + p.stop_loss):
                return f"stop-loss {p.stop_loss:.0%}"
            if price >= h.low * (1 + p.trailing_stop):
                return f"trailing stop {p.trailing_stop:.0%}"
        if p.style == "event_momentum" and h.bars >= p.time_stop_bars:
            return f"time stop {p.time_stop_bars} bars"   # speculation must not become investment
        return None

    def propose(self, symbol: str, row: dict, price: float, cur: float, sym_events: list[Event],
                tilt: float, liquidity_cap: float) -> Proposal:
        p = self.policy
        d = self.model(symbol).predict(row)  # always predict so the model keeps learning
        stop = self._stops(symbol, price, cur)
        if stop:
            self.cooldown[symbol] = self.reentry_cooldown
            return Proposal(symbol, self.category, 0.0, stop)
        if self.cooldown.get(symbol, 0) > 0 and abs(cur) < 1e-9:
            self.cooldown[symbol] -= 1
            return Proposal(symbol, self.category, 0.0, "cooldown after stop")

        if p.style == "event_momentum":
            good = [e for e in sym_events if e.sentiment > 0 and (e.type != EventType.UNEXPLAINED_PUMP or p.allow_unexplained)]
            pumps = [e for e in sym_events if e.type == EventType.UNEXPLAINED_PUMP]
            if abs(cur) > 1e-9:
                return Proposal(symbol, self.category, cur, "holding; stops armed")
            if not good and pumps:
                return Proposal(symbol, self.category, 0.0, "unexplained spike (possible pump): skipped")
            shock = any(e.type == EventType.PRICE_SHOCK for e in sym_events)
            if good and (shock or tilt > 0):
                strength = max(e.strength * e.reliability for e in good)
                w = min(p.max_position, p.max_position * (0.5 + strength), liquidity_cap)
                return Proposal(symbol, self.category, w, f"event entry: {good[0].type.value}", confidence=strength)
            return Proposal(symbol, self.category, 0.0, "no confirmed event")

        # "argo" style: learned signal, tilted by what events imply for this category
        mu = d.mu + tilt * p.event_sensitivity
        m = self.model(symbol)
        cost, h = m.cost, m.cfg.horizon
        if abs(mu) * h <= self.argo_cfg.edge_multiple * cost or d.status != "ACTIVE":
            w = 0.0
        else:
            w = self.argo_cfg.kelly_fraction * np.sign(mu) * (abs(mu) - cost / h) / d.sigma**2
        if not p.allow_short:
            w = max(w, 0.0)
        w = float(np.clip(w, -min(p.max_position, liquidity_cap), min(p.max_position, liquidity_cap)))
        why = f"argo mu={d.mu:+.4f} tilt={tilt:+.4f} regime={d.regime}"
        return Proposal(symbol, self.category, w, why, status=d.status, confidence=abs(mu) / max(d.sigma, 1e-6))

    def on_fill(self, symbol: str, price: float, new_weight: float, old_weight: float) -> None:
        if abs(new_weight) < 1e-9:
            self.holdings.pop(symbol, None)
        elif abs(old_weight) < 1e-9 or np.sign(new_weight) != np.sign(old_weight):
            self.holdings[symbol] = Holding(price, price, price)


@dataclass
class Decision:
    symbol: str
    category: str
    source: str
    proposed: float
    approved: float
    ok: bool
    reasons: list[str]
    override_id: int | None = None


class DecisionAgent:
    def __init__(self, instruments: dict[str, Instrument], gate: RiskGate | None = None,
                 overrides: OverrideBook | None = None, sleeve: float = SPECULATIVE_SLEEVE,
                 policies: dict[Category, CategoryPolicy] | None = None, argo_cfg: ArgoConfig | None = None,
                 cost_bps: dict[str, float] | None = None, allocator: CategoryAllocator | None = None) -> None:
        self.instruments = instruments
        self.category_of = {s: classify(i) for s, i in instruments.items()}
        pol = policies or POLICIES
        cfg = argo_cfg or ArgoConfig()
        self.agents = {c: CategoryAgent(c, pol[c], cfg, dict(cost_bps or {})) for c in set(self.category_of.values())}
        self.allocator = allocator
        self.gate = gate or RiskGate()
        self.overrides = overrides or OverrideBook()
        self.sleeve = sleeve
        self.shocks = ShockDetector()
        self.impact = ImpactGraph()
        self._last_events: list[Event] = []

    # ---------- learning ----------
    def learn(self, realized: dict[str, float], weights: dict[str, float] | None = None) -> None:
        for s, y in realized.items():
            if np.isfinite(y):
                self.agents[self.category_of[s]].learn(s, y)
        if self.allocator is not None and weights:
            pnl: dict[Category, float] = {}
            expo: dict[Category, float] = {}
            for s, w in weights.items():
                y = realized.get(s)
                if s in self.category_of and y is not None and np.isfinite(y) and abs(w) > 1e-9:
                    c = self.category_of[s]
                    pnl[c] = pnl.get(c, 0.0) + w * y
                    expo[c] = expo.get(c, 0.0) + abs(w)
            for c, e in expo.items():
                self.allocator.update(c.value, pnl[c] / e)
        by_cat: dict[Category, list[float]] = {}
        for s, y in realized.items():
            if np.isfinite(y):
                by_cat.setdefault(self.category_of[s], []).append(y)
        cat_ret = {c: float(np.mean(v)) for c, v in by_cat.items()}
        for e in self._last_events:
            self.impact.learn(e, cat_ret)

    # ---------- deciding ----------
    def step(self, now: float, rows: dict[str, dict], prices: dict[str, float], volumes: dict[str, float],
             text_events: list[Event], pf: PortfolioState, ts=None) -> list[Decision]:
        events = list(text_events)
        mentioned = {s for e in text_events for s in e.symbols}
        for s, row in rows.items():
            shock = self.shocks.update(s, float(np.nan_to_num(row.get("r1", 0.0))), volumes.get(s, 0.0),
                                       s in mentioned or any(self.category_of[s] in e.categories for e in text_events), ts)
            if shock:
                events.append(shock)
        self._last_events = events
        tilts = self.impact.impact(events)
        decisions: list[Decision] = []

        # 1) Human overrides first
        human_syms: set[str] = set()
        for o in self.overrides.pending_orders(now):
            cur = pf.weights.get(o.symbol, 0.0)
            target = OverrideBook.target_weight(o, cur)
            rd = self.gate.check(OrderIntent(o.symbol, target, source="human", note=o.reason), pf, now=now)
            if rd.approved:
                o.consumed = True
                human_syms.add(o.symbol)
                pf.weights[o.symbol] = rd.weight
            cat = self.category_of.get(o.symbol)
            decisions.append(Decision(o.symbol, cat.value if cat else "unknown", f"human:{o.user}", target, rd.weight,
                                      rd.approved, [o.reason, *rd.reasons], o.id))

        # 2) AI proposals per category agent
        props: list[Proposal] = []
        for s, row in rows.items():
            cat = self.category_of[s]
            if s in human_syms or self.overrides.pinned(s, now):
                continue
            cur = pf.weights.get(s, 0.0)
            if cat in self.overrides.paused and abs(cur) < 1e-9:
                continue
            inst = self.instruments[s]
            pol = self.agents[cat].policy
            liq = pol.max_adv_participation * inst.adv_value / pf.equity if inst.adv_value > 0 and pf.equity > 0 else pol.max_position
            sym_events = [e for e in events if s in e.symbols]
            p = self.agents[cat].propose(s, row, prices[s], cur, sym_events, tilts.get(cat, 0.0), liq)
            if cat in self.overrides.paused:
                p.weight, p.reason = min(p.weight, cur) if cur > 0 else max(p.weight, cur), "category paused: no adds"
            props.append(p)

        # 3) Budgets: per-category cap, then speculative sleeve across penny/alt/meme
        mult = (lambda c: self.allocator.multiplier(c.value)) if self.allocator is not None else (lambda c: 1.0)
        self._scale(props, pf, lambda c: {c}, lambda c: self.agents[c].policy.max_category * mult(c))
        self._scale(props, pf, lambda c: SPECULATIVE if c in SPECULATIVE else set(), lambda c: self.sleeve)

        # 4) Risk Gate on every AI intent
        for p in props:
            cur = pf.weights.get(p.symbol, 0.0)
            if abs(p.weight - cur) < 1e-9:
                if p.reason.startswith("unexplained"):  # surface skipped pumps in the audit trail
                    decisions.append(Decision(p.symbol, p.category.value, "ai", 0.0, cur, False, [p.reason]))
                continue
            rd = self.gate.check(OrderIntent(p.symbol, p.weight, atr_z=float(np.nan_to_num(rows[p.symbol].get("atr_z", 0.0))),
                                             model_status=p.status, max_position=self.agents[p.category].policy.max_position), pf, now=now)
            decisions.append(Decision(p.symbol, p.category.value, "ai", p.weight, rd.weight, rd.approved, [p.reason, *rd.reasons]))
        return decisions

    def _scale(self, props: list[Proposal], pf: PortfolioState, group, cap) -> None:
        seen: set[frozenset] = set()
        for p in props:
            g = frozenset(group(p.category))
            if not g or g in seen:
                continue
            seen.add(g)
            members = [q for q in props if q.category in g]
            proposed = {q.symbol for q in members}
            held_elsewhere = sum(abs(w) for s, w in pf.weights.items() if self.category_of.get(s) in g and s not in proposed)
            total = sum(abs(q.weight) for q in members) + held_elsewhere
            limit = cap(p.category)
            if total > limit + 1e-12:
                room = max(0.0, limit - held_elsewhere)
                k = room / max(sum(abs(q.weight) for q in members), 1e-12)
                for q in members:
                    q.weight *= k
                    q.reason += f" | scaled to budget {limit:.1%}"

    def on_fills(self, prices: dict[str, float], old: dict[str, float], new: dict[str, float]) -> None:
        for s in set(old) | set(new):
            if s in self.category_of and s in prices:
                self.agents[self.category_of[s]].on_fill(s, prices[s], new.get(s, 0.0), old.get(s, 0.0))

    def status(self) -> list[dict]:
        return [{"category": c.value, "style": a.policy.style, "symbols": sorted(a.models), "max_category": a.policy.max_category,
                 "max_position": a.policy.max_position, "paused": c in self.overrides.paused} for c, a in sorted(self.agents.items())]
