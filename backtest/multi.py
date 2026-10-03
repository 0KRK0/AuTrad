"""Multi-agent backtest: category agents + events + manual overrides, same broker/gate/costs as engine.py."""
from __future__ import annotations

import pandas as pd

from agents.events import Event
from agents.override import OverrideBook
from agents.system import DecisionAgent
from agents.taxonomy import Instrument
from argo import build_features
from audit.log import AuditLog
from execution.broker import CostModel, Order, PaperBroker
from risk.gate import PortfolioState, RiskGate, RiskLimits

from .engine import metrics


def run_multi_agent(data: dict[str, pd.DataFrame], instruments: dict[str, Instrument], events: list[Event] | None = None,
                    overrides: OverrideBook | None = None, cash: float = 1_000_000.0, limits: RiskLimits | None = None,
                    costs: CostModel | None = None, audit: AuditLog | None = None,
                    schedule: dict[pd.Timestamp, list] | None = None) -> dict:
    """schedule: optional {timestamp: [Override, ...]} to simulate you stepping in mid-run."""
    audit = audit or AuditLog()
    feats = {s: build_features(df) for s, df in data.items()}
    gate = RiskGate(limits)
    brain = DecisionAgent(instruments, gate, overrides)
    broker = PaperBroker(cash, costs)
    by_ts: dict[pd.Timestamp, list[Event]] = {}
    for e in events or []:
        by_ts.setdefault(pd.Timestamp(e.ts).normalize(), []).append(e)
    dates = sorted(set().union(*[df.index for df in data.values()]))
    curve, log, peak, sod, last = [], [], cash, cash, {}
    prev_w: dict[str, float] = {}

    for i, t in enumerate(dates):
        for s, df in data.items():
            if t in df.index:
                b = df.loc[t]
                v = feats[s].at[t, "vol20"]
                broker.on_bar(s, b.open, b.high, b.low, b.volume, 0.01 if pd.isna(v) else float(v))
                last[s] = float(b.close)
        broker.cancel_all()
        pos = broker.get_positions()
        eq = broker.cash + sum(q * last.get(s, 0.0) for s, q in pos.items())
        peak = max(peak, eq)
        w = {s: q * last[s] / eq for s, q in pos.items()} if eq > 0 else {}
        brain.on_fills(last, prev_w, w)
        curve.append((t, eq))

        rows = {s: feats[s].loc[t].to_dict() for s, df in data.items() if t in df.index}
        brain.learn({s: float(r["r1"]) for s, r in rows.items()})
        for o in (schedule or {}).get(t, []):
            brain.overrides.add(o)
        pf = PortfolioState(eq, sod, peak, dict(w))
        vols = {s: float(data[s].at[t, "volume"]) for s in rows}
        decisions = brain.step(float(i) * 86400, rows, {s: last[s] for s in rows}, vols,
                               by_ts.get(pd.Timestamp(t).normalize(), []), pf, ts=t)
        for d in decisions:
            log.append({"ts": str(t.date()), **d.__dict__})
            audit.write("decision", {"ts": t.isoformat(), **d.__dict__})
            if d.ok and abs(d.approved - w.get(d.symbol, 0.0)) >= 1e-4:
                broker.place_order(Order(d.symbol, (d.approved - w.get(d.symbol, 0.0)) * eq / last[d.symbol]))
        if gate.kill.active:
            for s, q in broker.get_positions().items():
                broker.place_order(Order(s, -q))
        prev_w, sod = w, eq

    eq_s = pd.Series(dict(curve)).astype(float)
    return {"metrics": metrics(eq_s.iloc[60:]), "decisions": log, "equity": eq_s, "agents": brain.status(),
            "kill_switch": gate.kill.active}
