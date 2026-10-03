"""Event-loop backtester: ARGO -> Risk Gate -> PaperBroker, with costs, latency, partial fills, audit."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from argo import ArgoConfig, ArgoModel, build_features
from audit.log import AuditLog
from execution.broker import CostModel, Order, PaperBroker
from risk.gate import OrderIntent, PortfolioState, RiskGate, RiskLimits

BARS_PER_YEAR = 252


@dataclass
class BacktestConfig:
    cash: float = 100_000.0
    min_trade_frac: float = 0.002
    argo: ArgoConfig = field(default_factory=ArgoConfig)
    limits: RiskLimits = field(default_factory=RiskLimits)
    costs: CostModel = field(default_factory=CostModel)
    folds: int = 4


def metrics(equity: pd.Series) -> dict[str, float]:
    r = equity.pct_change().dropna()
    if len(r) < 2:
        return {}
    years = len(r) / BARS_PER_YEAR
    total = equity.iloc[-1] / equity.iloc[0] - 1
    downside = r[r < 0].std()
    dd = 1 - equity / equity.cummax()
    return {
        "total_return": float(total),
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else -1.0,
        "sharpe": float(r.mean() / r.std() * np.sqrt(BARS_PER_YEAR)) if r.std() > 0 else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(BARS_PER_YEAR)) if downside and downside > 0 else 0.0,
        "max_drawdown": float(dd.max()),
        "volatility": float(r.std() * np.sqrt(BARS_PER_YEAR)),
        "win_rate": float((r > 0).mean()),
        "var_95": float(-np.quantile(r, 0.05)),
        "cvar_95": float(-r[r <= np.quantile(r, 0.05)].mean()),
    }


def run_backtest(data: dict[str, pd.DataFrame], cfg: BacktestConfig | None = None,
                 sectors: dict[str, str] | None = None, audit: AuditLog | None = None) -> dict:
    cfg = cfg or BacktestConfig()
    audit = audit or AuditLog()
    sectors = sectors or {}
    feats = {s: build_features(df) for s, df in data.items()}
    models = {s: ArgoModel(s, cfg.argo) for s in data}
    broker = PaperBroker(cfg.cash, cfg.costs)
    gate = RiskGate(cfg.limits)
    dates = sorted(set().union(*[df.index for df in data.values()]))

    equity_curve, exposure, turnover, rejections, n_orders = [], [], 0.0, 0, 0
    peak = sod = cfg.cash
    last_close: dict[str, float] = {}

    for i, t in enumerate(dates):
        # 1) Fill orders placed at t-1 close against bar t (1-bar latency, never same bar)
        for s, df in data.items():
            if t in df.index:
                b = df.loc[t]
                sig = feats[s].at[t, "vol20"]
                broker.on_bar(s, b.open, b.high, b.low, b.volume, 0.01 if pd.isna(sig) else float(sig))
                last_close[s] = float(b.close)
        broker.cancel_all()  # day orders: unfilled remainder expires

        # 2) Mark to market
        pos = broker.get_positions()
        equity = broker.cash + sum(q * last_close.get(s, 0.0) for s, q in pos.items())
        peak = max(peak, equity)
        weights = {s: q * last_close[s] / equity for s, q in pos.items()} if equity > 0 else {}
        equity_curve.append((t, equity))
        exposure.append(sum(abs(w) for w in weights.values()))
        pf = PortfolioState(equity=equity, sod_equity=sod, peak_equity=peak, weights=weights, sectors=sectors)

        # 3) Learn from the bar that just closed, then 4) decide for the next bar
        for s, df in data.items():
            if t not in df.index:
                continue
            row = feats[s].loc[t]
            models[s].learn(float(row["r1"]))
            d = models[s].predict(row.to_dict())
            intent = OrderIntent(s, d.target_weight, atr_z=float(np.nan_to_num(row["atr_z"])), model_status=d.status)
            rd = gate.check(intent, pf, now=float(i) * 86400)
            if not rd.approved:
                rejections += 1
            delta_w = rd.weight - weights.get(s, 0.0)
            if rd.approved and abs(delta_w) >= cfg.min_trade_frac and equity > 0:
                qty = delta_w * equity / last_close[s]
                o = broker.place_order(Order(s, qty))
                n_orders += 1
                turnover += abs(delta_w)
                audit.write("decision", {
                    "ts": t.isoformat(), "symbol": s, "order_id": o.id, "qty": qty,
                    "regime": d.regime, "mu": d.mu, "sigma": d.sigma, "status": d.status,
                    "proposed_weight": d.target_weight, "approved_weight": rd.weight,
                    "expert_weights": d.expert_weights, "risk_checks": rd.checks,
                    "risk_reasons": rd.reasons, "mode": rd.mode,
                })
            elif rd.reasons and not rd.approved:
                audit.write("rejection", {"ts": t.isoformat(), "symbol": s, "reasons": rd.reasons, "mode": rd.mode})
        if gate.kill.active:
            # Flatten: market-sell everything at next bar
            for s, q in broker.get_positions().items():
                broker.place_order(Order(s, -q))
        sod = equity  # daily bars: next bar's start-of-day equity

    eq = pd.Series(dict(equity_curve)).astype(float)
    bench = pd.concat([df["close"] / df["close"].iloc[0] for df in data.values()], axis=1).mean(axis=1) * cfg.cash
    warm = cfg.argo.warmup_bars
    folds = np.array_split(np.arange(warm, len(eq)), cfg.folds)
    wf = [{"start": str(eq.index[f[0]].date()), "end": str(eq.index[f[-1]].date()), **metrics(eq.iloc[f])} for f in folds if len(f) > 5]
    return {
        "metrics": metrics(eq.iloc[warm:]),
        "benchmark": metrics(bench.loc[eq.index].iloc[warm:]),
        "walk_forward": wf,
        "orders": n_orders,
        "rejections": rejections,
        "turnover_annual": turnover / max(len(eq) / BARS_PER_YEAR, 1e-9),
        "avg_gross_exposure": float(np.mean(exposure)),
        "drift_events": {s: m.drift_events for s, m in models.items()},
        "kill_switch": {"active": gate.kill.active, "reason": gate.kill.reason},
        "audit_records": len(audit.records),
        "equity": eq,
    }
