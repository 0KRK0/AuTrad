"""End-to-end real-market validation for one asset group."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.ledger import trade_stats
from backtest.sim import SimConfig
from marketdata.store import DataStore

from . import stats, tax
from .metrics import benchmark_equity, excess_returns, perf, periods_per_year
from .scorecard import evaluate
from .walkforward import EXTRA_TRIALS, GRID, calibrate_random, group_assets, run_jobs, walk_forward

BASELINES = ["buy_hold", "momentum_6m", "ma_50_200"]


def _window(e: pd.Series, t0: pd.Timestamp) -> pd.Series:
    prev = e[e.index < t0]
    return pd.concat([prev.iloc[-1:], e[e.index >= t0]]) if len(prev) else e[e.index >= t0]


def regimes(oos_equity: pd.Series, nifty_close: pd.Series, cash_rate: float) -> dict:
    """Sharpe of the OOS record split by NIFTY trend (above/below 200-day avg) x volatility (60-day vs median)."""
    c = nifty_close.dropna()
    trend = np.where(c > c.rolling(200).mean(), "bull", "bear")
    vol = np.log(c).diff().rolling(60).std()
    hv = np.where(vol > vol.expanding(250).median(), "high_vol", "low_vol")
    lab = pd.Series([f"{a}/{b}" for a, b in zip(trend, hv, strict=True)], index=c.index)
    ex = excess_returns(oos_equity, cash_rate)
    lab = lab.reindex(ex.index, method="ffill")
    ppy = periods_per_year(oos_equity.index)
    out = {}
    for k, r in ex.groupby(lab):
        if len(r) > 20:
            out[k] = {"days": int(len(r)), "ann_excess_return": float(r.mean() * ppy),
                      "sharpe": float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else 0.0}
    return out


def validate_group(group: str, data_dir: str = "data", cfg: SimConfig | None = None, first_test_year: int = 2018,
                   n_random: int = 100, workers: int | None = None, log=print) -> dict:
    cfg = cfg or SimConfig()
    assets = group_assets(group)
    log(f"[{group}] {len(assets)} assets: running {len(GRID)} AuTrad candidates + {len(BASELINES)} baselines")
    jobs = [("argo", c) for c in GRID] + [("baseline", b) for b in BASELINES]
    res = run_jobs(group, data_dir, cfg, jobs, workers)
    cands = {r.name: r for r in res[: len(GRID)]}
    bases = {r.name: r for r in res[len(GRID):]}
    t0 = pd.Timestamp(f"{first_test_year}-01-01")

    wf = walk_forward(cands, first_test_year, cfg.cash_rate, cfg.cash)
    m = perf(wf.equity, cfg.cash_rate, wf.exposure, wf.costs, wf.turnover)
    tr = trade_stats(wf.ledger) if not wf.ledger.empty else trade_stats(pd.DataFrame())

    log(f"[{group}] running {n_random} random traders")
    size, expo, hold = calibrate_random(cands, wf)
    target_expo = expo
    if n_random:
        # Pilot run: calendar effects (NSE trades 5 days, crypto 7) make realised exposure drift from the
        # target, so measure it on a few pilots and rescale. The null must take the SAME risk as AuTrad.
        pilots = run_jobs(group, data_dir, cfg, [("random", (10_000 + k, size, expo, hold)) for k in range(4)], workers)
        realised = float(np.mean([p.exposure.mean() for p in pilots]))
        if realised > 0:
            expo = expo * target_expo / realised
    rjobs = [("random", (seed, size, expo, hold)) for seed in range(n_random)]
    rnd = run_jobs(group, data_dir, cfg, rjobs, workers) if n_random else []
    rnd_expo = float(np.mean([r.exposure.mean() for r in rnd])) if rnd else float("nan")
    rnd_sharpes = [perf(_window(r.equity, t0), cfg.cash_rate).get("sharpe", 0.0) for r in rnd]

    base_m = {n: perf(_window(r.equity, t0), cfg.cash_rate, r.exposure, r.costs, r.turnover) for n, r in bases.items()}
    for n, r in bases.items():
        base_m[n].update({f"trade_{k}": v for k, v in trade_stats(r.ledger[r.ledger["exit"] >= t0]).items()} if not r.ledger.empty else {})
    nifty = DataStore(data_dir).load(["NIFTY50"]).get("NIFTY50")
    if nifty is not None and group in ("all", "nse_equity"):
        base_m["nifty50_index"] = perf(_window(benchmark_equity(nifty["close"], cfg.cash), t0), cfg.cash_rate)
    best_base = max(base_m.items(), key=lambda kv: kv[1].get("sharpe", -9))

    ex = excess_returns(wf.equity, cfg.cash_rate)
    per_period_sharpes = []
    for r in cands.values():
        e = excess_returns(_window(r.equity, t0), cfg.cash_rate)
        per_period_sharpes.append(float(e.mean() / e.std()) if e.std() > 0 else 0.0)
    n_trials = len(GRID) + EXTRA_TRIALS
    dsr = stats.deflated_sharpe(ex, per_period_sharpes, n_trials)
    ci = stats.bootstrap_sharpe_ci(ex, periods_per_year(wf.equity.index))
    p_rand = stats.random_pvalue(m["sharpe"], rnd_sharpes)

    # Cost share must be measured against TRADING profit, not against interest earned on idle cash.
    trading_net = float(wf.ledger["pnl"].sum()) if not wf.ledger.empty else 0.0
    trading_gross = trading_net + m.get("costs_paid", 0.0)
    m["trading_pnl_net"], m["trading_pnl_gross"] = trading_net, trading_gross
    m["cost_share_of_gross_profit"] = m.get("costs_paid", 0.0) / trading_gross if trading_gross > 0 else float("inf")
    taxes = tax.estimate(wf.ledger) if not wf.ledger.empty else pd.DataFrame()
    at_eq = tax.after_tax_equity(wf.equity, taxes)
    at = perf(at_eq, cfg.cash_rate)
    after_tax_cash = cfg.cash_rate * (1 - 0.30 * tax.CESS)
    # a test year counts as positive only if it beat cash; the final partial year is prorated
    last_year = wf.folds.index.max()
    frac = (wf.equity.index[-1] - pd.Timestamp(f"{last_year}-01-01")).days / 365.25
    beats = [(row.test_return > ((1 + cfg.cash_rate) ** (min(frac, 1.0) if y == last_year else 1.0) - 1)) for y, row in wf.folds.iterrows()]
    pos_share = float(np.mean(beats)) if beats else 0.0
    violations = sum(r.violations for r in cands.values())
    card = evaluate(m, best_baseline=(best_base[0], best_base[1].get("sharpe", 0.0)), random_p=p_rand, dsr=dsr,
                    positive_year_share=pos_share, trades=tr, violations=violations,
                    after_tax_cagr=at.get("cagr", -1.0), after_tax_cash=after_tax_cash)

    attribution = {}
    if not wf.ledger.empty:
        attribution = wf.ledger.groupby("category").agg(trades=("pnl", "size"), pnl=("pnl", "sum"), fees=("fees", "sum")).round(0).to_dict("index")
    cand_m = {n: {"full": perf(r.equity, cfg.cash_rate, r.exposure, r.costs, r.turnover),
                  "oos_window": perf(_window(r.equity, t0), cfg.cash_rate),
                  "kill_events": r.kill_events} for n, r in cands.items()}
    out = {
        "group": group, "symbols": sorted(cands[GRID[0].name].weights.columns.tolist() or [a.symbol for a in assets]),
        "oos": m, "trades": tr, "after_tax": {"cagr": at.get("cagr"), "cash_after_tax": after_tax_cash,
                                               "taxes_by_fy": taxes.round(0).reset_index().to_dict("records") if not taxes.empty else []},
        "folds": wf.folds.reset_index().to_dict("records"),
        "candidates": cand_m, "baselines": base_m, "best_baseline": best_base[0],
        "random": {"n": len(rnd_sharpes), "calibration": {"position_size": size, "target_exposure": target_expo,
                                                           "realised_exposure": rnd_expo, "hold_bars": hold},
                   "median_sharpe": float(np.median(rnd_sharpes)) if rnd_sharpes else None,
                   "p95_sharpe": float(np.quantile(rnd_sharpes, 0.95)) if rnd_sharpes else None, "p_value": p_rand},
        "stats": {"psr": stats.psr(ex), "deflated_sharpe": dsr, "n_trials": n_trials, "sharpe_ci_95": ci},
        "regimes": regimes(wf.equity, nifty["close"], cfg.cash_rate) if nifty is not None else {},
        "attribution": attribution, "scorecard": card,
        "_series": {"autrad_oos": wf.equity, **{n: _window(r.equity, t0) for n, r in bases.items()}},
        "_ledger": wf.ledger,
    }
    log(f"[{group}] verdict {card['verdict']} ({card['passed']}/{card['total']} gates)  "
        f"OOS CAGR {m['cagr']:+.2%}  Sharpe {m['sharpe']:.2f}  maxDD {m['max_drawdown']:.1%}")
    return out
