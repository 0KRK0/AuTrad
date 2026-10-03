"""The judge: one frozen evaluation that every strategy (any horizon) must pass. No special treatment.

Stages
 1. Grid on point-in-time data, net of versioned Indian costs.
 2. Config chosen on VALIDATION only (2021-2023); the selector never receives test-period data.
 3. Sealed TEST (2024 onward) evaluated once with the chosen config.
 4. Walk-forward (yearly from 2018, selection by past only) for robustness.
 5. Cost shock 1x / 1.5x / 2x / 3x.  6. Capacity Rs1L -> Rs5Cr.  7. Parameter perturbation.
 8. Nulls: random stock selection and random timing (same weights, rebalance dates, costs).
 9. Bootstrap CI, PSR, Deflated Sharpe (all trials ever run), PBO (CSCV).
10. Regime table.  11. Survivorship-bias test (PIT vs survivors-only).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd

from backtest.ledger import trade_stats
from validation import stats
from validation.metrics import excess_returns, perf, periods_per_year

from . import engine
from .panel import Panel
from .pbo import pbo
from .portfolio import tradable
from .regime import asof, regime_series
from .strategies import StrategySpec


@dataclass(frozen=True)
class JudgeConfig:
    oos_start: str = "2018-01-01"
    val_start: str = "2021-01-01"
    test_start: str = "2024-01-01"
    cash_rate: float = 0.06
    capital: float = 10_00_000
    cost_shocks: tuple = (0.0, 1.0, 1.5, 2.0, 3.0)   # 0.0 = gross diagnostic
    capitals: tuple = (1e5, 5e5, 1e6, 5e6, 1e7, 5e7)
    n_null: int = 200
    pbo_blocks: int = 12
    workers: int = 2


GATES = {"min_test_sharpe": 0.5, "min_oos_sharpe": 0.5, "max_null_p": 0.05, "min_dsr": 0.95, "max_pbo": 0.5,
         "max_dd": 0.25, "min_pos_years": 0.6, "min_stability": 0.6}


def window(nav: pd.Series, start: str, end: str | None = None) -> pd.Series:
    t0 = pd.Timestamp(start)
    prev = nav[nav.index < t0]
    seg = nav[(nav.index >= t0) & ((nav.index < pd.Timestamp(end)) if end else True)]
    return pd.concat([prev.iloc[-1:], seg]) if len(prev) else seg


def sharpe(nav: pd.Series, rf: float) -> float:
    ex = excess_returns(nav, rf)
    return float(ex.mean() / ex.std() * np.sqrt(periods_per_year(nav.index))) if len(ex) > 5 and ex.std() > 0 else 0.0


def key(cfg: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in cfg.items()) or "default"


# ----------------------------------------------------------------- nulls (parallel)
_G: dict = {}


def _null_init(P: Panel, targets: pd.DataFrame, jc: JudgeConfig) -> None:
    _G.update(P=P, targets=targets, jc=jc)


def _random_selection(P: Panel, targets: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = targets.copy() * 0.0
    for d, row in targets.iterrows():
        held = row[row.abs() > 1e-9]
        if held.empty:
            continue
        t = P.dates.get_loc(d)
        klass = {P.klass[P.idx[s]] for s in held.index}
        pool = [P.symbols[j] for j in np.nonzero(tradable(P, t, klass))[0]]
        if len(pool) < len(held):
            out.loc[d] = row
            continue
        pick = rng.choice(pool, size=len(held), replace=False)
        out.loc[d, list(pick)] = rng.permutation(held.to_numpy())
    return out


def _random_timing(P: Panel, targets: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    T = len(P.dates)
    k = int(rng.integers(63, T - 63))
    pos = (np.array([P.dates.get_loc(d) for d in targets.index]) + k) % T
    out = targets.copy()
    out.index = P.dates[pos]
    return out.sort_index()


def selection_null_applicable(P: Panel, targets: pd.DataFrame) -> bool:
    """A random-selection null only means something when the strategy chose among many candidates."""
    ratios = []
    for d, row in targets.iloc[:: max(len(targets) // 40, 1)].iterrows():
        held = row[row.abs() > 1e-9]
        if held.empty:
            continue
        klass = {P.klass[P.idx[s]] for s in held.index}
        ratios.append(tradable(P, P.dates.get_loc(d), klass).sum() / len(held))
    return bool(ratios) and float(np.median(ratios)) >= 3.0


def _null_job(args: tuple) -> float:
    kind, seed = args
    P, targets, jc = _G["P"], _G["targets"], _G["jc"]
    rng = np.random.default_rng(seed)
    tg = _random_selection(P, targets, rng) if kind == "selection" else _random_timing(P, targets, rng)
    r = engine.run(P, tg, capital=jc.capital, cash_rate=jc.cash_rate)
    return sharpe(window(r.nav, jc.oos_start), jc.cash_rate)


def run_nulls(P: Panel, targets: pd.DataFrame, jc: JudgeConfig) -> dict[str, list[float]]:
    jobs = [("selection", s) for s in range(jc.n_null)] + [("timing", 10_000 + s) for s in range(jc.n_null)]
    if jc.workers <= 1 or not hasattr(os, "fork"):
        _null_init(P, targets, jc)
        res = [_null_job(j) for j in jobs]
    else:
        with get_context("fork").Pool(jc.workers, initializer=_null_init, initargs=(P, targets, jc)) as pool:
            res = pool.map(_null_job, jobs, chunksize=8)
    return {"selection": res[: jc.n_null], "timing": res[jc.n_null:]}


# ----------------------------------------------------------------- main
def judge(spec: StrategySpec, P: Panel, jc: JudgeConfig, n_trials_before: int, benchmark_nav: pd.Series | None = None,
          survivors: Panel | None = None, log=print) -> dict:
    rf = jc.cash_rate
    cfgs = spec.configs()
    targets = {key(c): spec.fn(P, **c) for c in cfgs}
    runs = {k: engine.run(P, tg, capital=jc.capital, cash_rate=rf, name=spec.name, params=c)
            for (k, tg), c in zip(targets.items(), cfgs, strict=True)}

    # 2) selection on validation only: the selector sees navs truncated before the test period
    sealed = {k: r.nav[r.nav.index < pd.Timestamp(jc.test_start)] for k, r in runs.items()}
    val_sharpe = {k: sharpe(window(n, jc.val_start), rf) for k, n in sealed.items()}
    sel = max(val_sharpe, key=val_sharpe.get)
    R = runs[sel]

    # 3) sealed test + full out-of-sample window
    tw, ow = window(R.nav, jc.test_start), window(R.nav, jc.oos_start)
    m_test = perf(tw, rf, R.exposure, R.costs, R.turnover * len(tw) / len(R.nav))
    m_oos = perf(ow, rf, R.exposure, R.costs, R.turnover * len(ow) / len(R.nav))
    tr = R.trades
    tr_oos = tr[tr["exit"] >= pd.Timestamp(jc.oos_start)] if len(tr) else tr
    tstats = trade_stats(tr_oos.rename(columns={"pnl": "pnl"})) if len(tr_oos) else trade_stats(pd.DataFrame())

    # 4) walk-forward yearly: pick by expanding past Sharpe
    years = sorted({y for y in R.nav.index.year if y >= int(jc.oos_start[:4])})
    rets, folds = [], []
    for y in years:
        t0 = pd.Timestamp(f"{y}-01-01")
        past = {k: sharpe(r.nav[r.nav.index < t0].iloc[252:], rf) if (r.nav.index < t0).sum() > 300 else -9 for k, r in runs.items()}
        best = max(past, key=past.get)
        seg = window(runs[best].nav, f"{y}-01-01", f"{y + 1}-01-01")
        rets.append(seg.pct_change().dropna())
        folds.append({"year": y, "chosen": best, "return": float(seg.iloc[-1] / seg.iloc[0] - 1)})
    wf_nav = jc.capital * (1 + pd.concat(rets)).cumprod()
    m_wf = perf(wf_nav, rf)
    last = folds[-1]["year"]
    frac = (R.nav.index[-1] - pd.Timestamp(f"{last}-01-01")).days / 365.25
    pos_years = float(np.mean([f["return"] > (1 + rf) ** (min(frac, 1) if f["year"] == last else 1) - 1 for f in folds]))

    # 5) cost shock, 6) capacity
    shock = {}
    for mult in jc.cost_shocks:
        r = R if mult == 1.0 else engine.run(P, targets[sel], capital=jc.capital, cost_mult=mult, cash_rate=rf)
        shock[str(mult)] = {"oos_sharpe": sharpe(window(r.nav, jc.oos_start), rf), "test_sharpe": sharpe(window(r.nav, jc.test_start), rf),
                            "oos_cagr": perf(window(r.nav, jc.oos_start), rf)["cagr"]}
    capacity = {}
    for cap in jc.capitals:
        r = R if cap == jc.capital else engine.run(P, targets[sel], capital=cap, cash_rate=rf)
        capacity[f"{cap:,.0f}"] = {"oos_sharpe": sharpe(window(r.nav, jc.oos_start), rf), "oos_cagr": perf(window(r.nav, jc.oos_start), rf)["cagr"]}

    # 7) parameter perturbation: neighbours differing in exactly one parameter
    selc = runs[sel].params
    neigh = [k for k, r in runs.items() if k != sel and sum(r.params.get(p) != selc.get(p) for p in selc) == 1]
    post = {k: sharpe(window(runs[k].nav, jc.val_start), rf) for k in [sel, *neigh]}
    stability = float(np.mean([post[k] > 0 for k in neigh])) if neigh else float("nan")

    # 8) nulls
    log(f"  [{spec.name}] nulls x{2 * jc.n_null}")
    nulls = run_nulls(P, targets[sel], jc)
    s_oos = sharpe(ow, rf)
    sel_ok = selection_null_applicable(P, targets[sel])
    tg = targets[sel]
    tim_ok = bool(len(tg) > 1 and (tg.diff().abs().to_numpy()[1:] > 1e-9).any())   # constant weights: timing is meaningless
    p_sel = stats.random_pvalue(s_oos, nulls["selection"]) if sel_ok else float("nan")
    p_tim = stats.random_pvalue(s_oos, nulls["timing"]) if tim_ok else float("nan")

    # 9) statistics
    ex = excess_returns(ow, rf)
    trial_sr = [float(e.mean() / e.std()) if e.std() > 0 else 0.0 for e in (excess_returns(window(r.nav, jc.oos_start), rf) for r in runs.values())]
    n_trials = n_trials_before + len(cfgs)
    dsr = stats.deflated_sharpe(ex, trial_sr, n_trials)
    ci = stats.bootstrap_sharpe_ci(ex, periods_per_year(ow.index))
    M = np.column_stack([window(r.nav, jc.oos_start).pct_change().dropna().to_numpy() for r in runs.values()])
    p_bo = pbo(M, jc.pbo_blocks) if M.shape[1] > 1 else {"pbo": float("nan"), "n_splits": 0}

    # 10) regimes
    lab = asof(regime_series(P.bench.dropna()), ex.index)
    ppy = periods_per_year(ow.index)
    regimes = {k: {"days": int(len(g)), "ann_excess": float(g.mean() * ppy),
                   "sharpe": float(g.mean() / g.std() * np.sqrt(ppy)) if g.std() > 0 else 0.0}
               for k, g in ex.groupby(lab) if len(g) > 20}

    # 11) survivorship bias
    surv = None
    if survivors is not None:
        rs = engine.run(survivors, spec.fn(survivors, **selc), capital=jc.capital, cash_rate=rf)
        surv = {"survivors_oos_sharpe": sharpe(window(rs.nav, jc.oos_start), rf), "pit_oos_sharpe": s_oos,
                "survivors_oos_cagr": perf(window(rs.nav, jc.oos_start), rf)["cagr"]}

    bench = {}
    if benchmark_nav is not None:
        bench = {"test_sharpe": sharpe(window(benchmark_nav, jc.test_start), rf), "oos_sharpe": sharpe(window(benchmark_nav, jc.oos_start), rf),
                 "oos_cagr": perf(window(benchmark_nav, jc.oos_start), rf)["cagr"], "test_cagr": perf(window(benchmark_nav, jc.test_start), rf)["cagr"]}

    s_test = sharpe(tw, rf)
    G = GATES
    gates = [
        ("Sealed test Sharpe", s_test >= G["min_test_sharpe"], f"{s_test:.2f}", f">= {G['min_test_sharpe']}"),
        ("Out-of-sample Sharpe (2018+)", s_oos >= G["min_oos_sharpe"], f"{s_oos:.2f}", f">= {G['min_oos_sharpe']}"),
        ("Beats cash in test and OOS", m_test["excess_cagr"] > 0 and m_oos["excess_cagr"] > 0,
         f"{m_test['excess_cagr']:+.1%} / {m_oos['excess_cagr']:+.1%}", "> 0"),
        ("Beats NIFTY ETF (Sharpe, OOS)", bool(bench) and s_oos > bench.get("oos_sharpe", 9), f"{s_oos:.2f} vs {bench.get('oos_sharpe', float('nan')):.2f}", "higher"),
        ("Random selection null", (p_sel < G["max_null_p"]) if sel_ok else True,
         f"p = {p_sel:.3f}" if sel_ok else "n/a (too few candidates to pick from)", f"< {G['max_null_p']}"),
        ("Random timing null", (p_tim < G["max_null_p"]) if tim_ok else True,
         f"p = {p_tim:.3f}" if tim_ok else "n/a (static allocation)", f"< {G['max_null_p']}"),
        ("Deflated Sharpe", dsr >= G["min_dsr"], f"{dsr:.2f} ({n_trials} trials)", f">= {G['min_dsr']}"),
        ("Overfitting (PBO)", np.isfinite(p_bo["pbo"]) and p_bo["pbo"] < G["max_pbo"], f"{p_bo['pbo']:.2f}", f"< {G['max_pbo']}"),
        ("Survives 2x costs", shock["2.0"]["oos_sharpe"] > 0, f"{shock['2.0']['oos_sharpe']:.2f}", "> 0"),
        ("Parameter stability", not np.isfinite(stability) or stability >= G["min_stability"], f"{stability:.0%}" if np.isfinite(stability) else "n/a", f">= {G['min_stability']:.0%}"),
        ("Max drawdown (OOS)", m_oos["max_drawdown"] <= G["max_dd"], f"{m_oos['max_drawdown']:.1%}", f"<= {G['max_dd']:.0%}"),
        ("Profitable years (walk-forward)", pos_years >= G["min_pos_years"], f"{pos_years:.0%}", f">= {G['min_pos_years']:.0%}"),
    ]
    passed = all(g[1] for g in gates)
    verdict = "QUALIFIED_FOR_PAPER" if passed else ("REJECT" if s_oos <= 0 and s_test <= 0 else "RESEARCH")
    if spec.post_test_peek and verdict != "REJECT":
        verdict = "HYPOTHESIS_ONLY"      # designed after seeing the test window: can never qualify on this data
    return {
        "strategy": spec.name, "family": spec.family, "horizon": spec.horizon, "description": spec.description,
        "post_test_peek": spec.post_test_peek, "selected": selc, "val_sharpe": val_sharpe, "verdict": verdict, "gates_passed": sum(g[1] for g in gates), "gates_total": len(gates),
        "gates": [{"name": n, "passed": bool(p), "value": v, "target": t} for n, p, v, t in gates],
        "test": m_test, "oos": m_oos, "walk_forward": {"metrics": m_wf, "folds": folds, "positive_years": pos_years},
        "trades": tstats, "cost_shock": shock, "capacity": capacity,
        "perturbation": {"neighbours": {k: post[k] for k in neigh}, "selected": post[sel], "stability": stability},
        "nulls": {"selection_applicable": sel_ok, "timing_applicable": tim_ok, "p_selection": p_sel, "p_timing": p_tim, "median_selection": float(np.median(nulls["selection"])),
                  "median_timing": float(np.median(nulls["timing"])), "p95_selection": float(np.quantile(nulls["selection"], 0.95)),
                  "p95_timing": float(np.quantile(nulls["timing"], 0.95))},
        "stats": {"psr": stats.psr(ex), "dsr": dsr, "n_trials": n_trials, "sharpe_ci": ci, "pbo": p_bo},
        "regimes": regimes, "survivorship": surv, "benchmark": bench, "cost_model": R.cost_model,
        "_nav": R.nav, "_returns": R.returns, "_wf_nav": wf_nav, "_configs": [r.params for r in runs.values()],
    }
