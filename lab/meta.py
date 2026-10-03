"""ARGO-Meta: a regime-aware Bayesian allocator across strategy sleeves, with cash as a real choice.

It does not predict prices. Each month it answers one question: given how each sleeve has behaved in
regimes like the current one (shrunk towards its all-weather record), how much capital, if any, does it
deserve? Anything not allocated stays in cash.

For each sleeve i at a monthly decision date (using data up to the previous close only):
  mu_i   = (n_g * mean_g + k * mean_all) / (n_g + k)      regime mean shrunk towards the global mean
  se_i   = sd_i / sqrt(n_g + k)
  gate   : mu_i / se_i > z          (confidence gate; z = 0 turns it off)
  kill   : rolling 126-day Sharpe < -1  -> sleeve gets nothing until it recovers
  weight : kelly * mu_i / var_i, capped per sleeve, total <= 1, then scaled down to a volatility target
  cash   = 1 - sum(weights), earning the cash rate
Rebalancing charges |change in allocation| x rebalance_bps. Weights drift between rebalances.

Eligible sleeves are fixed before the test period (pre-test Sharpe > 0), so the sealed test stays sealed.
Caveat: sleeve configs were themselves chosen on 2021-2023, so the meta's validation window is optimistic;
only its TEST window is clean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from validation import stats
from validation.metrics import excess_returns, perf, periods_per_year

from .judge import GATES, JudgeConfig, sharpe, window
from .pbo import pbo
from .regime import asof, regime_series


@dataclass(frozen=True)
class MetaParams:
    lookback: int = 756          # days of history used (0 = expanding)
    z: float = 0.5               # confidence gate
    kelly: float = 0.5           # fraction of the Kelly weight
    use_regime: bool = True
    prior_k: float = 250.0       # shrinkage strength (days)
    sleeve_cap: float = 0.5
    vol_target: float = 0.12
    kill_sharpe: float = -1.0
    rebalance_bps: float = 15.0


GRID = {"lookback": [252, 756], "z": [0.0, 1.0], "kelly": [0.5, 1.0], "use_regime": [True, False]}


def grid_configs() -> list[MetaParams]:
    out = [{}]
    for k, vs in GRID.items():
        out = [{**c, k: v} for c in out for v in vs]
    return [MetaParams(**c) for c in out]


def sleeve_returns(navs: dict[str, pd.Series], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Align sleeve NAVs on one calendar (the NSE calendar) and turn them into daily returns."""
    cols = {}
    for k, n in navs.items():
        n = n.reindex(n.index.union(dates)).ffill().reindex(dates)
        cols[k] = n.pct_change(fill_method=None)
    return pd.DataFrame(cols, index=dates).iloc[1:]


def _month_starts(idx: pd.DatetimeIndex) -> np.ndarray:
    m = idx.to_period("M")
    return np.r_[0, np.nonzero(m[1:] != m[:-1])[0] + 1]


@dataclass
class MetaResult:
    nav: pd.Series
    alloc: pd.DataFrame        # target allocation at each rebalance
    costs: pd.Series
    exposure: pd.Series
    params: dict


def allocate(R: pd.DataFrame, regimes: pd.Series, p: MetaParams, cash_rate: float, eligible: list[str], t: int) -> pd.Series:
    """Target weights at decision index t using rows < t only."""
    lo = 0 if p.lookback == 0 else max(0, t - p.lookback)
    hist = R.iloc[lo:t][eligible]
    w = pd.Series(0.0, index=R.columns)
    if len(hist) < 126:
        return w
    rf_d = (1 + cash_rate) ** (1 / 252) - 1
    ex = hist - rf_d
    ok = ex.notna().sum() >= 126
    ex = ex.loc[:, ok]
    if ex.empty:
        return w
    g = regimes.iloc[t - 1]
    lab = regimes.iloc[lo:t]
    mu_all, sd = ex.mean(), ex.std()
    if p.use_regime and g != "UNKNOWN":
        sub = ex[lab.to_numpy() == g]
        n_g = sub.notna().sum()
        mu = (n_g * sub.mean().fillna(0) + p.prior_k * mu_all) / (n_g + p.prior_k)
        se = sd / np.sqrt(n_g + p.prior_k)
    else:
        mu, se = mu_all, sd / np.sqrt(ex.notna().sum())
    recent = ex.iloc[-126:]
    roll_sr = recent.mean() / recent.std() * np.sqrt(252)
    keep = (mu / se > p.z) & (mu > 0) & (roll_sr > p.kill_sharpe)
    if not keep.any():
        return w
    raw = (p.kelly * mu[keep] / (sd[keep] ** 2)).clip(lower=0, upper=p.sleeve_cap)
    if raw.sum() > 1:
        raw = raw / raw.sum()
    cov = ex[raw.index].iloc[-252:].cov() * 252
    vol = float(np.sqrt(raw @ cov @ raw)) if len(raw) else 0.0
    if vol > p.vol_target:
        raw *= p.vol_target / vol
    w[raw.index] = raw
    return w


def run_meta(R: pd.DataFrame, regimes: pd.Series, p: MetaParams, eligible: list[str], cash_rate: float = 0.06,
             capital: float = 10_00_000, fixed: dict[int, pd.Series] | None = None) -> MetaResult:
    """Simulate. `fixed` (decision index -> weights) replaces the allocator (used for baselines and nulls)."""
    dates = R.index
    Rv = R.fillna(0.0).to_numpy()
    days = np.r_[1, np.diff(dates.values).astype("timedelta64[D]").astype(float)]
    rf = (1 + cash_rate) ** (days / 365.0) - 1
    reb = set(_month_starts(dates).tolist())
    h = np.zeros(R.shape[1])          # value held in each sleeve
    cash = capital
    nav, cst, expo, allocs = np.empty(len(dates)), np.zeros(len(dates)), np.zeros(len(dates)), {}
    for t in range(len(dates)):
        if t in reb and t > 0:
            v = cash + h.sum()
            w = fixed[t] if fixed is not None and t in fixed else (allocate(R, regimes, p, cash_rate, eligible, t) if fixed is None else None)
            if w is not None:
                w = w.reindex(R.columns).fillna(0.0).to_numpy()
                new = w * v
                c = np.abs(new - h).sum() * p.rebalance_bps / 1e4
                h, cash = new, v - new.sum() - c
                cst[t] = c
                allocs[dates[t]] = w
        h = h * (1 + Rv[t])
        cash *= 1 + rf[t]
        nav[t] = cash + h.sum()
        expo[t] = h.sum() / nav[t] if nav[t] > 0 else 0.0
    alloc = pd.DataFrame(allocs, index=R.columns).T
    return MetaResult(pd.Series(nav, dates), alloc, pd.Series(np.cumsum(cst), dates), pd.Series(expo, dates), p.__dict__)


def _fixed_equal(R: pd.DataFrame, names: list[str]) -> dict[int, pd.Series]:
    w = pd.Series(0.0, index=R.columns)
    if names:
        w[names] = 1.0 / len(names)
    return {int(t): w for t in _month_starts(R.index) if t > 0}


def judge_meta(navs: dict[str, pd.Series], dates: pd.DatetimeIndex, bench_close: pd.Series, jc: JudgeConfig,
               n_trials_before: int, benchmark_nav: pd.Series | None = None, n_null: int = 200, log=print) -> dict:
    rf = jc.cash_rate
    R = sleeve_returns(navs, dates)
    pre = pd.Timestamp(jc.test_start)
    pre_sharpe = {k: sharpe(window(n[n.index < pre], jc.oos_start), rf) for k, n in navs.items()}
    eligible = [k for k, s in pre_sharpe.items() if s > 0]
    regimes = asof(regime_series(bench_close.dropna()), R.index)
    cfgs = grid_configs()
    runs = {repr(p.__dict__): run_meta(R, regimes, p, eligible, rf, jc.capital) for p in cfgs}
    val = {k: sharpe(window(r.nav[r.nav.index < pre], jc.val_start), rf) for k, r in runs.items()}
    sel = max(val, key=val.get)
    M = runs[sel]
    tw, ow = window(M.nav, jc.test_start), window(M.nav, jc.oos_start)
    s_test, s_oos = sharpe(tw, rf), sharpe(ow, rf)
    m_test = perf(tw, rf, M.exposure, M.costs)
    m_oos = perf(ow, rf, M.exposure, M.costs)

    # baselines: equal-weight eligible sleeves, best single sleeve (chosen pre-test), NIFTY ETF, cash
    base = {}
    p0 = MetaParams()
    ew = run_meta(R, regimes, p0, eligible, rf, jc.capital, fixed=_fixed_equal(R, eligible))
    base["equal_weight_sleeves"] = ew.nav
    if eligible:
        best = max(eligible, key=pre_sharpe.get)
        base[f"best_single ({best})"] = run_meta(R, regimes, p0, eligible, rf, jc.capital, fixed=_fixed_equal(R, [best])).nav
    if benchmark_nav is not None:
        base["NIFTY ETF buy & hold"] = benchmark_nav
    baselines = {k: {"test_sharpe": sharpe(window(n, jc.test_start), rf), "oos_sharpe": sharpe(window(n, jc.oos_start), rf),
                     "test_cagr": perf(window(n, jc.test_start), rf).get("cagr"), "oos_cagr": perf(window(n, jc.oos_start), rf).get("cagr"),
                     "oos_max_dd": perf(window(n, jc.oos_start), rf).get("max_drawdown")} for k, n in base.items()}
    baselines["cash"] = {"test_sharpe": 0.0, "oos_sharpe": 0.0, "test_cagr": rf, "oos_cagr": rf, "oos_max_dd": 0.0}

    # random allocation null: same rebalance dates and same total allocation, random split across sleeves
    log(f"  [argo_meta] random allocation null x{n_null}")
    rng = np.random.default_rng(11)
    gross = M.alloc.sum(axis=1)
    pos = {d: i for i, d in enumerate(R.index)}
    null_oos, null_test = [], []
    for _ in range(n_null):
        fixed = {}
        for d, g in gross.items():
            w = pd.Series(0.0, index=R.columns)
            if eligible and g > 0:
                w[eligible] = rng.dirichlet(np.ones(len(eligible))) * g
            fixed[pos[d]] = w
        rn = run_meta(R, regimes, p0, eligible, rf, jc.capital, fixed=fixed).nav
        null_oos.append(sharpe(window(rn, jc.oos_start), rf))
        null_test.append(sharpe(window(rn, jc.test_start), rf))
    p_null = stats.random_pvalue(s_oos, null_oos)
    p_null_test = stats.random_pvalue(s_test, null_test)

    ex = excess_returns(ow, rf)
    trial_sr = [float(e.mean() / e.std()) if e.std() > 0 else 0.0 for e in (excess_returns(window(r.nav, jc.oos_start), rf) for r in runs.values())]
    n_trials = n_trials_before + len(cfgs)
    dsr = stats.deflated_sharpe(ex, trial_sr, n_trials)
    Mx = np.column_stack([window(r.nav, jc.oos_start).pct_change().dropna().to_numpy() for r in runs.values()])
    p_bo = pbo(Mx, jc.pbo_blocks)
    years = sorted({y for y in ow.index.year if y >= int(jc.oos_start[:4])})
    yr = [float(window(M.nav, f"{y}-01-01", f"{y + 1}-01-01").pipe(lambda s: s.iloc[-1] / s.iloc[0] - 1)) for y in years]
    pos_years = float(np.mean([r > rf * 0.9 for r in yr])) if yr else 0.0
    ew_s = baselines["equal_weight_sleeves"]["oos_sharpe"]
    lab = asof(regime_series(bench_close.dropna()), ex.index)
    ppy = periods_per_year(ow.index)
    regimes_tbl = {k: {"days": int(len(g)), "ann_excess": float(g.mean() * ppy),
                       "avg_allocation": float(M.exposure.reindex(g.index).mean())} for k, g in ex.groupby(lab) if len(g) > 20}
    G = GATES
    gates = [
        ("Sealed test Sharpe", s_test >= G["min_test_sharpe"], f"{s_test:.2f}", f">= {G['min_test_sharpe']}"),
        ("Out-of-sample Sharpe (2018+)", s_oos >= G["min_oos_sharpe"], f"{s_oos:.2f}", f">= {G['min_oos_sharpe']}"),
        ("Beats cash in test and OOS", m_test.get("excess_cagr", -1) > 0 and m_oos.get("excess_cagr", -1) > 0,
         f"{m_test.get('excess_cagr', float('nan')):+.1%} / {m_oos.get('excess_cagr', float('nan')):+.1%}", "> 0"),
        ("Beats equal-weight sleeves (Sharpe, OOS)", s_oos > ew_s, f"{s_oos:.2f} vs {ew_s:.2f}", "higher"),
        ("Random allocation null (OOS)", p_null < G["max_null_p"], f"p = {p_null:.3f}", f"< {G['max_null_p']}"),
        ("Deflated Sharpe", dsr >= G["min_dsr"], f"{dsr:.2f} ({n_trials} trials)", f">= {G['min_dsr']}"),
        ("Overfitting (PBO)", np.isfinite(p_bo["pbo"]) and p_bo["pbo"] < G["max_pbo"], f"{p_bo['pbo']:.2f}", f"< {G['max_pbo']}"),
        ("Max drawdown (OOS)", m_oos.get("max_drawdown", 1) <= G["max_dd"], f"{m_oos.get('max_drawdown', float('nan')):.1%}", f"<= {G['max_dd']:.0%}"),
        ("Profitable years", pos_years >= G["min_pos_years"], f"{pos_years:.0%}", f">= {G['min_pos_years']:.0%}"),
    ]
    passed = all(g[1] for g in gates)
    verdict = "QUALIFIED_FOR_PAPER" if passed else ("REJECT" if s_oos <= 0 and s_test <= 0 else "RESEARCH")
    last = M.alloc.iloc[-1] if len(M.alloc) else pd.Series(dtype=float)
    return {
        "strategy": "argo_meta", "family": "META", "horizon": "monthly allocation",
        "description": "Regime-aware Bayesian allocator across sleeves; cash is the default",
        "eligible_sleeves": eligible, "pre_test_sharpe": pre_sharpe, "selected": M.params, "val_sharpe": val,
        "verdict": verdict, "gates_passed": sum(g[1] for g in gates), "gates_total": len(gates),
        "gates": [{"name": n, "passed": bool(p), "value": v, "target": t} for n, p, v, t in gates],
        "test": m_test, "oos": m_oos, "baselines": baselines,
        "nulls": {"p_allocation_oos": p_null, "p_allocation_test": p_null_test, "median_oos": float(np.median(null_oos)),
                  "p95_oos": float(np.quantile(null_oos, 0.95))},
        "stats": {"psr": stats.psr(ex), "dsr": dsr, "n_trials": n_trials, "pbo": p_bo,
                  "sharpe_ci": stats.bootstrap_sharpe_ci(ex, ppy)},
        "regimes": regimes_tbl, "yearly": dict(zip([str(y) for y in years], yr, strict=True)),
        "avg_allocation_oos": float(M.exposure[M.exposure.index >= pd.Timestamp(jc.oos_start)].mean()),
        "latest_allocation": {k: float(v) for k, v in last.items() if v > 1e-4},
        "_nav": M.nav, "_alloc": M.alloc, "_configs": [r.params for r in runs.values()],
    }
