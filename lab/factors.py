"""v0.4 factor engine: the pre-registered factors (research/preregistration_v0.4.md), scored point-in-time
and evaluated by IC, a net-of-cost long-only tercile, a long-short diagnostic, and stability. No tuning.

Operational notes (wording-level clarifications of the pre-registration, fixed before results):
  * RMOM: beta from a 252-session market-model OLS with intercept; the score is the sum over t-252..t-21 of
    (r - beta * r_market). The intercept is left out of the residual, because OLS residuals including it sum to
    zero over the fit window by construction.
  * EY: TTM EPS made comparable across splits/bonuses (verified factors) divided by the exchange's raw close.
  * EAR uses the first session that could react: the dissemination day if before 15:30 IST, else the next session.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from marketdata.fundamentals import FINANCIAL_SECTORS, PITFundamentals

from . import engine
from .panel import Panel

PRICE_FACTORS = ["MOM12", "MOM6", "RMOM", "REV1", "LVOL", "LBETA", "LDD"]
FUND_FACTORS = ["EY", "BM", "ROE", "MARGIN", "LEV", "STAB", "GREV", "GEPS", "SUE"]
EVENT_FACTORS = ["EAR"]
ALL_FACTORS = PRICE_FACTORS + FUND_FACTORS + EVENT_FACTORS


@dataclass
class FactorContext:
    P: Panel                       # point-in-time equity panel (exchange-anchored, adjusted prices)
    raw_close: pd.DataFrame        # unadjusted exchange close (dates x symbols) for market values
    fund: PITFundamentals | None
    sector: dict[str, str]


def decision_dates(P: Panel, start: str = "2016-05-01") -> list[int]:
    m = P.dates.to_period("M")
    last = [i for i in range(len(P.dates) - 1) if m[i] != m[i + 1]]
    return [i for i in last if P.dates[i] >= pd.Timestamp(start)]


def stock_universe(P: Panel, t: int) -> list[int]:
    return [j for j in range(len(P.symbols)) if P.eligible[t, j] and P.klass[j] in ("largecap", "default")
            and np.isfinite(P.C[t, j])]


# ----------------------------------------------------------------- scores
def price_scores(P: Panel, t: int, js: list[int], C: np.ndarray, R: np.ndarray, m: np.ndarray) -> dict[str, pd.Series]:
    out = {k: {} for k in PRICE_FACTORS}
    if t < 260:
        return {k: pd.Series(dtype=float) for k in out}
    mw = m[t - 251:t + 1]
    for j in js:
        c = C[:, j]
        if not (np.isfinite(c[t]) and np.isfinite(c[t - 252]) and np.isfinite(c[t - 21])):
            continue
        r = R[t - 251:t + 1, j]
        ok = np.isfinite(r) & np.isfinite(mw)
        if ok.sum() < 200:
            continue
        s = P.symbols[j]
        out["MOM12"][s] = c[t - 21] / c[t - 252] - 1
        out["MOM6"][s] = c[t - 21] / c[t - 126] - 1 if np.isfinite(c[t - 126]) else np.nan
        out["REV1"][s] = -(c[t] / c[t - 21] - 1)
        out["LVOL"][s] = -np.std(r[ok], ddof=1)
        beta = np.polyfit(mw[ok], r[ok], 1)[0]
        out["LBETA"][s] = -beta
        win = slice(t - 251, t - 20)
        rr, mm = R[win, j], m[win]
        k = np.isfinite(rr) & np.isfinite(mm)
        out["RMOM"][s] = float(np.sum(rr[k] - beta * mm[k]))
        path = c[t - 251:t + 1]
        path = path[np.isfinite(path)]
        out["LDD"][s] = -float(np.max(1 - path / np.maximum.accumulate(path)))
    return {k: pd.Series(v, dtype=float).dropna() for k, v in out.items()}


def fundamental_scores(ctx: FactorContext, t: int, js: list[int]) -> dict[str, pd.Series]:
    out = {k: {} for k in FUND_FACTORS}
    if ctx.fund is None:
        return {k: pd.Series(dtype=float) for k in out}
    d = ctx.P.dates[t]
    for j in js:
        s = ctx.P.symbols[j]
        rp = ctx.raw_close[s].iloc[t] if s in ctx.raw_close else np.nan
        sn = ctx.fund.snapshot(s, d, float(rp) if np.isfinite(rp) else None)
        if sn.get("n_quarters", 0) == 0:
            continue
        fin = ctx.sector.get(s) in FINANCIAL_SECTORS
        eq, mcap = sn.get("equity", np.nan), sn.get("mcap", np.nan)
        if "ttm_eps" in sn and np.isfinite(rp) and rp > 0:
            k = ctx.fund.split_factor(s, sn["last_period"], d)
            out["EY"][s] = sn["ttm_eps"] * k / rp
        if np.isfinite(eq) and np.isfinite(mcap) and mcap > 0:
            out["BM"][s] = eq / mcap
        if np.isfinite(eq) and eq > 0 and np.isfinite(sn.get("ttm_np", np.nan)):
            out["ROE"][s] = sn["ttm_np"] / eq
        if not fin and np.isfinite(sn.get("ttm_rev", np.nan)) and sn["ttm_rev"] > 0:
            out["MARGIN"][s] = sn["ttm_np"] / sn["ttm_rev"]
        if not fin and np.isfinite(eq) and eq > 0 and np.isfinite(sn.get("borrowings", np.nan)):
            out["LEV"][s] = -sn["borrowings"] / eq
        if np.isfinite(sn.get("eps_yoy_growth_std", np.nan)):
            out["STAB"][s] = -sn["eps_yoy_growth_std"]
        if np.isfinite(sn.get("ttm_rev_prev", np.nan)) and sn["ttm_rev_prev"] > 0:
            out["GREV"][s] = sn["ttm_rev"] / sn["ttm_rev_prev"] - 1
        a, b = sn.get("ttm_eps", np.nan), sn.get("ttm_eps_prev", np.nan)
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            out["GEPS"][s] = a / b - 1
        if np.isfinite(sn.get("sue", np.nan)):
            out["SUE"][s] = sn["sue"]
    return {k: pd.Series(v, dtype=float).replace([np.inf, -np.inf], np.nan).dropna() for k, v in out.items()}


STALE_LAG_DAYS = 60     # SEBI: quarterly results within 45 days (60 for Q4); later timestamps are re-filings


def earnings_events(ctx: FactorContext) -> pd.DataFrame:
    """One row per (symbol, quarter): first dissemination, the reaction session and its market-adjusted return.
    Kept only if the stock was a point-in-time index member on the reaction day and the timestamp is not stale."""
    if ctx.fund is None:
        return pd.DataFrame()
    P = ctx.P
    f = ctx.fund.facts.sort_values("available_at").groupby(["symbol", "period_end"], as_index=False).first()
    C = P.ffill_close()
    lb = np.log(P.bench.ffill().to_numpy())
    rows = []
    for r in f.itertuples(index=False):
        if r.symbol not in P.idx:
            continue
        j = P.idx[r.symbol]
        i = P.dates.searchsorted(r.usable_from)              # first session on/after the usable day
        if i >= len(P.dates) or i == 0:
            continue
        if not (np.isfinite(C[i, j]) and np.isfinite(C[i - 1, j])):
            continue
        lag = getattr(r, "lag_days", 0)
        ear = np.log(C[i, j] / C[i - 1, j]) - (lb[i] - lb[i - 1])
        rows.append({"symbol": r.symbol, "period_end": r.period_end, "available_at": r.available_at, "reaction_idx": i,
                     "reaction_day": P.dates[i], "ear": float(ear), "member": bool(P.eligible[i, j]),
                     "stale": bool(lag is not None and lag > STALE_LAG_DAYS)})
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev.attrs["excluded_nonmember"] = int((~ev["member"]).sum())
    ev.attrs["excluded_stale"] = int((ev["member"] & ev["stale"]).sum())
    return ev[ev["member"] & ~ev["stale"]].reset_index(drop=True)


def event_scores(ev: pd.DataFrame, P: Panel, t: int, js: list[int]) -> pd.Series:
    if ev.empty:
        return pd.Series(dtype=float)
    syms = {P.symbols[j] for j in js}
    e = ev[(ev["reaction_idx"] <= t) & ev["symbol"].isin(syms)].sort_values("reaction_idx").groupby("symbol").last()
    s = pd.Series(np.where(t - e["reaction_idx"] <= 63, e["ear"], 0.0), index=e.index, dtype=float)
    return s


def all_scores(ctx: FactorContext, dates: list[int]) -> dict[str, pd.DataFrame]:
    P = ctx.P
    C = P.ffill_close()
    R = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(np.log(C), axis=0)])
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))]
    ev = earnings_events(ctx)
    frames: dict[str, dict] = {k: {} for k in ALL_FACTORS}
    for t in dates:
        js = stock_universe(P, t)
        sc = {**price_scores(P, t, js, C, R, m), **fundamental_scores(ctx, t, js), "EAR": event_scores(ev, P, t, js)}
        for k, s in sc.items():
            frames[k][P.dates[t]] = s
    return {k: pd.DataFrame(v).T.reindex(columns=P.symbols) for k, v in frames.items()}


# ----------------------------------------------------------------- evaluation
def forward_returns(P: Panel, dates: list[int]) -> pd.DataFrame:
    """Return from the next session's open to the close of the next decision date (the holding month)."""
    rows = {}
    for a, b in zip(dates[:-1], dates[1:], strict=True):
        if a + 1 >= len(P.dates):
            continue
        o = P.O[a + 1]
        c = P.ffill_close()[b]
        with np.errstate(invalid="ignore", divide="ignore"):
            rows[P.dates[a]] = pd.Series(c / o - 1, index=P.symbols)
    return pd.DataFrame(rows).T


def newey_west_t(x: np.ndarray, lags: int = 3) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return float("nan")
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else float("nan")


def ic_series(scores: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 15) -> pd.Series:
    out = {}
    for d in scores.index.intersection(fwd.index):
        s, r = scores.loc[d], fwd.loc[d]
        k = s.notna() & r.notna()
        if k.sum() >= min_names:
            out[d] = sps.spearmanr(s[k], r[k]).statistic
    return pd.Series(out, dtype=float)


def tercile_targets(scores: pd.DataFrame, side: str = "top", long_short: bool = False) -> pd.DataFrame:
    rows = {}
    for d, s in scores.iterrows():
        s = s.dropna()
        if len(s) < 15:
            continue
        n = len(s) // 3
        top, bot = s.nlargest(n).index, s.nsmallest(n).index
        w = pd.Series(0.0, index=scores.columns)
        if long_short:
            w[top] = 0.5 / n
            w[bot] = -0.5 / n
        else:
            w[top if side == "top" else bot] = 1.0 / n
        rows[d] = w
    return pd.DataFrame(rows).T


def universe_targets(scores_any: pd.DataFrame, P: Panel) -> pd.DataFrame:
    rows = {}
    for d in scores_any.index:
        t = P.dates.get_loc(d)
        js = stock_universe(P, t)
        w = pd.Series(0.0, index=P.symbols)
        w.iloc[js] = 1.0 / len(js)
        rows[d] = w
    return pd.DataFrame(rows).T


def _ann(nav: pd.Series) -> float:
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else float("nan")


def evaluate_factor(name: str, scores: pd.DataFrame, fwd: pd.DataFrame, P: Panel, F: Panel | None,
                    bench_nav: pd.Series, capital: float = 10_00_000, cash_rate: float = 0.06) -> dict:
    ic = ic_series(scores, fwd)
    out: dict = {"factor": name, "months": int(len(ic))}
    if len(ic) < 12:
        out["note"] = "fewer than 12 months with enough names"
        return out
    t = newey_west_t(ic.to_numpy())
    p = 2 * (1 - sps.t.cdf(abs(t), df=len(ic) - 1)) if np.isfinite(t) else float("nan")
    half = len(ic) // 2
    yearly = ic.groupby(ic.index.year).mean()
    out.update({"start": str(ic.index[0].date()), "end": str(ic.index[-1].date()), "mean_ic": float(ic.mean()),
                "nw_t": t, "p": float(p), "pct_months_pos": float((ic > 0).mean()),
                "ic_first_half": float(ic.iloc[:half].mean()), "ic_second_half": float(ic.iloc[half:].mean()),
                "pct_years_pos": float((yearly > 0).mean()), "ic_by_year": {int(k): float(v) for k, v in yearly.items()}})
    # long-only top tercile vs the equal-weight universe, both through the full cost model
    tg = tercile_targets(scores)
    r_top = engine.run(P, tg, capital=capital, cash_rate=cash_rate, name=f"{name}_top")
    r_gross = engine.run(P, tg, capital=capital, cash_rate=cash_rate, cost_mult=0.0, name=f"{name}_top_gross")
    start = tg.index[0]
    nav_top = r_top.nav[r_top.nav.index >= start]
    nav_b = bench_nav[bench_nav.index >= start]
    gross_top = r_gross.nav[r_gross.nav.index >= start]
    ex = np.log(nav_top).diff().dropna() - np.log(nav_b).diff().reindex(nav_top.index).dropna()
    ex_m = ex.groupby(ex.index.to_period("M")).sum()
    ex_y = ex.groupby(ex.index.year).sum()
    yrs = (nav_top.index[-1] - start).days / 365.25
    traded = float(r_top.turnover * r_top.nav.mean())                       # rupees traded over the whole run
    out.update({"top_cagr": _ann(nav_top), "universe_cagr": _ann(nav_b), "top_gross_cagr": _ann(gross_top),
                "net_excess": _ann(nav_top) - _ann(nav_b), "gross_excess": _ann(gross_top) - _ann(nav_b),
                "cost_drag": _ann(gross_top) - _ann(nav_top),
                "turnover_per_year": float(traded / nav_top.mean() / max(yrs, 1e-9)),
                "info_ratio": float(ex.mean() / ex.std() * np.sqrt(252)) if ex.std() > 0 else float("nan"),
                "excess_t_monthly": float(ex_m.mean() / ex_m.std() * np.sqrt(len(ex_m))) if ex_m.std() > 0 else float("nan"),
                "excess_by_year": {int(k): float(v) for k, v in ex_y.items()},
                "pct_years_excess_pos": float((ex_y > 0).mean()), "_nav_top": nav_top})
    if F is not None:
        r_ls = engine.run(F, tercile_targets(scores, long_short=True), capital=capital, cash_rate=cash_rate, name=f"{name}_ls")
        nav_ls = r_ls.nav[r_ls.nav.index >= start]
        rls = np.log(nav_ls).diff().dropna()
        rf = np.log(1 + cash_rate) / 252
        out.update({"ls_cagr": _ann(nav_ls), "ls_sharpe": float((rls - rf).mean() / rls.std() * np.sqrt(252)) if rls.std() > 0 else float("nan"),
                    "_nav_ls": nav_ls})
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted((p, k) for k, p in pvals.items() if np.isfinite(p))
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def passes(r: dict, holm_p: float) -> dict[str, bool]:
    """Pre-registered criteria, in the hypothesised direction only (a significantly negative IC is a failure)."""
    if "mean_ic" not in r:
        return {"data": False}
    return {"significant (Holm p < 0.05), positive IC": holm_p < 0.05 and r["mean_ic"] > 0,
            "net excess > 0 after costs": r.get("net_excess", -1) > 0,
            "positive IC in both halves": r["ic_first_half"] > 0 and r["ic_second_half"] > 0,
            "IC > 0 in >= 60% of years": r["pct_years_pos"] >= 0.6}
