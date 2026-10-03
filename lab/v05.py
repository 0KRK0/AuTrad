"""v0.5 research tools (research/preregistration_v0.5.md): low-volatility decomposition (Track B), earnings-reaction
tests (Track C) and the four-axis evidence scorecard. Nothing here is fitted to returns.

Operational notes, fixed before any result (wording-level clarifications of the pre-registration):
  * Monthly series are calendar month-end to month-end returns of engine NAVs.
  * Sector-neutral low-vol: in each sector, k = max(1, round(n/3)) lowest 252-session-vol names.
  * Brinson split (Brinson-Fachler) uses gross holding-period returns, from the open after each decision to the
    close of the next decision; cash forced by the sector cap is its own "sector" earning cash.
  * T1 overlay and its baseline both use a 1 % NAV no-trade band, so event days do not trigger rebalancing of untouched names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .panel import Panel
from .portfolio import features, month_starts, tradable

SUBPERIODS = [("2016-06", "2018-12"), ("2019-01", "2021-12"), ("2022-01", "2023-12"), ("2024-01", "2026-09")]
EPISODES = [("IL&FS", "2018-09", "2018-10"), ("COVID crash", "2020-02", "2020-03"), ("COVID rebound", "2020-04", "2020-12"),
            ("Rate shock", "2022-01", "2022-06"), ("FPI outflow", "2024-10", "2025-02")]


# ----------------------------------------------------------------------------- statistics
def nw_ols(y: np.ndarray, X: np.ndarray, lags: int = 3) -> dict:
    """OLS with an intercept and Newey-West (Bartlett) standard errors."""
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta
    n, k = A.shape
    S = (A * e[:, None]).T @ (A * e[:, None]) / n
    for L in range(1, lags + 1):
        G = (A[L:] * e[L:, None]).T @ (A[:-L] * e[:-L, None]) / n
        S += (1 - L / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(A.T @ A / n)
    V = Q @ S @ Q / n * n / (n - k)
    se = np.sqrt(np.diag(V))
    r2 = 1 - e @ e / ((y - y.mean()) @ (y - y.mean())) if np.var(y) > 0 else float("nan")
    return {"beta": beta, "se": se, "t": beta / se, "r2": float(r2), "n": n, "resid": e}


def cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> dict:
    """OLS with an intercept and cluster-robust (CR1) standard errors; df = clusters - 1."""
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ beta
    n, k = A.shape
    g = pd.Series(np.arange(n)).groupby(pd.Series(groups).to_numpy())
    M = np.zeros((k, k))
    for _, ix in g.groups.items():
        s = A[ix].T @ e[ix]
        M += np.outer(s, s)
    G = len(g.groups)
    Ainv = np.linalg.inv(A.T @ A)
    V = Ainv @ M @ Ainv * G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * (1 - sps.t.cdf(np.abs(t), df=G - 1))
    return {"beta": beta, "se": se, "t": t, "p": p, "n": n, "clusters": G}


def day_clustered_mean(x: pd.Series, days: pd.Series) -> tuple[float, float, int]:
    """Mean of x and its t-stat using one observation per day (the day's average), as in v0.4-r1."""
    d = x.groupby(days.to_numpy()).mean()
    t = float(d.mean() / d.std() * np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else float("nan")
    return float(x.mean()), t, int(len(d))


def monthly(nav: pd.Series) -> pd.Series:
    return nav.resample("ME").last().pct_change().dropna()


def ann(x: pd.Series) -> float:
    return float(x.mean() * 12) if len(x) else float("nan")


# ----------------------------------------------------------------------------- Track B portfolios
def sector_neutral_low_vol(P: Panel) -> pd.DataFrame:
    """Each sector at its share of the equal-weight universe; inside it, the lowest-vol third (>= 1 name), equal weight."""
    v = features(P).vol(252)
    rows = {}
    for t in month_starts(P):
        if t < 252:
            continue
        ok = tradable(P, t, {"largecap"}) & np.isfinite(v.iloc[t].to_numpy())
        js = np.nonzero(ok)[0]
        if len(js) < 10:
            continue
        w = pd.Series(0.0, index=P.symbols)
        sec = pd.Series([P.sector[j] for j in js], index=[P.symbols[j] for j in js])
        for s, g in sec.groupby(sec):
            k = max(1, int(round(len(g) / 3)))
            pick = v.iloc[t][g.index].nsmallest(k).index
            w[pick] = (len(g) / len(js)) / k
        rows[P.dates[t]] = w
    return pd.DataFrame(rows).T


def ex_ante_beta(P: Panel, t: int, weights: pd.Series, n: int = 252) -> float:
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))][t - n + 1:t + 1]
    b = 0.0
    for s, w in weights[weights != 0].items():
        r = P.ret[t - n + 1:t + 1, P.idx[s]]
        ok = np.isfinite(r) & np.isfinite(m)
        if ok.sum() >= 120:
            b += w * np.polyfit(m[ok], r[ok], 1)[0]
        else:
            b += w * 1.0
    return float(b)


def holding_returns(P: Panel, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Gross return of every symbol from the open after each decision to the close of the next decision."""
    C = P.ffill_close()
    out = {}
    for a, b in zip(dates[:-1], dates[1:], strict=True):
        i, k = P.dates.get_loc(a), P.dates.get_loc(b)
        if i + 1 >= len(P.dates):
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a] = pd.Series(C[k] / P.O[i + 1] - 1, index=P.symbols)
    return pd.DataFrame(out).T


def brinson(P: Panel, targets: pd.DataFrame, universe: pd.DataFrame, cash_rate: float = 0.06) -> pd.DataFrame:
    """Per holding period: allocation and selection of the portfolio vs the equal-weight universe (Brinson-Fachler)."""
    dates = list(targets.index)
    R = holding_returns(P, dates)
    sec = pd.Series(P.sector, index=P.symbols)
    rows = []
    for d in R.index:
        wp, wu, r = targets.loc[d], universe.loc[d], R.loc[d]
        ok = r.notna()
        wp, wu = wp.where(ok, 0.0), wu.where(ok, 0.0)
        wu = wu / wu.sum()
        days = (dates[dates.index(d) + 1] - d).days
        rc = (1 + cash_rate) ** (days / 365) - 1
        ru_tot = float((wu * r.fillna(0)).sum())
        alloc = sel = 0.0
        for s in set(sec[(wp > 0) | (wu > 0)]):
            m = sec == s
            Wp, Wu = wp[m].sum(), wu[m].sum()
            rus = float((wu[m] * r[m].fillna(0)).sum() / Wu) if Wu > 0 else 0.0
            rps = float((wp[m] * r[m].fillna(0)).sum() / Wp) if Wp > 0 else rus
            alloc += (Wp - Wu) * (rus - ru_tot)
            sel += Wp * (rps - rus)
        cash = max(0.0, 1 - wp.sum())
        alloc += cash * (rc - ru_tot)
        rp_tot = float((wp * r.fillna(0)).sum() + cash * rc)
        rows.append({"date": d, "allocation": alloc, "selection": sel, "excess": rp_tot - ru_tot})
    return pd.DataFrame(rows).set_index("date")


def concentration(P: Panel, targets: pd.DataFrame, universe: pd.DataFrame) -> dict:
    sec = pd.Series(P.sector, index=P.symbols)
    names, cash, maxw, hp, hu, top = [], [], [], [], [], {}
    for d in targets.index:
        w = targets.loc[d]
        u = universe.loc[d] if d in universe.index else None
        sw = w.groupby(sec).sum()
        sw = sw[sw > 1e-9]
        names.append(int((w > 1e-9).sum()))
        cash.append(float(max(0.0, 1 - w.sum())))
        maxw.append(float(sw.max()) if len(sw) else 0.0)
        hp.append(float(((sw / sw.sum()) ** 2).sum()) if len(sw) else np.nan)
        if u is not None:
            su = u.groupby(sec).sum()
            su = su[su > 0]
            hu.append(float(((su / su.sum()) ** 2).sum()))
        if len(sw):
            top[sw.idxmax()] = top.get(sw.idxmax(), 0) + 1
    n = len(targets)
    return {"avg_names": float(np.mean(names)), "avg_cash": float(np.mean(cash)), "avg_max_sector": float(np.mean(maxw)),
            "max_max_sector": float(np.max(maxw)), "sector_hhi": float(np.nanmean(hp)), "universe_sector_hhi": float(np.nanmean(hu)) if hu else None,
            "top_sector_share_of_months": {k: v / n for k, v in sorted(top.items(), key=lambda x: -x[1])[:4]}}


def regime_flags(P: Panel) -> pd.DataFrame:
    """At each session: NIFTY above its 200-session average; trailing 63-session NIFTY vol above its expanding median."""
    b = P.bench.ffill()
    ma = b.rolling(200, min_periods=200).mean()
    vol = np.log(b).diff().rolling(63, min_periods=63).std()
    med = vol.expanding(min_periods=252).median()
    return pd.DataFrame({"bull": b > ma, "high_vol": vol > med, "valid": ma.notna() & med.notna()}, index=P.dates)


# ----------------------------------------------------------------------------- Track C helpers
def past_only_flag(values: pd.Series, order: pd.Series, q: float, burn: int = 200, below: bool = True) -> pd.Series:
    """Flag each value against the q-quantile of all earlier values (in `order`); NaN during burn-in or if value is NaN."""
    idx = order.sort_values(kind="stable").index
    past: list[float] = []
    out = pd.Series(np.nan, index=values.index, dtype=float)
    for i in idx:
        v = values[i]
        if not np.isfinite(v):
            continue
        if len(past) >= burn:
            thr = np.quantile(past, q)
            out[i] = float(v < thr) if below else float(v > thr)
        past.append(v)
    return out


def holm(p: dict[str, float]) -> dict[str, float]:
    items = sorted((v, k) for k, v in p.items() if np.isfinite(v))
    m, out, run = len(items), {}, 0.0
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


# ----------------------------------------------------------------------------- evidence scorecard
def score_axes(*, stat_ok: bool | None, stat_weak: bool | None, econ_value: float | None, econ_kind: str,
               tradable: dict[str, bool] | None, forward: str) -> dict[str, str]:
    """Four separate axes (pre-registration v0.5): ✓ / ~ / ✗ each, never merged into one number."""
    def mark(ok, weak):
        return "✓" if ok else ("~" if weak else "✗")
    thr = 0.02 if econ_kind == "portfolio" else 0.005
    stat = mark(bool(stat_ok), bool(stat_weak))
    econ = "n/a" if econ_value is None or not np.isfinite(econ_value) else mark(econ_value >= thr, econ_value > 0)
    if tradable is None:
        trad = "n/a"
    else:
        trad = "✗" if not tradable.get("net_positive", False) else mark(all(tradable.values()), True)
    return {"statistical": stat, "economic": econ, "tradable": trad, "forward": forward}
