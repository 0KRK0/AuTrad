"""Performance metrics on real calendars (NSE 5-day weeks, crypto 7-day weeks, or mixed).

Annualisation uses the actual number of observations per calendar year in the series,
and Sharpe/Sortino are computed on returns in excess of the cash rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def periods_per_year(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 2:
        return 252.0
    years = (idx[-1] - idx[0]).days / 365.25
    return (len(idx) - 1) / years if years > 0 else 252.0


def excess_returns(equity: pd.Series, cash_rate: float) -> pd.Series:
    r = equity.pct_change().dropna()
    days = equity.index.to_series().diff().dt.days.dropna()
    rf = (1 + cash_rate) ** (days / 365.0) - 1
    return r - rf.reindex(r.index).fillna(0.0)


def perf(equity: pd.Series, cash_rate: float = 0.06, exposure: pd.Series | None = None,
         costs: pd.Series | None = None, turnover: float | None = None) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 3 or equity.iloc[0] <= 0:
        return {}
    r = equity.pct_change().dropna()
    ex = excess_returns(equity, cash_rate)
    ppy = periods_per_year(equity.index)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    total = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    vol = r.std() * np.sqrt(ppy)
    downside = ex[ex < 0].std() * np.sqrt(ppy) if (ex < 0).sum() > 1 else np.nan
    dd = 1 - equity / equity.cummax()
    monthly = equity.resample("ME").last().pct_change().dropna()
    q05 = np.quantile(r, 0.05)
    out = {
        "start": str(equity.index[0].date()), "end": str(equity.index[-1].date()), "years": years,
        "total_return": total, "cagr": cagr, "excess_cagr": (1 + cagr) / (1 + cash_rate) - 1,
        "volatility": vol,
        "sharpe": ex.mean() / ex.std() * np.sqrt(ppy) if ex.std() > 0 else 0.0,
        "sortino": ex.mean() * ppy / downside if downside and downside > 0 else 0.0,
        "max_drawdown": dd.max(),
        "calmar": cagr / dd.max() if dd.max() > 0 else 0.0,
        "win_rate_days": (r > 0).mean(),
        "var_95": -q05, "cvar_95": -r[r <= q05].mean(),
        "worst_day": r.min(), "worst_month": monthly.min() if len(monthly) else np.nan,
        "skew": float(r.skew()), "kurtosis": float(r.kurt()),
        "n_obs": len(r),
    }
    if exposure is not None:
        out["avg_exposure"] = float(exposure.reindex(equity.index).mean())
    if costs is not None:
        c = costs.reindex(equity.index)
        paid = float(c.iloc[-1] - c.iloc[0])
        gross = equity + (c - c.iloc[0])
        gtot = gross.iloc[-1] / gross.iloc[0] - 1
        gcagr = (1 + gtot) ** (1 / years) - 1 if years > 0 and gtot > -1 else -1.0
        gross_profit = float(gross.iloc[-1] - gross.iloc[0])
        out.update({"costs_paid": paid, "gross_cagr": gcagr, "cost_drag_cagr": gcagr - cagr,
                    "cost_share_of_gross_profit": paid / gross_profit if gross_profit > 0 else float("inf")})
    if turnover is not None:
        out["turnover_per_year"] = turnover / years if years > 0 else np.nan
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in out.items()}


def yearly(equity: pd.Series, cash_rate: float = 0.06) -> pd.DataFrame:
    rows = []
    for y, e in equity.groupby(equity.index.year):
        prev = equity[equity.index.year < y]
        seg = pd.concat([prev.iloc[-1:], e]) if len(prev) else e
        m = perf(seg, cash_rate)
        if m:
            rows.append({"year": int(y), "return": m["total_return"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def benchmark_equity(close: pd.Series, start_value: float) -> pd.Series:
    """Index buy-and-hold with no costs (reference only; an index cannot be bought directly)."""
    c = close.dropna()
    return c / c.iloc[0] * start_value
