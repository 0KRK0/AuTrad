"""Portfolio construction shared by every lab strategy (same limits as the production Risk Gate)."""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from .panel import Panel

NAME_CAP = {"largecap": 0.10, "diversified": 1.00, "default": 0.05, "fx": 0.50, "crypto": 0.05, "futures": 0.10}
SECTOR_CAP = 0.30
SECTOR_EXEMPT = {"index", "gold", "silver", "fx", "crypto", "banks_index"}


class Features:
    """Point-in-time feature matrices (row t uses data <= t only), computed once per panel."""

    def __init__(self, P: Panel) -> None:
        self.P = P
        self.Cdf = pd.DataFrame(P.ffill_close(), index=P.dates, columns=P.symbols)

    def ret_over(self, lookback: int, skip: int = 0) -> pd.DataFrame:
        return self.Cdf.shift(skip) / self.Cdf.shift(lookback) - 1

    def ma(self, n: int) -> pd.DataFrame:
        return self.Cdf.rolling(n, min_periods=n).mean()

    def vol(self, n: int) -> pd.DataFrame:
        return self.Cdf.pct_change(fill_method=None).rolling(n, min_periods=max(10, n // 2)).std()

    @cached_property
    def vol63(self) -> pd.DataFrame:
        return self.vol(63)

    @cached_property
    def ma200(self) -> pd.DataFrame:
        return self.ma(200)

    @cached_property
    def bench_ma200(self) -> pd.Series:
        return self.P.bench.rolling(200, min_periods=200).mean()

    def rsi(self, n: int) -> pd.DataFrame:
        d = self.Cdf.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + up / dn.replace(0, np.nan))

    def high_n(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(self.P.H, index=self.P.dates, columns=self.P.symbols).rolling(n, min_periods=n).max()

    def low_n(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(self.P.L, index=self.P.dates, columns=self.P.symbols).rolling(n, min_periods=n).min()


_FEATURES: dict[int, Features] = {}


def features(P: Panel) -> Features:
    if id(P) not in _FEATURES:
        _FEATURES.clear()
        _FEATURES[id(P)] = Features(P)
    return _FEATURES[id(P)]


def month_starts(P: Panel) -> list[int]:
    m = P.dates.to_series().dt.to_period("M")
    return list(np.nonzero((m != m.shift()).to_numpy())[0])


def every(P: Panel, n: int, offset: int = 0) -> list[int]:
    return list(range(offset, len(P.dates), n))


def tradable(P: Panel, t: int, klasses: set[str] | None = None) -> np.ndarray:
    ok = P.eligible[t] & ~np.isnan(P.C[t])
    if klasses is not None:
        ok &= np.array([k in klasses for k in P.klass])
    return ok


def cap_weights(P: Panel, w: pd.Series, budget: float = 1.0) -> pd.Series:
    """Apply name caps, sector caps and the gross budget; any excess stays in cash."""
    if w.empty:
        return w
    w = w.copy()
    caps = pd.Series({s: NAME_CAP.get(P.klass[P.idx[s]], 0.05) for s in w.index})
    w = np.sign(w) * np.minimum(w.abs(), caps)
    sec = pd.Series({s: P.sector[P.idx[s]] for s in w.index})
    for sname, g in w.groupby(sec):
        if sname in SECTOR_EXEMPT:
            continue
        tot = g.abs().sum()
        if tot > SECTOR_CAP:
            w[g.index] *= SECTOR_CAP / tot
    gross = w.abs().sum()
    if gross > budget:
        w *= budget / gross
    return w[w.abs() > 1e-6]


def rank_select(P: Panel, t: int, scores: pd.Series, n: int, weighting: str = "inv_vol", budget: float = 1.0) -> pd.Series:
    s = scores.dropna()
    s = s[s > -np.inf].sort_values(ascending=False).head(n)
    if s.empty:
        return pd.Series(dtype=float)
    if weighting == "inv_vol":
        v = features(P).vol63.iloc[t].reindex(s.index).replace(0, np.nan)
        raw = (1 / v).fillna((1 / v).median() if v.notna().any() else 1.0)
    else:
        raw = pd.Series(1.0, index=s.index)
    w = raw / raw.sum() * budget
    return cap_weights(P, w, budget)


def targets_frame(rows: dict[int, pd.Series], P: Panel) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=P.symbols)
    df = pd.DataFrame({P.dates[t]: r for t, r in rows.items()}).T
    return df.reindex(columns=P.symbols).fillna(0.0).sort_index()
