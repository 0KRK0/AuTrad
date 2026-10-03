"""Point-in-time market regime from the NIFTY 50 (row t uses closes <= t only).

BULL      index above its 200-day average and the 50-day average rising, volatility normal
BEAR      index below its 200-day average and the 50-day average falling
STRESS    20-day volatility above the 80th percentile of the past 3 years (overrides trend)
SIDEWAYS  everything else
UNKNOWN   not enough history yet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ("BULL", "SIDEWAYS", "BEAR", "STRESS")


def regime_series(close: pd.Series, vol_pct: float = 0.8) -> pd.Series:
    c = close.astype(float).ffill()
    ma200, ma50 = c.rolling(200, min_periods=200).mean(), c.rolling(50, min_periods=50).mean()
    slope = ma50.diff(20)
    vol = np.log(c).diff().rolling(20, min_periods=15).std() * np.sqrt(252)
    thr = vol.rolling(750, min_periods=250).quantile(vol_pct)
    stress = vol > thr
    up = (c > ma200) & (slope > 0)
    down = (c < ma200) & (slope < 0)
    lab = np.select([stress, up, down], ["STRESS", "BULL", "BEAR"], "SIDEWAYS")
    lab = pd.Series(lab, index=c.index)
    lab[ma200.isna() | thr.isna()] = "UNKNOWN"
    return lab


def asof(labels: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    return labels.reindex(labels.index.union(dates)).ffill().reindex(dates).fillna("UNKNOWN")
