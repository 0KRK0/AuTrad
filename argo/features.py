"""Point-in-time feature engineering. Row t uses data up to and including bar t only."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["r1", "r5", "r20", "vol20", "z20", "rsi14", "vol_ratio", "atr_z", "r126_skip5", "ma_gap"]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: columns open, high, low, close, volume indexed by timestamp (ascending)."""
    c = df["close"]
    lr = np.log(c).diff()
    out = pd.DataFrame(index=df.index)
    out["r1"] = lr
    out["r5"] = np.log(c / c.shift(5))
    out["r20"] = np.log(c / c.shift(20))
    out["vol20"] = lr.rolling(20).std()
    sma = c.rolling(20).mean()
    out["z20"] = (c - sma) / c.rolling(20).std()
    out["rsi14"] = (_rsi(c) - 50) / 50
    v = df["volume"].astype(float)
    out["vol_ratio"] = np.log((v + 1) / (v.rolling(20).mean() + 1))
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean() / c
    out["atr_z"] = (atr - atr.rolling(100, min_periods=20).mean()) / atr.rolling(100, min_periods=20).std()
    out["ema_slope"] = np.log(c.ewm(span=50, adjust=False).mean()).diff(10)
    out["vol_long"] = lr.rolling(250, min_periods=60).std()
    out["r126_skip5"] = np.log(c.shift(5) / c.shift(126))                 # 6-month momentum, skip last week
    out["ma_gap"] = np.log(c.rolling(50).mean() / c.rolling(200).mean())  # trend state
    # Label: next-bar log return (never used as a feature; only revealed after bar t+1 closes)
    out["y_next"] = lr.shift(-1)
    return out.replace([np.inf, -np.inf], np.nan)
