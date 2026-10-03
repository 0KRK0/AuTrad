"""Market data loaders: CSV, optional yfinance, and a regime-switching synthetic generator for tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts = "date" if "date" in df.columns else "timestamp"
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df = df.set_index(ts).sort_index()
    return validate(df[COLS])


def load_yfinance(symbol: str, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:  # pragma: no cover
    import yfinance as yf  # optional dependency

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    return validate(df[COLS])


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Data-quality gate: rejects bad bars instead of trading on them."""
    df = df[~df.index.duplicated(keep="first")].dropna()
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1)) | (df["close"] <= 0)
    if bad.mean() > 0.01:
        raise ValueError(f"data quality: {bad.sum()} corrupt bars")
    return df[~bad]


def synthetic(symbol: str, n: int = 1500, seed: int = 0, start: str = "2019-01-01") -> pd.DataFrame:
    """Regime-switching series: persistent trends + mean-reverting ranges + vol clusters.
    Used ONLY to test mechanics. Profit on synthetic data says nothing about real markets."""
    rng = np.random.default_rng(seed)
    regime, rets, p = 0, np.empty(n), 0.0
    for t in range(n):
        if rng.random() < 0.01:
            regime = rng.integers(0, 4)
        vol = 0.008 if regime in (0, 2) else 0.02
        if regime in (0, 1):   # trending: autocorrelated returns
            mu = 0.12 * (rets[t - 1] if t else 0.0) + 0.0006 * (1 if seed % 2 == 0 else -1) * (1 if regime == 0 else 0.3)
        else:                  # ranging: pull back to anchor
            mu = -0.05 * p
        rets[t] = mu + vol * rng.standard_normal()
        p = 0.95 * p + rets[t]
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * np.exp(0.002 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + np.abs(0.004 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - np.abs(0.004 * rng.standard_normal(n)))
    vol = rng.lognormal(14, 0.3, n)
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
