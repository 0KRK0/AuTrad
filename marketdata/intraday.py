"""Intraday bars (Yahoo): 1-hour bars ~730 days, 5-minute bars ~60 days. No order book or bid/ask.

Stored under <data_dir>/intraday/<interval>/<SYMBOL>.csv with Asia/Kolkata timestamps.
Enough to research opening-range, intraday momentum and VWAP rules on bars; NOT enough for
order-flow / microstructure strategies, which need tick and depth data (a paid feed).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INTRADAY_SYMBOLS = ["NIFTYBEES", "BANKBEES", "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN", "BHARTIARTL",
                    "AXISBANK", "KOTAKBANK", "LT", "ITC", "HINDUNILVR", "BAJFINANCE", "MARUTI", "SUNPHARMA", "TATASTEEL",
                    "M&M", "NTPC", "TITAN", "ULTRACEMCO"]
PERIOD = {"1h": "730d", "5m": "60d"}


def fetch(data_dir: str, interval: str = "1h", symbols: list[str] | None = None) -> dict[str, int]:  # pragma: no cover
    import yfinance as yf

    syms = symbols or INTRADAY_SYMBOLS
    tickers = [f"{s}.NS" for s in syms] + ["^NSEI"]
    raw = yf.download(tickers, period=PERIOD[interval], interval=interval, auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    out_dir = Path(data_dir) / "intraday" / interval
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for s, t in zip([*syms, "NIFTY50"], tickers, strict=True):
        try:
            d = raw[t].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        except KeyError:
            continue
        d.index = d.index.tz_convert("Asia/Kolkata")
        d.to_csv(out_dir / f"{s}.csv", index_label="ts", float_format="%.4f")
        counts[s] = len(d)
    return counts


def load(data_dir: str, interval: str = "1h") -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted((Path(data_dir) / "intraday" / interval).glob("*.csv")):
        d = pd.read_csv(p, index_col="ts")
        d.index = pd.to_datetime(d.index, utc=True).tz_convert("Asia/Kolkata")
        out[p.stem] = d
    return out
