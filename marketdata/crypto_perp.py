"""Binance USDⓈ-M perpetual funding rates and daily bars (v1.0, study M3) from the public archive
(data.binance.vision, monthly files). Stored compactly under data/crypto/binance/perp/:
  <SYM>_funding.csv.gz   time (UTC, funding settlement), interval_h, rate
  <SYM>_perp_1d.csv.gz   date (UTC), open, high, low, close, volume, quote_volume
The monthly archive starts in 2020-01, so 2019-12 (needed only for the first 7-day signal) and the single funding
print at 2026-10-01 00:00 (credited to 2026-09-30) were taken from the public REST endpoints /fapi/v1/fundingRate
and /fapi/v1/klines on 2026-10-04.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

FILES = "https://data.binance.vision/"
OUT = Path("data/crypto/binance/perp")


def months(start: str = "2020-01", end: str = "2026-09") -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        raw = z.read(z.namelist()[0]).decode()
    first = raw.split("\n", 1)[0]
    header = 0 if any(c.isalpha() for c in first) else None
    return pd.read_csv(io.StringIO(raw), header=header)


def _ms(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x)
    return x.where(x < 1e14, x // 1000)                    # some newer archive files use microseconds


def parse_funding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["calc_time", "interval_h", "rate"][: df.shape[1]] if df.columns[0] != "calc_time" else df.columns
    df = df.rename(columns={"funding_interval_hours": "interval_h", "last_funding_rate": "rate"})
    t = pd.to_datetime(_ms(df["calc_time"]), unit="ms", utc=True).dt.round("h")
    return pd.DataFrame({"time": t, "interval_h": pd.to_numeric(df["interval_h"]), "rate": pd.to_numeric(df["rate"])})


def parse_klines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[:, :8].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    d = pd.to_datetime(_ms(df["open_time"]), unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    out = df[["open", "high", "low", "close", "volume", "quote_volume"]].apply(pd.to_numeric)
    out.insert(0, "date", d)
    return out


def fetch(session, sym: str, start: str = "2020-01", end: str = "2026-09") -> dict:  # pragma: no cover - network
    OUT.mkdir(parents=True, exist_ok=True)
    fr, kl, missing = [], [], []
    for m in months(start, end):
        for kind, url in (("funding", f"data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip"),
                          ("klines", f"data/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{m}.zip")):
            r = session.get(FILES + url, timeout=60)
            if r.status_code != 200:
                missing.append(f"{kind}:{m}")
                continue
            (fr if kind == "funding" else kl).append(_csv(r.content))
    f = pd.concat([parse_funding(x) for x in fr]).drop_duplicates("time").sort_values("time")
    k = pd.concat([parse_klines(x) for x in kl]).drop_duplicates("date").sort_values("date")
    f.to_csv(OUT / f"{sym}_funding.csv.gz", index=False)
    k.to_csv(OUT / f"{sym}_perp_1d.csv.gz", index=False)
    return {"sym": sym, "funding_rows": len(f), "kline_rows": len(k), "missing": missing}


def load(sym: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_csv(OUT / f"{sym}_funding.csv.gz", parse_dates=["time"])
    k = pd.read_csv(OUT / f"{sym}_perp_1d.csv.gz")
    return f, k
