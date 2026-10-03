"""Independent exchange-sourced prices: BSE daily bhavcopy (official, UNADJUSTED closes).

Why BSE: every NIFTY 50 stock and the major ETFs are dual-listed on NSE and BSE. NSE's archive blocks
cloud IPs; BSE's is reachable. Same company, same day, prices differ only by a few paise of cross-venue
spread, so BSE is a fair independent check on the vendor (Yahoo/NSE) series.

Two file formats:
  old (to mid-2024):  EQddmmyy_CSV.ZIP     columns SC_CODE, SC_NAME, OPEN, HIGH, LOW, CLOSE, PREVCLOSE, NO_OF_SHRS, ...
  new (2024 onward):  BhavCopy_BSE_CM_0_0_0_YYYYMMDD_F_0000.CSV   FinInstrmId, ISIN, TckrSymb, OpnPric, ... PrvsClsgPric
Cache: <root>/bse/<YYYY>.csv.gz (only our scrip codes) + <root>/bse/raw/ (nothing kept; files are parsed on arrival)
plus <root>/bse/manifest.json with the sha256 of every yearly file. Raw prices are never edited.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://www.bseindia.com/download/BhavCopy/Equity/"
COLS = ["date", "code", "name", "isin", "open", "high", "low", "close", "prev_close", "volume", "turnover"]

# Our symbol -> BSE scrip code (stable across renames and face-value changes, unlike ISIN).
# Built from the 2026 bhavcopy ticker column and checked by name; see build_code_map().
CODE_OVERRIDES: dict[str, str] = {}


def _get(url: str, session, tries: int = 3) -> bytes | None:  # pragma: no cover - network
    for i in range(tries):
        try:
            r = session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (AuTrad research; daily bhavcopy)"})
            if r.status_code == 200 and len(r.content) > 20_000:
                return r.content
            if r.status_code == 200:
                return None                     # small body = BSE's "file not found" page (holiday / wrong format)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5 * (i + 1))
    return None


def parse_old(blob: bytes, d: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), dtype={"SC_CODE": str})
    df.columns = [c.strip() for c in df.columns]
    return pd.DataFrame({"date": pd.Timestamp(d), "code": df["SC_CODE"].str.strip(), "name": df["SC_NAME"].str.strip(),
                         "isin": "", "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"], "close": df["CLOSE"],
                         "prev_close": df["PREVCLOSE"], "volume": df["NO_OF_SHRS"], "turnover": df["NET_TURNOV"]})


def parse_new(blob: bytes, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(blob), dtype={"FinInstrmId": str})
    df = df[df["Sgmt"].astype(str).str.strip() == "CM"] if "Sgmt" in df else df
    return pd.DataFrame({"date": pd.Timestamp(d), "code": df["FinInstrmId"].str.strip(), "name": df["TckrSymb"].astype(str).str.strip(),
                         "isin": df["ISIN"].astype(str).str.strip(), "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"],
                         "close": df["ClsPric"], "prev_close": df["PrvsClsgPric"], "volume": df["TtlTradgVol"], "turnover": df["TtlTrfVal"]})


def fetch_day(d: date, session) -> pd.DataFrame | None:  # pragma: no cover - network
    new = f"{BASE}BhavCopy_BSE_CM_0_0_0_{d:%Y%m%d}_F_0000.CSV"
    old = f"{BASE}EQ{d:%d%m%y}_CSV.ZIP"
    order = [(new, parse_new), (old, parse_old)] if d >= date(2024, 1, 1) else [(old, parse_old), (new, parse_new)]
    for url, parser in order:
        blob = _get(url, session)
        if blob is not None:
            try:
                return parser(blob, d)
            except Exception:  # noqa: BLE001
                continue
    return None


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def download(root: str | Path, start: date, end: date, codes: set[str] | None = None, workers: int = 4,
             log=print, extra_dates: set[date] | None = None) -> dict[int, int]:  # pragma: no cover - network
    """Download every weekday's bhavcopy (plus `extra_dates`, e.g. weekend special sessions); keep rows for `codes`
    (all rows if None); write one gz CSV per year. Already-cached sessions are skipped."""
    import requests

    root = Path(root) / "bse"
    root.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    counts = {}
    for year in range(start.year, end.year + 1):
        path = root / f"{year}.csv.gz"
        days = weekdays(max(start, date(year, 1, 1)), min(end, date(year, 12, 31)))
        days += sorted(d for d in (extra_dates or set()) if d.year == year and d.weekday() >= 5)
        have = set()
        if path.exists():
            have = set(pd.read_csv(path, usecols=["date"])["date"].unique())
        todo = [d for d in days if d.isoformat() not in have]
        if not todo:
            counts[year] = len(have)
            continue
        with ThreadPoolExecutor(workers) as ex:
            frames = [f for f in ex.map(lambda d: fetch_day(d, s), todo) if f is not None]
        if codes is not None:
            frames = [f[f["code"].isin(codes)] for f in frames]
        old = [pd.read_csv(path, dtype={"code": str})] if path.exists() else []
        df = pd.concat(old + frames, ignore_index=True) if (old or frames) else pd.DataFrame(columns=COLS)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates(["date", "code"]).sort_values(["date", "code"])
        df.to_csv(path, index=False, compression="gzip")
        counts[year] = int(df["date"].nunique())
        log(f"BSE {year}: {counts[year]} sessions")
    man = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.csv.gz"))}
    (root / "manifest.json").write_text(json.dumps({"source": BASE, "files": man}, indent=1))
    return counts


def load(root: str | Path) -> pd.DataFrame:
    root = Path(root) / "bse"
    files = sorted(root.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no BSE cache under {root}; run scripts.fetch_exchange")
    df = pd.concat([pd.read_csv(p, dtype={"code": str}) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def series(df: pd.DataFrame, code: str) -> pd.DataFrame:
    g = df[df["code"] == code].set_index("date").sort_index()
    return g[["open", "high", "low", "close", "prev_close", "volume"]].astype(float)
