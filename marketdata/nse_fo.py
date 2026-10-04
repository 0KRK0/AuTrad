"""NSE F&O bhavcopy (futures only) and NSE index closes, for v1.1 (basis & relative value).

F&O formats: legacy  https://nsearchives.nseindia.com/content/historical/DERIVATIVES/YYYY/MON/foDDMONYYYYbhav.csv.zip
             UDiFF   https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip (2024-07+)
Indices:     https://nsearchives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv
Stored as data/exchange_all/fo/<year>.csv.gz and data/exchange_all/indices/<year>.csv.gz, with a manifest of sessions
for which NSE has no file (HTTP 404). Transient errors are never recorded as missing; they are retried next run.
Futures columns: date, kind (IDX|STK), symbol, expiry, open, high, low, close, settle, contracts, value (₹), oi, lot
(lot only in UDiFF files), underlying (UDiFF only).
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from marketdata.nse_bhav import UA

ROOT = Path("data/exchange_all")
FO_COLS = ["date", "kind", "symbol", "expiry", "open", "high", "low", "close", "settle", "contracts", "value", "oi", "lot", "underlying"]
IDX_COLS = ["date", "index", "open", "high", "low", "close", "pe", "pb", "div_yield", "turnover_cr"]


def legacy_url(d: date) -> str:
    m = d.strftime("%b").upper()
    return f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{m}/fo{d.strftime('%d')}{m}{d.year}bhav.csv.zip"


def udiff_url(d: date) -> str:
    return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"


def index_url(d: date) -> str:
    return f"https://nsearchives.nseindia.com/content/indices/ind_close_all_{d.strftime('%d%m%Y')}.csv"


def parse_legacy(raw: bytes, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["INSTRUMENT"].isin(["FUTIDX", "FUTSTK"])]
    return pd.DataFrame({"date": d.isoformat(), "kind": df["INSTRUMENT"].str[3:].to_numpy(), "symbol": df["SYMBOL"].str.strip().to_numpy(),
                         "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y").dt.strftime("%Y-%m-%d").to_numpy(),
                         "open": df["OPEN"].to_numpy(), "high": df["HIGH"].to_numpy(), "low": df["LOW"].to_numpy(),
                         "close": df["CLOSE"].to_numpy(), "settle": df["SETTLE_PR"].to_numpy(), "contracts": df["CONTRACTS"].to_numpy(),
                         "value": (pd.to_numeric(df["VAL_INLAKH"], errors="coerce") * 1e5).to_numpy(), "oi": df["OPEN_INT"].to_numpy(),
                         "lot": pd.NA, "underlying": pd.NA})


def parse_udiff(raw: bytes, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df = df[df["FinInstrmTp"].isin(["IDF", "STF"])]
    return pd.DataFrame({"date": d.isoformat(), "kind": df["FinInstrmTp"].map({"IDF": "IDX", "STF": "STK"}).to_numpy(),
                         "symbol": df["TckrSymb"].str.strip().to_numpy(), "expiry": pd.to_datetime(df["XpryDt"]).dt.strftime("%Y-%m-%d").to_numpy(),
                         "open": df["OpnPric"].to_numpy(), "high": df["HghPric"].to_numpy(), "low": df["LwPric"].to_numpy(),
                         "close": df["ClsPric"].to_numpy(), "settle": df["SttlmPric"].to_numpy(),
                         "contracts": (df["TtlTradgVol"] / df["NewBrdLotQty"].where(df["NewBrdLotQty"] > 0)).round().to_numpy(),
                         "value": df["TtlTrfVal"].to_numpy(), "oi": df["OpnIntrst"].to_numpy(), "lot": df["NewBrdLotQty"].to_numpy(),
                         "underlying": df["UndrlygPric"].to_numpy()})


def parse_index(raw: bytes, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return pd.DataFrame({"date": d.isoformat(), "index": df["Index Name"].str.strip().str.upper(), "open": num("Open Index Value"),
                         "high": num("High Index Value"), "low": num("Low Index Value"), "close": num("Closing Index Value"),
                         "pe": num("P/E"), "pb": num("P/B"), "div_yield": num("Div Yield"), "turnover_cr": num("Turnover (Rs. Cr.)")})


def _get(session, url: str, tries: int = 3):  # pragma: no cover - network
    for k in range(tries):
        try:
            r = session.get(url, headers=UA, timeout=25)
        except Exception:
            time.sleep(1 + k)
            continue
        if r.status_code == 404:
            return 404
        if r.status_code == 200:
            return r.content
        time.sleep(1 + k)
    return None


def fetch_fo(d: date, session):  # pragma: no cover - network
    order = [(udiff_url, parse_udiff), (legacy_url, parse_legacy)] if d >= date(2024, 7, 8) else [(legacy_url, parse_legacy), (udiff_url, parse_udiff)]
    seen404 = 0
    for fn, parse in order:
        c = _get(session, fn(d))
        if c == 404:
            seen404 += 1
            continue
        if isinstance(c, bytes) and c[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(c)) as z:
                return parse(z.read(z.namelist()[0]), d)
    return 404 if seen404 == 2 else None


def fetch_index(d: date, session):  # pragma: no cover - network
    c = _get(session, index_url(d))
    if c == 404 or c is None:
        return c
    try:
        return parse_index(c, d)
    except Exception:
        return None


def sessions(y0: int = 2015, y1: int = 2026, last: str = "2026-09-30") -> list[date]:
    ds = set()
    for p in sorted((ROOT / "nse").glob("*.csv.gz")):
        if y0 <= int(p.stem[:4]) <= y1:
            ds |= set(pd.read_csv(p, usecols=["date"])["date"].unique())
    return sorted(date.fromisoformat(x) for x in ds if x <= last)


def download(days: list[date], budget_s: float = 150, workers: int = 8) -> dict:  # pragma: no cover - network
    import requests
    t0 = time.time()
    s = requests.Session()
    out = {}
    for kind, sub, fetch, cols, key in (("fo", "fo", fetch_fo, FO_COLS, ["date", "symbol", "expiry", "kind"]),
                                        ("idx", "indices", fetch_index, IDX_COLS, ["date", "index"])):
        root = ROOT / sub
        root.mkdir(parents=True, exist_ok=True)
        man_p = root / "manifest.json"
        man = json.loads(man_p.read_text()) if man_p.exists() else {"no_file": []}
        no_file = set(man["no_file"])
        stat = {"fetched": 0, "no_file": 0, "failed": 0}
        for year in sorted({d.year for d in days}):
            if time.time() - t0 > budget_s:
                break
            path = root / f"{year}.csv.gz"
            have = set(pd.read_csv(path, usecols=["date"])["date"].unique()) if path.exists() else set()
            todo = [d for d in days if d.year == year and d.isoformat() not in have and d.isoformat() not in no_file]
            frames = []
            for k in range(0, len(todo), 48):
                if time.time() - t0 > budget_s:
                    break
                with ThreadPoolExecutor(workers) as ex:
                    got = list(ex.map(lambda d: (d, fetch(d, s)), todo[k:k + 48]))
                for d, f in got:
                    if isinstance(f, pd.DataFrame):
                        frames.append(f)
                        stat["fetched"] += 1
                    elif f == 404:
                        no_file.add(d.isoformat())
                        stat["no_file"] += 1
                    else:
                        stat["failed"] += 1
            if frames:
                old = [pd.read_csv(path)] if path.exists() else []
                allf = pd.concat(old + frames, ignore_index=True)[cols].drop_duplicates(key).sort_values(key)
                allf.to_csv(path, index=False)
        man["no_file"] = sorted(no_file)
        man_p.write_text(json.dumps(man, indent=1))
        out[kind] = stat
    out["seconds"] = round(time.time() - t0, 1)
    return out


def load_fo(y0: int = 2015, y1: int = 2026) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted((ROOT / "fo").glob("*.csv.gz")) if y0 <= int(p.name[:4]) <= y1], ignore_index=True)


RENAME = {"CNX NIFTY": "NIFTY 50", "CNX BANK": "NIFTY BANK"}       # names used until Nov 2015


def load_index(y0: int = 2015, y1: int = 2026) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in sorted((ROOT / "indices").glob("*.csv.gz")) if y0 <= int(p.name[:4]) <= y1], ignore_index=True)
    df["index"] = df["index"].replace(RENAME)
    return df.drop_duplicates(["date", "index"])


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2015-2026")
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    print(download(sessions(y0, y1), a.budget, a.workers))
