"""NSE official daily bhavcopy (cash market): the exchange's own open/high/low/close per security and session.

Formats: legacy  https://nsearchives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
         UDiFF   https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip (2024-07+)
Reachable from a home/office connection (not from most cloud IPs). Rows are kept only for the EQ/BE/BZ series of the
securities we track (by ISIN or symbol). Cache: <root>/nse/<year>.csv.gz plus manifest.json (sha256 per year).
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Accept": "*/*", "Referer": "https://www.nseindia.com/"}
SERIES = {"EQ", "BE", "BZ"}
COLS = ["date", "symbol", "series", "isin", "open", "high", "low", "close", "prev_close", "volume", "value"]


def legacy_url(d: date) -> str:
    m = d.strftime("%b").upper()
    return f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{d.year}/{m}/cm{d.strftime('%d')}{m}{d.year}bhav.csv.zip"


def udiff_url(d: date) -> str:
    return f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"


def parse_legacy(raw: bytes, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    out = pd.DataFrame({"symbol": df["SYMBOL"].str.strip(), "series": df["SERIES"].str.strip(), "isin": df.get("ISIN", pd.Series("", index=df.index)).fillna("").str.strip(),
                        "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"], "close": df["CLOSE"], "prev_close": df["PREVCLOSE"],
                        "volume": df["TOTTRDQTY"], "value": df["TOTTRDVAL"]})
    out.insert(0, "date", d.isoformat())
    return out


def parse_udiff(raw: bytes, d: date) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), dtype=str)
    out = pd.DataFrame({"symbol": df["TckrSymb"].str.strip(), "series": df["SctySrs"].fillna("").str.strip(), "isin": df["ISIN"].fillna("").str.strip(),
                        "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"], "close": df["ClsPric"], "prev_close": df["PrvsClsgPric"],
                        "volume": df["TtlTradgVol"], "value": df["TtlTrfVal"]})
    out.insert(0, "date", d.isoformat())
    return out


def _unzip(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        return z.read(z.namelist()[0])


def fetch_day(d: date, session, tries: int = 2) -> pd.DataFrame | None:  # pragma: no cover - network
    order = [(udiff_url, parse_udiff), (legacy_url, parse_legacy)] if d >= date(2024, 7, 1) else [(legacy_url, parse_legacy), (udiff_url, parse_udiff)]
    for url_fn, parse in order:
        for _ in range(tries):
            try:
                r = session.get(url_fn(d), headers=UA, timeout=20)
            except Exception:
                continue
            if r.status_code == 200 and r.content[:2] == b"PK":
                return parse(_unzip(r.content), d)
            if r.status_code == 404:
                break
    return None


def keep(df: pd.DataFrame, isins: set[str], symbols: set[str]) -> pd.DataFrame:
    df = df[df["series"].isin(SERIES) & (df["isin"].isin(isins) | df["symbol"].isin(symbols))].copy()
    for c in ["open", "high", "low", "close", "prev_close", "volume", "value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[COLS]


def download(root: str | Path, days: list[date], isins: set[str], symbols: set[str], workers: int = 8,
             budget_s: float | None = None, refetch: set[str] | None = None) -> dict:  # pragma: no cover - network
    """refetch: ISO dates to download again (e.g. after adding an identity); new rows are merged, cached rows kept."""
    """Fetch the given sessions that are not cached yet (a missing file = holiday, remembered in the manifest)."""
    import time

    import requests

    t0 = time.time()
    root = Path(root) / "nse"
    root.mkdir(parents=True, exist_ok=True)
    man_p = root / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"years": {}, "no_file": []}
    no_file = set(man.get("no_file", []))
    s = requests.Session()
    done = {"fetched": 0, "missing": 0}
    for year in sorted({d.year for d in days}):
        if budget_s and time.time() - t0 > budget_s:
            break
        path = root / f"{year}.csv.gz"
        have = set(pd.read_csv(path, usecols=["date"])["date"].unique()) if path.exists() else set()
        todo = [d for d in days if d.year == year and ((d.isoformat() not in have and d.isoformat() not in no_file)
                                                       or d.isoformat() in (refetch or set()))]
        if not todo:
            continue
        frames = []
        for k in range(0, len(todo), 40):
            if budget_s and time.time() - t0 > budget_s:
                break
            chunk = todo[k:k + 40]
            with ThreadPoolExecutor(workers) as ex:
                got = list(ex.map(lambda d: (d, fetch_day(d, s)), chunk))
            for d, f in got:
                if f is None:
                    no_file.add(d.isoformat())
                    done["missing"] += 1
                else:
                    frames.append(keep(f, isins, symbols))
                    done["fetched"] += 1
        if frames:
            old = [pd.read_csv(path, dtype={"isin": str})] if path.exists() else []
            allf = pd.concat(old + frames, ignore_index=True).drop_duplicates(["date", "symbol", "series"]).sort_values(["date", "symbol"])
            allf.to_csv(path, index=False)
            man["years"][str(year)] = hashlib.sha256(path.read_bytes()).hexdigest()
    man["no_file"] = sorted(no_file)
    man_p.write_text(json.dumps(man, indent=1))
    done["seconds"] = round(time.time() - t0, 1)
    return done


def load(root: str | Path) -> pd.DataFrame:
    files = sorted((Path(root) / "nse").glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"no NSE bhavcopy cache under {root}")
    df = pd.concat([pd.read_csv(p, dtype={"isin": str, "symbol": str, "series": str}) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def series_for(df: pd.DataFrame, isins: set[str], symbols: set[str]) -> pd.DataFrame:
    """One row per session for one security (matched by any of its ISINs or symbols); EQ preferred over BE/BZ."""
    g = df[df["isin"].isin(isins) | df["symbol"].isin(symbols)].copy()
    if g.empty:
        return g
    g["by_isin"] = (~g["isin"].isin(isins)).astype(int)            # 0 = matched by ISIN (preferred)
    g["rank"] = g["series"].map({"EQ": 0, "BE": 1, "BZ": 2}).fillna(3)
    g = g.sort_values(["date", "by_isin", "rank", "value"], ascending=[True, True, True, False]).drop_duplicates("date")
    return g.set_index("date")[["open", "high", "low", "close", "volume", "series", "symbol"]]
