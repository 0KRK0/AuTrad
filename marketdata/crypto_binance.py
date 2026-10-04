"""Crypto daily bars for every Binance spot pair that ever traded, including delisted ones (survivorship-aware), from
Binance's public data archive (data.binance.vision: monthly zip files of 1-day klines per symbol).

Prices are in the quote currency (USDT for the research universe). Stored per symbol in
data/crypto/binance/1d/<SYMBOL>.csv.gz with a sha256 manifest. Timestamps are UTC; from 2025 Binance writes them in
microseconds (both handled).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import pandas as pd

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision/"
ROOT = Path("data/crypto/binance")
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def _list(session, prefix: str, delimiter: bool = True) -> tuple[list[str], list[str]]:  # pragma: no cover - network
    keys, prefixes, marker = [], [], ""
    while True:
        u = f"{S3}?{'delimiter=/&' if delimiter else ''}prefix={prefix}" + (f"&marker={marker}" if marker else "")
        t = session.get(u, timeout=60).text
        keys += re.findall(r"<Key>([^<]+)</Key>", t)
        prefixes += re.findall(r"<Prefix>([^<]+)</Prefix>", t)[1:] if delimiter else []
        if "<IsTruncated>true</IsTruncated>" not in t:
            break
        marker = (keys[-1] if keys else prefixes[-1])
    return keys, prefixes


def list_symbols(session) -> list[str]:  # pragma: no cover - network
    _, pre = _list(session, "data/spot/monthly/klines/")
    return sorted({p.rstrip("/").split("/")[-1] for p in pre if p.count("/") >= 4})


def fetch_symbol(session, sym: str, root: Path = ROOT) -> int | None:  # pragma: no cover - network
    out = Path(root) / "1d" / f"{sym}.csv.gz"
    if out.exists():
        return None
    keys, _ = _list(session, f"data/spot/monthly/klines/{sym}/1d/", delimiter=False)
    zips = [k for k in keys if k.endswith(".zip")]
    frames = []
    for k in zips:
        for i in range(3):
            try:
                r = session.get(FILES + k, timeout=60)
                if r.status_code == 200 and r.content[:2] == b"PK":
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        raw = z.read(z.namelist()[0])
                    f = pd.read_csv(io.BytesIO(raw), header=None)
                    if not str(f.iloc[0, 0]).isdigit():                 # some files carry a header row
                        f = f.iloc[1:]
                    frames.append(f.iloc[:, :12])
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (i + 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        pd.DataFrame(columns=["date"] + COLS[1:6] + ["quote_volume", "trades"]).to_csv(out, index=False)
        return 0
    df = pd.concat(frames, ignore_index=True)
    df.columns = COLS
    ot = pd.to_numeric(df["open_time"])
    unit = ot.where(ot < 1e14, ot // 1000)                          # microseconds (2025+) -> milliseconds
    df["date"] = pd.to_datetime(unit, unit="ms").dt.normalize()
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("date").sort_values("date")
    df[["date", "open", "high", "low", "close", "volume", "quote_volume", "trades"]].to_csv(out, index=False)
    return len(df)


def manifest(root: Path = ROOT) -> dict:
    m = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((Path(root) / "1d").glob("*.csv.gz"))}
    (Path(root) / "manifest.json").write_text(json.dumps(m, indent=0))
    return m


STABLE = {"USDT", "USDC", "BUSD", "TUSD", "PAX", "USDP", "DAI", "FDUSD", "USDS", "USDSB", "SUSD", "UST", "USTC", "EUR", "GBP", "AUD",
          "TRY", "BRL", "RUB", "UAH", "NGN", "ZAR", "BIDR", "IDRT", "BVND", "PAXG", "AEUR", "EURI", "USD1", "XUSD", "BFUSD", "RLUSD", "USDE",
          "XAUT", "WBTC", "WBETH", "BETH", "BULL", "BEAR", "ETHBULL", "ETHBEAR", "BNBBULL", "BNBBEAR", "EOSBULL", "EOSBEAR",
          "XRPBULL", "XRPBEAR"}   # gold token, wrapped duplicates of BTC/ETH, leveraged tokens


def research_symbol(sym: str) -> str | None:
    """Base asset of a USDT pair that is a real coin: not a stablecoin/fiat/gold token or a leveraged token."""
    if not sym.endswith("USDT"):
        return None
    base = sym[:-4]
    if base in STABLE or re.search(r"(UP|DOWN|BULL|BEAR)$", base) and len(base) > 4:
        return None
    return base


def fill_recent(session, sym: str, root: Path = ROOT, until: str = "2026-10-03") -> int:  # pragma: no cover - network
    """Days after the last monthly archive file, from Binance's public klines API (same exchange, same bars).
    Only for pairs still trading; appended rows are listed in <root>/api_fill.json."""
    p = Path(root) / "1d" / f"{sym}.csv.gz"
    d = pd.read_csv(p, parse_dates=["date"])
    if d.empty:
        return 0
    last = d["date"].max()
    if last >= pd.Timestamp(until) - pd.Timedelta(days=1):
        return 0
    start_ms = int((last + pd.Timedelta(days=1)).timestamp() * 1000)
    r = session.get("https://api.binance.com/api/v3/klines", params={"symbol": sym, "interval": "1d", "startTime": start_ms, "limit": 1000}, timeout=30)
    if r.status_code != 200:
        return 0
    rows = [k for k in r.json() if pd.to_datetime(k[0], unit="ms") < pd.Timestamp(until)]      # only closed days
    if not rows:
        return 0
    f = pd.DataFrame(rows).iloc[:, :12]
    f.columns = COLS
    f["date"] = pd.to_datetime(f["open_time"], unit="ms").dt.normalize()
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "trades"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    new = f[f["date"] > last][["date", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    pd.concat([d, new], ignore_index=True).to_csv(p, index=False)
    return len(new)
