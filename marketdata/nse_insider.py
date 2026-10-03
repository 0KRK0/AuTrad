"""SEBI (Prohibition of Insider Trading) Regulation 7(2) disclosures from NSE: who traded, how, how much, and when
the exchange disseminated it (the point-in-time 'available' instant).

Raw responses are stored unmodified per symbol and year (data/pit/nse/insider/<symbol>/<year>.json) with a sha256
manifest; `load()` parses them into one table. Reachable from a home/office connection only.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

URL = "https://www.nseindia.com/api/corporates-pit?index=equities&symbol={sym}&from_date={a}&to_date={b}"


def fetch_year(root: Path, symbol: str, year: int, client, force: bool = False) -> int | None:  # pragma: no cover - network
    from urllib.parse import quote

    d = Path(root) / "insider" / symbol
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{year}.json"
    if p.exists() and not force:
        return None
    raw = client.get(URL.format(sym=quote(symbol), a=f"01-01-{year}", b=f"31-12-{year}"))
    if raw is None:
        return -1
    body = json.loads(raw)
    rows = body.get("data", body) if isinstance(body, dict) else body
    if not rows:                                     # verify an empty year quarter by quarter (guards against silent caps)
        rows = []
        for qa, qb in (("01-01", "31-03"), ("01-04", "30-06"), ("01-07", "30-09"), ("01-10", "31-12")):
            r2 = client.get(URL.format(sym=quote(symbol), a=f"{qa}-{year}", b=f"{qb}-{year}"))
            b2 = json.loads(r2) if r2 else []
            rows += b2.get("data", b2) if isinstance(b2, dict) else b2
    p.write_text(json.dumps({"symbol": symbol, "year": year, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}))
    return len(rows)


def manifest(root: Path) -> dict:
    files = sorted((Path(root) / "insider").glob("*/*.json"))
    m = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (Path(root) / "insider" / "manifest.json").write_text(json.dumps(m, indent=0))
    return m


def _num(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def load(root: Path, alias: dict[str, str] | None = None) -> pd.DataFrame:
    out = []
    for f in sorted((Path(root) / "insider").glob("*/*.json")):
        j = json.loads(f.read_text())
        for r in j["rows"]:
            out.append({"symbol": (alias or {}).get(j["symbol"], j["symbol"]), "raw_symbol": r.get("symbol"), "person": r.get("acqName"),
                        "category": (r.get("personCategory") or "").strip(), "mode": (r.get("acqMode") or "").strip(),
                        "side": (r.get("tdpTransactionType") or "").strip(), "sec_type": r.get("secType"), "qty": _num(r.get("secAcq")),
                        "value": _num(r.get("secVal")), "buy_value": _num(r.get("buyValue")), "sell_value": _num(r.get("sellValue")),
                        "before_pct": _num(r.get("befAcqSharesPer")), "after_pct": _num(r.get("afterAcqSharesPer")),
                        "trade_from": pd.to_datetime(r.get("acqfromDt"), format="%d-%b-%Y", errors="coerce"),
                        "intimated": pd.to_datetime(r.get("intimDt"), format="%d-%b-%Y", errors="coerce"),
                        "disseminated": pd.to_datetime(r.get("date"), format="%d-%b-%Y %H:%M", errors="coerce"), "did": r.get("did")})
    df = pd.DataFrame(out)
    if len(df):
        df["disseminated"] = df["disseminated"].dt.tz_localize("Asia/Kolkata")
        df = df.drop_duplicates(["symbol", "did", "person", "qty", "disseminated"])
    return df
