"""NSE corporate actions (all listed companies) -> per-security ex-date price factors for the whole-market panel.

NSE's archived bhavcopy does NOT adjust prev_close on ex-dates (checked v0.8.1: Bajaj Finance's 2016 split shows as
-90 %). Every corporate action is fetched from NSE's corporate-actions API, quarter by quarter, and parsed:

  * Face value split / sub-division "From Rs X To Rs Y"  -> price factor Y/X  (shares x X/Y)
  * Consolidation "From Rs X To Rs Y"                     -> price factor Y/X  (Y > X)
  * Bonus a:b (a new for every b held)                    -> price factor b/(a+b)
  * Dividend Rs d (interim/final/special)                 -> cash d (total-return adjustment)
  * Rights, demerger, scheme of arrangement, capital reduction -> no factor; the ex-date return is marked UNEXPLAINED
    if it is extreme, and never silently treated as a loss

Adjusted return on day t: close_t / (prev_close_t * f_t) - 1 + d_t / (prev_close_t * f_t), where f_t is the product of
that day's price factors (1 if none) and d_t the cash dividend per pre-split share.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd

URL = "https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date={a}&to_date={b}"
ROOT = Path("data/exchange_all/corp_actions")


def fetch(start: int = 2015, end: int = 2026, client=None) -> int:  # pragma: no cover - network
    from marketdata.nse_filings import Client

    c = client or Client(pause=0.3)
    ROOT.mkdir(parents=True, exist_ok=True)
    n = 0
    for y in range(start, end + 1):
        for qa, qb, k in (("01-01", "31-03", 1), ("01-04", "30-06", 2), ("01-07", "30-09", 3), ("01-10", "31-12", 4)):
            p = ROOT / f"{y}Q{k}.json"
            if p.exists():
                continue
            b = c.get(URL.format(a=f"{qa}-{y}", b=f"{qb}-{y}"))
            if b is None:
                continue
            p.write_bytes(b)
            n += 1
    return n


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def parse_subject(subject: str) -> dict:
    """{'price_factor': f, 'dividend': d, 'unadjusted': [kinds]} for one corporate-action subject line."""
    s = " ".join((subject or "").split())
    out = {"price_factor": 1.0, "dividend": 0.0, "unadjusted": [], "rights": None}
    for m in re.finditer(r"(?:Face Valu[es] Split|Sub-Division|Consolidation|F\.?V\.?\s*Spli?t|Split|Consl).{0,60}?(?:(?:From|Frm)\s+)?R[se]\.?\s*([\d.]+)\s*/?-?\s*(?:Per\s*(?:Share)?)?\s*To\s+R[se]\.?\s*([\d.]+)", s, re.I):
        x, y = _num(m.group(1)), _num(m.group(2))
        if x and y:
            out["price_factor"] *= y / x
    for m in re.finditer(r"(?:Bonus|Bns)(?!\s*(?:Ncrps|Debenture|Pref|-\s*\d+\s*Debenture))\s*(?:Issue)?\s*[-:]?\s*(\d+)\s*:\s*(\d+)", s, re.I):
        a, b = int(m.group(1)), int(m.group(2))
        if a > 0 and b > 0:
            out["price_factor"] *= b / (a + b)
    for m in re.finditer(r"(?:Dividend|Divid\w*|Divd\w*|Div)\b[^R]{0,40}?R[se]\.?\s*[-:]?\s*([\d][\d.]*)", s, re.I):
        d = _num(re.sub(r"\.{2,}", ".", m.group(1)).rstrip("."))
        if d:
            out["dividend"] += d
    for m in re.finditer(r"Return of Capital[^R]{0,20}?R[se]\.?\s*([\d.]+)", s, re.I):
        d = _num(m.group(1).rstrip("."))
        if d:
            out["dividend"] += d                             # cash returned: same treatment as a dividend
    m = re.search(r"Rights\s*(\d+)\s*:\s*(\d+)\s*@\s*Premium\s*R[se]\.?\s*([\d.]+)", s, re.I)
    if m:                                                    # theoretical ex-rights factor needs the price: kept as terms
        out["rights"] = (int(m.group(1)), int(m.group(2)), float(m.group(3)))
    for k, rx in (("rights_unparsed", r"\brights?\b(?!\s*\d+\s*:\s*\d+\s*@)"), ("demerger", r"demerg|spin.?off"), ("arrangement", r"scheme of arrangement|amalgamation"),
                  ("capital_reduction", r"capital reduction|reduction of capital")):
        if re.search(rx, s, re.I):
            out["unadjusted"].append(k)
    return out


def load(root: Path = ROOT) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).glob("*.json")):
        try:
            items = json.loads(p.read_bytes())
        except ValueError:
            continue
        for x in items if isinstance(items, list) else []:
            ex = pd.to_datetime(x.get("exDate"), format="%d-%b-%Y", errors="coerce")
            if pd.isna(ex):
                continue
            r = parse_subject(x.get("subject"))
            ra = r.pop("rights")
            rows.append({"isin": (x.get("isin") or "").strip(), "symbol": (x.get("symbol") or "").strip(), "series": x.get("series"),
                         "ex_date": ex, "subject": " ".join((x.get("subject") or "").split()), **r,
                         "unadjusted": ";".join(r["unadjusted"]), "face_value": _num(x.get("faceVal") or ""),
                         "rights_a": ra[0] if ra else 0, "rights_b": ra[1] if ra else 0, "rights_premium": ra[2] if ra else 0.0})
    df = pd.DataFrame(rows).drop_duplicates(["isin", "symbol", "ex_date", "subject"])
    # the same action listed twice with different spelling (e.g. "Rs 0.3" and "Rs 0.30"): keep one
    df = df.drop_duplicates(["isin", "symbol", "ex_date", "price_factor", "dividend", "unadjusted", "rights_a", "rights_b"])
    return df


def daily_factors(ca: pd.DataFrame) -> pd.DataFrame:
    """One row per (isin, symbol, ex_date): product of price factors, sum of dividends, unadjusted kinds."""
    g = ca.groupby(["isin", "symbol", "ex_date"])
    return pd.DataFrame({"price_factor": g["price_factor"].prod(), "dividend": g["dividend"].sum(),
                         "unadjusted": g["unadjusted"].agg(lambda v: ";".join(sorted({k for x in v if isinstance(x, str) for k in x.split(";") if k}))),
                         "rights_a": g["rights_a"].max(), "rights_b": g["rights_b"].max(), "rights_premium": g["rights_premium"].max(),
                         "face_value": g["face_value"].max()}).reset_index()


def rights_factor(prev_close: float, a: int, b: int, premium: float, face_value: float | None) -> float:
    """Theoretical ex-rights price / cum price for a:b rights at (face value + premium)."""
    if not (a and b) or not prev_close or prev_close <= 0:
        return 1.0
    issue = (face_value or 0.0) + premium
    if issue >= prev_close:
        return 1.0                                           # out of the money: no adjustment
    return (b * prev_close + a * issue) / ((a + b) * prev_close)
