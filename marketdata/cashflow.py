"""Operating cash flow from the stored NSE results XBRL (half-year Sep and full-year Mar statements, Sep 2019 on).

For each filing in facts.csv whose period ends in March or September, read CashFlowsFromUsedInOperatingActivities
for the context ending on the filing's period end, preferring a context without dimensions.

The period a value covers is set by regulation, not read from the context: SEBI LODR Reg 33(3)(g) requires the cash
flow statement with half-yearly (September) and annual (March) results. The XBRL context dates are unreliable here
(annual figures are tagged with a quarter-long context): verified against published annual reports, e.g. TCS FY23
consolidated OCF Rs 41,965 cr, Infosys FY23 Rs 22,467 cr, Reliance FY24 Rs 1,58,788 cr, HUL FY24 Rs 15,469 cr. Output one row per filing with
the same identifiers as facts.csv, so the statement basis (consolidated / standalone) and the PIT availability
match the results filing exactly.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

TAG = "CashFlowsFromUsedInOperatingActivities"
CTX = re.compile(r'<(?:[\w\-]+:)?context\s+id="([^"]+)"\s*>(.*?)</(?:[\w\-]+:)?context>', re.S)
VAL = re.compile(r'<([\w\-]+):' + TAG + r'\s+([^>]*?)>\s*([^<]+?)\s*</', re.S)


def _period(body: str) -> tuple[str | None, str | None, bool]:
    s = re.search(r"<(?:[\w\-]+:)?startDate>([^<]+)<", body)
    e = re.search(r"<(?:[\w\-]+:)?endDate>([^<]+)<", body)
    return (s.group(1).strip() if s else None, e.group(1).strip() if e else None, "segment" in body or "scenario" in body)


def parse_ocf(text: str) -> list[dict]:
    """Every operating-cash-flow value with its context. Older filings (2020-09 .. 2023) reference contexts such as
    OneD / FourD that the file never defines; those values are returned with defined=False."""
    ctx = {cid: _period(b) for cid, b in CTX.findall(text)}
    out = []
    for _, attrs, v in VAL.findall(text):
        m = re.search(r'contextRef="([^"]+)"', attrs)
        if not m:
            continue
        try:
            val = float(v.replace(",", ""))
        except ValueError:
            continue
        start, end, dims = ctx.get(m.group(1), (None, None, False))
        out.append({"ocf": val, "context": m.group(1), "defined": m.group(1) in ctx, "start": start, "end": end, "dims": dims,
                    "days": (pd.Timestamp(end) - pd.Timestamp(start)).days + 1 if start and end else None})
    return out


def build(facts_path: str | Path = "data/pit/nse/facts.csv", xbrl_dir: str | Path = "data/pit/nse/xbrl", since: str = "2019-09-01") -> pd.DataFrame:
    f = pd.read_csv(facts_path)
    f["period_end"] = pd.to_datetime(f["period_end"])
    f = f[(f["period_end"] >= since) & f["period_end"].dt.month.isin([3, 9]) & ~f["bank"].astype("boolean").fillna(False).astype(bool)]
    rows = []
    for r in f.itertuples(index=False):
        p = Path(xbrl_dir) / os.path.basename(str(r.xbrl))
        base = {"symbol": r.symbol, "period_end": r.period_end, "consolidated": bool(r.consolidated), "available_at": r.available_at,
                "filing_id": r.filing_id, "revision": r.revision, "xbrl": r.xbrl}
        if not p.exists():
            rows.append(base | {"status": "xbrl missing"})
            continue
        allv = parse_ocf(p.read_text(errors="ignore"))
        cands = [c for c in allv if c["end"] and pd.Timestamp(c["end"]) == r.period_end and not c["dims"]]
        if not cands and len(allv) == 1 and not allv[0]["defined"]:
            cands = allv                     # a single value under an undefined context: the regulated period (see module doc)
        if not cands:
            rows.append(base | {"status": "no operating cash flow for the period" if not allv else f"ambiguous ({len(allv)} values)"})
            continue
        if len({c["ocf"] for c in cands}) > 1:
            rows.append(base | {"status": f"ambiguous ({len(cands)} different values)"})
            continue
        c = cands[0]
        rows.append(base | {"status": "ok", "ocf": c["ocf"], "covers": "full year" if r.period_end.month == 3 else "half year (H1)",
                            "context": c["context"], "context_defined": c["defined"], "context_days": c["days"]})
    return pd.DataFrame(rows)
