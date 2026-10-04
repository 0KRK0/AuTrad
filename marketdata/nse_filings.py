"""Point-in-time company fundamentals from NSE financial-results filings (XBRL), with exchange timestamps.

Sources (NSE, the primary exchange; reachable from a residential connection, blocked from cloud IPs):
  * /api/corporates-financial-results?index=equities&symbol=X&period=Quarterly   results 2005 .. Dec 2024
  * /api/integrated-filing-results?index=equities&symbol=X&type=Integrated Filing- Financials   2025 onward
    (SEBI moved results into "Integrated Filing" in 2025)
  * each filing's XBRL file (nsearchives.nseindia.com/corporate/xbrl/...)

Every filing keeps three times:
  period_end    the quarter the numbers describe
  published_at  NSE broadcast time (when the company filed)
  available_at  exchange dissemination time (when the market could see it) -- the only time a backtest may use
plus ingested_at (when we downloaded it), the raw file's sha256 and a revision number. Raw index JSON and XBRL
files are stored untouched under <root>/nse/; facts.csv is derived from them and can be rebuilt at any time.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
API = "https://www.nseindia.com/api/"
LEGACY = API + "corporates-financial-results?index=equities&symbol={sym}&period=Quarterly"
INTEGRATED = API + "integrated-filing-results?index=equities&symbol={sym}&period_ended=all&type=Integrated%20Filing-%20Financials"

# XBRL element -> our metric name (first match wins; bank taxonomy fallbacks after the general ones)
METRICS = {
    "revenue": ["RevenueFromOperations", "TotalRevenueFromOperations", "InterestEarned", "Income"],
    "total_income": ["Income", "TotalIncome"],
    "pbt": ["ProfitBeforeTax", "ProfitLossBeforeTax", "ProfitLossFromOrdinaryActivitiesBeforeTax"],
    "net_profit": ["ProfitLossForPeriodAttributableToOwnersOfParent", "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
                   "ProfitLossForPeriod", "ProfitLossForThePeriod", "NetProfitLossForThePeriod"],
    "eps_basic": ["BasicEarningsLossPerShareFromContinuingOperations", "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                  "BasicEarningsPerShareAfterExtraordinaryItems", "BasicEarningsLossPerShare"],
    "operating_profit_bank": ["OperatingProfitBeforeProvisionAndContingencies"],
    "eps_diluted": ["DilutedEarningsLossPerShareFromContinuingOperations", "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                    "DilutedEarningsPerShareAfterExtraordinaryItems"],
    "finance_costs": ["FinanceCosts", "InterestExpended"],
    "depreciation": ["DepreciationDepletionAndAmortisationExpense"],
    "paid_up_capital": ["PaidUpValueOfEquityShareCapital"],
    "face_value": ["FaceValueOfEquityShareCapital"],
    "equity": ["EquityAttributableToOwnersOfParent", "Equity", "TotalEquity"],
    "total_assets": ["Assets", "TotalAssets"],
    "borrowings": ["Borrowings", "BorrowingsNoncurrent"],
    "gross_npa_pct": ["PercentageOfGrossNpa"],
}


def _parse_ts(s: str | None) -> pd.Timestamp | None:
    if not s or s in ("-", "null"):
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(s.strip().title(), fmt)).tz_localize("Asia/Kolkata")
        except ValueError:
            continue
    return None


class Client:  # pragma: no cover - network
    def __init__(self, pause: float = 0.6) -> None:
        import requests

        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                               "Referer": "https://www.nseindia.com/"})
        self.pause = pause

    def get(self, url: str, tries: int = 4) -> bytes | None:
        for i in range(tries):
            try:
                r = self.s.get(url, timeout=40)
                if r.status_code == 200 and r.content:
                    time.sleep(self.pause)
                    return r.content
                if r.status_code in (401, 403):
                    self.s.get("https://www.nseindia.com/", timeout=40)       # refresh cookies
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.0 * (i + 1))
        return None


# ----------------------------------------------------------------- index of filings
def filings_index(root: Path, symbol: str, client: Client | None = None) -> pd.DataFrame:
    """All quarterly results filings for a symbol (legacy + integrated), one row per filing."""
    d = Path(root) / "nse" / "index"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for kind, url in (("legacy", LEGACY), ("integrated", INTEGRATED)):
        p = d / f"{symbol}_{kind}.json"
        if not p.exists() and client is not None:
            blob = client.get(url.format(sym=symbol.replace("&", "%26")))
            if blob is not None:
                p.write_bytes(blob)
        if not p.exists():
            continue
        js = json.loads(p.read_bytes())
        items = js.get("data", []) if isinstance(js, dict) else js
        for x in items:
            if kind == "legacy":
                rows.append({"symbol": symbol, "source": "nse_legacy", "period_start": x.get("fromDate"), "period_end": x.get("toDate"),
                             "consolidated": (x.get("consolidated") or "")[:3] == "Con", "audited": x.get("audited"),
                             "published_at": _parse_ts(x.get("broadCastDate")), "disseminated_at": _parse_ts(x.get("exchdisstime")),
                             "filing_id": str(x.get("seqNumber")), "revision": 0, "xbrl": x.get("xbrl"), "bank": x.get("bank") == "Y"})
            else:
                rows.append({"symbol": symbol, "source": "nse_integrated", "period_start": None, "period_end": x.get("qe_Date"),
                             "consolidated": (x.get("consolidated") or "")[:3] == "Con", "audited": x.get("audited"),
                             "published_at": _parse_ts(x.get("broadcast_Date")), "disseminated_at": _parse_ts(x.get("creation_Date")),
                             "filing_id": str(x.get("seq_Id")), "revision": 0 if (x.get("type_Sub") or "Original") in ("Original", "New") else 1,
                             "xbrl": x.get("xbrl"), "bank": None})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["period_end"] = pd.to_datetime(df["period_end"], format="mixed", dayfirst=True)
    for c in ("published_at", "disseminated_at"):            # all-missing columns arrive as object/float
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert("Asia/Kolkata")
    df["available_at"] = df[["published_at", "disseminated_at"]].max(axis=1)
    return df.sort_values(["period_end", "consolidated", "available_at"]).reset_index(drop=True)


def select_filings(idx: pd.DataFrame, start: str = "2014-01-01") -> pd.DataFrame:
    """Per period: consolidated filings if the company files them, else standalone; all revisions kept."""
    if idx.empty:
        return idx
    x = idx[(idx["period_end"] >= pd.Timestamp(start)) & idx["xbrl"].fillna("-").str.lower().str.endswith(".xml")]
    has_cons = x.groupby("period_end")["consolidated"].transform("any")
    return x[x["consolidated"] == has_cons].copy()


# ----------------------------------------------------------------- XBRL
# namespace-agnostic: legacy filings use in-bse-fin:, 2025+ integrated filings use in-capmkt:
_CTX = re.compile(r'<(?:xbrli:)?context\b[^>]*\bid="([^"]+)"[^>]*>(.*?)</(?:xbrli:)?context>', re.S)
_FACT = re.compile(r'<[A-Za-z][\w\-]*:([A-Z][A-Za-z]+)\b[^>]*?contextRef="([^"]+)"[^>]*?>([^<]*)<', re.S)


def parse_xbrl(text: str) -> dict:
    """Quarter (3-month) and point-in-time facts of interest. Returns {metric: value} for the reporting quarter."""
    ctx = {}
    for cid, body in _CTX.findall(text):
        s = re.search(r"<(?:xbrli:)?startDate>([^<]+)", body)
        e = re.search(r"<(?:xbrli:)?endDate>([^<]+)", body)
        inst = re.search(r"<(?:xbrli:)?instant>([^<]+)", body)
        dim = "explicitMember" in body or "typedMember" in body
        ctx[cid] = {"start": s.group(1) if s else None, "end": e.group(1) if e else (inst.group(1) if inst else None),
                    "instant": inst is not None, "dim": dim}
    facts: dict[str, dict[str, str]] = {}
    for name, cid, val in _FACT.findall(text):
        facts.setdefault(name, {})[cid] = val.strip()
    end = facts.get("DateOfEndOfReportingPeriod", {})
    start = facts.get("DateOfStartOfReportingPeriod", {})
    # the quarter context: duration context whose length is ~3 months and ends at the reporting period end
    # NSE's convention: OneD = current quarter (duration), OneI = quarter-end instant, FourD = year to date.
    #   Older filings reference OneD/FourD without defining them; the reporting dates are then facts under OneD.
    used = {c for v in facts.values() for c in v}
    period_days = None
    if "OneD" in ctx and not ctx["OneD"]["dim"] and ctx["OneD"]["start"] and ctx["OneD"]["end"]:
        period_days = (pd.Timestamp(ctx["OneD"]["end"]) - pd.Timestamp(ctx["OneD"]["start"])).days
    elif "OneD" in used and "OneD" in start and "OneD" in end:
        period_days = (pd.Timestamp(end["OneD"]) - pd.Timestamp(start["OneD"])).days
    q_ctx = "OneD" if ("OneD" in ctx and not ctx["OneD"]["dim"]) or ("OneD" in used and "OneD" not in ctx) else None
    if q_ctx and period_days is not None and not 80 <= period_days <= 100:
        q_ctx = None                       # some Q4 filings put the full year under OneD: not a quarter
    for cid, c in ([] if q_ctx else ctx.items()):
        if c["dim"] or c["instant"] or not c["start"] or not c["end"]:
            continue
        days = (pd.Timestamp(c["end"]) - pd.Timestamp(c["start"])).days
        if 80 <= days <= 100 and (not end or c["end"] in end.values()):
            q_ctx = cid if q_ctx is None or c["end"] > ctx[q_ctx]["end"] else q_ctx
    in_ctx = q_ctx is not None and q_ctx in ctx
    out = {"quarter_start": ctx[q_ctx]["start"] if in_ctx else start.get("OneD", next(iter(start.values()), None)),
           "quarter_end": ctx[q_ctx]["end"] if in_ctx else end.get("OneD", next(iter(end.values()), None)),
           "rounding": next(iter({**facts.get("LevelOfRoundingUsedInFinancialStatements", {}),
                                  **facts.get("LevelOfRounding", {})}.values()), None),
           "currency": next(iter(facts.get("DescriptionOfPresentationCurrency", {}).values()), None),
           "nature": next(iter(facts.get("NatureOfReportStandaloneConsolidated", {}).values()), None),
           "period_days": period_days}
    flow = {"revenue", "total_income", "pbt", "net_profit", "eps_basic", "eps_diluted", "finance_costs", "depreciation",
            "operating_profit_bank"}
    for metric, names in METRICS.items():
        if metric in flow and q_ctx is None and period_days is not None:
            continue                       # no 3-month context: do not mistake annual or YTD figures for a quarter
        for n in names:
            vals = facts.get(n)
            if not vals:
                continue
            v = None
            if q_ctx and q_ctx in vals:
                v = vals[q_ctx]
            elif "OneI" in vals:                     # balance-sheet items: the quarter-end instant
                v = vals["OneI"]
            else:
                for cid, x in vals.items():
                    c = ctx.get(cid, {})
                    if c.get("instant") and not c.get("dim") and c.get("end") == out["quarter_end"]:
                        v = x
                        break
                if v is None and metric in ("face_value", "paid_up_capital"):
                    v = next(iter(vals.values()))
            if v not in (None, ""):
                try:
                    out[metric] = float(v)
                    break
                except ValueError:
                    continue
    return out


def ingest(root: str | Path, symbols: list[str], workers: int = 2, start: str = "2014-01-01", log=print) -> pd.DataFrame:  # pragma: no cover
    root = Path(root)
    xdir = root / "nse" / "xbrl"
    xdir.mkdir(parents=True, exist_ok=True)
    client = Client()
    client.get("https://www.nseindia.com/")
    jobs = []
    for i, s in enumerate(symbols):
        idx = filings_index(root, s, client)
        sel = select_filings(idx, start)
        jobs += [r for _, r in sel.iterrows()]
        log(f"[{i + 1}/{len(symbols)}] {s}: {len(idx)} filings, {len(sel)} selected")

    def fetch(r) -> dict | None:
        fn = xdir / (r["xbrl"].rsplit("/", 1)[-1])
        if not fn.exists():
            blob = client.get(r["xbrl"])
            if blob is None:
                return None
            fn.write_bytes(blob)
        raw = fn.read_bytes()
        f = parse_xbrl(raw.decode("utf-8", "ignore"))
        return {**{k: r[k] for k in ["symbol", "source", "period_end", "consolidated", "audited", "published_at",
                                     "disseminated_at", "available_at", "filing_id", "revision", "xbrl", "bank"]},
                **f, "sha256": hashlib.sha256(raw).hexdigest(), "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    with ThreadPoolExecutor(workers) as ex:
        rows = []
        for n, out in enumerate(ex.map(fetch, jobs)):
            if out is not None:
                rows.append(out)
            if n % 200 == 0:
                log(f"xbrl {n}/{len(jobs)}")
    facts = pd.DataFrame(rows)
    facts.to_csv(root / "nse" / "facts.csv", index=False)
    man = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "symbols": len(symbols), "filings": len(facts),
           "facts_sha256": hashlib.sha256((root / "nse" / "facts.csv").read_bytes()).hexdigest()}
    (root / "nse" / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"facts: {len(facts)} filings -> {root / 'nse' / 'facts.csv'}")
    return facts


def rebuild_facts(root: str | Path, symbols: list[str], start: str = "2014-01-01") -> pd.DataFrame:
    """Re-derive facts.csv from the stored raw files only (no network)."""
    root = Path(root)
    rows = []
    for s in symbols:
        sel = select_filings(filings_index(root, s, None), start)
        for _, r in sel.iterrows():
            fn = root / "nse" / "xbrl" / r["xbrl"].rsplit("/", 1)[-1]
            if not fn.exists():
                continue
            raw = fn.read_bytes()
            rows.append({**{k: r[k] for k in ["symbol", "source", "period_end", "consolidated", "audited", "published_at",
                                              "disseminated_at", "available_at", "filing_id", "revision", "xbrl", "bank"]},
                         **parse_xbrl(raw.decode("utf-8", "ignore")), "sha256": hashlib.sha256(raw).hexdigest()})
    return pd.DataFrame(rows)
