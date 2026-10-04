"""v0.8 data acquisition: build + audit the new point-in-time datasets.

    python -m scripts.v08_data shp-build          # parse BSE shareholding filings -> data/pit/bse/shareholding.csv
    python -m scripts.v08_data shp-crosscheck     # NSE master (Sep 2021+) vs BSE: promoter %, timestamps (network)
    python -m scripts.v08_data shp-audit          # PIT coverage audit -> research/data_audits/shareholding_coverage_<date>.json
    python -m scripts.v08_data shp-validate       # XBRL parser vs structured sample + full XBRL scan (network for the sample)
    python -m scripts.v08_data shp-events         # Study 2 power count (events only; no returns)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.bse_shp import load
from marketdata.pit import default_universe
from scripts.fetch_bse_shp import codes

ROOT = Path("data/pit/bse")
OUT = ROOT / "shareholding.csv"
AUDITS = Path("research/data_audits")


def shp_build() -> pd.DataFrame:
    df = load(ROOT, codes())
    tmp = OUT.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(OUT)
    print(f"{len(df)} filings, {df.symbol.nunique()} companies -> {OUT}")
    return df


def read() -> pd.DataFrame:
    df = pd.read_csv(OUT, parse_dates=["period_end"])
    for c in ("available_at", "filed_at", "revised_at"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
    return df


def shp_crosscheck() -> dict:  # pragma: no cover - network
    from marketdata.nse_filings import Client

    d = Path("data/pit/nse/shp_master")
    d.mkdir(parents=True, exist_ok=True)
    cl = Client(pause=0.3)
    df = read()
    rows = []
    for sym in sorted(df.symbol.unique()):
        p = d / f"{sym}.json"
        if not p.exists():
            b = cl.get(f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={sym.replace('&', '%26')}")
            if b is None:
                continue
            p.write_bytes(b)
        try:
            items = json.loads(p.read_bytes())
        except ValueError:
            continue
        for x in items if isinstance(items, list) else []:
            rows.append({"symbol": sym, "period_end": pd.to_datetime(x.get("date"), format="%d-%b-%Y", errors="coerce"),
                         "nse_promoter_pct": pd.to_numeric(x.get("pr_and_prgrp"), errors="coerce"),
                         "nse_broadcast": pd.to_datetime(x.get("broadcastDate"), format="%d-%b-%Y %H:%M:%S", errors="coerce"),
                         "nse_revised": x.get("revisedData"), "nse_revision_date": x.get("revisionDate") or x.get("revisedDate")})
    nse = pd.DataFrame(rows).dropna(subset=["period_end"])
    nse = nse.sort_values("nse_broadcast").groupby(["symbol", "period_end"]).head(1)        # first NSE filing per period
    bse = df[df.kind == "quarterly"].sort_values("available_at").groupby(["symbol", "period_end"]).head(1)
    m = bse.merge(nse, on=["symbol", "period_end"], how="inner")
    m = m[m["period_end"] >= pd.Timestamp("2021-09-30")]          # NSE master is complete only from Sep 2021
    allv = df[df.kind == "quarterly"][["symbol", "period_end", "promoter_pct"]]
    m["pct_diff"] = (m["promoter_pct"] - m["nse_promoter_pct"]).abs()
    m["bse_any_version_matches"] = [bool(((allv.symbol == s) & (allv.period_end == p) & ((allv.promoter_pct - v).abs() <= 0.05)).any())
                                    for s, p, v in zip(m.symbol, m.period_end, m.nse_promoter_pct)]
    m["ts_diff_h"] = (m["available_at"].dt.tz_localize(None) - m["nse_broadcast"]).dt.total_seconds() / 3600
    res = {"matched_filings": int(len(m)), "symbols": int(m.symbol.nunique()),
           "promoter_pct_abs_diff_le_0.05pp_share": round(float((m.pct_diff <= 0.05).mean()), 4),
           "window": "periods from 2021-09-30 (NSE master coverage)",
           "mismatches": int((m.pct_diff > 0.05).sum()),
           "mismatches_nse_record_is_revised": int(((m.pct_diff > 0.05) & (m.nse_revised.astype(str).str.upper() != "N")).sum()),
           "mismatches_matched_by_a_bse_revision": int(((m.pct_diff > 0.05) & m.bse_any_version_matches).sum()),
           "mismatches_unexplained": int(((m.pct_diff > 0.05) & (m.nse_revised.astype(str).str.upper() == "N") & ~m.bse_any_version_matches).sum()),
           "all_mismatches": m[m.pct_diff > 0.05][["symbol", "period_end", "promoter_pct", "nse_promoter_pct", "nse_revised",
                                                   "bse_any_version_matches"]].astype(str).to_dict("records"),
           "bse_minus_nse_hours_quantiles": {str(k): round(float(v), 2) for k, v in m.ts_diff_h.quantile([.01, .1, .5, .9, .99]).items()},
           "bse_later_than_nse_by_24h_share": round(float((m.ts_diff_h > 24).mean()), 4)}
    AUDITS.mkdir(parents=True, exist_ok=True)
    (AUDITS / f"shareholding_crosscheck_{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("promoter_pct_mismatches", "all_mismatches")}, indent=1, default=str))
    return res


def shp_audit() -> dict:
    df = read()
    q = df[df.kind == "quarterly"]
    uni = default_universe()
    qe = pd.date_range("2016-03-31", "2026-06-30", freq="QE")
    cover, lags, gaps = [], [], []
    for d in qe:
        mem = uni.members(d)
        for s in mem:
            x = q[(q.symbol == s) & (q.period_end == d)]
            ok = len(x) > 0
            first = x.available_at.min() if ok else pd.NaT
            lag = (first.tz_localize(None) - d).days if ok else np.nan
            cover.append({"q": d, "symbol": s, "has": ok, "lag": lag, "revised_only": bool(x.revised_only.all()) if ok else None})
            if not ok:
                gaps.append(f"{s} {d.date()}")
    c = pd.DataFrame(cover)
    by_year = c.groupby(c.q.dt.year).agg(member_quarters=("has", "size"), covered=("has", "mean"),
                                         median_lag_days=("lag", "median"), late_over_45d=("lag", lambda v: float((v > 45).mean())))
    res = {"dataset": str(OUT), "filings": int(len(df)), "quarterly": int(len(q)), "event_based": int((df.kind == "event").sum()),
           "companies": int(df.symbol.nunique()), "first_period": str(q.period_end.min().date()), "last_period": str(q.period_end.max().date()),
           "pit_member_quarters": int(len(c)), "pit_member_quarters_covered": round(float(c.has.mean()), 4),
           "revised_only_share_of_covered": round(float(c[c.has].revised_only.mean()), 4),
           "available_after_45d_share": round(float((c.lag > 45).mean()), 4),
           "gaps": gaps,
           "by_year": {int(k): {kk: (round(float(vv), 4) if isinstance(vv, (float, np.floating)) else int(vv)) for kk, vv in v.items()}
                       for k, v in by_year.to_dict("index").items()},
           "promoter_pledge_any_share": round(float((q.pledged_shares > 0).mean()), 4),
           "companies_with_any_pledge": sorted(q[q.pledged_shares > 0].symbol.unique().tolist()),
           "no_promoter_companies": sorted(q[q.get("no_promoter", False) == True].symbol.unique().tolist()),  # noqa: E712
           "pit_rule": "usable from BSE filing timestamp (IST); revised-only quarters from the revision timestamp; "
                       "periods before Mar-2016 not used (BSE gives no timestamp)",
           "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    AUDITS.mkdir(parents=True, exist_ok=True)
    p = AUDITS / f"shareholding_coverage_{time.strftime('%Y-%m-%d')}.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("gaps", "by_year")}, indent=1, default=str))
    print("gaps", len(gaps), gaps[:20])
    print(pd.DataFrame(res["by_year"]).T)
    return res


def shp_validate(n: int = 60, seed: int = 8) -> dict:  # pragma: no cover - network (fetches sample XBRL)
    """(1) Parser check: as-filed XBRL vs the structured record on a random sample of filings whose structured record IS
    their own version (so both must agree). (2) Full scan of every stored XBRL: encumbrance elements present, values
    out of range, and as-filed rows whose encumbrance is unknown (flag says encumbered, no figure)."""
    import gzip
    import random

    from marketdata.bse_shp import MATCH_TOLERANCE, Client, fetch_xbrl, promoter_summary, xbrl_summary

    random.seed(seed)
    own = []
    for ip in (ROOT / "shp").glob("*/index.json"):
        for r in json.loads(ip.read_text())["rows"]:
            ts = r.get("filing_date_time") or r.get("revised_date_time")
            if r["qtrid"] < 89 or not ts or not r.get("XbrlFile"):
                continue
            pj = json.loads((ip.parent / f"{r['qtrid']:.2f}_promoter.json").read_text())
            a = (pj.get("Table") or [{}])[0].get("Fld_AuthoriseDate")
            if a and abs(pd.Timestamp(a) - pd.Timestamp(ts)) <= MATCH_TOLERANCE:
                own.append((ip.parent.name, r, pj))
    cl, rows = Client(pause=0.2), []
    for code, r, pj in random.sample(own, n):
        fetch_xbrl(ROOT, code, r["XbrlFile"], cl)
        x = xbrl_summary(gzip.decompress((ROOT / "shp" / code / "xbrl" / (r["XbrlFile"] + ".gz")).read_bytes()).decode("utf-8-sig", "replace"))
        a = promoter_summary(pj)
        rows.append({"code": code, "qtrid": r["qtrid"], "api_pct": a.get("promoter_pct"), "x_pct": x.get("promoter_pct"),
                     "api_enc": a.get("encumbered_shares"), "x_enc": x.get("encumbered_shares")})
    v = pd.DataFrame(rows)
    pct_ok = ((v.api_pct - v.x_pct).abs() <= 0.05)
    enc_ok = ((v.api_enc - v.x_enc).abs() <= 1)
    scan = {"files": 0, "with_pledge_elements": 0, "taxonomies": {}}
    for f in (ROOT / "shp").glob("*/xbrl/*.gz"):
        t = gzip.decompress(f.read_bytes()).decode("utf-8-sig", "replace")
        scan["files"] += 1
        import re as _re
        tx = _re.search(r"/xbrl/shp/([^/\"]+)/in-bse-shp\"", t)
        k = tx.group(1) if tx else "unknown"
        scan["taxonomies"][k] = scan["taxonomies"].get(k, 0) + 1
        scan["with_pledge_elements"] += int(bool(_re.search(r"PledgedOrEncumberedNumberOfShares|NumberOfSharesEncumbered", t)))
    df = read()
    asf = df[df.source == "bse_xbrl_as_filed"]
    res = {"sample": {"n": int(len(v)), "design": "random own-version filings (structured record authorised within 5 min of the filing)",
                      "seed": seed, "promoter_pct_agree": int(pct_ok.sum()), "encumbered_shares_agree": int(enc_ok.sum()),
                      "sample_with_nonzero_encumbrance": int((v.api_enc > 0).sum())},
           "xbrl_scan": scan, "as_filed_rows": int(len(asf)),
           "as_filed_rows_encumbrance_unknown": int((asf.encumbered_shares.isna() & (asf.no_promoter.astype(str) != "True")).sum()),
           "pct_out_of_range_rows": int(sum(((df[c] < 0) | (df[c] > 100.0001)).sum() for c in df.columns if c.endswith("_pct") or c.endswith("_of_promoter"))),
           "structured_rows_rescaled_x100": int(df.get("pct_rescaled", pd.Series(dtype=object)).astype(str).eq("True").sum()),
           "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (AUDITS / f"shareholding_validation_{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return res


def shp_events(threshold: float = 5.0, start: str = "2016-05-01") -> dict:
    """Study 2 power count (no returns): H2a events = first (as-filed) quarterly filing per (symbol, period) whose
    encumbrance (% of promoter holding) is >= threshold pp above the maximum of the previous four periods' first filings
    (v0.8 Amendment 1), for PIT NIFTY 50 members at the filing time. Revisions are not events."""
    df = read()
    q = df[df.kind == "quarterly"].sort_values(["symbol", "period_end", "available_at"]).groupby(["symbol", "period_end"]).head(1)
    q = q.sort_values(["symbol", "period_end"]).copy()
    prev_max = q.groupby("symbol")["encumbered_pct_of_promoter"].transform(lambda v: v.shift(1).rolling(4, min_periods=1).max())
    q["dE"] = q["encumbered_pct_of_promoter"] - prev_max          # Amendment 1: above the max of the previous four filings
    uni = default_universe()
    q["member"] = [s in uni.members(pd.Timestamp(t.tz_convert("Asia/Kolkata").date())) for s, t in zip(q.symbol, q.available_at)]
    m = q[q.member & (q.available_at >= pd.Timestamp(start, tz="Asia/Kolkata"))]
    ev = m[m.dE >= threshold]
    res = {"rule": f"first as-filed quarterly filing per period; encumbrance >= {threshold}pp above the max of the previous four "
                   f"quarterly filings (Amendment 1); PIT NIFTY 50 member; from {start}",
           "events": int(len(ev)), "companies": int(ev.symbol.nunique()), "by_company": ev.symbol.value_counts().to_dict(),
           "events_list": ev[["symbol", "period_label", "available_at", "encumbered_pct_of_promoter", "dE"]].astype(str).to_dict("records"),
           "required_by_preregistration": {"events": 100, "companies": 30}}
    res["study2_powered"] = res["events"] >= 100 and res["companies"] >= 30
    (AUDITS / f"study2_power_{time.strftime('%Y-%m-%d')}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "events_list"}, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shp-build", "shp-crosscheck", "shp-audit", "shp-validate", "shp-events"])
    a = ap.parse_args()
    {"shp-build": shp_build, "shp-crosscheck": shp_crosscheck, "shp-audit": shp_audit, "shp-validate": shp_validate,
     "shp-events": shp_events}[a.cmd]()


if __name__ == "__main__":
    main()
