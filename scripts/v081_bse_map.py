"""ISIN -> BSE scrip code for every TOP500 member (v0.8.1), choosing among several candidate codes the one whose
shareholding filings cover the period the company traded on NSE (fixes wrong picks such as a delisted duplicate code
or a later re-listing code). Writes marketdata/reference/bse_codes_wide.csv.

    python -m scripts.v081_bse_map          # needs data/pit/bse/scrip_lists/*.json (BSE ListofScripData) + network for new indexes
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from marketdata.bse_shp import Client, FIRST_QTRID, fetch_index, period_end

ROOT = Path("data/pit/bse")


def main() -> None:
    L = []
    for st in ("Active", "Delisted", "Suspended"):
        for x in json.loads((ROOT / "scrip_lists" / f"bse_list_{st}.json").read_text()):
            L.append({"bse_code": x["SCRIP_CD"], "isin": (x.get("ISIN_NUMBER") or "").strip(), "bse_id": (x.get("scrip_id") or "").strip(), "bse_status": st})
    B = pd.DataFrame(L)
    u = pd.read_csv("data/exchange_all/universe_monthly.csv", dtype={"sec": str}, parse_dates=["date"])
    u5 = u[u["in_top500"]]
    a = pd.read_parquet("data/exchange_all/returns.parquet", columns=["date", "sec", "isin", "symbol"])
    life = a.groupby("sec")["date"].agg(["min", "max"])
    ids = a.drop_duplicates(["sec", "isin", "symbol"]).groupby("sec").agg(isins=("isin", set), syms=("symbol", set))
    last_sym = a.sort_values("date").groupby("sec")["symbol"].last()
    cl, rows = Client(), []
    for sec in sorted(u5["sec"].unique()):
        isins, syms = ids.loc[sec, "isins"], ids.loc[sec, "syms"]
        m, how = B[B["isin"].isin(isins)], "isin"
        if m.empty:
            m, how = B[B["bse_id"].isin(syms)], "symbol"
        if m.empty:
            rows.append({"sec": sec, "symbol": last_sym[sec], "bse_code": "", "method": "unmapped", "bse_status": "", "n_codes": 0, "filings_in_life": 0})
            continue
        best, best_n = None, -1
        lo, hi = life.loc[sec, "min"], life.loc[sec, "max"]
        for r in m.drop_duplicates("bse_code").itertuples(index=False):
            idx = fetch_index(ROOT, r.bse_code, cl) or []
            n = 0
            for x in idx:
                if float(x["qtrid"]) < FIRST_QTRID or not (x.get("filing_date_time") or x.get("revised_date_time")):
                    continue
                pe, _ = period_end(x.get("qtr"))
                n += int(pe is not None and lo - pd.Timedelta(days=120) <= pe <= hi + pd.Timedelta(days=1))
            if n > best_n or (n == best_n and r.bse_status == "Active"):
                best, best_n = r, n
        rows.append({"sec": sec, "symbol": last_sym[sec], "bse_code": best.bse_code, "method": how, "bse_status": best.bse_status,
                     "n_codes": int(m["bse_code"].nunique()), "filings_in_life": best_n})
    M = pd.DataFrame(rows)
    M.to_csv("marketdata/reference/bse_codes_wide.csv", index=False)
    print(len(M), M["method"].value_counts().to_dict(), "zero filings in life:", int((M["filings_in_life"] == 0).sum() - (M["method"] == "unmapped").sum()),
          M[(M["filings_in_life"] == 0) & (M["method"] != "unmapped")]["symbol"].tolist())


if __name__ == "__main__":
    main()
