"""v0.8.1: whole-market adjusted returns and point-in-time TOP100/200/500 liquidity universes, plus their audit.

    python -m scripts.fetch_nse_all --years 2015-2026          # whole-market bhavcopy (network)
    python -m scripts.v081_universe build                      # panel -> adjusted returns + monthly universes
    python -m scripts.v081_universe audit                      # -> research/data_audits/universe_prices_<date>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.nse_universe import ROOT, adjust, build_panel, liquidity_universe

RET = ROOT / "returns.parquet"
UNI = ROOT / "universe_monthly.csv"
SIZES = (100, 200, 500)


def build() -> None:
    """Year by year (the machine has ~3 GB): adjust, keep the compact columns, then pivot traded value once."""
    import gc

    from marketdata.nse_corp_actions import load as load_ca

    t0 = time.time()
    if not (ROOT / "panel.parquet").exists():
        build_panel()
    ca = load_ca()
    keep = ["date", "sec", "isin", "symbol", "open", "close", "prev_close", "value", "cc", "on", "oc", "factor", "dividend", "flag"]
    parts = []
    for y in range(2015, 2027):
        d = pd.read_parquet(ROOT / "panel.parquet", filters=[("date", ">=", pd.Timestamp(f"{y}-01-01")), ("date", "<", pd.Timestamp(f"{y + 1}-01-01"))])
        if d.empty:
            continue
        x = adjust(d, ca)[keep]
        for c in ("open", "close", "prev_close", "value", "cc", "on", "oc", "factor", "dividend"):
            x[c] = x[c].astype("float32")
        for c in ("sec", "isin", "symbol", "flag"):
            x[c] = x[c].astype("category")
        parts.append(x)
        del d
        gc.collect()
    a = pd.concat(parts, ignore_index=True)
    for c in ("sec", "isin", "symbol", "flag"):
        a[c] = a[c].astype(str)
    a.to_parquet(RET)
    value = a.pivot(index="date", columns="sec", values="value").astype("float32")
    ends = value.index.to_series().groupby(value.index.to_period("M")).max().tolist()
    ends = [t for t in ends if t.to_period("M") < value.index.max().to_period("M")]     # the current month is incomplete
    last_sym = a.sort_values("date").groupby("sec")[["isin", "symbol"]].last()
    rows = []
    for t in ends:
        top = liquidity_universe(value, t, max(SIZES))
        for k, sec in enumerate(top, 1):
            rows.append({"date": t.date(), "rank": k, "sec": sec, "in_top100": k <= 100, "in_top200": k <= 200, "in_top500": k <= 500})
    u = pd.DataFrame(rows)
    sym_at = a.set_index(["date", "sec"])[["isin", "symbol"]]
    u = u.merge(sym_at, left_on=[pd.to_datetime(u["date"]), "sec"], right_index=True, how="left")
    u.drop(columns=[c for c in u.columns if c.startswith("key_")], errors="ignore").to_csv(UNI, index=False)
    print(f"returns {len(a):,} rows; universe {len(u):,} rows over {u['date'].nunique()} month-ends; {time.time() - t0:.0f}s")


def _member_flags(a: pd.DataFrame, u: pd.DataFrame) -> dict:
    """Flagged or |r|>50% days for securities that were TOP500 members at the previous month-end."""
    x = a[(a["flag"] != "") | (a["cc"].abs() > 0.5)].copy()
    ends = np.sort(u["date"].unique())
    pos = np.searchsorted(ends, x["date"].values, side="left") - 1
    x = x[pos >= 0]
    x["me"] = ends[pos[pos >= 0]]
    m = x.merge(u[u["in_top500"]][["date", "sec"]].rename(columns={"date": "me"}), on=["me", "sec"])
    return {"count": int(len(m)), "by_flag": m["flag"].replace("", "none(|r|>50%)").value_counts().to_dict(),
            "list": m[["date", "symbol", "cc", "flag"]].astype(str).to_dict("records")}


def audit() -> dict:
    a = pd.read_parquet(RET, columns=["date", "sec", "isin", "symbol", "cc", "flag", "factor", "dividend"])
    u = pd.read_csv(UNI, parse_dates=["date"])
    flags = a["flag"].value_counts().to_dict()
    ext = a[a["flag"] == "unexplained_extreme"].sort_values("date")
    res = {"sessions": int(a["date"].nunique()), "first": str(a["date"].min().date()), "last": str(a["date"].max().date()),
           "securities": int(a["sec"].nunique()), "rows": int(len(a)),
           "corporate_action_days_with_price_factor": int((a["factor"] != 1).sum()), "dividend_days": int((a["dividend"] > 0).sum()),
           "flags": {k: int(v) for k, v in flags.items() if k},
           "unexplained_extremes_sample": ext[["date", "symbol", "cc"]].astype(str).head(30).to_dict("records"),
           "abs_daily_return_gt_50pct_after_adjustment": int((a["cc"].abs() > 0.5).sum()),
           "universe_month_ends": int(u["date"].nunique()), "universe_first": str(u["date"].min().date()),
           "universe_size_min": {f"top{n}": int(u[u[f"in_top{n}"]].groupby("date").size().min()) for n in SIZES},
           "distinct_members": {f"top{n}": int(u[u[f"in_top{n}"]]["sec"].nunique()) for n in SIZES},
           "members_later_delisted_top500": int((lambda m: m[m["last"] < a["date"].max() - pd.Timedelta(days=30)]["sec"].nunique())(
               u[u["in_top500"]].merge(a.groupby("sec")["date"].max().rename("last"), left_on="sec", right_index=True))),
           "top500_member_days_flagged": _member_flags(a, u),
           "validation_vs_nifty50_panel": "see v0.8.1 report: 99.26% of 193,110 stock-days within 10 bp of the vendor-corrected NIFTY 50 panel",
           "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = Path("research/data_audits") / f"universe_prices_{time.strftime('%Y-%m-%d')}.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "unexplained_extremes_sample"}, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "audit"])
    a = ap.parse_args()
    {"build": build, "audit": audit}[a.cmd]()


if __name__ == "__main__":
    main()
