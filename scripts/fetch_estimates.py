"""Daily analyst-consensus snapshot for the current NIFTY 50 (run once per day, e.g. after the close).

    python -m scripts.fetch_estimates --budget 160          # resumable within the day
    python -m scripts.fetch_estimates --audit               # snapshot quality + EPS cross-check vs filed results
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from marketdata.estimates import flatten, snapshot, surprises
from marketdata.pit import default_universe

ROOT = Path("data/estimates/yahoo")


def members() -> list[str]:
    return sorted(default_universe().members(pd.Timestamp.today().normalize()))


def members_fo() -> list[str]:
    """v1.3 Part B universe: every stock with single-stock futures on the latest F&O session in the store."""
    from marketdata.nse_fo import load_fo
    f = load_fo(2026, 2026)
    last = f["date"].max()
    return sorted(set(f[(f["kind"] == "STK") & (f["date"] == last)]["symbol"]) | set(members()))


def run(budget: float, universe: str = "n50") -> None:  # pragma: no cover - network
    t0, done = time.time(), 0
    syms = members_fo() if universe == "fo" else members()
    for s in syms:
        if time.time() - t0 > budget:
            break
        try:
            snapshot(s, ROOT)
            done += 1
        except Exception as e:  # noqa: BLE001
            print("failed", s, str(e)[:120])
    day = time.strftime("%Y-%m-%d", time.gmtime())
    have = len(list((ROOT / day).glob("*.json")))
    print(f"{day}: {have}/{len(syms)} snapshots")


def audit() -> dict:
    files = sorted(ROOT.glob("*/*.json"))
    flat = pd.DataFrame([x for p in files for x in flatten(p)])
    sur = pd.DataFrame([x for p in files for x in surprises(p)])
    errs = sum(1 for p in files if json.loads(p.read_text())["errors"])
    res = {"snapshot_days": sorted({p.parent.name for p in files}), "snapshots": len(files), "with_errors": errs,
           "rows": int(len(flat))}
    if len(flat):
        fy = flat[flat.horizon == "0y"]
        res.update({"fy0_coverage": int(fy.eps_avg.notna().sum()), "fy0_median_analysts": float(fy.eps_n.median()),
                    "fy0_few_analysts": int(fy["flags"].str.contains("few_analysts").sum()),
                    "fy0_trend_mismatch": int(fy["flags"].str.contains("trend_vs_estimate_mismatch").sum()),
                    "fy0_trend_mismatch_symbols": fy[fy["flags"].str.contains("trend_vs_estimate_mismatch")].symbol.tolist(),
                    "q0_coverage": int(flat[flat.horizon == "0q"].eps_avg.notna().sum())})
    if len(sur):
        f = pd.read_csv("data/pit/nse/facts.csv")
        if "eps_basic" in f.columns or "eps" in f.columns:
            col = "eps_basic" if "eps_basic" in f.columns else "eps"
            f = f[(f["period_days"].between(80, 100)) & (f["consolidated"].astype(str).str.lower().isin(["true", "1"]))]
            f["q"] = pd.to_datetime(f["period_end"]).dt.strftime("%Y-%m-%d")
            m = sur.merge(f[["symbol", "q", col]].rename(columns={"q": "quarter"}).drop_duplicates(["symbol", "quarter"], keep="last"),
                          on=["symbol", "quarter"], how="inner")
            m["abs_rel"] = (m.eps_actual / m[col] - 1).abs()
            res["vendor_actual_vs_filed_eps"] = {"matched": int(len(m)), "within_5pct": round(float((m.abs_rel <= 0.05).mean()), 4) if len(m) else None,
                                                 "worst": m.sort_values("abs_rel", ascending=False).head(8)[["symbol", "quarter", "eps_actual", col]]
                                                 .astype(str).to_dict("records")}
        else:
            res["vendor_actual_vs_filed_eps"] = "facts.csv has no EPS column"
    p = Path("research/data_audits") / f"estimates_snapshot_{time.strftime('%Y-%m-%d')}.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--universe", choices=["n50", "fo"], default="fo", help="fo = all stock-futures names (v1.3 Part B)")
    a = ap.parse_args()
    audit() if a.audit else run(a.budget, a.universe)


if __name__ == "__main__":
    main()
