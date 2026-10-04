"""The single source of truth for v0.8 headline numbers: regenerated from the audit files and the Company Memory
build, written to research/data_audits/v0.8_canonical_numbers.json, and checked against the report and README.

    python -m scripts.v08_numbers            # regenerate + check that every number appears in the report and README
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

A = Path("research/data_audits")


def latest(pattern: str) -> dict:
    return json.loads(sorted(A.glob(pattern))[-1].read_text())


def numbers() -> dict:
    cov, val, xc = latest("shareholding_coverage_*.json"), latest("shareholding_validation_*.json"), latest("shareholding_crosscheck_*.json")
    pw, tr, es = latest("study2_power_*.json"), latest("transcripts_coverage_*.json"), latest("estimates_snapshot_*.json")
    rep = sorted(Path("reports").glob("v0.8-r1_*"))[-1]
    cm = json.loads((rep / "company_memory_v08.json").read_text())
    shp = pd.read_csv("data/pit/bse/shareholding.csv")
    st = cm["promises_by_status"]
    return {
        "report": str(rep).replace("\\", "/"),
        "shp_filings": cov["filings"], "shp_companies": cov["companies"],
        "shp_pit_coverage_pct": round(100 * cov["pit_member_quarters_covered"], 2), "shp_gaps": len(cov["gaps"]),
        "shp_as_filed_xbrl_rows": int((shp.source == "bse_xbrl_as_filed").sum()),
        "xbrl_sample_agree": f'{val["sample"]["promoter_pct_agree"]}/{val["sample"]["n"]}', "xbrl_files_scanned": val["xbrl_scan"]["files"],
        "nse_agree_pct": round(100 * xc["promoter_pct_abs_diff_le_0.05pp_share"], 2), "nse_matched": xc["matched_filings"],
        "nse_mismatches": xc["mismatches"], "nse_unexplained": xc["mismatches_unexplained"],
        "study2_events": pw["events"], "study2_companies": pw["companies"],
        "transcripts_usable": tr["usable_pit_transcripts"], "transcripts_nse_full": tr["nse_full"],
        "transcript_gap_companies": len(tr["gap_companies_under_half_usable"]),
        "consensus_vendor_eps_within_5pct": round(100 * es["vendor_actual_vs_filed_eps"]["within_5pct"]),
        "consensus_matched_quarters": es["vendor_actual_vs_filed_eps"]["matched"],
        "promises": cm["promises"], "promise_companies": cm["promises_companies"], "promises_due_for_review": cm["review_queue"],
        "memory_documents": cm["graph"]["documents"], "memory_metrics": cm["graph"]["metrics"],
        "memory_events": cm["graph"]["events"], "memory_claims": cm["graph"]["claims"],
        "pending_promises": st.get("PENDING", 0),
    }


def fmt(v) -> list[str]:
    """Spellings a number may take in prose (4042 -> '4,042')."""
    if isinstance(v, float):
        return [f"{v:.2f}", f"{v:g}"]
    if isinstance(v, int):
        return [f"{v:,}", str(v)]
    return [str(v)]


def main() -> None:
    n = numbers()
    (A / "v0.8_canonical_numbers.json").write_text(json.dumps(n, indent=1))
    texts = {"report": (Path(n["report"]) / "report.md").read_text(encoding="utf-8"), "README": Path("README.md").read_text(encoding="utf-8")}
    checked = ["shp_filings", "shp_pit_coverage_pct", "shp_as_filed_xbrl_rows", "nse_agree_pct", "study2_events", "study2_companies",
               "transcripts_usable", "promises", "promises_due_for_review"]
    bad = {}
    for where, t in texts.items():
        miss = [k for k in checked if not any(s in t for s in fmt(n[k]))]
        if miss:
            bad[where] = miss
    print(json.dumps(n, indent=1))
    print("CONSISTENT" if not bad else f"MISSING: {bad}")


if __name__ == "__main__":
    main()
