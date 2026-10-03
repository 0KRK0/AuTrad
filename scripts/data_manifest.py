"""Write an immutable data-integrity manifest: exactly which data, sources, corrections and code produced the
v0.3.1 / v0.4 conclusions.

    python -m scripts.data_manifest --label v0.3.1 --audit reports/data_integrity_<ts>
Creates research/manifests/data_integrity_<label>.json (refuses to overwrite) plus a .sha256 sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from marketdata.corporate_actions import TRANSFORMS_VERSION
from marketdata.store import DataStore

PIPELINE_CODE = ["marketdata/store.py", "marketdata/exchange.py", "marketdata/corporate_actions.py", "marketdata/audit.py",
                 "marketdata/pit.py", "marketdata/universe.py", "marketdata/nse_filings.py", "marketdata/fundamentals.py",
                 "scripts/fetch_data.py", "scripts/fetch_exchange.py", "scripts/fetch_fundamentals.py", "scripts/data_audit.py",
                 "scripts/data_impact.py", "execution/costs.py", "lab/panel.py", "lab/engine.py"]
REFERENCE = ["marketdata/reference/corporate_actions.csv", "marketdata/reference/bse_codes.csv",
             "marketdata/reference/verified_real_moves.csv", "marketdata/reference/nifty50_pit_snapshots.csv"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        r = subprocess.run(["git", "--no-optional-locks", *args], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="v0.3.1")
    ap.add_argument("--data-dir", default="data/snap_20261001")
    ap.add_argument("--exchange-dir", default="data/exchange")
    ap.add_argument("--pit-dir", default="data/pit")
    ap.add_argument("--audit", required=True, help="reports/data_integrity_<ts> folder")
    ap.add_argument("--out", default="research/manifests")
    a = ap.parse_args()
    out = Path(a.out) / f"data_integrity_{a.label}.json"
    if out.exists():
        sys.exit(f"{out} exists: manifests are immutable; use a new label")
    st = DataStore(a.data_dir, a.exchange_dir)
    man = st.manifest()["assets"]
    audit = Path(a.audit)
    summ = json.loads((audit / "summary.json").read_text())
    ev = pd.read_csv(audit / "events.csv")
    art = pd.read_csv(audit / "artifacts.csv")
    lvl = pd.read_csv(audit / "level_check.csv")
    q = pd.read_csv(audit / "quality.csv")
    bse_man = json.loads((Path(a.exchange_dir) / "bse" / "manifest.json").read_text())
    B = pd.concat([pd.read_csv(p, usecols=["date"]) for p in sorted((Path(a.exchange_dir) / "bse").glob("*.csv.gz"))])
    sessions = sorted(B["date"].unique())
    pit = Path(a.pit_dir) / "nse"
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True, timeout=900).stdout
    tline = [x for x in tests.splitlines() if " passed" in x or " failed" in x][-1:] or ["unknown"]
    doc = {
        "manifest_version": 1, "label": a.label, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Exactly which data, sources, corrections and code produced the AuTrad conclusions recorded under this label.",
        "dataset": {"vendor_snapshot": Path(a.data_dir).name, "vendor_manifest_hash": st.manifest_hash(),
                    "instruments": len(man), "exchange_cross_checked": int(len(lvl)),
                    "assets": {s: {k: v.get(k) for k in ["ticker", "source", "asset_class", "fetched_at", "sha256"]} |
                               {"rows": v.get("quality", {}).get("rows"), "first": v.get("quality", {}).get("first"),
                                "last": v.get("quality", {}).get("last")} for s, v in sorted(man.items())}},
        "sources": {
            "vendor": "Yahoo Finance daily OHLCV (NSE listings), fetched per asset at the fetched_at times above",
            "exchange": {"name": "BSE daily equity bhavcopy (official, unadjusted)", "url": bse_man.get("source"),
                         "sessions": len(sessions), "first_session": sessions[0], "last_session": sessions[-1],
                         "file_sha256": bse_man.get("files"), "cache_hash": st.exchange_hash(),
                         "missing_sessions_vs_nse": ["2021-12-29"]},
            "corporate_action_registry": "marketdata/reference/corporate_actions.csv (each row cites a public source)",
            "fundamentals": ({"name": "NSE financial-results filings (XBRL), legacy + Integrated Filing",
                              "manifest": json.loads((pit / "manifest.json").read_text()) if (pit / "manifest.json").exists() else None,
                              "facts_sha256": sha(pit / "facts.csv") if (pit / "facts.csv").exists() else None}),
        },
        "reference_files": {p: {"sha256": sha(Path(p)), "rows": sum(1 for _ in open(p)) - 1} for p in REFERENCE if Path(p).exists()},
        "corrections": {
            "transforms_version": TRANSFORMS_VERSION,
            "events_applied": {"total": int(len(ev)), "by_kind": ev["kind"].value_counts().to_dict(), "by_method": ev["method"].value_counts().to_dict()},
            "non_dividend_events": ev[ev["kind"] != "dividend"][["symbol", "date", "kind", "method", "factor", "vendor_factor", "source"]]
            .fillna("").to_dict("records"),
            "vendor_artifacts_removed": art.to_dict("records"),
            "closes_or_stale_bars_replaced": int(lvl["prints_replaced"].sum()), "opens_replaced": int(lvl["opens_replaced"].sum()),
            "suspect_windows_flagged": summ.get("suspects"), "verified_real_moves": "marketdata/reference/verified_real_moves.csv",
        },
        "quality": {"asset_days_below_0.5": int(q["days_below_0.5"].sum()), "asset_days_replaced": int(q["days_replaced"].sum()),
                    "asset_days_unconfirmed": int(q["days_unconfirmed"].sum()), "mean_score": float(q["mean"].mean())},
        "exchange_comparison": {"stock_days": summ["stock_days"], "checks": summ["checks"], "exdates_checked": summ["exdates_checked"],
                                "missed_adjustments": summ["missed_adjustments"]},
        "code": {"pipeline_files": {p: sha(Path(p)) for p in PIPELINE_CODE if Path(p).exists()},
                 "git_head": git("rev-parse", "--verify", "HEAD") or "no commits yet (this manifest is part of the first commit)",
                 "git_branch": git("symbolic-ref", "--short", "HEAD") or "detached", "git_dirty_files": len(git("status", "--porcelain").splitlines()),
                 "python": platform.python_version(), "pandas": pd.__version__},
        "tests": {"command": "pytest -q", "result": tline[0]},
        "audit_report": str(audit),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, default=str)
    with open(out, "x", encoding="utf-8") as fh:                  # 'x': never overwrite
        fh.write(text)
    Path(str(out) + ".sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + f"  {out.name}\n")
    print(f"written {out}  sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
