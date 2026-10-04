"""Import human verdicts from the Promise Review Desk into Company Memory (v1.4).

    python -m scripts.import_promise_reviews --from data/intel/desk_reviews [--queue data/intel/promise_review_2026-10-04.json]
                                             [--reviewer "Rajesh"] [--dry-run]

`--from` is a folder of review documents exported from the desk's `reviews` collection (one JSON per claim; either the
bare review or {"id", "data", ...}), or a single JSON file holding a list of them. Each review becomes one row of the
standard review CSV (intel.promise_review.QUEUE_COLUMNS) and goes through `apply_reviews`, which re-validates it:
verdict in MET / PARTIAL / MISSED / UNRESOLVABLE, a named reviewer, evidence dated after the promise, and the evidence
quote verbatim in that document. Nothing is scored automatically; a review identical to the claim's latest stored
review is skipped so re-imports do not duplicate the append-only table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from intel.promise_review import QUEUE_COLUMNS, apply_reviews, ensure

DEFAULT_QUEUE = Path("data/intel/promise_review_2026-10-04.json")


def _norm(s) -> str:
    return " ".join(str(s or "").split())


def read_reviews(src: Path) -> list[dict]:
    src = Path(src)
    files = sorted(src.rglob("*.json")) if src.is_dir() else [src]
    out = []
    for f in files:
        j = json.loads(f.read_text(encoding="utf-8"))
        for r in (j if isinstance(j, list) else [j]):
            r = r.get("data", r) if isinstance(r, dict) else {}
            if r.get("claim_id") and r.get("verdict"):
                out.append(r)
    return out


def to_rows(reviews: list[dict], queue: list[dict], reviewer: str | None = None) -> pd.DataFrame:
    q = {p["claim_id"]: p for p in queue}
    rows = []
    for r in reviews:
        p = q.get(r["claim_id"], {})
        rows.append({"claim_id": r["claim_id"], "symbol": p.get("symbol", r.get("symbol", "")), "speaker": p.get("speaker", ""),
                     "said": p.get("said", ""), "quote": p.get("quote", ""), "metric": p.get("metric", ""), "target": p.get("target", ""),
                     "due": p.get("due", ""), "suggested": p.get("suggested", ""), "candidates": len(p.get("candidates", [])),
                     "verdict": str(r["verdict"]).upper(), "evidence_doc_id": r.get("evidence_doc_id", ""),
                     "evidence_quote": _norm(r.get("evidence_quote")), "reviewer": reviewer or f"desk:{r.get('reviewer_id', '')}".rstrip(":"),
                     "note": r.get("note", "")})
    return pd.DataFrame(rows, columns=QUEUE_COLUMNS)


def drop_already_applied(G, df: pd.DataFrame) -> pd.DataFrame:
    ensure(G)
    keep = []
    for row in df.to_dict("records"):
        last = G.db.execute("SELECT verdict, evidence_doc, evidence_quote FROM promise_reviews WHERE claim_id=? "
                            "ORDER BY reviewed_at DESC LIMIT 1", (row["claim_id"],)).fetchone()
        if last and (last[0], last[1] or "", _norm(last[2])) == (row["verdict"], row["evidence_doc_id"] or "", _norm(row["evidence_quote"])):
            continue
        keep.append(row)
    return pd.DataFrame(keep, columns=QUEUE_COLUMNS)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE))
    ap.add_argument("--reviewer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from intel.graph import EvidenceGraph
    queue = json.loads(Path(a.queue).read_text(encoding="utf-8"))
    df = to_rows(read_reviews(Path(a.src)), queue, a.reviewer)
    G = EvidenceGraph()
    df = drop_already_applied(G, df)
    out_csv = Path("data/intel") / f"desk_reviews_{pd.Timestamp.now(tz='Asia/Kolkata'):%Y%m%dT%H%M}.csv"
    res = {"csv": None, "rows": int(len(df))}
    if len(df):
        df.to_csv(out_csv, index=False)
        res["csv"] = str(out_csv)
    if not a.dry_run and len(df):
        res.update(apply_reviews(G, out_csv))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
