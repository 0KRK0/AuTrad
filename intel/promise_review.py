"""Management promise -> outcome workflow:  promise -> due date -> outcome evidence -> HUMAN verification ->
MET / PARTIAL / MISSED (or UNRESOLVABLE).

The machine only prepares the work: it lists promises whose due date has passed, attaches candidate outcome passages
and, where a filed number can be computed, a *suggested* verdict. It never sets MET/PARTIAL/MISSED itself. A verdict
enters Company Memory only through `apply_reviews`, which checks that

  * the verdict is one of MET, PARTIAL, MISSED, UNRESOLVABLE and a reviewer is named;
  * for MET/PARTIAL/MISSED, the cited outcome document exists, became public after the promise was made, and the
    cited evidence quote appears verbatim (whitespace-normalised) in that document's text.

Reviews are append-only (table promise_reviews); the latest valid review is the claim's status. A management
execution record is reported per company only as counts, and is labelled insufficient below MIN_OBS verified
promises. It is not a trading factor (v0.8 pre-registration, section 4).
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

VERDICTS = ("MET", "PARTIAL", "MISSED", "UNRESOLVABLE")
MIN_OBS = 10
SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_files (doc_id TEXT PRIMARY KEY, path TEXT);
CREATE TABLE IF NOT EXISTS promise_reviews (review_id TEXT PRIMARY KEY, claim_id TEXT, verdict TEXT, evidence_doc TEXT,
  evidence_quote TEXT, reviewer TEXT, reviewed_at TEXT, note TEXT, suggested TEXT);
"""
QUEUE_COLUMNS = ["claim_id", "symbol", "speaker", "said", "quote", "metric", "target", "due", "suggested", "candidates",
                 "verdict", "evidence_doc_id", "evidence_quote", "reviewer", "note"]


def ensure(G) -> None:
    G.db.executescript(SCHEMA)


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def review_queue(G, asof: pd.Timestamp, path: Path) -> pd.DataFrame:
    """Promises due on or before `asof` without a verified review, with candidate outcome passages."""
    ensure(G)
    done = {r[0] for r in G.db.execute("SELECT DISTINCT claim_id FROM promise_reviews")}
    rows = []
    for p in G.promises():
        if not p["due"] or pd.Timestamp(p["due"]) > asof or p["claim_id"] in done:
            continue
        cands = G.db.execute("SELECT src, note FROM links WHERE dst=? AND rel IN ('outcome','outcome_candidate')", (p["claim_id"],)).fetchall()
        sug = p["status"].split("SUGGESTED ")[1].split(" ")[0] if "SUGGESTED " in (p["status"] or "") else ""
        rows.append({"claim_id": p["claim_id"], "symbol": p["entity"], "speaker": p["speaker"], "said": (p["made_at"] or "")[:10],
                     "quote": p["quote"], "metric": p["metric"], "target": p["target"], "due": p["due"], "suggested": sug,
                     "candidates": " || ".join(f"{d}: {n}" for d, n in cands)[:2000],
                     "verdict": "", "evidence_doc_id": "", "evidence_quote": "", "reviewer": "", "note": ""})
    df = pd.DataFrame(rows, columns=QUEUE_COLUMNS).sort_values(["symbol", "due"]) if rows else pd.DataFrame(columns=QUEUE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    return df


def _doc_text(G, doc_id: str) -> str | None:
    r = G.db.execute("SELECT path FROM doc_files WHERE doc_id=?", (doc_id,)).fetchone()
    if not r or not Path(r[0]).exists():
        return None
    return Path(r[0]).read_text(errors="ignore")


def validate(G, row: dict) -> list[str]:
    errs = []
    v = (row.get("verdict") or "").strip().upper()
    if v not in VERDICTS:
        errs.append(f"verdict must be one of {VERDICTS}")
    if not (row.get("reviewer") or "").strip():
        errs.append("reviewer is required")
    c = G.db.execute("SELECT made_at FROM claims WHERE claim_id=? AND kind='promise'", (row.get("claim_id"),)).fetchone()
    if not c:
        return errs + ["unknown promise claim_id"]
    if v in ("MET", "PARTIAL", "MISSED"):
        d = G.db.execute("SELECT published_at FROM documents WHERE doc_id=?", (row.get("evidence_doc_id"),)).fetchone()
        if not d:
            errs.append("evidence_doc_id is not a document in Company Memory")
        else:
            if pd.Timestamp(d[0]) <= pd.Timestamp(c[0]):
                errs.append("evidence document is not later than the promise")
            text = _doc_text(G, row["evidence_doc_id"])
            q = _norm(row.get("evidence_quote"))
            if not q or len(q) < 15:
                errs.append("evidence_quote (verbatim, >= 15 chars) is required")
            elif text is None:
                errs.append("evidence document text is not available locally to verify the quote")
            elif q not in _norm(text):
                errs.append("evidence_quote does not appear verbatim in the evidence document")
    return errs


def apply_reviews(G, path: Path) -> dict:
    ensure(G)
    df = pd.read_csv(path, dtype=str).fillna("")
    applied, rejected = 0, []
    for row in df.to_dict("records"):
        if not row["verdict"].strip():
            continue
        errs = validate(G, row)
        if errs:
            rejected.append({"claim_id": row["claim_id"], "errors": errs})
            continue
        v = row["verdict"].strip().upper()
        at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rid = "R" + hashlib.sha256(json.dumps([row["claim_id"], v, row["evidence_doc_id"], row["evidence_quote"], row["reviewer"], at]).encode()).hexdigest()[:16]
        G.db.execute("INSERT INTO promise_reviews VALUES (?,?,?,?,?,?,?,?,?)", (rid, row["claim_id"], v, row["evidence_doc_id"],
                     " ".join(row["evidence_quote"].split()), row["reviewer"].strip(), at, row["note"], row.get("suggested", "")))
        G.set_status(row["claim_id"], f"VERIFIED {v} (review {rid})")
        if row["evidence_doc_id"]:
            G.link(row["evidence_doc_id"], "verified_outcome", row["claim_id"], " ".join(row["evidence_quote"].split())[:400])
        applied += 1
    G.commit()
    return {"applied": applied, "rejected": rejected}


def execution_record(G, entity: str | None = None) -> list[dict]:
    """Counts of human-verified verdicts per company (latest review per promise)."""
    ensure(G)
    q = ("SELECT c.entity, r.verdict, r.reviewed_at, r.claim_id FROM promise_reviews r JOIN claims c ON c.claim_id=r.claim_id"
         + (" WHERE c.entity=?" if entity else ""))
    df = pd.DataFrame(G.db.execute(q, (entity,) if entity else ()).fetchall(), columns=["entity", "verdict", "at", "claim_id"])
    out = []
    if len(df):
        df = df.sort_values("at").groupby("claim_id").tail(1)
    for e, g in (df.groupby("entity") if len(df) else []):
        n = int(g.verdict.isin(["MET", "PARTIAL", "MISSED"]).sum())
        out.append({"entity": e, "verified": n, **{v: int((g.verdict == v).sum()) for v in VERDICTS},
                    "status": "insufficient" if n < MIN_OBS else "reportable (descriptive only; not a trading factor)"})
    return out
