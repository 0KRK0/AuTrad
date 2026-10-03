"""Evidence & claim graph: AuTrad's persistent company memory (SQLite, data/intel/graph.sqlite).

Nodes: documents (source, tier 1-9, URL, publication / ingestion time, hash), claims (fact | derived | inference |
promise | thesis | finding; entity, speaker, verbatim quote, time, due date, status), metrics (filed numbers),
events (results reactions, meetings), decisions (Main Head memos).
Edges (links): claim -sourced_by-> document, claim -corroborates/contradicts-> claim, outcome -outcome_of-> promise,
decision -thesis-> claim, finding -challenges-> decision, claim -cites-> document.

Confidence measures provenance strength (does the claim rest on authoritative documents), not whether the numbers
support it: support is checked when a statement is created (DERIVED statements are computed from the quoted numbers,
and the Head cites filings for "fundamentals are stable" only when they show no decline). Confidence never rises by repetition: it is set by the best tier among a claim's own sources and by contradictions
(tier <= its best tier); AuTrad-derived sources (tier 9) alone give DERIVED-ONLY. trace() answers "why?" for a
decision: decision -> thesis and challenges -> statements -> documents.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, source TEXT, tier INTEGER, url TEXT, published_at TEXT,
  ingested_at TEXT, entities TEXT, title TEXT, text_sha256 TEXT, method TEXT);
CREATE TABLE IF NOT EXISTS claims (claim_id TEXT PRIMARY KEY, kind TEXT, entity TEXT, speaker TEXT, text TEXT, quote TEXT,
  made_at TEXT, due TEXT, metric TEXT, target TEXT, status TEXT, origin TEXT);
CREATE TABLE IF NOT EXISTS links (src TEXT, rel TEXT, dst TEXT, note TEXT, PRIMARY KEY (src, rel, dst));
CREATE TABLE IF NOT EXISTS metrics (entity TEXT, metric TEXT, period_end TEXT, value REAL, doc_id TEXT, PRIMARY KEY (entity, metric, period_end, doc_id));
CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, entity TEXT, type TEXT, date TEXT, value REAL, doc_id TEXT);
CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT, symbol TEXT, strategy TEXT, asof TEXT, action TEXT, evidence_status TEXT,
  historical_p REAL, memo_hash TEXT, PRIMARY KEY (decision_id, symbol));
CREATE INDEX IF NOT EXISTS ix_claims_entity ON claims(entity);
CREATE INDEX IF NOT EXISTS ix_links_dst ON links(dst);
"""
CONF_ORDER = ["HIGH", "MEDIUM", "LOW", "DERIVED-ONLY", "CONTESTED", "UNVERIFIED"]


def _id(*parts) -> str:
    return "C" + hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


class EvidenceGraph:
    def __init__(self, path: str | Path = "data/intel/graph.sqlite") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ writes (idempotent)
    def add_document(self, d) -> str:
        self.db.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (d.doc_id, d.source, int(d.tier), d.url, d.published_at, d.ingested_at, json.dumps(d.entities), d.title,
                         hashlib.sha256(d.text.encode()).hexdigest(), d.method))
        return d.doc_id

    def add_claim(self, kind: str, entity: str, text: str, made_at: str, sources: list[str] = (), quote: str = "", speaker: str = "",
                  due: str = "", metric: str = "", target: str = "", status: str = "", origin: str = "") -> str:
        cid = _id(kind, entity, text, made_at, quote, speaker)
        self.db.execute("INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(claim_id) DO UPDATE SET due=excluded.due, "
                        "metric=excluded.metric, target=excluded.target, status=excluded.status, origin=excluded.origin",
                        (cid, kind, entity, speaker, text, quote, made_at, due, metric, target, status, origin))
        for s in sources:
            self.link(cid, "sourced_by", s)
        return cid

    def link(self, src: str, rel: str, dst: str, note: str = "") -> None:
        self.db.execute("INSERT OR IGNORE INTO links VALUES (?,?,?,?)", (src, rel, dst, note))

    def set_status(self, claim_id: str, status: str) -> None:
        self.db.execute("UPDATE claims SET status=? WHERE claim_id=?", (status, claim_id))

    def commit(self) -> None:
        self.db.commit()

    # ------------------------------------------------------------------ reads
    def sources(self, claim_id: str) -> list[dict]:
        q = ("SELECT d.doc_id, d.source, d.tier, d.url, d.published_at FROM links l JOIN documents d ON d.doc_id = l.dst "
             "WHERE l.src=? AND l.rel IN ('sourced_by','cites')")
        return [dict(zip(["doc_id", "source", "tier", "url", "published_at"], r)) for r in self.db.execute(q, (claim_id,))]

    def confidence(self, claim_id: str) -> str:
        src = self.sources(claim_id)
        ext = [s for s in src if s["tier"] <= 8]
        if not src:
            return "UNVERIFIED"
        if not ext:
            return "DERIVED-ONLY"
        best = min(s["tier"] for s in ext)
        contra = self.db.execute("SELECT c.claim_id FROM links l JOIN claims c ON c.claim_id=l.src WHERE l.dst=? AND l.rel='contradicts'", (claim_id,)).fetchall()
        for (cid,) in contra:
            cs = [s["tier"] for s in self.sources(cid) if s["tier"] <= 8]
            if cs and min(cs) <= best:
                return "CONTESTED"
        return "HIGH" if best <= 2 else ("MEDIUM" if best <= 5 else "LOW")

    def trace(self, decision_id: str, symbol: str) -> dict:
        """Why? decision -> thesis claims and 10th-Man findings -> their documents."""
        row = self.db.execute("SELECT strategy, asof, action, evidence_status, historical_p FROM decisions WHERE decision_id=? AND symbol=?",
                              (decision_id, symbol)).fetchone()
        if not row:
            return {}
        out = {"decision": decision_id, "symbol": symbol, "strategy": row[0], "asof": row[1], "action": row[2],
               "evidence_status": row[3], "historical_probability": row[4], "because": [], "challenged_by": []}
        node = f"{decision_id}:{symbol}"
        for rel, key in (("thesis", "because"), ("challenged_by", "challenged_by")):
            for (cid,) in self.db.execute("SELECT dst FROM links WHERE src=? AND rel=?", (node, rel)):
                c = self.db.execute("SELECT kind, text, status FROM claims WHERE claim_id=?", (cid,)).fetchone()
                out[key].append({"claim": c[1], "kind": c[0], "status": c[2], "confidence": self.confidence(cid), "documents": self.sources(cid)})
        return out

    def promises(self, entity: str | None = None) -> list[dict]:
        q = "SELECT claim_id, entity, speaker, quote, made_at, due, metric, target, status FROM claims WHERE kind='promise'"
        rows = self.db.execute(q + (" AND entity=?" if entity else ""), (entity,) if entity else ()).fetchall()
        return [dict(zip(["claim_id", "entity", "speaker", "quote", "made_at", "due", "metric", "target", "status"], r)) for r in rows]

    def stats(self) -> dict:
        n = {t: self.db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ["documents", "claims", "links", "metrics", "events", "decisions"]}
        n["documents_by_tier"] = dict(self.db.execute("SELECT tier, COUNT(*) FROM documents GROUP BY tier ORDER BY tier").fetchall())
        n["claims_by_kind"] = dict(self.db.execute("SELECT kind, COUNT(*) FROM claims GROUP BY kind").fetchall())
        return n
