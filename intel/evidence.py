"""Evidence Ledger: every factual statement an agent makes must resolve to a source document.

Document = something that exists outside AuTrad (an exchange filing, an announcement, a news item, a dataset row),
stored with its source tier, URL / id, publication time, ingestion time, entities and a content hash.
Statement = what an agent says. It carries the ids of the documents it rests on and, for every cited document, the
exact quote(s) it relies on. The resolver labels it (all times compared as UTC instants):
  SOURCED       kind "fact": every cited document exists, was published at or before the as-of instant, and every
                quote is found verbatim in its document
  CORROBORATED  SOURCED by at least two independent publishers, at least one of tier <= 3
  DERIVED       kind "derived": a computation whose inputs are all quoted verbatim from cited documents
  INFERENCE     kind "inference": an interpretation that cites resolvable evidence (never a fact)
  LABEL         kind "label": AuTrad's own classification (e.g. sector bucket), no evidence claimed
  UNVERIFIED    anything else (no evidence, unknown id, missing or non-verbatim quote, a document from the future)
Agents never author evidence: the only documents AuTrad itself writes are tier-9 DATASET_EXTRACT records, which hold raw
numbers copied from its exchange-anchored datasets (no interpretation) and never count as tier 1-2 evidence.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path

import pandas as pd

from forward.ledger import Ledger


class Tier(IntEnum):
    REGULATORY_FILING = 1        # exchange / regulator filing or announcement (NSE, BSE, SEBI, SEC)
    AUDITED_FINANCIALS = 2
    COMPANY_STATEMENT = 3        # press release, investor presentation, official IR site
    EXECUTIVE_PUBLIC = 4         # CEO/CFO interview or post on a verified official account
    JOURNALISM = 5               # reputable news organisation
    ANALYST = 6
    SOCIAL_MEDIA = 7
    ANONYMOUS = 8
    DATASET_EXTRACT = 9          # raw numbers copied from AuTrad's exchange-anchored datasets (reproducible, not evidence of tier 1-2)


TIER_NAMES = {1: "regulatory filing / exchange announcement", 2: "audited financials / audited dataset", 3: "company statement",
              4: "executive public statement", 5: "journalism", 6: "analyst research / third-party reconstruction", 7: "social media",
              8: "anonymous / unverified", 9: "AuTrad dataset extract"}


def utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def content_hash(text: str | bytes) -> str:
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode()).hexdigest()


@dataclass
class Document:
    source: str                  # publisher / system, e.g. "NSE corporate announcements"
    tier: int
    url: str                     # URL or a stable identifier (dataset path + key)
    published_at: str            # ISO time the information became public
    entities: list[str]
    title: str
    text: str                    # the content we rely on (announcement text, headline, data row as text)
    method: str                  # how it was obtained (API endpoint, RSS, dataset loader)
    ingested_at: str = field(default_factory=_now)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = "D" + content_hash(f"{self.source}|{self.url}|{self.published_at}|{self.text}")[:16]


@dataclass
class Statement:
    text: str
    evidence: list[str] = field(default_factory=list)          # document ids
    quotes: dict[str, str | list[str]] = field(default_factory=dict)   # doc id -> verbatim quote(s) relied on
    kind: str = "fact"                                          # fact | derived | inference | label
    status: str = "UNVERIFIED"
    note: str = ""


class EvidenceLedger:
    """Append-only, hash-chained store of documents (and the statements resolved against them)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.docs: dict[str, Document] = {}
        self.ledger = Ledger(path) if path else None
        if self.ledger:
            for r in self.ledger.of_kind("document"):
                d = Document(**{k: r[k] for k in Document.__dataclass_fields__})
                self.docs[d.doc_id] = d

    def add(self, doc: Document) -> str:
        if doc.doc_id not in self.docs:
            self.docs[doc.doc_id] = doc
            if self.ledger:
                self.ledger.append("document", asdict(doc))
        return doc.doc_id

    def get(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)

    def as_of(self, entity: str, t: str) -> list[Document]:
        tt = utc(t)
        return sorted((d for d in self.docs.values() if entity in d.entities and d.published_at and utc(d.published_at) <= tt),
                      key=lambda d: utc(d.published_at))

    def external_view(self) -> "EvidenceLedger":
        """A separate in-memory ledger with external documents only (tiers 1-8): what an isolated reviewer may read."""
        v = EvidenceLedger()
        v.docs = {k: d for k, d in self.docs.items() if d.tier <= 8}
        return v

    def resolve(self, st: Statement, asof: str) -> Statement:
        if st.kind == "label":
            st.status = "LABEL"
            return st
        if not st.evidence:
            st.status, st.note = "UNVERIFIED", "no evidence cited"
            return st
        for e in st.evidence:
            if not self._ok(e, asof):
                st.status, st.note = "UNVERIFIED", f"evidence {e} missing or published after {asof}"
                return st
        if st.kind == "inference":
            st.status = "INFERENCE"
            return st
        for e in st.evidence:
            qs = st.quotes.get(e)
            if not qs:
                st.status, st.note = "UNVERIFIED", f"no verbatim quote from {e}"
                return st
            text = " ".join(self.docs[e].text.split()).lower()
            for q in [qs] if isinstance(qs, str) else qs:
                if " ".join(q.split()).lower() not in text:
                    st.status, st.note = "UNVERIFIED", f"quote not found verbatim in {e}"
                    return st
        if st.kind == "derived":
            st.status = "DERIVED"
            return st
        ext = [self.docs[e] for e in st.evidence if self.docs[e].tier <= 8]
        pubs = {d.source for d in ext}
        st.status = "CORROBORATED" if len(pubs) >= 2 and min(d.tier for d in ext) <= 3 else ("SOURCED" if ext else "DERIVED")
        return st

    def min_external_tier(self, st: Statement) -> int | None:
        t = [self.docs[e].tier for e in st.evidence if e in self.docs and self.docs[e].tier <= 8]
        return min(t) if t else None

    def _ok(self, doc_id: str, asof: str) -> bool:
        d = self.docs.get(doc_id)
        return d is not None and bool(d.published_at) and utc(d.published_at) <= utc(asof)


def render(st: Statement, ledger: EvidenceLedger) -> str:
    refs = ", ".join(f"[{e}: tier {ledger.docs[e].tier}, {ledger.docs[e].published_at[:10]}]" for e in st.evidence if e in ledger.docs)
    return f"{st.text} — {st.status}{(' ' + refs) if refs else ''}{(' (' + st.note + ')') if st.note else ''}"


def to_json(st: Statement) -> dict:
    return json.loads(json.dumps(asdict(st)))
