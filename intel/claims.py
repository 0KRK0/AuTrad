"""Management-claim tracker: what leaders said would happen, and what did happen.

A claim is admitted only if its quote is found verbatim in a ledger document (EvidenceLedger.resolve). Measurable
claims (revenue / profit growth, margin) are resolved automatically from later PIT results filings; others need a
later document as outcome evidence. Over time each company (and speaker) gets a reliability record:
Beta(1 + met, 1 + missed) posterior mean and the count behind it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from forward.ledger import Ledger

from .evidence import EvidenceLedger, Statement, content_hash

METRICS = {"revenue_growth_yoy", "net_profit_growth_yoy", "net_margin", "other"}


@dataclass
class ManagementClaim:
    symbol: str
    speaker: str                 # e.g. "CEO", or a name as it appears in the document
    made_at: str                 # publication time of the source document
    statement: str               # our paraphrase
    quote: str                   # verbatim text from the document
    evidence_id: str
    metric: str = "other"        # one of METRICS
    op: str = ">="               # comparison for numeric metrics
    target: float | None = None
    due: str = ""                # ISO date by which it should be observable (e.g. quarter end)
    status: str = "PENDING"      # PENDING | MET | MISSED | UNRESOLVABLE
    outcome: dict = field(default_factory=dict)
    resolved_at: str = ""         # publication time of the evidence that resolved it
    claim_id: str = ""

    def __post_init__(self) -> None:
        if not self.claim_id:
            self.claim_id = "C" + content_hash(f"{self.symbol}|{self.speaker}|{self.made_at}|{self.quote}")[:12]


class ClaimTracker:
    def __init__(self, evidence: EvidenceLedger, path: str | Path | None = None) -> None:
        self.ev = evidence
        self.claims: dict[str, ManagementClaim] = {}
        self.log = Ledger(path) if path else None
        if self.log:
            for r in self.log.records():
                if r["kind"] in ("claim", "claim_resolution"):
                    c = ManagementClaim(**{k: r[k] for k in ManagementClaim.__dataclass_fields__})
                    self.claims[c.claim_id] = c

    def add(self, c: ManagementClaim) -> ManagementClaim:
        st = self.ev.resolve(Statement(c.statement, [c.evidence_id], {c.evidence_id: c.quote}), c.made_at)
        if st.status not in ("SOURCED", "CORROBORATED"):
            raise ValueError(f"claim rejected: {st.note or st.status}")
        if c.metric not in METRICS:
            raise ValueError(f"unknown metric {c.metric}")
        self.claims[c.claim_id] = c
        if self.log:
            self.log.append("claim", asdict(c))
        return c

    def resolve_from_filings(self, facts: pd.DataFrame, asof: pd.Timestamp) -> list[ManagementClaim]:
        """Numeric claims whose due date has passed and whose quarter has been filed by `asof`."""
        done = []
        for c in self.claims.values():
            if c.status != "PENDING" or c.metric == "other" or not c.due or pd.Timestamp(c.due) > asof:
                continue
            a_utc = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
            f = facts[(facts["symbol"] == c.symbol) & (pd.to_datetime(facts["available_at"], utc=True) <= a_utc)]
            f = f.sort_values("period_end").drop_duplicates("period_end", keep="last")
            cur = f[pd.to_datetime(f["period_end"]) <= pd.Timestamp(c.due)].tail(1)
            if cur.empty:
                continue
            pe = pd.Timestamp(cur["period_end"].iloc[0])
            prev = f[pd.to_datetime(f["period_end"]) == pe - pd.DateOffset(years=1)]
            val = None
            if c.metric in ("revenue_growth_yoy", "net_profit_growth_yoy") and len(prev):
                col = "revenue" if c.metric == "revenue_growth_yoy" else "net_profit"
                a, b = float(cur[col].iloc[0]), float(prev[col].iloc[0])
                val = a / b - 1 if b > 0 else None
            elif c.metric == "net_margin":
                rv = float(cur["revenue"].iloc[0])
                val = float(cur["net_profit"].iloc[0]) / rv if rv > 0 else None
            if val is None:
                c.status = "UNRESOLVABLE"
            else:
                ok = val >= c.target if c.op == ">=" else val <= c.target
                c.status = "MET" if ok else "MISSED"
            c.outcome = {"value": val, "period_end": str(pe.date()), "filing": str(cur["xbrl"].iloc[0])}
            c.resolved_at = pd.Timestamp(cur["available_at"].iloc[0]).isoformat()
            if self.log:
                self.log.append("claim_resolution", asdict(c))
            done.append(c)
        return done

    def reliability(self, symbol: str, speaker: str | None = None, asof: str | None = None) -> dict:
        """Record as it was knowable at `asof`: claims made by then; outcomes count only if resolved by then."""
        from .evidence import utc

        cs = [c for c in self.claims.values() if c.symbol == symbol and (speaker is None or c.speaker == speaker)
              and (asof is None or utc(c.made_at) <= utc(asof))]

        def st(c):
            return c.status if c.status in ("MET", "MISSED") and (asof is None or (c.resolved_at and utc(c.resolved_at) <= utc(asof))) else "PENDING"
        met = sum(st(c) == "MET" for c in cs)
        missed = sum(st(c) == "MISSED" for c in cs)
        return {"claims": len(cs), "met": met, "missed": missed, "pending": len(cs) - met - missed,
                "reliability": (1 + met) / (2 + met + missed), "evidence_weight": met + missed, "evidence": [c.evidence_id for c in cs]}
