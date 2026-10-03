"""Evidence graph confidence rules and the management-promise extractor."""
from __future__ import annotations

import pandas as pd

from intel.evidence import Document
from intel.graph import EvidenceGraph
from intel.transcripts import due_date, promise_candidates

TRANSCRIPT = """
Moderator:      Ladies and gentlemen, welcome to the call. I now hand over to the management.
Ravi Kumar:     Thank you. We had a good quarter with volumes up.
Asha Rao:       Thank you. What is your capacity outlook for the next two years?
Ravi Kumar:     We will reach 50 million tonnes of capacity by FY27. We are also doing well in the south.
Asha Rao:       And margins? Will they improve?
Ravi Kumar:     Margins depend on fuel prices. We expect the plant at Pune to be commissioned by the end of fiscal '26.
Moderator:      Thank you. Next question.
Vik Shah:       Can you share the capex number for this year?
Ravi Kumar:     We do not guide on capex. We will keep investing.
"""


def test_promises_come_only_from_management_and_need_a_target_and_a_date():
    c = promise_candidates(TRANSCRIPT)
    quotes = [x["quote"] for x in c]
    assert "We will reach 50 million tonnes of capacity by FY27." in quotes
    assert any("commissioned by the end of fiscal '26" in q for q in quotes)
    assert not any("What is your capacity" in q or "We will keep investing" in q for q in quotes)
    assert {x["speaker"] for x in c} == {"Ravi Kumar"}
    assert due_date("by FY27", pd.Timestamp("2025-01-10")) == "2027-03-31"
    assert due_date("within this financial year", pd.Timestamp("2023-07-26")) == "2024-03-31"


def _doc(src, tier, text, when="2025-01-01T10:00:00+05:30"):
    return Document(src, tier, f"u:{src}:{text}", when, ["X"], "t", text, "test")


def test_confidence_never_rises_by_repetition_and_contradictions_are_tier_aware(tmp_path):
    G = EvidenceGraph(tmp_path / "g.sqlite")
    social = [G.add_document(_doc(f"post {k}", 8, "X will triple sales")) for k in range(10)]
    c1 = G.add_claim("fact", "X", "X will triple sales", "2025-01-02", social)
    assert G.confidence(c1) == "LOW"                                 # ten tier-8 posts are still tier 8
    f = G.add_document(_doc("NSE filing", 1, "revenue Rs 100 cr"))
    c2 = G.add_claim("derived", "X", "Revenue grew 10%", "2025-01-02", [f])
    assert G.confidence(c2) == "HIGH"
    j = G.add_document(_doc("news: Reuters", 5, "revenue fell"))
    c3 = G.add_claim("fact", "X", "Revenue fell", "2025-01-03", [j])
    G.link(c3, "contradicts", c2)
    assert G.confidence(c2) == "HIGH"                                # a tier-5 report cannot contest a tier-1 filing
    G.link(c2, "contradicts", c3)
    assert G.confidence(c3) == "CONTESTED"
    x = G.add_document(_doc("AuTrad dataset extract", 9, "close 100"))
    assert G.confidence(G.add_claim("derived", "X", "fell 9%", "2025-01-03", [x])) == "DERIVED-ONLY"
    assert G.confidence(G.add_claim("thesis", "X", "no adverse news", "2025-01-03", [])) == "UNVERIFIED"
