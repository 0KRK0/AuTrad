"""Advisory agents: evidence provenance, the 10th Man's isolation and honesty rules, the Head's authority."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.company_intel import CompanyIntelligenceAgent
from agents.context import MarketContext
from agents.tenth_man import FEATURES, BaseRateModel, ChallengeRequest, TenthMan, ThesisClaim
from head.main_head import MainHead, StrategyProposal
from intel.claims import ClaimTracker, ManagementClaim
from intel.evidence import Document, EvidenceLedger, Statement
from tests.test_factors import _panel

ROOT = Path(__file__).resolve().parents[1]


def _ctx(bad_symbol: str | None = None):
    P = _panel(T=500, N=24)
    P.sector = ["it"] * 8 + ["banks"] * 8 + ["fmcg"] * 8
    rows = []
    for s in P.symbols[:6]:
        for k, pe in enumerate(pd.date_range("2017-03-31", periods=8, freq="QE")):
            rows.append({"symbol": s, "period_end": pe, "available_at": (pe + pd.Timedelta(days=30)).tz_localize("Asia/Kolkata"),
                         "xbrl": f"https://example.org/{s}/{k}.xml", "sha256": "ab" * 32, "consolidated": True, "audited": "Un-Audited",
                         "revenue": 1e10 * (1 + 0.02 * k), "net_profit": 1e9 * (1 + 0.02 * k) if s != "S1" else 1e9 * (1 - 0.15 * k), "eps_basic": 10.0})
    facts = pd.DataFrame(rows)
    t = len(P.dates) - 1
    ev = pd.DataFrame([{"symbol": bad_symbol or "S0", "period_end": pd.Timestamp("2018-09-30"), "reaction_idx": t - 5, "ear": -0.06,
                        "bad": 1.0 if bad_symbol else 0.0, "reaction_day": P.dates[t - 5]}])
    return MarketContext(P, facts, ev, {}, "testhash", {s: f"Company {s}" for s in P.symbols})


# ----------------------------------------------------------------------------- evidence rules
def test_resolver_never_turns_missing_fabricated_or_future_evidence_into_fact():
    L = EvidenceLedger()
    d = L.add(Document("NSE corporate announcements", 1, "u1", "2026-09-01T10:00:00+05:30", ["X"], "t", "Board approved a buyback of Rs 500 cr", "api"))
    d2 = L.add(Document("news: Reuters", 5, "u2", "2026-09-02T10:00:00+05:30", ["X"], "t", "X announces buyback", "rss"))
    asof = "2026-09-05T15:30:00+05:30"
    assert L.resolve(Statement("buyback", []), asof).status == "UNVERIFIED"
    assert L.resolve(Statement("buyback", ["Dnope"]), asof).status == "UNVERIFIED"
    assert L.resolve(Statement("buyback", [d], {d: "buyback of Rs 900 cr"}), asof).status == "UNVERIFIED"   # quote not in the document
    assert L.resolve(Statement("buyback", [d]), "2026-08-31T15:30:00+05:30").status == "UNVERIFIED"      # published later
    assert L.resolve(Statement("buyback", [d]), asof).status == "UNVERIFIED"                            # a fact must quote its source
    assert L.resolve(Statement("buyback", [d], {d: "buyback of Rs 500 cr"}), asof).status == "SOURCED"
    assert L.resolve(Statement("buyback", [d, d2], {d: "buyback", d2: "announces buyback"}), asof).status == "CORROBORATED"
    late = L.add(Document("news: Reuters", 5, "u3", "2026-09-05T10:30:00+00:00", ["X"], "t", "X cancels buyback", "rss"))   # 16:00 IST
    assert L.resolve(Statement("cancel", [late], {late: "cancels buyback"}), "2026-09-05T15:30:00+05:30").status == "UNVERIFIED"
    assert L.external_view().docs.keys() == L.docs.keys()


def test_claim_tracker_rejects_invented_quotes_and_scores_management():
    L = EvidenceLedger()
    facts = _ctx().facts
    d = L.add(Document("NSE corporate announcements", 1, "u", "2017-05-01T10:00:00+05:30", ["S0"], "call",
                       "CEO: we expect revenue to grow at least 5 percent this year", "api"))
    tr = ClaimTracker(L)
    with pytest.raises(ValueError):
        tr.add(ManagementClaim("S0", "CEO", "2017-05-01T10:00:00+05:30", "growth >= 5%", "revenue to grow 20 percent", d,
                               "revenue_growth_yoy", ">=", 0.05, "2018-06-30"))
    tr.add(ManagementClaim("S0", "CEO", "2017-05-01T10:00:00+05:30", "growth >= 5%", "revenue to grow at least 5 percent", d,
                           "revenue_growth_yoy", ">=", 0.05, "2018-06-30"))
    tr.add(ManagementClaim("S0", "CFO", "2017-05-01T10:00:00+05:30", "growth >= 10%", "at least 5 percent", d, "revenue_growth_yoy", ">=", 0.10, "2018-06-30"))
    res = tr.resolve_from_filings(facts, pd.Timestamp("2019-01-01"))
    assert {c.speaker: c.status for c in res} == {"CEO": "MET", "CFO": "MISSED"}
    assert tr.reliability("S0")["reliability"] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- 10th Man isolation
def _imports(path: Path, pkg: str) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.add(f"{pkg}.{n.module}" if n.level == 1 else n.module)
        elif isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
    return out


def test_tenth_man_reaches_no_agent_head_gate_or_broker_even_transitively():
    forbidden = {"agents.company_intel", "agents.system", "agents.override", "head", "head.main_head", "risk.gate", "execution.broker"}
    seen, todo = set(), ["agents.tenth_man"]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        p = ROOT / (m.replace(".", "/") + ".py")
        if not p.exists():
            p = ROOT / m.replace(".", "/") / "__init__.py"
        if not p.exists():
            continue                                            # third-party module
        todo += [x for x in _imports(p, m.rsplit(".", 1)[0]) if x.split(".")[0] in {"agents", "head", "intel", "forward", "lab", "marketdata", "risk", "execution", "argo"}]
    assert not (seen & forbidden), seen & forbidden
    params = list(inspect.signature(TenthMan.__init__).parameters)
    assert params == ["self", "ctx", "ledger", "model", "skill"]


def test_tenth_man_accepts_only_a_challenge_request_with_no_reasoning_fields():
    fields = set(ChallengeRequest.__dataclass_fields__)
    assert fields == {"decision_id", "action", "symbol", "weight", "horizon_sessions", "asof", "strategy", "thesis", "sector_weight_after"}
    ctx = _ctx()
    tm = TenthMan(ctx, EvidenceLedger(), BaseRateModel(), None)
    with pytest.raises(TypeError):
        tm.review({"symbol": "S0", "reasoning": "everyone agrees"})
    with pytest.raises(Exception):
        ChallengeRequest("d", "BUY", "S0", 0.1, 21, "2018-11-01", "REV1").symbol = "S1"      # frozen


def test_no_material_objection_is_allowed_and_bad_news_is_challenged():
    ctx = _ctx()
    asof = str(ctx.P.dates[-1].date())
    th = (ThesisClaim("No adverse company news drove the loss.", (), "no_adverse_news"),)
    tm = TenthMan(ctx, EvidenceLedger(), BaseRateModel(), None)
    rep = tm.review(ChallengeRequest("d1", "BUY", "S5", 0.06, 21, asof, "REV1", th))
    assert rep.status == "NO_MATERIAL_OBJECTION" and rep.recommendation == "PROCEED"
    ctx2 = _ctx(bad_symbol="S3")
    rep2 = TenthMan(ctx2, EvidenceLedger(), BaseRateModel(), None).review(ChallengeRequest("d2", "BUY", "S3", 0.06, 21, asof, "REV1", th))
    assert rep2.status == "MATERIAL_CHALLENGE" and rep2.severity == "high"
    assert any(f.attacks == "no_adverse_news" for f in rep2.findings)


def test_probability_comes_from_a_model_only_if_it_beats_the_base_rate():
    rng = np.random.default_rng(0)
    n = 4000
    dec = np.repeat(pd.date_range("2015-01-31", periods=100, freq="ME"), n // 100)
    X = rng.integers(0, 2, (n, len(FEATURES))).astype(float)
    p = 1 / (1 + np.exp(-(-0.2 + 1.2 * X[:, 0])))
    H = pd.DataFrame(X, columns=FEATURES).assign(decision=dec, known_at=dec + pd.offsets.MonthEnd(1), fail=(rng.random(n) < p).astype(float))
    assert BaseRateModel.walk_forward(H)["skill"] > 0.02
    H2 = H.assign(fail=(rng.random(n) < 0.5).astype(float))
    assert BaseRateModel.walk_forward(H2)["skill"] <= 0.005
    m = BaseRateModel().fit(H, pd.Timestamp("2016-01-01"))
    assert m.n == int((H["known_at"] < pd.Timestamp("2016-01-01")).sum())        # nothing known later is used


# ----------------------------------------------------------------------------- Company Intelligence + Head
def test_computed_findings_never_count_as_a_tier_one_contradiction():
    ctx = _ctx()
    P = ctx.P
    t = len(P.dates) - 1
    C = ctx.close()
    it = [P.idx[s] for s in P.symbols if P.sector[P.idx[s]] == "it"]
    C[t, it] = C[t - 21, it] * 0.8                                  # a sector-wide 20 % fall, incl. S5
    th = (ThesisClaim("stock-specific overreaction", (), "idiosyncratic_overreaction"),)
    rep = TenthMan(ctx, EvidenceLedger(), BaseRateModel(), None).review(ChallengeRequest("d", "BUY", "S5", 0.06, 21, str(P.dates[-1].date()), "REV1", th))
    assert any(f.check == "alternative explanation" for f in rep.findings)
    assert rep.status == "NO_MATERIAL_OBJECTION"


def test_dossier_facts_all_resolve_to_documents():
    ctx = _ctx()
    L = EvidenceLedger()
    d = CompanyIntelligenceAgent(ctx, L).dossier("S1", ctx.P.dates[-1])
    assert d.provenance_coverage() == 1.0
    assert any("Net profit" in s.text for s in d.sections["Financial trajectory (NSE filings)"])
    assert any("Profit falling" in s.text for s in d.sections["Red flags"])
    assert all(e in L.docs for s in d.statements() for e in s.evidence)


def test_head_in_shadow_executes_the_strategy_unchanged_and_agents_hold_no_order_path():
    ctx = _ctx(bad_symbol="S3")
    L = EvidenceLedger()
    cia = CompanyIntelligenceAgent(ctx, L)
    tm = TenthMan(ctx, L, BaseRateModel(), None)
    head = MainHead(cia, tm, "SHADOW")
    targets = {"S3": 0.5, "S5": 0.5}
    memo = head.deliberate(StrategyProposal("REV1 test", ctx.P.dates[-1], targets, {}, 21, {s: "it" for s in targets}))
    assert memo.final_targets == targets
    assert "S3" not in memo.shadow_targets                    # a challenge-aware head would have dropped it
    for agent in (cia, tm):
        assert not any(k in vars(agent) for k in ("gate", "broker", "head", "tenth", "cia"))
