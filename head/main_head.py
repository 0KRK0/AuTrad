"""Main Head: collects a strategy's proposal, asks Company Intelligence for a dossier on each name, builds a
sanitised ChallengeRequest for the 10th Man, and writes a Decision Memo.

Routing (enforced): strategy -> Head -> {Company Intelligence, 10th Man} -> Head -> Risk Gate. The advisory agents
never talk to each other, never see each other's output, and cannot reach the gate or a broker.
Policy SHADOW (v0.6): the memo records what a challenge-aware Head *would* do (drop on ABSTAIN, halve on
REDUCE_CONFIDENCE) but the final targets are the strategy's own, unchanged.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from agents.company_intel import CompanyIntelligenceAgent, Dossier
from agents.tenth_man import ChallengeReport, ChallengeRequest, TenthMan, ThesisClaim


@dataclass
class StrategyProposal:
    strategy: str
    asof: pd.Timestamp
    targets: dict[str, float]
    previous: dict[str, float] = field(default_factory=dict)
    horizon_sessions: int = 21
    sector: dict[str, str] = field(default_factory=dict)


@dataclass
class MemoItem:
    symbol: str
    action: str
    weight: float
    dossier_coverage: float
    challenge: dict
    would_do: str
    would_weight: float


@dataclass
class DecisionMemo:
    decision_id: str
    strategy: str
    asof: str
    policy: str
    items: list[MemoItem]
    final_targets: dict[str, float]
    shadow_targets: dict[str, float]
    thesis_template_sha: str = ""


class MainHead:
    def __init__(self, cia: CompanyIntelligenceAgent, tenth: TenthMan, policy: str = "SHADOW", audit=None) -> None:
        self._cia, self._tenth, self.policy, self.audit = cia, tenth, policy, audit
        self.dossiers: dict[str, Dossier] = {}

    def _log(self, kind: str, payload: dict) -> None:
        if self.audit is not None:
            self.audit.write(kind, payload)

    def _thesis(self, prop: StrategyProposal, sym: str, d: Dossier) -> tuple[ThesisClaim, ...]:
        ident = d.sections.get("Identity and price", [])
        fin = [s for s in d.sections.get("Financial trajectory (NSE filings)", []) if "year on year" in s.text]
        if prop.strategy.startswith("REV1"):
            return (ThesisClaim("Last month's loss is a stock-specific, liquidity-driven overreaction that tends to revert.",
                                tuple(ident[-1].evidence) if ident else (), "idiosyncratic_overreaction"),
                    ThesisClaim("No adverse company news drove the loss.", (), "no_adverse_news"),
                    ThesisClaim("Fundamentals are stable.", tuple(e for s in fin for e in s.evidence)
                                if d.metrics and d.metrics.get("net_profit", 0) >= 0 and d.metrics.get("revenue", 0) >= 0 else (), "fundamentals_stable"))
        return ()

    def deliberate(self, prop: StrategyProposal) -> DecisionMemo:
        did = "AT-" + hashlib.sha256(f"{prop.strategy}|{prop.asof.date()}|{json.dumps(prop.targets, sort_keys=True)}".encode()).hexdigest()[:10]
        items, shadow = [], dict(prop.targets)
        sec_w: dict[str, float] = {}
        for s, w in prop.targets.items():
            sec_w[prop.sector.get(s, "?")] = sec_w.get(prop.sector.get(s, "?"), 0.0) + w
        for sym, w in sorted(prop.targets.items()):
            action = "BUY" if prop.previous.get(sym, 0.0) <= 1e-9 else "HOLD"
            self._log("head->company_intel", {"decision": did, "symbol": sym, "asof": str(prop.asof.date())})
            d = self._cia.dossier(sym, prop.asof)
            self.dossiers[sym] = d
            self._log("company_intel->head", {"decision": did, "symbol": sym, "coverage": d.provenance_coverage(),
                                              "statements": len(d.statements())})
            req = ChallengeRequest(did, action, sym, float(w), prop.horizon_sessions, str(prop.asof.date()), prop.strategy,
                                   self._thesis(prop, sym, d), sec_w.get(prop.sector.get(sym, "?")))
            self._log("head->tenth_man", asdict(req))
            rep: ChallengeReport = self._tenth.review(req)
            rj = json.loads(json.dumps(asdict(rep), default=str)) | {"status": rep.status}
            self._log("tenth_man->head", {"decision": did, "symbol": sym, "evidence_status": rep.evidence_status, "recommendation": rep.recommendation,
                                          "historical_probability": rep.historical.value, "statistical_flag": rep.historical.statistical_flag})
            would = {"PROCEED": ("KEEP", w), "REDUCE_CONFIDENCE": ("HALVE", w / 2), "ABSTAIN": ("DROP", 0.0)}[rep.recommendation]
            shadow[sym] = would[1]
            items.append(MemoItem(sym, action, float(w), d.provenance_coverage(), rj, would[0], float(would[1])))
        final = dict(prop.targets) if self.policy == "SHADOW" else {k: v for k, v in shadow.items() if v > 0}
        import inspect
        tsha = hashlib.sha256(inspect.getsource(MainHead._thesis).encode()).hexdigest()[:16]
        memo = DecisionMemo(did, prop.strategy, str(prop.asof.date()), self.policy, items, final, {k: v for k, v in shadow.items() if v > 0}, tsha)
        self._log("head_memo", {"decision": did, "policy": self.policy, "final": final, "shadow": memo.shadow_targets})
        return memo
