"""The 10th Man: an isolated adversarial reviewer.

When the Main Head has a proposed decision, the 10th Man's job is to try to falsify it, not to agree. It works only
on a ChallengeRequest built by the Head (action, asset, size, horizon, and the thesis claims the Head chooses to
expose with their evidence ids). It never sees other agents' reasoning and holds no reference to any agent: its only
inputs are that request and read-only market data / the Evidence Ledger, from which it researches independently.

It may answer NO_MATERIAL_OBJECTION. Its failure probability comes from a statistical base-rate model fitted on
historical outcomes known strictly before the decision; if that model does not beat the plain base rate out of sample,
only the base rate is reported ("flags not predictive"). It never places or changes orders; it advises the Head.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from intel.evidence import EvidenceLedger, Statement

from .context import MarketContext

OBJECTIVES = {"BUY": "How can this investment fail?", "SELL": "Why could selling be a mistake?",
              "HOLD": "What would make continuing to hold wrong?", "SHORT": "What could make this short fail violently?",
              "NO_TRADE": "What evidence suggests we are missing an opportunity?"}
FEATURES = ["bad20", "extreme_loss", "vol_rank", "bear", "dq_lt1"]


@dataclass(frozen=True)
class ThesisClaim:
    text: str
    evidence: tuple[str, ...] = ()
    key: str = ""                        # e.g. "no_adverse_news", "fundamentals_stable" (what the claim asserts)


@dataclass(frozen=True)
class ChallengeRequest:
    """Everything the 10th Man is allowed to know about a decision. Built only by the Main Head."""
    decision_id: str
    action: str                          # BUY | SELL | HOLD | SHORT | NO_TRADE
    symbol: str
    weight: float
    horizon_sessions: int
    asof: str                            # ISO date of the decision close
    strategy: str
    thesis: tuple[ThesisClaim, ...] = ()
    sector_weight_after: float | None = None


@dataclass
class Finding:
    check: str
    severity: str                        # info | medium | high
    statement: Statement
    attacks: str = ""                    # thesis key it contradicts, if any


@dataclass
class HistoricalProbability:
    """Owned by the statistical layer, never by the reviewer's reasoning."""
    value: float | None                  # shown probability of underperforming over the horizon
    base_rate: float | None
    model_skill: float | None            # out-of-sample Brier skill of the flag model
    n_history: int
    statistical_flag: bool               # model beats the base rate AND puts this pick >= 5 pp above it
    note: str
    source: str = "statistical layer: expanding logistic model vs base rate (agents.tenth_man.BaseRateModel)"


@dataclass
class FailureScenario:
    condition: str                       # "If ..."
    consequence: str                     # "... then ..."
    evidence: list[str]
    status: str                          # status of the statement it rests on (no probability is ever attached)


@dataclass
class ChallengeReport:
    decision_id: str
    symbol: str
    action: str
    objective: str
    historical: HistoricalProbability
    evidence_status: str                 # MATERIAL | NONE  (from evidence only)
    severity: str                        # none | high
    recommendation: str                  # PROCEED | ABSTAIN  (advice to the Head only, from the evidence status)
    findings: list[Finding] = field(default_factory=list)
    scenarios: list[FailureScenario] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "MATERIAL_CHALLENGE" if self.evidence_status == "MATERIAL" else "NO_MATERIAL_OBJECTION"


SCENARIOS = {
    "adverse company news": ("If the market keeps repricing the weaker results", "the loss does not revert"),
    "fundamental deterioration": ("If profit keeps falling", "the price is repricing earnings, not overreacting"),
    "scheduled binary event": ("If the results inside the holding month disappoint", "a fresh bad reaction lands while the position is held"),
    "hostile regime": ("If NIFTY stays below its 200-session average", "market-wide selling can dominate stock-level reversal"),
    "alternative explanation": ("If the sector repricing continues", "the loss does not revert"),
    "tail risk": ("If the stock's high volatility persists", "a large adverse move within the holding month is possible"),
    "data trust": ("If the unconfirmed decision-day price is wrong", "the selection itself may be an artefact of bad data"),
    "concentration": ("If the sector falls as a block", "several positions lose together"),
}


# ----------------------------------------------------------------------------- base-rate model
class BaseRateModel:
    """Expanding-window logistic regression of 'the selected stock underperforms the equal-weight universe over the
    holding month' on fixed flags. Fit only on outcomes known before the decision."""

    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self.coef: np.ndarray | None = None
        self.base: float | None = None
        self.skill: float | None = None
        self.n = 0

    @staticmethod
    def _fit(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
        A = np.column_stack([np.ones(len(X)), X])
        w = np.zeros(A.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-A @ w))
            g = A.T @ (p - y) + ridge * np.r_[0, w[1:]]
            H = (A * (p * (1 - p))[:, None]).T @ A + ridge * np.diag(np.r_[0, np.ones(len(w) - 1)])
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return w

    def fit(self, hist: pd.DataFrame, before: pd.Timestamp) -> "BaseRateModel":
        h = hist[hist["known_at"] < before].dropna(subset=FEATURES + ["fail"])
        self.n = len(h)
        if len(h) < 100:
            self.coef, self.base = None, (float(h["fail"].mean()) if len(h) else None)
            return self
        self.coef = self._fit(h[FEATURES].to_numpy(float), h["fail"].to_numpy(float), self.ridge)
        self.base = float(h["fail"].mean())
        return self

    def predict(self, x: dict) -> float | None:
        if self.coef is None:
            return self.base
        v = np.r_[1.0, [float(x[k]) for k in FEATURES]]
        return float(1 / (1 + np.exp(-v @ self.coef)))

    @staticmethod
    def walk_forward(hist: pd.DataFrame, min_months: int = 24) -> dict:
        """Out-of-sample Brier score of the model vs the expanding base rate, month by month."""
        months = sorted(hist["decision"].unique())
        rows = []
        for k, m in enumerate(months):
            if k < min_months:
                continue
            mdl = BaseRateModel().fit(hist, m)
            cur = hist[hist["decision"] == m].dropna(subset=FEATURES + ["fail"])
            for r in cur.itertuples(index=False):
                p = mdl.predict({f: getattr(r, f) for f in FEATURES})
                rows.append({"decision": m, "fail": r.fail, "p_model": p, "p_base": mdl.base})
        R = pd.DataFrame(rows)
        if R.empty:
            return {"n": 0}
        bm = float(((R["p_model"] - R["fail"]) ** 2).mean())
        bb = float(((R["p_base"] - R["fail"]) ** 2).mean())
        hi = R[R["p_model"] - R["p_base"] >= 0.05]
        return {"n": int(len(R)), "months": int(R["decision"].nunique()), "brier_model": bm, "brier_base": bb,
                "skill": 1 - bm / bb if bb > 0 else None, "fail_rate": float(R["fail"].mean()),
                "flagged_n": int(len(hi)), "flagged_fail_rate": float(hi["fail"].mean()) if len(hi) else None, "_rows": R}


def features_for(ctx: MarketContext, t: int, symbols: list[str]) -> pd.DataFrame:
    """The fixed base-rate flags (pre-registration v0.6, section 4) for `symbols` at the close of session t, computed
    from data only (no agent output): BAD20, extreme last-month loss (top decile of the universe's loss), 1-year
    volatility rank, NIFTY below its 200-session average, decision-day data quality below 1."""
    from lab.factors import stock_universe

    P = ctx.P
    C = ctx.close()
    uni = [P.symbols[j] for j in stock_universe(P, t)]
    loss = pd.Series({s: -(C[t, P.idx[s]] / C[t - 21, P.idx[s]] - 1) for s in uni if t >= 21})
    vol = {}
    for s in uni:
        r = np.diff(np.log(C[max(t - 252, 0):t + 1, P.idx[s]]))
        r = r[np.isfinite(r)]
        vol[s] = np.std(r) if len(r) > 100 else np.nan
    vr = pd.Series(vol).rank(pct=True)
    b = P.bench.ffill()
    ma = b.rolling(200, min_periods=200).mean().iloc[t]
    ev = ctx.events
    bad = set(ev.loc[(ev["bad"] == 1) & (ev["reaction_idx"] <= t) & (ev["reaction_idx"] >= t - 19), "symbol"])
    rows = {}
    for s in symbols:
        q = ctx.quality.get(s)
        qd = float(q.get(P.dates[t], 1.0)) if q is not None else 1.0
        rows[s] = {"bad20": float(s in bad), "extreme_loss": float(loss.get(s, np.nan) >= loss.quantile(0.9)) if len(loss) else np.nan,
                   "vol_rank": float(vr.get(s, np.nan)), "bear": float(b.iloc[t] < ma) if np.isfinite(ma) else 0.0, "dq_lt1": float(qd < 1.0)}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- the reviewer
class TenthMan:
    def __init__(self, ctx: MarketContext, ledger: EvidenceLedger, model: BaseRateModel, skill: float | None) -> None:
        self.ctx, self.ev, self.model, self.skill = ctx, ledger, model, skill

    def review(self, req: ChallengeRequest) -> ChallengeReport:
        if not isinstance(req, ChallengeRequest):
            raise TypeError("the 10th Man only accepts a ChallengeRequest from the Main Head")
        from intel.sources import dataset_extract, filing_documents, filing_quotes

        ctx, sym = self.ctx, req.symbol
        t = ctx.idx(pd.Timestamp(req.asof))
        features = features_for(ctx, t, [sym]).iloc[0].to_dict()          # its own research, from data only
        a_iso = (pd.Timestamp(req.asof).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)).isoformat()
        F: list[Finding] = []
        missing: list[str] = []
        C = ctx.close()
        j = ctx.P.idx[sym]

        def st(text, quotes, kind="derived"):
            return self.ev.resolve(Statement(text, list(quotes), quotes, kind), a_iso)

        def extract(symbol, i, vals):
            d = self.ev.add(dataset_extract(symbol, ctx.P.dates[i], vals, ctx.dataset_hash))
            return {d: [f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}" for k, v in vals.items()]}

        f = ctx.filings(sym, pd.Timestamp(req.asof))
        fdocs = {d.url: d for d in filing_documents(f, sym)}
        for d in fdocs.values():
            self.ev.add(d)
        if req.action in ("BUY", "HOLD"):
            # 1. was the move news-driven? Results inside the BAD20 window are context; they contradict the premise
            #    "no adverse company news" only if the released results themselves show a year-on-year profit decline
            #    (tier-1 numbers). The size of the market reaction (tier-9 data, a statistical flag) never decides it.
            ev = ctx.events[(ctx.events["symbol"] == sym) & (ctx.events["reaction_idx"] <= t) & (ctx.events["reaction_idx"] >= t - 19)]
            if len(ev):
                r = ev.iloc[-1]
                i = int(r["reaction_idx"])
                q = extract(sym, i, {f"adjusted close {ctx.P.dates[i - 1].date()}": float(C[i - 1, j]), f"adjusted close {ctx.P.dates[i].date()}": float(C[i, j]),
                                     f"NIFTY50 {ctx.P.dates[i - 1].date()}": float(ctx.P.bench.iloc[i - 1]), f"NIFTY50 {ctx.P.dates[i].date()}": float(ctx.P.bench.iloc[i])})
                F.append(Finding("results in the loss window", "info",
                                 st(f"{sym} reported results within the last 20 sessions; market-adjusted reaction {r['ear']:+.1%} on {ctx.P.dates[i].date()}.", q)))
                pe = pd.Timestamp(r["period_end"])
                cur = f[pd.to_datetime(f["period_end"]) == pe]
                prv = f[pd.to_datetime(f["period_end"]) == pe - pd.DateOffset(years=1)]
                if len(cur) and len(prv) and pd.notna(cur.iloc[-1]["net_profit"]) and pd.notna(prv.iloc[-1]["net_profit"]) and prv.iloc[-1]["net_profit"] > 0:
                    g0 = cur.iloc[-1]["net_profit"] / prv.iloc[-1]["net_profit"] - 1
                    dc, dp = fdocs[str(cur.iloc[-1]["xbrl"])], fdocs[str(prv.iloc[-1]["xbrl"])]
                    if g0 < 0 and "net profit" in filing_quotes(dc) and "net profit" in filing_quotes(dp):
                        F.append(Finding("adverse company news", "high",
                                         st(f"The results released in the loss window show net profit {g0:+.0%} year on year: the loss followed adverse company news.",
                                            {dc.doc_id: [filing_quotes(dc)["net profit"], filing_quotes(dc)["quarter ended"]],
                                             dp.doc_id: [filing_quotes(dp)["net profit"], filing_quotes(dp)["quarter ended"]]}), "no_adverse_news"))
            # 2. fundamentals (computed from two tier-1 filings)
            if len(f) >= 5:
                last = f.iloc[-1]
                prev = f[pd.to_datetime(f["period_end"]) == pd.Timestamp(last["period_end"]) - pd.DateOffset(years=1)]
                if len(prev) and pd.notna(last["net_profit"]) and pd.notna(prev.iloc[-1]["net_profit"]) and prev.iloc[-1]["net_profit"] > 0:
                    g = last["net_profit"] / prev.iloc[-1]["net_profit"] - 1
                    dl, dp = fdocs[str(last["xbrl"])], fdocs[str(prev.iloc[-1]["xbrl"])]
                    if g < -0.2 and "net profit" in filing_quotes(dl) and "net profit" in filing_quotes(dp):
                        F.append(Finding("fundamental deterioration", "medium",
                                         st(f"Net profit {g:+.0%} year on year in the latest filed quarter.",
                                            {dl.doc_id: filing_quotes(dl)["net profit"], dp.doc_id: filing_quotes(dp)["net profit"]}), "fundamentals_stable"))
                age = (pd.Timestamp(req.asof) - pd.Timestamp(last["period_end"])).days
                if age > 120:
                    missing.append(f"Latest stored results are {age} days old: a newer quarter may already be public.")
            else:
                missing.append("Fewer than five quarters of PIT filings: fundamentals cannot be checked.")
            # 3. results due inside the holding period (binary event), from exchange board-meeting intimations
            bm = [d for d in self.ev.as_of(sym, a_iso) if d.source == "NSE board meetings"]
            seen_dates = set()
            for d in bm:
                if "result" not in d.text.lower():
                    continue
                try:
                    md = pd.Timestamp(d.title.split(":")[0].replace("Board meeting ", ""))
                except (ValueError, TypeError):
                    continue
                if pd.Timestamp(req.asof) <= md <= pd.Timestamp(req.asof) + pd.Timedelta(days=req.horizon_sessions * 1.45) and md not in seen_dates:
                    seen_dates.add(md)
                    F.append(Finding("scheduled binary event", "info", st(f"Results board meeting on {md.date()} falls inside the holding period.",
                                                                       {d.doc_id: d.text[:120]}, "fact")))
            if not bm:
                missing.append("Board-meeting calendar not available: a results date inside the holding period cannot be ruled out.")
            # 4. regime and tail risk
            if features.get("bear") == 1:
                ma = float(ctx.P.bench.ffill().rolling(200, min_periods=200).mean().iloc[t])
                F.append(Finding("hostile regime", "info", st("NIFTY 50 is below its 200-session average at the decision close.",
                                                             extract("NIFTY50", t, {f"NIFTY50 {ctx.P.dates[t].date()}": float(ctx.P.bench.iloc[t]), "200-session average": ma}))))
            if features.get("vol_rank", 0) >= 0.8:
                F.append(Finding("tail risk", "info", st(f"{sym}'s 1-year volatility is in the top fifth of the index.",
                                                        extract(sym, t, {"252-session volatility percentile in the index": float(features["vol_rank"])}))))
            # 5. alternative explanation: a sector-wide move (computed; never a tier-1/2 contradiction)
            peers = [s for s in ctx.P.symbols if ctx.P.sector[ctx.P.idx[s]] == ctx.P.sector[j] and s != sym and ctx.P.eligible[t, ctx.P.idx[s]]]
            if len(peers) >= 2:
                sr = float(np.nanmean([ctx.ret(s, t, 21) for s in peers]))
                own = ctx.ret(sym, t, 21)
                if sr < -0.05 and own - sr > -0.02:
                    F.append(Finding("alternative explanation", "medium",
                                     st(f"The month's loss is mostly sector-wide ({ctx.P.sector[j]} peers {sr:+.1%}, {sym} {own:+.1%}): possibly a sector "
                                        "repricing rather than a stock-specific overreaction.",
                                        extract(sym, t, {f"{sym} 21-session return": own, "sector peers' mean 21-session return": sr})),
                                     "idiosyncratic_overreaction"))
            # 6. concentration (from the Head's own request)
            if req.sector_weight_after is not None and req.sector_weight_after > 0.25:
                F.append(Finding("concentration", "medium", Statement(f"Sector weight after this trade would be {req.sector_weight_after:.0%}.", kind="label", status="LABEL")))
            # 7. data trust
            if features.get("dq_lt1") == 1:
                F.append(Finding("data trust", "medium", st("The decision-day price was not fully confirmed by the exchange (quality score < 1).",
                                                            extract(sym, t, {"exchange quality score < 1": "yes"}))))
        else:
            missing.append(f"Checks for {req.action} decisions are limited in v0.6 (objective recorded only).")
        missing.append("No analyst consensus, promoter pledge, insider trade or verified social-media feed is connected yet.")

        # 1. historical probability: statistical layer only
        p = self.model.predict(features)
        base = self.model.base
        usable = self.skill is not None and self.skill > 0 and p is not None
        note = (f"calibrated: expanding logistic model on {self.model.n} past selections, out-of-sample Brier skill {self.skill:+.3f}" if usable else
                f"flags not predictive (out-of-sample Brier skill {self.skill:+.3f} <= 0): base rate only" if self.skill is not None else "base rate only")
        hist = HistoricalProbability(p if usable else base, base, self.skill, self.model.n,
                                     bool(usable and base is not None and p - base >= 0.05), note)
        # 2. evidence challenge: a contradiction of a thesis claim resting on tier-1/2 external evidence; AuTrad's own
        #    computations (tier 9) can raise findings and scenarios but never a contradiction on their own
        contradiction = any(fd.attacks and fd.attacks in {c.key for c in req.thesis} and fd.statement.status in ("SOURCED", "CORROBORATED", "DERIVED")
                            and (self.ev.min_external_tier(fd.statement) or 9) <= 2 for fd in F)
        # 3. failure scenarios: "if ... then ..." from the findings, citing their evidence, never with a probability
        scen = []
        for fd in F:
            if fd.check in SCENARIOS and (fd.severity in ("medium", "high") or fd.check in ("scheduled binary event", "hostile regime")):
                c_, q_ = SCENARIOS[fd.check]
                scen.append(FailureScenario(c_, q_, list(fd.statement.evidence), fd.statement.status))
        return ChallengeReport(req.decision_id, sym, req.action, OBJECTIVES.get(req.action, ""), hist,
                               "MATERIAL" if contradiction else "NONE", "high" if contradiction else "none",
                               "ABSTAIN" if contradiction else "PROCEED", F, scen, missing)
