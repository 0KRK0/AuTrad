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
class ChallengeReport:
    decision_id: str
    symbol: str
    action: str
    objective: str
    status: str                          # NO_MATERIAL_OBJECTION | MATERIAL_CHALLENGE
    severity: str                        # none | medium | high
    recommendation: str                  # PROCEED | REDUCE_CONFIDENCE | ABSTAIN  (advice to the Head only)
    p_fail: float | None
    base_rate: float | None
    model_skill: float | None
    probability_note: str
    findings: list[Finding] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)


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
            # 1. was the move news-driven? (attacks the reversal premise "no adverse news"); same window as BAD20
            ev = ctx.events[(ctx.events["symbol"] == sym) & (ctx.events["reaction_idx"] <= t) & (ctx.events["reaction_idx"] >= t - 19)]
            if len(ev):
                r = ev.iloc[-1]
                i = int(r["reaction_idx"])
                q = extract(sym, i, {f"adjusted close {ctx.P.dates[i - 1].date()}": float(C[i - 1, j]), f"adjusted close {ctx.P.dates[i].date()}": float(C[i, j]),
                                     f"NIFTY50 {ctx.P.dates[i - 1].date()}": float(ctx.P.bench.iloc[i - 1]), f"NIFTY50 {ctx.P.dates[i].date()}": float(ctx.P.bench.iloc[i])})
                fd = f[pd.to_datetime(f["period_end"]) == pd.Timestamp(r["period_end"])]
                if len(fd):
                    dd = fdocs[str(fd.iloc[0]["xbrl"])]
                    q[dd.doc_id] = filing_quotes(dd)["quarter ended"]
                bad = r.get("bad", np.nan) == 1
                F.append(Finding("news-driven move", "high" if bad else "info",
                                 st(f"{sym} reported results within the last 20 sessions; market-adjusted reaction {r['ear']:+.1%} on {ctx.P.dates[i].date()}"
                                    + (" — a bad reaction (lower third of all earlier ones): the move is news, which has tended to keep drifting." if bad else "."), q),
                                 "no_adverse_news" if bad else ""))
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

        p = self.model.predict(features)
        base = self.model.base
        usable = self.skill is not None and self.skill > 0 and p is not None
        note = (f"calibrated: expanding logistic model on {self.model.n} past selections, out-of-sample Brier skill {self.skill:+.3f}" if usable else
                f"flags not predictive (out-of-sample Brier skill {self.skill:+.3f} <= 0): base rate only" if self.skill is not None else "base rate only")
        p_show = p if usable else base
        # a contradiction counts only if it rests on tier-1/2 external evidence (exchange filings, audited statements):
        # AuTrad's own computations (tier-9 extracts) can raise findings but never a contradiction on their own
        contradiction = any(fd.attacks and fd.attacks in {c.key for c in req.thesis} and fd.statement.status in ("SOURCED", "CORROBORATED", "DERIVED")
                            and (self.ev.min_external_tier(fd.statement) or 9) <= 2 for fd in F)
        material = contradiction or (usable and base is not None and p - base >= 0.05)
        sev = "none" if not material else ("high" if contradiction or (usable and p - base >= 0.10) else "medium")
        rec = "PROCEED" if not material else ("ABSTAIN" if sev == "high" else "REDUCE_CONFIDENCE")
        return ChallengeReport(req.decision_id, sym, req.action, OBJECTIVES.get(req.action, ""), "MATERIAL_CHALLENGE" if material else "NO_MATERIAL_OBJECTION",
                               sev, rec, p_show, base, self.skill, note, F, missing)
