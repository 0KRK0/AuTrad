"""v0.6 advisory agents (research/preregistration_v0.6.md, section 4).

    python -m scripts.agents_v06 calibrate --out reports/v0.6_<ts>     # 10th Man base-rate model: walk-forward Brier skill
    python -m scripts.agents_v06 shadow --out reports/v0.6_<ts> [--news] # Head + Company Intelligence + 10th Man on the
                                                                          # REV1 forward decisions (shadow; REV1 untouched)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from agents.company_intel import CompanyIntelligenceAgent
from agents.context import MarketContext
from agents.tenth_man import FEATURES, BaseRateModel, TenthMan, features_for
from forward.ledger import Ledger
from head.main_head import MainHead, StrategyProposal
from intel.claims import ClaimTracker
from intel.evidence import EvidenceLedger
from scripts.v05 import clean

DATA = "data/snap_20261001"
HIST = Path("data/intel/rev1_pick_history.csv")


class LedgerAudit:
    def __init__(self, path: Path) -> None:
        self.L = Ledger(path)

    def write(self, kind: str, payload: dict) -> str:
        return self.L.append(kind, json.loads(json.dumps(payload, default=str)))["hash"]


def research_context():
    from lab.panel import build_panel, india_equity_assets
    from marketdata.fundamentals import load_facts, split_events_from_store
    from marketdata.store import DataStore
    from scripts.v06 import bad_events

    P = build_panel(DATA, india_equity_assets(), name="india_pit_xopen", open_source="exchange")
    st = DataStore(DATA)
    st.load(list(P.symbols))
    facts = load_facts("data/pit/nse/facts.csv", split_events=split_events_from_store(st))
    ev = bad_events(P)
    ev["reaction_day"] = P.dates[ev["reaction_idx"].to_numpy()]
    names = pd.read_csv("marketdata/reference/nifty50_pit_snapshots.csv").drop_duplicates("symbol", keep="last").set_index("symbol")["company_name"].to_dict()
    pit = pd.read_csv("marketdata/reference/nifty50_pit_snapshots.csv", dtype=str)
    from marketdata.pit import ALIASES
    rows: dict = {}
    for r in pit.itertuples(index=False):
        rows.setdefault(ALIASES.get(r.symbol, r.symbol), []).append((r.effective_date, f"{r.effective_date},{r.symbol},{r.company_name},{r.isin}"))
    return MarketContext(P, facts, ev, dict(st.quality), st.manifest_hash(), names, rows)


def calibrate(out: Path) -> None:
    from lab.factors import decision_dates, forward_returns, price_scores, stock_universe, tercile_targets

    ctx = research_context()
    P = ctx.P
    C = P.ffill_close()
    R = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(np.log(C), axis=0)])
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))]
    dates = decision_dates(P)
    S = pd.DataFrame({P.dates[t]: price_scores(P, t, stock_universe(P, t), C, R, m)["REV1"] for t in dates}).T.reindex(columns=P.symbols)
    tg = tercile_targets(S.dropna(how="all"))
    fwd = forward_returns(P, dates)
    nxt = {P.dates[a]: P.dates[b] for a, b in zip(dates[:-1], dates[1:], strict=True)}
    rows = []
    for d, w in tg.iterrows():
        if d not in fwd.index:
            continue
        picks = list(w[w > 0].index)
        t = P.dates.get_loc(d)
        f = features_for(ctx, t, picks)
        uni_mean = fwd.loc[d, S.loc[d].dropna().index].mean()
        for s in picks:
            r = fwd.at[d, s]
            if np.isfinite(r):
                rows.append({"decision": d, "symbol": s, "known_at": nxt[d], **f.loc[s].to_dict(), "ret": r, "fail": float(r < uni_mean)})
    H = pd.DataFrame(rows)
    HIST.parent.mkdir(parents=True, exist_ok=True)
    H.to_csv(HIST, index=False)
    wf = BaseRateModel.walk_forward(H)
    rr = wf.pop("_rows")
    full = BaseRateModel().fit(H, H["known_at"].max() + pd.Timedelta(days=1))
    res = {"picks": int(len(H)), "months": int(H["decision"].nunique()), "walk_forward": wf,
           "coefficients_full_sample": dict(zip(["intercept"] + FEATURES, map(float, full.coef))) if full.coef is not None else None,
           "fail_rate_by_flag": {f: {"flag=1": float(H.loc[H[f] >= 0.5, "fail"].mean()) if (H[f] >= 0.5).any() else None,
                                     "n=1": int((H[f] >= 0.5).sum()), "flag=0": float(H.loc[H[f] < 0.5, "fail"].mean())}
                                 for f in ["bad20", "extreme_loss", "bear", "dq_lt1"]},
           "verdict": "calibrated (skill > 0): the 10th Man may show the model probability" if (wf.get("skill") or 0) > 0 else
                      "flags not predictive (skill <= 0): the 10th Man shows the base rate only"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "tenth_man_calibration.json").write_text(json.dumps(clean(res), indent=1))
    rr.to_csv(out / "tenth_man_walk_forward.csv", index=False)
    print(json.dumps(clean({k: v for k, v in res.items()}), indent=1)[:2500])


def shadow(out: Path, news: bool, supersede: dict | None = None) -> None:
    from forward.rev1 import load_panel
    from intel.sources import NewsRSS, NSEAnnouncements
    from marketdata.fundamentals import load_facts
    from marketdata.pit import SECTOR
    from marketdata.store import DataStore

    calib = json.loads((out / "tenth_man_calibration.json").read_text())
    skill = calib["walk_forward"].get("skill")
    H = pd.read_csv(HIST, parse_dates=["decision", "known_at"])
    research = research_context()
    fl = Ledger("forward/rev1/ledger.jsonl")
    decs = fl.of_kind("decision")
    sh = Ledger("forward/shadow/ledger.jsonl")
    if supersede:
        sh.append("memo_superseded", supersede)
    latest: dict = {}
    for r in sh.records():                                   # a month is done if its latest record is a memo
        if r["kind"] in ("memo", "memo_superseded"):
            latest[r["month"]] = r["kind"]
    done = {m for m, k in latest.items() if k == "memo"}
    ev_ledger = EvidenceLedger("data/intel/evidence_r1.jsonl")   # r1: after the tier fix (the first file is kept as the audit trail)
    claims = ClaimTracker(ev_ledger, "data/intel/claims_r1.jsonl")
    md = []
    for k, d in enumerate(decs):
        if d["month"] in done:
            continue
        asof = pd.Timestamp(d["date"])
        P = load_panel(Path("data/forward/snap"), "2024-06-01", asof=asof)
        st = DataStore("data/forward/snap")
        st.load(list(P.symbols), asof=str(asof.date()))
        ctx = MarketContext(P, research.facts, research.events, dict(st.quality), st.manifest_hash(), research.company_names, research.pit_rows)
        nse = NSEAnnouncements()
        ann, nw = [], []
        for s in d["selected"]:
            ann += nse.fetch(s, asof - pd.Timedelta(days=90), asof) + nse.board_meetings(s)
            if news:
                nw += NewsRSS().fetch(s, research.company_names.get(s, s))
        from intel.evidence import utc
        cut = utc((asof + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Kolkata").isoformat())
        ann = [a for a in ann if a.published_at and utc(a.published_at) <= cut]
        nw = [a for a in nw if a.published_at and utc(a.published_at) <= cut]
        cia = CompanyIntelligenceAgent(ctx, ev_ledger, claims, announcements=ann, news=nw if news else None)
        model = BaseRateModel().fit(H, asof)
        tm_view = ev_ledger.external_view()                                  # its own copy: external documents only
        tenth = TenthMan(ctx, tm_view, model, skill)
        head = MainHead(cia, tenth, "SHADOW", LedgerAudit(Path("forward/shadow/audit.jsonl")))
        prev = decs[k - 1]["weights"] if k else {}
        try:
            memo = head.deliberate(StrategyProposal("REV1 (frozen, forward)", asof, d["weights"], prev, 21, SECTOR))
        except Exception as e:
            head.audit.write("run_aborted", {"month": d["month"], "error": repr(e)[:300]})
            raise
        tm_store = EvidenceLedger("data/intel/evidence_tenth_man.jsonl")      # persist what the 10th Man cited (raw extracts + filings)
        for doc in tm_view.docs.values():
            if doc.doc_id not in ev_ledger.docs:
                tm_store.add(doc)
        sh.append("memo", {"month": d["month"], "rev1_decision_hash": d["hash"],
                           "memo": json.loads(json.dumps(asdict(memo), default=str)), "evidence_docs": len(ev_ledger.docs)})
        md.append(render_memo(memo, head, ev_ledger, calib))
    if md:
        (out / "shadow_memo.md").write_text("\n".join(md), encoding="utf-8")
        Path("forward/shadow").mkdir(parents=True, exist_ok=True)
        for m_ in md:
            month = m_.split("\n")[0].split()[-1]
            (Path("forward/shadow") / f"memo_{month}.md").write_text(m_, encoding="utf-8")
    print(f"shadow memos written: {len(md)}; evidence documents: {len(ev_ledger.docs)}")


def scorecard(out: Path) -> None:
    from forward.rev1 import load_panel
    from head.scorecard import evaluate, outcomes

    sh, fl = Ledger("forward/shadow/ledger.jsonl"), Ledger("forward/rev1/ledger.jsonl")
    P = load_panel(Path("data/forward/snap"), "2024-06-01")
    df = outcomes(sh, fl, P)
    done = {(r["month"], r["symbol"]) for r in sh.of_kind("outcome")}
    for r in df.itertuples(index=False):
        if (r.month, r.symbol) not in done:
            sh.append("outcome", {"month": r.month, "symbol": r.symbol, "evidence_status": r.evidence_status, "excess": r.excess})
    res = evaluate(df)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shadow_scorecard.json").write_text(json.dumps(clean(res), indent=1))
    print(res)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def render_memo(memo, head: MainHead, ev: EvidenceLedger, calib: dict) -> str:
    L = [f"# Main Head decision memo — REV1 forward, decision close {memo.asof[:7]}", "",
         f"Decision `{memo.decision_id}` · policy **{memo.policy}** (advisory only: REV1's frozen targets are executed unchanged) · "
         f"10th Man probability: {calib['verdict']}", "",
         "| Stock | Action | Weight | Evidence challenge | Recommendation | Historical P(underperform) | Base rate | Dossier provenance | Shadow |", "|---|---|---|---|---|---|---|---|---|"]
    for it in memo.items:
        c = it.challenge
        L.append(f"| {it.symbol} | {it.action} | {it.weight:.2%} | {c['evidence_status']} | {c['recommendation']} | "
                 f"{_pct(c['historical']['value'])} | {_pct(c['historical']['base_rate'])} | "
                 f"{it.dossier_coverage:.0%} | {it.would_do} |")
    L += ["", f"Shadow portfolio (what a challenge-aware Head would hold): {len(memo.shadow_targets)} names; "
          f"final executed targets: {len(memo.final_targets)} names (unchanged).", ""]
    for it in memo.items:
        c = it.challenge
        L += [f"## {it.symbol}", "", f"**10th Man** — {c['objective']}", "",
              f"- Historical probability (statistical layer): {_pct(c['historical']['value'])} — {c['historical']['note']}.",
              f"- Evidence challenge: **{c['evidence_status']}** (a contradiction needs tier-1/2 evidence).", "", "Findings:"]
        L += [f"- [{f['severity']}] {f['check']}: {f['statement']['text']} ({f['statement']['status']})" for f in c["findings"]] or ["- No finding."]
        if c["scenarios"]:
            L += ["", "Failure scenarios (no probabilities):"] + [f"- {x['condition']}, then {x['consequence']}. ({x['status']})" for x in c["scenarios"]]
        if c["missing_information"]:
            L += ["", "Missing information: " + " ".join(c["missing_information"])]
        L += ["", head.dossiers[it.symbol].markdown(ev)]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "shadow", "scorecard"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--news", action="store_true", help="also read Google News RSS headlines (personal research use)")
    ap.add_argument("--supersede", nargs=2, metavar=("MONTH", "REASON"), help="re-run a month's memo; the old memo stays, marked superseded")
    a = ap.parse_args()
    if a.step == "calibrate":
        calibrate(Path(a.out))
    elif a.step == "scorecard":
        scorecard(Path(a.out))
    else:
        shadow(Path(a.out), a.news, {"month": a.supersede[0], "reason": a.supersede[1]} if a.supersede else None)


if __name__ == "__main__":
    main()
