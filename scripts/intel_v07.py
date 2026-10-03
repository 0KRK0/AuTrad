"""v0.7 company memory: evidence & claim graph, and the management-promise pilot.

    python -m scripts.intel_v07 transcripts --budget 150      # on your machine: NSE-filed earnings-call transcripts (pilot companies)
    python -m scripts.intel_v07 build --out reports/v0.7_<ts>  # graph from filings, reactions, evidence ledger, shadow memo, promises
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from intel.evidence import EvidenceLedger, Statement, utc
from intel.graph import EvidenceGraph
from intel.transcripts import (METRICS, NUMBER, document_for, download_text, due_date, is_full_transcript, promise_candidates,
                               transcript_announcements)

PILOT = ["ULTRACEMCO", "MARUTI", "TATASTEEL", "TCS", "HINDUNILVR", "LT"]
TR = Path("data/intel/transcripts")
TODAY = pd.Timestamp("2026-10-03")


def fetch(budget: float) -> None:
    from marketdata.nse_filings import Client

    c, t0 = Client(pause=0.3), time.time()
    for s in PILOT:
        idx = TR / s / "index.json"
        if not idx.exists():
            items = []
            for y in range(2022, 2026):
                items += transcript_announcements(c, s, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"))
            idx.parent.mkdir(parents=True, exist_ok=True)
            idx.write_text(json.dumps(items, indent=0))
        for a in json.loads(idx.read_text()):
            if time.time() - t0 > budget:
                print("budget reached; run again")
                return
            download_text(c, a, TR / s)
    print("transcripts:", {s: len(list((TR / s).glob("*.txt"))) for s in PILOT})


def annual_revenue_growth(facts: pd.DataFrame, sym: str, fy_end: pd.Timestamp, known_by: pd.Timestamp) -> tuple[float | None, list[str]]:
    f = facts[(facts["symbol"] == sym) & (pd.to_datetime(facts["available_at"], utc=True) <= known_by.tz_localize("UTC"))].copy()
    f["pe"] = pd.to_datetime(f["period_end"])
    f = f.sort_values("available_at").drop_duplicates("pe", keep="last")

    def fy(end):
        q = f[(f["pe"] > end - pd.DateOffset(years=1)) & (f["pe"] <= end)]
        return (float(q["revenue"].sum()), list(q["xbrl"])) if len(q) == 4 and q["revenue"].notna().all() else (None, [])
    a, da = fy(fy_end)
    b, db = fy(fy_end - pd.DateOffset(years=1))
    return (a / b - 1, da + db) if a and b else (None, [])


def build(out: Path) -> None:
    from intel.sources import filing_documents
    from marketdata.fundamentals import load_facts

    G = EvidenceGraph("data/intel/graph.sqlite")
    E = EvidenceLedger("data/intel/evidence_r1.jsonl")
    if Path("data/intel/evidence_tenth_man.jsonl").exists():
        E.docs.update(EvidenceLedger("data/intel/evidence_tenth_man.jsonl").docs)
    for d in E.docs.values():
        if d.source.startswith("news:") and d.tier == 8:
            d.tier = 7                                     # v0.6 tier 8 'unknown media' = v2 tier 7 'other media'
        G.add_document(d)
    facts = pd.read_csv("data/pit/nse/facts.csv")
    lf = load_facts("data/pit/nse/facts.csv")
    doc_by_url = {}
    for sym in sorted(facts["symbol"].unique()):
        for d in filing_documents(lf, sym):
            G.add_document(d)
            doc_by_url[d.url] = d.doc_id
    for r in lf.itertuples(index=False):
        for m in ("revenue", "net_profit", "total_assets"):
            v = getattr(r, m)
            if pd.notna(v) and str(r.xbrl) in doc_by_url:
                G.db.execute("INSERT OR IGNORE INTO metrics VALUES (?,?,?,?,?)", (r.symbol, m, str(pd.Timestamp(r.period_end).date()), float(v), doc_by_url[str(r.xbrl)]))
    # results reactions (from the v0.6 research context)
    ev_path = Path("data/intel/events_v06.csv")
    if ev_path.exists():
        ev = pd.read_csv(ev_path)
        for r in ev.itertuples(index=False):
            G.db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)", (f"R:{r.symbol}:{r.period_end}", r.symbol, "results_reaction" + ("_bad" if r.bad == 1 else ""),
                                                                               str(r.reaction_day)[:10], float(r.ear), None))
    # latest valid shadow memo -> decisions, thesis, challenges
    from forward.ledger import Ledger
    from head.scorecard import valid_memos

    for month, rec in valid_memos(Ledger("forward/shadow/ledger.jsonl")).items():
        m = rec["memo"]
        for it in m["items"]:
            c = it["challenge"]
            hp = (c.get("historical") or {}).get("value", c.get("p_fail"))
            es = c.get("evidence_status", "MATERIAL" if c.get("status") == "MATERIAL_CHALLENGE" else "NONE")
            G.db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)", (m["decision_id"], it["symbol"], m["strategy"], m["asof"], it["action"], es, hp, rec["hash"]))
            node = f"{m['decision_id']}:{it['symbol']}"
            for f in c["findings"]:
                st = f["statement"]
                have = [e for e in st["evidence"] if e in E.docs]
                miss = [e for e in st["evidence"] if e not in E.docs]
                cid = G.add_claim("finding", it["symbol"], st["text"], m["asof"], have,
                                  status=st["status"] + (f" (evidence not persisted: {len(miss)} id(s))" if miss else ""), origin="10th Man")
                G.link(node, "challenged_by", cid, f["check"])
        # thesis claims are in the request logged in the audit trail
    aud = Ledger("forward/shadow/audit.jsonl").records()
    valid_ids = {r["memo"]["decision_id"] for r in valid_memos(Ledger("forward/shadow/ledger.jsonl")).values()}
    last_memo = max((r["seq"] for r in aud if r["kind"] == "head_memo"), default=-1)
    prev_memo = max((r["seq"] for r in aud if r["kind"] == "head_memo" and r["seq"] < last_memo), default=-1)
    for r in [x for x in aud if x["kind"] == "head->tenth_man" and prev_memo < x["seq"] < last_memo and x["decision_id"] in valid_ids]:
        node = f"{r['decision_id']}:{r['symbol']}"
        for th in r.get("thesis", []):
            cid = G.add_claim("thesis", r["symbol"], th["text"], r["asof"], [e for e in th["evidence"] if e in E.docs], status="ASSERTED BY HEAD", origin=th["key"])
            G.link(node, "thesis", cid)
    # management-promise pilot
    rows = []
    for sym in PILOT:
        idx = TR / sym / "index.json"
        if not idx.exists():
            continue
        items = json.loads(idx.read_text())
        docs = []
        for a in items:
            txt = TR / sym / (Path(a["attchmntFile"]).stem + ".txt")
            if txt.exists():
                text = txt.read_text(errors="ignore")
                if not is_full_transcript(text):
                    continue
                d = document_for(a, sym, text)
                E.docs.setdefault(d.doc_id, d)
                G.add_document(d)
                docs.append((d, text))
        docs.sort(key=lambda x: x[0].published_at)
        for d, text in docs:
            said = utc(d.published_at).tz_convert(None)
            for c in promise_candidates(text):
                st = E.resolve(Statement(c["quote"], [d.doc_id], {d.doc_id: c["quote"]}), d.published_at)
                if st.status not in ("SOURCED", "CORROBORATED"):
                    continue                                   # never store a promise we cannot quote verbatim
                due = due_date(c["anchor"], said)
                status, outcome, value = "", [], None
                last_doc = max(utc(dd.published_at).tz_convert(None) for dd, _ in docs)
                if not due:
                    status = "UNRESOLVABLE (no usable due date)"
                elif pd.Timestamp(due) > TODAY:
                    status = "PENDING (due date not reached)"
                elif c["metric"] == "revenue_growth" and c["target"].endswith("%") and pd.Timestamp(due).month == 3:
                    g, srcs = annual_revenue_growth(facts, sym, pd.Timestamp(due), TODAY)
                    if g is not None:
                        value = g
                        status = "MET" if g >= float(re.sub(r"[^\d.]", "", c["target"])) / 100 else "MISSED"
                        outcome = [(doc_by_url[u], "") for u in srcs if u in doc_by_url]
                if not status:
                    kw = dict(METRICS).get(c["metric"], "")
                    if not kw:
                        status = "DUE: needs human review (no measurable keyword)"
                    elif pd.Timestamp(due) + pd.Timedelta(days=400) > last_doc:
                        status = "DUE: outcome window not yet covered by the fetched transcripts"
                    else:
                        for dd, tt in docs:
                            when = utc(dd.published_at).tz_convert(None)
                            if dd.published_at > d.published_at and pd.Timestamp(due) - pd.Timedelta(days=120) <= when <= pd.Timestamp(due) + pd.Timedelta(days=400):
                                for sent in re.split(r"(?<=[.!?])\s+", " ".join(tt.split())):
                                    if re.search(kw, sent, re.I) and NUMBER.search(sent) and len(sent) < 400:
                                        outcome.append((dd.doc_id, sent))
                                        break
                        status = "DUE: candidate outcome passages found (human review)" if outcome else "DUE: no outcome passage found"
                cid = G.add_claim("promise", sym, c["quote"], d.published_at, [d.doc_id], quote=c["quote"], speaker=c["speaker"], due=due,
                                  metric=c["metric"], target=c["target"], status=status, origin="transcript")
                for od, passage in outcome[:3]:
                    G.link(od, "outcome" if status in ("MET", "MISSED") else "outcome_candidate", cid, passage[:400])
                rows.append({"symbol": sym, "said": d.published_at[:10], "speaker": c["speaker"], "metric": c["metric"], "target": c["target"],
                             "due": due, "status": status, "value": value, "quote": c["quote"]})
    G.commit()
    P = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "promises_pilot.csv", index=False)
    res = {"graph": G.stats(), "promises": int(len(P)),
           "by_status": P["status"].value_counts().to_dict() if len(P) else {}, "by_metric": P["metric"].value_counts().to_dict() if len(P) else {},
           "by_company": P["symbol"].value_counts().to_dict() if len(P) else {},
           "transcripts": {s: {"filed": len(list((TR / s).glob("*.txt"))),
                               "full_text": sum(is_full_transcript(p.read_text(errors="ignore")) for p in (TR / s).glob("*.txt"))} for s in PILOT}}
    d0 = G.db.execute("SELECT decision_id, symbol FROM decisions ORDER BY symbol LIMIT 1").fetchone()
    res["trace_example"] = G.trace(*d0) if d0 else {}
    (out / "company_memory.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "trace_example"}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["transcripts", "build"])
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--out", default="reports/v0.7")
    a = ap.parse_args()
    fetch(a.budget) if a.step == "transcripts" else build(Path(a.out))


if __name__ == "__main__":
    main()
