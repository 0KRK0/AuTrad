"""v0.8 company memory expansion: ownership & pledges, transcripts for the whole NIFTY 50 (exchange + company
website), consensus snapshots, and the promise -> outcome review workflow.

    python -m scripts.intel_v08 build --out reports/v0.8_<ts>
    python -m scripts.intel_v08 queue --out reports/v0.8_<ts>          # promises due for human review -> CSV
    python -m scripts.intel_v08 apply --csv <filled queue csv>          # validated human verdicts -> Company Memory
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import pandas as pd

from intel.company_ir import document_for as web_document_for, load_index
from intel.evidence import Document, EvidenceLedger, Statement, Tier, utc
from intel.graph import EvidenceGraph
from intel.promise_review import apply_reviews, ensure, execution_record, review_queue
from intel.transcripts import METRICS, NUMBER, document_for, due_date, is_full_transcript, promise_candidates
from scripts.intel_v07 import annual_revenue_growth

TR = Path("data/intel/transcripts")
WEB = Path("data/intel/transcripts_web")
GRAPH = Path("data/intel/graph.sqlite")
TODAY = pd.Timestamp("2026-10-03")


def transcript_docs(sym: str) -> list[tuple[Document, str, str]]:
    """(document, text, local path) for every usable transcript of `sym`: NSE full-text filings (tier 1) and
    website transcripts with an exchange-intimation timestamp (tier 2) that do not duplicate an NSE filing."""
    out = []
    idx = TR / sym / "index.json"
    nse_times = []
    for a in (json.loads(idx.read_text()) if idx.exists() else []):
        txt = TR / sym / (Path(a["attchmntFile"]).stem + ".txt")
        if not txt.exists():
            continue
        text = txt.read_text(errors="ignore")
        if not is_full_transcript(text):
            continue
        d = document_for(a, sym, text)
        out.append((d, text, str(txt)))
        nse_times.append(utc(d.published_at))
    for w in load_index(WEB):
        if w.symbol != sym or not w.path or not w.full_text or not w.available_at:
            continue
        t = utc(w.available_at)
        if any(abs((t - n).total_seconds()) < 21 * 86400 for n in nse_times):
            continue                                            # same call already on the exchange record
        txt = Path(w.path).with_suffix(".txt")
        text = txt.read_text(errors="ignore")
        d = web_document_for(w, text)
        if d is not None:
            out.append((d, text, str(txt)))
    seen, uniq = set(), []
    for d, text, path in sorted(out, key=lambda x: x[0].published_at):      # earliest copy of an identical text wins
        h = hash(" ".join(text.split()))
        if h not in seen:
            seen.add(h)
            uniq.append((d, text, path))
    return uniq


def build(out: Path) -> dict:
    snap = GRAPH.with_name("graph_v07_snapshot.sqlite")
    if not snap.exists():
        shutil.copy(GRAPH, snap)                                 # the v0.7 graph is kept as it was
    G = EvidenceGraph(GRAPH)
    ensure(G)
    E = EvidenceLedger("data/intel/evidence_v08.jsonl")
    facts = pd.read_csv("data/pit/nse/facts.csv")
    doc_by_url = dict(G.db.execute("SELECT url, doc_id FROM documents").fetchall())
    syms = sorted({p.name for p in TR.iterdir() if p.is_dir()})
    rows, n_docs, dropped_past = [], 0, 0
    for sym in syms:
        docs = transcript_docs(sym)
        for d, text, path in docs:
            E.docs.setdefault(d.doc_id, d)
            G.add_document(d)
            G.db.execute("INSERT OR REPLACE INTO doc_files VALUES (?,?)", (d.doc_id, path))
            n_docs += 1
        if not docs:
            continue
        last_doc = max(utc(dd.published_at).tz_convert(None) for dd, _, _ in docs)
        for d, text, _ in docs:
            said = utc(d.published_at).tz_convert(None)
            for c in promise_candidates(text):
                st = E.resolve(Statement(c["quote"], [d.doc_id], {d.doc_id: c["quote"]}), d.published_at)
                if st.status not in ("SOURCED", "CORROBORATED"):
                    continue                                     # never store a promise we cannot quote verbatim
                due = due_date(c["anchor"], said)
                if due and pd.Timestamp(due) < said.normalize():
                    dropped_past += 1
                    continue                                     # anchor is a past period ("78% in FY'24"): not a promise
                status, outcome, value = "", [], None
                if not due:
                    status = "UNRESOLVABLE (no usable due date)"
                elif pd.Timestamp(due) > TODAY:
                    status = "PENDING (due date not reached)"
                else:
                    if c["metric"] == "revenue_growth" and c["target"].endswith("%") and pd.Timestamp(due).month == 3:
                        g, srcs = annual_revenue_growth(facts, sym, pd.Timestamp(due), TODAY)
                        if g is not None:
                            value = g
                            hit = g >= float(re.sub(r"[^\d.]", "", c["target"])) / 100
                            status = f"DUE: SUGGESTED {'MET' if hit else 'MISSED'} from filed revenue (human review required)"
                            outcome = [(doc_by_url[u], f"filed revenue growth {g:.1%}") for u in srcs if u in doc_by_url]
                    if not status:
                        kw = dict(METRICS).get(c["metric"], "")
                        if not kw:
                            status = "DUE: needs human review (no measurable keyword)"
                        elif pd.Timestamp(due) + pd.Timedelta(days=400) > last_doc:
                            status = "DUE: outcome window not yet covered by the transcripts"
                        else:
                            for dd, tt, _ in docs:
                                when = utc(dd.published_at).tz_convert(None)
                                if dd.published_at > d.published_at and pd.Timestamp(due) - pd.Timedelta(days=120) <= when <= pd.Timestamp(due) + pd.Timedelta(days=400):
                                    for sent in re.split(r"(?<=[.!?])\s+", " ".join(tt.split())):
                                        if re.search(kw, sent, re.I) and NUMBER.search(sent) and len(sent) < 400:
                                            outcome.append((dd.doc_id, sent))
                                            break
                            status = "DUE: candidate outcome passages found (human review)" if outcome else "DUE: no outcome passage found"
                cid = G.add_claim("promise", sym, c["quote"], d.published_at, [d.doc_id], quote=c["quote"], speaker=c["speaker"], due=due,
                                  metric=c["metric"], target=c["target"], status=status, origin=f"transcript tier {d.tier}")
                for od, passage in outcome[:3]:
                    G.link(od, "outcome_candidate", cid, passage[:400])
                rows.append({"symbol": sym, "said": d.published_at[:10], "tier": d.tier, "speaker": c["speaker"], "metric": c["metric"],
                             "target": c["target"], "due": due, "status": status, "value": value, "quote": c["quote"], "claim_id": cid})
    # ownership & pledges (tier 1: exchange filings)
    shp = pd.read_csv("data/pit/bse/shareholding.csv")
    for r in shp.itertuples(index=False):
        txt = (f"{r.symbol} shareholding pattern {r.period_label}: promoter {r.promoter_pct:.2f}%, promoter shares encumbered "
               f"{0 if pd.isna(r.encumbered_pct_of_promoter) else r.encumbered_pct_of_promoter:.2f}% of promoter holding")
        d = Document("BSE shareholding pattern filing", int(Tier.REGULATORY), f"https://www.bseindia.com/XBRLFILES/SHPXBRLDataXML/{r.xbrl_file}",
                     pd.Timestamp(r.available_at).isoformat(), [r.symbol], f"Shareholding pattern {r.period_label} ({r.status})", txt,
                     f"BSE API / as-filed XBRL ({r.source})")
        G.add_document(d)
        pe = str(pd.Timestamp(r.period_end).date()) if pd.notna(r.period_end) else r.period_label
        for m in ("promoter_pct", "pledged_pct_of_promoter", "encumbered_pct_of_promoter", "mf_pct", "fpi_pct", "insurance_pct"):
            v = getattr(r, m)
            if pd.notna(v):
                G.db.execute("INSERT OR IGNORE INTO metrics VALUES (?,?,?,?,?)", (r.symbol, m, pe, float(v), d.doc_id))
    q = shp[shp.kind == "quarterly"].sort_values(["symbol", "available_at"])
    q["d_enc"] = q.groupby("symbol")["encumbered_pct_of_promoter"].diff()
    for r in q[q.d_enc.abs() >= 5].itertuples(index=False):
        G.db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)", (f"P:{r.symbol}:{r.period_end}", r.symbol,
                     "promoter_encumbrance_up" if r.d_enc > 0 else "promoter_encumbrance_down", str(r.available_at)[:10], float(r.d_enc), None))
    # consensus snapshots (tier 6)
    from marketdata.estimates import flatten
    n_est = 0
    for p in sorted(Path("data/estimates/yahoo").glob("*/*.json")):
        fl = flatten(p)
        if not fl:
            continue
        j = json.loads(p.read_text())
        d = Document("Yahoo Finance consensus (yfinance)", int(Tier.SPECIALIST), f"yahoo:{j['vendor_symbol']}:{j['fetched_at']}",
                     j["fetched_at"], [j["symbol"]], f"Consensus snapshot {j['fetched_at'][:10]}",
                     json.dumps(fl, default=str), "marketdata.estimates.snapshot")
        G.add_document(d)
        for x in fl:
            if x["eps_avg"] is not None:
                G.db.execute("INSERT OR IGNORE INTO metrics VALUES (?,?,?,?,?)", (x["symbol"], f"consensus_eps_{x['horizon']}", x["fetched_at"][:10], float(x["eps_avg"]), d.doc_id))
        n_est += 1
    G.commit()
    out.mkdir(parents=True, exist_ok=True)
    P = pd.DataFrame(rows)
    P.to_csv(out / "promises_v08.csv", index=False)
    Q = review_queue(G, TODAY, out / "promise_review_queue.csv")
    res = {"graph": G.stats(), "transcript_documents": n_docs, "consensus_snapshots": n_est,
           "promises": int(len(P)), "promises_companies": int(P.symbol.nunique()) if len(P) else 0,
           "promises_by_status": {k.split(" (")[0] if k.startswith(("PENDING", "UNRESOLVABLE")) else k: int(v)
                                  for k, v in P.status.value_counts().items()} if len(P) else {},
           "promises_by_metric": P.metric.value_counts().to_dict() if len(P) else {},
           "promises_by_tier": P.tier.value_counts().to_dict() if len(P) else {},
           "dropped_past_period_candidates": dropped_past, "review_queue": int(len(Q)), "suggested_in_queue": Q.suggested.value_counts().to_dict() if len(Q) else {},
           "execution_record": execution_record(G),
           "encumbrance_events": int((q.d_enc.abs() >= 5).sum())}
    (out / "company_memory_v08.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "queue", "apply"])
    ap.add_argument("--out", default=f"reports/v0.8_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    ap.add_argument("--csv")
    a = ap.parse_args()
    if a.step == "build":
        build(Path(a.out))
    elif a.step == "queue":
        G = EvidenceGraph(GRAPH)
        print(len(review_queue(G, TODAY, Path(a.out) / "promise_review_queue.csv")), "promises due for review")
    else:
        print(json.dumps(apply_reviews(EvidenceGraph(GRAPH), Path(a.csv)), indent=1))


if __name__ == "__main__":
    main()
