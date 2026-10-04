"""v0.8 data layer: BSE shareholding (version selection, PIT), website transcripts (PIT stamping), consensus
snapshots (flags) and the human-verified promise workflow."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from intel.company_ir import WebTranscript, call_date, cover_letter_links, quarter_end_from_label, stamp, tatasteel_label
from intel.evidence import Document
from intel.graph import EvidenceGraph
from intel.promise_review import apply_reviews, ensure, execution_record, review_queue, validate
from marketdata import bse_shp
from marketdata.estimates import flatten


# ------------------------------------------------------------------------------------------- BSE shareholding
def _prom(auth, pct=62.3, shares=1000.0, pledged=455.1):
    return {"Table": [{"Fld_AuthoriseDate": auth, "Fld_IsPledge": True}],
            "Table1": [{"Fld_Code": "A1", "Fld_TotalNoOfShares": 0},
                       {"Fld_Code": "STA1A2", "Fld_NoOfShareHolders": 9, "Fld_TotalNoOfShares": shares,
                        "Fld_TotalPercentageOf_A_B_C2": pct, "Fld_PledgeEncumberedNoOfShares": pledged}]}


def _pub():
    return {"Table1": [{"Fld_Level": "Mutual Funds/", "Fld_ShareHolderName": None, "Fld_TotalPercentageOf_A_B_C2": 2.86},
                       {"Fld_Level": "Mutual Funds/", "Fld_ShareHolderName": "SBI Equity Hybrid Fund", "Fld_TotalPercentageOf_A_B_C2": 1.24},
                       {"Fld_Level": "Foreign Portfolio Investors", "Fld_ShareHolderName": None, "Fld_TotalPercentageOf_A_B_C2": 21.8},
                       {"Fld_Level": "Any Other (specify)", "Fld_ShareHolderName": "Foreign Institutional Investor", "Fld_NoOfShareHolders": 6,
                        "Fld_TotalPercentageOf_A_B_C2": 0.09},
                       {"Fld_Level": "Sub Total B1", "Fld_ShareHolderName": None, "Fld_TotalPercentageOf_A_B_C2": 34.4}]}


XBRL = """<?xml version="1.0"?><xbrli:xbrl xmlns:in-bse-shp="http://www.bseindia.com/xbrl/shp/2022-09-30/in-bse-shp">
<in-bse-shp:NumberOfShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI" unitRef="s">{ps}</in-bse-shp:NumberOfShares>
<in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">{pp}</in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares>
<in-bse-shp:NumberOfShares contextRef="ShareholdingPattern_ContextI" unitRef="s">2000</in-bse-shp:NumberOfShares>
<in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares contextRef="ShareholdingPattern_ContextI">{tot}</in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares>
<in-bse-shp:PledgedOrEncumberedNumberOfShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">{pl}</in-bse-shp:PledgedOrEncumberedNumberOfShares>
</xbrli:xbrl>"""


def test_period_end_and_kind():
    assert bse_shp.period_end("June 2026") == (pd.Timestamp("2026-06-30"), "quarterly")
    assert bse_shp.period_end("23 Dec 2025") == (pd.Timestamp("2025-12-23"), "event")


def test_promoter_and_public_summary():
    s = bse_shp.promoter_summary(_prom("2019-01-18T10:56:20"))
    assert s["promoter_pct"] == 62.3 and s["pledged_pct_of_promoter"] == pytest.approx(45.51)
    assert s["no_promoter"] is False
    none = bse_shp.promoter_summary({"Table": [{}], "Table1": [{"Fld_Code": "A1", "Fld_TotalNoOfShares": 0}]})
    assert none["no_promoter"] is True and none["promoter_pct"] == 0.0
    p = bse_shp.public_summary(_pub())
    assert p["mf_pct"] == pytest.approx(2.86)                    # named fund rows are not double counted
    assert p["fpi_pct"] == pytest.approx(21.89)                  # FPI + 'Foreign Institutional Investor' category row
    assert p["public_sub_total_b1_pct"] == pytest.approx(34.4)


def test_xbrl_summary_scales_fractions_and_handles_no_promoter():
    a = bse_shp.xbrl_summary(XBRL.format(ps=1000, pp=0.5, tot=1, pl=250))
    assert a["promoter_pct"] == pytest.approx(50.0) and a["pledged_pct_of_promoter"] == pytest.approx(25.0)
    b = bse_shp.xbrl_summary(XBRL.format(ps=1000, pp=50, tot=100, pl=0))
    assert b["promoter_pct"] == pytest.approx(50.0)
    c = bse_shp.xbrl_summary(XBRL.replace("ShareholdingOfPromoterAndPromoterGroup_ContextI", "Other_ContextI").format(ps=1, pp=1, tot=100, pl=0))
    assert c["no_promoter"] is True and c["promoter_pct"] == 0.0
    assert pd.isna(a["mf_pct"])                                   # institutional mix is never read from XBRL


def _write_case(tmp_path, rows, prom_auth, xbrl=None):
    d = tmp_path / "shp" / "532921"
    (d / "xbrl").mkdir(parents=True)
    (d / "index.json").write_text(json.dumps({"rows": rows}))
    (d / "100.00_promoter.json").write_text(json.dumps(_prom(prom_auth)))
    (d / "100.00_public.json").write_text(json.dumps(_pub()))
    if xbrl:
        (d / "xbrl" / "F1.xml.gz").write_bytes(gzip.compress(xbrl.encode()))
    return bse_shp.load(tmp_path, {"532921": "ADANIPORTS"})


ROW = {"qtrid": 100.0, "qtr": "December 2018", "status": "New", "filing_date_time": "2019-01-18T10:56:20", "revised_date_time": None,
       "XbrlFile": "F1.xml", "revised_reson": ""}


def test_structured_record_used_only_when_it_is_this_filings_version(tmp_path):
    df = _write_case(tmp_path, [ROW], "2019-01-18T10:56:21")
    assert df.loc[0, "source"] == "bse_api" and str(df.loc[0, "available_at"]).startswith("2019-01-18 10:56:21")   # max(filing, auth)


def test_later_version_is_replaced_by_as_filed_xbrl(tmp_path):
    df = _write_case(tmp_path, [ROW], "2019-03-01T09:00:00", XBRL.format(ps=1000, pp=60, tot=100, pl=0))
    assert df.loc[0, "source"] == "bse_xbrl_as_filed" and df.loc[0, "promoter_pct"] == pytest.approx(60.0)
    assert str(df.loc[0, "available_at"]).startswith("2019-01-18")


def test_later_version_without_xbrl_is_never_backdated(tmp_path):
    df = _write_case(tmp_path, [ROW], "2019-03-01T09:00:00")
    assert df.loc[0, "source"] == "bse_api_later_version"
    assert df.loc[0, "available_at"] == pd.Timestamp("2019-03-01T09:00:00", tz="Asia/Kolkata")


def test_as_of_is_strict_and_quarterly_only(tmp_path):
    df = _write_case(tmp_path, [ROW], "2019-01-18T10:56:20")
    assert bse_shp.as_of(df, pd.Timestamp("2019-01-18 10:56:20", tz="Asia/Kolkata")).empty
    assert len(bse_shp.as_of(df, pd.Timestamp("2019-01-18 10:57:00"))) == 1
    ev = df.copy()
    ev["kind"], ev["available_at"] = "event", pd.Timestamp("2019-01-10", tz="Asia/Kolkata")
    assert bse_shp.as_of(ev, pd.Timestamp("2019-01-15")).empty         # event-dated filings are never 'the quarter'


def _frame(rows):
    df = pd.DataFrame(rows)
    df["available_at"] = pd.to_datetime(df["available_at"]).dt.tz_localize("Asia/Kolkata")
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["kind"], df["qtrid"] = "quarterly", 100.0
    return df


def test_as_of_late_revision_of_old_quarter_does_not_displace_newer_quarter():
    df = _frame([{"symbol": "H", "period_end": "2024-12-31", "available_at": "2025-01-20", "promoter_pct": 34.6},
                 {"symbol": "H", "period_end": "2025-09-30", "available_at": "2025-10-20", "promoter_pct": 34.7},
                 {"symbol": "H", "period_end": "2024-12-31", "available_at": "2025-10-30", "promoter_pct": 34.5}])
    x = bse_shp.as_of(df, pd.Timestamp("2025-11-01"))
    assert x.loc["H", "period_end"] == pd.Timestamp("2025-09-30")
    y = _frame([{"symbol": "H", "period_end": "2025-09-30", "available_at": "2025-10-20", "promoter_pct": 34.7},
                {"symbol": "H", "period_end": "2025-09-30", "available_at": "2025-10-25", "promoter_pct": 34.9}])
    assert bse_shp.as_of(y, pd.Timestamp("2025-10-26")).loc["H", "promoter_pct"] == 34.9     # latest public version
    assert bse_shp.as_of(y, pd.Timestamp("2025-10-21")).loc["H", "promoter_pct"] == 34.7


def test_qtrid_collision_gives_structured_record_only_to_nearest_filing(tmp_path):
    a = dict(ROW, filing_date_time="2019-01-18T10:56:20", XbrlFile="F1.xml")
    b = dict(ROW, filing_date_time="2019-01-18T10:56:50", XbrlFile="F2.xml", status="Revised")
    df = _write_case(tmp_path, [a, b], "2019-01-18T10:56:51", XBRL.format(ps=1000, pp=61, tot=100, pl=0))
    by = df.set_index("xbrl_file")
    assert by.loc["F2.xml", "source"] == "bse_api"
    assert by.loc["F2.xml", "available_at"] == pd.Timestamp("2019-01-18T10:56:51", tz="Asia/Kolkata")    # max(filing, authorisation)
    assert by.loc["F1.xml", "source"] == "bse_xbrl_as_filed" and by.loc["F1.xml", "promoter_pct"] == pytest.approx(61.0)
    assert {r["XbrlFile"] for r in bse_shp.needs_xbrl(tmp_path, "532921", [a, b])} == {"F1.xml", "F2.xml"}


def test_xbrl_2025_taxonomy_encumbrance_and_unknown_flag():
    t = XBRL.format(ps=1000, pp=0.5, tot=1, pl=0).replace(
        '<in-bse-shp:PledgedOrEncumberedNumberOfShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">0</in-bse-shp:PledgedOrEncumberedNumberOfShares>',
        '<in-bse-shp:NumberOfSharesEncumbered contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">508</in-bse-shp:NumberOfSharesEncumbered>'
        '<in-bse-shp:NumberOfSharesEncumberedUnderPledged contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">500</in-bse-shp:NumberOfSharesEncumberedUnderPledged>')
    x = bse_shp.xbrl_summary(t)
    assert x["encumbered_shares"] == 508 and x["pledged_shares"] == 500 and x["encumbered_pct_of_promoter"] == pytest.approx(50.8)
    u = XBRL.format(ps=1000, pp=50, tot=100, pl=0).replace(
        '<in-bse-shp:PledgedOrEncumberedNumberOfShares contextRef="ShareholdingOfPromoterAndPromoterGroup_ContextI">0</in-bse-shp:PledgedOrEncumberedNumberOfShares>',
        '<in-bse-shp:WhetherAnySharesHeldByPromotersArePledgeOrOtherwiseEncumbered contextRef="MainI">true</in-bse-shp:WhetherAnySharesHeldByPromotersArePledgeOrOtherwiseEncumbered>')
    assert pd.isna(bse_shp.xbrl_summary(u)["encumbered_shares"])          # declared encumbered, figure unreadable -> unknown, not 0


def test_structured_percentages_x100_are_rescaled():
    s = bse_shp.promoter_summary(_prom("2025-07-01T10:00:00", pct=5504.0))
    assert s["promoter_pct"] == pytest.approx(55.04) and s["pct_rescaled"] is True
    pub = _pub()
    for r in pub["Table1"]:
        r["Fld_TotalPercentageOf_A_B_C2"] *= 100
    assert bse_shp.public_summary(pub)["mf_pct"] == pytest.approx(2.86)


# ------------------------------------------------------------------------------------------- website transcripts
def test_collector_label_keeps_four_digit_years():
    assert tatasteel_label("/media/1/2qfy2025-transcript-v7.pdf") == "2QFY2025"
    assert quarter_end_from_label(tatasteel_label("/media/1/2qfy2025-transcript-v7.pdf")) == pd.Timestamp("2024-09-30")
    assert tatasteel_label("/media/1/4qfy26-earnings-call-transcript.pdf") == "4QFY26"


def test_quarter_labels():
    assert quarter_end_from_label("4QFY26") == pd.Timestamp("2026-03-31")
    assert quarter_end_from_label("Q2, FY2025-26") == pd.Timestamp("2025-09-30")
    assert quarter_end_from_label("2qfy2025-transcript-v7") == pd.Timestamp("2024-09-30")
    assert quarter_end_from_label("1qfy27-transcript-v7") == pd.Timestamp("2026-06-30")
    assert quarter_end_from_label("annual-report") is None


def test_call_date_skips_period_ends():
    t = "Results for the quarter ended March 31, 2026\nEarnings conference call, April 22, 2026"
    assert call_date(t) == pd.Timestamp("2026-04-22")


def _w(tmp_path, text, **kw):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"%PDF")
    p.with_suffix(".txt").write_text(text)
    return WebTranscript("LT", "https://x/y.pdf", "Earnings Call Transcript Q1FY27", path=str(p), **kw)


FULL = "Earnings call, July 28, 2026\n" + "\n".join(f"Speaker{i % 3}: line {i} " + "x" * 300 for i in range(40))


def test_stamp_uses_exchange_intimation_never_call_date(tmp_path):
    T = {"desc": "Transcript of Analysts/Institutional Investor Meet/Con. Call", "attchmntText": "Sub.: Transcript of the earnings call"}
    idx = [dict(T, exchdisstime="05-Aug-2026 18:00:00", attchmntFile="a.pdf"), dict(T, exchdisstime="01-Jan-2026 10:00:00", attchmntFile="old.pdf")]
    w = stamp(_w(tmp_path, FULL), idx)
    assert w.available_at.startswith("2026-08-05T18:00:00") and w.intimation_url == "a.pdf"
    w2 = stamp(_w(tmp_path, FULL), [dict(T, exchdisstime="30-Sep-2026 10:00:00", attchmntFile="late.pdf")])
    assert w2.available_at == ""                                   # outside the window: first-seen only


def test_stamp_ignores_recording_notices_and_other_quarters(tmp_path):
    audio = {"exchdisstime": "28-Jul-2026 21:56:50", "attchmntFile": "audio.pdf", "desc": "Recording of Analysts/Institutional Investor Meet/Con. Call",
             "attchmntText": "Sub.: Audio recording of Q1 FY27 Earnings Call. See https://x/transcripts.aspx"}
    other_q = {"exchdisstime": "29-Jul-2026 10:00:00", "attchmntFile": "q4.pdf", "desc": "Transcript of Analysts/Institutional Investor Meet/Con. Call",
               "attchmntText": "Sub.: Transcript of Q4 FY26 Earnings Call"}
    real = {"exchdisstime": "04-Aug-2026 11:23:19", "attchmntFile": "q1.pdf", "desc": "Transcript of Analysts/Institutional Investor Meet/Con. Call",
            "attchmntText": "Sub.: Transcript of Q1 FY27 Earnings Call"}
    w = stamp(_w(tmp_path, FULL), [audio, other_q, real])
    assert w.intimation_url == "q1.pdf" and w.available_at.startswith("2026-08-04T11:23:19")


def test_cover_letter_links_rejoin_and_filter(tmp_path):
    text = ("transcript is available at https://www.axis.bank.in/docs/call-transcript/q3fy26-earnings-call-\n"
            "transcript.pdf and the AGM at https://x.com/AGM_Transcript_2024.pdf\n"
            "https://ind01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fa.com%2Fq1-earnings-call-transcript.pdf&data=1")
    links = cover_letter_links(tmp_path / "missing.pdf", text)
    assert "https://www.axis.bank.in/docs/call-transcript/q3fy26-earnings-call-transcript.pdf" in links
    assert "https://a.com/q1-earnings-call-transcript.pdf" in links
    assert not any("AGM" in u for u in links)


# ------------------------------------------------------------------------------------------- consensus snapshots
def test_estimate_flags(tmp_path):
    j = {"symbol": "RELIANCE", "vendor_symbol": "RELIANCE.NS", "fetched_at": "2026-10-03T18:00:00Z", "tier": 6, "errors": {},
         "tables": {"earnings_estimate": [{"period": "0y", "avg": 63.9, "numberOfAnalysts": 27, "currency": "INR"},
                                          {"period": "0q", "avg": 16.3, "numberOfAnalysts": 2, "currency": "INR"}],
                    "eps_trend": [{"period": "0y", "current": 33.3, "30daysAgo": 25.8}, {"period": "0q", "current": 16.3}]}}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(j))
    f = {r["horizon"]: r for r in flatten(p)}
    assert "trend_vs_estimate_mismatch" in f["0y"]["flags"] and "few_analysts" in f["0q"]["flags"]
    assert f["0y"]["vendor_lookback_30d"] == 25.8


# ------------------------------------------------------------------------------------------- promise review
@pytest.fixture
def graph(tmp_path):
    G = EvidenceGraph(tmp_path / "g.sqlite")
    ensure(G)
    said = Document("NSE", 1, "u1", "2024-05-01T10:00:00+05:30", ["UC"], "t1", "We will reach 200 mtpa by FY25.", "m")
    later = Document("NSE", 1, "u2", "2025-05-01T10:00:00+05:30", ["UC"], "t2", "Our capacity has now reached 201 mtpa as of March.", "m")
    for d, name in ((said, "a.txt"), (later, "b.txt")):
        G.add_document(d)
        (tmp_path / name).write_text(d.text)
        G.db.execute("INSERT INTO doc_files VALUES (?,?)", (d.doc_id, str(tmp_path / name)))
    cid = G.add_claim("promise", "UC", "We will reach 200 mtpa by FY25.", said.published_at, [said.doc_id], quote="We will reach 200 mtpa by FY25.",
                      speaker="CFO", due="2025-03-31", metric="capacity", target="200 mtpa", status="DUE: candidate outcome passages found (human review)")
    G.commit()
    return G, cid, said, later, tmp_path


def _row(cid, **kw):
    r = {"claim_id": cid, "verdict": "MET", "evidence_doc_id": "", "evidence_quote": "", "reviewer": "RK", "note": "", "suggested": ""}
    r.update(kw)
    return r


def test_review_requires_later_document_and_verbatim_quote(graph):
    G, cid, said, later, _ = graph
    assert validate(G, _row(cid, evidence_doc_id=later.doc_id, evidence_quote="capacity has now reached 201 mtpa")) == []
    assert any("verbatim" in e for e in validate(G, _row(cid, evidence_doc_id=later.doc_id, evidence_quote="capacity reached 250 mtpa easily")))
    assert any("later" in e for e in validate(G, _row(cid, evidence_doc_id=said.doc_id, evidence_quote="We will reach 200 mtpa by FY25")))
    assert any("reviewer" in e for e in validate(G, _row(cid, reviewer="", evidence_doc_id=later.doc_id, evidence_quote="capacity has now reached 201 mtpa")))
    assert any("verdict" in e for e in validate(G, _row(cid, verdict="PROBABLY")))
    assert validate(G, _row(cid, verdict="UNRESOLVABLE")) == []


def test_queue_apply_and_record(graph):
    G, cid, said, later, tmp = graph
    q = review_queue(G, pd.Timestamp("2026-10-03"), tmp / "q.csv")
    assert list(q.claim_id) == [cid]
    q.loc[0, ["verdict", "evidence_doc_id", "evidence_quote", "reviewer"]] = ["MET", later.doc_id, "capacity has now reached 201 mtpa", "RK"]
    q.to_csv(tmp / "q.csv", index=False)
    res = apply_reviews(G, tmp / "q.csv")
    assert res["applied"] == 1 and not res["rejected"]
    assert G.db.execute("SELECT status FROM claims WHERE claim_id=?", (cid,)).fetchone()[0].startswith("VERIFIED MET")
    rec = execution_record(G)
    assert rec[0]["MET"] == 1 and rec[0]["status"] == "insufficient"
    assert review_queue(G, pd.Timestamp("2026-10-03"), tmp / "q2.csv").empty     # reviewed promises leave the queue


def test_rebuild_does_not_erase_verified_status(graph):
    G, cid, said, later, tmp = graph
    G.set_status(cid, "VERIFIED MET (review R1)")
    G.add_claim("promise", "UC", "We will reach 200 mtpa by FY25.", said.published_at, [said.doc_id], quote="We will reach 200 mtpa by FY25.",
                speaker="CFO", due="2025-03-31", metric="capacity", target="200 mtpa", status="DUE: machine status")
    assert G.db.execute("SELECT status FROM claims WHERE claim_id=?", (cid,)).fetchone()[0].startswith("VERIFIED MET")
