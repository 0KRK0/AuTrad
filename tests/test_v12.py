import numpy as np
import pandas as pd
import pytest

from lab import aem
from lab.execution_rules import RULES, roll_advice
from lab.registry import Registry
from marketdata.index_changes import index_key, parse_text
from marketdata.nse_events import parse_ban, parse_dividend


def test_registry_requires_six_questions_from_v12(tmp_path):
    r = Registry(tmp_path / "t.jsonl")
    base = {"kind": "experiment", "version": "v1.2", "experiment_id": "x", "economic_mechanism": "m"}
    with pytest.raises(ValueError):
        r.add([base])
    r.add([{**base, "economic_counterparty": "c", "mechanism_constraint": "k", "our_advantage": "a", "failure_condition": "f"}])
    r.add([{"kind": "experiment", "version": "v1.1", "experiment_id": "y", "economic_mechanism": "m"}])


def test_allocator_drawdown_margin_and_fill_constraints():
    v = aem.with_overrides(aem.venues()["nse_index_fut"], tax_rate=0.0, taker_bp=0.0, half_spread_bp=0.0)
    o1 = aem.Opportunity("risky", "m", 50, 400, 5, v, cash_rate=0.0)                      # sigma 4 %
    o2 = aem.Opportunity("margin_hungry", "m", 30, 10, 5, v, cash_rate=0.0, margin_per_notional=0.5)
    o3 = aem.Opportunity("unlikely_fill", "m", 80, 10, 5, v, cash_rate=0.0, p_fill=0.1)
    r = aem.allocate([o1, o2, o3], capital=1e7, max_share=1.0, group_share=1.0, drawdown_budget=0.02, margin_available=1e6, min_fill_prob=0.5)
    a = {x["name"]: x for x in r["allocations"]}
    assert a["risky"]["binding"] == "drawdown" and a["risky"]["capital"] == pytest.approx(0.02 * 1e7 / 0.08)
    assert a["margin_hungry"]["binding"] == "margin" and a["margin_hungry"]["capital"] == pytest.approx(2e6)
    assert a["unlikely_fill"]["capital"] == 0


def test_roll_rule():
    assert roll_advice(5)["expected_extra_cost_bp"] == 0 and roll_advice(2)["expected_extra_cost_bp"] == RULES["stock_futures_roll_window"].effect_bp


def test_dividend_and_ban_parsers():
    p = parse_dividend("declared interim dividend at the rate of Rs. 5.29 (Rupees Five and Twenty Nine Paise only) per equity share")
    assert p["amounts"] == [5.29] and p["type"] == "interim" and p["action"] == "declared"
    assert parse_dividend("did not recommend any final dividend")["negative"]
    assert parse_ban(b"Securities in Ban For Trade Date 30-SEP-2026:\n1,SAIL\n2,IDEA\n", None) == ["SAIL", "IDEA"]
    assert parse_ban(b"Securities in Ban For Trade Date 01-MAR-2016:NIL\n", None) == []


def test_press_release_parser_and_index_keys():
    txt = """These changes shall become effective from March 28, 2025 (close of March 27, 2025).
     a) Nifty 50

     The following companies are being excluded:

         Sr. No.                  Company Name                                                Symbol
           1     Bharat Petroleum Corporation Ltd.                                    BPCL

     The following companies are being included:

         Sr. No.                  Company Name                                             Symbol
           1     Zomato Ltd.                                                         ZOMATO
    Nifty 50 index
           1     Not A Row Ltd.                                                      NOTAROW
     d) Nifty Next 50

     The following companies are being excluded:
       1       Zomato Ltd.                                                        ZOMATO
"""
    eff, rows = parse_text(txt)
    assert eff == pd.Timestamp("2025-03-28")
    got = {(r["index_key"], r["action"], r["symbol"]) for r in rows}
    assert ("NIFTY50", "exclude", "BPCL") in got and ("NIFTY50", "include", "ZOMATO") in got and ("NIFTYNEXT50", "exclude", "ZOMATO") in got
    assert ("NIFTY50", "include", "NOTAROW") in got          # a wrapped note line is not a header: rows stay in the current section
    assert index_key("CNX Nifty") == "NIFTY50" and index_key("CNX Nifty Junior") == "NIFTYNEXT50"


def test_t0_rule():
    from scripts.v12_events import first_close_after
    s = first_close_after(pd.Timestamp("2023-10-12 16:04:50", tz="Asia/Kolkata"))
    assert s == pd.Timestamp("2023-10-13")
    assert first_close_after(pd.Timestamp("2023-10-12 11:00:00", tz="Asia/Kolkata")) == pd.Timestamp("2023-10-12")
