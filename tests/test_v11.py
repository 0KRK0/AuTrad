import math

import numpy as np
import pandas as pd
import pytest

from lab import aem
from marketdata import nse_fo


def _nse_fut():
    return aem.venues()["nse_index_fut"]


def test_price_requires_mechanism():
    with pytest.raises(ValueError):
        aem.price(aem.Opportunity("x", " ", 10, 5, 1, _nse_fut()))


def test_no_fill_cost_and_capital_enter_ev():
    v = aem.with_overrides(_nse_fut(), tax_rate=0.0)
    a = aem.price(aem.Opportunity("a", "m", 50, 10, 2, v, cash_rate=0.0))
    b = aem.price(aem.Opportunity("b", "m", 50, 10, 2, v, p_fill=0.5, no_fill_cost_bp=10, cash_rate=0.0))
    assert math.isclose(b["ev_bp"], 0.5 * a["net_if_filled_bp"] - 5)
    c = aem.price(aem.Opportunity("c", "m", 50, 10, 2, v, capital_per_notional=2.0, cash_rate=0.0))
    assert math.isclose(c["excess_bp_per_capital_day"], a["excess_bp_per_capital_day"] / 2)


def test_allocator_ranks_by_capital_day_and_respects_capacity_and_groups():
    v = aem.with_overrides(_nse_fut(), tax_rate=0.0, taker_bp=0.0, half_spread_bp=0.0)
    ops = [aem.Opportunity("short_rich", "forced flow", 30, 5, 1, v, capacity_inr=1e6, risk_group="g", cash_rate=0.0),
           aem.Opportunity("slow_big", "information diffusion", 300, 50, 60, v, capacity_inr=1e9, cash_rate=0.0),
           aem.Opportunity("same_group", "forced flow", 25, 5, 1, v, capacity_inr=1e9, risk_group="g", cash_rate=0.0),
           aem.Opportunity("loser", "none really", -5, 5, 1, v, cash_rate=0.0)]
    r = aem.allocate(ops, capital=1e7, max_share=0.5, group_share=0.4)
    a = {x["name"]: x for x in r["allocations"]}
    assert a["short_rich"]["capital"] == 1e6                      # capacity binds
    assert a["same_group"]["capital"] == pytest.approx(3e6)       # group cap 40 % minus the 1e6 already used
    assert a["slow_big"]["capital"] == pytest.approx(5e6)         # max_share binds
    assert a["loser"]["capital"] == 0 and r["cash"] == pytest.approx(1e6)
    order = [x["name"] for x in r["allocations"]]
    assert order.index("short_rich") < order.index("slow_big")    # 30 bp/day beats 5 bp/day


def test_cash_and_carry_at_measured_v11_numbers_is_abandoned():
    # HB1 measured: locked 65 bp over 27 days, round-trip costs 37 bp, capital 1.2 x notional, cash 5.8 %
    v = aem.with_overrides(_nse_fut(), taker_bp=0.0, half_spread_bp=16.7, tax_rate=0.312)   # 33 bp round trip
    r = aem.price(aem.Opportunity("cash_carry", "financing demand of futures longs", 65, 5, 27, v, capital_per_notional=1.2, cash_rate=0.058))
    assert r["decision"] == "ABANDON" and r["ann_excess_on_capital"] < 0


def test_parsers_on_synthetic_files():
    raw = (b"INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,\n"
           b"FUTIDX,NIFTY,31-Mar-2016,0,XX,1,2,0.5,1.5,1.5,10,2.5,100,1,01-MAR-2016,\n"
           b"OPTIDX,NIFTY,31-Mar-2016,7000,CE,1,2,0.5,1.5,1.5,10,2.5,100,1,01-MAR-2016,\n")
    d = nse_fo.parse_legacy(raw, pd.Timestamp("2016-03-01").date())
    assert len(d) == 1 and d.iloc[0]["kind"] == "IDX" and d.iloc[0]["expiry"] == "2016-03-31" and d.iloc[0]["value"] == 250000
    idx = b"Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield\nCNX Nifty,01-03-2016,1,2,0.5,1.5,0,0,1,1,20,3,1.5\n"
    i = nse_fo.parse_index(idx, pd.Timestamp("2016-03-01").date())
    assert i.iloc[0]["index"] == "CNX NIFTY" and nse_fo.RENAME["CNX NIFTY"] == "NIFTY 50"


def test_costs_and_tax_rules():
    from scripts import v11_basis as B
    assert B.fut_stt("2023-03-31") == 1.0 and B.fut_stt("2024-10-01") == 2.0 and B.fut_stt("2026-04-01") == 5.0
    c = B.costs_bp(pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-30"), 8e5, "STK", 1.0, price=5.0)
    assert c["cash"] > 50 + 20                                       # half-tick floor on entry: 0.025 / 5 = 50 bp
    c0 = B.costs_bp(pd.Timestamp("2017-01-02"), pd.Timestamp("2017-01-25"), 8e5, "STK", 1.0, price=5.0)
    assert c0["cash"] > 2 * 50 + 20                                  # before physical settlement the exit pays it too
    t = pd.DataFrame({"expiry": pd.to_datetime(["2024-06-27", "2024-07-25"]), "w": [1.0, 1.0],
                      "cash_leg": [0.05, 0.0], "fut_leg": [-0.03, 0.0], "cost_cash": [0.0, 0.0], "cost_fut": [0.0, 0.0]})
    tx = B.after_tax_excess(t, None)
    C, Bz = 0.05 / 1.2, -0.03 / 1.2
    assert tx[2024]["tax"] == pytest.approx(B.cg_rate("2024-06-27") * (C + Bz))   # business loss set off; STCG 15.6 % before 23 Jul 2024
    t2 = t.assign(cash_leg=[-0.05, 0.0], fut_leg=[0.03, 0.0])
    assert B.after_tax_excess(t2, None)[2024]["tax"] == pytest.approx(B.TAX_BUS * 0.03 / 1.2)   # capital loss not set off


def test_registry_requires_mechanism_from_v11(tmp_path):
    from lab.registry import Registry
    r = Registry(tmp_path / "t.jsonl")
    with pytest.raises(ValueError):
        r.add([{"kind": "experiment", "version": "v1.1", "experiment_id": "x"}])
    r.add([{"kind": "experiment", "version": "v1.0", "experiment_id": "old"}])
    r.add([{"kind": "experiment", "version": "v1.1", "experiment_id": "y", "economic_mechanism": "forced flow"}])
    with pytest.raises(ValueError):
        r.add([{"kind": "experiment", "experiment_id": "no_version"}])


def test_expiry_normalisation_merges_relabelled_contracts():
    from scripts.v11_basis import normalise_expiry
    f = pd.DataFrame({"kind": ["IDX"] * 5, "symbol": ["NIFTY"] * 5,
                      "date": pd.to_datetime(["2023-06-26", "2023-06-27", "2023-06-28", "2023-06-28", "2023-07-27"]),
                      "expiry": pd.to_datetime(["2023-06-29", "2023-06-29", "2023-06-28", "2023-07-27", "2023-07-27"])})
    g = normalise_expiry(f.copy())
    assert set(g["expiry"].iloc[:3]) == {pd.Timestamp("2023-06-28")} and g["expiry"].iloc[3] == pd.Timestamp("2023-07-27")


def test_depth_day_seal_and_qc(tmp_path):
    from marketdata import nse_depth_recorder as R
    day = "2026-10-05"
    t0 = int(pd.Timestamp(f"{day} 09:15", tz="Asia/Kolkata").value // 10**6)
    n = 3000
    rows = {c: np.zeros(n) for c in R.TICK_COLS}
    rows["ts_local"] = t0 + np.arange(n) * 7500           # every 7.5 s through the session
    rows["ts_exchange"] = rows["ts_local"] - 40
    rows["symbol"] = ["NIFTY-FUT"] * n
    rows["bid1_px"], rows["ask1_px"] = np.full(n, 100.0), np.full(n, 100.05)
    rows["volume_cum"] = np.arange(n, dtype=float)
    d = tmp_path / "NIFTY-FUT" / day
    d.mkdir(parents=True)
    pd.DataFrame(rows)[R.TICK_COLS].to_parquet(d / "part-1-0.parquet")
    r = R.seal_day("NIFTY-FUT", day, tmp_path)
    assert r["status"] == "sealed" and r["qc"]["ok"] and r["qc"]["latency_ms_median"] == 40
    assert (d / "manifest.json").exists() and (d / "day.parquet").exists()
    assert R.seal_day("NIFTY-FUT", day, tmp_path)["status"] == "already sealed"
