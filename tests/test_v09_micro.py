"""v0.9 M0: event-level OFI, the queue-aware passive fill simulator, greedy non-overlap, NSE recorder -> grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketdata.micro_binance import ofi_events
from marketdata.nse_depth_recorder import DepthRecorder, flatten, nse_index_futures_round_trip_bp, to_grid
from scripts.v09_analysis import _greedy, simulate


def test_ofi_matches_definition():
    bp = np.array([100.0, 100.0, 100.1, 100.0])
    bq = np.array([5.0, 7.0, 2.0, 4.0])
    ap = np.array([100.2, 100.2, 100.2, 100.1])
    aq = np.array([3.0, 3.0, 1.0, 6.0])
    e = ofi_events(bp, bq, ap, aq, (100.0, 5.0, 100.2, 3.0))
    # n=0: no change -> 5-5-3+3=0; n=1: bid size +2 -> +2; n=2: bid up (+2), ask size 3->1 at same px -> -(1)+3=+2 => 4; n=3: bid down -> -2, ask down -> -6
    assert e.tolist() == pytest.approx([0.0, 2.0, 4.0, -8.0])


def test_greedy_one_position_at_a_time():
    start = np.array([0, 2, 5, 6, 10])
    end = np.array([5, 7, 9, 12, 15])
    assert _greedy(start, end).tolist() == [0, 2, 4]


def _grid(n=200, bid=100.0, ask=100.1, bq=1.0, aq=1.0):
    return pd.DataFrame({"bid": np.full(n, bid), "ask": np.full(n, ask), "bidq": np.full(n, bq), "askq": np.full(n, aq),
                         "sell_min_px": np.full(n, np.nan), "sell_vol_at_min": np.zeros(n), "buy_max_px": np.full(n, np.nan), "buy_vol_at_max": np.zeros(n)})


def test_passive_fill_needs_queue_ahead_to_trade_through():
    d = _grid()
    pred = np.zeros(len(d))
    pred[10] = 1.0                                              # one buy signal
    model = {"pred_deciles": [-0.5, 0.5]}
    d.loc[12, ["sell_min_px", "sell_vol_at_min"]] = [100.0, 0.5]   # less than queue ahead (1.0) + size
    r = simulate(d, pred, model, 0, (0.0, 0.0))
    assert r["E2"]["orders"] == 1 and r["E2"]["trades"] == 0          # not filled
    d.loc[13, ["sell_min_px", "sell_vol_at_min"]] = [100.0, 0.6]   # cumulative 1.1 >= 1.0 + 0.01
    r = simulate(d, pred, model, 0, (0.0, 0.0))
    assert r["E2"]["trades"] == 1 and r["E2"]["mean_net_bp"] == pytest.approx(0.0)   # filled at the bid, exits at the same bid


def test_taker_pays_spread_and_fees():
    d = _grid()
    pred = np.zeros(len(d))
    pred[10] = 100.0
    r = simulate(d, pred, {"pred_deciles": [-1e9, 1e9]}, 0, (2.0, 5.0))
    spread_bp = np.log(100.1 / 100.0) * 1e4
    assert r["E1"]["trades"] == 1 and r["E1"]["mean_net_bp"] == pytest.approx(-spread_bp - 10.0)


def test_recorder_flatten_and_grid(tmp_path):
    import datetime as dt
    day = "2026-10-05"
    t0_local = int(pd.Timestamp("2026-10-05 09:15", tz="Asia/Kolkata").value // 10**6)
    rows = []
    for k, (bpx, apx, vol, lp) in enumerate([(25000.0, 25000.5, 100, 25000.5), (25000.0, 25000.5, 175, 25000.5), (25000.5, 25001.0, 250, 25000.0)]):
        t = {"exchange_timestamp": dt.datetime(2026, 10, 5, 9, 15, k),        # kiteconnect: naive IST, whole seconds
             "last_price": lp, "last_traded_quantity": 75, "volume_traded": vol,
             "depth": {"buy": [{"price": bpx, "quantity": 300, "orders": 3}], "sell": [{"price": apx, "quantity": 150, "orders": 2}]}}
        r = flatten(t, "NIFTY-FUT")
        r["ts_local"] = t0_local + 1000 * k + 120                              # received 120 ms later
        rows.append(r)
    assert rows[0]["ts_exchange"] == t0_local                                   # localised as IST, not read as UTC
    rec = DepthRecorder(tmp_path, flush_every_s=0)
    rec.on_rows(rows)
    assert rec.drain(force=True) == 3
    df = pd.concat([pd.read_parquet(p) for p in (tmp_path / "NIFTY-FUT" / day).glob("*.parquet")])
    g = to_grid(df, day)
    assert len(g) == int(6.25 * 3600)                                           # 1 s grid
    assert g["buy_vol"].iloc[1] == 75 and g["sell_vol"].iloc[2] == 75          # quote rule: at the ask = buy, below mid = sell
    assert g["bid"].iloc[2] == 25000.5


def test_nse_futures_cost_is_a_few_bp():
    assert 5.0 < nse_index_futures_round_trip_bp() < 7.0               # STT 0.05 % since Apr 2026
    assert 2.0 < nse_index_futures_round_trip_bp(stt_rate=0.0002, lot=75) < 4.0
