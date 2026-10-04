import numpy as np
import pandas as pd

from scripts import v10_micro as V


def _grid(n=1000, q=1.0):
    return pd.DataFrame({"bid": np.full(n, 100.0), "ask": np.full(n, 100.1), "bidq": np.full(n, q), "askq": np.full(n, q),
                         "sell_min_px": np.full(n, np.nan), "sell_vol_at_min": np.zeros(n),
                         "buy_max_px": np.full(n, np.nan), "buy_vol_at_max": np.zeros(n)})


def test_m1_policies_prices_and_fills():
    d = _grid()
    d.loc[50, ["sell_min_px", "sell_vol_at_min"]] = [100.0, 2.0]        # enough volume at the bid to clear Q + size
    o = V.m1_orders(d, np.zeros(len(d), np.float32))
    buy, sell = o[o.side == 1].iloc[0], o[o.side == -1].iloc[0]
    assert buy["maker1"] and buy["p1"] == 100.0 and buy["p0"] == 100.1
    assert not sell["maker1"] and sell["p1"] == 100.0                  # unfilled: crosses at the bid after 60 s
    assert buy["p2"] == buy["p1"] and not buy["triggered"]
    x = V.is_bp(o, (2.0, 5.0))
    b = x[x.side == 1].iloc[0]
    assert abs(b["IS0"] - (np.log(100.1 / 100.05) * 1e4 + 5)) < 1e-9
    assert abs(b["IS1"] - (np.log(100.0 / 100.05) * 1e4 + 2)) < 1e-9


def test_m1_queue_not_cleared_means_no_fill():
    d = _grid()
    d.loc[50, ["sell_min_px", "sell_vol_at_min"]] = [100.0, 1.0]        # 1.0 < Q + size = 1.01
    o = V.m1_orders(d, np.zeros(len(d), np.float32))
    assert not o[o.side == 1].iloc[0]["maker1"]


def test_m1_p2_crosses_on_adverse_prediction():
    d = _grid()
    o = V.m1_orders(d, np.ones(len(d), np.float32))                       # predicts +1 bp: adverse for a buy
    buy, sell = o[o.side == 1].iloc[0], o[o.side == -1].iloc[0]
    assert buy["triggered"] and buy["trig_at_decision"] and buy["p2"] == buy["p0"] and not buy["maker2"]
    assert not sell["triggered"]


def test_mm_round_trips_capture_spread_and_inventory_bounded():
    n = 2000
    d = _grid(n, q=0.001)
    for k, t in enumerate(range(10, n - 50, 20)):
        if k % 2 == 0:
            d.loc[t, ["sell_min_px", "sell_vol_at_min"]] = [100.0, 1.0]
        else:
            d.loc[t, ["buy_max_px", "buy_vol_at_max"]] = [100.1, 1.0]
    r = V.mm_day(d, np.zeros(n, np.float32), -0.33, 0.33, skew=False)
    assert r["fills"] > 50 and r["flattens"] == 0 and abs(r["end_inventory"]) <= 1
    pnl = sum(r["pnl_h"])
    assert abs(pnl - 0.1 * (r["fills"] // 2)) < 0.1


def test_mm_flattens_stale_inventory_as_taker():
    n = 2000
    d = _grid(n, q=0.001)
    d.loc[10, ["sell_min_px", "sell_vol_at_min"]] = [100.0, 1.0]         # buy fill, never unwound passively
    r = V.mm_day(d, np.zeros(n, np.float32), -0.33, 0.33, skew=False)
    assert r["fills"] == 1 and r["flattens"] == 1
    assert abs(sum(r["taker_h"]) - 100.0) < 1e-9                           # sold at the bid
