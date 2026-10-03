"""Indian capital-gains tax rules."""
from __future__ import annotations

import pandas as pd
import pytest

from accounting.tax_india import LotBook, Realised, fy_of, post_tax_nav, tax_by_fy

T = pd.Timestamp


def test_fy_and_rates_by_sale_date():
    assert fy_of(T("2025-03-31")) == 2025 and fy_of(T("2025-04-01")) == 2026
    r = tax_by_fy([Realised(T("2024-07-01"), "A", 100_000, False), Realised(T("2024-08-01"), "A", 100_000, False)])
    assert r.loc[2025, "tax"] == pytest.approx((15_000 + 20_000) * 1.04)


def test_ltcg_exemption_and_old_exempt_regime():
    r = tax_by_fy([Realised(T("2025-06-01"), "A", 225_000, True)])
    assert r.loc[2026, "tax"] == pytest.approx(100_000 * 0.125 * 1.04)
    r0 = tax_by_fy([Realised(T("2017-06-01"), "A", 500_000, True)])
    assert r0.empty or r0["tax"].sum() == 0.0


def test_long_term_losses_cannot_offset_short_term_gains_but_short_term_losses_can_offset_long_term():
    a = tax_by_fy([Realised(T("2025-06-01"), "A", 100_000, False), Realised(T("2025-07-01"), "B", -100_000, True)])
    assert a.loc[2026, "tax"] == pytest.approx(20_000 * 1.04) and a.loc[2026, "loss_cf_lt"] == pytest.approx(100_000)
    b = tax_by_fy([Realised(T("2025-06-01"), "A", 300_000, True), Realised(T("2025-07-01"), "B", -100_000, False)])
    assert b.loc[2026, "tax"] == pytest.approx((200_000 - 125_000) * 0.125 * 1.04)


def test_carry_forward_expires_after_eight_years():
    r = tax_by_fy([Realised(T("2015-06-01"), "A", -100_000, False), Realised(T("2024-06-01"), "A", 100_000, False)])
    assert r.loc[2025, "tax"] == pytest.approx(15_000 * 1.04)          # FY2016 loss expired by FY2025 (9 years)
    r = tax_by_fy([Realised(T("2017-06-01"), "A", -100_000, False), Realised(T("2024-06-01"), "A", 100_000, False)])
    assert r.loc[2025, "tax"] == 0.0


def test_fifo_lots_and_holding_period():
    b = LotBook()
    b.buy(T("2023-01-02"), "X", 1000, 10)           # 100 sh
    b.buy(T("2024-03-01"), "X", 1200, 12)           # 100 sh
    b.sell(T("2024-06-03"), "X", 1500, 15)          # 100 sh from the first lot: +500, long term
    b.sell(T("2024-06-04"), "X", 1500, 15)          # 100 sh from the second lot: +300, short term
    assert [(round(r.gain), r.long_term) for r in b.realised] == [(500, True), (300, False)]


def test_post_tax_nav_pays_at_fy_end():
    d = pd.bdate_range("2024-01-01", "2025-06-30")
    nav = pd.Series(1_000_000.0, index=d)
    nav.loc[d >= "2024-06-01"] = 1_100_000.0
    fills = [{"t": 0, "symbol": "X", "delta_w": 1.0, "value": 1_000_000.0, "price": 100.0},
             {"t": int(d.get_loc(T("2024-06-03"))), "symbol": "X", "delta_w": -1.0, "value": 1_100_000.0, "price": 110.0}]
    out = post_tax_nav(nav, fills, d)
    paid_on = out["post_tax_nav"].diff().idxmin()
    assert paid_on == T("2024-03-29") or paid_on == T("2025-03-31")
    assert out["tax_paid_along_the_way"] == pytest.approx(100_000 * 0.15 * 1.04)


def test_exempt_era_long_term_results_neither_absorb_nor_carry_losses():
    a = tax_by_fy([Realised(T("2017-06-01"), "A", -100_000, False), Realised(T("2017-07-01"), "B", 100_000, True),
                   Realised(T("2018-06-01"), "A", 100_000, False)])
    assert a.loc[2019, "tax"] == 0.0                     # the FY2018 short-term loss was not eaten by the exempt LTCG
    b = tax_by_fy([Realised(T("2017-06-01"), "A", -300_000, True), Realised(T("2019-06-01"), "B", 300_000, True)])
    assert b.loc[2020, "tax"] == pytest.approx((300_000 - 100_000) * 0.10 * 1.04)


def test_twelve_months_exactly_is_short_term():
    b = LotBook()
    b.buy(T("2023-03-01"), "X", 1000, 10)
    b.sell(T("2024-03-01"), "X", 1100, 11)
    assert b.realised[0].long_term is False
    b.buy(T("2023-03-01"), "Y", 1000, 10)
    b.sell(T("2024-03-04"), "Y", 1100, 11)
    assert b.realised[1].long_term is True


def test_capital_efficiency_gate_is_sequential():
    from lab.gates import GateInput, capital_efficiency_gate

    good = GateInput(True, 0.03, 0.15, 0.13, 0.01, 0.005, 0.0055, 0.029, [0.01, 0.02, -0.01])
    r = capital_efficiency_gate(good)
    assert r["verdict"] == "ELIGIBLE FOR A FROZEN FORWARD TEST" and r["steps"][-1]["result"] == "PENDING"
    bad_tax = GateInput(True, 0.03, 0.12, 0.13, 0.01, 0.005, 0.0055, 0.029, [0.01, 0.02, 0.01])
    r = capital_efficiency_gate(bad_tax)
    assert r["verdict"] == "REJECTED at after_tax_estimate"
    assert [s["result"] for s in r["steps"]][3:] == ["NOT REACHED"] * 4
    assert capital_efficiency_gate(GateInput(False, 0.05, 0.2, 0.1, 0.04, 0.0, 0.0, 0.05, [1, 1, 1]))["verdict"] == "REJECTED at statistical"
    assert capital_efficiency_gate(GateInput(True, 0.05, 0.2, 0.1, 0.04, 0.001, 0.001, 0.05, [1, 1, 1], "passed"))["verdict"].startswith("MAY BE")
