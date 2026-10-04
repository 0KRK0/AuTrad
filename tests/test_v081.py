"""v0.8.1: corporate-action parsing and adjustment, security linking, point-in-time liquidity universe, Indian
crypto tax (no loss set-off, never a refund), crypto redenomination splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from accounting.tax_crypto_india import RATE, crypto_post_tax
from marketdata.nse_corp_actions import parse_subject, rights_factor
from marketdata.nse_universe import adjust, liquidity_universe, link_ids
from scripts.v081_crypto import split_redenominations


@pytest.mark.parametrize("subject,factor,div", [
    ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share", 0.1, 0),
    ("Fv Splt Frm Rs 10 To Re 1", 0.1, 0),
    ("Face Value Split Rs 10 To Rs 1", 0.1, 0),
    ("Bonus 1:2", 2 / 3, 0),
    ("Bonus- 1:1", 0.5, 0),
    ("Bonus 4:1/Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 5/- Per Share", 0.1, 0),
    ("Consolidation Of Shares From Re 1/- Per Share To Rs 10/- Per Share", 10.0, 0),
    ("Scheme Of Arrangement - Bonus Ncrps 4:1", 1.0, 0),            # preference shares: not an equity bonus
    ("Interim Divided - Rs 2.80 Per Share", 1.0, 2.8),
    ("Final Dividend - Rs. - 2.50 Per Share", 1.0, 2.5),
    ("Return Of Capital Rs 5 Per Share", 1.0, 5.0),
])
def test_parse_subject(subject, factor, div):
    r = parse_subject(subject)
    assert r["price_factor"] == pytest.approx(factor) and r["dividend"] == pytest.approx(div)


def test_rights_and_unpriced_actions():
    r = parse_subject("Rights 1:6 @ Premium Rs 49/-")
    assert r["rights"] == (1, 6, 49.0) and r["price_factor"] == 1.0
    assert rights_factor(100.0, 1, 6, 49.0, 1.0) == pytest.approx((6 * 100 + 50) / (7 * 100))
    assert rights_factor(40.0, 1, 6, 49.0, 1.0) == 1.0                       # out of the money
    assert "demerger" in parse_subject("Demerger")["unadjusted"]


def _ca(rows):
    df = pd.DataFrame(rows)
    for c, v in (("dividend", 0.0), ("unadjusted", ""), ("rights_a", 0), ("rights_b", 0), ("rights_premium", 0.0), ("face_value", 10.0), ("series", "EQ"), ("subject", "")):
        if c not in df:
            df[c] = v
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    return df


def _px(rows):
    df = pd.DataFrame(rows, columns=["date", "isin", "symbol", "open", "close", "prev_close"])
    df["date"] = pd.to_datetime(df["date"])
    df["sec"], df["value"], df["high"], df["low"] = df["isin"], 1e7, df["close"], df["close"]
    return df


def test_adjust_split_dividend_already_applied_unpriced_and_extremes():
    px = _px([("2020-01-02", "INEA00001", "A", 102, 101, 1000),   # 1:10 split, prev_close not adjusted
              ("2020-01-02", "INEB00001", "B", 98, 99, 100),      # Rs 2 dividend
              ("2020-01-02", "INEC00001", "C", 100, 101, 100),    # split record but NSE already adjusted prev_close
              ("2020-01-02", "INED00001", "D", 90, 85, 100),      # demerger: -15% overnight is value moved out
              ("2020-01-02", "INEE00001", "E", 50, 51, 100),      # no record, ratio 2: unrecorded split
              ("2020-01-02", "INEF00001", "F", 60, 45, 100)])     # no record, not a standard ratio: genuine crash
    ca = _ca([{"isin": "INEA00001", "symbol": "A", "ex_date": "2020-01-02", "price_factor": 0.1},
              {"isin": "INEB00001", "symbol": "B", "ex_date": "2020-01-02", "price_factor": 1.0, "dividend": 2.0},
              {"isin": "INEC00001", "symbol": "C", "ex_date": "2020-01-02", "price_factor": 0.1},
              {"isin": "INED00001", "symbol": "D", "ex_date": "2020-01-02", "price_factor": 1.0, "unadjusted": "demerger"}])
    a = adjust(px, ca).set_index("isin")
    assert a.loc["INEA00001", "cc"] == pytest.approx(101 / 100 - 1)
    assert a.loc["INEB00001", "cc"] == pytest.approx((99 + 2) / 100 - 1)
    assert a.loc["INEC00001", "cc"] == pytest.approx(0.01) and bool(a.loc["INEC00001", "factor_skipped"])
    assert a.loc["INED00001", "on"] == 0.0 and a.loc["INED00001", "cc"] == pytest.approx(85 / 90 - 1)   # keeps the day's real move
    assert a.loc["INEE00001", "flag"] == "unrecorded_split" and a.loc["INEE00001", "cc"] == pytest.approx(51 / 50 - 1)
    assert a.loc["INEF00001", "flag"] == "unexplained_extreme" and a.loc["INEF00001", "cc"] == pytest.approx(-0.55)


def test_link_ids_symbol_and_issuer():
    d = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-03-02"])
    df = pd.DataFrame({"date": [d[0], d[1], d[2], d[0], d[3], d[0], d[1]],
                       "isin": ["INE111A01011", "INE111A01011", "INE111A01029", "INE222B01011", "INE222B01029", "INE333C01011", "INE444D01011"],
                       "symbol": ["AAA", "AAA", "AAA", "BBB", "BBBNEW", "CCC", "DDD"]})
    s = link_ids(df)
    assert s.iloc[2] == "INE111A01011"                       # same symbol, next session
    assert s.iloc[4] == "INE222B01011"                       # same issuer code, re-listed within 60 sessions
    assert s.iloc[6] == "INE444D01011"                       # different issuer: never joined


def test_liquidity_universe_is_point_in_time():
    idx = pd.bdate_range("2020-01-01", periods=300)
    v = pd.DataFrame({"A": 10.0, "B": 5.0, "C": 1.0}, index=idx)
    v.loc[idx[200]:, "C"] = 100.0                           # C becomes liquid only later
    t = idx[150]
    assert liquidity_universe(v, t, 2) == ["A", "B"]
    assert liquidity_universe(v.iloc[:151], t, 2) == liquidity_universe(v, t, 2)   # future rows change nothing
    v2 = v.copy()
    v2.loc[idx[50]:idx[100], "B"] = 0.0                       # traded on < 80 % of the window -> ineligible
    assert "B" not in liquidity_universe(v2, t, 3)


def _nav(vals, start="2021-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def test_crypto_tax_no_loss_set_off_and_never_a_refund():
    nav = _nav([100.0] * 400)
    dates = nav.index
    fills = [{"t": 1, "symbol": "X", "delta_w": 1, "value": 100.0, "price": 10.0},
             {"t": 2, "symbol": "X", "delta_w": -1, "value": 50.0, "price": 5.0},     # loss: ignored
             {"t": 3, "symbol": "Y", "delta_w": 1, "value": 50.0, "price": 10.0},
             {"t": 4, "symbol": "Y", "delta_w": -1, "value": 100.0, "price": 20.0}]   # +50 gain taxed in full
    tx = crypto_post_tax(nav, fills, dates, {}, recoverable=True)
    assert tx["tax_paid"] == pytest.approx(RATE * 50.0) and tx["losses_not_set_off"] == pytest.approx(50.0)
    big = [{"t": 1, "symbol": "Z", "delta_w": 1, "value": 100.0, "price": 1.0},
           {"t": 2, "symbol": "Z", "delta_w": -1, "value": 10_000.0, "price": 100.0}]
    tx2 = crypto_post_tax(_nav([100.0] * 400), big, dates, {})                       # tax far above the account
    assert tx2["liquidation_value"] == 0.0 and (tx2["post_tax_nav"] >= 0).all() and tx2["wiped_out_on"]


def test_split_redenominations():
    d = pd.DataFrame({"date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-02-01", "2021-02-02"]),
                      "open": [1.0, 1.0, 1000.0, 1000.0], "close": [1.0, 1.0, 1000.0, 1010.0], "sym": "COCOS"})
    out = split_redenominations(d)
    assert list(out["sym"]) == ["COCOS", "COCOS", "COCOS#2", "COCOS#2"]
