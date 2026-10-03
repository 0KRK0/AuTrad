"""Regression tests for the Oct-2026 data-integrity audit: one test per vendor failure mode found against
BSE exchange prices, plus the factor arithmetic. Synthetic data reproduces each real case; a final group
runs on the real snapshot when the exchange cache is present."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketdata.corporate_actions import (
    _ratio_factor,
    _terp_factor,
    correct_with_exchange,
    load_registry,
)

N = 300


def _exchange(seed: int = 0, events: dict[int, float] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Unadjusted exchange bars with raw drops on event days, plus a flat-ish market index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=N)
    r = rng.normal(0.0004, 0.012, N)
    for t, k in (events or {}).items():
        r[t] += np.log(k)
    c = 1000 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * np.exp(np.r_[0.0, r[1:]] * 0.5)
    for t, k in (events or {}).items():
        o[t] = c[t - 1] * k                      # price discovery opens at the adjusted price
    hi, lo = np.maximum(o, c) * 1.005, np.minimum(o, c) * 0.995
    bse = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "prev_close": np.r_[c[0], c[:-1]], "volume": 1e5}, idx)
    bench = pd.Series(np.full(N, 20000.0), idx)
    return bse, bench


def _vendor_from(bse: pd.DataFrame, factor: pd.Series) -> pd.DataFrame:
    v = bse[["open", "high", "low", "close", "volume"]].copy()
    for c in ["open", "high", "low", "close"]:
        v[c] = v[c] * factor
    return v


def _reg(rows: list[dict]) -> pd.DataFrame:
    r = pd.DataFrame(rows, columns=["symbol", "ex_date", "kind", "terms", "method", "source", "note"])
    r["ex_date"] = pd.to_datetime(r["ex_date"])
    return r


def _total_return_ok(out: pd.DataFrame, bse: pd.DataFrame, ex_days: dict[int, float], tol: float = 0.003) -> None:
    """corrected log return == exchange raw log return - log(k) on event days, == raw elsewhere."""
    r = np.log(out["close"]).diff().to_numpy()
    raw = np.log(bse["close"]).diff().to_numpy()
    for t in range(1, N):
        expect = raw[t] - (np.log(ex_days[t]) if t in ex_days else 0.0)
        assert abs(r[t] - expect) < tol, (t, r[t], expect)


# ----------------------------------------------------------------- factor arithmetic
def test_ratio_factors_for_bonus_and_split():
    assert _ratio_factor("bonus", "1:2") == pytest.approx(2 / 3)      # TRENT 2026
    assert _ratio_factor("bonus", "1:1") == pytest.approx(1 / 2)
    assert _ratio_factor("split", "1:10") == pytest.approx(0.1)       # NIFTYBEES 2019
    assert _ratio_factor("split", "1:100") == pytest.approx(0.01)     # GOLDBEES 2019


def test_terp_matches_published_rights_terms():
    # ADANIENT rights 3:25 @ Rs1800, cum price 2516.85 -> vendor and exchange both show ~3.05%
    assert _terp_factor("3:25@1800", 2516.85) == pytest.approx(0.96948, abs=1e-4)
    # Vodafone Idea 87:38 @ Rs12.5 at Rs28.5 -> TERP Rs17.35 (Capitalmind's worked example)
    assert _terp_factor("87:38@12.5", 28.5) * 28.5 == pytest.approx(17.36, abs=0.02)


def test_registry_rows_all_have_a_source_and_a_known_method():
    r = load_registry()
    assert len(r) >= 15
    assert r["source"].astype(str).str.len().gt(5).all()
    assert set(r["method"]) <= {"ratio", "terp", "discovery", "close_mkt", "dividend"}


# ----------------------------------------------------------------- failure mode 1: right factor, wrong window
def test_adjustment_applied_to_a_truncated_window_is_rebuilt_over_full_history():
    """TRENT pattern: bonus on day 200, vendor scaled only days 150-199 and left earlier history raw."""
    k = 2 / 3
    bse, bench = _exchange(1, {200: k})
    f = pd.Series(1.0, bse.index)
    f.iloc[150:200] = k                                  # only the window
    v = _vendor_from(bse, f)
    reg = _reg([{"symbol": "X", "ex_date": bse.index[200], "kind": "bonus", "terms": "1:2", "method": "ratio",
                 "source": "test", "note": ""}])
    c = correct_with_exchange("X", v, bse, bench, reg)
    assert any(r["date"] == str(bse.index[150].date()) for r in c.removed)   # the fake drop is removed
    _total_return_ok(c.df, bse, {200: k})


# ----------------------------------------------------------------- failure mode 2: wrong demerger factor
def test_wrong_vendor_demerger_factor_is_replaced_by_exchange_price_discovery():
    """CESC/GRASIM pattern: exchange discovers 0.775, vendor used 0.897."""
    k_true = 0.775
    bse, bench = _exchange(2, {120: k_true})
    f = pd.Series(1.0, bse.index)
    f.iloc[:120] = 0.897
    v = _vendor_from(bse, f)
    reg = _reg([{"symbol": "X", "ex_date": bse.index[120], "kind": "demerger", "terms": "", "method": "discovery",
                 "source": "test", "note": ""}])
    c = correct_with_exchange("X", v, bse, bench, reg)
    ev = [e for e in c.events if e.kind == "demerger"][0]
    assert ev.factor == pytest.approx(k_true, rel=1e-6)
    assert ev.vendor_factor == pytest.approx(0.897, rel=0.01)
    _total_return_ok(c.df, bse, {120: k_true})


def test_unregistered_big_event_with_disagreeing_vendor_factor_is_flagged_unverified():
    bse, bench = _exchange(3, {120: 0.7})
    f = pd.Series(1.0, bse.index)
    f.iloc[:120] = 0.62
    c = correct_with_exchange("X", _vendor_from(bse, f), bse, bench, _reg([]))
    ev = [e for e in c.events if "UNVERIFIED" in e.evidence]
    assert ev and ev[0].factor == pytest.approx(0.7, rel=1e-6)
    assert c.quality.loc[bse.index[120]] <= 0.3                         # kept out of trading until verified


# ----------------------------------------------------------------- failure mode 3: no adjustment at all
def test_unadjusted_demerger_gets_the_registry_factor():
    """Tata Motors / Vedanta pattern: vendor shows the raw -40% / -63% drop."""
    bse, bench = _exchange(4, {100: 0.6037})
    v = _vendor_from(bse, pd.Series(1.0, bse.index))
    reg = _reg([{"symbol": "X", "ex_date": bse.index[100], "kind": "demerger", "terms": "", "method": "discovery",
                 "source": "test", "note": ""}])
    c = correct_with_exchange("X", v, bse, bench, reg)
    _total_return_ok(c.df, bse, {100: 0.6037})


# ----------------------------------------------------------------- failure mode 4: misprints at the wrong scale
def test_two_day_misprint_at_wrong_scale_is_replaced_by_exchange_bars():
    """ETF pattern (Dec 2019): two vendor prints 10x too low, then back to normal."""
    bse, bench = _exchange(5)
    f = pd.Series(1.0, bse.index)
    f.iloc[80:82] = 0.1
    c = correct_with_exchange("X", _vendor_from(bse, f), bse, bench, _reg([]))
    assert len(c.replaced_days) == 2
    assert np.allclose(c.df["close"], bse["close"], rtol=1e-9)
    assert (c.quality.iloc[80:82] == 0.5).all()


# ----------------------------------------------------------------- things that must NOT be touched
def test_dividends_adjusted_correctly_by_the_vendor_are_kept():
    bse, bench = _exchange(6, {90: 0.98, 210: 0.97})
    f = pd.Series(1.0, bse.index)
    f.iloc[:90] *= 0.98
    f.iloc[:210] *= 0.97
    v = _vendor_from(bse, f)
    c = correct_with_exchange("X", v, bse, bench, _reg([]))
    assert not c.removed
    assert np.allclose(c.df["close"], v["close"], rtol=1e-3)


def test_real_crash_is_not_mistaken_for_a_corporate_action():
    """Yes Bank / COVID pattern: exchange and vendor both fall 30% with no factor change."""
    bse, bench = _exchange(7)
    bse = bse.copy()
    bse.iloc[150:, bse.columns.get_indexer(["open", "high", "low", "close"])] *= 0.7
    v = _vendor_from(bse, pd.Series(1.0, bse.index))
    c = correct_with_exchange("X", v, bse, bench, _reg([]))
    assert not [e for e in c.events if e.kind != "dividend"]
    assert np.allclose(c.df["close"], v["close"], rtol=1e-9)


# ----------------------------------------------------------------- real snapshot (skipped without the caches)
SNAP = Path("data/snap_20261001")
needs_data = pytest.mark.skipif(not (SNAP / "raw").exists() or not Path("data/exchange/bse").exists(),
                                reason="real snapshot or BSE cache not present")


@needs_data
def test_real_trent_has_no_fake_january_crash_and_bonus_on_june_4():
    from marketdata.store import DataStore

    st = DataStore(SNAP)
    d = st.load(["TRENT"])["TRENT"]
    r = np.log(d["close"]).diff()
    assert abs(r.loc["2026-01-01"]) < 0.05
    assert abs(r.loc["2026-06-04"]) < 0.05
    assert any(e.kind == "bonus" and str(e.date.date()) == "2026-06-04" for e in st.events["TRENT"])


@needs_data
def test_real_snapshot_has_no_level_errors_against_the_exchange():
    from marketdata.exchange import load, series
    from marketdata.store import DataStore

    codes = pd.read_csv("marketdata/reference/bse_codes.csv", dtype={"bse_code": str})
    st = DataStore(SNAP)
    data = st.load(list(codes["symbol"]))
    B = load("data/exchange")
    worst = {}
    for s, c in zip(codes["symbol"], codes["bse_code"], strict=True):
        b = series(B, c)
        b = b[b["close"] > 0]
        A = pd.Series(1.0, b.index)
        for e in st.events[s]:
            A[A.index < e.date] *= e.factor
        g = (np.log(data[s]["close"]).reindex(b.index) - np.log(b["close"] * A)).dropna()
        worst[s] = float(g.abs().max())
    assert max(worst.values()) < 0.05, sorted(worst.items(), key=lambda x: -x[1])[:5]


def test_large_dividend_is_not_snapped_to_a_bonus_ratio():
    """Bug found in the first audit pass: a ~8% dividend (Coal India, Vedanta) was rounded to a 1:10 bonus (10/11)."""
    bse, bench = _exchange(8, {150: 0.9173})
    f = pd.Series(1.0, bse.index)
    f.iloc[:150] = 0.9173
    c = correct_with_exchange("X", _vendor_from(bse, f), bse, bench, _reg([]))
    ev = [e for e in c.events if e.date == bse.index[150]][0]
    assert ev.factor == pytest.approx(0.9173, rel=2e-3)
    assert ev.kind != "ratio_event"


def test_split_uses_the_exact_published_ratio_not_the_vendors_residual():
    """TCS/HCLTECH/LT pattern: vendor factor 1.3% off the 1:1 bonus with no dividend that day (independent review)."""
    bse, bench = _exchange(9, {150: 0.4936})
    f = pd.Series(1.0, bse.index)
    f.iloc[:150] = 0.4936
    c = correct_with_exchange("X", _vendor_from(bse, f), bse, bench, _reg([]))
    ev = [e for e in c.events if e.date == bse.index[150]][0]
    assert ev.kind == "ratio_event" and ev.factor == pytest.approx(0.5)


# ----------------------------------------------------------------- failure modes found by the independent review
def test_vendor_window_eight_sessions_before_a_rights_issue_is_not_booked_as_a_factor_above_one():
    """ADANIENT Nov-2025 pattern: vendor scaled 8 sessions before the ex-date only; the early step must not become x1.03."""
    k = 0.9695
    bse, bench = _exchange(10, {200: k})
    f = pd.Series(1.0, bse.index)
    f.iloc[192:200] = k
    reg = _reg([{"symbol": "X", "ex_date": bse.index[200], "kind": "rights", "terms": "3:25@1800", "method": "terp",
                 "source": "test", "note": ""}])
    c = correct_with_exchange("X", _vendor_from(bse, f), bse, bench, reg)
    assert all(e.factor <= 1.0 for e in c.events if e.kind != "rights")     # the vendor window is not a factor
    assert any(r["date"] == str(bse.index[192].date()) for r in c.removed)
    kt = [e for e in c.events if e.kind == "rights"][0].factor
    _total_return_ok(c.df, bse, {200: kt})


def test_reversed_vendor_step_is_not_a_dividend():
    """Aug-2026 vendor noise: a 1.5% 'dividend' undone 3 sessions later."""
    bse, bench = _exchange(11)
    f = pd.Series(1.0, bse.index)
    f.iloc[:150] = 0.985
    f.iloc[:147] = 1.0
    c = correct_with_exchange("X", _vendor_from(bse, f), bse, bench, _reg([]))
    assert not [e for e in c.events if e.kind == "dividend"]
    assert np.allclose(c.df["close"], bse["close"], rtol=1e-9)


def test_persistent_dividend_is_kept_even_when_the_stock_moved_a_lot_that_day():
    """FEDERALBNK 2022-07-19 pattern: 1.8% dividend on a +5% day was dropped by the first version."""
    bse, bench = _exchange(12, {150: 0.982})
    bse = bse.copy()
    bse.iloc[150:, bse.columns.get_indexer(["open", "high", "low", "close"])] *= 1.05     # big move that day too
    f = pd.Series(1.0, bse.index)
    f.iloc[:150] = 0.982
    c = correct_with_exchange("X", _vendor_from(bse, f), bse, bench, _reg([]))
    assert [e for e in c.events if e.kind == "dividend" and e.date == bse.index[150]]
    assert np.allclose(c.df["close"], _vendor_from(bse, f)["close"], rtol=1e-3)


def test_stale_zero_volume_bar_is_replaced_and_holiday_phantom_bar_dropped():
    bse, bench = _exchange(13)
    v = _vendor_from(bse, pd.Series(1.0, bse.index))
    d = bse.index[100]
    v.loc[d, ["open", "high", "low", "close"]] = v["close"].iloc[99]       # flat copy of yesterday
    v.loc[d, "volume"] = 0
    holiday = bse.index[150]
    bse2 = bse.drop(holiday)                                                # exchange closed that day
    v.loc[holiday, ["open", "high", "low", "close"]] = v["close"].iloc[149]
    v.loc[holiday, "volume"] = 0
    c = correct_with_exchange("X", v, bse2, bench, _reg([]))
    assert holiday not in c.df.index
    assert c.df.loc[d, "close"] == pytest.approx(bse.loc[d, "close"])
    assert c.quality[d] == 0.5


def test_bad_vendor_open_is_replaced_by_the_exchange_open():
    """NIFTYBEES pattern: vendor opens far from the exchange (the engine trades at the open)."""
    bse, bench = _exchange(14)
    v = _vendor_from(bse, pd.Series(1.0, bse.index))
    d = bse.index[60]
    v.loc[d, "open"] = v.loc[d, "close"] * 0.8
    c = correct_with_exchange("X", v, bse, bench, _reg([]))
    assert c.df.loc[d, "open"] == pytest.approx(bse.loc[d, "open"])
    assert str(d.date()) in c.opens_replaced


def test_registry_event_missing_from_exchange_data_is_flagged_not_silently_dropped():
    bse, bench = _exchange(15, {120: 0.6})
    bse2 = bse.drop(bse.index[120])
    reg = _reg([{"symbol": "X", "ex_date": bse.index[120], "kind": "demerger", "terms": "", "method": "discovery",
                 "source": "test", "note": ""}])
    c = correct_with_exchange("X", _vendor_from(bse, pd.Series(1.0, bse.index)), bse2, bench, reg)
    assert any("NOT applied" in n for n in c.notes)
    assert c.quality.loc[bse.index[118]] <= 0.3
