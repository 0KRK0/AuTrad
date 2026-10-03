"""Tests for the v0.3 research lab: engine reconciliation, point-in-time blocking, corporate actions,
PBO sanity, trial registry, sealed-test selection, regimes, intraday accounting and ARGO-Meta."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from execution.costs import PROFILES
from lab import engine
from lab.intraday import DayBars, IntradayData, orb, random_direction, simulate
from lab.judge import JudgeConfig, judge, window
from lab.meta import MetaParams, run_meta, sleeve_returns
from lab.panel import Panel
from lab.pbo import pbo
from lab.regime import regime_series
from lab.registry import Registry
from lab.strategies import StrategySpec
from marketdata.corporate_actions import repair_blips
from marketdata.pit import PITUniverse


def make_panel(C: np.ndarray, eligible: np.ndarray | None = None, profile: str = "zero", klass: str = "default",
               start: str = "2015-01-01") -> Panel:
    T, N = C.shape
    dates = pd.bdate_range(start, periods=T)
    V = np.full_like(C, 1e9)
    el = np.ones_like(C, dtype=bool) if eligible is None else eligible
    return Panel(dates, [f"S{i}" for i in range(N)], C.copy(), C * 1.001, C * 0.999, C.copy(), V, el,
                 ["X"] * N, [klass] * N, [PROFILES[profile]] * N, pd.Series(C[:, 0], dates), "test")


def trending(T: int = 600, N: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.01, (T, N))
    return 100 * np.cumprod(1 + r, axis=0)


# ---------- engine
def test_engine_buy_and_hold_reconciles_with_price_path():
    C = trending()
    P = make_panel(C)
    t = pd.DataFrame({"S0": [1.0]}, index=[P.dates[0]])
    r = engine.run(P, t, capital=1e6, cash_rate=0.0)
    # bought at the open of day 1 (== close here), then rides the price
    expected = 1e6 * C[-1, 0] / C[1, 0]
    assert r.nav.iloc[-1] == pytest.approx(expected, rel=1e-6)
    assert r.costs.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_engine_costs_reduce_nav_and_scale_with_multiplier():
    C = trending()
    P = make_panel(C, profile="nse_largecap")
    rows = {P.dates[t]: pd.Series({"S0": 1.0 if (t // 20) % 2 == 0 else 0.0}) for t in range(0, 500, 20)}
    tg = pd.DataFrame(rows).T
    r1 = engine.run(P, tg, cash_rate=0.0)
    r2 = engine.run(P, tg, cash_rate=0.0, cost_mult=2.0)
    assert r1.costs.iloc[-1] > 0
    assert r2.costs.iloc[-1] == pytest.approx(2 * r1.costs.iloc[-1], rel=0.05)
    assert r2.nav.iloc[-1] < r1.nav.iloc[-1]


def test_engine_blocks_entries_into_ineligible_names():
    C = trending()
    el = np.ones_like(C, dtype=bool)
    el[:300, 1] = False                     # S1 not an index member for the first 300 days
    P = make_panel(C, el)
    tg = pd.DataFrame({"S1": [1.0]}, index=[P.dates[10]])
    r = engine.run(P, tg, cash_rate=0.0)
    assert np.all(np.abs(r.weights[:300, 1]) < 1e-12)


def test_engine_floors_non_shortable_at_zero():
    C = trending()
    P = make_panel(C, profile="nse_largecap")
    tg = pd.DataFrame({"S0": [-0.5]}, index=[P.dates[5]])
    r = engine.run(P, tg, cash_rate=0.0)
    assert r.weights.min() >= -1e-12


# ---------- point-in-time universe
def test_pit_membership_uses_latest_snapshot_on_or_before_date(tmp_path):
    p = tmp_path / "snap.csv"
    p.write_text("effective_date,symbol,company_name,isin,weight_pct\n"
                 "2020-01-01,AAA,A,I1,1\n2020-01-01,BBB,B,I2,1\n2021-01-01,AAA,A,I1,1\n2021-01-01,CCC,C,I3,1\n")
    U = PITUniverse(p)
    assert U.members(pd.Timestamp("2020-06-01")) == {"AAA", "BBB"}
    assert U.members(pd.Timestamp("2021-06-01")) == {"AAA", "CCC"}
    assert "BBB" in U.ever_removed()


# ---------- corporate actions
def test_repair_blips_fixes_power_of_ten_glitch():
    idx = pd.bdate_range("2019-12-16", periods=8)
    c = np.array([100, 101, 102, 10.2, 10.3, 103, 104, 105], dtype=float)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    fixed, notes = repair_blips(df)
    assert fixed["close"].iloc[3] == pytest.approx(102.0)
    assert notes


# ---------- statistics
def test_pbo_low_for_real_edge_high_for_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, (1200, 10))
    assert pbo(noise, 10)["pbo"] > 0.25
    edge = noise.copy()
    edge[:, 0] += 0.004                      # one config genuinely better everywhere
    assert pbo(edge, 10)["pbo"] < 0.05


def test_registry_counts_distinct_trials_and_charges_for_reruns_once(tmp_path):
    R = Registry(tmp_path / "t.jsonl")
    base = R.count()
    R.add([{"strategy": "x", "config": {"a": 1}}, {"strategy": "x", "config": {"a": 2}}])
    R.add([{"strategy": "x", "config": {"a": 1}}])
    assert R.count() == base + 2
    assert all(json.loads(line)["ts"] for line in (tmp_path / "t.jsonl").open())


def test_regime_labels_bull_and_bear():
    idx = pd.bdate_range("2010-01-01", periods=1200)
    noise = np.random.default_rng(2).normal(0, 0.002, 1200)
    up = pd.Series(100 * np.exp(np.cumsum(0.001 + noise)), index=idx)
    down = pd.Series(100 * np.exp(np.cumsum(-0.001 + noise)), index=idx)
    assert (regime_series(up).iloc[-300:].isin(["BULL", "STRESS"])).mean() > 0.9
    assert (regime_series(up).iloc[-300:] == "BULL").mean() > 0.6
    assert (regime_series(down).iloc[-300:] == "BEAR").mean() > 0.6
    assert (regime_series(up).iloc[:150] == "UNKNOWN").all()


# ---------- the judge never selects on test data
def test_judge_selection_ignores_the_sealed_test_period():
    T = 2800
    C = trending(T, 2, seed=3)
    P = make_panel(C, klass="diversified")
    test_start = P.dates[2400]

    def fn(P, pick: int = 0):
        # config 1 is terrible before the test and wonderful during it; the judge must still pick config 0
        rows = {}
        for t in range(0, len(P.dates), 21):
            w = pd.Series(0.0, index=P.symbols)
            w[P.symbols[0]] = 1.0 if pick == 0 or P.dates[t] >= test_start else 0.0
            rows[P.dates[t]] = w
        return pd.DataFrame(rows).T

    spec = StrategySpec("leak_probe", "TEST", "months", "probe", fn, {"pick": [0, 1]})
    jc = JudgeConfig(oos_start=str(P.dates[800].date()), val_start=str(P.dates[1800].date()),
                     test_start=str(test_start.date()), n_null=4, workers=1, pbo_blocks=6,
                     cost_shocks=(1.0, 2.0), capitals=(1e6,))
    out = judge(spec, P, jc, 0, log=lambda *_: None)
    assert out["selected"] == {"pick": 0}


def test_window_prepends_the_last_pre_window_point():
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"]))
    w = window(s, "2021-01-01")
    assert list(w.values) == [2.0, 3.0, 4.0]


# ---------- intraday
def _intraday_fixture() -> IntradayData:
    days = list(pd.bdate_range("2024-01-01", periods=3))
    bars, prev = {"AAA": {}}, {"AAA": {}}
    last = 100.0
    for d in days:
        ts = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(hours=h) for h in range(7)])
        o = np.array([100, 101, 102, 103, 104, 105, 106], float)
        c = o + 1
        bars["AAA"][d] = DayBars(np.arange(7) * 60, o, c + 0.5, o - 0.5, c, np.full(7, 1e7), ts)
        prev["AAA"][d] = last
        last = c[-1]
    return IntradayData("1h", 60, ["AAA"], days, bars, prev)


def test_orb_enters_next_open_after_breakout_and_exits_at_close():
    D = _intraday_fixture()
    tr = orb(D, or_min=60, direction="long")
    assert len(tr) == 3
    r = tr.iloc[0]
    assert r.entry_px == 102.0 and r.exit_px == 107.0     # breakout on bar 1 close (102 > 101.5), next open 102
    res = simulate(D, tr, capital=1e6, max_pos=1, cash_rate=0.0)
    assert res.trades["cost"].gt(0).all()
    assert res.trades["gross"].iloc[0] == pytest.approx(1e6 / 102 * 5, rel=1e-9)


def test_random_direction_null_keeps_timing():
    D = _intraday_fixture()
    tr = orb(D, direction="long")
    rd = random_direction(tr, np.random.default_rng(0))
    assert (rd[["entry_ts", "exit_ts", "symbol"]].values == tr[["entry_ts", "exit_ts", "symbol"]].values).all()
    assert set(rd.side.unique()) <= {-1, 1}


# ---------- ARGO-Meta
def test_meta_all_cash_earns_cash_rate_and_allocations_never_exceed_one():
    dates = pd.bdate_range("2016-01-01", periods=900)
    rng = np.random.default_rng(4)
    navs = {"good": pd.Series(np.cumprod(1 + rng.normal(0.0008, 0.005, 900)), dates),
            "bad": pd.Series(np.cumprod(1 + rng.normal(-0.0008, 0.005, 900)), dates)}
    R = sleeve_returns(navs, dates)
    reg = pd.Series("BULL", index=R.index)
    cash = run_meta(R, reg, MetaParams(), [], cash_rate=0.06, capital=1e6)
    yrs = (R.index[-1] - R.index[0]).days / 365
    assert cash.nav.iloc[-1] == pytest.approx(1e6 * 1.06 ** yrs, rel=0.01)
    m = run_meta(R, reg, MetaParams(z=0.0), ["good", "bad"], cash_rate=0.06, capital=1e6)
    assert (m.alloc.sum(axis=1) <= 1 + 1e-9).all()
    assert m.alloc["bad"].iloc[-5:].max() < 1e-9 and m.alloc["good"].iloc[-5:].min() > 0
