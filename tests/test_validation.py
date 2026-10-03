import numpy as np
import pandas as pd
import pytest

from agents.allocator import CategoryAllocator
from argo import ArgoConfig, ArgoModel
from backtest.ledger import build_ledger, trade_stats
from backtest.sim import Market, SimConfig, SimResult, simulate
from backtest.strategies import BuyHold, RandomTrader, Strategy
from execution.broker import Fill, Order, PaperBroker
from execution.costs import PROFILES
from marketdata.loaders import synthetic
from marketdata.store import DataStore, clean
from marketdata.universe import Asset
from validation import stats, tax
from validation.metrics import excess_returns, perf, periods_per_year
from validation.walkforward import walk_forward


# ---------- costs + broker ----------
def test_equity_delivery_charges_match_published_rates():
    p = PROFILES["nse_largecap"]
    buy = p.fees(1, 1_00_000)                     # STT 100 + stamp 15 + txn 3.07 + SEBI 0.01 + GST 0.57
    sell = p.fees(-1, 1_00_000)                   # STT 100 + txn 3.07 + SEBI 0.01 + GST 0.57 + DP 15.34
    assert buy == pytest.approx(118.7, abs=0.2)
    assert sell == pytest.approx(103.7 + 15.34, abs=0.2)


def test_currency_brokerage_is_capped_at_20_rupees():
    p = PROFILES["nse_currency_fut"]
    assert p.fees(1, 10_00_000) < 20 * 1.18 + 10_00_000 * p.buy_bps / 1e4 + 1e-6


def test_broker_fx_without_volume_fills_fully_and_tracks_costs():
    b = PaperBroker(10_00_000, profiles={"USDINR": PROFILES["nse_currency_fut"]})
    o = b.place_order(Order("USDINR", 1000))
    b.on_bar("USDINR", 88.0, 88.2, 87.9, volume=0, sigma=0.003)
    assert o.status == "FILLED" and b.fees_paid > 0 and b.slippage_paid > 0
    assert not b.open_orders()


def test_broker_open_order_index_drops_filled_orders():
    b = PaperBroker(1e7, profiles={"X": PROFILES["nse_largecap"]})
    for _ in range(50):
        b.place_order(Order("X", 10))
    b.on_bar("X", 100, 101, 99, volume=1e6, sigma=0.01)
    assert not b.open_orders() and len(b.orders) == 50


# ---------- data quality ----------
def test_clean_tolerates_vendor_rounding_but_flags_real_breaches():
    df = synthetic("X", n=300)
    top = df[["open", "close"]].max(axis=1)
    df.loc[df.index[:30], "high"] = top.iloc[:30] * 0.9999           # tiny breach on 10% of bars
    _, q = clean(df, "fx")
    assert q.passed and q.ohlc_repaired >= 30
    df.loc[df.index[:30], "high"] = top.iloc[:30] * 0.9               # 10% breach: corrupt
    _, q = clean(df, "equity")
    assert not q.passed


def test_store_detects_tampered_files(tmp_path):
    s = DataStore(tmp_path)
    s.save(Asset("X", "X", "equity", "in_largecap", "it", "nse_largecap"), synthetic("X", n=300), "test")
    assert "X" in s.load(["X"])
    p = tmp_path / "raw" / "X.csv"
    p.write_text(p.read_text().replace("\n2019", "\n2019", 1) + "\n")
    with pytest.raises(ValueError):
        s.load(["X"])


# ---------- ledger ----------
def test_ledger_round_trip_and_flip():
    t = pd.Timestamp
    fills = [(t("2024-01-01"), Fill(1, "A", 10, 100.0, 1.0)), (t("2024-01-10"), Fill(2, "A", -10, 110.0, 1.0)),
             (t("2024-02-01"), Fill(3, "A", 5, 100.0, 0.5)), (t("2024-02-05"), Fill(4, "A", -8, 90.0, 0.5))]
    led = build_ledger(fills, {"A": 95.0}, {"A": "in_largecap"}, t("2024-03-01"))
    assert led.iloc[0].pnl == pytest.approx(100 - 2)            # (110-100)*10 - fees
    assert led.iloc[1].pnl == pytest.approx(-50 - 1.0)          # (90-100)*5 - fees
    assert led.iloc[2]["open"] and led.iloc[2].side == "short"  # flipped remainder, marked at 95
    assert trade_stats(led)["trades"] == 3


# ---------- metrics + stats + tax ----------
def test_cash_only_equity_has_zero_excess_and_right_calendar():
    idx = pd.bdate_range("2020-01-01", "2022-12-31")
    days = idx.to_series().diff().dt.days.fillna(0)
    eq = pd.Series(1e6 * (1.06 ** (days.cumsum() / 365)).to_numpy(), idx)
    assert abs(excess_returns(eq, 0.06)).max() < 1e-12
    assert 255 < periods_per_year(idx) < 265
    assert 364 < periods_per_year(pd.date_range("2020-01-01", "2022-12-31")) < 366
    assert perf(eq, 0.06)["excess_cagr"] == pytest.approx(0, abs=1e-4)   # 365 vs 365.25 day-count only


def test_psr_and_deflation():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.001, 0.01, 1500))
    assert stats.psr(good) > 0.95
    trials = list(rng.normal(0.0, 0.03, 50))
    assert stats.deflated_sharpe(good, trials, 50) < stats.psr(good)
    assert stats.random_pvalue(1.0, [0.1] * 99) == pytest.approx(0.01)


def test_tax_equity_stcg_and_crypto_no_loss_relief():
    led = pd.DataFrame([
        {"symbol": "A", "category": "in_largecap", "exit": pd.Timestamp("2024-06-01"), "days": 30, "pnl": 10_000, "open": False},
        {"symbol": "B", "category": "in_largecap", "exit": pd.Timestamp("2024-07-01"), "days": 30, "pnl": -4_000, "open": False},
        {"symbol": "C", "category": "crypto_major", "exit": pd.Timestamp("2024-08-01"), "days": 5, "pnl": 5_000, "open": False},
        {"symbol": "D", "category": "crypto_major", "exit": pd.Timestamp("2024-09-01"), "days": 5, "pnl": -5_000, "open": False},
    ])
    t = tax.estimate(led).loc[2025]
    assert t.equity_tax == pytest.approx(0.20 * 6_000 * 1.04)
    assert t.crypto_tax == pytest.approx(0.30 * 5_000 * 1.04)   # the crypto loss gives no relief


# ---------- simulator ----------
def _market(n=400, syms=("A", "B")):
    data = {s: synthetic(s, n=n, seed=i) for i, s in enumerate(syms)}
    for d in data.values():
        d.index = d.index.tz_localize(None)
    return Market(data, {s: "x" for s in syms}, {s: "in_largecap" for s in syms}, {s: PROFILES["nse_largecap"] for s in syms})


class _Peek(Strategy):
    name = "peek"

    def targets(self, ctx):
        self.seen_last = ctx.t
        assert all(len(h) == self.bars[s] + 1 for s, h in ctx.history.items() if (self.bars.__setitem__(s, self.bars.get(s, -1) + 1) or True))
        return {}

    def __init__(self):
        self.bars = {}


def test_sim_history_never_includes_future_bars():
    simulate(_market(n=120), _Peek(), SimConfig())


def test_sim_orders_fill_at_next_open_with_costs():
    mk = _market(n=300)
    r = simulate(mk, BuyHold(), SimConfig(cash_rate=0.0))
    first_fill_day = r.exposure[r.exposure > 0].index[0]
    assert first_fill_day == mk.dates[1]                      # decided on bar 0, filled on bar 1
    assert r.fees > 0 and r.costs.iloc[-1] == pytest.approx(r.fees + r.slippage)


def test_random_trader_matches_target_exposure_roughly():
    mk = _market(n=800, syms=tuple("ABCDEFGH"))
    r = simulate(mk, RandomTrader(1, 0.05, 0.2, 10), SimConfig(cash_rate=0.0, min_trade_frac=0.0))
    assert 0.08 < r.exposure.iloc[100:].mean() < 0.35


# ---------- ARGO horizon + allocator + walk-forward ----------
def test_argo_horizon_scores_predictions_only_after_h_bars():
    m = ArgoModel("X", ArgoConfig(horizon=5))
    row = {"r1": 0.0, "vol20": 0.01}
    for _ in range(4):
        m.predict(row)
        m.learn(0.01)
    assert len(m._queue) == 4 and m.alloc.n.sum() == 0      # nothing scored yet
    m.predict(row)
    m.learn(0.01)
    assert m.alloc.n.sum() > 0 and len(m._queue) == 4       # first prediction scored after 5 returns


def test_allocator_shrinks_losing_category_but_never_exceeds_one():
    a = CategoryAllocator()
    for _ in range(100):
        a.update("loser", -0.01 + 0.001 * np.sin(_))
        a.update("winner", 0.01 + 0.001 * np.sin(_))
    assert a.multiplier("loser") < 0.3 and 0.99 < a.multiplier("winner") <= 1.0


def _fake(name, rets):
    idx = pd.bdate_range("2018-01-01", periods=len(rets))
    eq = pd.Series(1e6 * np.cumprod(1 + np.array(rets)), idx)
    z = pd.Series(0.0, idx)
    return SimResult(name, eq, z, 0.0, 0.0, z + 0.5, pd.DataFrame(index=idx), pd.DataFrame(), 0.0, [], 0)


def test_walk_forward_selects_using_only_the_past():
    rng = np.random.default_rng(3)
    n1 = len(pd.bdate_range("2018-01-01", "2019-12-31"))
    n2 = len(pd.bdate_range("2020-01-01", "2020-12-31"))
    a = list(rng.normal(0.002, 0.005, n1)) + list(rng.normal(-0.002, 0.005, n2))   # good early, bad later
    b = list(rng.normal(-0.001, 0.005, n1)) + list(rng.normal(0.003, 0.005, n2))   # bad early, good later
    wf = walk_forward({"A": _fake("A", a), "B": _fake("B", b)}, 2020, 0.0, 1e6)
    assert wf.folds.loc[2020, "chosen"] == "A"     # hindsight would pick B; honest selection picks A
    assert wf.folds.loc[2020, "test_return"] < 0
