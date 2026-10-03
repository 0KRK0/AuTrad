import numpy as np
import pandas as pd
import pytest

from argo import ArgoModel, build_features
from argo.experts import RLSExpert
from audit.log import AuditLog
from backtest.engine import BacktestConfig, run_backtest
from execution.broker import CostModel, Order, PaperBroker
from marketdata.loaders import synthetic, validate
from risk.gate import OrderIntent, PortfolioState, RiskGate, RiskLimits


def pf(**kw):
    base = dict(equity=100_000, sod_equity=100_000, peak_equity=100_000, weights={}, sectors={})
    base.update(kw)
    return PortfolioState(**base)


# ---------- Risk Gate ----------
def test_position_cap_resizes():
    d = RiskGate().check(OrderIntent("AAPL", 0.20), pf())
    assert d.approved and d.weight == pytest.approx(0.05)
    assert not d.checks["position_limit"]


def test_sector_cap():
    state = pf(weights={"MSFT": 0.05, "NVDA": 0.05, "GOOG": 0.05, "META": 0.05, "AMD": 0.05, "INTC": 0.04},
               sectors={s: "TECH" for s in ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AMD", "INTC"]})
    d = RiskGate().check(OrderIntent("AAPL", 0.05), state)
    assert d.weight == pytest.approx(0.01)


def test_daily_loss_halts_new_entries_but_allows_reduction():
    state = pf(equity=97_500, peak_equity=100_000, weights={"AAPL": 0.04})
    g = RiskGate()
    assert g.check(OrderIntent("AAPL", 0.05), state).weight == pytest.approx(0.04)
    assert g.check(OrderIntent("AAPL", 0.01), state).weight == pytest.approx(0.01)


def test_capital_preservation_halves_size():
    state = pf(equity=91_000, sod_equity=91_000, peak_equity=100_000)
    d = RiskGate().check(OrderIntent("AAPL", 0.04), state)
    assert d.mode == "PRESERVE" and d.weight == pytest.approx(0.02)


def test_max_drawdown_trips_kill_switch_and_blocks_everything():
    g = RiskGate()
    d = g.check(OrderIntent("AAPL", 0.01), pf(equity=87_000, sod_equity=87_000))
    assert not d.approved and g.kill.active
    assert not g.check(OrderIntent("MSFT", 0.01), pf()).approved  # latched


def test_stale_data_blocks_orders():
    assert not RiskGate().check(OrderIntent("AAPL", 0.03, data_age_s=30), pf()).approved


def test_order_rate_breaker():
    g = RiskGate(RiskLimits(max_orders_per_min=5))
    results = [g.check(OrderIntent(f"S{i}", 0.01), pf(), now=1000.0 + i) for i in range(7)]
    assert g.kill.active and not results[-1].approved


def test_crash_scenario_reduces_exposure_no_limit_violation():
    """Market drops ~10%: gate must never approve weight above caps; mode escalates."""
    g = RiskGate()
    w = {f"S{i}": 0.05 for i in range(10)}
    for eq in np.linspace(100_000, 90_000, 15):
        d = g.check(OrderIntent("S0", 0.5, atr_z=4.0), pf(equity=eq, sod_equity=100_000, weights=w))
        assert abs(d.weight) <= 0.05 + 1e-12
    assert d.mode in ("HALTED", "PRESERVE", "KILLED")


# ---------- Broker ----------
def test_paper_broker_partial_fill_costs_and_idempotency():
    b = PaperBroker(100_000, CostModel(max_participation=0.1))
    o = Order("AAPL", 1000, client_id="abc")
    b.place_order(o)
    assert b.place_order(Order("AAPL", 1000, client_id="abc")) is o  # idempotent
    b.on_bar("AAPL", 100, 101, 99, volume=6_500, sigma=0.01)
    assert o.status == "PARTIAL" and o.filled_qty == pytest.approx(650)
    assert o.avg_price > 100  # paid spread + impact
    assert b.cash < 100_000 - 650 * 100


# ---------- ARGO ----------
def test_rls_learns_linear_signal():
    rng = np.random.default_rng(1)
    e = RLSExpert(name="t", cols=[0, 1])
    for _ in range(500):
        x = rng.normal(size=8)
        e.update(x, 0.01 * x[0] - 0.02 * x[1])
    assert e.w[0] == pytest.approx(0.01, abs=1e-3) and e.w[1] == pytest.approx(-0.02, abs=1e-3)


def test_features_have_no_lookahead():
    df = synthetic("X", n=300)
    f1 = build_features(df)
    f2 = build_features(df.iloc[:200])
    cols = [c for c in f1.columns if c != "y_next"]
    pd.testing.assert_frame_equal(f1[cols].iloc[:200], f2[cols], check_freq=False)


def test_argo_warmup_then_decides():
    df = synthetic("X", n=200)
    f = build_features(df)
    m = ArgoModel("X")
    statuses = []
    for t in f.index:
        m.learn(float(f.at[t, "r1"]))
        statuses.append(m.predict(f.loc[t].to_dict()).status)
    assert statuses[0] == "WARMUP" and statuses[-1] in ("ACTIVE", "PAPER_ONLY")


# ---------- Data + audit + backtest ----------
def test_data_quality_gate_rejects_corrupt_series():
    df = synthetic("X", n=100)
    df.loc[df.index[:10], "high"] = 0
    with pytest.raises(ValueError):
        validate(df)


def test_audit_chain_detects_tampering():
    a = AuditLog()
    a.write("decision", {"x": 1.5})
    a.write("decision", {"x": 2})
    assert a.verify()
    a.records[0]["x"] = 99
    assert not a.verify()


def test_backtest_runs_and_respects_limits():
    data = {f"S{i}": synthetic(f"S{i}", n=400, seed=i) for i in range(2)}
    res = run_backtest(data, BacktestConfig(folds=2))
    assert res["orders"] > 0 and res["audit_records"] >= res["orders"]
    assert res["metrics"]["max_drawdown"] < 0.12 + 0.02
    assert res["avg_gross_exposure"] <= 1.0
