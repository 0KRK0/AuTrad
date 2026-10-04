import math

import numpy as np

from lab import aem


def _v():
    return aem.venues()


def test_positive_part_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for mu, s in [(0.0, 10.0), (3.0, 50.0), (-5.0, 20.0), (20.0, 5.0)]:
        x = rng.normal(mu, s, 400_000)
        assert abs(aem.expected_positive_part(mu, s) - np.maximum(x, 0).mean()) < 0.15


def test_microstructure_signal_is_abandoned_at_crypto_fees():
    # with the measured calibration (fill prob ~0.8, adverse ~0.9 bp) and with an optimistic one (front of queue)
    v = _v()
    v = {**v, "crypto_best_case_optimistic": aem.with_overrides(v["crypto_best_case"], passive_fill_prob=0.9, passive_adverse_bp=0.1)}
    # v0.9: |5 s move| ~ 0.85 bp; best prediction ~0.5 bp; per-trade sigma ~1.5 bp
    s = aem.Signal(gross_bp=0.85, sigma_bp=1.5, horizon_days=5 / 86400, latency_s=0.25, half_life_s=5)
    for name in ("crypto_retail", "crypto_best_case", "crypto_best_case_optimistic"):
        r = aem.evaluate(s, v[name])
        assert r["decision"] in ("ABANDON",), r
        assert r["net_cross_bp"] < 0


def test_slow_event_signal_executes_on_nse():
    v = _v()
    s = aem.Signal(gross_bp=150.0, sigma_bp=300.0, horizon_days=5, qty_over_adv=0.001, daily_vol_bp=150)
    r = aem.evaluate(s, v["nse_delivery"])
    assert r["decision"] == "EXECUTE"
    assert 0 < r["net_cross_bp"] < 150


def test_no_setoff_tax_kills_noisy_small_edges():
    v = _v()["crypto_best_case"]
    quiet = aem.Signal(gross_bp=20.0, sigma_bp=1.0, horizon_days=1, cash_rate=0.0)
    noisy = aem.Signal(gross_bp=20.0, sigma_bp=300.0, horizon_days=1, cash_rate=0.0)
    rq, rn = aem.evaluate(quiet, v), aem.evaluate(noisy, v)
    assert rq["net_cross_bp"] > 0 > rn["net_cross_bp"]
    # with set-off the same noisy trade would be positive
    rs = aem.evaluate(noisy, aem.with_overrides(v, tax_setoff=True))
    assert rs["net_cross_bp"] > 0


def test_wait_when_only_passive_route_is_positive():
    v = aem.with_overrides(_v()["crypto_retail"], passive_fill_prob=0.8, passive_adverse_bp=0.5, tax_rate=0.0)
    s = aem.Signal(gross_bp=10.0, sigma_bp=1.0, horizon_days=0.0, cash_rate=0.0)
    r = aem.evaluate(s, v)
    assert r["net_cross_bp"] <= 0 and r["ev_passive_bp"] > 0
    assert r["decision"] == "WAIT" and r["route"] == "passive"


def test_costs_monotone_and_latency_decay():
    v = _v()["nse_index_fut"]
    a = aem.evaluate(aem.Signal(30.0, 10.0, 1), v)["net_cross_bp"]
    b = aem.evaluate(aem.Signal(30.0, 10.0, 1), aem.with_overrides(v, taker_bp=v.taker_bp * 2))["net_cross_bp"]
    assert b < a
    s = aem.Signal(30.0, 10.0, 1, latency_s=5, half_life_s=5)
    r = aem.evaluate(s, v)
    assert math.isclose(r["components_bp"]["latency_drift"], 15.0)


def test_calibration_loads_from_absolute_path():
    assert aem.CALIBRATION.is_absolute()
    if aem.CALIBRATION.exists():
        v = aem.venues()["crypto_retail"]
        assert 0 < v.passive_fill_prob <= 1 and v.passive_adverse_bp > 0
