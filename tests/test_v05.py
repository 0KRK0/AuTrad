"""v0.5 research tools: regression maths, past-only flags, Brinson identities, sector-neutral weights, scorecard."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab import v05 as V
from tests.test_factors import _panel


def test_nw_ols_recovers_coefficients_and_matches_ols_point_estimates():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    y = 0.01 + X @ np.array([0.8, -0.3]) + rng.normal(0, 0.05, 400)
    r = V.nw_ols(y, X)
    ols, *_ = np.linalg.lstsq(np.column_stack([np.ones(400), X]), y, rcond=None)
    assert np.allclose(r["beta"], ols)
    assert abs(r["beta"][1] - 0.8) < 0.02 and r["t"][1] > 50


def test_nw_inflates_se_under_autocorrelation():
    rng = np.random.default_rng(2)
    e = np.zeros(600)
    for i in range(1, 600):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    a = V.nw_ols(e, np.zeros((600, 0)), lags=0)
    b = V.nw_ols(e, np.zeros((600, 0)), lags=6)
    assert b["se"][0] > 1.5 * a["se"][0]


def test_cluster_ols_counts_clusters_not_rows():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(30), 20)                      # 30 days x 20 identical-shock events
    shock = rng.normal(0, 1, 30)[g]
    d = (np.arange(600) % 2).astype(float)
    y = shock + rng.normal(0, 0.1, 600)
    r = V.cluster_ols(y, d, g)
    assert r["clusters"] == 30
    naive = V.nw_ols(y, d, lags=0)
    assert r["se"][0] > 3 * naive["se"][0]               # the intercept's uncertainty is driven by 30 day-shocks


def test_past_only_flag_never_uses_later_values():
    order = pd.Series(np.arange(500, dtype=float))
    v = pd.Series(np.random.default_rng(4).normal(size=500))
    f1 = V.past_only_flag(v, order, 1 / 3)
    v2 = v.copy()
    v2.iloc[400:] = 100.0                                  # change the future
    f2 = V.past_only_flag(v2, order, 1 / 3)
    assert f1.iloc[:400].equals(f2.iloc[:400])                # flags before the change are unaffected
    assert (f2.iloc[401:] == 0).all()                       # and each later value is judged only on earlier ones
    assert f1.iloc[:200].isna().all()


def test_brinson_allocation_plus_selection_equals_excess():
    P = _panel(T=500, N=12)
    P.sector = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    dates = [P.dates[i] for i in range(300, 480, 21)]
    rng = np.random.default_rng(5)
    tg = pd.DataFrame([pd.Series(rng.dirichlet(np.ones(12)) * 0.9, index=P.symbols) for _ in dates], index=dates)
    U = pd.DataFrame([pd.Series(1 / 12, index=P.symbols) for _ in dates], index=dates)
    br = V.brinson(P, tg, U)
    assert np.allclose(br["allocation"] + br["selection"], br["excess"], atol=1e-12)


def test_sector_neutral_matches_universe_sector_weights():
    P = _panel(T=600, N=15)
    P.sector = ["a"] * 6 + ["b"] * 6 + ["c"] * 3
    tg = V.sector_neutral_low_vol(P)
    sec = pd.Series(P.sector, index=P.symbols)
    for _, w in tg.iterrows():
        sw = w.groupby(sec).sum()
        assert np.allclose(sw.to_numpy(), [6 / 15, 6 / 15, 3 / 15])
        assert ((w > 0).groupby(sec).sum().to_numpy() == [2, 2, 1]).all()


def test_scorecard_keeps_axes_separate():
    s = V.score_axes(stat_ok=False, stat_weak=False, econ_value=0.03, econ_kind="portfolio",
                     tradable={"net_positive": True, "cost_2x": True}, forward="not started")
    assert s == {"statistical": "✗", "economic": "✓", "tradable": "✓", "forward": "not started"}
    s = V.score_axes(stat_ok=True, stat_weak=True, econ_value=-0.01, econ_kind="portfolio", tradable={"net_positive": False}, forward="running")
    assert s["statistical"] == "✓" and s["economic"] == "✗" and s["tradable"] == "✗"


def test_v06_replace_and_skip_variants():
    from scripts.v06 import variants

    P = _panel(T=300, N=30)
    d = P.dates[250]
    S = pd.DataFrame([pd.Series(np.arange(30, 0, -1, dtype=float), index=P.symbols)], index=[d])   # S0 best ... S29 worst
    ev = pd.DataFrame({"symbol": ["S1", "S4", "S20"], "reaction_idx": [245, 200, 249], "bad": [1.0, 1.0, 1.0]})
    rep, skp, nveto, flags = variants(S, P, ev, 20)
    top = [f"S{i}" for i in range(10)]
    assert nveto.iloc[0] == 1                                       # S1 vetoed; S4's event is older than 20 sessions; S20 not selected
    assert sorted(rep.iloc[0][rep.iloc[0] > 0].index) == sorted([s for s in top if s != "S1"] + ["S10"])
    assert sorted(skp.iloc[0][skp.iloc[0] > 0].index) == sorted(s for s in top if s != "S1")
    assert abs(skp.iloc[0].sum() - 1) < 1e-12 and abs(rep.iloc[0].sum() - 1) < 1e-12
    assert flags.loc[d, "S20"] == 1.0 and flags.loc[d, "S4"] == 0.0
