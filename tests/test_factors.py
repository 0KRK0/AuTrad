"""Factor engine: scores never use the future; IC, Newey-West, Holm and tercile construction behave."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from execution.costs import PROFILES
from lab.factors import (
    FactorContext,
    all_scores,
    decision_dates,
    forward_returns,
    holm,
    ic_series,
    newey_west_t,
    passes,
    tercile_targets,
)
from lab.panel import Panel
from marketdata.fundamentals import PITFundamentals, usable_from


def _panel(T: int = 700, N: int = 24, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=T)
    r = rng.normal(0.0004, 0.015, (T, N)) + rng.normal(0, 0.008, (T, 1))
    C = 100 * np.exp(np.cumsum(r, axis=0))
    O = C * np.exp(rng.normal(0, 0.002, (T, N)))  # noqa: E741
    bench = pd.Series(1000 * np.exp(np.cumsum(r.mean(axis=1))), dates)
    return Panel(dates, [f"S{i}" for i in range(N)], O, np.maximum(O, C) * 1.002, np.minimum(O, C) * 0.998, C, np.full_like(C, 1e7),
                 np.ones_like(C, dtype=bool), ["it"] * N, ["largecap"] * N, [PROFILES["nse_largecap"]] * N, bench, "t")


def _facts(P: Panel) -> pd.DataFrame:
    rows = []
    for s in P.symbols:
        for i, pe in enumerate(pd.date_range("2016-12-31", "2019-09-30", freq="QE")):
            rows.append({"symbol": s, "period_end": pe, "available_at": (pe + pd.Timedelta(days=25, hours=12)).tz_localize("Asia/Kolkata"),
                         "revenue": 100 + i + hash(s) % 7, "net_profit": 10 + i % 3 + hash(s) % 5, "eps_basic": 5 + i % 4,
                         "equity": 400.0, "paid_up_capital": 1e8, "face_value": 1.0, "borrowings": 50.0})
    f = pd.DataFrame(rows)
    f["available_at"] = f["available_at"].dt.tz_convert("UTC")
    f["usable_from"] = usable_from(f["available_at"])
    return f


def test_scores_at_t_ignore_any_change_after_t():
    P = _panel()
    raw = pd.DataFrame(P.C, index=P.dates, columns=P.symbols)
    ctx = FactorContext(P, raw, PITFundamentals(_facts(P)), {s: "it" for s in P.symbols})
    dates = decision_dates(P, "2018-01-01")
    t = dates[5]
    a = all_scores(ctx, [t])
    # change every price after t and every filing published after t
    P2 = _panel()
    P2.C[t + 1:] *= 3.0
    P2.O[t + 1:] *= 3.0
    f2 = _facts(P2)
    later = f2["usable_from"] > P.dates[t]
    f2.loc[later, ["net_profit", "eps_basic", "revenue"]] *= 10
    ctx2 = FactorContext(P2, pd.DataFrame(P2.C, index=P2.dates, columns=P2.symbols), PITFundamentals(f2), ctx.sector)
    b = all_scores(ctx2, [t])
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], check_exact=False, rtol=1e-12)


def test_forward_return_runs_from_next_open_to_next_decision_close():
    P = _panel()
    dates = decision_dates(P, "2018-01-01")[:3]
    fwd = forward_returns(P, dates)
    d0, d1 = dates[0], dates[1]
    assert fwd.loc[P.dates[d0], "S0"] == pytest.approx(P.C[d1, 0] / P.O[d0 + 1, 0] - 1)


def test_perfect_signal_has_ic_one_and_noise_is_insignificant():
    P = _panel()
    dates = decision_dates(P, "2018-01-01")
    fwd = forward_returns(P, dates)
    assert ic_series(fwd, fwd).mean() == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns)
    assert abs(newey_west_t(ic_series(noise, fwd).to_numpy())) < 3


def test_holm_is_monotone_and_never_below_raw_p():
    p = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}
    h = holm(p)
    assert h["a"] == pytest.approx(0.004) and h["b"] == pytest.approx(0.06) and h["d"] == pytest.approx(0.5)
    assert all(h[k] >= p[k] for k in p)


def test_tercile_weights_sum_correctly():
    s = pd.DataFrame([np.arange(30, dtype=float)], index=[pd.Timestamp("2020-01-31")], columns=[f"S{i}" for i in range(30)])
    lo = tercile_targets(s).iloc[0]
    ls = tercile_targets(s, long_short=True).iloc[0]
    assert lo.sum() == pytest.approx(1.0) and (lo > 0).sum() == 10 and lo["S29"] > 0
    assert ls.sum() == pytest.approx(0.0) and ls.abs().sum() == pytest.approx(1.0)


def test_negative_ic_never_passes_even_if_significant():
    r = {"mean_ic": -0.05, "ic_first_half": -0.05, "ic_second_half": -0.05, "pct_years_pos": 0.0, "net_excess": 0.01}
    assert not all(passes(r, 0.001).values())
