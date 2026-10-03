"""Statistical tests: is the Sharpe real, or luck and multiple testing?

- Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012): P(true SR > benchmark SR)
  given sample length, skew and kurtosis.
- Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): PSR against the Sharpe you would
  expect from the BEST of N random configurations, i.e. it charges for every config tried.
- Random-trader p-value: share of random traders (same exposure, size, holding period)
  that did at least as well.
- Stationary block bootstrap confidence interval for the annualised Sharpe.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EULER = 0.5772156649


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p: float) -> float:
    # Acklam's approximation, accurate to ~1e-9 on (0, 1)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    lo, hi = 0.02425, 1 - 0.02425
    if p < lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > hi:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def psr(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """sr_benchmark is per-period (not annualised)."""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()
    g3, g4 = float(r.skew()), float(r.kurt()) + 3.0
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return _norm_cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom)


def expected_max_sharpe(trial_sharpes: list[float], n_trials: int) -> float:
    """Expected maximum per-period Sharpe among n_trials unskilled configurations."""
    if n_trials <= 1:
        return 0.0
    var = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sd = math.sqrt(max(var, 1e-12))
    return sd * ((1 - EULER) * _norm_ppf(1 - 1 / n_trials) + EULER * _norm_ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float], n_trials: int) -> float:
    return psr(returns, expected_max_sharpe(trial_sharpes, n_trials))


def random_pvalue(strategy_sharpe: float, random_sharpes: list[float]) -> float:
    if not random_sharpes:
        return float("nan")
    return (1 + sum(s >= strategy_sharpe for s in random_sharpes)) / (1 + len(random_sharpes))


def bootstrap_sharpe_ci(returns: pd.Series, ppy: float, n: int = 1000, block: int = 20,
                        alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    r = returns.dropna().to_numpy()
    if len(r) < 2 * block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    T, out = len(r), []
    for _ in range(n):
        idx = np.empty(T, dtype=int)
        k = 0
        while k < T:
            start = rng.integers(0, T)
            length = min(rng.geometric(1 / block), T - k)
            idx[k:k + length] = (start + np.arange(length)) % T
            k += length
        s = r[idx]
        out.append(s.mean() / s.std() * math.sqrt(ppy) if s.std() > 0 else 0.0)
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))
