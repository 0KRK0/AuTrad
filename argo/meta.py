"""ARGO L3 regime gate, L4 Thompson-sampling allocator, L7 drift sentinel."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REGIMES = ("trend_calm", "trend_volatile", "range_calm", "range_volatile")


def classify_regime(ema_slope: float, vol20: float, vol_long: float, vol20_now_vs_long: float = 1.25) -> int:
    """Deterministic, point-in-time 4-way regime from trend strength x volatility state."""
    if not np.isfinite(ema_slope) or not np.isfinite(vol20) or not np.isfinite(vol_long) or vol_long <= 0:
        return 2
    trending = abs(ema_slope) > 1.0 * vol_long * np.sqrt(10) * 0.5
    volatile = vol20 > vol20_now_vs_long * vol_long
    return (0 if trending else 2) + (1 if volatile else 0)


@dataclass
class ThompsonAllocator:
    """Per-regime Gaussian Thompson sampling over experts.

    Reward = expert's counterfactual next-bar PnL after costs, divided by volatility.
    Every expert is scored every bar (full-information), so learning is fast.
    """

    n_experts: int
    n_regimes: int = len(REGIMES)
    decay: float = 0.99
    samples: int = 64
    seed: int = 7
    s: np.ndarray = field(init=False)   # discounted reward sum
    s2: np.ndarray = field(init=False)  # discounted squared reward sum
    n: np.ndarray = field(init=False)   # discounted count

    def __post_init__(self) -> None:
        shape = (self.n_regimes, self.n_experts)
        self.s = np.zeros(shape)
        self.s2 = np.zeros(shape)
        self.n = np.zeros(shape)
        self.rng = np.random.default_rng(self.seed)

    def weights(self, regime: int) -> np.ndarray:
        n = self.n[regime] + 1.0
        mean = self.s[regime] / n
        var = np.maximum(self.s2[regime] / n - mean**2, 1e-4) / n + 1e-4 / n
        draws = self.rng.normal(mean, np.sqrt(var), size=(self.samples, self.n_experts))
        wins = np.bincount(draws.argmax(axis=1), minlength=self.n_experts)
        return wins / wins.sum()

    def update(self, regime: int, rewards: np.ndarray) -> None:
        self.s[regime] = self.decay * self.s[regime] + rewards
        self.s2[regime] = self.decay * self.s2[regime] + rewards**2
        self.n[regime] = self.decay * self.n[regime] + 1.0


@dataclass
class PageHinkley:
    """Detects an upward shift in a monitored error stream (model broke / world changed)."""

    delta: float = 0.005
    threshold: float = 0.5
    alpha: float = 0.999
    mean: float = 0.0
    cum: float = 0.0
    min_cum: float = 0.0
    t: int = 0

    def update(self, value: float) -> bool:
        self.t += 1
        self.mean = self.mean + (value - self.mean) / min(self.t, 1000)
        self.cum = self.alpha * self.cum + (value - self.mean - self.delta)
        self.min_cum = min(self.min_cum, self.cum)
        if self.cum - self.min_cum > self.threshold:
            self.cum = self.min_cum = 0.0
            return True
        return False
