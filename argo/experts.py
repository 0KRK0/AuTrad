"""ARGO L1 experts: small predictors that each output an expected next-bar log return."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Expert:
    name = "expert"

    def predict(self, x: np.ndarray) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, x: np.ndarray, y: float) -> None:
        """Called once the realized return for the bar is known."""


@dataclass(kw_only=True)
class RLSExpert(Expert):
    """Recursive Least Squares with exponential forgetting: learns every bar, O(d^2)."""

    name: str
    cols: list[int]
    lam: float = 0.995
    delta: float = 100.0
    clip: float = 0.05
    w: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        d = len(self.cols) + 1
        self.w = np.zeros(d)
        self.P = np.eye(d) * self.delta

    def _x(self, x: np.ndarray) -> np.ndarray:
        return np.append(x[self.cols], 1.0)

    def predict(self, x: np.ndarray) -> float:
        return float(np.clip(self._x(x) @ self.w, -self.clip, self.clip))

    def update(self, x: np.ndarray, y: float) -> None:
        z = self._x(x)
        Pz = self.P @ z
        k = Pz / (self.lam + z @ Pz)
        self.w = self.w + k * (y - z @ self.w)
        self.P = (self.P - np.outer(k, Pz)) / self.lam
        # Numerical guard: keep P symmetric and bounded
        self.P = (self.P + self.P.T) / 2
        if not np.isfinite(self.P).all() or np.trace(self.P) > 1e8:
            self.P = np.eye(len(z)) * self.delta


@dataclass
class MomentumExpert(Expert):
    name: str = "momentum"
    col: int = 2  # r20
    k: float = 1.0

    def predict(self, x: np.ndarray) -> float:
        # Trailing 20-bar mean drift extrapolated one bar ahead
        return float(self.k * x[self.col] / 20)


@dataclass
class MeanReversionExpert(Expert):
    name: str = "mean_reversion"
    col: int = 4  # z20
    vol_col: int = 3
    k: float = 0.25

    def predict(self, x: np.ndarray) -> float:
        return float(-self.k * np.tanh(x[self.col] / 2) * x[self.vol_col])


@dataclass
class TrendExpert(Expert):
    """Extrapolates the 6-month drift (skipping the last week): the classic time-series momentum signal."""

    name: str = "trend_6m"
    col: int = 8

    def predict(self, x: np.ndarray) -> float:
        return float(np.clip(x[self.col] / 121, -0.01, 0.01))


@dataclass
class MAExpert(Expert):
    """Long-trend state: drift of 5% of daily volatility in the direction of the 50/200-day average gap."""

    name: str = "ma_50_200"
    col: int = 9
    vol_col: int = 3
    k: float = 0.05

    def predict(self, x: np.ndarray) -> float:
        return float(self.k * x[self.vol_col] * np.sign(x[self.col]))


@dataclass
class FlatExpert(Expert):
    """'Do nothing' is a first-class action: it wins whenever edge < costs."""

    name: str = "flat"

    def predict(self, x: np.ndarray) -> float:
        return 0.0


def default_experts() -> list[Expert]:
    return [
        RLSExpert(name="rls_trend", cols=[1, 2, 3]),
        RLSExpert(name="rls_full", cols=[0, 1, 2, 3, 4, 5, 6, 7], lam=0.998),
        MomentumExpert(),
        MeanReversionExpert(),
        TrendExpert(),
        MAExpert(),
        FlatExpert(),
    ]
