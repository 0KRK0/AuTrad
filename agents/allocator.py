"""Category capital allocator: ARGO's portfolio-level brain.

Each category's budget = policy max x multiplier, multiplier in [floor, 1].
The multiplier is the posterior probability that the category's recent return on exposure
is positive (discounted Gaussian estimate). Categories that keep losing shrink toward the
floor; categories that keep earning grow back to their policy cap, never above it.

Speculative categories (penny, alt, meme) are additionally bound by the fixed speculative
sleeve in DecisionAgent, so one lucky meme trade can never pull the portfolio into memes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class CategoryAllocator:
    decay: float = 0.98          # ~50-bar memory
    floor: float = 0.25
    min_obs: float = 20.0        # below this effective count the multiplier stays at 1 (no evidence yet)
    s: dict[str, float] = field(default_factory=dict)
    s2: dict[str, float] = field(default_factory=dict)
    n: dict[str, float] = field(default_factory=dict)

    def update(self, category: str, ret_on_exposure: float) -> None:
        d = self.decay
        self.s[category] = d * self.s.get(category, 0.0) + ret_on_exposure
        self.s2[category] = d * self.s2.get(category, 0.0) + ret_on_exposure**2
        self.n[category] = d * self.n.get(category, 0.0) + 1.0

    def multiplier(self, category: str) -> float:
        n = self.n.get(category, 0.0)
        if n < self.min_obs:
            return 1.0
        mean = self.s[category] / n
        var = max(self.s2[category] / n - mean**2, 1e-12)
        z = mean / math.sqrt(var / n)
        return self.floor + (1 - self.floor) * _phi(z)

    def snapshot(self) -> dict[str, float]:
        return {c: round(self.multiplier(c), 3) for c in sorted(self.n)}
