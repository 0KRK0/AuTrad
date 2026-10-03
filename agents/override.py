"""Manual override: you can buy, sell, flatten, pin or pause anything, any time.

Overrides go through the Risk Gate as source="human": they skip the AI's soft brakes but never
the kill switch, stale-data block, gross-exposure cap or the human position cap. Every override
is audited with who, what and why.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

from .taxonomy import Category

ACTIONS = {"TARGET", "BUY", "SELL", "FLATTEN", "PIN", "PAUSE_CATEGORY", "RESUME_CATEGORY"}


@dataclass
class Override:
    action: str
    user: str
    reason: str
    symbol: str | None = None
    category: Category | None = None
    weight: float = 0.0          # TARGET: absolute weight; BUY/SELL: weight to add/remove
    expires_at: float = float("inf")
    id: int = 0
    consumed: bool = False

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action}")
        if self.action in {"PAUSE_CATEGORY", "RESUME_CATEGORY"} and self.category is None:
            raise ValueError("category required")
        if self.action not in {"PAUSE_CATEGORY", "RESUME_CATEGORY"} and not self.symbol:
            raise ValueError("symbol required")


@dataclass
class OverrideBook:
    items: list[Override] = field(default_factory=list)
    paused: set[Category] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def add(self, o: Override) -> Override:
        with self._lock:
            o.id = next(self._ids)
            if o.action == "PAUSE_CATEGORY":
                self.paused.add(o.category)
            elif o.action == "RESUME_CATEGORY":
                self.paused.discard(o.category)
            self.items.append(o)
            return o

    def pending_orders(self, now: float) -> list[Override]:
        """One-shot trade instructions not yet executed."""
        return [o for o in self.items if o.action in {"TARGET", "BUY", "SELL", "FLATTEN"} and not o.consumed and o.expires_at > now]

    def pinned(self, symbol: str, now: float) -> bool:
        """A pinned symbol is left alone by the AI (your manual position is not undone)."""
        return any(o.action == "PIN" and o.symbol == symbol and o.expires_at > now for o in self.items) or any(
            o.symbol == symbol and o.action in {"TARGET", "BUY", "SELL"} and o.consumed and o.expires_at > now for o in self.items
        )

    @staticmethod
    def target_weight(o: Override, current: float) -> float:
        return {"TARGET": o.weight, "BUY": current + abs(o.weight), "SELL": current - abs(o.weight), "FLATTEN": 0.0}[o.action]
