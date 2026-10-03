"""Broker abstraction + realistic PaperBroker (commission, spread, slippage, impact, partial fills)."""
from __future__ import annotations

import itertools
import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .costs import CostProfile
from .costs import impact as impact_of


@dataclass
class Order:
    symbol: str
    qty: float                       # signed: + buy, - sell
    type: str = "MARKET"             # MARKET | LIMIT
    limit_price: float | None = None
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # idempotency key
    id: int = 0
    status: str = "NEW"              # NEW | PARTIAL | FILLED | CANCELLED | REJECTED
    filled_qty: float = 0.0
    avg_price: float = 0.0


@dataclass
class Fill:
    order_id: int
    symbol: str
    qty: float
    price: float
    commission: float          # all fees (statutory + brokerage + flat), Rs
    slippage: float = 0.0      # spread + impact paid vs the bar open, Rs


class BrokerInterface(ABC):
    @abstractmethod
    def place_order(self, order: Order) -> Order: ...
    @abstractmethod
    def cancel_order(self, order_id: int) -> bool: ...
    @abstractmethod
    def get_positions(self) -> dict[str, float]: ...
    @abstractmethod
    def get_balance(self) -> float: ...
    @abstractmethod
    def get_orders(self) -> list[Order]: ...


@dataclass
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 1.5
    impact_coef: float = 0.1          # sqrt-impact: coef * sigma * sqrt(qty / adv)
    max_participation: float = 0.05   # cap fill at 5% of bar volume -> partial fills


class PaperBroker(BrokerInterface):
    """Fills orders against the NEXT bar (latency = 1 bar), never the signal bar.

    `profiles` maps symbol -> CostProfile (India charges); symbols without a profile use `costs`.
    Open orders are indexed per symbol, so matching is O(open orders), not O(all orders ever).
    """

    def __init__(self, cash: float, costs: CostModel | None = None, profiles: dict[str, CostProfile] | None = None) -> None:
        self.cash = cash
        self.costs = costs or CostModel()
        self.profiles = profiles or {}
        self.positions: dict[str, float] = {}
        self.orders: dict[int, Order] = {}
        self._open: dict[str, dict[int, Order]] = {}
        self._by_client: dict[str, int] = {}
        self._ids = itertools.count(1)
        self.fills: list[Fill] = []
        self.fees_paid = 0.0
        self.slippage_paid = 0.0

    def place_order(self, order: Order) -> Order:
        if order.client_id in self._by_client:  # idempotent retry
            return self.orders[self._by_client[order.client_id]]
        order.id = next(self._ids)
        self.orders[order.id] = order
        self._open.setdefault(order.symbol, {})[order.id] = order
        self._by_client[order.client_id] = order.id
        return order

    def _close(self, o: Order) -> None:
        self._open.get(o.symbol, {}).pop(o.id, None)

    def cancel_order(self, order_id: int) -> bool:
        o = self.orders.get(order_id)
        if o and o.status in ("NEW", "PARTIAL"):
            o.status = "CANCELLED"
            self._close(o)
            return True
        return False

    def cancel_all(self) -> int:
        ids = [i for book in self._open.values() for i in book]
        return sum(self.cancel_order(i) for i in ids)

    def open_orders(self) -> list[Order]:
        return [o for book in self._open.values() for o in book.values()]

    def get_positions(self) -> dict[str, float]:
        return {k: v for k, v in self.positions.items() if abs(v) > 1e-9}

    def get_balance(self) -> float:
        return self.cash

    def get_orders(self) -> list[Order]:
        return list(self.orders.values())

    def on_bar(self, symbol: str, open_: float, high: float, low: float, volume: float, sigma: float) -> list[Fill]:
        """Match open orders for `symbol` against this bar."""
        out: list[Fill] = []
        book = self._open.get(symbol)
        if not book:
            return out
        prof = self.profiles.get(symbol)
        c = self.costs
        max_part = prof.max_participation if prof else c.max_participation
        for o in list(book.values()):
            remaining = o.qty - o.filled_qty
            if volume > 0 and max_part > 0:
                cap = max_part * volume
                qty = math.copysign(min(abs(remaining), cap), remaining)
            elif volume <= 0 and prof is not None:
                qty = remaining          # no volume data (e.g. FX): unlimited liquidity, spread still paid
            else:
                qty = 0.0
            if qty == 0:
                continue
            side = 1 if qty > 0 else -1
            if prof is not None:
                slip = prof.half_spread_bps / 1e4 + impact_of(prof, qty, volume, sigma)
            else:
                slip = c.half_spread_bps / 1e4 + c.impact_coef * sigma * math.sqrt(abs(qty) / max(volume, 1.0))
            px = open_ * (1 + side * slip)
            if o.type == "LIMIT" and o.limit_price is not None:
                if (side > 0 and low > o.limit_price) or (side < 0 and high < o.limit_price):
                    continue
                px = min(px, o.limit_price) if side > 0 else max(px, o.limit_price)
            fee = prof.fees(side, qty * px) if prof is not None else abs(qty * px) * c.commission_bps / 1e4
            slippage = abs(qty) * abs(px - open_)
            self.cash -= qty * px + fee
            self.fees_paid += fee
            self.slippage_paid += slippage
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
            prev = o.filled_qty
            o.filled_qty += qty
            o.avg_price = (o.avg_price * abs(prev) + px * abs(qty)) / abs(o.filled_qty)
            o.status = "FILLED" if abs(o.filled_qty - o.qty) < 1e-9 else "PARTIAL"
            if o.status == "FILLED":
                self._close(o)
            f = Fill(o.id, symbol, qty, px, fee, slippage)
            self.fills.append(f)
            out.append(f)
        return out


# Live adapters implement the same interface. Stubs kept explicit so Live stays disabled until built + approved.
class AlpacaAdapter(BrokerInterface):  # pragma: no cover
    def __init__(self, *_, **__):
        raise NotImplementedError("Live broker disabled: implement + pass promotion gates first")

    place_order = cancel_order = get_positions = get_balance = get_orders = None  # type: ignore
