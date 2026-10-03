"""Asset taxonomy: every instrument maps to exactly one category, and every category gets its own agent + policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    IN_LARGECAP = "in_largecap"        # NIFTY 100-type names
    IN_MIDSMALL = "in_midsmall"
    IN_PENNY = "in_penny"              # low price / illiquid / under surveillance: speculative
    IN_INDEX_FNO = "in_index_fno"      # NIFTY / BANKNIFTY futures & options
    CRYPTO_MAJOR = "crypto_major"      # BTC, ETH
    CRYPTO_ALT = "crypto_alt"
    CRYPTO_MEME = "crypto_meme"        # DOGE, SHIB, PEPE...: speculative, social-media driven
    FX_INR = "fx_inr"                  # USDINR, GBPINR, EURINR, JPYINR (NSE currency derivatives)
    COMMODITY = "commodity"            # MCX gold, silver, crude


SPECULATIVE = {Category.IN_PENNY, Category.CRYPTO_ALT, Category.CRYPTO_MEME}
CRYPTO_MAJORS = {"BTC", "ETH"}
MEME_COINS = {"DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "TRUMP"}


@dataclass
class Instrument:
    symbol: str
    asset_class: str                    # equity | index | crypto | fx | commodity
    price: float = 0.0
    adv_value: float = 0.0              # average daily traded value, INR
    market_cap: float = 0.0             # INR
    surveillance: bool = False          # on NSE/BSE ASM/GSM list
    tags: set[str] = field(default_factory=set)
    category: str | None = None         # explicit category (e.g. from marketdata.universe) wins
    sector: str = "UNKNOWN"


def base_asset(symbol: str) -> str:
    for quote in ("USDT", "INR", "USDC", "USD"):
        if symbol.upper().endswith(quote) and len(symbol) > len(quote):
            return symbol.upper()[: -len(quote)]
    return symbol.upper()


def classify(inst: Instrument) -> Category:
    if inst.category:
        return Category(inst.category)
    ac = inst.asset_class
    if ac == "crypto":
        b = base_asset(inst.symbol)
        if b in CRYPTO_MAJORS:
            return Category.CRYPTO_MAJOR
        if b in MEME_COINS or "meme" in inst.tags:
            return Category.CRYPTO_MEME
        return Category.CRYPTO_ALT
    if ac == "fx":
        return Category.FX_INR
    if ac == "commodity":
        return Category.COMMODITY
    if ac == "index":
        return Category.IN_INDEX_FNO
    # Indian equity
    if inst.surveillance or inst.price < 20 or inst.adv_value < 1e7 or inst.market_cap < 5e9:
        return Category.IN_PENNY
    if inst.market_cap >= 2e12 or "nifty100" in inst.tags:
        return Category.IN_LARGECAP
    return Category.IN_MIDSMALL


@dataclass(frozen=True)
class CategoryPolicy:
    style: str                   # argo (learned) | event_momentum (ride confirmed shocks, exit fast)
    max_category: float          # total weight this category may hold
    max_position: float          # per symbol
    stop_loss: float             # exit if price falls this far below entry
    trailing_stop: float         # exit if price falls this far below the high since entry
    time_stop_bars: int          # exit after N bars regardless (speculation must not become investment)
    max_adv_participation: float = 0.02   # order <= 2% of daily traded value: you can always get out
    allow_short: bool = False
    allow_unexplained: bool = False       # trade price spikes with NO news behind them? (pump risk)
    event_sensitivity: float = 1.0


POLICIES: dict[Category, CategoryPolicy] = {
    Category.IN_LARGECAP:  CategoryPolicy("argo", 0.40, 0.05, 0.08, 0.12, 60),
    Category.IN_MIDSMALL:  CategoryPolicy("argo", 0.25, 0.03, 0.10, 0.15, 40),
    Category.IN_PENNY:     CategoryPolicy("event_momentum", 0.03, 0.005, 0.08, 0.10, 5, 0.01),
    Category.IN_INDEX_FNO: CategoryPolicy("argo", 0.20, 0.05, 0.03, 0.05, 5, allow_short=True),
    Category.CRYPTO_MAJOR: CategoryPolicy("argo", 0.15, 0.05, 0.10, 0.15, 30),
    Category.CRYPTO_ALT:   CategoryPolicy("event_momentum", 0.05, 0.01, 0.12, 0.15, 5),
    Category.CRYPTO_MEME:  CategoryPolicy("event_momentum", 0.02, 0.005, 0.15, 0.20, 3, event_sensitivity=2.0),
    Category.FX_INR:       CategoryPolicy("argo", 0.10, 0.05, 0.01, 0.015, 10, allow_short=True, event_sensitivity=1.5),
    Category.COMMODITY:    CategoryPolicy("argo", 0.10, 0.03, 0.05, 0.08, 20, allow_short=True),
}

SPECULATIVE_SLEEVE = 0.05  # all speculative categories together: max 5% of equity
