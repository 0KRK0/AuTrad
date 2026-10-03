"""AuTrad India validation universe.

Every tradable instrument carries its asset class, AuTrad category, sector and cost profile.
Benchmarks are used for comparison only and are never traded.

Survivorship bias warning: equity names are chosen from today's liquid universe, so companies
that were delisted or fell out of the index since 2015 are missing. This flatters every
long-only strategy (ARGO and baselines alike) and must be remembered when reading results.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Asset:
    symbol: str            # AuTrad symbol
    ticker: str            # data-source ticker (Yahoo)
    asset_class: str       # equity | etf | crypto | fx | index
    category: str          # AuTrad category (agents.taxonomy.Category value)
    sector: str
    cost_profile: str      # key into execution.costs.PROFILES
    tags: frozenset[str] = field(default_factory=frozenset)


def _eq(sym: str, cat: str, sector: str, profile: str) -> Asset:
    return Asset(sym, f"{sym}.NS", "equity", cat, sector, profile)


LARGE = [
    ("RELIANCE", "energy"), ("TCS", "it"), ("HDFCBANK", "banks"), ("ICICIBANK", "banks"), ("INFY", "it"),
    ("ITC", "fmcg"), ("LT", "industrials"), ("SBIN", "banks"), ("BHARTIARTL", "telecom"), ("HINDUNILVR", "fmcg"),
    ("KOTAKBANK", "banks"), ("AXISBANK", "banks"), ("BAJFINANCE", "nbfc"), ("MARUTI", "auto"), ("SUNPHARMA", "pharma"),
]
MID = [
    ("FEDERALBNK", "banks"), ("ASHOKLEY", "auto"), ("VOLTAS", "consumer_durables"), ("MPHASIS", "it"),
    ("CUMMINSIND", "industrials"), ("INDHOTEL", "hotels"), ("BHARATFORG", "auto"), ("PAGEIND", "textiles"),
]
SMALL = [("NCC", "construction"), ("CESC", "utilities"), ("CANFINHOME", "nbfc"), ("KEC", "industrials")]

UNIVERSE: list[Asset] = [
    *[_eq(s, "in_largecap", sec, "nse_largecap") for s, sec in LARGE],
    *[_eq(s, "in_midsmall", sec, "nse_midcap") for s, sec in MID],
    *[_eq(s, "in_midsmall", sec, "nse_smallcap") for s, sec in SMALL],
    Asset("NIFTYBEES", "NIFTYBEES.NS", "etf", "in_index_fno", "index", "nse_etf"),
    Asset("BANKBEES", "BANKBEES.NS", "etf", "in_index_fno", "banks", "nse_etf"),
    Asset("GOLDBEES", "GOLDBEES.NS", "etf", "commodity", "gold", "nse_etf"),
    Asset("SILVERBEES", "SILVERBEES.NS", "etf", "commodity", "silver", "nse_etf"),
    Asset("BTCINR", "BTC-INR", "crypto", "crypto_major", "crypto", "crypto_inr"),
    Asset("ETHINR", "ETH-INR", "crypto", "crypto_major", "crypto", "crypto_inr"),
    Asset("USDINR", "INR=X", "fx", "fx_inr", "fx", "nse_currency_fut"),
    Asset("GBPINR", "GBPINR=X", "fx", "fx_inr", "fx", "nse_currency_fut"),
    Asset("EURINR", "EURINR=X", "fx", "fx_inr", "fx", "nse_currency_fut"),
]

BENCHMARKS: list[Asset] = [
    Asset("NIFTY50", "^NSEI", "index", "benchmark", "index", "none"),
    Asset("NIFTYMID50", "^NSEMDCP50", "index", "benchmark", "index", "none"),
]

BY_SYMBOL = {a.symbol: a for a in [*UNIVERSE, *BENCHMARKS]}


def select(groups: list[str] | None = None) -> list[Asset]:
    """groups: any of equity, etf, crypto, fx (default: all tradable)."""
    return [a for a in UNIVERSE if groups is None or a.asset_class in groups]
