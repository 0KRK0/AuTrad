"""Point-in-time NIFTY 50 universe: which stocks were actually in the index on a given date.

Source: marketdata/reference/nifty50_pit_snapshots.csv, extracted from the Marketcalls
"NIFTY 50 survivorship-bias-free constituents 2016-2026" workbook (22 semi-annual snapshots,
reconstructed from NSE's current constituent list walked back through NSE Indices press
releases; weights from NSE Indices monthly files). See marketdata/reference/SOURCE.md.

Rules:
- membership on date t = the latest snapshot with effective_date <= t
- before the first snapshot (1 Apr 2016) the first snapshot is used (documented approximation)
- renamed / restructured companies are mapped to the ticker that carries their price history
- companies with no surviving price history are excluded, with the reason recorded
"""
from __future__ import annotations

import bisect
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .universe import Asset

REFERENCE = Path(__file__).parent / "reference" / "nifty50_pit_snapshots.csv"

ALIASES = {
    "INFRATEL": "INDUSTOWER",     # Bharti Infratel merged with Indus Towers (Nov 2020), same listing
    "IBULHSGFIN": "SAMMAANCAP",   # Indiabulls Housing Finance renamed Sammaan Capital
    "LTIM": "LTM",                # LTIMindtree (Yahoo ticker LTM.NS)
    "TATAMOTORS": "TMPV",         # Tata Motors listed entity after the 2025 demerger
    "ZOMATO": "ETERNAL",          # Zomato renamed Eternal
}
EXCLUDED = {
    "HDFC": "merged into HDFC Bank (Jul 2023); delisted price history not available from the data source",
    "TATAMTRDVR": "DVR share class merged into Tata Motors (2024); delisted price history not available",
}

SECTOR = {
    "ACC": "cement", "ADANIENT": "conglomerate", "ADANIPORTS": "infrastructure", "AMBUJACEM": "cement",
    "APOLLOHOSP": "healthcare", "ASIANPAINT": "consumer", "AUROPHARMA": "pharma", "AXISBANK": "banks",
    "BAJAJ-AUTO": "auto", "BAJAJFINSV": "nbfc", "BAJFINANCE": "nbfc", "BANKBARODA": "banks", "BEL": "industrials",
    "BHARTIARTL": "telecom", "BHEL": "industrials", "BOSCHLTD": "auto", "BPCL": "energy", "BRITANNIA": "fmcg",
    "CIPLA": "pharma", "COALINDIA": "metals", "DIVISLAB": "pharma", "DRREDDY": "pharma", "EICHERMOT": "auto",
    "ETERNAL": "consumer_services", "GAIL": "energy", "GRASIM": "cement", "HCLTECH": "it", "HDFCBANK": "banks",
    "HDFCLIFE": "insurance", "HEROMOTOCO": "auto", "HINDALCO": "metals", "HINDPETRO": "energy", "HINDUNILVR": "fmcg",
    "SAMMAANCAP": "nbfc", "ICICIBANK": "banks", "IDEA": "telecom", "INDIGO": "consumer_services", "INDUSINDBK": "banks",
    "INDUSTOWER": "telecom", "INFY": "it", "IOC": "energy", "ITC": "fmcg", "JIOFIN": "nbfc", "JSWSTEEL": "metals",
    "KOTAKBANK": "banks", "LT": "industrials", "LTM": "it", "LUPIN": "pharma", "M&M": "auto", "MARUTI": "auto",
    "MAXHEALTH": "healthcare", "NESTLEIND": "fmcg", "NTPC": "power", "ONGC": "energy", "POWERGRID": "power",
    "RELIANCE": "energy", "BSE": "capital_markets", "SBILIFE": "insurance", "SBIN": "banks", "SHREECEM": "cement", "SHRIRAMFIN": "nbfc",
    "SUNPHARMA": "pharma", "TATACONSUM": "fmcg", "TMPV": "auto", "TATAPOWER": "power", "TATASTEEL": "metals",
    "TCS": "it", "TECHM": "it", "TITAN": "consumer", "TRENT": "consumer", "ULTRACEMCO": "cement", "UPL": "chemicals",
    "VEDL": "metals", "WIPRO": "it", "YESBANK": "banks", "ZEEL": "media",
}


def normalise(symbol: str) -> str:
    return ALIASES.get(symbol, symbol)


class PITUniverse:
    def __init__(self, path: str | Path = REFERENCE) -> None:
        df = pd.read_csv(path, parse_dates=["effective_date"])
        df = df[~df["symbol"].isin(EXCLUDED)]
        df["symbol"] = df["symbol"].map(normalise)
        self.snapshots: dict[pd.Timestamp, frozenset[str]] = {
            d: frozenset(g["symbol"]) for d, g in df.groupby("effective_date")
        }
        self.dates = sorted(self.snapshots)

    def members(self, t: pd.Timestamp) -> frozenset[str]:
        i = bisect.bisect_right(self.dates, t) - 1
        return self.snapshots[self.dates[max(i, 0)]]

    def symbols(self) -> list[str]:
        return sorted(set().union(*self.snapshots.values()))

    def ever_removed(self) -> list[str]:
        return sorted(set(self.symbols()) - self.snapshots[self.dates[-1]])


@lru_cache(maxsize=1)
def default_universe() -> PITUniverse:
    return PITUniverse()


def nifty50_assets() -> list[Asset]:
    return [Asset(s, f"{s}.NS", "equity", "in_largecap", SECTOR.get(s, "unknown"), "nse_largecap")
            for s in default_universe().symbols()]
