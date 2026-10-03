"""Aligned, point-in-time market panel for the research lab.

Every matrix is (dates x symbols). `eligible[t, i]` says whether symbol i could be BOUGHT on the
decision date t: for index stocks that means it was an index member that day (point-in-time),
for ETFs / FX / crypto it means it had a price. Survivors-only panels exist only to measure bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from execution.costs import PROFILES, CostProfile
from marketdata.pit import PITUniverse, default_universe, nifty50_assets
from marketdata.store import DataStore
from marketdata.universe import BY_SYMBOL, Asset


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    symbols: list[str]
    O: np.ndarray  # noqa: E741  (OHLC convention)
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    V: np.ndarray
    eligible: np.ndarray
    sector: list[str]
    klass: list[str]                 # largecap | diversified | fx | crypto | futures | default
    profiles: list[CostProfile]
    bench: pd.Series                 # NIFTY 50 close on this calendar (reference only)
    name: str = "panel"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.value = np.nan_to_num(self.C * self.V)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.C[1:] / self.ffill_close()[:-1] - 1
        self.ret = np.vstack([np.full((1, len(self.symbols)), np.nan), r])
        self.vol20 = pd.DataFrame(self.ret).rolling(20, min_periods=10).std().to_numpy()
        self.idx = {s: i for i, s in enumerate(self.symbols)}

    def ffill_close(self) -> np.ndarray:
        return pd.DataFrame(self.C).ffill().to_numpy()

    def frame(self, m: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(m, index=self.dates, columns=self.symbols)

    def subset(self, symbols: list[str], name: str | None = None) -> Panel:
        ix = [self.idx[s] for s in symbols if s in self.idx]
        pick = lambda m: m[:, ix]  # noqa: E731
        return Panel(self.dates, [self.symbols[i] for i in ix], pick(self.O), pick(self.H), pick(self.L), pick(self.C),
                     pick(self.V), pick(self.eligible), [self.sector[i] for i in ix], [self.klass[i] for i in ix],
                     [self.profiles[i] for i in ix], self.bench, name or self.name, list(self.notes))


def _klass(a: Asset) -> str:
    if a.asset_class == "etf":
        return "diversified"
    if a.asset_class in ("fx", "crypto"):
        return a.asset_class
    return "largecap" if a.category == "in_largecap" else "default"


def build_panel(data_dir: str, assets: list[Asset], pit: PITUniverse | None = None, calendar: str = "nse",
                survivors_only: bool = False, start: str = "2015-01-01", end: str | None = None,
                profile_override: dict[str, str] | None = None, name: str = "panel", asof: str | None = None) -> Panel:
    store = DataStore(data_dir)
    data = store.load([a.symbol for a in assets], start, end, asof=asof)
    q = store.quality
    bench_df = store.load(["NIFTY50"], start, end, asof=asof).get("NIFTY50")
    store.quality = q
    assets = [a for a in assets if a.symbol in data]
    if calendar == "nse" and bench_df is not None:
        dates = bench_df.index
    else:
        dates = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    syms = [a.symbol for a in assets]
    mats = {k: pd.DataFrame({s: data[s][k] for s in syms}).reindex(dates)[syms].to_numpy(dtype=float)
            for k in ["open", "high", "low", "close", "volume"]}
    pit = pit if pit is not None else default_universe()
    pit_syms = set(pit.symbols())
    elig = ~np.isnan(mats["close"])
    notes = []
    if pit_syms & set(syms):
        latest = pit.snapshots[pit.dates[-1]]
        for j, s in enumerate(syms):
            if s in pit_syms:
                if survivors_only:
                    mask = np.full(len(dates), s in latest)
                else:
                    mask = np.array([s in pit.members(t) for t in dates])
                elig[:, j] &= mask
        notes.append("survivors-only universe (bias test)" if survivors_only else "point-in-time NIFTY 50 membership")
    # data quality: a day whose price was not confirmed by the exchange (score < 0.5) cannot be a buy decision
    qm = pd.DataFrame({s: store.quality.get(s, pd.Series(dtype=float)) for s in syms}).reindex(dates)[syms].to_numpy(dtype=float)
    low = np.nan_to_num(qm, nan=1.0) < 0.5
    if low.any():
        elig &= ~low
        notes.append(f"{int(low.sum())} asset-days blocked by the data-quality score (< 0.5)")
    notes.append(f"exchange cross-check {store.exchange_hash()}")
    override = profile_override or {}
    profiles = [PROFILES[override.get(a.symbol, a.cost_profile)] for a in assets]
    klass = ["futures" if not p.cash_settled and a.asset_class != "fx" else _klass(a) for a, p in zip(assets, profiles, strict=True)]
    bench = bench_df["close"].reindex(dates).ffill() if bench_df is not None else pd.Series(np.nan, index=dates)
    return Panel(dates, syms, mats["open"], mats["high"], mats["low"], mats["close"], mats["volume"], elig,
                 [a.sector for a in assets], klass, profiles, bench, name, notes)


def india_equity_assets() -> list[Asset]:
    """PIT NIFTY 50 stocks + the NIFTY ETF (diversified core) + gold ETF (defensive)."""
    return [*nifty50_assets(), BY_SYMBOL["NIFTYBEES"], BY_SYMBOL["GOLDBEES"]]


def cross_asset_assets() -> list[Asset]:
    return [BY_SYMBOL[s] for s in ["NIFTYBEES", "BANKBEES", "GOLDBEES", "SILVERBEES", "USDINR", "BTCINR"]]
