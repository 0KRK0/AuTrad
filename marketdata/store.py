"""Versioned market-data cache: fetch -> validate -> store CSV -> manifest with SHA-256.

Every backtest records the manifest hash it ran on, so any result can be reproduced exactly.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .universe import Asset

COLS = ["open", "high", "low", "close", "volume"]
DEFAULT_DIR = Path("data")


@dataclass
class QualityReport:
    rows: int
    first: str
    last: str
    duplicates: int
    non_positive: int
    ohlc_repaired: int
    stale_runs: int          # runs of >=5 identical closes
    extreme_moves: int       # |daily return| > 25% (flagged, not removed)
    max_gap_days: int
    passed: bool
    notes: list[str]


def clean(df: pd.DataFrame, asset_class: str) -> tuple[pd.DataFrame, QualityReport]:
    notes: list[str] = []
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) else pd.to_datetime(df.index)
    df.index = df.index.normalize()
    df = df.sort_index()
    dup = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="last")]
    df = df[COLS].astype(float)
    df = df.dropna(subset=["open", "high", "low", "close"])
    nonpos = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    hi = df[["open", "high", "close"]].max(axis=1)
    lo = df[["open", "low", "close"]].min(axis=1)
    repaired = int(((df["high"] < hi) | (df["low"] > lo)).sum())
    # Vendors often print a close a hair outside the day's range (rounding); only a breach
    # larger than 0.5% of price is treated as a corrupt bar.
    breach = np.maximum((hi - df["high"]) / df["close"], (df["low"] - lo) / df["close"])
    corrupt = int((breach > 0.005).sum())
    df["high"], df["low"] = hi, lo
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    c = df["close"]
    same = (c.diff() == 0).astype(int)
    runs = same.groupby((same == 0).cumsum()).sum()
    stale = int((runs >= 5).sum())
    r = c.pct_change().abs()
    extreme = int((r > 0.25).sum())
    gaps = df.index.to_series().diff().dt.days.fillna(0)
    max_gap = int(gaps.max()) if len(gaps) else 0
    if asset_class == "fx" and (df["volume"] == 0).all():
        notes.append("source has no FX volume; broker treats liquidity as unlimited, spread still charged")
    passed = len(df) >= 250 and (nonpos + corrupt) / max(len(df), 1) < 0.01
    if corrupt:
        notes.append(f"{corrupt} bars with OHLC breach > 0.5%")
    if max_gap > 10:
        notes.append(f"max gap {max_gap} days")
    if stale > 3 and asset_class in ("equity", "etf", "crypto"):
        notes.append(f"{stale} stale-price runs")
    q = QualityReport(len(df), str(df.index.min().date()) if len(df) else "", str(df.index.max().date()) if len(df) else "",
                      dup, nonpos, repaired, stale, extreme, max_gap, passed, notes)
    return df, q


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_yahoo(tickers: list[str], start: str, end: str | None, retries: int = 3) -> dict[str, pd.DataFrame]:  # pragma: no cover - network
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    pending = list(tickers)
    for attempt in range(retries):
        if not pending:
            break
        raw = yf.download(pending, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True)
        still = []
        for t in pending:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                d = d.rename(columns=str.lower).dropna(how="all")
                if len(d) == 0:
                    still.append(t)
                    continue
                out[t] = d
            except KeyError:
                still.append(t)
        pending = still
        if pending:
            time.sleep(2 ** (attempt + 1))  # backoff before retrying only the failures
    return out


class DataStore:
    def __init__(self, root: str | Path = DEFAULT_DIR, exchange_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.exchange_root = Path(exchange_root) if exchange_root else None
        self.raw = self.root / "raw"
        self.manifest_path = self.root / "manifest.json"

    def manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {"assets": {}}

    def manifest_hash(self) -> str:
        m = self.manifest()
        body = json.dumps({k: v["sha256"] for k, v in sorted(m.get("assets", {}).items())}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def save(self, asset: Asset, df: pd.DataFrame, source: str) -> QualityReport:
        self.raw.mkdir(parents=True, exist_ok=True)
        clean_df, q = clean(df, asset.asset_class)
        path = self.raw / f"{asset.symbol}.csv"
        clean_df.to_csv(path, index_label="date", float_format="%.6f")
        m = self.manifest()
        m["assets"][asset.symbol] = {
            "ticker": asset.ticker, "source": source, "asset_class": asset.asset_class,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": _sha(path), "quality": asdict(q),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(m, indent=2))
        return q

    def refresh(self, assets: list[Asset], start: str = "2015-01-01", end: str | None = None) -> dict[str, QualityReport]:  # pragma: no cover
        got = fetch_yahoo([a.ticker for a in assets], start, end)
        reports = {}
        for a in assets:
            if a.ticker in got:
                reports[a.symbol] = self.save(a, got[a.ticker], "yahoo")
        return reports

    def load(self, symbols: list[str], start: str | None = None, end: str | None = None,
             adjust: bool = True) -> dict[str, pd.DataFrame]:
        """adjust=True applies the exchange-anchored corporate-action correction (marketdata.corporate_actions)
        for every asset with a BSE listing when the exchange cache exists (default <root>/../exchange), else
        only the vendor blip repair. Raw files are never modified. After a load:
          self.adjustments[symbol] -> human-readable notes,  self.events[symbol] -> applied corporate actions,
          self.quality[symbol] -> per-date data-quality score (1 = confirmed by the exchange that day)."""
        from .corporate_actions import apply as apply_blips
        from .corporate_actions import correct_with_exchange, load_registry

        m = self.manifest().get("assets", {})
        out: dict[str, pd.DataFrame] = {}
        self.adjustments: dict[str, list[str]] = {}
        self.events: dict[str, list] = {}
        self.quality: dict[str, pd.Series] = {}
        frames = {}
        for s in symbols:
            path = self.raw / f"{s}.csv"
            if not path.exists():
                continue
            if s in m and _sha(path) != m[s]["sha256"]:
                raise ValueError(f"{s}: file changed since fetch (sha256 mismatch); re-run fetch_data")
            frames[s] = pd.read_csv(path, parse_dates=["date"], index_col="date")
        exch = self._exchange() if adjust else None
        bench = None
        if exch is not None:
            bp = self.raw / "NIFTY50.csv"
            bench = pd.read_csv(bp, parse_dates=["date"], index_col="date")["close"] if bp.exists() else None
            registry = load_registry()
        for s, df in frames.items():
            if adjust:
                if exch is not None and s in exch["codes"]:
                    from .exchange import series
                    c = correct_with_exchange(s, df, series(exch["prices"], exch["codes"][s]), bench, registry)
                    df, notes = c.df, c.notes
                    self.events[s] = c.events
                    self.quality[s] = c.quality
                else:
                    df, notes = apply_blips(s, df)
                    self.quality[s] = pd.Series(0.9, df.index)          # internal checks only
                if notes:
                    self.adjustments[s] = notes
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]
            if len(df):
                out[s] = df
        return out

    def _exchange(self) -> dict | None:
        """BSE cache + symbol->scrip-code map, if present."""
        from .exchange import load as load_bse

        root = self.exchange_root if getattr(self, "exchange_root", None) else self.root.parent / "exchange"
        codes = Path(__file__).parent / "reference" / "bse_codes.csv"
        if not (Path(root) / "bse").exists() or not codes.exists():
            return None
        if not hasattr(self, "_exch_cache"):
            cm = pd.read_csv(codes, dtype={"bse_code": str})
            self._exch_cache = {"prices": load_bse(root), "codes": dict(zip(cm["symbol"], cm["bse_code"], strict=True)),
                                "manifest": _exchange_hash(Path(root))}
        return self._exch_cache

    def exchange_hash(self) -> str:
        e = self._exchange()
        return e["manifest"] if e else "none"


def _exchange_hash(root: Path) -> str:
    p = root / "bse" / "manifest.json"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "none"


def align_returns(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Close-to-close returns on the union calendar; NaN where an asset did not trade."""
    closes = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    return closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
