"""Read-only market data shared by the advisory agents (data is not an agent: sharing it does not break isolation)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MarketContext:
    P: object                          # lab.panel.Panel (exchange-anchored)
    facts: pd.DataFrame                # PIT results filings (marketdata.fundamentals.load_facts)
    events: pd.DataFrame               # earnings events with past-only 'bad' flags (scripts.v06.bad_events)
    quality: dict[str, pd.Series] = field(default_factory=dict)
    dataset_hash: str = ""
    company_names: dict[str, str] = field(default_factory=dict)
    pit_rows: dict[str, list[tuple[str, str]]] = field(default_factory=dict)   # symbol -> [(effective date, csv row)]

    def __post_init__(self) -> None:
        """Events may come from another panel: re-index them onto this panel's sessions by date."""
        self.facts = self.facts.copy()
        self.facts.attrs = {}
        if len(self.events) and "reaction_day" in self.events:
            e = self.events.copy()
            e["reaction_idx"] = self.P.dates.get_indexer(pd.to_datetime(e["reaction_day"]))
            self.events = e[e["reaction_idx"] >= 0].reset_index(drop=True)

    def idx(self, d: pd.Timestamp) -> int:
        return int(self.P.dates.searchsorted(pd.Timestamp(d), side="right") - 1)

    def close(self) -> np.ndarray:
        if not hasattr(self, "_C"):
            self._C = self.P.ffill_close()
        return self._C

    def ret(self, symbol: str, t: int, n: int) -> float:
        j = self.P.idx[symbol]
        C = self.close()
        return float(C[t, j] / C[t - n, j] - 1) if t - n >= 0 and np.isfinite(C[t, j]) and np.isfinite(C[t - n, j]) else float("nan")

    def filings(self, symbol: str, asof: pd.Timestamp) -> pd.DataFrame:
        a = pd.Timestamp(asof)
        a = a.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30) if a.tzinfo is None else a
        f = self.facts[(self.facts["symbol"] == symbol) & (self.facts["available_at"] <= a)]
        return f.sort_values(["period_end", "available_at"]).drop_duplicates("period_end", keep="last")
