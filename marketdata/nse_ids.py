"""Our symbol -> (ISINs, NSE trading symbols) used to match NSE bhavcopy rows, from the point-in-time file,
the BSE code map and the rename aliases. ISIN matches always win over symbol matches (symbols get reused)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pit import ALIASES

REF = Path(__file__).parent / "reference"
EXTRA_SYMBOLS = {"NIFTYBEES": {"NIFTYBEES"}, "GOLDBEES": {"GOLDBEES"}, "BANKBEES": {"BANKBEES"}, "SILVERBEES": {"SILVERBEES"},
                 "SHRIRAMFIN": {"SRTRANSFIN"}}                    # Shriram Transport Finance, renamed Dec 2022
EXTRA_ISINS = {"SHRIRAMFIN": {"INE721A01013"}}


def identity_sets() -> dict[str, tuple[set[str], set[str]]]:
    out: dict[str, tuple[set[str], set[str]]] = {}

    def add(sym: str, isin: str | None = None, nse_sym: str | None = None) -> None:
        i, s = out.setdefault(sym, (set(), set()))
        if isin and isinstance(isin, str) and isin.startswith("IN"):
            i.add(isin.strip())
        if nse_sym:
            s.add(nse_sym.strip())

    pit = pd.read_csv(REF / "nifty50_pit_snapshots.csv", dtype=str)
    for r in pit.itertuples(index=False):
        add(ALIASES.get(r.symbol, r.symbol), r.isin, r.symbol)
    bse = pd.read_csv(REF / "bse_codes.csv", dtype=str)
    for r in bse.itertuples(index=False):
        add(r.symbol, getattr(r, "isin", None), r.symbol)
    for old, new in ALIASES.items():
        add(new, None, old)
        add(new, None, new)
    for k, v in EXTRA_SYMBOLS.items():
        for s in v:
            add(k, None, s)
    for k, v in EXTRA_ISINS.items():
        for i in v:
            add(k, i, None)
    return out
