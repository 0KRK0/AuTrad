"""Download NSE results filings (index + XBRL) for the equity universe and build data/pit/nse/facts.csv.

    python -m scripts.fetch_fundamentals --root data/pit            # resumable; ~4,000 filings, about 1 hour
Must run from a connection NSE does not block (NSE refuses cloud/datacenter IPs).
"""
from __future__ import annotations

import argparse

import pandas as pd

from marketdata.nse_filings import ingest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/pit")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--symbols", nargs="*")
    a = ap.parse_args()
    cm = pd.read_csv("marketdata/reference/bse_codes.csv")
    etfs = {"NIFTYBEES", "BANKBEES", "GOLDBEES", "SILVERBEES"}
    syms = a.symbols or [s for s in cm["symbol"] if s not in etfs]
    ingest(a.root, syms, a.workers, a.start, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
