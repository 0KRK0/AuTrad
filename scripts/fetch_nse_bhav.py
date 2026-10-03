"""Download NSE's official daily bhavcopy for every session in a vendor snapshot (run on a home/office connection).

    python -m scripts.fetch_nse_bhav --data-dir data/snap_20261001 --exchange-dir data/exchange --budget 160
Re-run until it reports todo 0; only missing sessions are fetched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from marketdata.nse_bhav import download
from marketdata.nse_ids import identity_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/snap_20261001")
    ap.add_argument("--exchange-dir", default="data/exchange")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--budget", type=float, default=None, help="stop after this many seconds (resume later)")
    ap.add_argument("--refetch", nargs=2, default=None, metavar=("FROM", "TO"), help="download these dates again (new identities)")
    a = ap.parse_args()
    cal = pd.read_csv(Path(a.data_dir) / "raw" / "NIFTY50.csv", usecols=["date"], parse_dates=["date"])["date"]
    days = [d.date() for d in cal if d >= pd.Timestamp(a.start)]
    isins, symbols = set(), set()
    for i, s in identity_sets().values():
        isins |= i
        symbols |= s
    ref = {d.isoformat() for d in days if a.refetch and a.refetch[0] <= d.isoformat() <= a.refetch[1]}
    print(download(a.exchange_dir, days, isins, symbols, budget_s=a.budget, refetch=ref))


if __name__ == "__main__":
    main()
