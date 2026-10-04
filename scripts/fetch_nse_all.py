"""Whole-market NSE bhavcopy cache (every EQ/BE/BZ security, every session) for point-in-time universes beyond the
NIFTY 50. Separate from the frozen research cache: data/exchange_all/nse/<year>.csv.gz.

    python -m scripts.fetch_nse_all --years 2015-2026 --budget 160      # resumable
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from marketdata.nse_bhav import download


def sessions() -> list[date]:
    """Every session the research cache knows (it already contains all trading days, incl. special sessions)."""
    ds = set()
    for p in sorted(Path("data/exchange/nse").glob("*.csv.gz")):
        ds |= set(pd.read_csv(p, usecols=["date"])["date"].unique())
    for p in sorted(Path("data/exchange/bse").glob("*.csv.gz")):        # BSE sessions incl. special/Muhurat ones
        ds |= set(pd.read_csv(p, usecols=[0]).iloc[:, 0].astype(str).str[:10])
    ds |= SPECIAL
    return sorted(date.fromisoformat(d) for d in ds if d >= "2015-01-01")


SPECIAL = {"2015-02-28", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27", "2020-02-01", "2020-11-14", "2021-11-04",
           "2022-10-24", "2023-11-12", "2024-01-20", "2024-03-02", "2024-05-18", "2024-11-01", "2025-02-01", "2025-10-21", "2026-02-01"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2015-2026")
    ap.add_argument("--budget", type=float, default=160)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    y0, y1 = (int(x) for x in a.years.split("-"))
    days = [d for d in sessions() if y0 <= d.year <= y1]
    print(download("data/exchange_all", days, None, None, workers=a.workers, budget_s=a.budget))


if __name__ == "__main__":
    main()
