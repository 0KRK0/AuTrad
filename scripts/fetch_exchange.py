"""Download BSE's official daily bhavcopy (unadjusted) for every symbol in marketdata/reference/bse_codes.csv.

    python -m scripts.fetch_exchange --exchange-dir data/exchange --data-dir data/snap_20261001
Re-running only fetches missing sessions (including weekend special sessions such as Diwali muhurat trading
or budget-day Saturdays that appear in the vendor data). ~3,000 sessions take about 5 minutes.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from marketdata.exchange import download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange-dir", default="data/exchange")
    ap.add_argument("--data-dir", default="data/snap_20261001", help="vendor snapshot whose weekend sessions are fetched too")
    ap.add_argument("--start", default="2014-12-01")
    ap.add_argument("--end", default=str(date.today()))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cm = pd.read_csv("marketdata/reference/bse_codes.csv", dtype={"bse_code": str})
    extra: set[date] = set()
    for sym in cm["symbol"]:
        p = Path(a.data_dir, "raw", f"{sym}.csv")
        if p.exists():
            extra |= {d.date() for d in pd.read_csv(p, usecols=["date"], parse_dates=["date"])["date"] if d.weekday() >= 5}
    print(f"{len(extra)} weekend session(s) in the vendor data for exchange-listed symbols")
    print(download(a.exchange_dir, date.fromisoformat(a.start), date.fromisoformat(a.end), set(cm["bse_code"]), a.workers,
                   extra_dates=extra))


if __name__ == "__main__":
    main()
