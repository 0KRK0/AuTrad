"""Download and cache real market data.

python -m scripts.fetch_data                    # full India universe + benchmarks from 2015
python -m scripts.fetch_data --groups equity crypto --start 2018-01-01
"""
from __future__ import annotations

import argparse

from marketdata.pit import nifty50_assets
from marketdata.store import DataStore
from marketdata.universe import BENCHMARKS, select


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=None, help="equity etf crypto fx (default all)")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--no-pit", action="store_true", help="skip the point-in-time NIFTY 50 members (incl. removed stocks)")
    a = ap.parse_args()

    assets = [*select(a.groups), *BENCHMARKS]
    if not a.no_pit:
        have = {x.symbol for x in assets}
        assets += [x for x in nifty50_assets() if x.symbol not in have]
    store = DataStore(a.data_dir)
    reports = store.refresh(assets, a.start, a.end)
    missing = [x.symbol for x in assets if x.symbol not in reports]
    print(f"{'symbol':12} {'rows':>5}  {'first':10}  {'last':10}  quality")
    for s, q in sorted(reports.items()):
        flag = "OK " if q.passed else "FAIL"
        print(f"{s:12} {q.rows:5}  {q.first}  {q.last}  {flag} {'; '.join(q.notes)}")
    if missing:
        print("MISSING:", ", ".join(missing))
    print(f"manifest hash: {store.manifest_hash()}  ({len(reports)} assets)")


if __name__ == "__main__":
    main()
