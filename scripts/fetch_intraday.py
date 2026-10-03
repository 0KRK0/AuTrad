"""python -m scripts.fetch_intraday --data-dir data/snap_20261001        # 1h (~2 years) and 5m (~60 days)"""
from __future__ import annotations

import argparse

from marketdata.intraday import fetch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--intervals", nargs="*", default=["1h", "5m"])
    a = ap.parse_args()
    for iv in a.intervals:
        c = fetch(a.data_dir, iv)
        print(f"{iv}: {len(c)} symbols, bars per symbol {min(c.values())}-{max(c.values())}")


if __name__ == "__main__":
    main()
