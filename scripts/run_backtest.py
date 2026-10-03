"""Run a backtest.  python -m scripts.run_backtest [--csv AAPL=path.csv ...] [--yf AAPL MSFT]"""
from __future__ import annotations

import argparse
import json

from backtest.engine import run_backtest
from marketdata.loaders import load_csv, synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="*", default=[], help="SYMBOL=path.csv")
    ap.add_argument("--yf", nargs="*", default=[], help="symbols via yfinance (pip install yfinance)")
    ap.add_argument("--start", default="2015-01-01")
    args = ap.parse_args()

    data = {s: load_csv(p) for s, p in (a.split("=", 1) for a in args.csv)}
    if args.yf:
        from marketdata.loaders import load_yfinance
        data |= {s: load_yfinance(s, args.start) for s in args.yf}
    if not data:
        print("No data given: using SYNTHETIC series (tests mechanics only, not profitability).")
        data = {f"SYN{i}": synthetic(f"SYN{i}", seed=i) for i in range(4)}

    res = run_backtest(data)
    res.pop("equity")
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
