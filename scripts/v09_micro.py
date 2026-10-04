"""v0.9 study M0 (research/preregistration_v0.9.md).

    python -m scripts.v09_micro sample                 # list archived days, draw the pre-registered sample (seed 2026)
    python -m scripts.v09_micro fetch --day N          # download + process day N of the sample (one per call; raw zips deleted)
    python -m scripts.v09_micro audit                  # data audit (no returns)
    python -m scripts.v09_micro run --out reports/v0.9_<ts>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from marketdata.micro_binance import ROOT, SCRATCH, available_days, download, process_day, sample

SAMPLE = ROOT / "sample.json"


def cmd_sample() -> None:
    days = available_days(requests.Session())
    s = sample(days)
    ROOT.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(json.dumps(s | {"days_available_first": days[0], "days_available_last": days[-1]}, indent=1))
    print(json.dumps(s, indent=1))


def cmd_fetch(i: int, stage: str, force: bool = False) -> None:
    s = json.loads(SAMPLE.read_text())
    day = (s["train"] + s["test"])[i]
    done = ROOT / f"{day}.json"
    if done.exists() and "raw_backward_steps" in json.loads(done.read_text()) and not force:
        print(day, "done")
        return
    t0 = time.time()
    if stage in ("download", "all"):
        download(requests.Session(), day)
        print(day, "downloaded", round(time.time() - t0, 1), "s")
    if stage in ("process", "all"):
        paths = {k: SCRATCH / f"{k}-{day}.zip" for k in ("book", "trades", "spot")}
        p = process_day(day, paths)
        for q in paths.values():
            q.unlink(missing_ok=True)
        print(day, "processed", round(time.time() - t0, 1), "s", round(p.stat().st_size / 1e6, 1), "MB")


def cmd_nse_grid() -> None:
    """Recorded NSE depth (scripts/record_nse_depth.py) -> 1 s grids in data/micro/nse/<symbol>/grid/<day>.parquet.
    An NSE study needs >= 20 recorded days and its own pre-registration (NSE tick size, costs, session hours)."""
    from marketdata.nse_depth_recorder import ROOT as NSE, load_day, to_grid

    for sym_dir in sorted(p for p in NSE.glob("*") if p.is_dir()):
        for day_dir in sorted(p for p in sym_dir.glob("20*") if p.is_dir()):
            out = sym_dir / "grid" / f"{day_dir.name}.parquet"
            if out.exists():
                continue
            spot = load_day("NIFTY-SPOT", day_dir.name) if sym_dir.name.startswith("NIFTY") else None
            out.parent.mkdir(parents=True, exist_ok=True)
            to_grid(load_day(sym_dir.name, day_dir.name), day_dir.name, spot=spot).to_parquet(out, compression="zstd")
            print(sym_dir.name, day_dir.name, "->", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "fetch", "audit", "run", "nse-grid"])
    ap.add_argument("--day", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", default="all", choices=["download", "process", "all"])
    ap.add_argument("--out", default=f"reports/v0.9_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    a = ap.parse_args()
    if a.cmd == "sample":
        cmd_sample()
    elif a.cmd == "fetch":
        cmd_fetch(a.day, a.stage, a.force)
    elif a.cmd == "nse-grid":
        cmd_nse_grid()
    else:
        from scripts import v09_analysis as V
        V.audit() if a.cmd == "audit" else V.run(Path(a.out))


if __name__ == "__main__":
    main()
