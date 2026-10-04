"""Record NSE market depth forward (run on a machine that stays on during market hours, 09:00-15:35 IST).

    set KITE_API_KEY=...  &  set KITE_ACCESS_TOKEN=...      (your own Kite Connect credentials; never stored)
    python -m scripts.record_nse_depth --symbols NIFTY-FUT,BANKNIFTY-FUT,NIFTY-SPOT

After each session (or any time later):  python -m scripts.record_nse_depth --seal
    -> every recorded, unsealed day gets QC + a compressed research file + a sha256 manifest; raw parts become read-only.
Then, after >= 20 recorded days:  python -m scripts.v09_micro nse-grid  -> the M0 analysis on NSE data.
"""
from __future__ import annotations

import argparse
import os

from marketdata.nse_depth_recorder import DepthRecorder, KiteSource


def resolve(kite, names: list[str]) -> dict[int, str]:  # pragma: no cover - needs credentials
    import datetime as dt

    out = {}
    nfo = [i for i in kite.instruments("NFO") if i["instrument_type"] == "FUT"]
    for n in names:
        if n.endswith("-FUT"):
            base = n[:-4]
            c = sorted([i for i in nfo if i["name"] == base and i["expiry"] >= dt.date.today()], key=lambda i: i["expiry"])[0]
            out[c["instrument_token"]] = n
        elif n == "NIFTY-SPOT":
            out[256265] = n                                   # NIFTY 50 index token on Kite
        elif n == "BANKNIFTY-SPOT":
            out[260105] = n
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="NIFTY-FUT,BANKNIFTY-FUT,NIFTY-SPOT")
    ap.add_argument("--seal", action="store_true", help="QC + seal every recorded day not sealed yet (no credentials needed)")
    a = ap.parse_args()
    if a.seal:
        import json
        from marketdata.nse_depth_recorder import recorded_days, seal_day
        for sym, day in recorded_days():
            print(json.dumps(seal_day(sym, day), default=str))
        return
    from kiteconnect import KiteConnect

    key, tok = os.environ["KITE_API_KEY"], os.environ["KITE_ACCESS_TOKEN"]
    kite = KiteConnect(api_key=key)
    kite.set_access_token(tok)
    KiteSource(key, tok, resolve(kite, a.symbols.split(","))).run(DepthRecorder())


if __name__ == "__main__":
    main()
