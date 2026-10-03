"""Demo: category agents reacting to a tweet, an RBI speech, a real penny-stock story and a pump, plus you overriding.

python -m scripts.run_agents_demo      (synthetic data: shows mechanics, not expected profits)
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from agents.events import Event, EventType
from agents.override import Override
from agents.taxonomy import Category, Instrument
from backtest.multi import run_multi_agent
from marketdata.loaders import synthetic


def spike(df: pd.DataFrame, day: int, move: float, vol_mult: float = 8.0) -> pd.DataFrame:
    df = df.copy()
    t = df.index[day]
    df.loc[df.index[day:], ["open", "high", "low", "close"]] *= 1 + move
    df.loc[t, "high"] = df.loc[t, ["open", "close"]].max() * 1.01
    df.loc[t, "volume"] *= vol_mult
    return df


def main() -> None:
    n = 400
    data = {
        "RELIANCE": synthetic("RELIANCE", n, seed=0),
        "SMALLCO": spike(synthetic("SMALLCO", n, seed=1), 250, 0.12),   # penny, real news
        "PUMPCO": spike(synthetic("PUMPCO", n, seed=3), 260, 0.15),     # penny, no news: pump
        "BTCINR": synthetic("BTCINR", n, seed=4),
        "DOGEINR": spike(synthetic("DOGEINR", n, seed=5), 300, 0.20),   # meme coin, tweet
        "USDINR": synthetic("USDINR", n, seed=6),
    }
    inst = {
        "RELIANCE": Instrument("RELIANCE", "equity", 1400, 5e10, 1.9e13, tags={"nifty100"}),
        "SMALLCO": Instrument("SMALLCO", "equity", 12, 5e7, 2e9),
        "PUMPCO": Instrument("PUMPCO", "equity", 4, 2e7, 5e8, surveillance=True),
        "BTCINR": Instrument("BTCINR", "crypto", 8e6, 5e9),
        "DOGEINR": Instrument("DOGEINR", "crypto", 15, 1e9),
        "USDINR": Instrument("USDINR", "fx", 88, 1e11),
    }
    d = data["DOGEINR"].index
    events = [
        Event(d[300], EventType.SOCIAL_POST, "Elon Musk tweeted about Dogecoin payments", "x.com", ["DOGEINR"],
              {Category.CRYPTO_MEME, Category.CRYPTO_ALT}, 0.9, 0.9, 0.6),
        Event(d[250], EventType.EARNINGS, "SmallCo Q2 results: profit rose 80%, new order win", "bse_filings", ["SMALLCO"],
              set(), 0.8, 0.8, 0.8),
        Event(d[320], EventType.POLICY_SPEECH, "RBI governor signals rate cut; rupee weakens", "rbi", [],
              {Category.FX_INR, Category.IN_LARGECAP, Category.IN_INDEX_FNO}, 0.6, 0.7, 0.9),
    ]
    schedule = {  # you step in twice
        d[200]: [Override("TARGET", "rajesh", "my conviction buy", symbol="RELIANCE", weight=0.08, expires_at=210 * 86400)],
        d[330]: [Override("PAUSE_CATEGORY", "rajesh", "meme coins too hot", category=Category.CRYPTO_MEME)],
    }
    res = run_multi_agent(data, inst, events, schedule=schedule)
    log = pd.DataFrame(res["decisions"])
    focus = log[log["ts"].isin([str(d[i].date()) for i in (200, 250, 251, 260, 261, 300, 301, 320, 321)])]
    print(focus[["ts", "symbol", "category", "source", "proposed", "approved", "ok", "reasons"]].to_string(index=False, max_colwidth=70))
    print("\nDecisions by category/source:", dict(Counter(zip(log.category, log.source, strict=False))))
    print("Agents:", json.dumps(res["agents"], indent=1))
    print("Metrics:", {k: round(v, 4) for k, v in res["metrics"].items()})
    assert np.isfinite(res["equity"].iloc[-1])


if __name__ == "__main__":
    main()
