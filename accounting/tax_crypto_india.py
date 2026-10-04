"""Indian tax on crypto (virtual digital assets, Income-tax Act s.115BBH and s.194S) for research estimates.

* Every transfer's gain is taxed at 30 % + 4 % cess = 31.2 %. The cost of acquisition is the only deduction.
* A loss on one transfer is never set off against any gain, not even another crypto gain, and is never carried
  forward. So tax = 31.2 % x sum of the positive gains of individual transfers.
* 1 % TDS on each sale. It is credited against the year's tax, so with recoverable=True it changes only timing (not
  modelled). recoverable=False treats it as a 1 % cost on every sale (a conservative sensitivity).
* Tax is paid at each fiscal year's last session. At the end, open lots are sold at the final prices and taxed.

`fills` are engine fills: {"t", "symbol", "delta_w", "value", "price"}.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

RATE = 0.30 * 1.04
TDS = 0.01


def _fy(d: pd.Timestamp) -> int:
    return d.year + 1 if d.month > 3 else d.year


def crypto_post_tax(nav: pd.Series, fills: list[dict], dates: pd.DatetimeIndex, end_prices: dict[str, float],
                    start: pd.Timestamp | None = None, recoverable: bool = True) -> dict:
    lots: dict[str, deque] = defaultdict(deque)               # symbol -> [qty, unit cost]
    tax_by_fy: dict[int, float] = defaultdict(float)
    gains = []
    for f in sorted(fills, key=lambda x: x["t"]):
        d, sym, px = dates[f["t"]], f["symbol"], f["price"]
        qty = f["value"] / px if px > 0 else 0.0
        if f["delta_w"] > 0:
            lots[sym].append([qty, px])
            continue
        left, cost = qty, 0.0
        while left > 1e-12 and lots[sym]:
            q, c = lots[sym][0]
            take = min(q, left)
            cost += take * c
            left -= take
            if take >= q - 1e-12:
                lots[sym].popleft()
            else:
                lots[sym][0][0] = q - take
        gain = f["value"] - cost
        if start is None or d >= start:
            tax = RATE * max(gain, 0.0) + (0.0 if recoverable else TDS * f["value"])
            tax_by_fy[_fy(d)] += tax
            gains.append(gain)
    end = nav.index[-1]
    liq_tax = 0.0
    for sym, q in lots.items():
        px = end_prices.get(sym)
        if px is None:
            continue
        for qty, c in q:
            liq_tax += RATE * max(qty * (px - c), 0.0) + (0.0 if recoverable else TDS * qty * px)
    nav = nav[nav.index >= start] if start is not None else nav
    pay = {}
    for fy, tax in tax_by_fy.items():
        sess = nav.index[nav.index <= pd.Timestamp(f"{fy}-03-31")]
        if len(sess) and pd.Timestamp(f"{fy}-03-31") <= end:
            pay[sess[-1]] = pay.get(sess[-1], 0.0) + tax
        else:
            liq_tax += tax                                    # the final partial year is paid at liquidation
    post, scale = [nav.iloc[0]], 1.0
    r = nav / nav.shift(1)
    wiped = None
    for d, g in r.iloc[1:].items():
        v = post[-1] * g - pay.get(d, 0.0) * scale
        if v <= 0 or wiped is not None:                    # the tax bill exceeded the account: wiped out, stays at 0
            wiped = wiped or d
            v = 0.0
        post.append(v)
        scale = v / nav.loc[d] if nav.loc[d] > 0 else 0.0
    s = pd.Series(post, index=nav.index)
    liq = max(float(s.iloc[-1] - liq_tax * scale), 0.0)
    return {"post_tax_nav": s, "liquidation_value": liq, "wiped_out_on": str(wiped.date()) if wiped is not None else None,
            "tax_paid": sum(pay.values()),
            "losses_not_set_off": float(-sum(g for g in gains if g < 0)), "transfers": len(gains)}
