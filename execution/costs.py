"""India transaction-cost profiles (statutory charges + brokerage + spread + market impact).

Statutory rates follow Zerodha's published charges page (checked 30 Sep 2026):
  Equity delivery : brokerage 0, STT 0.1% buy & sell, NSE txn 0.00307%, SEBI Rs10/crore,
                    stamp 0.015% buy, GST 18% on (brokerage + SEBI + txn), DP Rs15.34 per scrip sold.
  Currency futures: brokerage min(0.03%, Rs20)/order, no STT, NSE txn 0.00035%, SEBI Rs10/crore,
                    stamp 0.0001% buy, GST 18% on (brokerage + SEBI + txn).
Crypto (CoinDCX INR spot, below Rs2 lakh 30-day volume): 0.50% fee + 18% GST, both sides.
  1% TDS on sells is tax deducted at source (creditable), so it is NOT modelled as a cost.
  Equity intraday (MIS): brokerage min(0.03%, Rs20)/order, STT 0.025% sell, NSE txn 0.00307%, stamp 0.003% buy.
  Equity futures: brokerage min(0.03%, Rs20)/order, STT 0.05% sell (from Apr 2026), NSE txn 0.00183%, stamp 0.002% buy.
Spreads and impact are estimates by liquidity tier and should be recalibrated from live fills.
ETFs are charged like equity delivery (conservative: STT may not apply to non-equity ETFs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

COST_MODEL_VERSION = "india_zerodha_2026_09"   # stored with every backtest; bump when any rate changes
GST = 0.18
SEBI_BPS = 10 / 1e7 * 1e4  # Rs10 per crore = 0.001 bps


@dataclass(frozen=True)
class CostProfile:
    name: str
    buy_bps: float                  # proportional statutory charges on buys (incl. GST), bps of notional
    sell_bps: float                 # proportional statutory charges on sells
    half_spread_bps: float          # paid on every fill vs mid/open
    impact_coef: float = 0.1        # sqrt-impact: coef * daily_vol * sqrt(qty / bar_volume)
    max_participation: float = 0.02 # max fraction of bar volume per fill (0 volume = unlimited)
    brokerage_pct: float = 0.0      # proportional brokerage ...
    brokerage_cap: float = 0.0      # ... capped at Rs per order (0 = no cap)
    flat_per_sell: float = 0.0      # e.g. DP charge per scrip sold
    carry_bps_per_day: float = 0.0  # holding cost, e.g. futures roll (both legs' charges spread over ~21 days)
    cash_settled: bool = True       # False for futures: notional does not consume cash
    shortable: bool = False

    def fees(self, side: int, notional: float) -> float:
        notional = abs(notional)
        prop = notional * (self.buy_bps if side > 0 else self.sell_bps) / 1e4
        brok = notional * self.brokerage_pct / 100
        if self.brokerage_cap > 0:
            brok = min(brok, self.brokerage_cap)
        return prop + brok * (1 + GST) + (self.flat_per_sell if side < 0 else 0.0)

    def round_trip_bps(self, notional: float = 1_00_000.0) -> float:
        """Approximate cost of buying and later selling `notional`, incl. spread, in bps."""
        f = self.fees(1, notional) + self.fees(-1, notional)
        return f / notional * 1e4 + 2 * self.half_spread_bps


def _equity(name: str, half_spread: float, participation: float = 0.02) -> CostProfile:
    txn = 0.307  # bps
    gst_on = (txn + SEBI_BPS) * GST
    buy = 10 + txn + SEBI_BPS + 1.5 + gst_on    # STT + txn + SEBI + stamp + GST
    sell = 10 + txn + SEBI_BPS + gst_on         # STT + txn + SEBI + GST
    return CostProfile(name, buy, sell, half_spread, 0.1, participation, flat_per_sell=15.34)


def _intraday(name: str, half_spread: float) -> CostProfile:
    txn = 0.307
    gst_on = (txn + SEBI_BPS) * GST
    return CostProfile(name, txn + SEBI_BPS + 0.3 + gst_on, 2.5 + txn + SEBI_BPS + gst_on, half_spread, 0.1, 0.02,
                       brokerage_pct=0.03, brokerage_cap=20.0, shortable=True)


def _stock_fut(name: str, half_spread: float) -> CostProfile:
    txn = 0.183
    gst_on = (txn + SEBI_BPS) * GST
    buy, sell = txn + SEBI_BPS + 0.2 + gst_on, 5.0 + txn + SEBI_BPS + gst_on
    roll = (buy + sell + 2 * half_spread) / 21.0      # monthly roll: sell expiring + buy next month
    return CostProfile(name, buy, sell, half_spread, 0.1, 0.02, brokerage_pct=0.03, brokerage_cap=20.0,
                       carry_bps_per_day=roll, cash_settled=False, shortable=True)


def _currency_fut() -> CostProfile:
    txn = 0.035
    gst_on = (txn + SEBI_BPS) * GST
    return CostProfile("nse_currency_fut", txn + SEBI_BPS + 0.01 + gst_on, txn + SEBI_BPS + gst_on,
                       half_spread_bps=1.5, impact_coef=0.0, max_participation=0.0,
                       brokerage_pct=0.03, brokerage_cap=20.0, carry_bps_per_day=0.2, cash_settled=False, shortable=True)


PROFILES: dict[str, CostProfile] = {
    "nse_largecap": _equity("nse_largecap", 2.5),
    "nse_midcap": _equity("nse_midcap", 6.0),
    "nse_smallcap": _equity("nse_smallcap", 12.0, 0.01),
    "nse_etf": _equity("nse_etf", 4.0),
    "nse_currency_fut": _currency_fut(),
    "crypto_inr": CostProfile("crypto_inr", 50 * (1 + GST), 50 * (1 + GST), 10.0, 0.1, 0.01),
    "nse_intraday": _intraday("nse_intraday", 3.0),
    "nse_stock_fut": _stock_fut("nse_stock_fut", 3.0),
    "zero": CostProfile("zero", 0.0, 0.0, 0.0, 0.0, 0.0),
}


def impact(profile: CostProfile, qty: float, volume: float, sigma: float) -> float:
    if volume <= 0 or profile.impact_coef == 0:
        return 0.0
    return profile.impact_coef * sigma * math.sqrt(abs(qty) / volume)
