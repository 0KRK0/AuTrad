"""Execution rules learned from research results that are too small to be strategies (v1.2).

Each rule records the measured effect, its source and how the execution layer / AEM should use it. Rules are
advisory inputs to execution; they never create trades."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    applies_to: str
    effect_bp: float
    source: str
    action: str


RULES = {
    "stock_futures_roll_window": Rule(
        "stock_futures_roll_window", "rolling single-stock futures (sell near / buy next, or the reverse)",
        2.8, "v1.1 HB4: liquid stock-futures calendar spread +0.36 %/yr richer in the last 3 sessions before expiry "
             "(t 4.9, 142 cycles) = about 2.8 bp per roll; NIFTY/BANKNIFTY show no effect",
        "roll long stock-futures positions before the last 3 sessions; rolling a short position inside the window is cheaper"),
    "fo_ban_no_new_exposure": Rule(
        "fo_ban_no_new_exposure", "names in, or approaching, an F&O ban (open interest >= 95 % of MWPL)",
        0.5, "v1.2 HF1: during bans the near-month future trades at a larger DISCOUNT to fair value (signed -16 bp on "
             "average) but the effect is concentrated in distressed names (YESBANK 2020, JETAIRWAYS, DHFL, RELCAPITAL, "
             "PCJEWELLER); the median episode is +0.5 bp, +5 bp without the 20 largest; month-clustered t 2.1",
        "no new F&O exposure is possible during a ban; for names near a ban in financial distress, price a futures "
        "discount of up to several hundred bp (tail risk), otherwise treat the basis as normal"),
}


def roll_advice(sessions_to_expiry: int, position: str = "long") -> dict:
    """Advice for rolling a single-stock futures position given the number of sessions left to expiry."""
    r = RULES["stock_futures_roll_window"]
    inside = sessions_to_expiry <= 3
    if position == "long":
        return {"advice": "roll now (window not yet open)" if not inside else "inside expensive window: roll only if forced",
                "expected_extra_cost_bp": r.effect_bp if inside else 0.0, "rule": r.name}
    return {"advice": "rolling a short inside the window earns the premium", "expected_extra_cost_bp": -r.effect_bp if inside else 0.0,
            "rule": r.name}
