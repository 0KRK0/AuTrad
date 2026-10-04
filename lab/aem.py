"""AEM — Alpha Economics Model (research/preregistration_v1.0.md §1). Sits between every signal and the Risk Gate:
it turns an expected *gross* edge into an expected *net* edge and a decision EXECUTE / WAIT / ABANDON.

    E[net] = E[gross] - spread - fees(+rebates) - E[adverse selection | passive] - E[impact] - latency drift
             - tax drag - opportunity cost

Every cost parameter comes from a measured or statutory source:
* India equity / stock futures: execution/costs.py (PROFILES);
* NSE index futures: marketdata/nse_depth_recorder.nse_index_futures_round_trip_bp;
* crypto: Binance retail / best-case fee tiers (v0.9), M0 spreads, and the v1.0 M1 measurements of passive fill
  probability and adverse selection (research/data_audits/v1.0_aem_calibration.json, written by the M1 run).

Tax: a per-trade tax without loss set-off (crypto, s.115BBH: 31.2 %) is charged on the expected *positive part* of
the trade outcome, E[max(X, 0)] for X ~ N(mu, sigma), so a noisy trade with a small mean pays tax on its winners
while its losers are lost. With set-off (India equity STCG 20 % within a year) the drag is rate * max(mu, 0).
Fees: under s.115BBH no expense other than the cost of acquisition is deductible, so for crypto the tax base is the
price outcome before fees (fees_deductible=False). The passive route is a passive entry plus a crossing exit.
min_ratio (E[net]/sigma per trade) is a per-venue knob, 0 by default (not yet calibrated).
The AEM is an architecture component; it is unit-tested and is not part of any study's statistics.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

CALIBRATION = Path(__file__).resolve().parents[1] / "research/data_audits/v1.0_aem_calibration.json"


@dataclass(frozen=True)
class Venue:
    name: str
    taker_bp: float                 # fee per side when crossing (bp of notional); statutory charges included
    maker_bp: float                 # fee per side when posting (negative = rebate)
    half_spread_bp: float           # paid per side when crossing
    tax_rate: float                 # on gains
    tax_setoff: bool                # True: losses offset gains (drag on the mean); False: per-trade tax on winners
    passive_fill_prob: float = 0.0  # probability a passive order fills within the horizon budget (measured)
    passive_adverse_bp: float = 0.0 # expected mid move AGAINST us shortly after a passive fill (+ = cost; measured)
    fees_deductible: bool = True    # False under s.115BBH: only the cost of acquisition is deductible
    impact_coef: float = 0.1        # square-root impact: coef * daily_vol_bp * sqrt(qty / adv)
    min_ratio: float = 0.0          # minimum E[net] / sigma(net) per trade to EXECUTE
    source: str = ""


def _load_calibration() -> dict:
    try:
        return json.loads(CALIBRATION.read_text())
    except (OSError, ValueError):
        return {}


def venues() -> dict[str, Venue]:
    from execution.costs import PROFILES
    from marketdata.nse_depth_recorder import nse_index_futures_round_trip_bp

    cal = _load_calibration().get("crypto_btcusdt_perp", {})
    lc = PROFILES["nse_largecap"]
    sf = PROFILES["nse_stock_fut"]
    eq_side = (lc.round_trip_bps(10_00_000.0) - 2 * lc.half_spread_bps) / 2           # ₹10 lakh ticket
    sf_side = (sf.round_trip_bps(10_00_000.0) - 2 * sf.half_spread_bps) / 2
    return {
        "nse_delivery": Venue("nse_delivery", eq_side, eq_side, lc.half_spread_bps, 0.20, True,
                              source="execution/costs.py nse_largecap at ₹10 lakh; STCG 20 %"),
        "nse_stock_fut": Venue("nse_stock_fut", sf_side, sf_side, sf.half_spread_bps, 0.30, True,
                               source="execution/costs.py nse_stock_fut; business income ~30 % slab (assumption)"),
        "nse_index_fut": Venue("nse_index_fut", nse_index_futures_round_trip_bp() / 2, nse_index_futures_round_trip_bp() / 2,
                               0.2, 0.30, True, source="nse_index_futures_round_trip_bp (STT 0.05 %, lot 65); spread assumed 0.4 bp"),
        "crypto_retail": Venue("crypto_retail", 5.0, 2.0, cal.get("half_spread_bp", 0.012), 0.312, False,
                               cal.get("passive_fill_prob_60s", 0.0), cal.get("adverse_mid_move_after_passive_fill_5s_bp", 0.0),
                               fees_deductible=False,
                               source="Binance USDⓈ-M retail fees; M0 spread; v1.0 M1 calibration; s.115BBH"),
        "crypto_best_case": Venue("crypto_best_case", 1.7, 0.0, cal.get("half_spread_bp", 0.012), 0.312, False,
                                  cal.get("passive_fill_prob_60s", 0.0), cal.get("adverse_mid_move_after_passive_fill_5s_bp", 0.0),
                                  fees_deductible=False,
                                  source="Binance best published tier; M0 spread; v1.0 M1 calibration; s.115BBH"),
    }


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def expected_positive_part(mu: float, sigma: float) -> float:
    """E[max(X, 0)] for X ~ N(mu, sigma^2)."""
    if sigma <= 0:
        return max(mu, 0.0)
    z = mu / sigma
    return mu * _norm_cdf(z) + sigma * _norm_pdf(z)


def tax_drag(mu: float, sigma: float, v: Venue) -> float:
    if v.tax_rate <= 0:
        return 0.0
    return v.tax_rate * (max(mu, 0.0) if v.tax_setoff else expected_positive_part(mu, sigma))


@dataclass
class Signal:
    gross_bp: float                  # expected gross return over the horizon, bp (sign-adjusted: > 0 means favourable)
    sigma_bp: float                  # per-trade standard deviation of the outcome, bp
    horizon_days: float             # holding period
    latency_s: float = 0.0           # decision-to-order latency
    half_life_s: float = math.inf    # signal half-life (decay of the expected gross with delay)
    qty_over_adv: float = 0.0        # order size / average daily volume
    daily_vol_bp: float = 0.0
    cash_rate: float = 0.06          # opportunity cost of capital (INR cash proxy)
    extra: dict = field(default_factory=dict)


def evaluate(sig: Signal, v: Venue) -> dict:
    drift = sig.gross_bp * (1 - 2 ** (-sig.latency_s / sig.half_life_s)) if math.isfinite(sig.half_life_s) else 0.0
    g = sig.gross_bp - drift
    impact = v.impact_coef * sig.daily_vol_bp * math.sqrt(max(sig.qty_over_adv, 0.0)) * 2
    opp = sig.cash_rate * sig.horizon_days / 365 * 1e4
    # crossing both legs
    cross_px = 2 * v.half_spread_bp + impact                           # price costs (always reduce the taxable outcome)
    cross_fee = 2 * v.taker_bp
    pre_tax_cross = g - cross_px - cross_fee - opp
    tax_c = tax_drag(g - cross_px - (cross_fee if v.fees_deductible else 0.0), sig.sigma_bp, v)
    net_cross = pre_tax_cross - tax_c
    # passive entry (fills with probability p; the mid then moves against us by the measured adverse selection),
    # exit by crossing: passive exits are not assumed
    pas_px = v.passive_adverse_bp + v.half_spread_bp + impact
    pas_fee = v.maker_bp + v.taker_bp
    pre_tax_passive = g - pas_px - pas_fee - opp
    net_passive_if_filled = pre_tax_passive - tax_drag(g - pas_px - (pas_fee if v.fees_deductible else 0.0), sig.sigma_bp, v)
    ev_passive = v.passive_fill_prob * net_passive_if_filled            # unfilled: no trade, no cost
    ratio = net_cross / sig.sigma_bp if sig.sigma_bp > 0 else math.inf
    if net_cross > 0 and ratio >= v.min_ratio:
        decision, route = "EXECUTE", ("passive" if ev_passive > net_cross else "cross")
    elif ev_passive > 0:
        decision, route = "WAIT", "passive"
    else:
        decision, route = "ABANDON", None
    return {"venue": v.name, "decision": decision, "route": route, "gross_bp": sig.gross_bp,
            "components_bp": {"latency_drift": drift, "spread": 2 * v.half_spread_bp, "fees_cross": cross_fee,
                              "fees_passive_route": pas_fee, "adverse_selection_passive": v.passive_adverse_bp,
                              "impact": impact, "opportunity": opp, "tax_cross": tax_c},
            "net_cross_bp": net_cross, "net_passive_if_filled_bp": net_passive_if_filled, "ev_passive_bp": ev_passive,
            "net_over_sigma": ratio}


def with_overrides(v: Venue, **kw) -> Venue:
    return replace(v, **kw)


# ================================================================================================ v1.1: opportunity pricing
@dataclass
class Opportunity:
    """One way of getting paid, priced per unit of scarce capital and time (v1.1 Opportunity Pricing Engine).

    expected_bp      E[R | filled], gross, over the holding period, bp of notional
    sigma_bp         per-trade standard deviation of the outcome, bp of notional
    holding_days     capital lock-up (calendar days; fractions allowed)
    capital_per_notional  capital tied up per unit notional (cash leg 1.0; + futures margin 0.2 for cash-and-carry)
    p_fill           probability the entry is obtained on the intended (cheap) route
    no_fill_cost_bp  what a miss costs (e.g. crossing late, or the signal lost), bp of notional
    capacity_inr     notional the opportunity can absorb without breaking its own assumptions
    economic_mechanism  why the edge should exist (who pays us and why it persists); required
    risk_group       opportunities sharing a risk driver (allocator caps the group)
    """
    name: str
    economic_mechanism: str
    expected_bp: float
    sigma_bp: float
    holding_days: float
    venue: Venue
    route: str = "cross"
    capital_per_notional: float = 1.0
    p_fill: float = 1.0
    no_fill_cost_bp: float = 0.0
    capacity_inr: float = math.inf
    risk_group: str = ""
    cash_rate: float = 0.06
    cash_tax_rate: float = 0.312                 # the cash alternative's interest is taxed too (slab rate)
    margin_per_notional: float = 0.0            # exchange margin consumed per unit notional (part of capital_per_notional)
    reserve_capital_unfilled: bool = False       # True if capital is blocked while waiting for a fill


def price(o: Opportunity) -> dict:
    if not o.economic_mechanism.strip():
        raise ValueError(f"{o.name}: economic_mechanism is required")
    v = o.venue
    if o.route == "cross":
        px, fee = 2 * v.half_spread_bp, 2 * v.taker_bp
    else:                                                                 # passive entry, crossing exit
        px, fee = v.passive_adverse_bp + v.half_spread_bp, v.maker_bp + v.taker_bp
    base = o.expected_bp - px - (fee if v.fees_deductible else 0.0)
    tax = tax_drag(base, o.sigma_bp, v)
    net_if_filled = o.expected_bp - px - fee - tax
    ev_bp = o.p_fill * net_if_filled - (1 - o.p_fill) * o.no_fill_cost_bp          # bp of notional
    cap = o.capital_per_notional
    used = 1.0 if o.reserve_capital_unfilled else o.p_fill                            # capital is tied up only when filled
    opp_bp = o.cash_rate * (1 - o.cash_tax_rate) * o.holding_days / 365 * 1e4 * cap * used   # after-tax cash forgone
    excess_bp = ev_bp - opp_bp
    per_cap_day = excess_bp / cap / max(o.holding_days, 1e-9)                         # bp of capital per day
    return {"name": o.name, "economic_mechanism": o.economic_mechanism, "ev_bp": ev_bp, "net_if_filled_bp": net_if_filled,
            "tax_bp": tax, "opportunity_cost_bp": opp_bp, "excess_bp": excess_bp,
            "excess_bp_per_capital_day": per_cap_day, "ann_excess_on_capital": per_cap_day * 365 / 1e4,
            "decision": "EXECUTE" if excess_bp > 0 else "ABANDON"}


def allocate(opps: list[Opportunity], capital: float, max_share: float = 0.25, group_share: float = 0.40,
             drawdown_budget: float | None = None, margin_available: float | None = None, min_fill_prob: float = 0.0) -> dict:
    """Greedy capital allocator: where should the next rupee go?

    Objective (v1.2): OpportunityScore = (E[P&L] - after-tax cash alternative) / (capital x time), i.e. excess bp of
    capital per day, after costs, tax and the probability of not being filled. Opportunities are filled best-first up to
    the tightest of these constraints:
      * capacity (notional the opportunity absorbs, converted to capital);
      * concentration: max_share of the book per opportunity, group_share per risk_group (correlation proxy);
      * drawdown budget: a 2-sigma loss of the position <= drawdown_budget x book;
      * margin: the sum of exchange margin used <= margin_available;
      * execution probability: opportunities with p_fill < min_fill_prob are skipped.
    Capital left over stays in cash (excess 0)."""
    priced = sorted(((price(o), o) for o in opps), key=lambda x: -x[0]["excess_bp_per_capital_day"])
    left, groups, alloc = capital, {}, []
    margin_left = margin_available if margin_available is not None else math.inf
    for pr, o in priced:
        if pr["excess_bp"] <= 0 or left <= 0 or o.p_fill < min_fill_prob:
            alloc.append({**pr, "capital": 0.0, "binding": "not attractive" if pr["excess_bp"] <= 0 else "skipped"})
            continue
        g_left = group_share * capital - groups.get(o.risk_group, 0.0) if o.risk_group else math.inf
        caps = {"capacity": o.capacity_inr * o.capital_per_notional, "max_share": max_share * capital, "book": left, "group": g_left}
        if drawdown_budget is not None and o.sigma_bp > 0:                  # 2 sigma of notional loss <= budget
            caps["drawdown"] = drawdown_budget * capital / (2 * o.sigma_bp / 1e4) * o.capital_per_notional
        if o.margin_per_notional > 0 and math.isfinite(margin_left):
            caps["margin"] = margin_left / o.margin_per_notional * o.capital_per_notional
        binding = min(caps, key=caps.get)
        cap_need = max(caps[binding], 0.0)
        if math.isfinite(margin_left) and o.margin_per_notional > 0:
            margin_left -= cap_need / o.capital_per_notional * o.margin_per_notional
        left -= cap_need
        if o.risk_group:
            groups[o.risk_group] = groups.get(o.risk_group, 0.0) + cap_need
        alloc.append({**pr, "capital": cap_need, "binding": binding})
    used = sum(a["capital"] for a in alloc)
    book_ann_excess = sum(a["capital"] * a["ann_excess_on_capital"] for a in alloc) / capital if capital else 0.0
    return {"allocations": alloc, "capital_used": used, "cash": capital - used, "book_ann_excess": book_ann_excess}
