"""Model qualification scorecard: the automatic gate between research and paper trading.

All checks use the stitched walk-forward (out-of-sample) record, net of Indian costs.
Every gate must pass for QUALIFIED_FOR_PAPER; otherwise the verdict is RESEARCH, and a
system already in paper or live is demoted automatically.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

THRESHOLDS = {
    "min_sharpe": 0.5,
    "max_drawdown": 0.20,
    "min_positive_year_share": 0.6,
    "max_random_pvalue": 0.05,
    "min_deflated_sharpe": 0.95,
    "max_cost_share": 0.5,
    "min_profit_factor": 1.1,
    "min_trades": 100,
}


@dataclass
class Gate:
    name: str
    passed: bool
    value: str
    target: str


def evaluate(m: dict, *, best_baseline: tuple[str, float], random_p: float, dsr: float, positive_year_share: float,
             trades: dict, violations: int, after_tax_cagr: float, after_tax_cash: float) -> dict:
    T = THRESHOLDS
    bname, bsharpe = best_baseline
    g = [
        Gate("Beats cash (liquid fund) after costs", m["excess_cagr"] > 0, f"{m['excess_cagr']:+.2%}/yr vs cash", "> 0"),
        Gate("Beats cash after tax", after_tax_cagr > after_tax_cash, f"{after_tax_cagr:.2%} vs {after_tax_cash:.2%}", "higher"),
        Gate("Sharpe ratio (excess of cash)", m["sharpe"] >= T["min_sharpe"], f"{m['sharpe']:.2f}", f">= {T['min_sharpe']}"),
        Gate(f"Beats best simple baseline ({bname})", m["sharpe"] > bsharpe, f"{m['sharpe']:.2f} vs {bsharpe:.2f}", "higher Sharpe"),
        Gate("Beats random traders with same risk", random_p < T["max_random_pvalue"], f"p = {random_p:.3f}", f"< {T['max_random_pvalue']}"),
        Gate("Deflated Sharpe (charges for configs tried)", dsr >= T["min_deflated_sharpe"], f"{dsr:.2f}", f">= {T['min_deflated_sharpe']}"),
        Gate("Profitable in most test years", positive_year_share >= T["min_positive_year_share"],
             f"{positive_year_share:.0%} of years", f">= {T['min_positive_year_share']:.0%}"),
        Gate("Max drawdown", m["max_drawdown"] <= T["max_drawdown"], f"{m['max_drawdown']:.1%}", f"<= {T['max_drawdown']:.0%}"),
        Gate("Costs eat less than half of gross profit", m.get("cost_share_of_gross_profit", 1e9) <= T["max_cost_share"],
             f"{m.get('cost_share_of_gross_profit', float('nan')):.0%}", f"<= {T['max_cost_share']:.0%}"),
        Gate("Profit factor", trades.get("profit_factor", 0) >= T["min_profit_factor"],
             f"{trades.get('profit_factor', 0):.2f}", f">= {T['min_profit_factor']}"),
        Gate("Enough trades to judge", trades.get("trades", 0) >= T["min_trades"], f"{trades.get('trades', 0)}", f">= {T['min_trades']}"),
        Gate("Risk limit violations", violations == 0, str(violations), "0"),
    ]
    passed = all(x.passed for x in g)
    return {
        "verdict": "QUALIFIED_FOR_PAPER" if passed else "RESEARCH",
        "passed": sum(x.passed for x in g), "total": len(g),
        "gates": [asdict(x) for x in g],
        "action": ("Promote to real-time paper trading" if passed
                   else "Stay in research. Any paper or live deployment of this model is demoted to RESEARCH."),
    }
