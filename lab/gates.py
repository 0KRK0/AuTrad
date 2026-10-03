"""Capital-efficiency gate (research/preregistration_v0.7.md, section 1): strictly sequential; the first failure rejects.

    statistical -> economic after costs -> after-tax (estimate) -> 2x costs -> capacity -> stability -> forward
Only a candidate that reaches the end may be *considered* for capital by the Main Head, and only with owner approval.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STEPS = ["statistical", "economic_after_costs", "after_tax_estimate", "cost_2x", "capacity", "stability", "forward"]


@dataclass
class GateInput:
    statistical_pass: bool
    net_excess: float | None             # per year, 1x costs, vs the pre-registered benchmark (None = not measured)
    post_tax_cagr: float | None          # after-tax estimate (v0.6 model)
    benchmark_post_tax_cagr: float | None
    net_excess_2x: float | None
    cost_drag_10L: float | None
    cost_drag_1Cr: float | None
    net_excess_1Cr: float | None
    subperiod_excess: list[float] = field(default_factory=list)
    forward: str = "not started"         # not started | running | passed | failed


def _p(v) -> str:
    return "n/a" if v is None else f"{v:+.2%}"


def capital_efficiency_gate(x: GateInput, min_excess: float = 0.02) -> dict:
    """A step whose inputs were never measured fails (with 'n/a'), but only matters if it is reached."""
    ok = lambda v: v is not None  # noqa: E731
    checks = {
        "statistical": (x.statistical_pass, "pre-registered primary test significant after Holm, right sign"),
        "economic_after_costs": (ok(x.net_excess) and x.net_excess >= min_excess, f"net excess {_p(x.net_excess)}/yr (needs >= {min_excess:+.0%})"),
        "after_tax_estimate": (ok(x.post_tax_cagr) and ok(x.benchmark_post_tax_cagr) and x.post_tax_cagr > x.benchmark_post_tax_cagr,
                               f"post-tax {_p(x.post_tax_cagr)} vs benchmark {_p(x.benchmark_post_tax_cagr)} (estimate)"),
        "cost_2x": (ok(x.net_excess_2x) and x.net_excess_2x > 0, f"net excess at 2x costs {_p(x.net_excess_2x)}/yr"),
        "capacity": (ok(x.net_excess_1Cr) and ok(x.cost_drag_1Cr) and ok(x.cost_drag_10L) and x.net_excess_1Cr > 0
                     and x.cost_drag_1Cr <= 1.25 * max(x.cost_drag_10L, 1e-9),
                     f"at Rs 1 Cr: net excess {_p(x.net_excess_1Cr)}, cost drag {_p(x.cost_drag_1Cr)} vs {_p(x.cost_drag_10L)} at Rs 10 lakh"),
        "stability": (len(x.subperiod_excess) > 0 and sum(v > 0 for v in x.subperiod_excess) >= 2 / 3 * len(x.subperiod_excess) - 1e-9,
                      f"{sum(v > 0 for v in x.subperiod_excess)} of {len(x.subperiod_excess)} subperiods positive"),
        "forward": (x.forward == "passed", f"forward test {x.forward}"),
    }
    out, verdict = [], None
    for s in STEPS:
        ok, why = checks[s]
        if verdict is None:
            out.append({"step": s, "result": "PASS" if ok else ("PENDING" if s == "forward" and x.forward in ("not started", "running") else "FAIL"), "detail": why})
            if not ok:
                verdict = (f"REJECTED at {s}" if out[-1]["result"] == "FAIL" else
                           "ELIGIBLE FOR A FROZEN FORWARD TEST" if x.forward == "not started" else "FORWARD TEST RUNNING")
        else:
            out.append({"step": s, "result": "NOT REACHED", "detail": why})
    return {"steps": out, "verdict": verdict or "MAY BE CONSIDERED FOR CAPITAL (owner approval required)"}
