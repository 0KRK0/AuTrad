"""Research dashboard (v1.1; v1.4 adds the data clock): what the research machine has tested, rejected and kept, from research/trials.jsonl.

    python -m scripts.research_dashboard            -> research/dashboard.md

Discovery (hypotheses per version, rejected, surviving), statistics (t, Holm p) and economics (net return field as
logged) per experiment, plus whether each v1.1+ row states its economic mechanism. Proven alpha is the number of
experiments whose decision starts with PASS and has also passed a forward paper test (none yet).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

PATH = Path("research/trials.jsonl")


def load() -> list[dict]:
    return [json.loads(x) for x in PATH.open() if x.strip()]


def classify(dec: str | None) -> str:
    d = (dec or "").upper()
    if d.startswith("PASS"):
        return "surviving"
    if "REJECT" in d or "NOT SIGNIFICANT" in d or "WIPED" in d or "FAIL" in d:
        return "rejected"
    if "SIGNIFICANT" in d:
        return "significant (no economic test)"
    return "other"


def build() -> str:
    rows = load()
    ex = [r for r in rows if r.get("kind") == "experiment"]
    pre = [r for r in rows if r.get("kind") == "preregistration"]
    by_v = defaultdict(list)
    for r in ex:
        by_v[r.get("version", "?")].append(r)
    lines = ["# AuTrad research dashboard", "", f"Generated from `{PATH}`: {len(pre)} pre-registration records, {len(ex)} experiment rows.", "",
             "## Discovery", "", "| Version | Experiments | Rejected | Surviving | Significant only | With mechanism |", "|---|---|---|---|---|---|"]
    tot = Counter()
    for v in sorted(by_v, key=lambda x: [int(p) if p.isdigit() else 0 for p in x.lstrip("v").split(".")]):
        c = Counter(classify(r.get("decision")) for r in by_v[v])
        mech = sum(1 for r in by_v[v] if str(r.get("economic_mechanism", "")).strip())
        tot.update(c)
        lines.append(f"| {v} | {len(by_v[v])} | {c['rejected']} | {c['surviving']} | {c['significant (no economic test)']} | {mech} |")
    lines += ["", f"**Surviving the full gate: {tot['surviving']}. Proven alpha (gate + forward paper test): 0.**", "",
              "## Latest experiments", "", "| Version | Experiment | t | Holm p | Net (as logged) | Decision | Mechanism |", "|---|---|---|---|---|---|---|"]
    fmt = lambda x: "" if x is None else (f"{x:.3g}" if isinstance(x, (int, float)) else str(x))
    for r in ex[-25:]:
        lines.append(f"| {r.get('version')} | {r.get('experiment_id')} | {fmt(r.get('t'))} | {fmt(r.get('holm_p'))} | {fmt(r.get('net_return'))} | "
                     f"{r.get('decision', '')} | {r.get('economic_mechanism', '')} |")
    try:  # v1.4: forward data clock (counts only)
        from forward.data_clock import clock, markdown
        lines += ["", markdown(clock())]
    except Exception as e:  # pragma: no cover - the clock must never break the dashboard
        lines += ["", f"_Data clock unavailable: {e}_"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    out = Path("research/dashboard.md")
    out.write_text(build())
    print(out.read_text()[:3000])
