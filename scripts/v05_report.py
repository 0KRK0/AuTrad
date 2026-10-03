"""Write reports/v0.5_<ts>/report.md from trackB.json, trackC.json and the forward ledger, and log every v0.5 test
in research/trials.jsonl (trials + one experiment record each)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from lab.registry import Registry, git_sha


def f(x, fmt="{:.2f}"):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def pct(x, d=1):
    return f(None if x is None else x * 100, "{:+.%df}%%" % d)


def write_report(out: Path) -> None:
    B = json.loads((out / "trackB.json").read_text())
    C = json.loads((out / "trackC.json").read_text())
    fwd_status = Path("forward/rev1/status.md").read_text(encoding="utf-8") if Path("forward/rev1/status.md").exists() else ""
    spec = json.loads(Path("forward/rev1/spec.json").read_text()) if Path("forward/rev1/spec.json").exists() else {}
    from forward.ledger import Ledger
    from forward.rev1 import summarize
    L = Ledger("forward/rev1/ledger.jsonl")
    fs = summarize(L.records(), spec) if spec else {}
    sha = B["prereg_sha256"]
    o = B["objects"]
    s, lv, sn = o["low_vol_sleeve"], o["LVOL_tercile"], o["sector_neutral_low_vol"]
    t = C["tests"]
    t1 = C["T1_avoid_overlay"]
    A = json.loads((out / "rev1_research.json").read_text())
    a_sub = sum(v > 0 for v in A["1.0x"]["subperiods"].values())
    Lines = [
        "# AuTrad AI v0.5 — alpha validation\n",
        f"Pre-registration `research/preregistration_v0.5.md` (sha256 `{sha[:16]}`, logged before any result; Amendments 1–2 are "
        "implementation only) · prices exchange-anchored, cost model `india_zerodha_2026_09` · research data to 29 Sep 2026 · "
        f"forward ledger head `{L.head()[:16]}` · git `{git_sha()}`\n",
        "v0.5 added **no new factors**. It validates the one candidate (REV1, forward), takes apart the low-volatility "
        "\"alpha\", and tests the earnings-reaction finding under frozen rules.\n",
        "## Scorecard: four separate questions\n",
        "Statistical = distinguishable from noise (pre-registered test). Economic = big enough after costs (portfolios ≥ +2 %/yr). "
        "Tradable = long-only, survives 2× costs, ≥ 3 of 4 subperiods, capacity ≥ ₹1 Cr. Forward = held up on data nobody had seen. "
        "None of these axes stands in for another.\n",
        "| Candidate | Statistical | Economic | Tradable | Forward | Verdict |", "|---|---|---|---|---|---|",
        f"| REV1 one-month reversal | ✓ IC t 3.32, Holm p 0.020 (v0.4); but portfolio excess t only 0.87 | ✓ {pct(A['1.0x']['net_excess'], 2)}/yr net vs equal-weight | "
        f"~ {pct(A['2.0x']['net_excess'], 2)}/yr at 2× costs; {a_sub} of 4 subperiods positive; 16.8× turnover | "
        f"running since 1 Oct 2026 ({fs.get('sessions', 0)} session(s)) | candidate under forward test; cost-fragile |",
        f"| Low-vol sleeve (v0.3) | {s['scorecard']['statistical']} M1 alpha {pct(s['M1']['alpha_ann'])} (t {f(s['M1']['alpha_t'])}) | "
        f"{s['scorecard']['economic']} {pct(s['net_excess_vs_beta_matched']['ann'])}/yr vs beta-matched | {s['scorecard']['tradable']} | not started | **{B['decision'].split(' (')[0]}** |",
        f"| LVOL tercile | {lv['scorecard']['statistical']} M1 alpha {pct(lv['M1']['alpha_ann'])} (t {f(lv['M1']['alpha_t'])}) | "
        f"{lv['scorecard']['economic']} {pct(lv['net_excess_vs_beta_matched']['ann'])}/yr vs beta-matched | {lv['scorecard']['tradable']} | not started | diagnostic |",
        f"| Sector-neutral low-vol | {sn['scorecard']['statistical']} M1 alpha {pct(sn['M1']['alpha_ann'])} (t {f(sn['M1']['alpha_t'])}) | "
        f"{sn['scorecard']['economic']} {pct(sn['net_excess_vs_beta_matched']['ann'])}/yr vs beta-matched | {sn['scorecard']['tradable']} | not started | diagnostic |",
        f"| Bad earnings reaction → avoid | {C['scorecard']['statistical']} C1–C3 Holm p ≤ {f(max(t[k]['holm_p'] for k in ('C1','C2','C3')), '{:.3f}')} | "
        f"{C['scorecard']['economic']} avoid-overlay {pct(t1['1.0x']['value_ann'], 2)}/yr | {C['scorecard']['tradable']} ({pct(t1['2.0x']['value_ann'], 2)}/yr at 2× costs) | not started | **{C['decision'].split(' (')[0]}** |",
        "",
        "## Track A — REV1 forward paper test (frozen)\n",
        "Running exactly as frozen in v0.4: top third of point-in-time NIFTY 50 members by minus last-month return, equal weight, "
        "month-end decision, next-open fill, full Indian costs, ₹10 lakh paper. Every decision, fill and daily mark is appended to a "
        "hash-chained ledger (`forward/rev1/ledger.jsonl`), and the code that defines the strategy is hashed. Any change voids the test. "
        "Research and forward results are kept apart.\n",
        "Before the first record, an independent review of the runner found that the 'truncated' panel could still be influenced "
        "by later bars through the corporate-action fit, that the VOID rule could never fire, and that month-ends depended on one "
        "vendor calendar. All three were fixed before initialisation (Amendment 2).\n",
        f"**Honest expectation:** at REV1's historical information ratio (~0.26), forward data would need ~59 years to reach t = 2. "
        "So the forward test can catch REV1 breaking, and it checks the pipeline, but it cannot prove the edge. First review after "
        "the October 2027 month-end; REJECT if z < −2 against history, or if turnover runs above 1.5× the backtest.\n",
        "Current state (also in `forward/rev1/status.md`):\n",
        "\n".join(("#" + x) if x.startswith("## ") else x for x in fwd_status.split("\nA forward test")[0].splitlines()[2:]) + "\n",
        "## Track B — what the low-volatility \"alpha\" really is\n",
        "v0.4 found +3.2 %/yr unexplained return in the v0.3 `low_vol` sleeve (15 lowest-volatility NIFTY 50 stocks). v0.5 explains "
        "it with factors defined in advance: market (NIFTYBEES), EWCW (equal-weight minus cap-weight NIFTY 50: the small-name tilt of "
        "any equal-weight portfolio), SMB (small minus big by traded value) and MOM (12-1 momentum). Factor premia over the sample: "
        + ", ".join(f"{k} {pct(v)}" for k, v in B["factor_means_ann"].items()) + " a year.\n",
        "| | Low-vol sleeve | LVOL tercile | Sector-neutral low-vol |", "|---|---|---|---|",
    ]
    rows = [
        ("Return over cash", lambda x: pct(x["M1"]["waterfall_ann"]["total_over_cash"])),
        ("… from market beta", lambda x: pct(x["M1"]["waterfall_ann"]["MKT"])),
        ("… from EWCW (small-name tilt)", lambda x: pct(x["M1"]["waterfall_ann"]["EWCW"])),
        ("… from SMB", lambda x: pct(x["M1"]["waterfall_ann"]["SMB"])),
        ("… from momentum", lambda x: pct(x["M1"]["waterfall_ann"]["MOM"])),
        ("**… unexplained (M1 alpha)**", lambda x: f"**{pct(x['M1']['alpha_ann'])}** (t {f(x['M1']['alpha_t'])})"),
        ("v0.4-style alpha (M0: MKT, MOM, LVOL)", lambda x: f"{pct(x['M0']['alpha_ann'])} (t {f(x['M0']['alpha_t'])})"),
        ("Beta ex-ante / realized", lambda x: f"{f(x['ex_ante_beta_avg'])} / {f(x['realized_beta'])}"),
        ("Net excess vs beta-matched benchmark", lambda x: f"{pct(x['net_excess_vs_beta_matched']['ann'])} (t {f(x['net_excess_vs_beta_matched']['t'])})"),
        ("Net excess vs equal-weight NIFTY 50", lambda x: f"{pct(x['net_excess_vs_EW']['ann'])} (t {f(x['net_excess_vs_EW']['t'])})"),
        ("Net excess vs NIFTYBEES", lambda x: f"{pct(x['net_excess_vs_NIFTYBEES']['ann'])} (t {f(x['net_excess_vs_NIFTYBEES']['t'])})"),
        ("Brinson vs EW: sector allocation / stock selection (gross)", lambda x: f"{pct(x['brinson_ann']['allocation'])} / {pct(x['brinson_ann']['selection'])}"),
        ("Avg largest sector · months FMCG is largest", lambda x: f"{x['concentration']['avg_max_sector']:.0%} · {x['concentration']['top_sector_share_of_months'].get('fmcg', 0):.0%}"),
        ("Sector HHI (universe)", lambda x: f"{f(x['concentration']['sector_hhi'])} ({f(x['concentration']['universe_sector_hhi'])})"),
        ("Turnover · cost drag at ₹10L / ₹1 Cr", lambda x: f"{f(x['costs']['turnover_per_year'], '{:.1f}')}×/yr · {pct(x['costs']['cost_drag_10L'], 2)} / {pct(x['costs']['cost_drag_1Cr'], 2)}"),
        ("vs beta-matched at 2× / 3× costs", lambda x: f"{pct(x['cost_shock']['2.0']['vs_beta_matched_ann'])} / {pct(x['cost_shock']['3.0']['vs_beta_matched_ann'])}"),
        ("Up / down capture vs NIFTYBEES", lambda x: f"{f(x['capture']['up'])} / {f(x['capture']['down'])}"),
        ("vs beta-matched: NIFTY above / below 200-day avg", lambda x: f"{pct(x['regimes']['bull=True']['vs_beta_matched']['ann'])} ({x['regimes']['bull=True']['vs_beta_matched']['months']} m) / "
                                                                        f"{pct(x['regimes']['bull=False']['vs_beta_matched']['ann'])} ({x['regimes']['bull=False']['vs_beta_matched']['months']} m)"),
    ]
    for name, fn in rows:
        Lines.append(f"| {name} | {fn(s)} | {fn(lv)} | {fn(sn)} |")
    Lines += ["", "**Crashes and stress (cumulative):**\n", "| Episode | Low-vol sleeve | NIFTYBEES | Equal-weight |", "|---|---|---|---|"]
    for k, v in s["episodes"].items():
        Lines.append(f"| {k} | {pct(v['object'])} | {pct(v['NIFTYBEES'])} | {pct(v['EW'])} |")
    Lines += ["", "**Subperiods (low-vol sleeve):**\n", "| Period | M1 alpha (t) | vs beta-matched | vs equal-weight |", "|---|---|---|---|"]
    for k, v in s["subperiods"].items():
        Lines.append(f"| {k} | {pct(v['M1_alpha_ann'])} ({f(v['M1_alpha_t'])}) | {pct(v['vs_beta_matched_ann'])} | {pct(v['vs_EW_ann'])} |")
    Lines += ["", "**Pre-registered rule for \"serious candidate\" (all must hold):** " + "; ".join(f"{k} {'✓' if v else '✗'}" for k, v in B["criteria"].items())
              + f" → **{B['decision']}**.\n",
              f"Size proxy check: traded-value rank vs market-cap rank (2019+), mean Spearman {f(B['size_proxy_vs_mcap_spearman'].get('mean'))} "
              f"(min {f(B['size_proxy_vs_mcap_spearman'].get('min'))}). Traded value partly measures trading activity, which is linked to volatility, "
              "so SMB here is part size, part low-turnover.\n",
              "## Track C — bad earnings reactions\n",
              f"{C['events']} results announcements by point-in-time members ({C['span']}). \"Bad\" = below the lower third of all earlier "
              "reactions on the same measure (past-only). Drift = market-relative return from the next open (equal-weight members over the same "
              "window). t-stats are clustered by reaction day.\n",
              "| Test | n | Mean CAR20 | t | Holm p | Result |", "|---|---|---|---|---|---|"]
    names = {"C1": "Bad market-adjusted reaction", "C2": "Bad sector-adjusted reaction", "C3": "Bad absolute reaction"}
    for k in ("C1", "C2", "C3"):
        v = t[k]
        Lines.append(f"| {k} {names[k]} | {v['n']} | {f(v['mean_bps'], '{:+.0f}')} bps | {f(v['t'])} | {f(v['holm_p'], '{:.3f}')} | "
                     f"{'significant' if v['holm_p'] < 0.05 else 'not significant'} |")
    for k in ("C4", "C5", "C6"):
        v = t[k]
        Lines.append(f"| {k} {v['split']} (within bad-MKT) | {v['n1']} vs {v['n0']} | {f(v['mean1_bps'], '{:+.0f}')} vs {f(v['mean0_bps'], '{:+.0f}')} bps | "
                     f"{f(v['t'])} | {f(v['holm_p'], '{:.3f}')} | {'significant' if v['holm_p'] < 0.05 else 'no difference shown'} |")
    cl = C["bad_MKT_cells"]
    Lines += ["", "Drift by horizon (bad-MKT): " + ", ".join(f"+{h} sessions {f(v['mean_bps'], '{:+.0f}')} bps (t {f(v['t'])})" for h, v in t["C1"]["by_h"].items())
              + ". Stability: " + ", ".join(f"{k} {f(cl[k]['mean_bps'], '{:+.0f}')} bps (t {f(cl[k]['t'])}, n {cl[k]['n']})"
                                            for k in ("first half", "second half", "bull", "bear", "high vol", "low vol")) + ".\n",
              "By year: " + ", ".join(f"{k} {f(v['mean_bps'], '{:+.0f}')}" for k, v in cl.items() if k.isdigit()) + " bps.\n",
              f"**Can a long-only portfolio use it?** Avoid-overlay on the equal-weight NIFTY 50: sell a stock at the open after a bad reaction, "
              f"park the money in NIFTYBEES, buy back 20 sessions later. {t1['events_used']} events since {t1['active_from']}. Value "
              f"{pct(t1['1.0x']['value_ann'], 2)}/yr (t {f(t1['1.0x']['value_t'])}) at 1× costs, {pct(t1['2.0x']['value_ann'], 2)}/yr at 2× costs. "
              "The drift is real, but each name is only ~2 % of the portfolio, so selling and re-buying costs most of what it saves.\n",
              f"Short diagnostic (not a proposal): stock futures, 20 sessions, {f(C['T2_short_diagnostic']['net_bps_per_event'], '{:+.0f}')} bps per event "
              f"after a {f(C['T2_short_diagnostic']['futures_round_trip_bps'], '{:.0f}')} bps round trip (t {f(C['T2_short_diagnostic']['t'])}).\n",
              "**Pre-registered candidate rule:** " + "; ".join(f"{k} {'✓' if v else '✗'}" for k, v in C["criteria"].items()) + f" → **{C['decision']}**.\n",
              "## What this means\n",
              "- **REV1 remains the only candidate, and it is cost-fragile.** Its +2.3 %/yr disappears at double costs and only "
              "2 of 4 subperiods were positive. It is now running forward, untouched. Its expected edge is too small to prove "
              "quickly, so it cannot earn capital on its own.",
              "- **Low volatility is mostly beta, not alpha.** The sleeve took ~0.7–0.8 of market risk and was heavily in FMCG. Its "
              "\"alpha\" depended on which factors it was measured against, and under the pre-registered model it is not significant. "
              "It lagged the equal-weight index, and its whole beta-adjusted edge came from the months when NIFTY was below its 200-day "
              "average. Making it sector-neutral removes most of the defensiveness (down-capture 0.73 → 0.91) but leaves about the same "
              "beta-adjusted excess, so the FMCG tilt was not the source either; nothing is significant either way. It also lost in "
              "2024–26, including the Oct 2024 – Feb 2025 sell-off.",
              "- **Bad earnings reactions are the strongest effect AuTrad has measured.** The drift holds on market-, sector- and "
              "absolute-adjusted measures. But surprise size, volume and prior momentum don't sharpen it, and a stand-alone long-only "
              "filter doesn't survive double costs. The cheapest place to use it is inside trades AuTrad is already making, e.g. "
              "not buying a reversal candidate that just had a bad results reaction. That is a factor interaction, so it needs its own "
              "pre-registration later; it was not tested here.",
              "- Live trading stays disabled.\n",
              "## Known limitations\n",
              "- About 50 stocks; each test has limited power, and 2024–26 had already been seen.",
              "- No analyst consensus: SUE uses a seasonal random walk, and only 205 bad-reaction events have a matching SUE.",
              "- SMB uses traded value as the size measure (it overlaps with low turnover); EWCW captures the cleaner small-name tilt.",
              "- Fills use the vendor's dividend-adjusted open; taxes are not deducted.\n"]
    (out / "report.md").write_text("\n".join(Lines) + "\n", encoding="utf-8")

    # ledger: every v0.5 test is a trial with an experiment record
    reg = Registry("research/trials.jsonl")
    done = {(r.get("version"), r.get("strategy")) for r in (json.loads(x) for x in open("research/trials.jsonl")) if r.get("version") == "v0.5"}
    trials = [("lowvol:low_vol_sleeve", s, B["decision"]), ("lowvol:LVOL_tercile", lv, "diagnostic"), ("lowvol:sector_neutral", sn, "diagnostic")]
    recs, exps = [], []
    n_prior = reg.count()
    for name, x, dec in trials:
        if ("v0.5", name) in done:
            continue
        recs.append({"version": "v0.5", "strategy": name, "config": {"definition": "preregistered v0.5 Track B"}})
        exps.append({"kind": "experiment", "version": "v0.5", "experiment_id": hashlib.sha1(f"{out}{name}".encode()).hexdigest()[:12],
                     "git_sha": git_sha(), "preregistration_sha256": sha, "features": name, "model": "M1 = MKT+EWCW+SMB+MOM",
                     "alpha_ann": x["M1"]["alpha_ann"], "alpha_t": x["M1"]["alpha_t"], "net_return": x["net_excess_vs_beta_matched"]["ann"],
                     "number_of_prior_trials": n_prior, "scorecard": x["scorecard"], "decision": dec})
    for k, v in t.items():
        name = f"earnings:{k}"
        if ("v0.5", name) in done:
            continue
        recs.append({"version": "v0.5", "strategy": name, "config": {"definition": "preregistered v0.5 Track C"}})
        exps.append({"kind": "experiment", "version": "v0.5", "experiment_id": hashlib.sha1(f"{out}{name}".encode()).hexdigest()[:12],
                     "git_sha": git_sha(), "preregistration_sha256": sha, "features": name, "t": v["t"], "holm_p": v["holm_p"],
                     "mean_bps": v.get("mean_bps", v.get("diff_bps")), "number_of_prior_trials": n_prior,
                     "decision": "significant" if v["holm_p"] < 0.05 else "not significant"})
    if ("v0.5", "earnings:T1_avoid_overlay") not in done:
        recs.append({"version": "v0.5", "strategy": "earnings:T1_avoid_overlay", "config": {"band": 0.01, "hold": 20}})
        exps.append({"kind": "experiment", "version": "v0.5", "experiment_id": hashlib.sha1(f"{out}T1".encode()).hexdigest()[:12],
                     "git_sha": git_sha(), "preregistration_sha256": sha, "features": "T1 avoid-overlay", "net_return": t1["1.0x"]["value_ann"],
                     "net_return_2x_costs": t1["2.0x"]["value_ann"], "number_of_prior_trials": n_prior, "scorecard": C["scorecard"], "decision": C["decision"]})
    if recs:
        reg.add(recs)
        reg.add(exps)
    print(f"report written to {out / 'report.md'}; logged {len(recs)} trials")
