"""Write reports/v0.6-r1_<ts>/report.md from v06.json, tenth_man_calibration.json and shadow_memo.md, and log the
v0.6 tests in research/trials.jsonl."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab.registry import Registry, git_sha


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def write_report(out: Path) -> None:
    R = json.loads((out / "v06.json").read_text())
    K = json.loads((out / "tenth_man_calibration.json").read_text())
    S, T, ex, os_ = R["stats"], R["tests"], R["execution"], R["open_stats"]
    wf = K["walk_forward"]
    memo = (out / "shadow_memo.md").read_text(encoding="utf-8") if (out / "shadow_memo.md").exists() else ""
    head_table = "\n".join(memo.split("\n")[4:24]) if memo else "(no shadow memo)"
    L = ["# AuTrad AI v0.6 — execution prices, taxes, REV1 × earnings veto, and the advisory agents\n",
         f"Pre-registration `research/preregistration_v0.6.md` (logged before any result; Amendment 1 after an independent review, no hypothesis "
         f"changes) · prereg sha256 `{R['prereg_sha256'][:16]}` · git `{git_sha()}` · REV1's forward test untouched\n",
         "## 1. Execution prices: exchange opens instead of Yahoo opens\n",
         f"NSE's official bhavcopy for all {os_['exchange'] + os_['vendor_fallback']:,} stock-days (2015–2026, downloaded on your machine). "
         f"Yahoo's open matches the exchange open almost exactly: median difference {os_['abs_diff_bps']['median']:.4f} bps. "
         f"Only {os_['share_above_10bps']:.2%} of stock-days differ by more than 10 bps, and {os_['share_above_50bps']:.2%} by more than 50 bps. "
         f"{os_['discontinuity_fallback']} stock-days around splits keep the vendor open (continuity guard). ETFs keep vendor opens: NSE's "
         "official ETF opens are often stray auction prints (more than 50 bps off on about 7 % of NIFTYBEES days).\n",
         f"Frozen REV1, full sample, net excess over the equal-weight index: vendor opens {pct(ex['REV1_vendor_open'], 2)}/yr → exchange opens "
         f"{pct(ex['REV1_exchange_open'], 2)}/yr; at double costs {pct(ex['REV1_exchange_open_2x'], 2)}/yr. **The open-price source was not "
         "the problem; costs are.**\n",
         "## 2. After tax (India: STCG 20 %, LTCG 12.5 % above ₹1.25 lakh, older rates by date, set-off and carry-forward)\n",
         f"Window {R['active_from']} to Sep 2026. Pre-tax CAGR → post-tax CAGR (open positions liquidated and taxed at the end):\n",
         "| Portfolio | Pre-tax CAGR | Post-tax CAGR | Turnover / yr | Max drawdown |", "|---|---|---|---|---|"]
    for k, name in (("REV1", "REV1 (frozen)"), ("REPLACE", "REV1 + veto, replace"), ("SKIP", "REV1 + veto, skip"), ("EW", "Equal-weight NIFTY 50"),
                    ("NIFTYBEES", "NIFTYBEES held")):
        s = S[k]
        L.append(f"| {name} | {pct(s['cagr'])} | {pct(s['post_tax_cagr_liquidated'])} | {s['turnover_per_year']:.1f}× | {pct(s['max_dd'], 0)} |")
    gap_pre = S["REV1"]["cagr"] - S["EW"]["cagr"]
    gap_post = S["REV1"]["post_tax_cagr_liquidated"] - S["EW"]["post_tax_cagr_liquidated"]
    L += ["", f"**REV1 beats the equal-weight index by {pct(gap_pre)}/yr before tax but trails it by {-gap_post * 100:.1f}%/yr after tax** "
          "(17× turnover realises short-term gains at 15–20 % every year; the index defers most of its gains). REV1 is still ahead "
          "of NIFTYBEES after tax, so it is not TAX-NEGATIVE under the pre-registered definition. But tax removes the edge that "
          "mattered.\n",
          "## 3. Does a bad earnings reaction add information to REV1?\n",
          f"Months {R['active_from']}..2026-08 ({R['vetoed_per_month']['months']} months). A recent bad reaction (BAD20) vetoed "
          f"{R['vetoed_per_month']['BAD20_total']} REV1 picks in {R['vetoed_per_month']['BAD20_months_with_veto']} months "
          f"(~{R['vetoed_per_month']['BAD20_mean']:.1f} a month).\n",
          "| Test | Estimate | t | Holm p | Result |", "|---|---|---|---|---|",
          f"| H1 Fama–MacBeth: BAD20 beyond the REV1 rank | {T['H1']['mean_coef_bps']:+.0f} bps / month ({T['H1']['months']} months) | {T['H1']['t']:.2f} | {T['H1']['holm_p']:.3f} | no incremental information shown |",
          f"| H2 REPLACE − REV1 | {pct(T['H2']['mean_diff_ann'])}/yr | {T['H2']['t']:.2f} | {T['H2']['holm_p']:.3f} | not significant |",
          f"| H3 SKIP − REV1 | {pct(T['H3']['mean_diff_ann'])}/yr | {T['H3']['t']:.2f} | {T['H3']['holm_p']:.3f} | not significant |",
          "", "REPLACE − REV1 by subperiod: " + ", ".join(f"{k} {pct(v)}" for k, v in T["H2"]["subperiods"].items()) +
          f". BAD60 sensitivity: {pct(R['sensitivity_BAD60_replace']['ann'])}/yr (t {R['sensitivity_BAD60_replace']['t']:.2f}).\n",
          "| | REV1 | REPLACE | SKIP |", "|---|---|---|---|"]
    for lab, key, f_ in (("Net excess vs EW (1× costs)", "excess_vs_EW", pct), ("… at 2× costs", "excess_vs_EW_2x", pct), ("… at ₹1 Cr", "excess_vs_EW_1Cr", pct),
                         ("Sharpe", "sharpe", lambda x: f"{x:.2f}"), ("Turnover / yr", "turnover_per_year", lambda x: f"{x:.1f}×"),
                         ("Post-tax excess vs NIFTYBEES", "post_tax_excess_vs_NIFTYBEES", pct)):
        L.append(f"| {lab} | {f_(S['REV1'][key])} | {f_(S['REPLACE'][key])} | {f_(S['SKIP'][key])} |")
    L += ["", "**REV2 rule (all must hold):** " + "; ".join(f"{k} {'✓' if v else '✗'}" for k, v in R["criteria"].items()) + f" → **{R['decision']}**.\n",
          "REPLACE looks better on paper (+1.7 %/yr over REV1, positive at double costs). But the difference is not significant (t 1.15), "
          "it was negative in 2022–23, and the stock-level test (H1) finds no extra information in the earnings flag once the "
          "REV1 rank is known. Most likely this is noise, so REV1 stays as it is and no REV2 is started.\n",
          "## 4. The advisory agents (shadow, no order authority)\n",
          "**Main Head → Company Intelligence and 10th Man → Main Head → Risk Gate.** The agents never talk to each other. The 10th Man "
          "gets only a sanitised request (action, stock, size, horizon, the thesis claims and their evidence ids), reads its own copy "
          "of the external documents, and holds no reference to any agent, the gate or a broker. Isolation is tested, including "
          "transitive imports. Every fact quotes its source verbatim. AuTrad's own numbers are tier-9 data extracts: they support "
          "DERIVED statements but can never count as tier-1/2 proof.\n",
          f"**10th Man failure probability:** walk-forward over {wf['n']:,} historical REV1 picks ({wf['months']} months). Brier score: model "
          f"{wf['brier_model']:.4f} vs plain base rate {wf['brier_base']:.4f}, **skill {wf['skill']:+.3f}**. The pre-registered flags (recent bad "
          "reaction, extreme loss, volatility, bear market, data quality) do not predict which REV1 picks fail. So, as pre-registered, "
          f"the 10th Man reports the base rate only (~{wf['fail_rate']:.0%}). It cannot use a language model to invent a number.\n",
          "**First shadow memo: REV1's 30 Sep 2026 picks** (`forward/shadow/memo_2026-09.md`):\n", head_table, "",
          "Each dossier holds:\n",
          "- the PIT index membership;",
          "- the price path (derived from exchange-anchored data);",
          "- four quarters of NSE results with year-on-year changes, each quoting the exchange filing;",
          "- every results reaction;",
          "- the latest NSE announcements and upcoming board meetings (for example TCS results on 8 Oct, inside the holding month);",
          "- news headlines, labelled by source tier (reputable publishers tier 5, others tier 8; a headline is a claim, not a fact);",
          "- red flags, and what is unknown.\n",
          "**The review loop worked again.** The first memo challenged 9 of 16 names. The cause was a defect: the 10th Man's own "
          "computation (\"the loss is sector-wide\") had been stored as tier-2 evidence. I then changed the Head's thesis wording, and "
          "all 16 challenges disappeared. The independent review caught both the defect and the wording edit, which changed the "
          "10th Man's input after its output was known. Fixes:\n",
          "- the evidence tiers;",
          "- the original thesis restored;",
          "- the thesis template is now hashed in every memo;",
          "- all three memos stay in the ledger, the first two marked superseded.\n",
          "Final memo: no material objection on any pick. There are medium findings (sector-wide losses for 9 names) and info flags "
          "(NIFTY below its 200-day average, results due inside the month).\n",
          "## What this means\n",
          "- **REV1 stays frozen and running.** Exchange opens confirm its numbers. But it is cost-fragile and, after Indian tax, it trails "
          "the plain equal-weight index. Even if the forward test holds up, REV1 is not a path to money.",
          "- **The earnings veto adds no measurable information to REV1.** No REV2.",
          "- **The two agents are built and running in shadow.** Their value will be judged on forward evidence, and every claim they make "
          "is traceable. The 10th Man is allowed to say \"no objection\", and it did.",
          "- **Most useful next step:** a strategy whose edge survives 2× costs and tax, which points to lower turnover. Candidates "
          "would have to be pre-registered; nothing new was tested here.\n",
          "## Limitations\n",
          "- About 80 months with flags; limited power.",
          "- Tax: dividends are inside the adjusted prices (taxed here as capital gains, in reality at slab rates); no surcharge; costs "
          "are not deducted from gains.",
          "- Google News RSS is used only for headlines, for personal research. X/Twitter is read only through the official API, for "
          "accounts a person has verified; none are configured yet.\n"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    reg = Registry("research/trials.jsonl")
    done = {r.get("strategy") for r in (json.loads(x) for x in open("research/trials.jsonl")) if r.get("version") == "v0.6"}
    n_prior = reg.count()
    recs, exps = [], []
    for k, v in T.items():
        name = f"rev1_earnings_veto:{k}"
        if name in done:
            continue
        recs.append({"version": "v0.6", "strategy": name, "config": {"window": 20, "definition": "preregistered v0.6"}})
        exps.append({"kind": "experiment", "version": "v0.6", "experiment_id": hashlib.sha1(f"{out}{name}".encode()).hexdigest()[:12],
                     "git_sha": git_sha(), "preregistration_sha256": R["prereg_sha256"], "features": name, "t": v["t"], "holm_p": v["holm_p"],
                     "number_of_prior_trials": n_prior, "decision": R["decision"]})
    if "tenth_man:base_rate_model" not in done:
        recs.append({"version": "v0.6", "strategy": "tenth_man:base_rate_model", "config": {"features": ["bad20", "extreme_loss", "vol_rank", "bear", "dq_lt1"]}})
        exps.append({"kind": "experiment", "version": "v0.6", "experiment_id": hashlib.sha1(f"{out}tm".encode()).hexdigest()[:12], "git_sha": git_sha(),
                     "features": "10th Man base-rate model", "brier_skill": wf["skill"], "number_of_prior_trials": n_prior, "decision": K["verdict"]})
    if recs:
        reg.add(recs)
        reg.add(exps)
    print(f"report written to {out / 'report.md'}; logged {len(recs)} trials")
