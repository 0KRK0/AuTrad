"""reports/v0.7_<ts>/report.md and the ledger records for v0.7."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from lab.registry import Registry, git_sha


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def write_report(out: Path) -> None:
    C = json.loads((out / "cfq.json").read_text())
    REG = json.loads((out / "candidate_register.json").read_text())
    M = json.loads((out / "company_memory.json").read_text())
    S = json.loads((out / "shadow_scorecard.json").read_text()) if (out / "shadow_scorecard.json").exists() else {}
    P = pd.read_csv(out / "promises_pilot.csv") if (out / "promises_pilot.csv").exists() else pd.DataFrame()
    pf, s1, s2 = C["portfolio"], C["S1"], C["S2"]
    L = ["# AuTrad AI v0.7 — low-turnover alpha programme, company memory, and a stricter 10th Man\n",
         f"Pre-registration `research/preregistration_v0.7.md` (logged before any result; Amendment 1 = data audit, Amendment 2 = fixes after an independent review; no definition changes) · "
         f"sha256 `{C['prereg_sha256'][:16]}` · git `{git_sha()}`\n",
         "## 1. The reversal branch is closed\n",
         "REV1 is now a **frozen benchmark and forward experiment**: does the historical reversal pattern keep appearing in unseen data? "
         "Nothing will be added to it. Any idea built on it is REV2+, with its own pre-registration and forward test. Its paper ledger "
         "keeps running, and neither research nor the shadow agents can write to it.\n",
         "## 2. The capital-efficiency gate (code: `lab/gates.py`)\n",
         "Strictly in order, and the first failure rejects:\n",
         "1. statistically significant (pre-registered, after Holm correction);",
         "2. at least +2 %/yr net of costs;",
         "3. positive after tax (estimate);",
         "4. still positive at 2× costs;",
         "5. capacity at ₹1 Cr;",
         "6. positive in ⅔ of subperiods;",
         "7. a frozen forward test.\n",
         "Only after step 7 may the Main Head even *consider* capital, and then only with your approval. Every candidate so far, "
         "run through it:\n",
         "| Candidate | Verdict |", "|---|---|"]
    for r in REG:
        L.append(f"| {r['candidate']} | {r['verdict']} |")
    r1 = next(r for r in REG if r["candidate"].startswith("REV1 one-month"))
    L += ["", "Each candidate is measured over its own pre-registered window, on one consistent source (exchange opens, full costs, "
          "the same tax estimate). **No candidate passes.**\n",
          f"REV1 over 2016–26 is the closest. It clears statistics, +{r1['inputs']['net_excess'] * 100:.1f} %/yr and, barely, the "
          f"after-tax estimate ({pct(r1['inputs']['post_tax_cagr'])} vs {pct(r1['inputs']['benchmark_post_tax_cagr'])} for the equal-weight "
          f"index). It then fails at double costs ({pct(r1['inputs']['net_excess_2x'])}). Over the v0.6 window (2019–26) it trailed "
          "the index after tax. REV1 keeps running only as the frozen benchmark it was already, before this gate existed.\n",
          "## 3. Study 1: cash-flow quality (accruals), semi-annual, non-financial NIFTY 50\n",
          "A data audit came first:\n",
          "- Operating cash flow is in NSE's results XBRL from Mar 2021. It was checked against annual reports: TCS FY21 ₹38,802 cr, "
          "Infosys FY21 ₹23,224 cr, Infosys FY23 ₹22,467 cr, TCS FY25 ₹48,908 cr.",
          "- **Total assets only start in Sep 2022**, so the pre-registered definition (accruals / total assets) was kept and the "
          f"window shortened: {C['months_ic']} months of tests and {pf['rebalances']} semi-annual rebalances. Power is very low, and "
          "this was stated before any result.",
          "- Insider trading and promoter-pledge data were audited too. Both are **blocked**: NSE's insider-trade feed thins out after "
          "2021 (companies covered: 67 in 2021, 51 in 2022, 41 in 2023), and shareholding history starts only in Sep 2021.",
          "- The independent review found a gap the first audit missed: **the June-2022 quarter is missing for 15 companies.** They "
          "have no score until Sep-2023 data, so in the first two rebalances only 25 and 28 of 39 eligible stocks were scored. The "
          "pre-registered benchmark stays; a descriptive comparison against only the scored stocks is shown as well.\n",
          "| Test | Estimate | NW t | Holm p | Result |", "|---|---|---|---|---|",
          f"| S1 CFQ predicts next-month returns (IC) | {s1['mean_ic']:+.3f} | {s1['nw_t']:.2f} | {s1['holm_p']:.2f} | no |",
          f"| S2 CFQ adds to REV1, MOM12, EY, ROE (Fama–MacBeth) | {s2['mean_coef_bps']:+.0f} bps / month | {s2['nw_t']:.2f} | {s2['holm_p']:.2f} | no (wrong sign) |",
          "", "| Portfolio (top third, ~11 names, rebalanced each June and December) | Value |", "|---|---|",
          f"| Net excess vs equal-weight non-financials | **{pct(pf['net_excess'])}/yr** (monthly t {pf['monthly_excess_t']:.2f}) |",
          f"| At 2× costs · at ₹1 Cr | {pct(pf['net_excess_2x'])} · {pct(pf['net_excess_1Cr'])} |",
          f"| Turnover · cost drag | {pf['turnover_per_year']:.1f}×/yr · {pct(pf['cost_drag_10L'], 2)}/yr |",
          f"| After-tax estimate (portfolio vs benchmark) | {pct(pf['post_tax_cagr'])} vs {pct(pf['benchmark_post_tax_cagr'])} |",
          "| Subperiods | " + ", ".join(f"{k} {pct(v)}" for k, v in pf["subperiod_excess"].items()) + " |",
          f"| Descriptive: vs the scored stocks only | {pct(pf['sensitivity_vs_scored_names_only']['net_excess'])}/yr; " +
          ", ".join(f"{k} {pct(v)}" for k, v in pf["sensitivity_vs_scored_names_only"]["subperiods"].items()) + " |",
          "", f"**Verdict: {C['gate']['verdict']}.**\n",
          "This is the trap the gate exists for. The portfolio looks excellent: +5.7 %/yr, 1.4× turnover, survives double costs, "
          "ahead after tax. But the signal itself shows nothing (IC t 0.41), and once the known factors are included its "
          "coefficient is negative. All the excess came in Dec 2022 – Jun 2024 (+20 %); since then it is −0.8 %. With 11 stocks "
          "and 8 rebalances, a few lucky picks can produce this. Accruals stay unproven. They can be re-tested in about two "
          "years, when more balance-sheet history exists, under a new pre-registration.\n",
          "## 4. The 10th Man now separates probability, evidence and scenarios\n",
          "Each report now has three parts that never mix:\n",
          "- **Historical probability:** owned by the statistical layer. It is the base rate (~49 %) because the flag model has no "
          "skill (Brier skill −0.005).",
          "- **Evidence challenge:** MATERIAL or NONE. Only tier-1/2 contradicting evidence can make it MATERIAL. After the review, "
          "\"adverse company news\" must be shown by the released results themselves (a year-on-year profit decline in the "
          "exchange filing). A big market reaction alone is a statistical flag, and the flag has no predictive skill.",
          "- **Failure scenarios:** \"if … then …\" statements, citing evidence, **with no probabilities**.\n",
          "Qualitative research never becomes a number. A test shows that even a confident model probability cannot turn into an "
          "objection. The September memo was re-run in this format (the old memos stay in the log, marked superseded), and the "
          "documents the 10th Man cites are now stored. Result: no material objection on any of the 16 picks. Every pick still "
          "lists its findings and if/then scenarios, for example TCS: results on 8 Oct fall inside the holding month.\n",
          f"**Shadow scorecard** (`head/scorecard.py`): {S.get('status', 'n/a')}. The pre-registered test runs after 24 months and "
          "30 MATERIAL picks. Until it passes, and a separate pre-registration follows, the Head's authority over allocation stays at zero.\n",
          "## 5. Company memory: the evidence and claim graph\n",
          f"`data/intel/graph.sqlite` (`intel/graph.py`) holds **{M['graph']['documents']:,} documents**, {M['graph']['claims']} claims, "
          f"{M['graph']['metrics']:,} filed numbers, {M['graph']['events']:,} results reactions and {M['graph']['decisions']} Head decisions. "
          "Documents by tier: " + ", ".join(f"tier {k}: {v:,}" for k, v in M["graph"]["documents_by_tier"].items()) + ".\n",
          "Confidence comes from the best tier of a claim's **own** sources, and **never rises by repetition**:\n",
          "- ten tier-8 posts are still LOW;",
          "- a tier-5 report cannot contest a tier-1 filing;",
          "- AuTrad-derived numbers alone are only DERIVED-ONLY;",
          "- a claim with no source stays UNVERIFIED.\n"]
    t = M.get("trace_example") or {}
    if t:
        L += [f"**\"Why did the Head buy {t['symbol']}?\"** — `trace()` walks the decision ({t['asof']}) back to its documents:\n"]
        for b in t["because"]:
            srcs = ", ".join(sorted({"tier %s %s" % (d["tier"], d["source"]) for d in b["documents"]})) or "no source"
            L.append(f"- thesis: \"{b['claim']}\" → **{b['confidence']}** ({srcs})")
        for b in t["challenged_by"]:
            L.append(f"- 10th Man finding: \"{b['claim'][:110]}\" → **{b['confidence']}**")
        L += ["", "The trace makes the weak point visible:",
              "- The Head's \"no adverse company news\" premise rests on no evidence at all (UNVERIFIED).",
              "- \"Stock-specific overreaction\" rests only on AuTrad's own numbers (DERIVED-ONLY).",
              "- Only \"fundamentals are stable\" rests on exchange filings (HIGH). The filings are cited only because they show no "
              "decline.\n"]
    full = {k: v["full_text"] for k, v in M["transcripts"].items()}
    filed = {k: v["filed"] for k, v in M["transcripts"].items()}
    L += ["### Management promises (pilot: 6 companies, 2022–2025)\n",
          f"NSE-filed earnings-call transcripts: {sum(filed.values())} filed, of which {sum(full.values())} contain the transcript text. The rest are "
          "cover letters linking to the company website: Maruti and L&T entirely, Tata Steel mostly; not fetched yet. A conservative "
          "rule-based extractor keeps only **management** statements (speakers who answer rather than ask) that include a future "
          "commitment, a measurable target and a date. Every promise is stored with its verbatim quote and filing; nothing is "
          "paraphrased into a fact.\n",
          f"Result: **{len(P)} promises**: " + ", ".join(f"{k} {v}" for k, v in P["symbol"].value_counts().items()) + ". UltraTech states "
          "dated capacity targets openly. TCS and HUL management almost never give dated numeric targets, so their transcripts yield "
          "nothing, which is itself informative. Status: " + ", ".join(f"{v} {k}" for k, v in P["status"].value_counts().items()) + ".\n",
          "| Said | Who | Promise (verbatim, shortened) | Due | Status |", "|---|---|---|---|---|"]
    for r in P.sort_values("said").head(8).itertuples(index=False):
        L.append(f"| {r.said} {r.symbol} | {r.speaker} | \"{r.quote[:110]}…\" | {r.due if isinstance(r.due, str) else '—'} | {r.status} |")
    L += ["", "No promise was auto-scored MET or MISSED:",
          "- The pilot found no revenue-growth promises, the only kind the filings can check automatically.",
          "- Capacity and completion promises need outcome evidence from later documents. The graph links the *matching passage* "
          "of a later transcript for a person to confirm.",
          "- Several promises fall due after the last fetched transcript, so they are marked \"not yet covered\", not \"failed\".",
          "", "Turning \"candidate\" into \"resolved\" without a human, or a verbatim-quoting model, would be inventing outcomes.\n",
          "## 6. Tax wording\n",
          "All tax figures are an **after-tax estimate**, never \"actual investor after-tax return\". Not yet modelled:\n",
          "- dividends taxed at slab rates (they are inside total-return prices);",
          "- surcharge;",
          "- non-STT costs deducted from gains;",
          "- 2018 grandfathering.\n",
          "## What this means\n",
          "- **No strategy has passed the gate.** The first low-turnover study looked profitable but failed the statistics. "
          "That is the result, not a near-miss.",
          "- **The next family with complete data is still to be chosen.** Insider, pledge and analyst-revision data are blocked "
          "until better sources exist (BSE history, a consensus provider). Pre-register it only when its data audit passes.",
          "- **Company memory is now a real primitive.** The highest-value additions are:",
          "  - fetching the company-website transcripts (tier 2) that the exchange filings only link to;",
          "  - a verbatim-quoting extractor with wider recall;",
          "  - human confirmation of promise outcomes.",
          "- Live trading stays disabled. The evening paper-test task is not scheduled; that decision is yours.\n"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    reg = Registry("research/trials.jsonl")
    done = {r.get("strategy") for r in (json.loads(x) for x in open("research/trials.jsonl")) if r.get("version") == "v0.7"}
    n_prior = reg.count()
    recs, exps = [], []
    for k, v in (("S1", s1), ("S2", s2)):
        name = f"cfq:{k}"
        if name in done:
            continue
        recs.append({"version": "v0.7", "strategy": name, "config": {"definition": "preregistered v0.7 study 1", "rebalance": "semi-annual"}})
        exps.append({"kind": "experiment", "version": "v0.7", "experiment_id": hashlib.sha1(f"{out}{name}".encode()).hexdigest()[:12], "git_sha": git_sha(),
                     "preregistration_sha256": C["prereg_sha256"], "features": name, "t": v["nw_t"], "holm_p": v["holm_p"], "number_of_prior_trials": n_prior,
                     "net_return": pf["net_excess"], "decision": C["gate"]["verdict"]})
    if recs:
        reg.add(recs)
        reg.add(exps)
    print(f"report written to {out / 'report.md'}; logged {len(recs)} trials")
