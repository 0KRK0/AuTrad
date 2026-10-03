# AuTrad v0.7 — pre-registration: low-turnover alpha programme, study 1 (cash-flow quality), and agent rules

Written 2026-10-03, **before any v0.7 result was computed**. The sha256 is logged before any computation.

## 0. Status of the reversal branch (closed)

The v0.4–v0.6 reversal branch is **closed**. REV1 is now a **frozen benchmark and forward experiment** with one job:
does the historical reversal observation keep appearing in unseen data? Nothing is ever added to REV1: no filters,
machine learning, regimes, sizing or re-ranking. Any such idea is REV2+, with its own pre-registration and its own
forward test. The REV1 ledger is never modified by research or by shadow agents.

## 1. Programme: low-turnover, point-in-time, incremental net alpha

Goal: information that is (a) point-in-time, (b) slow-moving (holding months, not weeks), (c) economically motivated,
and (d) **incremental** to what AuTrad already measures (REV1, MOM12, EY, ROE, LVOL). Families are tested **one at a
time**, each with its own pre-registration. Before any test, a data audit must show complete point-in-time coverage
over the whole test window.

| Order | Family | Data status (audited before any test) |
|---|---|---|
| 1 | **Cash-flow quality / accruals** | Ready. Half-yearly cash-flow statements are in the stored NSE XBRL filings (sample check: operating cash flow present in 50 of 60 half-year filings since Sep 2019). |
| 2 | Insider and promoter open-market transactions | **Blocked.** NSE's PIT endpoint coverage collapses from 2022 (system-driven disclosures are not returned), and the latest item is 2026-04-30 (`research/data_audits/insider_coverage_2026-10-03.json`). |
| 3 | Shareholding pattern and promoter pledges | **Blocked.** The NSE endpoint only goes back to Sep 2021; BSE history is needed. |
| 4 | Earnings revisions and guidance changes | **Blocked.** There is no analyst-consensus source. |
| 5 | Regime and allocation (equity / gold / cash) | Contaminated (seen in v0.3). **Forward test only.** |

### Capital-efficiency gate (applies to every candidate; strictly sequential; the first failure rejects)

1. **Statistical:** the family's pre-registered primary test is significant after Holm correction across its tests,
   with the hypothesised sign.
2. **Economic after costs:** net excess ≥ +2.0 %/yr over the pre-registered benchmark, at 1× costs.
3. **After-tax (estimate) positive:** post-tax CAGR above the post-tax benchmark CAGR (benchmark held under the same
   tax model, open lots liquidated at the end). This is a v0.6-model *estimate*; see section 4.
4. **2× costs:** net excess > 0 with all trading costs doubled.
5. **Capacity:** at ₹1 Cr, cost drag ≤ 1.25 × the drag at ₹10 lakh, and net excess still > 0.
6. **Stability:** net excess > 0 in at least two thirds of the pre-registered subperiods.
7. **Forward:** only a candidate that passes 1–6 gets a frozen forward test (as REV1).

Only after step 7 may the Main Head even *consider* allocating capital, and then only with the owner's approval.

## 2. Study 1 — cash-flow quality (accruals) in non-financial NIFTY 50 stocks

**Rationale.** Earnings that are not backed by operating cash (high accruals) tend to persist less, and the market
over-weights them (Sloan, 1996). The signal changes only twice a year, so turnover is naturally low. AuTrad has never
tested it.

**Data.**
- Operating cash flow (`CashFlowsFromUsedInOperatingActivities`, the same statement basis as the company's results
  in `facts.csv`), parsed from the stored NSE XBRL files.
- Half-year (Sep) and full-year (Mar) statements.
- Usable from the filing's dissemination time (15:30 IST rule).
- Total assets from the latest half-yearly balance sheet in `facts.csv`.
- Banks, NBFCs and insurers are excluded: cash flow is not comparable for them.

**Definitions (fixed).**
- **OCF_TTM:** at a March filing, the annual figure. At a September filing, H1 current + (annual previous − H1
  previous).
- **NP_TTM:** the sum of the last four quarterly net profits on the same basis.
- **ACC** = (NP_TTM − OCF_TTM) / total assets. The score is **CFQ = −ACC**: higher means better cash backing.
- Only computed when all inputs exist and are usable by the decision date.

**Tests (one Holm family of two, one-sided):**

| Id | Hypothesis | Test |
|---|---|---|
| S1 | CFQ predicts next-month returns | Monthly Spearman IC of the latest usable CFQ with the next-month return (next open → next month-end close), among non-financial PIT members. Mean IC > 0, Newey–West t with 6 lags |
| S2 | CFQ is **incremental** to known factors | Monthly Fama–MacBeth OLS of the next-month return on the ranks (0–1) of CFQ, REV1, MOM12, EY and ROE. Mean CFQ coefficient > 0, NW t with 6 lags |

**Portfolio (for gates 2–6).**
- Top third of eligible non-financial members by CFQ, equal weight.
- Rebalanced **semi-annually**, at the last session of June and of December (after the March and September filings are
  public), at the next open. Held until the next rebalance.
- Benchmark: the equal-weight non-financial PIT universe, rebalanced on the same dates.
- Exchange-anchored opens, full costs, ₹10 lakh.
- Subperiods: 2020-07..2022-06, 2022-07..2024-06, 2024-07..2026-09.

**Sample.** The first decision with a complete CFQ is expected around Dec 2020. Tests run to Sep 2026: about 70
monthly IC observations and 11 semi-annual rebalances. **Power is low.** A pass earns a forward test; a failure is
reported as a failure.

## 3. Agent rules (v0.7)

**Evidence tiers v2** (replace v0.6 section 4's list):

| Tier | Source |
|---|---|
| 1 | Regulators, exchange filings and announcements, audited filings |
| 2 | Official company disclosures (company IR site, annual report) |
| 3 | Direct executive statements |
| 4 | Primary documents and transcripts |
| 5 | Established financial journalism |
| 6 | Specialist research |
| 7 | Other media |
| 8 | Social media and unverified sources |
| 9 | AuTrad-derived (never proof) |

- A claim is **never** upgraded by repetition. Each claim records its source, tier, corroboration, contradiction,
  timestamp and outcome.
- Contradiction-based challenges still need tier ≤ 2 evidence.

**10th Man report.** Three separate parts:
1. **Historical probability:** owned by the statistical layer. It is the base rate unless a model beats it out of sample.
2. **Evidence challenge:** MATERIAL or NONE, from evidence only.
3. **Failure scenarios:** "if X then Y" statements citing evidence, **with no probabilities**.

Qualitative research never becomes a number.

**Shadow scorecard (does the 10th Man improve decisions?).** For every shadow memo, record each pick's evidence status
and its realised holding-month return minus the equal-weight universe. The test runs after **≥ 24 completed months
and ≥ 30 MATERIAL picks**:

| Element | Rule |
|---|---|
| Hypothesis | MATERIAL picks underperform NONE picks |
| Test | One-sided t of the month-clustered difference |
| Pass | p < 0.05 **and** the mean difference ≤ −0.5 % per month |

Until a pass **and** a separate pre-registration, the Head's authority over allocation stays **zero**: shadow only.
No advisory agent may place, modify, cancel or size orders.

## 4. Tax output wording

All tax figures are an **after-tax estimate**. Not yet modelled:
- dividends taxed at slab rates (they are currently inside total-return prices);
- surcharge;
- deduction of transaction costs other than STT from capital gains;
- 2018 grandfathering.

They are never called "actual investor after-tax return".

## What v0.7 will not do

No change to REV1. No new technical indicators. No machine learning. No more than one family tested. No agent with
order authority.
