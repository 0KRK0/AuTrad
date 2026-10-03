# AuTrad v0.6 — pre-registration: execution prices, taxes, REV1 × earnings-reaction veto, and the advisory agents

Written 2026-10-03, **before any v0.6 result was computed**. The REV1 forward paper test (v0.5, Track A) is not touched:
anything learnt here can only ever become a *new* strategy (REV2) with its own forward test. Every test is logged in
`research/trials.jsonl`; the sha256 of this file is logged before any computation.

## 1. Exchange-anchored execution prices

Problem (v0.5 report, limitations): fills use the vendor's adjusted open. REV1 trades at the next open, so the open matters.

- Source: NSE's official daily bhavcopy (equity segment, EQ/BE series), downloaded on the user's machine
  (`marketdata/nse_bhav.py`; legacy `cmDDMMMYYYYbhav.csv.zip` and the 2024+ UDiFF format).
- Exchange-anchored adjusted open: `O_x = NSE_open × (C_adj / NSE_close)` on the same day, where `C_adj` is the
  exchange-anchored adjusted close already in use. So the open carries the same day's adjustment factor as the close.
  If NSE has no row for a stock-day, the vendor open is kept and the day is counted as a fallback.
- Measurements: the distribution of |vendor open − O_x| / O_x, the share of stock-days above 10 bps and 50 bps,
  fallback counts, and the frozen REV1's research result with each open source (sensitivity; **the forward test
  keeps its frozen pipeline**).
- Rule: from v0.6 on, research results use `O_x` (`build_panel(..., open_source="exchange")`).

## 2. India tax layer (reported beside, not instead of, pre-tax results)

- Lots: FIFO per symbol from the engine's fills. Holding > 12 months = long-term (listed equity, STT paid).
- Rates by sale date. **Before 2018-04-01:** STCG 15 %, LTCG exempt. **2018-04-01 to 2024-07-22:** STCG 15 %, LTCG 10 %
  above ₹1 lakh. **From 2024-07-23:** STCG 20 % (s.111A), LTCG 12.5 % above ₹1.25 lakh (s.112A). The FY 2024-25
  exemption is ₹1.25 lakh.
- Cess: 3 % before FY 2018-19, 4 % after. No surcharge (assumes total income below ₹50 lakh). 2018 grandfathering is
  ignored (it does not matter for lots bought after 2017).
- Set-off: short-term losses against short- or long-term gains; long-term losses only against long-term gains. Losses
  carry forward 8 years. The exemption applies after set-off.
- Payment: the financial year's tax is deducted from NAV at its last session (31 March). This is conservative against
  advance-tax timing.
- Reported: the post-tax CAGR of each portfolio and of NIFTYBEES held (taxed on terminal liquidation), and the
  post-tax excess. A portfolio whose post-tax excess over the post-tax NIFTYBEES is ≤ 0 is flagged **TAX-NEGATIVE**.

## 3. Main hypothesis — does a bad earnings reaction add information to REV1?

**Rationale.** REV1 buys last month's losers on the premise that the losses were liquidity-driven and revert. A loss
that came from bad earnings news is information, and v0.5 showed it keeps drifting (−107 bps over 20 sessions).
So reversal should work worse for news-driven losers ("news vs no-news reversal").

**Definitions (fixed):**
- **BAD20(t):** the stock had a results announcement whose market-adjusted reaction (v0.5 measure MKT) was "bad"
  (below the expanding lower-third quantile of all earlier events; past-only, 200-event burn-in). Its reaction
  session *i* must satisfy t − 19 ≤ i ≤ t. Everything is known at the close of t.
- **REV1:** exactly the frozen definition (top third, equal weight, month-end close, next-open fill).
- **REPLACE (primary variant):** REV1's ranking with BAD20 names removed; take the top n = floor(N/3) of what
  remains (same number of names as REV1).
- **SKIP (secondary variant):** REV1's top third with BAD20 names dropped; the remaining names equal weight.
- **Window:** months where at least one BAD20 flag can exist (the first flagged event is in Oct 2019) to Aug 2026.
  Before that the variants equal REV1 by construction.
- Execution with exchange-anchored opens (section 1). Full costs. ₹10 lakh.

**Tests (one Holm family of three, one-sided):**

| Id | Hypothesis | Test |
|---|---|---|
| H1 | BAD20 predicts lower next-month returns **beyond** REV1 | Monthly Fama–MacBeth cross-sectional OLS of the next-month return on the REV1 rank (0–1) and the BAD20 dummy, over all eligible members. The mean BAD20 coefficient is < 0, with a Newey–West t (3 lags) across months |
| H2 | REPLACE beats REV1 | Monthly net-return difference REPLACE − REV1 > 0 (NW t) |
| H3 | SKIP beats REV1 | Monthly net-return difference SKIP − REV1 > 0 (NW t) |

**Reported, not decision tests:**
- Net excess over the equal-weight universe (1× and 2× costs).
- Sharpe, maximum drawdown, turnover and replaced names per month.
- Capacity at ₹1 Cr, subperiods, post-tax results, and vendor-open vs exchange-open results.
- The BAD60 window (t − 59 ≤ i ≤ t), as a sensitivity check.

**REV2 candidate rule (all must hold, for REPLACE only):**
1. H1 or H2 significant after Holm, with the right sign.
2. REPLACE's net excess over the equal-weight universe is higher than REV1's at 1× costs **and** > 0 at 2× costs.
   REV1 itself failed 2× costs.
3. REPLACE − REV1 > 0 in at least 3 of the 4 subperiods. Subperiod 1 (2016-06..2018-12) has no flags; it counts as 0
   and therefore as not positive.
4. Turnover ≤ 1.1 × REV1's.

A pass earns REV2 a separate forward pre-registration. It never changes REV1.

**Honesty notes:**
- The v0.5 event results came from these same data, so the hypothesis is motivated by data already seen.
- Only about 80 months have any flag. Power is limited.

## 4. Advisory agents (Company Intelligence, 10th Man) — rules fixed before they run

**Authority.** Neither agent can place or change an order. Each reports only to the Main Head. The Head's decision
still goes through the deterministic Risk Gate. In v0.6 the Head runs them in **shadow**: their reports are logged and
nothing they say changes any portfolio.

**Isolation of the 10th Man.** It receives only a sanitised `ChallengeRequest` from the Head: proposed action, asset,
size, horizon, the thesis claims the Head chooses to expose, and their evidence ids. It never receives other agents'
reasoning, and it has no reference to any other agent. This is enforced by the code structure and tested (import graph
plus the request type).

**No manufactured evidence.** Every factual statement in either agent's output must resolve to an Evidence Ledger
record: source, URL or document id, publication time, ingestion time, entity, extraction method and content hash.
Anything else is labelled UNVERIFIED, which is never treated as a fact. Source tiers, in order of reliability:
1. Regulatory filing or exchange announcement.
2. Audited financial statement.
3. Company statement.
4. Executive public statement.
5. Reputable journalism.
6. Analyst research.
7. Social media.
8. Anonymous source.

**10th Man probability.** A failure probability may be shown only from a statistical model, never from a language
model. For a REV1-selected stock, "failure" means it underperforms the equal-weight universe over the holding month.
- Model: an expanding-window logistic regression, refitted each month on outcomes known strictly before the decision.
- Fixed features: BAD20; extreme last-month loss (bottom decile of REV1 score); 252-session volatility rank; NIFTY
  below its 200-session average; data quality < 1 on the decision day.
- The model's out-of-sample Brier score is compared with the constant base rate. **If it does not beat the base rate
  (Brier skill ≤ 0), the 10th Man reports the base rate only, labelled "flags not predictive".**

The 10th Man may return NO_MATERIAL_OBJECTION. It returns MATERIAL_CHALLENGE only when the calibrated probability
exceeds the base rate by at least 5 percentage points, **or** a tier-1/2 evidence item contradicts a thesis claim.

**Company Intelligence** builds per-company dossiers:
- Business and sector; PIT financial trajectory from the NSE filings already stored.
- Results timeline and reactions.
- Corporate announcements and leadership from NSE.
- Official communication channels (registry of verified handles; posts are ingested only through a source's official
  API or feed — no scraping behind logins).
- A **management-claim tracker**: claim, speaker, date, expected outcome and date, and the resolved outcome, which
  builds a reliability record over time.

It is evaluated in v0.6 only on provenance coverage (100 % of facts resolved) and freshness, never on returns.

## What v0.6 will not do

No new price factors, no machine learning, no change to REV1, no live trading, no agent with order authority.

---

## Amendment 1 — 2026-10-03, after an independent review of the first run (no hypothesis, threshold or decision rule changes)

The first run's hypothesis results (H1–H3, REV2 decision) are unaffected. These items are fixed, and the run is repeated as `v0.6-r1`:

1. **Evidence tiers.** In the first run, AuTrad's own computed sentences were stored as tier-2 documents. That let a
   10th-Man computation (a sector-wide move) count as a "tier-1/2 contradiction", and 9 of 16 names were challenged
   because of it. Now:
   - AuTrad writes only tier-9 *dataset extracts* (raw numbers).
   - Statements built on them are DERIVED.
   - A fact must quote its source verbatim.
   - Only tier-1/2 *external* evidence (exchange filings, audited statements) can make a contradiction.
2. **Time comparisons** are made on UTC instants. Before, news stamped in UTC could pass an IST cut-off: 29 documents
   from after the close were labelled SOURCED.
3. **The Head's REV1 thesis.** After the first memo, the thesis was changed to drop the "stock-specific overreaction"
   claim, which removed every challenge. That was an edit to the 10th Man's input made after seeing its output, and it
   is reverted:
   - The original thesis is restored.
   - The challenges are now governed by rule 1, not by wording.
   - The thesis template is hashed in every memo, and changes to it are logged.
4. **Tax corrections.**
   - Exempt-era (pre-FY 2018-19) long-term results neither absorb short-term losses nor carry losses forward.
   - "Long-term" means held for more than 12 calendar months.
   - Post-tax results build lots from the whole history, so positions held at the start of the window keep their cost basis.
5. **Execution prices.**
   - ETFs keep vendor opens. NSE's official ETF opens differ by more than 50 bps on about 7 % of NIFTYBEES days, often
     stray auction prints. The rule in section 1 applies to stocks.
   - A continuity guard (the day's factor must stay within 25 % of its trailing 21-session median) protects against a
     reused symbol.
6. **Smaller items.**
   - The 10th Man's news window is t − 19..t, the same as BAD20.
   - The claims record is filtered as of the decision.
   - The 10th Man reads its own copy of the external documents only.
