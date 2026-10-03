# AuTrad v0.5 — pre-registration: alpha validation (reversal forward test, low-volatility decomposition, earnings reaction)

Written 2026-10-03, **before any v0.5 result was computed**. Everything below is frozen: definitions, data rules, tests,
thresholds and decision rules. Any later change is added as a dated amendment with a reason, and the original text stays.
Every test is logged in `research/trials.jsonl`, failures included. The sha256 of this file is logged before any computation.

v0.5 adds **no new factors**. It goes deeper on three findings from v0.4-r1 (`reports/v0.4-r1_20261003T102305Z`):

| Finding (v0.4-r1, historical, already seen) | v0.5 track |
|---|---|
| REV1 (one-month reversal) passed the pre-registered IC gate (IC 0.056, NW t 3.32, Holm p 0.020). Its long-only tercile earned +2.25 %/yr net of costs over the equal-weight universe, but with a monthly excess t of only 0.87, and 2024 was −9.3 %. | A — forward paper test, frozen |
| The v0.3 `low_vol` sleeve showed +3.2 %/yr unexplained alpha (t 2.2) after market, MOM12 and LVOL, yet the LVOL factor itself had no IC (0.005, t 0.26). | B — decomposition |
| Earnings announcements with a bad market-adjusted reaction kept drifting down: −103 bps over 20 sessions (t −3.3). | C — hypothesis tests |

## Core principle: three separate questions

Every result in v0.5 is scored on four separate axes, and one axis is never allowed to stand in for another:

| Axis | ✓ | ~ | ✗ |
|---|---|---|---|
| **Statistical** — is the effect distinguishable from noise? | the pre-registered test passes (Holm-adjusted p < 0.05 within its family, or t ≥ 2.0 for a single pre-declared object) | unadjusted p < 0.10 | otherwise |
| **Economic** — is it big enough to matter after costs? | portfolios: net excess ≥ +2.0 %/yr; events: net ≥ +50 bps per position | > 0 but below that | ≤ 0 |
| **Tradable** — can AuTrad actually capture it? | long-only implementable, still > 0 at 2× costs, positive in ≥ 3 of 4 subperiods, capacity ≥ ₹1 Cr with < 25 % extra cost | some of these | net ≤ 0 at 1× costs, or needs shorting that is not available in cash equities |
| **Forward** — has it held up on data nobody had seen? | see Track A decision rules | running | contradicted |

Subperiods everywhere: 2016-06..2018-12, 2019..2021, 2022..2023, 2024..2026-09.

## Data (unchanged from v0.4 except where stated)

- Prices: `data/snap_20261001`, exchange-anchored (`ca_2026_10_03_exchange_anchored`), immutable manifest
  `research/manifests/data_integrity_v0.3.1.json` (sha256 a8f59244e533bb71…). Fundamentals: `data/pit/nse/facts.csv` (sha256 854fa9925d268260…).
- Universe: point-in-time NIFTY 50; cost model `india_zerodha_2026_09`; cash 6 %/yr; capital ₹10 lakh unless stated.
- **Data maintenance amendment (does not touch any historical date):** NSE Indices press release of 10 Aug 2026 —
  BSE Ltd (BSE) replaces Wipro (WIPRO) in the NIFTY 50 with effect from 30 Sep 2026 (after the close of 29 Sep 2026).
  A 2026-09-30 snapshot is added to `nifty50_pit_snapshots.csv`. BSE Ltd is listed only on NSE, so its prices cannot
  be anchored to BSE's bhavcopy; it is flagged as vendor-only.

---

## Track A — REV1 forward paper test (frozen; separate from research)

### Frozen strategy (identical to the v0.4 research definition; code hashed)

- Decision: at the close of the last NSE session of each calendar month.
- Universe: the stocks that are point-in-time NIFTY 50 members on the decision date and eligible in the panel
  (exchange-quality ≥ 0.5, class large-cap/default, a price on the decision date).
- Score: REV1 = −(C_t / C_{t−21} − 1), adjusted closes. Required: closes at t, t−21 and t−252, and at least 200 valid
  daily returns in the last 252 sessions (exactly `lab.factors.price_scores`).
- Portfolio: the top third by score (n = floor(N/3)), equal weight 1/n, fully invested, no sector cap, no other rules
  (exactly `lab.factors.tercile_targets`).
- Execution: at the next session's open, through `lab.engine.run`: full Indian delivery costs, half-spread, sqrt
  impact, 2 % participation cap, and entries blocked on non-eligible days.
- Capital: ₹10,00,000 notional at the first fill (open of 1 Oct 2026). Cash earns 6 %/yr.
- Frozen code: the sha256 of the source of `lab.factors.price_scores`, `lab.factors.stock_universe`,
  `lab.factors.tercile_targets`, `lab.engine.run` and the whole `execution/costs.py` is stored in
  `forward/rev1/spec.json`. The runner refuses to run if any hash differs.

**Nothing may be changed:** no parameters, no filters, no stops, no sizing changes, no "skip this month". Data errors
are fixed only through the same audited correction pipeline, and every correction is logged in the forward ledger.

### Forward data and records

- Prices come from a fresh vendor download into `data/forward/` (never the research snapshot), anchored to BSE
  bhavcopies the same way. The panel starts 2024-06-01, which is enough for the 252-session requirement.
- `forward/rev1/ledger.jsonl` is append-only and hash-chained: every record carries the sha256 of the previous record.
  - **decision** (once per month, never rewritten): universe, every score, the selected names, target weights, data
    hashes, and the frozen-code hashes.
  - **fills**: per symbol, the side, weight change, traded value, open price, shares and modelled cost.
  - **mark** (per session): NAV, weights, cumulative costs and both benchmark NAVs.
  - **revision**: written if a later data download changes an already-marked NAV by more than 1 bp.
  - **gate_shadow**: what the production Risk Gate would have done with each order (approve, resize, reasons).
    This is information only; the frozen strategy is not gated, exactly as in research.
- The decision for a month-end is recorded when the first session of the next month appears in the data, using data up
  to the month-end close only. **The first decision (30 Sep 2026) is computed on 3 Oct 2026** from data up to the
  30 Sep close, with the fill at the recorded 1 Oct open. The rule is mechanical, so nothing in it can depend on what
  happened after 30 Sep.
- If a month-end's data are not available within 5 sessions, the previous holdings are kept and the month is flagged
  `missed_decision`; it still counts.

### Benchmarks

- Primary: the equal-weight point-in-time NIFTY 50 universe, through the same engine and costs, rebalanced on the same
  dates. This is the benchmark REV1 was selected against.
- Secondary: NIFTYBEES (NIFTY 50 ETF), bought at the 1 Oct 2026 open with costs, then held.

### What the forward test can and cannot show (computed from v0.4-r1, written down before any forward data)

The historical information ratio is about 0.26 (+2.25 %/yr excess over roughly 8.7 %/yr tracking error). At that
ratio, reaching t = 2 on forward data alone would take (2 / 0.26)² ≈ **59 years**. A forward test therefore
**cannot prove** REV1. It can show that REV1 is broken (much worse than history), and it can check the
implementation: data pipeline, decisions, fills and costs. REV1 cannot earn capital on forward evidence of
this size.

### Decision rules (frozen)

- Monthly: a status line only. Nothing is decided before 12 complete months (first review: after the October 2027
  month-end).
- **Integrity failure → VOID**: a frozen-code hash changes, the ledger chain breaks, or a decision is written after
  its fill session's close. The forward test restarts under a new pre-registration.
- **REJECT** (at any review from 12 months on): the cumulative forward excess over the primary benchmark is more
  than 2 standard errors below the historical expectation, i.e.
  z = (excess_fwd − 2.25 %·years) / (8.7 % · √years) < −2. Also REJECT if realized turnover is more than 1.5× the
  historical 16.8×/yr, which would mean an implementation fault.
- **KEEP OBSERVING**: otherwise. REV1 is never promoted to real capital on its own, because the information ratio
  is too low. Capital needs a signal with a materially higher information ratio. A composite with another
  independently established factor is a separate future study, not v0.5.

---

## Track B — low-volatility decomposition (research only; nothing goes live)

### Objects (fixed; no tuning)

1. **Primary:** the v0.3 `low_vol` sleeve with its selected setting `{top_n: 15, market_filter: False}`: the 15
   lowest 252-session-volatility large-cap PIT members, equal weight, decided at the first session of each month,
   with name caps and a 30 % sector cap (any excess held in cash). This is where the +3.2 % "alpha" came from.
2. LVOL top tercile (the v0.4 factor portfolio).
3. Diagnostic **sector-neutral low-vol**: in every sector, hold the lowest-volatility third of its eligible members
   (at least one name). Each sector is weighted at its share of the equal-weight universe, with equal weights inside
   the sector. Decided on the same dates as object 1.

### Factor returns used to explain them (monthly, 2016-06 .. 2026-09)

- MKT = NIFTYBEES − cash.
- EWCW = equal-weight PIT universe − NIFTYBEES. This is the "smaller NIFTY 50 names" tilt that any equal-weight
  NIFTY 50 portfolio carries.
- SMB = small-minus-big tercile within the PIT universe, ranked by the 63-session median traded value, run as
  futures long-short through the engine. Its correlation with free-float market cap on 2019+ data is reported.
- MOM = the MOM12 long-short (futures, as in v0.4).
- LVOL = the LVOL long-short (futures, as in v0.4); only in model M2.

Models (OLS on monthly returns over cash, Newey–West t-stats with 3 lags):
- M0 = MKT + MOM + LVOL (replicates v0.4).
- **M1 = MKT + EWCW + SMB + MOM (primary).**
- M2 = M1 + LVOL.

### Measurements

1. Return waterfall for each object: total over cash = Σ β_k · mean(f_k) + alpha (model M1), annualized.
2. Sector: a Brinson split of the excess over the equal-weight universe into allocation and selection; average
   and maximum sector weight; sector HHI against the universe; and the average cash forced by the sector cap.
3. Beta: the ex-ante beta of the holdings (past-only, 252-session), the realized beta, and the net excess over a
   beta-matched benchmark (β_ex-ante × NIFTYBEES + (1 − β) × cash).
4. Turnover, cost drag, and net results at 2× and 3× costs; capacity at ₹1 Cr and ₹5 Cr.
5. Regimes, set at each decision date: NIFTY above or below its 200-session average, and trailing 63-session NIFTY
   volatility above or below its expanding median. Excess and up/down capture in each.
6. Crashes: the 10 worst NIFTYBEES months (data-defined), and named episodes fixed in advance — Sep–Oct 2018
   (IL&FS), Feb–Mar 2020 (COVID), Apr–Dec 2020 (rebound), Jan–Jun 2022 (rate shock), Oct 2024–Feb 2025 (FPI outflow).
7. Subperiods: the M1 alpha and the net excess in each of the four.

### "Serious candidate" rule for the primary object (all must hold)

- B1: M1 alpha > 0 with NW t ≥ 2.0.
- B2: Brinson selection > 0, and the sector-neutral variant's M1 alpha > 0.
- B3: net excess over the beta-matched benchmark > 0.
- B4: M1 alpha > 0 in at least 3 of the 4 subperiods.
- B5: net excess over the beta-matched benchmark still > 0 at 2× costs.

The +3.2 % was found on these same data, so even a pass is **not independent evidence**. A pass earns a separate
forward pre-registration; nothing more.

---

## Track C — earnings reaction (research only; no shorting proposal)

### Events

The same set as v0.4-r1: the first dissemination per (company, quarter) by a point-in-time member, excluding stale
timestamps. Reaction session *i* is the first session that could trade the filing (15:30 IST rule).

### Reaction measures (all known at the close of *i*)

- ABS = log C_i/C_{i−1}.
- MKT = ABS − log NIFTY_i/NIFTY_{i−1} (the v0.4 EAR).
- SEC = ABS − the mean log return of the other PIT members in the same sector on *i*. It needs at least 2 such
  members; otherwise the event is left out of the SEC tests.

"Bad" means below the expanding lower-third quantile of the **same measure** over all earlier events, past-only
(the first 200 events are burn-in), as in v0.4-r1.

### Conditioning variables (on bad-MKT events, known at the close of *i*)

- Surprise: the SUE of the same filing (seasonal random walk, PIT). Groups: SUE < 0 against SUE ≥ 0. Left out if the
  latest usable filing at *i* is not this event's quarter.
- Volume confirmation: VR = V_i / median(V_{i−20..i−1}). High means above the expanding median of VR over earlier
  events.
- Prior momentum: PM = log C_{i−1}/C_{i−127} minus the median of the same over members on *i*. Groups: PM > 0
  (prior winner) against PM ≤ 0 (prior loser).

### Outcome

CAR_h = log(C_{i+h} / O_{i+1}) − log(mean over members k of C_{i+h,k} / O_{i+1,k}). **Primary h = 20**; 5 and 60 are
secondary. Standard errors are clustered by reaction day.

### Hypotheses (one Holm family, primary horizon)

| Id | Hypothesis | Test |
|---|---|---|
| C1 | bad-MKT CAR20 < 0 (same-data replication of v0.4-r1; not independent) | one-sided |
| C2 | bad-SEC CAR20 < 0 | one-sided |
| C3 | bad-ABS CAR20 < 0 | one-sided |
| C4 | within bad-MKT: SUE < 0 minus SUE ≥ 0 ≠ 0 | two-sided, cluster-robust OLS on a group dummy |
| C5 | within bad-MKT: high VR minus low VR ≠ 0 | as C4 |
| C6 | within bad-MKT: prior loser minus prior winner ≠ 0 | as C4 |

Reported but not tested for decisions: every cell at 5 and 60 sessions, by year, bull/bear (NIFTY vs its
200-session average on *i*), high/low volatility regime, and the first vs second half of events.

### Tradability (long-only first)

- **T1 avoid-overlay (primary).** Baseline: the equal-weight PIT member portfolio, rebalanced monthly. Overlay: the
  same, but a stock with a bad-MKT reaction is sold at the open of *i+1*. The proceeds go into NIFTYBEES (keeping
  market exposure), and the stock is bought back at the open of *i+21* if it is still a member. Through the engine
  with full costs. The value of the filter is the overlay's net excess over the baseline, reported at 1× and 2× costs.
- T2 short (diagnostic only; **not a proposal**): per event, the net of −CAR20 minus the stock-futures round-trip cost.

### "Candidate" rule

C1, C2 or C3 significant after Holm; T1 net value > 0 at both 1× and 2× costs; bad-MKT CAR20 < 0 in both halves
of the event sample **and** in both the bull and bear regimes. A pass earns a separate forward pre-registration as an
avoid-filter. No short strategy is built in v0.5.

---

## What v0.5 will not do

No new factors, no factor combinations or interactions (they come only after each factor is established
independently), no machine learning, no parameter grids, no intraday work, and no change to REV1. Live trading stays
disabled.

---

## Amendment 1 — 2026-10-03, before any forward record or v0.5 result

**Track A integrity rule.** The original text says a decision written after its fill session's close voids the test.
An end-of-day runner only learns that a session was the last of the month when the next session's bar arrives (or,
in the usual case, on the evening of the last weekday). So that rule would void months mechanically, including
months where nothing could have gone wrong. Replacement:
- The runner **truncates the price panel at the decision close before scoring**, so no later price can enter a
  decision, whenever it is computed. Every decision record stores a recomputation check.
- A decision is **ON_TIME** if it is written on the evening of the month-end session. That is possible when the
  latest session in the data is the last weekday of its month. Otherwise it is **LATE** (written after the next
  session's bar arrived). The paper fill is at the next session's open in both cases, exactly as in research.
  LATE is reported, never hidden.
- **VOID** remains for a changed frozen-code hash, a broken ledger chain, or a recorded decision that differs from a
  truncated recomputation for any reason other than a logged data revision.

No definition, threshold or test changes. Reason: implementation feasibility, found while building the runner.

## Amendment 2 — 2026-10-03, after an independent code review, before the forward ledger was initialised

No strategy definition, threshold or hypothesis changes. These items make the implementation match the text, or
remove an ambiguity in it:

1. **No later data at all.** "Truncated" now means the raw vendor and exchange rows after the decision close are
   dropped *before* the corporate-action correction runs. The correction fits step functions and rolling medians over
   whole series, so cutting only its output could still let later bars move earlier prices.
2. **One decision per month, confirmed by the exchange.** A month-end is accepted only once BSE's bhavcopy for that
   session is published (final data) and BSE shows no later session in the same month that the vendor lacks.
   Otherwise the runner waits, for at most 5 sessions. After that, a missing vendor month-end is MISSED (previous
   holdings kept, still counted). Nothing is booked past a month-end whose decision is still pending.
3. **Re-checks that can actually fail.** Each decision stores a hash of its exact inputs (raw vendor and BSE rows up
   to the close) and the data-pipeline file hashes. The last three decisions, and all of them after any pipeline
   change, are recomputed every run. A difference with identical inputs and pipeline is **VOID**. A difference after
   inputs or pipeline changed is a logged **revision**, and the recorded decision stands. Pipeline changes and runner
   changes are logged as records.
4. **More of the definition is hashed:** `lab.engine._profile_arrays` (cost-profile arrays) and the runner functions
   for month-end status, replay and decision rules. Coverage and wait limits now live in `spec.json`.
5. **Review timing.** "12 complete months" is the minimum. The first review is after the **October 2027** month-end, as
   stated (13 months). The turnover REJECT rule is implemented, and the initial purchase is excluded from turnover.
6. **Data-quality gate bug (pre-existing, found by the review work).** `lab.panel.build_panel` lost the per-day
   exchange-quality scores when it loaded the benchmark. So since v0.3.1 the rule "no buys on prices the exchange did
   not confirm (score < 0.5)" was silently off in research runs. It affects 77 stock-days in 6 stocks. Fixed, with a
   regression test. Impact on the frozen candidate: REV1 net excess goes from +2.245 %/yr to +2.255 %/yr. Tracks B and
   C are run with the fix.
7. **Track B/C reporting.** The economic axis uses net excess over the beta-matched benchmark, not the regression
   alpha; the alpha stays on the statistical axis. C1–C3 report the per-event mean with a CR1 day-clustered t (as
   C4–C6), and the v0.4-style day-average t is shown beside it. Turnover excludes the pre-start cash period. T1's
   tradable check uses the 4 pre-registered subperiods.
