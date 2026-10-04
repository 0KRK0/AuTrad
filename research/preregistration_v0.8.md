# AuTrad v0.8 — pre-registration: data and company-intelligence expansion

Written 2026-10-03. **No return, alpha or portfolio statistic was computed on any v0.8 dataset** before this file
was hashed into `research/trials.jsonl`. The only numbers below are data-coverage counts from the audits in
`research/data_audits/*_2026-10-03.json`.

v0.8 is a **data** version. It runs no alpha study. Its job is to make three blocked families from the v0.7 programme
(section 1 of the v0.7 pre-registration) testable, or to state plainly why they are still blocked.

## 1. Datasets and acceptance rules

A dataset is **accepted for research** only if all of these hold. Each is checked by a script, not by eye.

| Rule | Shareholding & pledges (BSE) | Earnings-call transcripts | Analyst consensus |
|---|---|---|---|
| Point-in-time stamp from the publisher or exchange | BSE filing timestamp; the revision timestamp for revised-only quarters | NSE dissemination time; for website copies, the company's own NSE intimation (call date to +21 days), otherwise first-seen only | Our own fetch time (collected forward) |
| No silent later versions | Structured record used only if its authorisation stamp is within 5 minutes of the filing; otherwise the as-filed XBRL; otherwise the later version **from its own authorisation time** | Hash of each stored PDF | Raw vendor response stored per day |
| Coverage over the test window | ≥ 95 % of PIT member-quarters | Reported per company; no family uses transcripts yet | ≥ 24 months of daily snapshots |
| Cross-source check | NSE master (Sep 2021+): promoter % agrees within 0.05 pp in ≥ 95 % of matched filings; every disagreement explained or listed | Not applicable (single source per document) | Vendor's "actual EPS" vs filed EPS within 5 % in ≥ 90 % of matched quarters |

**Status at registration (counts only):**

* **Shareholding & pledges:** ACCEPTED. 4,042 filings, 89 companies, Mar 2016 to Jun 2026. 99.66 % of PIT member-quarters covered (7 gaps, all Mar 2016). 346 filings use the as-filed XBRL. On a 60-filing sample, XBRL and structured records agree 60/60 on promoter % and encumbered shares. NSE cross-check: 97.76 % agree within 0.05 pp; 38 of 40 disagreements are later NSE revisions and 2 are unexplained (listed in the audit).
* **Transcripts:** ACCEPTED AS COMPANY MEMORY ONLY, not for any return study. The current NIFTY 50 has 900 NSE transcript filings, 482 with full text. Another 102 company-website transcripts carry an exchange-intimation timestamp. 14 companies are still mostly cover-letter-only.
* **Analyst consensus:** NOT ACCEPTED. Day 1 of the snapshot store exists. Basis check: 61 % of 188 matched quarters fall within 5 %, against a ≥ 90 % threshold. The vendor's "actual" appears to be an adjusted EPS, so a surprise measured against it is not comparable to filed EPS.

## 2. Study 2 (ownership: promoter encumbrance) — registered, NOT run, power-gated

Hypotheses (signs fixed now):

* **H2a (events):** a quarterly filing whose promoter encumbrance (pledge, NDU and other, as % of promoter holding)
  rises by ≥ 5 pp over the previous quarterly filing is followed by a negative abnormal return. The window is
  63 sessions from the next exchange open after `available_at`, measured vs the equal-weight PIT universe. Test: mean
  CAR < 0, with t-statistics clustered by calendar month.
* **H2b (level):** in monthly Fama–MacBeth cross-sections, encumbrance ≥ 25 % predicts a lower next-month return
  (Newey–West t, 6 lags).
* Holm correction across H2a and H2b. The implementation is long-only exclusion: the equal-weight universe minus
  flagged names, rebalanced monthly at exchange-confirmed month-ends with full costs. It goes through the v0.7
  capital-efficiency gate unchanged.

**Power rule:** the study may run only if the test universe contains **≥ 100 H2a events across ≥ 30 companies**.
The PIT NIFTY 50 has **24 events across 8 companies** (a count of filings; no returns were examined). **Study 2 is
therefore blocked** until the universe is expanded to NIFTY 200/500, which needs (a) the BSE shareholding fetch for
those companies and (b) an exchange-anchored price panel with PIT membership for them. Any change to the hypotheses
or thresholds must be logged as an amendment before results.

## 3. Consensus-based families (revisions, surprise) — blocked

They stay blocked until 24 months of snapshots exist **and** the basis rule in section 1 passes. A licensed historical
source (LSEG I/B/E/S, FactSet, S&P Capital IQ, Bloomberg, or Trendlyne's paid history) would unblock them sooner. If
one is adopted, its point-in-time semantics must be audited first: the estimate date versus when it became visible.

## 4. Management promises — human-verified only

Promise → due date → outcome evidence → human verification → MET / PARTIAL / MISSED / UNRESOLVABLE
(`intel/promise_review.py`).

* The machine may only *suggest* a verdict.
* A verdict is accepted only with a named reviewer and a verbatim quote from a document published after the promise.
* Promises whose anchor is a past period are dropped.
* A per-company execution record is descriptive and labelled insufficient below 10 verified promises.
* It is **not** a trading factor in v0.8. Making it one later needs its own pre-registration and ≥ 30 verified
  promises in ≥ 15 companies.

## 5. Unchanged

* REV1 stays frozen.
* The 10th Man and Company Intelligence remain advisory (SHADOW) with zero order authority.
* Live trading stays disabled.
* Nothing in v0.8 can change a portfolio.

---

## Amendment 1 — after the independent review (2026-10-03, still before any return was computed)

An independent reviewer found 1 critical, 5 major and 6 minor defects. All are fixed, and the data, audits and
Company Memory were rebuilt (`reports/v0.8-r1_*`). Corrections to the text above:

1. **Transcript stamping (critical).**
   * The website-transcript stamp could take an *audio-recording* notice. One L&T transcript (Q4 FY22) was stamped
     6.6 days early. Now only transcript notices count, and a notice whose subject names a different quarter is
     rejected.
   * Cover-letter links take the earliest cover letter.
   * Duplicate files are removed.
   * **Corrected counts (2022-01-01 onward, current NIFTY 50):**
     * 899 NSE announcements: 848 transcript notices and 51 recording notices.
     * 482 full-text NSE transcripts; 375 link-only transcript notices.
     * **572 distinct usable transcripts** in Company Memory.
     * 14 companies remain under half covered.
2. **Shareholding parser.**
   * The 2025-05-31 XBRL taxonomy names encumbrance differently (`NumberOfSharesEncumbered*`). Five as-filed rows had
     lost their pledges (e.g. INDUSINDBK Jun/Sep 2025, 50.86 %). Fixed. An "encumbered" flag with no readable figure
     now gives *unknown*, not 0; 0 rows are affected.
   * One structured record with percentages ×100 (BAJAJ-AUTO Jun 2025 revision) is rescaled and flagged.
     0 percentages are now out of range.
3. **`as_of`.** It now returns the newest quarter-end period, in its latest public version. A late revision of an old
   quarter no longer displaces a newer quarter.
4. **Version matching.**
   * Within a qtrid shared by several filings, the structured record goes only to the nearest filing. The others use
     their own as-filed XBRL, fetched for every shared qtrid.
   * Availability is now max(filing time, authorisation time).
5. **The XBRL check is now scripted** (`scripts.v08_data shp-validate`, seed 8): 60/60 agree on promoter % and on
   encumbered shares. 11 of the 60 have non-zero encumbrance. A full scan covers 421 as-filed XBRL files across seven
   taxonomies (2016-06-23 … 2025-10-31).
6. **NSE cross-check**, restricted to periods from Sep 2021 (NSE's coverage): 99.14 % agree within 0.05 pp
   (1,744 matched). All 15 mismatches are NSE *revised* records that match a BSE revision of the same quarter;
   0 are unexplained. All mismatches are now listed in the audit.
7. **Study 2 power count is now scripted** (`scripts.v08_data shp-events`; rule: first as-filed filing per quarter,
   PIT member at filing time). **23 events across 8 companies** (the text above said 24): still blocked.
8. **H2a event definition, amended before any return was seen.**
   * As-filed encumbrance can flicker: a filing reports 0 and the next restores the earlier level (e.g. INDUSINDBK
     Mar 2019 and Jun–Dec 2020).
   * H2a events are therefore re-defined as ΔE ≥ 5 pp **above the maximum encumbrance of the previous four quarterly
     filings**, so a restored level is not a new event.
   * The power rule (≥ 100 events, ≥ 30 companies) applies to this definition.
9. **Review statuses.** Rebuilding Company Memory no longer overwrites a human-verified promise status.

**Amendment 1a (count only):** under the amended H2a definition, the scripted count is **16 events across
7 companies** (`research/data_audits/study2_power_2026-10-03.json`). Study 2 remains blocked by the power rule.
