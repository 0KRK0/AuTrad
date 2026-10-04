# AuTrad AI

Enterprise-grade AI trading **research and paper-trading** platform. AI proposes; deterministic risk code decides; the broker only ever sees orders the Risk Gate approved.

> No system can guarantee profit. AuTrad AI guarantees **bounded, known loss** (daily cap, drawdown caps, kill switch) and a model that re-weights itself toward what is working. Live trading ships disabled. Not a regulated investment service.

## Quick start

```bash
pip install -r requirements-dev.txt
pytest -q                                   # 185 tests: risk, no-look-ahead, costs, ledger, stats, walk-forward, lab judge, corporate actions, PIT fundamentals, factors, forward test, tax, advisory agents, company memory, gate, v0.8 data layer, API
python -m scripts.fetch_data                # v0.2: real India data (NSE, NIFTY, BTC/ETH INR, USD/GBP/EUR INR, gold/silver ETFs)
python -m scripts.validate                  # v0.2: walk-forward validation + baselines + random traders + scorecard -> reports/
python -m scripts.run_backtest              # synthetic data (mechanics only)
python -m scripts.run_agents_demo          # category agents: tweet, RBI speech, penny news vs pump, your overrides
python -m scripts.run_collector --once      # pull filings / central-bank feeds as evidence
uvicorn api.main:app --reload               # API on :8000  (token: dev-admin)
cd web && npm i && npm run dev              # control plane on :5173
```

Full stack: `cp .env.example .env` (set secrets) then `docker compose up --build` → http://localhost

## Layout

| Path | What |
| --- | --- |
| `argo/` | ARGO model: point-in-time features, RLS online experts, regime gate, Thompson-sampling allocator, Page-Hinkley drift sentinel, fractional-Kelly sizing |
| `agents/` | One agent per category (large/mid/penny stocks, index F&O, BTC/ETH, alts, meme coins, INR FX, commodities), event classifier, shock + pump detector, learned impact graph, Decision agent, manual override book |
| `marketdata/universe.py`, `store.py` | India universe; fetch, clean, quality gate, SHA-256 manifest (every result names the data it ran on) |
| `execution/costs.py` | Indian charges per asset class: STT, stamp, exchange, SEBI, GST, DP, currency-futures brokerage cap, crypto exchange fees; spread and impact tiers |
| `backtest/sim.py`, `strategies.py`, `ledger.py` | Simulator v2 (one engine for AuTrad and baselines, cash yield, kill cooldown), buy & hold, momentum, MA 50/200, random traders, round-trip trade ledger |
| `validation/` | Metrics, PSR / deflated Sharpe / bootstrap CI, India tax estimate, walk-forward selection, qualification scorecard, report writer |
| `agents/allocator.py` | Category capital allocator (budgets shrink for losing categories, never above policy caps) |
| `backtest/multi.py` | Multi-agent backtest with injected events and scheduled overrides |
| `risk/gate.py` | Deterministic Risk Gate + latched Kill Switch |
| `execution/broker.py` | `BrokerInterface`, `PaperBroker` (spread, commission, sqrt impact, participation-capped partial fills, idempotent orders, 1-bar latency) |
| `backtest/engine.py` | Event-loop backtester, walk-forward folds, Sharpe/Sortino/DD/VaR/CVaR, benchmark |
| `intel/collector.py` | Internet Intelligence Layer: robots-aware async collector, SimHash dedupe, entity linking, sentiment, learned source reliability, point-in-time evidence features |
| `audit/log.py` | Hash-chained, tamper-evident decision log |
| `api/` | FastAPI: RBAC, rate limit, GZip, Redis/LRU cache, asyncpg pool, keyset pagination, batched (non-N+1) queries |
| `web/` | React + Vite control plane: route code-splitting, lazy chart, skeletons, debounced search, memoised rows |
| `db/schema.sql` | Postgres + TimescaleDB: indexes, hypertables, compression, continuous aggregate |
| `deploy/nginx.conf` | Load balancer, CDN-grade static caching, gzip, request limiting |

## v0.2 real-market validation

`scripts/validate` answers one question per asset group: does AuTrad have an edge after Indian costs, out of sample?
It runs 6 ARGO candidates (prediction horizon 1/5/20 bars, rebalance cadence, cost hurdle), chooses one per year using only
past data (walk-forward from 2018), stitches the test years into one out-of-sample record, and compares it with buy & hold,
6-month momentum, MA 50/200, the NIFTY 50 index and 100 random traders taking identical risk. The scorecard verdict is
`QUALIFIED_FOR_PAPER` only if all 12 gates pass; otherwise `RESEARCH`.

## v0.3 multi-horizon research lab

`lab/` is a research laboratory. It decides which strategies, at any horizon, have a real net edge in Indian markets, and whether ARGO should allocate between them. **The judge was built first, and every strategy is judged the same way.**

```bash
python -m scripts.fetch_data --data-dir data/snap_20261001        # daily bars + point-in-time NIFTY 50 members
python -m scripts.fetch_intraday --data-dir data/snap_20261001    # hourly (~2y) and 5-minute (~60d) bars
python -m scripts.lab --data-dir data/snap_20261001               # full run, ~25 min; --quick for a smoke test
```

| Path | What |
| --- | --- |
| `marketdata/pit.py`, `reference/` | Point-in-time NIFTY 50 membership (22 snapshots, 2016–2026). No survivorship bias. |
| `marketdata/corporate_actions.py` | Versioned fixes for splits, bonuses, demergers and vendor price blips. Raw files are never edited. |
| `lab/panel.py`, `lab/engine.py` | Aligned date×symbol panel. Fast engine: next-open execution, exact Indian charges, spread, sqrt impact, participation cap, futures carry. |
| `lab/strategies.py` | Families:<br>• CORE: momentum, low-vol, index trend, NIFTY+gold mix<br>• SWING: breakout, RSI-2, reversal, relative strength<br>• CROSS_ASSET: trend across equity, gold, silver, USDINR and BTC<br>• REL_VALUE: sector pairs on futures<br>• BLOCKED: value, quality and events, which have no point-in-time data yet |
| `lab/intraday.py` | Hourly MIS lab: opening-range breakout, intraday momentum, VWAP trend. Direction and entry-time nulls, plus a 5-minute resolution check. |
| `lab/judge.py` | The frozen judge, in order:<br>1. Select on 2021–23 only.<br>2. Sealed test from 2024 onward.<br>3. Walk-forward.<br>4. Cost shock at 0/1/1.5/2/3×.<br>5. Capacity from ₹1L to ₹5Cr.<br>6. Parameter perturbation.<br>7. Random selection and timing nulls.<br>8. PSR, DSR (all trials ever run), bootstrap CI, PBO.<br>9. Regime table.<br>10. Survivorship test.<br>11. The 12 gates. |
| `lab/meta.py` | ARGO-Meta: regime-aware Bayesian allocator across sleeves. Shrinkage, a confidence gate, fractional Kelly, a volatility target, a kill switch, and cash as the default. Judged against equal-weight, best-single, NIFTY, cash and a random-allocation null. |
| `lab/registry.py`, `research/trials.jsonl` | Every configuration ever tried (kept in git), so Deflated Sharpe charges for all of our searching. |

**Latest result (`reports/v0.3_20260930T233002Z`):** 13 strategies from hourly to multi-year, plus ARGO-Meta. None qualified for paper trading on the sealed 2024–26 test.
- Intraday and swing rules lose to Indian costs: hourly rules make −5 to +2 bps gross per trade against 12–14 bps of costs.
- The closest candidates are slow: a NIFTY+gold mix (9/12 gates, added after the test was seen) and NIFTY trend (7/12).
- ARGO-Meta lowers drawdowns but is no better than random allocation across the same sleeves (p = 0.26).
- Next: forward paper-test the two slow candidates, and recast ARGO as the risk layer.

Outputs go to `reports/v0.3_<ts>/report.md`, `results.json` and `navs.csv`. Each run records the data manifest hash, the cost-model version and the corporate-action version.

## v0.4 data integrity (exchange-anchored prices)

Vendor prices (Yahoo/NSE) are no longer trusted for corporate actions. Every NSE stock and ETF is re-anchored on **BSE's
official unadjusted daily bhavcopy**, an independent exchange record.

```bash
python -m scripts.fetch_exchange --data-dir data/snap_20261001     # BSE prices for every listed symbol (~5 min, cached)
python -m scripts.data_audit --data-dir data/snap_20261001         # reports/data_integrity_<ts>/report.md + CSVs
```

| Path | What |
| --- | --- |
| `marketdata/exchange.py` | BSE bhavcopy downloader and cache (old and new file formats, weekend special sessions, sha256 manifest) |
| `marketdata/corporate_actions.py` | Fits log(vendor/exchange) as a step function and classifies every step: split/bonus (exact ratio), dividend, rights, demerger (exchange price discovery), or vendor artifact (removed). It then rebuilds the adjustment from verified factors only, repairs misprints, bad opens, stale and phantom bars, and scores every asset-date for data quality |
| `marketdata/reference/corporate_actions.csv` | Verified non-standard events (demergers, rights, unit splits, special dividends), each with terms and a public source |
| `marketdata/reference/bse_codes.csv` | Symbol → BSE scrip code, with an ISIN cross-check against the point-in-time file |
| `marketdata/reference/verified_real_moves.csv` | Large drops confirmed as real market moves (not missed adjustments) |
| `marketdata/audit.py`, `scripts/data_audit.py` | Audit and report: fake one-day moves, ex-date accuracy against published terms, missed adjustments, level consistency |

`DataStore.load()` applies this automatically when `data/exchange/` exists. The lab will not open a position on a day
whose data-quality score is below 0.5. The research ledger (`research/trials.jsonl`) now also stores one full
experiment record per strategy run: git sha, data and exchange hashes, periods, cost model, all metrics, and the decision.

## v0.4 information + factor research

Point-in-time fundamentals and earnings events from NSE's own results filings, and a pre-registered factor study.

```bash
python -m scripts.fetch_fundamentals --root data/pit           # NSE results index + XBRL (~2,750 filings); needs a non-cloud IP
python -m scripts.rebuild_fundamentals --root data/pit         # re-parse the stored raw files offline
python -m scripts.factors --label v0.4-r1 --sleeves reports/v0.3.1_<ts>/navs.csv
python -m scripts.data_manifest --label v0.3.1 --audit reports/data_integrity_<ts>   # immutable data manifest
```

| Path | What |
| --- | --- |
| `research/preregistration_v0.4.md` | 17 factor definitions, timing rule, tests and pass criteria. Written and hashed into the ledger before any result |
| `marketdata/nse_filings.py` | NSE results index (legacy and 2025+ Integrated Filing) and an XBRL parser that handles both namespaces, bank taxonomies and undefined contexts. Each filing keeps period_end, published_at, available_at, revision and sha256 |
| `marketdata/fundamentals.py` | Point-in-time store: "what was known at 15:30 IST on day t". Revisions apply only from their own date. Splits are keyed on publication time, one statement basis is used per comparison, and validation blanks parse errors (never fills them) |
| `lab/factors.py`, `scripts/factors.py` | Factor scores, IC with Newey–West t, Holm correction across all 17, a net-of-cost long-only tercile, a long-short diagnostic, robustness checks, an earnings event study, and sleeve attribution |
| `scripts/data_manifest.py`, `research/manifests/` | Immutable manifest of exactly which data, sources, corrections and code produced a result |

**Latest result (`reports/v0.4-r1_<ts>`, after an independent review):**

- **Only REV1 passed** (one-month reversal within the NIFTY 50) of the 17 pre-registered factors. Its signal is robust (IC 0.056, t 3.3), but its after-cost portfolio gain is not significant (+2.3% a year, t 0.9). It is a weak candidate for the forward paper test, not a strategy.
- **Quality, growth, SUE, low volatility and beta show no premium** in Indian large caps over 2019–26. Value is positive but not significant.
- **After bad earnings reactions, stocks keep falling** (−103 bps over 20 sessions, t −3.3). Good reactions show no tradable drift.

## v0.5 alpha validation

No new factors. One candidate is tested forward, and two findings are taken apart under frozen rules
(`research/preregistration_v0.5.md`, hashed before any result).

```bash
python -m scripts.forward_rev1 --fetch      # every NSE session evening (~18:30 IST+): download, decide at month-ends, mark, re-check
python -m scripts.v05 A|B|C|report --out reports/v0.5_<ts>     # research side: REV1 cost check, low-vol decomposition, earnings reaction
```

| Path | What |
| --- | --- |
| `forward/rev1.py`, `forward/ledger.py` | Frozen REV1 forward paper test from 1 Oct 2026. Hash-chained ledger; hashed strategy code (any change = VOID). Month-ends are confirmed by the BSE bhavcopy, raw data are cut at the decision close *before* corrections, decisions are re-checked every run, and the production Risk Gate runs in shadow |
| `forward/rev1/status.md` | Live state: paper NAV vs equal-weight NIFTY 50 and NIFTYBEES, decisions, audit events |
| `lab/v05.py`, `scripts/v05.py` | Newey–West and cluster-robust OLS, Brinson split, ex-ante beta, regime flags, past-only flags, and a four-axis evidence scorecard (statistical / economic / tradable / forward) |

**Result (`reports/v0.5_<ts>/report.md`, after an independent review):**

- **REV1 is now running forward, untouched.** Research-side it is cost-fragile: +2.3 %/yr net, −0.5 %/yr at double costs, and
  2 of 4 subperiods positive. At its information ratio a forward test can catch it breaking, but cannot prove it.
- **Low volatility is mostly market beta.** The "+3.2 % alpha" falls to +2.7 % (t 1.4) once the equal-weight and size tilts
  are included. All its beta-adjusted edge came in months when NIFTY was below its 200-day average, and it lost in
  2024–26. Not a candidate.
- **Bad earnings reactions keep falling** on market-, sector- and absolute-adjusted measures (−92 to −122 bps over 20
  sessions, Holm p ≤ 0.011). But a stand-alone long-only "avoid" filter earns +0.7 %/yr at normal costs and loses at
  double costs. Not a candidate on its own.
- Found and fixed a pre-existing bug: the exchange-quality buy block had been silently off in research since v0.3.1
  (77 stock-days; REV1 effect +0.01 %/yr).

## v0.6 execution, tax, earnings veto, and the advisory agents

```bash
python -m scripts.fetch_nse_bhav --budget 160          # NSE official bhavcopy (on your machine; repeat until todo 0)
python -m scripts.v06 run --out reports/v0.6_<ts>       # exchange opens, tax, REV1 x bad-earnings veto (pre-registered)
python -m scripts.agents_v06 calibrate --out <same>     # 10th Man base-rate model, walk-forward Brier skill
python -m scripts.agents_v06 shadow --news --out <same> # Head + Company Intelligence + 10th Man on REV1's forward decisions
python -m scripts.v06 report --out <same>
```

| Path | What |
| --- | --- |
| `marketdata/nse_bhav.py`, `nse_ids.py` | NSE official bhavcopy (legacy and UDiFF) matched by ISIN first; `build_panel(open_source="exchange")` uses the exchange open with a continuity guard |
| `accounting/tax_india.py` | FIFO lots; STCG/LTCG by sale date (15 %→20 %, exempt→10 %→12.5 % above ₹1–1.25 lakh); set-off, 8-year carry-forward, cess; post-tax NAV |
| `intel/evidence.py` | Evidence Ledger. Every fact quotes a source verbatim. Tiers: 1 exchange filing … 8 anonymous; 9 = AuTrad data extract (never proof). All times compared in UTC |
| `intel/sources.py`, `intel/claims.py` | NSE announcements and board meetings, PIT filings, news headlines (opt-in), and official X accounts (official API, verified handles only). Management-claim tracker with a reliability record |
| `agents/company_intel.py` | Company Intelligence: a per-company dossier (identity, filings, reactions, announcements, scheduled meetings, news, claims, red flags, unknowns) |
| `agents/tenth_man.py` | The 10th Man: an isolated adversarial reviewer. Sanitised request only; no reference to any agent, gate or broker. Its probability comes from a base-rate model, shown only if it beats the base rate out of sample. It may say "no material objection" |
| `head/main_head.py` | Main Head: routes everything; SHADOW policy (memo of what a challenge-aware Head would do; targets unchanged); thesis template hashed |
| `forward/shadow/` | Hash-chained shadow memos and the full message audit, separate from the frozen REV1 ledger |

**Result (`reports/v0.6-r1_<ts>/report.md`, after an independent review):**

- Exchange opens ≈ Yahoo opens (median difference 0.0002 bps). REV1 net excess changes from +2.26 %/yr to +2.22 %/yr. Costs are the problem, not prices.
- **After Indian tax, REV1 trails the equal-weight NIFTY 50.** Before tax it was +1.6 %/yr ahead, after tax it is −0.6 %/yr; 17× turnover pays short-term gains tax every year.
- **A bad-earnings veto adds no measurable information to REV1** (Fama–MacBeth t −0.3; REPLACE − REV1 +1.7 %/yr, t 1.15, negative in 2022–23). No REV2.
- The 10th Man's flags do not predict which REV1 picks fail (Brier skill −0.005), so it shows the base rate only. First shadow memo (30 Sep picks): no material objection; findings and unknowns are listed per stock with sources.

## v0.7 low-turnover programme, company memory, stricter 10th Man

The reversal branch is **closed**: REV1 is a frozen benchmark and forward experiment, and every idea built on it is REV2+.

```bash
python -m scripts.v07 audit|study|register|report --out reports/v0.7_<ts>   # data audit, study 1, every candidate through the gate
python -m scripts.intel_v07 transcripts --budget 150                         # NSE-filed earnings-call transcripts (pilot companies)
python -m scripts.intel_v07 build --out <same>                               # evidence & claim graph + management-promise pilot
python -m scripts.agents_v06 scorecard --out <same>                          # does the 10th Man improve decisions? (shadow)
```

| Path | What |
| --- | --- |
| `lab/gates.py` | Capital-efficiency gate, strictly in order: statistical → +2 %/yr after costs → after-tax estimate → 2× costs → capacity ₹1 Cr → ⅔ subperiods → forward test. Only then may the Head *consider* capital, with owner approval |
| `marketdata/cashflow.py`, `lab/cfq.py` | Operating cash flow from NSE XBRL (from Mar 2021; checked against annual reports) and point-in-time cash-flow quality (accruals) |
| `agents/tenth_man.py`, `head/scorecard.py` | Report in three parts: historical probability (statistical layer only), evidence challenge (MATERIAL/NONE, tier-1/2 only), if/then failure scenarios (no probabilities). Pre-registered shadow scorecard |
| `intel/graph.py` | Evidence & claim graph (SQLite): documents (tiers 1–9), claims, metrics, events, decisions; tier-aware confidence that never rises by repetition; `trace()` answers "why did the Head buy X?" |
| `intel/transcripts.py` | NSE-filed earnings-call transcripts. A rule-based extractor of dated, measurable management promises (verbatim quotes) |

**Result (`reports/v0.7-r1_<ts>/report.md`, after an independent review):**

- **No candidate passes the gate.** REV1 (2016–26) clears statistics, +2.2 %/yr and, barely, the after-tax estimate, then fails at 2× costs.
- **Study 1, cash-flow quality, is rejected at the statistical step** (IC t 0.41; Fama–MacBeth coefficient negative), despite a portfolio that looked strong (+5.7 %/yr, 1.4× turnover). All its excess came in Dec 2022 – Jun 2024.
- Insider, pledge and analyst-revision data are blocked by coverage (audited); there is no family to test next until the data exist.
- The 10th Man found no material objection on the 30 Sep picks. The shadow scorecard is accumulating.
- Company memory: 4,330 documents and 6,157 filed numbers. Promise pilot: 15 verbatim promises (UltraTech 14, TCS 1); none auto-scored, outcomes go to human review.

## v0.8 data + company-intelligence expansion

v0.8 is a data version: it computes no returns. Run it on your own machine; the exchange and company sites are not
reachable from the cloud.

```bash
python -m scripts.fetch_bse_shp --budget 160                   # BSE shareholding & pledge history (resumable; repeat until todo 0)
python -m scripts.v08_data shp-build|shp-audit|shp-validate|shp-crosscheck|shp-events
python -m scripts.v08_transcripts nse|web|links --budget 160   # NSE + company-website transcripts (resumable)
python -m scripts.v08_transcripts audit
python -m scripts.fetch_estimates --budget 160                 # daily consensus snapshot (run once a day); --audit
python -m scripts.intel_v08 build --out reports/v0.8_<ts>      # Company Memory + promise review queue
python -m scripts.intel_v08 apply --csv <filled queue>         # human-verified promise verdicts
```

| Path | What |
| --- | --- |
| `marketdata/bse_shp.py` | BSE shareholding pattern from Mar 2016: promoter holding, pledged and encumbered shares, institutional mix. Usable from BSE's filing time. Uses as-filed XBRL whenever BSE's feed serves a later version |
| `intel/company_ir.py` | Company-website transcripts. Collectors for Maruti, L&T and Tata Steel, plus links followed from NSE cover letters. Dated by the company's NSE *transcript* notice, otherwise first-seen only |
| `scripts/v08_numbers.py` | Single source of truth for the v0.8 headline numbers (regenerated from the audits; checks the report and README) |
| `marketdata/estimates.py` | Daily consensus snapshots (tier 6). The vendor's look-back columns are labelled as the vendor's claim |
| `intel/promise_review.py` | Promise → due → outcome evidence → human verdict (MET / PARTIAL / MISSED / UNRESOLVABLE). Requires a named reviewer and a verbatim quote from a later document |

**Result (`reports/v0.8-r1_<ts>/report.md`, after an independent review):**

- **Shareholding & pledges accepted:** 4,042 filings, 99.66 % of PIT member-quarters (347 read from the as-filed XBRL), 99.14 % agreement with NSE.
  It is too sparse inside the NIFTY 50 for a pledge study (16 events across 7 companies, against 100/30 required).
  That study needs the NIFTY 200/500.
- **Transcripts:** 572 distinct usable transcripts for the current NIFTY 50 (Company Memory only). 14 companies are
  still mostly cover-letter-only.
- **Consensus: not accepted.** No free point-in-time history exists, so the snapshot clock starts now. The vendor's
  EPS basis matches filed EPS only 61 % of the time.
- **Promises:** 209 verbatim promises, 86 due for human review. None are scored by the machine.
- Proven alpha: still none. Live trading stays disabled. The pre-live checklist is in the report, §6.

## v0.8.1 wider universe (TOP100/200/500), pledge study, crypto (all coins)

```bash
python -m scripts.fetch_nse_all --years 2015-2026 --budget 160              # whole-market NSE bhavcopy (resumable)
python -m scripts.v081_universe build|audit                                 # corporate-action-adjusted total returns + PIT TOP100/200/500
python -m scripts.v081_bse_map                                              # ISIN -> BSE code for every TOP500 member
python -m scripts.fetch_bse_shp --wide --parts promoter --shard 0/2         # promoter/pledge filings for the wide universe (repeat; both shards)
python -m scripts.v081_study2 data|run --out reports/v0.8.1_<ts>            # audit + power count, then Study 2 (only if both pass)
python -m scripts.fetch_crypto --budget 160                                 # every Binance USDT coin, incl. delisted
python -m scripts.v081_crypto audit|run --out <same>                        # crypto C1/C2 with Indian crypto tax
```

| Path | What |
| --- | --- |
| `marketdata/nse_corp_actions.py` | NSE corporate actions for every company → split/bonus/consolidation/rights factors and cash. NSE's archived prev_close is *not* split-adjusted |
| `marketdata/nse_universe.py` | Whole-market panel (ISIN-linked securities), adjusted total returns with flags, point-in-time liquidity universes |
| `marketdata/crypto_binance.py`, `accounting/tax_crypto_india.py` | All Binance USDT pairs from the public archive (delisted included); Indian crypto tax: 31.2 % per transfer, no loss set-off, 1 % TDS, never a refund |

**Result (`reports/v0.8.1-r1_<ts>/report.md`, after an independent review; numbers in `research/data_audits/v0.8.1_canonical_numbers.json`):**

- **Nothing passed. Proven alpha is still 0.**
- **Pledge study (TOP500, 317 events across 187 companies):** sharp pledge increases are followed by −2.50 % over 63
  sessions (t −1.76, Holm p 0.167, not significant). Encumbrance level predicts nothing (t −0.18). Avoiding pledged
  names: +1.10 %/yr net, but only +0.62 %/yr gross.
- **Crypto (658 coins, 194 delisted):**
  - The BTC and ETH 140-day trend rules cut drawdowns but are not statistically better than holding (t −0.66 and
    −0.18).
  - Weekly top-50 momentum lost 48.9 %/yr; after Indian tax it was wiped out.

## v0.9 market microstructure & execution alpha lab (study M0)

```bash
python -m scripts.v09_micro sample                       # pre-registered sample of archive days (seed 2026)
python -m scripts.v09_micro fetch --day N                # download + process one day (BTCUSDT perp book + trades + spot; raw deleted)
python -m scripts.v09_micro audit|run --out reports/v0.9_<ts>
python -m scripts.record_nse_depth                       # record NSE depth forward (your Kite Connect key); then: v09_micro nse-grid
```

| Path | What |
| --- | --- |
| `marketdata/micro_binance.py` | Binance public archive -> 100 ms grid; event-level order-flow imbalance; raw files sorted per hour (some interleave two time streams) |
| `scripts/v09_analysis.py` | 11 microstructure hypotheses (test-day hours, Holm), training-day OLS model, queue-aware execution simulator (market and limit orders), latency curve, microstructure gate |
| `marketdata/nse_depth_recorder.py` | NSE depth recorder (Kite full mode, 5 levels) and converter to the same grid; NSE index-futures cost (~5.9 bp round trip) |

**Result (`reports/v0.9-r1_20261004T044028Z/report.md`, after an independent review):**

- **Order flow is predictable, but not profitably.** All 11 hypotheses are significant out of sample (for example,
  order-book imbalance IC 0.34 and order-flow imbalance IC 0.19 for the next 5 s), but the predictable move is
  under 1 bp.
- Market orders never trigger: the predicted move never exceeds the 3.4–10 bp round trip.
- Queue-aware limit orders lose 7.9 bp per trade at retail fees and 2.6 bp at best-case fees. After a fill the mid
  drifts 0.93 bp against the order (adverse selection), more than the 0.37 bp spread captured. This holds at every
  latency from 0 to 5,000 ms.
- **REJECTED at the economic step. Proven alpha is still 0.**

## v1.0 Monetisation Lab (M1 execution alpha, M2 market making, M3 funding carry, AEM)

```bash
python -m scripts.v10_micro m1 --day N                   # M1 parent orders on M0 test day N (P0 cross / P1 post-then-cross / P2 AEM-timed)
python -m scripts.v10_micro m2 --day N --variant plain   # M2 market-making loop (plain | skew | plain_q0.5 | plain_q0)
python -m scripts.v10_micro results --out reports/v1.0_<ts>
python -m scripts.v10_m3 reports/v1.0_<ts>               # M3 funding/basis carry, BTC & ETH 2020-2026
```

| Path | What |
| --- | --- |
| `lab/aem.py` | Alpha Economics Model: gross -> net edge (spread, fees, adverse selection, impact, latency, tax without set-off, opportunity) -> EXECUTE / WAIT / ABANDON |
| `scripts/v10_micro.py` | M1 implementation-shortfall study and M2 queue-aware market-making simulator on the M0 grids |
| `marketdata/crypto_perp.py`, `scripts/v10_m3.py` | Binance funding rates + perpetual bars; carry with costs, Indian tax, subperiods |

**Result (`reports/v1.0-r1_20261004T100818Z/report.md`, after an independent review):**

- **M1:** posting and crossing only at a 60 s deadline cuts BTC-perpetual execution cost from 5.01 to 3.39 bp at
  retail fees. Adding the microstructure trigger makes it worse (+0.10 bp), because crossing pays the 3 bp
  taker–maker fee gap. REJECTED.
- **M2:** market making needs a rebate: the break-even maker fee is −0.96 bp (−0.09 bp even at the front of the
  queue). The mid moves 0.90 bp against us within 5 s of a fill. REJECTED.
- **M3:** funding carry earns +4.0 %/yr (BTC) and +5.2 %/yr (ETH) before tax (t ≈ 6.5), below the 8 % hurdle. It is
  negative after Indian tax without set-off. REJECTED.
- **Proven alpha is still 0.** The edges belong to fee tier, queue priority and tax treatment. Next: the NSE
  cash–futures basis, where set-off applies.

## v1.1 Indian Basis & Relative-Value Lab (cash–futures carry, index carry, roll-window forced flow)

```bash
python -m marketdata.nse_fo --years 2015-2026            # NSE F&O bhavcopy (futures only) + index closes, resumable
python -m scripts.v11_basis audit|run --out reports/v1.1_<ts>
python -m scripts.research_dashboard                     # research/dashboard.md: tested / rejected / surviving, mechanisms
python -m scripts.record_nse_depth --seal                # QC + seal recorded NSE depth days (immutable archive)
```

| Path | What |
| --- | --- |
| `marketdata/nse_fo.py` | NSE F&O bhavcopy (legacy + UDiFF), futures only; `ind_close_all` index closes |
| `scripts/v11_basis.py` | HB1 monthly stock cash-and-carry, HB2 one-day dislocations, HB3 index carry via ETF, HB4 roll window; retail costs with dated STT, Indian tax (business income + STCG, set-off rules), capital incl. margin, excess over RBI repo |
| `lab/aem.py` (`Opportunity`, `price`, `allocate`) | Opportunity Pricing Engine: EV per unit of capital per day after costs, tax and the after-tax cash alternative; capital allocator with capacity and risk-group caps |
| `lab/registry.py` | from v1.1 every experiment row must state its `economic_mechanism` |

**Result (`reports/v1.1-r1_20261004T112103Z/report.md`, after an independent review):**

- **The NSE cash–futures basis is efficiently priced for retail.** Stock-futures carry is about repo + 0.5 %. A retail
  round trip costs about 33 bp, most of it the 0.1 % STT on each cash leg. The monthly cash-and-carry book returns
  −0.91 %/yr against cash.
- One-day dislocations fade: 39 % of the signalled carry is left the next day, and the strategy returns −7 %/yr on
  deployed capital.
- Index futures trade slightly below repo carry, so there is no ETF carry trade (NIFTY −0.36 %/yr, BANKNIFTY
  −0.17 %/yr).
- **Forced flow:** the NIFTY roll window shows nothing. Stock futures' calendar spread rises 0.36 %/yr in the last 3
  sessions (t 4.9), about 3 bp per roll; the execution layer should roll early.
- **Disclosed:** the pre-registered signal (+1.0 %/yr) relied on dividend ex-dates that were probably not yet
  announced at the signal date, so it was removed (Amendment 1). The next lead is "do futures underprice *announced*
  dividends?", which needs announcement timestamps.
- **Proven alpha is still 0.**

## v1.2 Forced Flow & Events (dividend announcements, index changes, F&O bans)

```bash
python -m marketdata.nse_events div --shard 0/3      # timestamped dividend announcements (NSE), resumable
python -m marketdata.nse_events ban                  # F&O ban lists + MWPL / open interest
python -m marketdata.index_changes                   # NSE Indices press releases -> data/events/index_changes.csv
python -m scripts.v12_events audit|run --out reports/v1.2_<ts>
```

| Path | What |
| --- | --- |
| `scripts/v12_events.py` | HD dividend-incorporation event study and trade, HI NIFTY 50 / NEXT 50 inclusion-exclusion, HF ban measurement |
| `lab/registry.py` | from v1.2: economic_counterparty, mechanism_constraint, our_advantage, failure_condition required |
| `lab/aem.py` `allocate()` | objective (E[P&L] - after-tax cash)/(capital x time); capacity, concentration, drawdown, margin, fill-probability limits |
| `lab/execution_rules.py` | research results that become execution rules (stock-futures roll window; F&O ban) |

**Result (`reports/v1.2-r1_20261004T135141Z/report.md`, after an independent review):**

- **Dividends:** futures had priced 87 % of a dividend before the filing and 93 % by the first close after it. The
  trade earns +0.5 %/yr (t 0.3), and being first is worth nothing (same-close placebos match). This settles the v1.1
  "+1 %" as look-ahead.
- **Index changes:** announcement-to-rebalance drift is +0.4 % for NIFTY 50 (t 0.5) and +1.2 % for NEXT 50 (t 1.0),
  with no reversal. The hedged futures trade loses 49 bp per trip.
- **F&O bans:** futures go to a discount in distressed names only (median +0.5 bp). The open-interest "forced
  unwinding" disappears when compared with stocks at the same OI level. Measurement only.
- **Proven alpha is still 0.**

## v1.3 Information Advantage Lab (results-news diffusion outside NIFTY 50; forward-only analyst, depth and ban studies)

```bash
python -m scripts.v13_data index|xbrl [--needed] --shard 0/3        # NSE results index + XBRL facts (stock-futures names)
python -m scripts.v13_info audit|run --out reports/v1.3_<ts>        # Part A
python -m scripts.fetch_estimates --universe fo                       # Part B: daily consensus snapshot, 213 names (schedule = your decision)
python -m scripts.v13_depth status                                     # Part C: sealed until 20 recorded NSE depth days
python -m forward.ban_expiry update|status                             # Part D: forward-only ledger (no backfill)
```

**Result (`reports/v1.3-r1_20261004T162648Z/report.md`, after an independent review):**

- **Bad results do not keep drifting outside NIFTY 50.** Over 1,562 events, CAR20 is +0.74 % (wrong sign) and the
  hedged futures short loses 93 bp per trade, mostly because it is a short-midcap bet (slope −0.99). REJECTED.
- **v0.5's −107 bp NIFTY 50 drift does not survive either.** It was about half the equal-weight benchmark and half the
  first day after the reaction. Only a one-day continuation (−33 bp) is robust, and it has not been tested.
- **HM1b (XBRL deterioration) was not testable at this run:** NSE throttled downloads.
- **What remains needs forward data:** analyst revisions (daily snapshot), NSE depth (your Kite key) and human verdicts
  on Company Memory promises.
- **Proven alpha is still 0.**

## v1.4 Data acquisition & forward research (no new backtest)

```bash
python -m scripts.fetch_estimates --universe fo          # daily 213-name analyst snapshot (scheduling = your decision)
python -m forward.data_clock                              # what each sealed study is waiting for (counts only)
python -m scripts.import_promise_reviews --from <dir>     # Promise Review Desk verdicts -> apply_reviews (re-validated)
set KITE_API_KEY=... & set KITE_ACCESS_TOKEN=... & python -m scripts.record_nse_depth   # NSE depth (your Kite key)
```

**Status (`reports/v1.4_20261004T165216Z/report.md`):**

- **Analyst snapshots:** 213 of 213 names today (186 valid). Sealed until ≥ 12 months and ≥ 24 monthly cross-sections
  with ≥ 100 valid stocks, which is the end of October 2028 at the earliest.
- **Company Memory:** 35 promises (16 companies, 83 candidate passages) are on the Promise Review Desk for human
  verdicts. Nothing is scored automatically; imports are re-checked by `apply_reviews`.
- **NSE depth:** recorder ready, not running until your Kite key is set. Sealed until 20 QC-passed days.
- **Candidate ledger** (`research/candidates.jsonl`): C-013 one-day post-results continuation (FORWARD /
  PRE-REGISTRATION REQUIRED) and C-014 ban at expiry (FORWARD / LEDGER RUNNING). Candidates are not strategies and
  cannot be traded.
- **ARGO is frozen** (`argo/FROZEN.md`; hashes pinned in `tests/test_v14.py`). Its later role is the risk layer.
- **Proven alpha is still 0.** Live trading stays off.

## Promotion path

Research → Paper (default) → Limited Live → Live. A strategy is promoted only when it beats a cost-adjusted benchmark on every walk-forward fold **and** in paper trading, and a human with the `risk_officer` role approves.
