# AuTrad AI

Enterprise-grade AI trading **research and paper-trading** platform. AI proposes; deterministic risk code decides; the broker only ever sees orders the Risk Gate approved.

> No system can guarantee profit. AuTrad AI guarantees **bounded, known loss** (daily cap, drawdown caps, kill switch) and a model that re-weights itself toward what is working. Live trading ships disabled. Not a regulated investment service.

## Quick start

```bash
pip install -r requirements-dev.txt
pytest -q                                   # 101 tests: risk, no-look-ahead, costs, ledger, stats, walk-forward, lab judge, corporate actions, PIT fundamentals, factors, API
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

## Promotion path

Research → Paper (default) → Limited Live → Live. A strategy is promoted only when it beats a cost-adjusted benchmark on every walk-forward fold **and** in paper trading, and a human with the `risk_officer` role approves.
