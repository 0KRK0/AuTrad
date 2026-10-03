"""REV1 forward paper test (research/preregistration_v0.5.md, Track A, with Amendments 1-2).

Frozen strategy: at the last NSE session of each month, rank point-in-time NIFTY 50 members by
REV1 = -(C_t / C_{t-21} - 1), hold the top third equally weighted from the next open, with full Indian costs.
Every function that defines the strategy, its timing, the paper book or the decision rules is hashed in spec.json;
if any of them changes, the runner refuses to run (VOID). Data-pipeline and runner changes are allowed only as
logged records (pipeline_change / runner_version), and every recorded decision is re-checked against them.

Ledger records (forward/rev1/ledger.jsonl, hash-chained): genesis, runner_version, pipeline_change, decision,
gate_shadow, missed_decision, fills, mark, revision, integrity.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .ledger import Ledger

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent / "rev1"
PREREG = ROOT / "research" / "preregistration_v0.5.md"
FROZEN_FUNCTIONS = [("lab.factors", "price_scores"), ("lab.factors", "stock_universe"), ("lab.factors", "tercile_targets"),
                    ("lab.engine", "run"), ("lab.engine", "_profile_arrays"),
                    ("forward.rev1", "rev1_decision"), ("forward.rev1", "month_end_status"), ("forward.rev1", "replay"),
                    ("forward.rev1", "summarize")]
FROZEN_FILES = ["execution/costs.py"]
PIPELINE_FILES = ["lab/panel.py", "marketdata/store.py", "marketdata/corporate_actions.py", "marketdata/pit.py",
                  "marketdata/exchange.py", "marketdata/universe.py", "marketdata/reference/nifty50_pit_snapshots.csv",
                  "marketdata/reference/corporate_actions.csv", "marketdata/reference/bse_codes.csv",
                  "marketdata/reference/verified_real_moves.csv"]
RUNNER_FILES = ["forward/rev1.py", "forward/ledger.py", "scripts/forward_rev1.py"]
HISTORICAL = {"net_excess_per_year": 0.02245, "tracking_error_per_year": 0.087, "info_ratio": 0.26,
              "turnover_per_year": 16.84, "source": "reports/v0.4-r1_20261003T102305Z"}


def _sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    out = {f"{m}.{f}": _sha_text(inspect.getsource(getattr(importlib.import_module(m), f))) for m, f in FROZEN_FUNCTIONS}
    return out | {f: _sha_file(ROOT / f) for f in FROZEN_FILES}


def file_hashes(files: list[str]) -> dict[str, str]:
    return {f: _sha_file(ROOT / f) for f in files if (ROOT / f).exists()}


def make_spec(first_decision: str = "2026-09-30") -> dict:
    from execution.costs import COST_MODEL_VERSION
    from marketdata.corporate_actions import TRANSFORMS_VERSION

    return {"strategy": "REV1 top tercile, equal weight, monthly (frozen v0.4 definition)",
            "preregistration": "research/preregistration_v0.5.md", "preregistration_sha256": _sha_file(PREREG),
            "first_decision": first_decision, "first_fill": "next NSE session open (1 Oct 2026)", "capital": 1_000_000.0,
            "cash_rate": 0.06, "panel_start": "2024-06-01", "universe": "point-in-time NIFTY 50 (lab.factors.stock_universe)",
            "min_coverage": 0.9, "max_wait_sessions": 5, "first_review_month": "2027-10",
            "cost_model": COST_MODEL_VERSION, "transforms_version": TRANSFORMS_VERSION,
            "benchmarks": {"primary": "equal-weight PIT universe, same engine and costs, same dates",
                           "secondary": "NIFTYBEES bought at the first fill and held"},
            "code_hashes": code_hashes(), "historical_expectation": HISTORICAL,
            "decision_rules": {"VOID": "frozen code or spec changed, ledger chain broken, or a decision differs from a recomputation on identical inputs and pipeline",
                               "REJECT": "from the 2027-10 month-end: z = (excess - 2.25%*yrs) / (8.7%*sqrt(yrs)) < -2, or realized turnover > 1.5 x 16.84/yr",
                               "KEEP_OBSERVING": "otherwise; never promoted to capital on forward evidence alone (IR too low)"}}


# --------------------------------------------------------------------------- data
def load_panel(snap: Path, start: str, asof: pd.Timestamp | None = None):
    """asof cuts the RAW vendor and exchange data before any correction (nothing after it can influence the panel)."""
    from lab.panel import build_panel, india_equity_assets

    a = None if asof is None else str(pd.Timestamp(asof).date())
    return build_panel(str(snap), india_equity_assets(), start=start, end=a, asof=a, name="forward_rev1")


def exchange_sessions(snap: Path) -> set[pd.Timestamp]:
    from marketdata.exchange import load as load_bse

    try:
        return set(pd.to_datetime(load_bse(Path(snap).parent / "exchange")["date"].unique()))
    except FileNotFoundError:
        return set()


def input_hash(snap: Path, symbols: list[str], asof: pd.Timestamp) -> str:
    """sha256 of every raw vendor row (universe + NIFTY50) and BSE bhavcopy row up to `asof`."""
    h = hashlib.sha256()
    for s in sorted(set(symbols) | {"NIFTY50"}):
        p = Path(snap) / "raw" / f"{s}.csv"
        if p.exists():
            d = pd.read_csv(p, dtype=str)
            h.update(d[pd.to_datetime(d["date"]) <= asof].to_csv(index=False).encode())
    ex = Path(snap).parent / "exchange" / "bse"
    cm = pd.read_csv(ROOT / "marketdata" / "reference" / "bse_codes.csv", dtype={"bse_code": str})
    codes = set(cm.loc[cm["symbol"].isin(symbols), "bse_code"])
    for f in sorted(ex.glob("*.csv.gz")) if ex.exists() else []:
        d = pd.read_csv(f, dtype=str)
        d = d[(pd.to_datetime(d["date"]) <= asof) & d["code"].isin(codes)].sort_values(["date", "code"])
        h.update(d.to_csv(index=False).encode())
    return h.hexdigest()


def month_end_status(dates: pd.DatetimeIndex, exch: set[pd.Timestamp], first: pd.Timestamp, max_wait: int) -> list[dict]:
    """For every month from `first` on: is its last session known, and how? One entry per month (YYYY-MM).
    READY/LATE: the next vendor session is in a later month. READY/ON_TIME: the latest vendor session is the last weekday
    of its month. Either way the exchange must have published that session's bhavcopy (final data) and must not show a
    later session in the same month that the vendor lacks; otherwise WAIT, and after `max_wait` vendor sessions a
    missing exchange file is tolerated (flagged) while a missing vendor month-end becomes MISSED."""
    out = []
    months = sorted({d.to_period("M") for d in dates if d >= first})
    for M in months:
        ix = [i for i, d in enumerate(dates) if d.to_period("M") == M]
        i = ix[-1]
        d = dates[i]
        if i + 1 < len(dates):
            timing = "LATE"
        elif (d + pd.offsets.BDay(1)).to_period("M") != M:
            timing = "ON_TIME"
        else:
            continue                                            # month not finished
        waited = len(dates) - 1 - i
        later_exch = sorted(x for x in exch if x.to_period("M") == M and x > d)
        rec = {"month": str(M), "index": i, "date": d, "timing": timing, "waited": waited, "flags": []}
        if later_exch:
            rec["status"] = "MISSED" if waited >= max_wait else "WAIT"
            rec["flags"].append(f"exchange has a later session {later_exch[-1].date()} the vendor lacks")
        elif exch and d not in exch:
            if waited >= max_wait:
                rec["status"] = "READY"
                rec["flags"].append("exchange bhavcopy for the decision date still missing; decided on vendor data")
            else:
                rec["status"] = "WAIT"
                rec["flags"].append("waiting for the exchange bhavcopy (final data)")
        else:
            rec["status"] = "READY"
        out.append(rec)
    return out


def rev1_decision(P, t: int) -> dict:
    """The frozen rule on a panel that ends at t (callers pass a panel cut at the decision close)."""
    from lab.factors import price_scores, stock_universe, tercile_targets

    assert t == len(P.dates) - 1, "decision panel must end at the decision session"
    C = P.ffill_close()
    R = np.vstack([np.full((1, C.shape[1]), np.nan), np.diff(np.log(C), axis=0)])
    m = np.r_[np.nan, np.diff(np.log(P.bench.ffill().to_numpy()))]
    js = stock_universe(P, t)
    s = price_scores(P, t, js, C, R, m)["REV1"]
    tg = tercile_targets(pd.DataFrame([s.reindex(P.symbols)], index=[P.dates[t]]))
    w = tg.iloc[0] if len(tg) else pd.Series(dtype=float)
    w = w[w > 0]
    ranked = s.sort_values(ascending=False, kind="stable")
    return {"date": str(P.dates[t].date()), "universe": sorted(P.symbols[j] for j in js), "scored": int(len(s)),
            "ranking": [[k, round(float(v), 6)] for k, v in ranked.items()],
            "selected": sorted(w.index), "weights": {k: round(float(v), 10) for k, v in sorted(w.items())}}


def members_without_close(P, t: int) -> list[str]:
    from marketdata.pit import default_universe

    mem = default_universe().members(P.dates[t])
    return sorted(s for s in mem if not (s in P.idx and np.isfinite(P.C[t, P.idx[s]])))


def gate_shadow(decision: dict, held: dict[str, float], sectors: dict[str, str]) -> dict:
    """What the production Risk Gate would do with these orders, starting from the drifted holdings.
    Information only: the frozen strategy is ungated, exactly as in research."""
    from risk.gate import OrderIntent, PortfolioState, RiskGate

    gate = RiskGate()
    pf = PortfolioState(equity=1.0, sod_equity=1.0, peak_equity=1.0, weights=dict(held), sectors=sectors)
    rows, k = [], 0
    targets = {s: 0.0 for s in held if s not in decision["weights"]} | decision["weights"]
    for s in sorted(targets, key=lambda x: targets[x] - held.get(x, 0.0)):        # sells first, then buys
        k += 1
        d = gate.check(OrderIntent(s, targets[s], instrument_class="largecap", source="ai", note="forward REV1"), pf, now=k * 10.0)
        pf.weights[s] = d.weight
        rows.append({"symbol": s, "held": round(held.get(s, 0.0), 6), "requested": round(targets[s], 6), "approved": d.approved,
                     "weight": round(d.weight, 6), "reasons": d.reasons})
    changed = [r for r in rows if abs(r["weight"] - r["requested"]) > 1e-9 or not r["approved"]]
    return {"date": decision["date"], "orders": rows, "would_change": len(changed),
            "summary": "gate approves every order unchanged" if not changed else
            "; ".join(f"{r['symbol']}: {', '.join(r['reasons']) or 'rejected'}" for r in changed)}


# --------------------------------------------------------------------------- paper book
def replay(P, decisions: list[dict], capital: float, cash_rate: float) -> dict:
    """Run the recorded decisions (never recomputed) and both benchmarks through the research engine, starting with the
    cash amount that grows to exactly `capital` at the first decision close (fills and impact at the real paper size)."""
    from lab import engine

    if not decisions:
        return {}
    idx = [pd.Timestamp(d["date"]) for d in decisions]
    tg = pd.DataFrame([pd.Series(d["weights"], dtype=float) for d in decisions], index=idx).reindex(columns=P.symbols).fillna(0.0)
    ew = pd.DataFrame([pd.Series(1.0 / len(d["universe"]), index=d["universe"]) for d in decisions], index=idx)
    ew = ew.reindex(columns=P.symbols).fillna(0.0)
    nb = pd.DataFrame([pd.Series({"NIFTYBEES": 1.0})], index=idx[:1]).reindex(columns=P.symbols).fillna(0.0)
    d0 = idx[0]
    i0 = P.dates.get_loc(d0)
    days = np.r_[0, np.diff(P.dates.values).astype("timedelta64[D]").astype(float)]
    start_cap = capital / float(np.prod((1 + cash_rate) ** (days[1:i0 + 1] / 365.0)))
    fills: list[dict] = []
    r = engine.run(P, tg, capital=start_cap, cash_rate=cash_rate, name="REV1_forward", fills=fills)
    r_ew = engine.run(P, ew, capital=start_cap, cash_rate=cash_rate, name="EW_universe_forward")
    r_nb = engine.run(P, nb, capital=start_cap, cash_rate=cash_rate, name="NIFTYBEES_forward")

    def norm(nav: pd.Series) -> pd.Series:
        return nav[nav.index >= d0] / nav.loc[d0] * capital

    W = pd.DataFrame(r.weights, index=P.dates, columns=P.symbols)
    return {"nav": norm(r.nav), "ew": norm(r_ew.nav), "nifty": norm(r_nb.nav), "costs": r.costs[r.costs.index >= d0] - r.costs.loc[d0],
            "weights": W[W.index >= d0], "fills": [f | {"date": P.dates[f["t"]]} for f in fills if P.dates[f["t"]] > d0]}


def _r(x, n: int = 2):
    return None if x is None or not math.isfinite(float(x)) else round(float(x), n)


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}


def run(snap: Path, ledger_path: Path = HERE / "ledger.jsonl", spec_path: Path = HERE / "spec.json",
        status_path: Path = HERE / "status.md", log=print) -> dict:
    from marketdata.pit import SECTOR
    from marketdata.store import DataStore

    L = Ledger(ledger_path)
    probs = L.verify()
    if not spec_path.exists() or not L.of_kind("genesis"):
        raise SystemExit("ledger not initialised: run with --init first")
    spec = json.loads(spec_path.read_text())
    if _sha_file(spec_path) != L.of_kind("genesis")[0]["spec_sha256"]:
        probs.append("spec.json changed after genesis")
    now_hash = code_hashes()
    changed = [k for k, v in spec["code_hashes"].items() if now_hash.get(k) != v]
    if changed:
        probs.append("frozen code changed: " + ", ".join(changed))
    if probs:
        if not L.verify() and not any(r["kind"] == "integrity" and r.get("status") == "VOID" for r in L.records()):
            L.append("integrity", {"status": "VOID", "problems": probs})
        raise SystemExit("INTEGRITY FAILURE (forward test VOID): " + "; ".join(probs))

    # runner and pipeline versions: changes are allowed but always logged
    rv = file_hashes(RUNNER_FILES)
    last_rv = (L.of_kind("runner_version") or [{}])[-1].get("files")
    if rv != last_rv:
        L.append("runner_version", {"files": rv, "changed": sorted(k for k in rv if (last_rv or {}).get(k) != rv[k])})
    pv = file_hashes(PIPELINE_FILES)
    last_pv = (L.of_kind("pipeline_change") or [{}])[-1].get("files")
    pipeline_changed = pv != last_pv
    if pipeline_changed:
        L.append("pipeline_change", {"files": pv, "changed": sorted(k for k in pv if (last_pv or {}).get(k) != pv[k])})

    first = pd.Timestamp(spec["first_decision"])
    P = load_panel(snap, spec["panel_start"])
    exch = exchange_sessions(snap)
    store = DataStore(str(snap))
    data_hash = store.manifest_hash()
    done = {r["month"] for r in L.of_kind("decision")} | {r["month"] for r in L.of_kind("missed_decision")}
    pending: pd.Timestamp | None = None
    for st in month_end_status(P.dates, exch, first, spec["max_wait_sessions"]):
        if st["month"] in done:
            continue
        d, ds = st["date"], str(st["date"].date())
        if st["status"] == "MISSED":
            L.append("missed_decision", {"month": st["month"], "date": ds, "flags": st["flags"], "rule": "previous holdings kept"})
            continue
        missing = members_without_close(P, st["index"])
        cov = 1 - len(missing) / 50
        if st["status"] == "WAIT" or (cov < spec["min_coverage"] and st["waited"] < spec["max_wait_sessions"]):
            pending = d if pending is None else min(pending, d)
            log(f"{st['month']}: waiting ({'; '.join(st['flags']) or f'coverage {cov:.0%}'})")
            continue
        if cov < spec["min_coverage"]:
            L.append("missed_decision", {"month": st["month"], "date": ds, "coverage": round(cov, 3), "missing": missing,
                                         "rule": "previous holdings kept"})
            continue
        Pt = load_panel(snap, spec["panel_start"], asof=d)
        dec = rev1_decision(Pt, len(Pt.dates) - 1)
        computed_on = datetime.now(timezone.utc).date().isoformat()
        dec |= {"month": st["month"], "timing": st["timing"], "flags": st["flags"], "computed_on_utc": computed_on,
                "data_to_at_computation": str(P.dates[-1].date()), "members_without_close": missing,
                "input_hash": input_hash(snap, Pt.symbols, d), "vendor_manifest_hash": data_hash,
                "pipeline_hashes": pv, "code_hashes": now_hash, "versions": _versions()}
        prev = L.of_kind("decision")
        held = {}
        marks = [m for m in L.of_kind("mark") if m["date"] == ds]
        if marks:
            held = {k: v for k, v in marks[-1]["weights"].items()}
        elif prev:
            held = dict(prev[-1]["weights"])
        L.append("decision", dec)
        L.append("gate_shadow", gate_shadow(dec, held, SECTOR))
        log(f"decision {st['month']} at {ds} ({dec['timing']}): {len(dec['selected'])} of {dec['scored']} -> {', '.join(dec['selected'])}")

    # re-check recorded decisions on cut panels: always the last three, and all of them after a pipeline change
    decs = L.of_kind("decision")
    for d in (decs if pipeline_changed else decs[-3:]):
        dd = pd.Timestamp(d["date"])
        Pt = load_panel(snap, spec["panel_start"], asof=dd)
        ih = input_hash(snap, Pt.symbols, dd)
        again = rev1_decision(Pt, len(Pt.dates) - 1)
        same = again["selected"] == d["selected"] and all(abs(again["weights"].get(k, 0) - v) < 1e-9 for k, v in d["weights"].items())
        if same:
            continue
        identical_inputs = ih == d["input_hash"] and pv == d["pipeline_hashes"]
        kind = "integrity" if identical_inputs else "revision"
        key = (d["month"], ih, json.dumps(pv, sort_keys=True))
        seen = {(r.get("decision_month"), r.get("input_hash"), json.dumps(r.get("pipeline_hashes"), sort_keys=True)) for r in L.of_kind(kind)}
        if key not in seen:
            L.append(kind, {"decision_month": d["month"], "input_hash": ih, "pipeline_hashes": pv, "recorded": d["selected"],
                            "recomputed": again["selected"], "versions": _versions(),
                            "status": "VOID: same inputs and pipeline, different decision" if kind == "integrity" else
                            "inputs or pipeline changed; the recorded decision stands (no decision is ever rewritten)"})

    decs = L.of_kind("decision")
    rp = replay(P, decs, spec["capital"], spec["cash_rate"]) if decs else {}
    if rp:
        stop = pending if pending is not None else P.dates[-1]          # nothing is booked past an undecided month-end
        logged = {r["date"]: r for r in L.of_kind("fills")}
        by_day: dict = {}
        for f in rp["fills"]:
            if f["date"] <= stop:
                by_day.setdefault(str(f["date"].date()), []).append(f)
        for day, fs in sorted(by_day.items()):
            body = {"date": day, "orders": [
                {"symbol": f["symbol"], "side": "BUY" if f["delta_w"] > 0 else "SELL", "delta_w": _r(f["delta_w"], 6),
                 "value_inr": _r(f["value"]), "vendor_open_adj": _r(f["price"], 4),
                 "shares_equiv": _r(f["value"] / f["price"], 3) if f["price"] else None, "modelled_cost_inr": _r(f["cost"])} for f in fs],
                "total_cost_inr": _r(sum(f["cost"] for f in fs)), "traded_inr": _r(sum(f["value"] for f in fs))}
            if day not in logged:
                L.append("fills", body)
            elif abs((logged[day]["traded_inr"] or 0) - (body["traded_inr"] or 0)) > 1.0:
                L.append("revision", {"fills_date": day, "recorded_traded_inr": logged[day]["traded_inr"],
                                      "replayed_traded_inr": body["traded_inr"], "status": "replayed fills differ from the first record"})
        known = {r["date"]: r["nav"] for r in L.of_kind("mark")}
        for r in L.of_kind("revision"):
            known.update(r.get("navs", {}))
        drift = {str(d.date()): _r(v) for d, v in rp["nav"].items() if str(d.date()) in known and abs(v / known[str(d.date())] - 1) > 1e-4}
        if drift:
            L.append("revision", {"navs": drift, "vendor_manifest_hash": data_hash,
                                  "status": "past NAV changed on re-download or pipeline change; first-observed marks stay in the ledger"})
        last_mark = max([pd.Timestamp(r["date"]) for r in L.of_kind("mark")], default=pd.Timestamp(spec["first_decision"]))
        for d in rp["nav"].index:
            if d <= last_mark or d > stop:
                continue
            w = rp["weights"].loc[d]
            w = w[w.abs() > 1e-6]
            L.append("mark", {"date": str(d.date()), "nav": _r(rp["nav"][d]), "ew_nav": _r(rp["ew"][d]), "nifty_nav": _r(rp["nifty"][d]),
                              "cum_cost_inr": _r(rp["costs"][d]), "cash_w": _r(1 - w.sum(), 4),
                              "weights": {k: _r(v, 4) for k, v in w.sort_values(ascending=False).items()}, "vendor_manifest_hash": data_hash})
    res = summarize(L.records(), spec)
    status_path.write_text(status_md(res, L, spec), encoding="utf-8")
    log(f"{res['as_of']}: NAV {res.get('nav')}, excess vs EW {res.get('excess_ew_pct')}%, verdict {res.get('verdict')}, ledger head {L.head()[:12]}")
    return res


def summarize(records: list[dict], spec: dict) -> dict:
    """Pre-registered decision rules on the ledger records."""
    kind = lambda k: [r for r in records if r["kind"] == k]  # noqa: E731
    marks = kind("mark")
    void = any(r.get("status", "").startswith("VOID") for r in records)
    month_recs = sorted({r["month"] for r in kind("decision") + kind("missed_decision")})
    complete = [m for m in month_recs if m > spec["first_decision"][:7]]
    out = {"as_of": marks[-1]["date"] if marks else spec["first_decision"], "sessions": len(marks),
           "decisions": len(kind("decision")), "missed": len(kind("missed_decision")), "complete_months": len(complete),
           "integrity": "VOID" if void else "OK"}
    if not marks:
        out["verdict"] = "VOID" if void else "KEEP OBSERVING (nothing booked yet)"
        return out
    cap = spec["capital"]
    m = marks[-1]
    yrs = max((pd.Timestamp(m["date"]) - pd.Timestamp(spec["first_decision"])).days / 365.25, 1e-9)
    ex = math.log(m["nav"] / cap) - math.log(m["ew_nav"] / cap)
    h = spec["historical_expectation"]
    z = (ex - h["net_excess_per_year"] * yrs) / (h["tracking_error_per_year"] * math.sqrt(yrs))
    fills = kind("fills")
    traded = sum(f["traded_inr"] or 0 for f in fills[1:])                     # the initial purchase is not turnover
    turnover = traded / float(np.mean([x["nav"] for x in marks])) / yrs
    review = spec["first_review_month"] in month_recs
    out |= {"nav": m["nav"], "ew_nav": m["ew_nav"], "nifty_nav": m["nifty_nav"], "ret_pct": _r((m["nav"] / cap - 1) * 100),
            "ew_ret_pct": _r((m["ew_nav"] / cap - 1) * 100), "nifty_ret_pct": _r((m["nifty_nav"] / cap - 1) * 100),
            "excess_ew_pct": _r(ex * 100), "excess_nifty_pct": _r((math.log(m["nav"] / cap) - math.log(m["nifty_nav"] / cap)) * 100),
            "cum_cost_inr": m["cum_cost_inr"], "years": _r(yrs, 3), "z_vs_history": _r(z), "turnover_per_year": _r(turnover),
            "verdict": ("VOID" if void else
                        f"KEEP OBSERVING (first review after the {spec['first_review_month']} month-end)" if not review else
                        "REJECT (much worse than history)" if z < -2 else
                        "REJECT (turnover fault)" if turnover > 1.5 * h["turnover_per_year"] else "KEEP OBSERVING")}
    return out


def status_md(res: dict, L: Ledger, spec: dict) -> str:
    decs = L.of_kind("decision")
    gates = {g["date"]: g for g in L.of_kind("gate_shadow")}
    lines = ["# REV1 forward paper test — status", "",
             f"As of **{res['as_of']}** · ledger head `{L.head()[:16]}` · integrity **{res['integrity']}** · "
             f"pre-registration `{spec['preregistration_sha256'][:16]}` · paper only, never live", "",
             "Frozen rule: top third of point-in-time NIFTY 50 members by minus their last-month return, equal weight, "
             "rebalanced at each month-end close, filled at the next open with full Indian costs. Nothing may be changed.", ""]
    if "nav" in res:
        lines += ["| | Paper REV1 | Equal-weight NIFTY 50 (primary) | NIFTYBEES (secondary) |", "|---|---|---|---|",
                  f"| Value (₹) | {res['nav']:,.0f} | {res['ew_nav']:,.0f} | {res['nifty_nav']:,.0f} |",
                  f"| Return since the 1 Oct 2026 open | {res['ret_pct']:+.2f}% | {res['ew_ret_pct']:+.2f}% | {res['nifty_ret_pct']:+.2f}% |", "",
                  f"Excess vs primary benchmark **{res['excess_ew_pct']:+.2f}%** (log) · vs NIFTYBEES {res['excess_nifty_pct']:+.2f}% · "
                  f"modelled costs ₹{res['cum_cost_inr']:,.0f} · {res['sessions']} sessions · {res['complete_months']} complete months · "
                  f"z vs history {res['z_vs_history']:+.2f} (REJECT below −2, only at review)", ""]
    lines += [f"**Verdict: {res['verdict']}**", "", "## Decisions", "",
              "| Month | Decision close | Timing | Universe | Selected | Risk Gate (shadow) |", "|---|---|---|---|---|---|"]
    for d in decs:
        g = gates.get(d["date"], {})
        lines.append(f"| {d['month']} | {d['date']} | {d['timing']} (computed {d['computed_on_utc']}) | {len(d['universe'])} | "
                     f"{', '.join(d['selected'])} | {g.get('summary', '—')} |")
    for r in L.of_kind("missed_decision"):
        lines.append(f"| {r['month']} | {r['date']} | MISSED | — | previous holdings kept | — |")
    other = [r for r in L.records() if r["kind"] in ("revision", "integrity", "pipeline_change", "runner_version")]
    if other:
        lines += ["", "## Audit events", ""] + [f"- {r['ts_utc'][:10]} {r['kind']}: {r.get('status') or ', '.join(r.get('changed', [])) or ''}" for r in other]
    lines += ["", "A forward test this short cannot prove REV1: at its historical information ratio (~0.26), forward data would need "
              "about 59 years to reach t = 2. It can catch REV1 breaking, and it checks the data, decision and cost pipeline in real time.", ""]
    return "\n".join(lines) + "\n"


def init(spec_path: Path = HERE / "spec.json", ledger_path: Path = HERE / "ledger.jsonl") -> None:
    L = Ledger(ledger_path)
    if L.records() or spec_path.exists():
        raise SystemExit("already initialised; the spec is frozen")
    spec = make_spec()
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8")
    L.append("genesis", {"spec_sha256": _sha_file(spec_path), "preregistration_sha256": spec["preregistration_sha256"],
                         "note": "REV1 forward paper test starts; first decision at the 30 Sep 2026 close, first fill at the 1 Oct 2026 open"})
