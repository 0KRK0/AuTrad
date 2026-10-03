"""Corporate-action handling, anchored on exchange prices (v2, after the Oct-2026 data-integrity audit).

The vendor (Yahoo) series is adjusted price = exchange price x factor. The audit against BSE's official
unadjusted bhavcopy found that the vendor's factor is usually right but has four failure modes:
  1. right factor, wrong window: the adjustment is applied only back to some cutoff (TRENT bonus applied
     only back to 1 Jan 2026; ADANIENT/UPL/TATACONSUM rights only back about one week), creating fake moves;
  2. wrong factor for demergers (ADANIENT 2015 and 2018, CESC 2018);
  3. no adjustment at all (Tata Motors demerger Oct 2025, Vedanta demerger Apr 2026);
  4. short misprints at the wrong scale (NIFTYBEES/BANKBEES/GOLDBEES around the Dec-2019 unit splits).

So the vendor's factor is no longer trusted. For every NSE/BSE-listed asset:
  * F_t = log(vendor close / BSE close) is fitted as a step function (blips masked).
    vendor x exp(-F_t) is then the NSE price at the exchange's unadjusted scale (NSE prices are kept).
  * Every step of F is classified using exchange evidence on that day (see classify()):
      - verified registry event (reference/corporate_actions.csv, each with a public source) -> its factor;
      - split/bonus whose size matches a simple ratio on an exchange event day -> the exact ratio;
      - dividend or rights step that the exchange's own move explains -> the vendor's factor;
      - big event where the vendor's factor disagrees with the exchange's price discovery -> the exchange's;
      - step on a day the exchange price did nothing unusual -> vendor artifact, removed.
  * A_t = product of accepted factors after t;  corrected = vendor x exp(-F_t) x A_t.
  * Days where the vendor print disagrees with the exchange by > 3% (misprints) are replaced by the BSE
    bar at the same adjusted scale and flagged.
Raw vendor files are never modified. Assets with no exchange cross-check (FX, crypto, indices) only get
the power-of-ten blip repair and are marked unverified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

TRANSFORMS_VERSION = "ca_2026_10_03_exchange_anchored"
REGISTRY = Path(__file__).parent / "reference" / "corporate_actions.csv"

EVENT_MOVE = 0.12          # |raw exchange move net of market| (close or open gap) that marks an exchange event day
MISPRINT = 0.03            # vendor print vs exchange disagreement that triggers replacement
# outcomes of common splits (1:2 ... 1:100) and bonuses (1:1, 1:2, 1:3, 1:4, 1:5, 1:10, 2:1, 3:1)
STD_RATIOS = sorted({float(f) for f in [Fraction(1, 2), Fraction(1, 3), Fraction(1, 4), Fraction(1, 5), Fraction(1, 10),
                                        Fraction(1, 20), Fraction(1, 50), Fraction(1, 100), Fraction(2, 3), Fraction(3, 4),
                                        Fraction(4, 5), Fraction(5, 6), Fraction(10, 11)]})


# ----------------------------------------------------------------- legacy fallback (no exchange cache)
def repair_blips(df: pd.DataFrame, max_len: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Power-of-ten misprints of 1-5 bars that snap back (vendor-only repair; no exchange needed)."""
    df = df.copy()
    c = df["close"].to_numpy()
    notes, i, n = [], 1, len(df)
    while i < n:
        ratio = c[i] / c[i - 1] if c[i - 1] > 0 else 1.0
        if ratio < 0.2 or ratio > 5.0:
            k = 10 ** round(-np.log10(ratio))
            for j in range(i + 1, min(i + max_len + 1, n)):
                if 0.7 < c[j] / c[i - 1] < 1.3:
                    cols = ["open", "high", "low", "close"]
                    df.iloc[i:j, df.columns.get_indexer(cols)] *= k
                    df.iloc[i:j, df.columns.get_loc("volume")] /= k
                    notes.append(f"repaired {j - i} bar(s) from {df.index[i].date()} (scale x{k:g})")
                    c = df["close"].to_numpy()
                    i = j
                    break
        i += 1
    return df, notes


# ----------------------------------------------------------------- registry
@dataclass
class Event:
    symbol: str
    date: pd.Timestamp
    factor: float               # multiply pre-event prices by this (k < 1 for splits, bonuses, demergers, dividends)
    kind: str                   # split | bonus | rights | demerger | dividend | ratio_event | vendor_event
    method: str                 # ratio | terp | discovery | close_mkt | vendor | registry
    evidence: str = ""
    vendor_factor: float | None = None
    source: str = ""

    def row(self) -> dict:
        d = asdict(self)
        d["date"] = str(self.date.date())
        return d


def load_registry(path: Path = REGISTRY) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "ex_date", "kind", "terms", "method", "source", "note"])
    r = pd.read_csv(path)
    r["ex_date"] = pd.to_datetime(r["ex_date"])
    return r


def _ratio_factor(kind: str, terms: str) -> float:
    a, b = (float(x) for x in terms.split(":"))
    if kind == "split":            # "1:10" = one unit becomes ten
        return a / b
    if kind == "bonus":            # "1:2" = one bonus share for every two held
        return b / (a + b)
    raise ValueError(kind)


def _terp_factor(terms: str, cum_price: float) -> float:
    ratio, price = terms.split("@")
    m, n = (float(x) for x in ratio.split(":"))     # m new for every n held
    terp = (n * cum_price + m * float(price)) / (n + m)
    return terp / cum_price


# ----------------------------------------------------------------- step fitting
def _fit_steps(lr: np.ndarray, w: int = 5, thr: float = 0.004) -> list[int]:
    n = len(lr)
    if n < 2 * w + 1:
        return []
    d = np.zeros(n)
    for t in range(w, n - w + 1):
        d[t] = np.median(lr[t:t + w]) - np.median(lr[t - w:t])
    out, t = [], w
    while t < n - w + 1:
        if abs(d[t]) > thr:
            j = t
            while j + 1 < n - w + 1 and abs(d[j + 1]) > thr and np.sign(d[j + 1]) == np.sign(d[t]):
                j += 1
            k = t + int(np.argmax(np.abs(d[t:j + 1])))
            lo, hi = max(k - w + 1, 1), min(k + w, n)
            out.append(lo + int(np.argmax(np.abs(np.diff(lr[lo - 1:hi])))))
            t = j + 1
        else:
            t += 1
    return sorted(set(out))


def _segments(lr: np.ndarray, cps: list[int]) -> np.ndarray:
    F = np.empty_like(lr)
    edges = [0, *cps, len(lr)]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        F[a:b] = np.median(lr[a:b])
    return F


def _nearest_std(k: float, tol: float = 0.03) -> float | None:
    best = min(STD_RATIOS, key=lambda r: abs(np.log(k) - np.log(r)))
    return best if abs(np.log(k) - np.log(best)) < tol else None


# ----------------------------------------------------------------- main
@dataclass
class Correction:
    df: pd.DataFrame
    events: list[Event] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)       # vendor artifacts dropped
    replaced_days: list[str] = field(default_factory=list)  # vendor misprints replaced by exchange bars
    quality: pd.Series | None = None                        # per-date data-quality score 0..1
    notes: list[str] = field(default_factory=list)
    opens_replaced: list[str] = field(default_factory=list)
    suspect: list[str] = field(default_factory=list)        # possible events hidden in vendor windows (flagged)


def correct_with_exchange(symbol: str, vendor: pd.DataFrame, bse: pd.DataFrame, bench: pd.Series | None,
                          registry: pd.DataFrame | None = None) -> Correction:
    """Rebuild the vendor series on exchange evidence. Output columns as the vendor's. Note on volume: it is
    rescaled so that close x volume = the rupee value actually traded that day (what the engine uses for impact
    and participation); it is therefore not a share count before demergers."""
    reg = load_registry() if registry is None else registry
    reg = reg[reg["symbol"] == symbol]
    bse = bse[bse["close"] > 0].astype(float)
    v = vendor.copy()
    notes: list[str] = []
    # 0) phantom/stale vendor bars: zero volume and a flat bar. On an exchange holiday -> drop the row;
    #    on a real session -> replaced by the exchange bar further down.
    flat = (v["volume"] <= 0) & (v["open"] == v["close"]) & (v["high"] == v["low"]) & (v["high"] == v["close"])
    phantom = flat & ~v.index.isin(bse.index) & (v.index >= bse.index.min()) & (v.index <= bse.index.max())
    if phantom.any():
        notes.append(f"dropped {int(phantom.sum())} phantom vendor bar(s) on exchange holidays")
        v = v[~phantom]
        flat = flat[~phantom]
    stale = pd.Series(flat.to_numpy() & v.index.isin(bse.index), index=v.index)

    idx = v.index.intersection(bse.index)
    b = bse.reindex(idx)
    vc = v["close"].reindex(idx).astype(float)
    ok = (vc > 0) & ~stale.reindex(idx).to_numpy()
    idx, b, vc = idx[ok], b[ok], vc[ok]
    lr = np.log(vc / b["close"]).to_numpy()
    mkt = np.log(bench.reindex(idx).ffill()).diff().fillna(0).to_numpy() if bench is not None else np.zeros(len(idx))
    braw = np.r_[0.0, np.diff(np.log(b["close"].to_numpy()))]
    bopen_gap = np.r_[0.0, np.log(b["open"].to_numpy()[1:] / b["close"].to_numpy()[:-1])]
    vret = np.r_[0.0, np.diff(np.log(vc.to_numpy()))]

    # 1) mask misprints (vendor disagrees with the exchange for a few sessions, then agrees again)
    med = pd.Series(lr).rolling(11, center=True, min_periods=3).median().to_numpy()
    mis = np.abs(lr - med) > MISPRINT
    lr_m = lr.copy()
    lr_m[mis] = med[mis]
    # 2) vendor factor as a step function
    cps = _fit_steps(lr_m)
    F = _segments(lr_m, cps)
    still = np.abs(lr - F) > MISPRINT                      # prints that disagree with the fitted factor
    sizes = {i: float(F[i] - F[i - 1]) for i in cps}
    # 2b) window artifacts: a step undone by an opposite step within ~20 sessions means the vendor scaled only a
    #     window of history. Neither step is a corporate action the vendor applied to the whole history.
    paired: dict[int, int] = {}
    for x, i in enumerate(cps):
        for j in cps[x + 1:]:
            if (idx[j] - idx[i]).days > 28:
                break
            si, sj = sizes[i], sizes[j]
            if np.sign(si) != np.sign(sj) and abs(si + sj) < 0.3 * max(abs(si), abs(sj)) and max(abs(si), abs(sj)) > 0.004:
                paired[i], paired[j] = j, i
                break
    # 3) classify every step
    events: list[Event] = []
    removed: list[dict] = []
    suspect: list[pd.Timestamp] = []
    for i in cps:
        d, size = idx[i], sizes[i]
        k_vendor = float(np.exp(-size))
        br, gap = braw[i] - mkt[i], bopen_gap[i] - mkt[i]
        vr = vret[i] - mkt[i]
        if len(reg[(reg["ex_date"] - d).abs() <= pd.Timedelta(days=5)]):
            continue                                       # registry events are added below (authoritative)
        art = {"symbol": symbol, "date": str(d.date()), "vendor_step": round(size, 4),
               "exchange_move": round(float(br), 4), "vendor_move": round(float(vr), 4)}
        if abs(size) < 0.004:
            continue                                       # venue noise
        if i in paired:
            removed.append({**art, "reason": f"vendor window: undone by the step on {idx[paired[i]].date()}"})
            if size > 0.01 and abs(br + size) < max(0.02, 0.25 * size):
                suspect.append(d)                          # a real event may hide in the window: flag, do not guess
            continue
        if size < 0:
            removed.append({**art, "reason": "negative step: no corporate action raises pre-event prices"})
            continue
        explained = abs(br + size) < max(0.045, 0.25 * size)   # exchange fell by about the step that day
        std = _nearest_std(k_vendor, 0.02) if size > 0.25 else None
        if std is not None and (explained or ex_event_day(br, gap)):
            events.append(Event(symbol, d, std, "ratio_event", "ratio",
                                f"vendor x{k_vendor:.4f} ~ standard ratio {std:.4f} (exact ratio used); BSE raw x{np.exp(braw[i]):.4f}",
                                k_vendor))
        elif size < 0.05 and abs(vr) < EVENT_MOVE:
            events.append(Event(symbol, d, k_vendor, "dividend", "vendor",
                                f"persistent vendor step {size:+.4f}; exchange {br:+.3f} net of market", k_vendor))
        elif size < 0.10 and abs(vr) < 0.08:
            events.append(Event(symbol, d, k_vendor, "vendor_event", "vendor",
                                f"large dividend/rights: exchange {br:+.3f} net of market vs vendor step {size:+.3f}"
                                + ("" if explained else " (exchange move not fully explained; crash day?)"), k_vendor))
        elif size >= 0.10 and (explained or ex_event_day(br, gap)):
            k_disc = _discovery(b, i, mkt[i])
            if abs(np.log(k_disc) - np.log(k_vendor)) < 0.03:      # vendor factor and exchange discovery agree
                events.append(Event(symbol, d, k_vendor, "vendor_event", "vendor",
                                    f"vendor x{k_vendor:.4f} agrees with exchange discovery x{k_disc:.4f}", k_vendor))
            else:
                events.append(Event(symbol, d, k_disc, "vendor_event", "discovery",
                                    f"UNVERIFIED: vendor x{k_vendor:.4f} disagrees with exchange discovery x{k_disc:.4f}", k_vendor))
        else:
            removed.append({**art, "reason": "step on a day the exchange did not move accordingly"})
    # registry events (verified with a public source); factors come from the full exchange series
    for _, r in reg.iterrows():
        d = r["ex_date"]
        if d not in bse.index or bse.index.get_loc(d) == 0:
            notes.append(f"WARNING registry {r['kind']} {d.date()} not in exchange data; NOT applied")
            suspect.append(d)
            continue
        j = bse.index.get_loc(d)
        prev = float(bse["close"].iloc[j - 1])
        m = float(np.log(bench.reindex([bse.index[j - 1], d]).ffill()).diff().iloc[-1]) if bench is not None else 0.0
        if r["method"] == "ratio":
            k = _ratio_factor(r["kind"], r["terms"])
        elif r["method"] == "terp":
            k = _terp_factor(r["terms"], prev)
        elif r["method"] == "discovery":
            k = _discovery(bse, j, m)
        elif r["method"] == "dividend":
            k = 1.0 - float(r["terms"]) / prev
        else:
            k = float(bse["close"].iloc[j] / prev / np.exp(m))
        near = [c for c in cps if abs((idx[c] - d).days) <= 5 and c not in paired]
        kv = float(np.exp(-sizes[near[0]])) if near else 1.0
        events.append(Event(symbol, d, k, r["kind"], r["method"], str(r.get("note", "")), kv, str(r.get("source", ""))))
    events.sort(key=lambda e: e.date)

    # 4) rebuild: corrected = vendor x exp(-F) x A on the vendor calendar
    Fs = pd.Series(F, idx).reindex(v.index.union(idx)).ffill().bfill().reindex(v.index)
    A = pd.Series(1.0, v.index)
    for e in events:
        A[A.index < e.date] *= e.factor
    mult = np.exp(-Fs) * A
    out = v.copy()
    for c in ["open", "high", "low", "close"]:
        out[c] = v[c] * mult
    traded_value = v["close"] * v["volume"] / np.exp(Fs)      # vendor close / exp(F) = unadjusted price
    # 5) misprints and stale bars -> the exchange bar at the same adjusted scale
    stale_days = v.index[stale.reindex(v.index, fill_value=False).to_numpy()]
    bad = idx[still].union(stale_days)
    vol_ratio = float(np.nanmedian((v["volume"].reindex(idx) / bse["volume"].reindex(idx)).replace([np.inf, 0], np.nan))) \
        if len(idx) else 1.0
    for d in bad:
        for c in ["open", "high", "low", "close"]:
            out.loc[d, c] = float(bse.loc[d, c]) * A[d]
        traded_value[d] = float(bse.loc[d, "close"] * bse.loc[d, "volume"]) * (vol_ratio if np.isfinite(vol_ratio) else 1.0)
    # 6) opens: the engine trades at the open. When the NSE (vendor) and BSE opens disagree by > 3%, one venue
    #    printed a stale/wrong open; keep the one closer to that day's close (both venues have such prints).
    common = out.index.intersection(bse.index)
    ex_open = bse["open"].reindex(common) * A.reindex(common)
    vo, cl = out["open"].reindex(common), out["close"].reindex(common)
    disagree = np.log(vo / ex_open).abs() > MISPRINT
    bse_better = np.log(ex_open / cl).abs() < np.log(vo / cl).abs()
    bad_open = common[(disagree & bse_better).to_numpy()]
    if len(bad_open):
        out.loc[bad_open, "open"] = ex_open.reindex(bad_open)
        notes.append(f"{len(bad_open)} vendor open(s) replaced by the exchange open")
    out["high"] = out[["high", "open", "close"]].max(axis=1)
    out["low"] = out[["low", "open", "close"]].min(axis=1)
    out["volume"] = (traded_value / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    q = pd.Series(1.0, out.index)
    q[~out.index.isin(bse.index)] = 0.8                   # no same-day exchange print to confirm
    q[q.index.isin(bad) | q.index.isin(bad_open)] = 0.5   # vendor print replaced by exchange
    for e in [*[x.date for x in events if "UNVERIFIED" in x.evidence], *suspect]:
        q[(q.index >= e - pd.Timedelta(days=7)) & (q.index <= e + pd.Timedelta(days=7))] = 0.3
    notes = [f"{e.kind} {e.date.date()}: x{e.factor:.4f} ({e.method})" for e in events if e.kind != "dividend"] + notes
    notes += [f"removed vendor artifact {r['date']} (step {r['vendor_step']:+.3f}: {r['reason']})" for r in removed]
    if suspect:
        notes.append("SUSPECT (needs a source): " + ", ".join(str(pd.Timestamp(x).date()) for x in suspect))
    if len(bad):
        notes.append(f"{len(bad)} vendor print(s) replaced by exchange bars")
    c = Correction(out, events, removed, [str(d.date()) for d in bad], q, notes)
    c.opens_replaced = [str(d.date()) for d in bad_open]
    c.suspect = [str(pd.Timestamp(x).date()) for x in suspect]
    return c


def ex_event_day(br: float, gap: float) -> bool:
    return abs(br) > EVENT_MOVE or abs(gap) > EVENT_MOVE


def _discovery(b: pd.DataFrame, i: int, mkt: float) -> float:
    """Exchange price discovery on the ex-date: the opening (special pre-open) price if it is a valid print,
    otherwise the close net of the market's move that day."""
    prev = float(b["close"].iloc[i - 1])
    o, h, lo, c = (float(b[k].iloc[i]) for k in ["open", "high", "low", "close"])
    if lo <= o <= h and abs(np.log(o / c)) < 0.15:
        return o / prev
    return c / prev / np.exp(mkt)


# legacy entry point kept for assets with no exchange cross-check
def apply(symbol: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df, notes = repair_blips(df)
    return df, [*notes, "unverified: no exchange cross-check for this asset"] if notes else ["unverified: no exchange cross-check"]
