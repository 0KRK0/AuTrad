"""v1.2 event data (research/preregistration_v1.2.md):
* dividend announcements with exchange timestamps (NSE corporate announcements API), per F&O symbol-year;
* NSE F&O ban lists (fo_secban) and market-wide position limit / open interest (combineoi);
* NSE Indices press releases on index changes (parsed in marketdata/index_changes.py).
Everything is stored under data/events/. Network functions are resumable and never record a transient failure as
"no data".
"""
from __future__ import annotations

import io
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path("data/events")
ANN = ROOT / "div_ann"
ANN_URL = "https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={sym}&from_date={a}&to_date={b}"


# ------------------------------------------------------------------------------------------------ dividend announcements
def symbol_years(y0: int = 2015, last: str = "2026-09-30") -> list[tuple[str, int]]:
    from marketdata.nse_fo import load_fo
    f = load_fo()
    f = f[(f["kind"] == "STK") & (f["date"] <= last)]
    sy = f.assign(y=f["date"].str[:4].astype(int)).groupby(["symbol", "y"]).size().index.tolist()
    return sorted((s, y) for s, y in sy if y >= y0)


def _keep(a: dict) -> bool:
    return bool(re.search(r"divid", (a.get("desc") or "") + " " + (a.get("attchmntText") or ""), re.I))


def fetch_dividend_announcements(shard: int = 0, nshard: int = 1, budget_s: float = 150, workers: int = 3) -> dict:  # pragma: no cover
    from marketdata.nse_filings import Client
    ANN.mkdir(parents=True, exist_ok=True)
    todo = [(s, y) for k, (s, y) in enumerate(symbol_years()) if k % nshard == shard and not (ANN / f"{s}_{y}.json").exists()]
    t0 = time.time()
    clients = [Client(pause=0.25) for _ in range(workers)]
    stat = {"done": 0, "failed": 0, "left": len(todo)}

    def one(i_item):
        i, (s, y) = i_item
        if time.time() - t0 > budget_s:
            return None
        c = clients[i % workers]
        b = min(pd.Timestamp(f"{y}-12-31"), pd.Timestamp("2026-09-30"))
        raw = c.get(ANN_URL.format(sym=s.replace("&", "%26"), a=f"01-01-{y}", b=b.strftime("%d-%m-%Y")))
        if raw is None:
            return False
        try:
            js = json.loads(raw)
        except ValueError:
            return False
        rows = [{k: a.get(k) for k in ("symbol", "desc", "an_dt", "exchdisstime", "sort_date", "seq_id", "sm_isin")}
                | {"text": (a.get("attchmntText") or "")[:800]} for a in (js if isinstance(js, list) else []) if _keep(a)]
        (ANN / f"{s}_{y}.json").write_text(json.dumps({"symbol": s, "year": y, "n_all": len(js) if isinstance(js, list) else None, "rows": rows}))
        return True

    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, enumerate(todo)):
            if r is True:
                stat["done"] += 1
            elif r is False:
                stat["failed"] += 1
    stat["left"] -= stat["done"]
    stat["seconds"] = round(time.time() - t0, 1)
    return stat


AMT = re.compile(r"(?:Rs\.?|INR|₹|Re\.?)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/-)?\s*(?:\([^)]{0,90}\)\s*)?(?:/-)?\s*(?:\(?\s*[0-9.]+\s*%\s*\)?\s*)?"
                 r"per\s+(?:fully\s*paid[- ]?up\s+)?(?:ordinary\s+)?(?:equity\s+)?share", re.I)


def parse_dividend(text: str) -> dict:
    t = " ".join((text or "").split())
    typ = "special" if re.search(r"special", t, re.I) else "interim" if re.search(r"interim", t, re.I) else "final" if re.search(r"final", t, re.I) else None
    act = ("declared" if re.search(r"declar", t, re.I) else "recommended" if re.search(r"recommend", t, re.I)
           else "approved" if re.search(r"approv", t, re.I) else None)
    amts = [float(x) for x in AMT.findall(t)]
    neg = bool(re.search(r"(not|did not|no)\s+(recommend|declare)", t, re.I))
    pay = re.search(r"paid[^.]{0,60}?on or (?:before|after)\s+([A-Z][a-z]+ \d{1,2},? \d{4}|\d{1,2}[-/][A-Za-z]{3}[-/]\d{4})", t)
    return {"type": typ, "action": act, "amounts": amts, "negative": neg, "payment_text": pay.group(1) if pay else None}


def announcement_time(r: dict) -> pd.Timestamp | None:
    from marketdata.nse_filings import _parse_ts
    return _parse_ts(r.get("exchdisstime")) or _parse_ts(r.get("an_dt"))


def load_announcements() -> pd.DataFrame:
    rows = []
    for p in sorted(ANN.glob("*.json")):
        js = json.loads(p.read_text())
        for r in js["rows"]:
            ts = announcement_time(r)
            pr = parse_dividend(r.get("text", ""))
            rows.append({"symbol": js["symbol"], "ts": ts, "desc": r.get("desc"), "text": r.get("text"), "seq_id": r.get("seq_id"), **pr})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ ban / MWPL
BAN_URL = "https://nsearchives.nseindia.com/archives/fo/sec_ban/fo_secban_{d}.csv"
OI_URL = "https://nsearchives.nseindia.com/archives/nsccl/mwpl/combineoi_{d}.zip"


def parse_ban(raw: bytes, d: date) -> list[str]:
    lines = raw.decode(errors="ignore").splitlines()
    if not lines or "NIL" in lines[0].upper():
        return []
    return [ln.split(",", 1)[1].strip() for ln in lines[1:] if "," in ln and ln.split(",", 1)[1].strip()]


def parse_oi(raw: bytes, d: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(io.BytesIO(z.read(name)))
    df.columns = [c.strip() for c in df.columns]
    lim = df["Limit for Next Day"].astype(str).str.strip()
    return pd.DataFrame({"date": d.isoformat(), "symbol": df["NSE Symbol"].astype(str).str.strip(), "mwpl": pd.to_numeric(df["MWPL"], errors="coerce"),
                         "oi": pd.to_numeric(df["Open Interest"], errors="coerce"),
                         "fut_eq_oi": pd.to_numeric(df["Future Equivalent Open Interest"], errors="coerce") if "Future Equivalent Open Interest" in df else pd.NA,
                         "no_fresh": lim.str.contains("No Fresh", case=False)})


def fetch_ban_oi(days: list[date], budget_s: float = 150, workers: int = 10) -> dict:  # pragma: no cover - network
    import requests
    from marketdata.nse_bhav import UA
    ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    s = requests.Session()
    man_p = ROOT / "ban_manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"done": [], "no_ban_file": [], "no_oi_file": []}
    done = set(man["done"])
    todo = [d for d in days if d.isoformat() not in done]
    ban_rows, oi_frames = [], []

    def get(url):
        for k in range(3):
            try:
                r = s.get(url, headers=UA, timeout=25)
                if r.status_code in (200, 404):
                    return r.status_code, r.content
            except Exception:
                pass
            time.sleep(1 + k)
        return None, None

    def one(d):
        if time.time() - t0 > budget_s:
            return d, None
        k = d.strftime("%d%m%Y")
        bs, bc = get(BAN_URL.format(d=k))
        os_, oc = get(OI_URL.format(d=k))
        if bs is None or os_ is None:
            return d, None
        return d, (bs, bc, os_, oc)

    with ThreadPoolExecutor(workers) as ex:
        for d, r in ex.map(one, todo):
            if r is None:
                continue
            bs, bc, os_, oc = r
            if os_ == 200 and oc[:2] == b"PK":
                try:
                    parsed = parse_oi(oc, d)
                except Exception:                        # corrupt download: retry next run, do not mark done
                    continue
            if bs == 200:
                ban_rows += [{"date": d.isoformat(), "symbol": x} for x in parse_ban(bc, d)]
            else:
                man["no_ban_file"].append(d.isoformat())
            if os_ == 200 and oc[:2] == b"PK":
                oi_frames.append(parsed)
            else:
                man["no_oi_file"].append(d.isoformat())
            done.add(d.isoformat())
    for name, new in (("fo_ban.csv", pd.DataFrame(ban_rows, columns=["date", "symbol"])), ("mwpl_oi.csv.gz", pd.concat(oi_frames) if oi_frames else None)):
        if new is None:
            continue
        p = ROOT / name
        old = [pd.read_csv(p)] if p.exists() else []
        pd.concat(old + [new], ignore_index=True).drop_duplicates().sort_values(["date", "symbol"]).to_csv(p, index=False)
    man["done"] = sorted(done)
    man_p.write_text(json.dumps(man))
    return {"fetched": len(done) - (len(set(man["done"])) - len(todo)) if False else None, "done_total": len(done), "todo_left": len([d for d in days if d.isoformat() not in done]),
            "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["div", "ban"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--budget", type=float, default=150)
    ap.add_argument("--years", default="2016-2026")
    a = ap.parse_args()
    if a.cmd == "div":
        k, n = (int(x) for x in a.shard.split("/"))
        print(fetch_dividend_announcements(k, n, a.budget))
    else:
        from marketdata.nse_fo import sessions
        y0, y1 = (int(x) for x in a.years.split("-"))
        print(fetch_ban_oi(sessions(y0, y1), a.budget))
