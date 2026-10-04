"""BSE shareholding-pattern history (SEBI LODR Reg 31): promoter holding, promoter pledges / encumbrances and the
institutional mix, one record per exchange filing, stamped with BSE's own filing (or revision) timestamp.

Point-in-time rule: a filing is usable from its BSE filing timestamp (IST). When BSE lists only a *revised* filing for a
quarter (no original timestamp) the revision timestamp is used, because the content we hold is the revised content.
Quarters before March 2016 (qtrid < 89) carry no timestamp on BSE and are not fetched.

Version check: BSE's structured API serves the *latest* version of a filing's data (its Fld_AuthoriseDate can be weeks
after the listed filing), and several filings can share one qtrid. A structured record is used for a filing only when
its authorisation stamp is within 5 minutes of that filing's timestamp; otherwise the filing's own as-filed XBRL is
parsed, and if that is unavailable the structured record is used from its authorisation time (never earlier).

Raw API responses are stored unmodified (data/pit/bse/shp/<code>/...) with a sha256 manifest; `load()` parses them.
Reachable from a home/office connection only (api.bseindia.com).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

API = "https://api.bseindia.com/BseIndiaAPI/api"
INDEX = API + "/SHPQNewFormat/w?scripcode={code}"
PROMOTER = API + "/Corp_shpPromoterNGroup_ng/w?SCRIPCODE={code}&QtrCode={q:.2f}"
PUBLIC = API + "/Corp_shpSec_SHPPubShold_ng/w?SCRIPCODE={code}&QtrCode={q:.2f}"
FIRST_QTRID = 89.0                      # March 2016: first quarter with a BSE filing timestamp
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


class Client:  # pragma: no cover - network
    def __init__(self, pause: float = 0.3) -> None:
        import requests

        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                               "Referer": "https://www.bseindia.com/", "Origin": "https://www.bseindia.com"})
        self.pause = pause

    def get_json(self, url: str, tries: int = 4):
        for i in range(tries):
            try:
                r = self.s.get(url, timeout=40)
                if r.status_code == 200 and r.content:
                    time.sleep(self.pause)
                    return r.json()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.0 * (i + 1))
        return None


def qtr_key(q: float) -> str:
    return f"{q:.2f}"


def fetch_index(root: Path, code: str, client, force: bool = False) -> list[dict] | None:  # pragma: no cover - network
    d = Path(root) / "shp" / code
    d.mkdir(parents=True, exist_ok=True)
    p = d / "index.json"
    if p.exists() and not force:
        return json.loads(p.read_text())["rows"]
    j = client.get_json(INDEX.format(code=code))
    if j is None:
        return None
    rows = j.get("Table", [])
    p.write_text(json.dumps({"code": code, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}))
    return rows


def todo(root: Path, code: str, rows: list[dict], parts: tuple[str, ...] = ("promoter", "public")) -> list[tuple[str, float]]:
    out = []
    for r in rows:
        q = float(r["qtrid"])
        if q < FIRST_QTRID or not (r.get("filing_date_time") or r.get("revised_date_time")):
            continue
        for part in parts:
            if not (Path(root) / "shp" / code / f"{qtr_key(q)}_{part}.json").exists():
                out.append((part, q))
    return out


def fetch_part(root: Path, code: str, part: str, q: float, client) -> bool:  # pragma: no cover - network
    url = (PROMOTER if part == "promoter" else PUBLIC).format(code=code, q=q)
    j = client.get_json(url)
    if j is None:
        return False
    (Path(root) / "shp" / code / f"{qtr_key(q)}_{part}.json").write_text(json.dumps(j))
    return True


XBRL_URL = "https://www.bseindia.com/XBRLFILES/SHPXBRLDataXML/{f}"
MATCH_TOLERANCE = pd.Timedelta(minutes=5)


def needs_xbrl(root: Path, code: str, rows: list[dict]) -> list[dict]:
    """Index rows whose structured record is not this filing's own version (see module docstring)."""
    out = []
    for r in rows:
        q = float(r["qtrid"])
        ts = r.get("filing_date_time") or r.get("revised_date_time")
        if q < FIRST_QTRID or not ts or not r.get("XbrlFile"):
            continue
        p = Path(root) / "shp" / code / f"{qtr_key(q)}_promoter.json"
        auth = (json.loads(p.read_text()).get("Table") or [{}])[0].get("Fld_AuthoriseDate") if p.exists() else None
        if auth is None or abs(pd.Timestamp(auth) - pd.Timestamp(ts)) > MATCH_TOLERANCE:
            out.append(r)
        elif sum(1 for x in rows if float(x["qtrid"]) == q and (x.get("filing_date_time") or x.get("revised_date_time"))) > 1:
            out.append(r)                                     # qtrid shared by several filings: keep every as-filed copy
    return out


def fetch_xbrl(root: Path, code: str, xbrl_file: str, client) -> bool:  # pragma: no cover - network
    import gzip

    d = Path(root) / "shp" / code / "xbrl"
    d.mkdir(parents=True, exist_ok=True)
    p = d / (xbrl_file + ".gz")
    if p.exists():
        return True
    for i in range(3):
        try:
            r = client.s.get(XBRL_URL.format(f=xbrl_file), timeout=60)
            if r.status_code == 200 and b"<xbrli:xbrl" in r.content[:2000]:
                p.write_bytes(gzip.compress(r.content))
                time.sleep(client.pause)
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0 * (i + 1))
    return False


XBRL_BUCKETS = (("mf_pct", ("MutualFund",)), ("fpi_pct", ("ForeignPortfolioInvestor", "ForeignInstitutionalInvestor")),
                ("insurance_pct", ("InsuranceCompan",)), ("banks_pct", ("Banks", "FinancialInstitutions")),
                ("aif_pct", ("AlternativeInvestmentFund", "AlternateInvestmentFund")),
                ("retail_pct", ("ResidentIndividualShareholders", "IndividualShareholdersHolding", "IndividualsHolding")))


def xbrl_summary(text: str) -> dict:
    """As-filed shareholding-pattern XBRL (BSE taxonomies 2016-06-23, 2018-03-31, 2022-09-30, 2025-05-31)."""
    import re

    m = re.search(r'xmlns:([\w-]+)="http://www\.bseindia\.com/xbrl/shp/[^"]+/in-bse-shp"', text)
    if not m:
        return {}
    pre = re.escape(m.group(1))

    def facts(el: str) -> dict[str, float]:
        return {c: _f(v) for c, v in re.findall(rf'<{pre}:{el} contextRef="([^"]+)"[^>]*>([^<]*)<', text)}

    shares, pct = facts("NumberOfShares"), facts("ShareholdingAsAPercentageOfTotalNumberOfShares")
    enc = facts("PledgedOrEncumberedNumberOfShares")            # 2016-2022 taxonomies: pledge + other encumbrance
    enc_total = facts("NumberOfSharesEncumbered")               # 2025-05-31 taxonomy: total encumbered ...
    enc_pledge = facts("NumberOfSharesEncumberedUnderPledged")  # ... of which pledged
    flag = re.search(rf"<{pre}:WhetherAnySharesHeldByPromotersArePledgeOrOtherwiseEncumbered[^>]*>\s*(true|yes)\s*<", text, re.I)
    prom = next((c for c in shares if re.fullmatch(r"ShareholdingOfPromoterAndPromoterGroup(_Context)?I", c)), None)
    if prom is None:
        if any(re.fullmatch(r"ShareholdingPattern(_Context)?I", c) for c in shares):   # a complete filing with no promoter
            out = {"no_promoter": True, "promoter_holders": 0.0, "promoter_shares": 0.0, "promoter_pct": 0.0, "pledged_shares": 0.0,
                   "pledged_pct_of_promoter": float("nan"), "ndu_shares": float("nan"), "other_encumbered_shares": float("nan"),
                   "encumbered_shares": 0.0, "encumbered_pct_of_promoter": float("nan"), "flag_pledge": None, "authorised_at": None,
                   "pledge_includes_other_encumbrance": True}
            out.update({k: float("nan") for k, _ in XBRL_BUCKETS})
            return out
        return {}
    ps = shares.get(prom, float("nan"))
    if prom in enc_total or prom in enc:
        e = enc_total.get(prom, enc.get(prom))
        pl = enc_pledge.get(prom, e)
    elif flag:                                                  # declared encumbered but no figure we can read
        e = pl = float("nan")
    else:
        e = pl = 0.0
    total = next((v for c, v in pct.items() if re.fullmatch(r"ShareholdingPattern(_Context)?I", c)), float("nan"))
    scale = 100.0 if total == total and total <= 1.0 + 1e-6 else 1.0          # newer filings report fractions (1.0 = 100%)
    out = {"no_promoter": not (ps > 0), "promoter_holders": float("nan"), "promoter_shares": ps,
           "promoter_pct": scale * pct.get(prom, float("nan")), "pledged_shares": pl,
           "pledged_pct_of_promoter": 100.0 * pl / ps if ps > 0 else float("nan"), "ndu_shares": float("nan"),
           "other_encumbered_shares": float("nan"), "encumbered_shares": e,
           "encumbered_pct_of_promoter": 100.0 * e / ps if ps > 0 else float("nan"),
           "flag_pledge": bool(flag), "authorised_at": None, "pledge_includes_other_encumbrance": not enc_pledge}
    for k, _ in XBRL_BUCKETS:          # institutional mix is not taken from XBRL: category contexts differ across
        out[k] = float("nan")           # taxonomies and can collide with promoter sub-categories (e.g. a bank promoter)
    return out


def manifest(root: Path) -> dict:
    files = sorted((Path(root) / "shp").glob("*/*.json")) + sorted((Path(root) / "shp").glob("*/xbrl/*.gz"))
    m = {str(f.relative_to(root)).replace("\\", "/"): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (Path(root) / "shp" / "manifest.json").write_text(json.dumps(m, indent=0))
    return m


# ----------------------------------------------------------------------------------------------- parsing
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def period_end(label: str) -> tuple[pd.Timestamp | None, str]:
    """'June 2026' -> (2026-06-30, 'quarterly'); '23 Dec 2025' -> (2025-12-23, 'event')."""
    s = (label or "").strip()
    parts = s.split()
    if len(parts) == 2 and parts[0].lower() in MONTHS:
        return pd.Timestamp(int(parts[1]), MONTHS[parts[0].lower()], 1) + pd.offsets.MonthEnd(0), "quarterly"
    try:
        return pd.Timestamp(pd.to_datetime(s, format="%d %b %Y")), "event"
    except (ValueError, TypeError):
        return None, "unknown"


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fix_scale(x: float) -> tuple[float, bool]:
    """A few structured records carry percentages x100 (e.g. 5504 for 55.04). Rescale and flag; NaN if still invalid."""
    if x != x or 0.0 <= x <= 100.0 + 1e-6:
        return x, False
    if 0.0 <= x / 100.0 <= 100.0 + 1e-6:
        return x / 100.0, True
    return float("nan"), True


def promoter_summary(j: dict) -> dict:
    t = j.get("Table1") or []
    flags = (j.get("Table") or [{}])[0]
    tot = next((r for r in t if (r.get("Fld_Code") or "").upper() == "STA1A2"), None)
    if tot is None:                                          # fall back: sum the A1 and A2 subtotals
        subs = [r for r in t if (r.get("Fld_Code") or "").upper() in ("STA1", "STA2")]
        if not subs:
            if t and all(not _f(r.get("Fld_TotalNoOfShares") or 0) for r in t):   # filed table with no promoter rows
                return {"no_promoter": True, "promoter_holders": 0.0, "promoter_shares": 0.0, "promoter_pct": 0.0,
                        "pledged_shares": 0.0, "pledged_pct_of_promoter": float("nan"), "ndu_shares": 0.0,
                        "other_encumbered_shares": 0.0, "encumbered_shares": 0.0, "encumbered_pct_of_promoter": float("nan"),
                        "flag_pledge": flags.get("Fld_IsPledge"), "authorised_at": flags.get("Fld_AuthoriseDate")}
            return {}
        keys = ["Fld_TotalNoOfShares", "Fld_TotalPercentageOf_A_B_C2", "Fld_PledgeEncumberedNoOfShares", "Fld_NDUNoOfShares",
                "Fld_OtherencumbrancesNoOfShares", "Fld_TotalencumberedNoOfShares", "Fld_NoOfShareHolders"]
        tot = {k: sum((_f(r.get(k)) if r.get(k) is not None else 0.0) for r in subs) for k in keys}
    shares = _f(tot.get("Fld_TotalNoOfShares"))
    pledged = _f(tot.get("Fld_PledgeEncumberedNoOfShares") or 0)
    ndu = _f(tot.get("Fld_NDUNoOfShares") or 0)
    other = _f(tot.get("Fld_OtherencumbrancesNoOfShares") or 0)
    total_enc = tot.get("Fld_TotalencumberedNoOfShares")
    total_enc = _f(total_enc) if total_enc not in (None, 0) else pledged + ndu + other
    ppct, rescaled = _fix_scale(_f(tot.get("Fld_TotalPercentageOf_A_B_C2")))
    return {"no_promoter": False, "promoter_holders": _f(tot.get("Fld_NoOfShareHolders")), "promoter_shares": shares,
            "promoter_pct": ppct, "pct_rescaled": rescaled,
            "pledged_shares": pledged, "pledged_pct_of_promoter": 100.0 * pledged / shares if shares and shares > 0 else float("nan"),
            "ndu_shares": ndu, "other_encumbered_shares": other, "encumbered_shares": total_enc,
            "encumbered_pct_of_promoter": 100.0 * total_enc / shares if shares and shares > 0 else float("nan"),
            "flag_pledge": flags.get("Fld_IsPledge"), "authorised_at": flags.get("Fld_AuthoriseDate")}


BUCKETS = (("mf_pct", ("mutual fund",)), ("fpi_pct", ("foreign portfolio", "foreign institutional")),
           ("insurance_pct", ("insurance",)), ("banks_pct", ("financial institutions", "banks")),
           ("aif_pct", ("alternate investment",)), ("retail_pct", ("individual",)))


def public_summary(j: dict) -> dict:
    """Category-level percentages of total shares. Only category rows (no named holder) are used, except the
    'Any Other (specify)' rows where BSE puts the category label in the holder-name field (e.g. 'Foreign Institutional
    Investor'), which are counted only when their label matches a bucket."""
    t = j.get("Table1") or []
    out = {k: 0.0 for k, _ in BUCKETS}
    seen = {k: False for k, _ in BUCKETS}
    for r in t:
        lvl = (r.get("Fld_Level") or "").lower()
        name = r.get("Fld_ShareHolderName")
        pct = _f(r.get("Fld_TotalPercentageOf_A_B_C2"))
        if pct != pct or lvl.startswith("sub total") or "=" in lvl:
            continue
        if name in (None, ""):
            label = lvl
        elif "any other" in lvl and (r.get("Fld_NoOfShareHolders") or 0) >= 1 and any(
                w in name.lower() for _, ws in BUCKETS for w in ws):
            label = name.lower()
        else:
            continue
        for k, ws in BUCKETS:
            if any(w in label for w in ws):
                if k == "retail_pct" and "any other" in lvl:
                    break
                out[k] += pct
                seen[k] = True
                break
    inst = [r for r in t if (r.get("Fld_Level") or "").lower().startswith("sub total b1")]
    res = {k: (v if seen[k] else 0.0) for k, v in out.items()}
    res["public_sub_total_b1_pct"] = _f(inst[0].get("Fld_TotalPercentageOf_A_B_C2")) if inst else float("nan")
    grand = [_f(r.get("Fld_TotalPercentageOf_A_B_C2")) for r in t if "b=b1" in (r.get("Fld_Level") or "").lower()]
    if any(v == v and v > 100.0 + 1e-6 for v in list(res.values()) + grand):   # x100 record: rescale all
        res = {k: (v / 100.0 if v == v else v) for k, v in res.items()}
        res["public_pct_rescaled"] = True
    return res


def load(root: Path, codes: dict[str, str]) -> pd.DataFrame:
    """One row per BSE filing. `codes` maps BSE scrip code -> our symbol."""
    rows = []
    for code, sym in codes.items():
        d = Path(root) / "shp" / code
        ip = d / "index.json"
        if not ip.exists():
            continue
        idx_rows = json.loads(ip.read_text())["rows"]
        nearest: dict[float, pd.Timedelta] = {}               # per qtrid: smallest |authorisation - filing| gap
        for r in idx_rows:
            q = float(r["qtrid"])
            ts0 = r.get("filing_date_time") or r.get("revised_date_time")
            pp0 = d / f"{qtr_key(q)}_promoter.json"
            if q < FIRST_QTRID or not ts0 or not pp0.exists():
                continue
            a0 = (json.loads(pp0.read_text()).get("Table") or [{}])[0].get("Fld_AuthoriseDate")
            if a0:
                g0 = abs(pd.Timestamp(a0) - pd.Timestamp(ts0))
                nearest[q] = min(nearest.get(q, g0), g0)
        for r in idx_rows:
            q = float(r["qtrid"])
            if q < FIRST_QTRID:
                continue
            pp, pu = d / f"{qtr_key(q)}_promoter.json", d / f"{qtr_key(q)}_public.json"
            filed, revised = r.get("filing_date_time"), r.get("revised_date_time")
            if not (filed or revised) or not pp.exists():
                continue
            pe, kind = period_end(r.get("qtr"))
            ts = pd.Timestamp(filed or revised)
            pj = json.loads(pp.read_text())
            auth = (pj.get("Table") or [{}])[0].get("Fld_AuthoriseDate")
            gap = abs(pd.Timestamp(auth) - ts) if auth is not None else None
            own = gap is not None and gap <= MATCH_TOLERANCE and gap == nearest.get(q, gap)   # only the closest row in a qtrid
            xp = d / "xbrl" / ((r.get("XbrlFile") or "none") + ".gz")
            if own:
                source, summ, avail_naive = "bse_api", None, max(ts, pd.Timestamp(auth))
            elif xp.exists():
                import gzip

                summ = xbrl_summary(gzip.decompress(xp.read_bytes()).decode("utf-8-sig", errors="replace"))
                source, avail_naive = ("bse_xbrl_as_filed", ts) if summ else ("bse_api_later_version", max(ts, pd.Timestamp(auth)))
            else:
                source, summ, avail_naive = "bse_api_later_version", None, max(ts, pd.Timestamp(auth)) if auth else ts
            avail = avail_naive.tz_localize("Asia/Kolkata")
            rec = {"symbol": sym, "bse_code": code, "qtrid": q, "period_label": r.get("qtr"), "period_end": pe, "kind": kind,
                   "status": r.get("status"), "filed_at": pd.Timestamp(filed).tz_localize("Asia/Kolkata") if filed else pd.NaT,
                   "revised_at": pd.Timestamp(revised).tz_localize("Asia/Kolkata") if revised else pd.NaT,
                   "available_at": avail, "revised_only": filed is None, "revision_reason": r.get("revised_reson") or None,
                   "xbrl_file": r.get("XbrlFile") or None, "source": source, "content_authorised_at": auth}
            if summ:
                rec.update(summ)
            else:
                rec.update(promoter_summary(pj))
                if pu.exists():
                    rec.update(public_summary(json.loads(pu.read_text())))
            rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["symbol", "available_at", "qtrid"]).reset_index(drop=True)
    return df


def as_of(df: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    """Per symbol: the newest quarter-end period with a filing public strictly before `when`, in its latest version
    public before `when`. A late revision of an old quarter never displaces a newer quarter."""
    w = pd.Timestamp(when)
    w = w.tz_localize("Asia/Kolkata") if w.tzinfo is None else w
    x = df[(df["kind"] == "quarterly") & (df["available_at"] < w)]
    return x.sort_values(["period_end", "available_at", "qtrid"]).groupby("symbol").tail(1).set_index("symbol")
