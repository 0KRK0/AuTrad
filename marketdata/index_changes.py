"""NSE Indices press releases on index changes (v1.2 study HI). Lists the dated releases on
https://www.niftyindices.com/press-release, downloads the relevant PDFs, extracts text with pdftotext (kept under
data/events/press_text/) and parses, per index: excluded and included symbols and the effective date.
Output: data/events/index_changes.csv (announce_date, effective_date, index_key, index_name, action, symbol, company,
url, title).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path("data/events")
TXT = ROOT / "press_text"
LIST_URL = "https://www.niftyindices.com/press-release"
BAD = r"Fixed Income|IPO|SME|methodology|launch|Maturity|REIT|Bharat Bond|G-Sec|SDL|Target Maturity|T-Bill|Debt|Bond"
HEADER = re.compile(r"^\s*(?:\(?(?:[0-9]{1,2}|[a-zA-Z]|[ivxIVX]{1,4})[).])?\s*((?:CNX|Nifty|NIFTY|S&P CNX)[A-Za-z0-9 &\-]{0,40}?)\s*(?:index|Index)?\s*:?\s*$")
ROW = re.compile(r"^\s*\d{1,3}\s+(.+?)\s{2,}([A-Z0-9][A-Z0-9&\-]{1,19})\s*$")
EFF = re.compile(r"effective from\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", re.I)


def index_key(name: str) -> str:
    k = re.sub(r"[^A-Z0-9]", "", name.upper().replace("S&P", ""))
    return {"CNXNIFTY": "NIFTY50", "NIFTY": "NIFTY50", "CNXNIFTYJUNIOR": "NIFTYNEXT50", "NIFTYJUNIOR": "NIFTYNEXT50",
            "CNXBANK": "NIFTYBANK", "BANKNIFTY": "NIFTYBANK"}.get(k, k)


def releases() -> pd.DataFrame:  # pragma: no cover - network
    import requests
    from marketdata.nse_bhav import UA
    h = requests.get(LIST_URL, headers=UA, timeout=30).text
    items = re.findall(r'data-date="([^"]+)"[^>]*>\s*<p>[^<]*</p>\s*<a href=\'([^\']+)\'[^>]*>([^<]+)</a>', h)
    df = pd.DataFrame(items, columns=["date", "url", "title"])
    df["date"] = pd.to_datetime(df["date"])
    df["title"] = df["title"].str.strip()
    return df


def relevant(df: pd.DataFrame, start: str = "2015-01-01", end: str = "2026-09-30") -> pd.DataFrame:
    m = df["title"].str.contains(r"change|replacement|inclusion|exclusion|review", case=False) & ~df["title"].str.contains(BAD, case=False)
    return df[m & (df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


def fetch_texts(rel: pd.DataFrame) -> int:  # pragma: no cover - network
    import requests
    from marketdata.nse_bhav import UA
    TXT.mkdir(parents=True, exist_ok=True)
    n = 0
    for u in rel["url"]:
        out = TXT / (Path(u).stem + ".txt")
        if out.exists():
            continue
        r = requests.get("https://www.niftyindices.com" + u, headers=UA, timeout=40)
        if r.status_code != 200 or r.content[:4] != b"%PDF":
            continue
        pdf = Path("/tmp") / Path(u).name
        pdf.write_bytes(r.content)
        subprocess.run(["pdftotext", "-layout", str(pdf), str(out)], check=False, timeout=60)
        pdf.unlink(missing_ok=True)
        n += 1
    return n


def parse_text(text: str) -> tuple[pd.Timestamp | None, list[dict]]:
    m = EFF.search(" ".join(text.split()))
    eff = pd.to_datetime(m.group(1).replace(",", ""), errors="coerce") if m else None
    rows, idx, mode = [], None, None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        h = HEADER.match(line)
        if h and len(line.strip()) <= 60:
            nxt = next((x for x in lines[i + 1:i + 6] if x.strip()), "")       # a real section header is followed by
            if re.search(r"(being|to be) (excluded|included)|following compan", nxt, re.I):   # the exclusion/inclusion line
                idx, mode = h.group(1).strip(), None
            continue
        low = line.lower()
        if "being excluded" in low or "to be excluded" in low:
            mode = "exclude"
            continue
        if "being included" in low or "to be included" in low:
            mode = "include"
            continue
        r = ROW.match(line)
        if r and idx and mode:
            rows.append({"index_name": idx, "index_key": index_key(idx), "action": mode, "company": r.group(1).strip(), "symbol": r.group(2)})
    return eff, rows


def build(rel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, x in rel.iterrows():
        p = TXT / (Path(x["url"]).stem + ".txt")
        if not p.exists():
            continue
        eff, rows = parse_text(p.read_text(errors="ignore"))
        for r in rows:
            out.append({"announce_date": x["date"].date(), "effective_date": eff.date() if eff is not None and not pd.isna(eff) else None,
                        **r, "url": x["url"], "title": x["title"]})
    df = pd.DataFrame(out)
    df.to_csv(ROOT / "index_changes.csv", index=False)
    return df


if __name__ == "__main__":  # pragma: no cover
    rel = relevant(releases())
    rel.to_csv(ROOT / "index_press_releases.csv", index=False)
    print("relevant", len(rel), "fetched", fetch_texts(rel))
    df = build(rel)
    print(len(df), df["index_key"].value_counts().head(12).to_dict())
