"""Legitimate sources for the Company Intelligence agent. Each connector turns what it reads into Evidence Ledger
documents (intel/evidence.py) with a source tier, URL, publication time and content hash.

  * NSE results filings (XBRL) already stored by marketdata/nse_filings.py             tier 1
  * NSE corporate announcements and board-meeting intimations (official exchange API)  tier 1
  * AuTrad's audited datasets (exchange-anchored prices)                               tier 2
  * News headlines via Google News RSS (opt-in; headline + link only, personal research use) tier 5 for listed
    publishers, else tier 8
  * Official social channels: only through the platform's official API with your own credentials and only for
    accounts in reference/official_channels.csv (verified by a person). Disabled when no credentials are set.
No scraping behind logins, no paywalled content, no private data about individuals.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from .evidence import Document, Tier

REF = Path(__file__).parent / "reference"
REPUTABLE = {"Reuters", "The Economic Times", "Economic Times", "Mint", "Livemint", "Business Standard", "Moneycontrol",
             "Financial Express", "The Financial Express", "BusinessLine", "The Hindu BusinessLine", "Bloomberg", "CNBC TV18",
             "CNBC-TV18", "NDTV Profit", "The Hindu", "Hindustan Times", "Times of India", "Financial Times"}


def _ist(s: str, fmt: str) -> str:
    return pd.Timestamp(datetime.strptime(s.strip(), fmt)).tz_localize("Asia/Kolkata").isoformat()


# ----------------------------------------------------------------------------- offline (stored) sources
def filing_documents(facts: pd.DataFrame, symbol: str) -> list[Document]:
    """One document per stored NSE results filing: the reported numbers, at the time NSE disseminated them."""
    out = []
    for r in facts[facts["symbol"] == symbol].itertuples(index=False):
        basis = "consolidated" if getattr(r, "consolidated", False) else "standalone"
        def cr(x):
            return "n/a" if pd.isna(x) else f"Rs {x / 1e7:,.0f} cr"
        text = (f"{symbol} results for the quarter ended {pd.Timestamp(r.period_end).date()} ({basis}, {r.audited}): "
                f"revenue {cr(r.revenue)}; net profit {cr(r.net_profit)}; basic EPS {r.eps_basic if not pd.isna(r.eps_basic) else 'n/a'}")
        out.append(Document("NSE results filing (XBRL)", int(Tier.REGULATORY_FILING), str(r.xbrl), pd.Timestamp(r.available_at).isoformat(),
                            [symbol], f"{symbol} quarterly results {pd.Timestamp(r.period_end).date()}", text,
                            f"marketdata/nse_filings.py; xbrl sha256 {str(r.sha256)[:16]}"))
    return out


def dataset_extract(symbol: str, date: pd.Timestamp, values: dict[str, float | str], dataset_hash: str) -> Document:
    """Raw numbers copied from AuTrad's exchange-anchored datasets as of a session close (tier 9: reproducible data,
    no interpretation, never counted as tier 1-2 evidence). Statements built on it are DERIVED."""
    body = "; ".join(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}" for k, v in values.items())
    return Document("AuTrad dataset extract", int(Tier.DATASET_EXTRACT), f"dataset:{dataset_hash}/{symbol}/{pd.Timestamp(date).date()}",
                    (pd.Timestamp(date).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)).isoformat(), [symbol],
                    f"{symbol} data extract {pd.Timestamp(date).date()}", f"{symbol} {body}", "lab.panel.build_panel (BSE/NSE anchored)")


def pit_document(symbol: str, eff: pd.Timestamp, row_text: str) -> Document:
    return Document("NIFTY 50 PIT snapshots (third-party reconstruction of NSE Indices releases)", 6,
                    f"marketdata/reference/nifty50_pit_snapshots.csv#{pd.Timestamp(eff).date()}/{symbol}",
                    pd.Timestamp(eff).tz_localize("Asia/Kolkata").isoformat(), [symbol], f"{symbol} NIFTY 50 membership", row_text, "reference file")


def filing_quotes(doc: Document) -> dict[str, str]:
    """The verbatim pieces of a filing document a statement may quote."""
    import re

    out = {}
    m = re.search(r"quarter ended \d{4}-\d{2}-\d{2}", doc.text)
    if m:
        out["quarter ended"] = m.group(0)
    for key in ("revenue", "net profit", "basic EPS"):
        m = re.search(rf"{key} [^;]+", doc.text)
        if m:
            out[key] = m.group(0).strip()
    return out


# ----------------------------------------------------------------------------- network sources (run on your machine)
class NSEAnnouncements:  # pragma: no cover - network
    def __init__(self, client=None) -> None:
        from marketdata.nse_filings import Client

        self.c = client or Client()

    def fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[Document]:
        url = (f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={quote_plus(symbol)}"
               f"&from_date={start.strftime('%d-%m-%Y')}&to_date={end.strftime('%d-%m-%Y')}")
        raw = self.c.get(url)
        out = []
        for a in json.loads(raw) if raw else []:
            t = a.get("exchdisstime") or a.get("an_dt")
            out.append(Document("NSE corporate announcements", int(Tier.REGULATORY_FILING), a.get("attchmntFile") or url,
                                _ist(t, "%d-%b-%Y %H:%M:%S"), [symbol], a.get("desc", ""), a.get("attchmntText", "") or a.get("desc", ""),
                                "NSE /api/corporate-announcements"))
        return out

    def board_meetings(self, symbol: str) -> list[Document]:
        raw = self.c.get(f"https://www.nseindia.com/api/corporate-board-meetings?index=equities&symbol={quote_plus(symbol)}")
        out = []
        for a in json.loads(raw) if raw else []:
            out.append(Document("NSE board meetings", int(Tier.REGULATORY_FILING), a.get("attachment") or "nse:board-meetings",
                                _ist(a["bm_timestamp"], "%d-%b-%Y %H:%M:%S"), [symbol], f"Board meeting {a.get('bm_date')}: {a.get('bm_purpose')}",
                                a.get("bm_desc", ""), "NSE /api/corporate-board-meetings"))
        return out


class NewsRSS:  # pragma: no cover - network
    """Headlines and links only (no article bodies). Opt-in: personal, non-commercial research use of the feed."""

    def fetch(self, symbol: str, company: str, days: int = 30) -> list[Document]:
        import requests
        import xml.etree.ElementTree as ET

        q = quote_plus(f'"{company}" when:{days}d')
        r = requests.get(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en", headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        out = []
        if r.status_code != 200:
            return out
        for it in ET.fromstring(r.content).iter("item"):
            title = (it.findtext("title") or "").strip()
            src = (it.findtext("source") or "").strip()
            pub = it.findtext("pubDate")
            when = pd.Timestamp(pub).tz_convert("UTC").isoformat() if pub else ""
            tier = int(Tier.JOURNALISM) if src in REPUTABLE else int(Tier.OTHER_MEDIA)
            out.append(Document(f"news: {src or 'unknown'}", tier, it.findtext("link") or "", when, [symbol], title, title, "Google News RSS"))
        return out


def official_channels() -> pd.DataFrame:
    p = REF / "official_channels.csv"
    return pd.read_csv(p, comment="#") if p.exists() else pd.DataFrame(columns=["symbol", "channel", "handle", "role", "verified_by", "verified_on"])


class XOfficialAPI:  # pragma: no cover - network
    """Posts from verified official accounts via the X API v2 (needs X_BEARER_TOKEN). Executive accounts are tier 4,
    company accounts tier 3. A post is a claim by its author, never a verified fact."""

    def __init__(self) -> None:
        self.token = os.environ.get("X_BEARER_TOKEN")

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def fetch(self, symbol: str, max_results: int = 20) -> list[Document]:
        if not self.enabled:
            return []
        import requests

        out = []
        h = {"Authorization": f"Bearer {self.token}"}
        for r in official_channels().query("symbol == @symbol and channel == 'x'").itertuples(index=False):
            u = requests.get(f"https://api.x.com/2/users/by/username/{r.handle}", headers=h, timeout=20).json().get("data", {})
            if not u:
                continue
            tw = requests.get(f"https://api.x.com/2/users/{u['id']}/tweets", headers=h, timeout=20,
                              params={"max_results": max_results, "tweet.fields": "created_at"}).json().get("data", [])
            tier = int(Tier.EXECUTIVE)
            for t in tw:
                out.append(Document(f"X @{r.handle}", tier, f"https://x.com/{r.handle}/status/{t['id']}", t["created_at"], [symbol],
                                    f"post by @{r.handle}", t["text"], "X API v2 (official)"))
        return out
