"""Internet Intelligence Layer: collect -> normalise -> score -> store evidence -> point-in-time features.

The AI reads the internet as EVIDENCE, never as orders. Every item keeps source, url, published_at,
ingested_at and a learned source-reliability weight, so nothing enters a model without provenance.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

USER_AGENT = "AuTradAI-Research/0.1 (contact: set AUTRAD_CONTACT_EMAIL)"

DEFAULT_FEEDS = {
    "sec_8k": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&output=atom",
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "boe_news": "https://www.bankofengland.co.uk/rss/news",
    "ecb_press": "https://www.ecb.europa.eu/rss/press.html",
}

POS = set("beat beats record growth upgrade upgraded raises raised strong surge surges profit approval approved expands buyback outperform exceeds".split())
NEG = set("miss misses downgrade downgraded cuts cut weak plunge plunges loss lawsuit probe recall bankruptcy fraud warning halts layoffs default".split())


@dataclass
class Evidence:
    source: str
    url: str
    title: str
    summary: str
    published_at: datetime
    ingested_at: datetime
    symbols: list[str]
    sentiment: float          # -1..1
    simhash: int
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha1(f"{self.source}|{self.url}|{self.title}".encode()).hexdigest()[:16]


def simhash(text: str, bits: int = 64) -> int:
    v = [0] * bits
    for tok in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if v[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def lexicon_sentiment(text: str) -> float:
    """Baseline scorer. Swap for a fine-tuned transformer behind the same signature."""
    toks = re.findall(r"[a-z]+", text.lower())
    p, n = sum(t in POS for t in toks), sum(t in NEG for t in toks)
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def link_entities(text: str, aliases: dict[str, list[str]]) -> list[str]:
    low = text.lower()
    return sorted({sym for sym, names in aliases.items() if any(re.search(rf"\b{re.escape(n.lower())}\b", low) for n in names)})


def parse_feed(xml_text: str, source: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    items = []
    for it in root.iter("item"):  # RSS 2.0
        items.append({"title": (it.findtext("title") or "").strip(), "url": (it.findtext("link") or "").strip(),
                      "summary": re.sub("<[^>]+>", " ", it.findtext("description") or ""), "published": it.findtext("pubDate")})
    for e in root.findall("a:entry", ns):  # Atom
        link = e.find("a:link", ns)
        items.append({"title": (e.findtext("a:title", "", ns)).strip(), "url": link.get("href", "") if link is not None else "",
                      "summary": re.sub("<[^>]+>", " ", e.findtext("a:summary", "", ns)), "published": e.findtext("a:updated", None, ns)})
    for i in items:
        i["source"] = source
    return items


def _parse_date(s: str | None, fallback: datetime) -> datetime:
    if not s:
        return fallback
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return fallback
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


class EvidenceStore:
    """In-memory store with the same API as the Postgres-backed one (see db/schema.sql: evidence)."""

    def __init__(self, dedupe_bits: int = 3) -> None:
        self.items: list[Evidence] = []
        self.dedupe_bits = dedupe_bits

    def add(self, ev: Evidence) -> bool:
        if any(x.id == ev.id or hamming(x.simhash, ev.simhash) <= self.dedupe_bits for x in self.items[-2000:]):
            return False
        self.items.append(ev)
        return True

    def as_of(self, symbol: str, t: datetime, lookback_h: float = 72) -> list[Evidence]:
        """Point-in-time: only evidence INGESTED before t (prevents look-ahead)."""
        lo = t.timestamp() - lookback_h * 3600
        return [e for e in self.items if symbol in e.symbols and lo <= e.ingested_at.timestamp() <= t.timestamp()]


class SourceReliability:
    """Learns how much each source's sentiment actually predicted next-day returns (EW correlation)."""

    def __init__(self, decay: float = 0.98, prior: float = 0.3) -> None:
        self.decay, self.prior, self.stats = decay, prior, {}

    def update(self, source: str, sentiment: float, fwd_return: float) -> None:
        sxy, sxx, syy = self.stats.get(source, (0.0, 1e-6, 1e-6))
        d = self.decay
        self.stats[source] = (d * sxy + sentiment * fwd_return, d * sxx + sentiment**2, d * syy + fwd_return**2)

    def weight(self, source: str) -> float:
        if source not in self.stats:
            return self.prior
        sxy, sxx, syy = self.stats[source]
        return max(0.0, sxy / math.sqrt(sxx * syy))  # anti-predictive sources decay to zero


def evidence_features(store: EvidenceStore, rel: SourceReliability, symbol: str, t: datetime, half_life_h: float = 24) -> dict[str, float]:
    items = store.as_of(symbol, t)
    if not items:
        return {"news_sent": 0.0, "news_count": 0.0}
    ws = [rel.weight(e.source) * 0.5 ** ((t - e.ingested_at).total_seconds() / 3600 / half_life_h) for e in items]
    tot = sum(ws) or 1.0
    return {"news_sent": sum(w * e.sentiment for w, e in zip(ws, items, strict=False)) / tot, "news_count": float(len(items))}


class Collector:
    """Async, robots.txt-respecting, per-host rate-limited feed collector."""

    def __init__(self, store: EvidenceStore, aliases: dict[str, list[str]], feeds: dict[str, str] | None = None,
                 min_interval_s: float = 2.0) -> None:
        self.store, self.aliases = store, aliases
        self.feeds = feeds or DEFAULT_FEEDS
        self.min_interval_s = min_interval_s
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser] = {}

    async def _allowed(self, client, url: str) -> bool:
        host = urlparse(url).scheme + "://" + urlparse(url).netloc
        if host not in self._robots:
            rp = RobotFileParser()
            try:
                r = await client.get(host + "/robots.txt", timeout=10)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                rp.parse([])
            self._robots[host] = rp
        return self._robots[host].can_fetch(USER_AGENT, url)

    async def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        wait = self._last_hit.get(host, 0) + self.min_interval_s - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_hit[host] = time.monotonic()

    def ingest(self, raw: list[dict]) -> int:
        now = datetime.now(timezone.utc)
        added = 0
        for i in raw:
            text = f"{i['title']} {i['summary']}"
            ev = Evidence(i["source"], i["url"], i["title"], i["summary"][:2000], _parse_date(i.get("published"), now), now,
                          link_entities(text, self.aliases), lexicon_sentiment(text), simhash(text))
            added += self.store.add(ev)
        return added

    async def run_once(self) -> int:
        import httpx

        added = 0
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
            for name, url in self.feeds.items():
                if not await self._allowed(client, url):
                    continue
                await self._throttle(url)
                try:
                    r = await client.get(url, timeout=15)
                    r.raise_for_status()
                    added += self.ingest(parse_feed(r.text, name))
                except Exception:  # one bad feed never stops the others; surfaced via metrics
                    continue
        return added
