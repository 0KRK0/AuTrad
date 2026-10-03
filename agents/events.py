"""Events: how one tweet, speech or data release ripples into one asset, a category, or everything.

1. EventClassifier turns evidence text (news, filings, posts) into typed events with target categories.
2. ShockDetector spots abnormal price/volume moves and labels them EXPLAINED (news behind it) or
   UNEXPLAINED (possible pump). Unexplained spikes are the classic penny/meme trap.
3. ImpactGraph holds how strongly each event type moves each category; it starts from priors and
   learns from what actually happened after each event.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from .taxonomy import Category as C


class EventType(str, Enum):
    SOCIAL_POST = "social_post"          # e.g. a celebrity/CEO post about a coin or company
    POLICY_SPEECH = "policy_speech"      # RBI governor, Finance Minister, Fed chair, budget
    MACRO_RELEASE = "macro_release"      # CPI, GDP, IIP, PMI, jobs
    EARNINGS = "earnings"
    REGULATORY = "regulatory"            # SEBI order, crypto tax/ban news, exchange action
    GEOPOLITICAL = "geopolitical"
    PRICE_SHOCK = "price_shock"          # detected from market data, not text
    UNEXPLAINED_PUMP = "unexplained_pump"


# (pattern, event type, categories hit). Order matters: first match sets the type; categories accumulate.
RULES: list[tuple[str, EventType, set[C]]] = [
    (r"\b(elon|musk|tweet(ed)?|posted on x)\b", EventType.SOCIAL_POST, {C.CRYPTO_MEME, C.CRYPTO_ALT}),
    (r"\b(rbi|repo rate|monetary policy|governor)\b", EventType.POLICY_SPEECH, {C.FX_INR, C.IN_LARGECAP, C.IN_INDEX_FNO}),
    (r"\b(finance minister|union budget|budget|nirmala|gst council)\b", EventType.POLICY_SPEECH, {C.IN_LARGECAP, C.IN_MIDSMALL, C.IN_INDEX_FNO, C.FX_INR}),
    (r"\b(fed|fomc|powell|rate hike|rate cut)\b", EventType.POLICY_SPEECH, {C.FX_INR, C.CRYPTO_MAJOR, C.IN_INDEX_FNO, C.COMMODITY}),
    (r"\b(bank of england|chancellor|sterling|pound)\b", EventType.POLICY_SPEECH, {C.FX_INR}),
    (r"\b(cpi|inflation|gdp|iip|pmi|payrolls|jobs report)\b", EventType.MACRO_RELEASE, {C.FX_INR, C.IN_INDEX_FNO, C.COMMODITY}),
    (r"\b(quarterly results|q[1-4] results|earnings|profit rose|profit fell|guidance)\b", EventType.EARNINGS, set()),
    (r"\b(sebi|ban|crackdown|tds|crypto tax|delist(ed|ing)?|asm|gsm)\b", EventType.REGULATORY, {C.CRYPTO_MAJOR, C.CRYPTO_ALT, C.CRYPTO_MEME, C.IN_PENNY}),
    (r"\b(war|sanction|opec|strike on|border tension|tariff)\b", EventType.GEOPOLITICAL, {C.COMMODITY, C.FX_INR, C.IN_INDEX_FNO, C.CRYPTO_MAJOR}),
]
_COMPILED = [(re.compile(p, re.I), t, cats) for p, t, cats in RULES]


@dataclass
class Event:
    ts: datetime
    type: EventType
    text: str
    source: str
    symbols: list[str] = field(default_factory=list)
    categories: set[C] = field(default_factory=set)
    sentiment: float = 0.0       # -1..1
    strength: float = 0.5        # 0..1
    reliability: float = 0.5     # learned source reliability


def classify_text(text: str) -> tuple[EventType | None, set[C]]:
    etype, cats = None, set()
    for rx, t, c in _COMPILED:
        if rx.search(text):
            etype = etype or t
            cats |= c
    return etype, cats


def event_from_evidence(ev, reliability: float = 0.5) -> Event | None:
    """Adapter from intel.collector.Evidence."""
    etype, cats = classify_text(f"{ev.title} {ev.summary}")
    if etype is None and not ev.symbols:
        return None
    return Event(ev.ingested_at, etype or EventType.EARNINGS, ev.title, ev.source, list(ev.symbols), cats,
                 ev.sentiment, min(1.0, 0.3 + abs(ev.sentiment)), reliability)


class ShockDetector:
    """Rolling z-scores of return and volume; confirms a shock only if BOTH are extreme."""

    def __init__(self, window: int = 30, ret_z: float = 2.5, vol_z: float = 2.5) -> None:
        self.window, self.ret_z, self.vol_z = window, ret_z, vol_z
        self.hist: dict[str, list[tuple[float, float]]] = {}

    def update(self, symbol: str, ret: float, volume: float, has_news: bool, ts: datetime) -> Event | None:
        h = self.hist.setdefault(symbol, [])
        out = None
        if len(h) >= 10:
            r = np.array([x[0] for x in h])
            v = np.log1p(np.array([x[1] for x in h]))
            rz = (ret - r.mean()) / (r.std() + 1e-9)
            vz = (np.log1p(volume) - v.mean()) / (v.std() + 1e-9)
            if abs(rz) >= self.ret_z and vz >= self.vol_z:
                etype = EventType.PRICE_SHOCK if has_news else EventType.UNEXPLAINED_PUMP
                out = Event(ts, etype, f"{symbol} move z={rz:.1f}, volume z={vz:.1f}", "market",
                            [symbol], set(), float(np.sign(rz)), float(min(1.0, abs(rz) / 6)), 1.0)
        h.append((ret, volume))
        if len(h) > self.window:
            h.pop(0)
        return out


class ImpactGraph:
    """Expected signed move of each category per unit of event (sentiment x strength x reliability).

    Priors are small; every realized outcome updates the edge with an exponentially-weighted
    regression, so the graph learns e.g. that 'social_post -> crypto_meme' is strong while
    'social_post -> crypto_major' is weak.
    """

    def __init__(self, prior: float = 0.002, decay: float = 0.97) -> None:
        self.decay = decay
        self.sxy: dict[tuple[EventType, C], float] = {}
        self.sxx: dict[tuple[EventType, C], float] = {}
        self.prior = prior

    def beta(self, etype: EventType, cat: C) -> float:
        k = (etype, cat)
        # prior acts as one pseudo-observation; real outcomes dominate once they accumulate
        return (self.sxy.get(k, 0.0) + self.prior) / (self.sxx.get(k, 0.0) + 1.0)

    def impact(self, events: list[Event]) -> dict[C, float]:
        out: dict[C, float] = {}
        for e in events:
            x = e.sentiment * e.strength * e.reliability
            for cat in e.categories:
                out[cat] = out.get(cat, 0.0) + self.beta(e.type, cat) * x
        return out

    def learn(self, event: Event, realized: dict[C, float]) -> None:
        x = event.sentiment * event.strength * event.reliability
        for cat, y in realized.items():
            if cat not in event.categories:
                continue
            k = (event.type, cat)
            self.sxy[k] = self.decay * self.sxy.get(k, 0.0) + x * y
            self.sxx[k] = self.decay * self.sxx.get(k, 0.0) + x * x
