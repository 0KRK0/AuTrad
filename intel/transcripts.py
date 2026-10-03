"""Earnings-call transcripts filed with NSE (tier 1: on the exchange's record of what was said) and a rule-based
extractor of management promises: forward-looking statements with a measurable target and a time anchor.

Extraction is deliberately conservative and produces CANDIDATES. Each keeps the speaker (from the transcript's
"Name:" turn labels), the verbatim sentence and the document id; nothing is paraphrased into a fact. A promise is a
claim that something will happen; its truth is decided only later, by outcome evidence.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from .evidence import Document, Tier

FUTURE = re.compile(r"\b(we will|we'll|we expect|we are targeting|we target|our target|target(?:ing)? to|we aim|we plan to|we intend|"
                    r"should (?:reach|be|cross|touch|achieve)|will (?:reach|be|cross|touch|achieve|grow|complete|commission|add)|guidance)\b", re.I)
NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?\s?(?:%|percent|per cent|million|mn|crore|cr|billion|bn|mtpa|gw|mw|megawatts?|gigawatts?|units|stores|tonnes?|x)(?!\w))", re.I)
ANCHOR = re.compile(r"\b(?:by|in|before|end of|within|over|during)\s+(?:the\s+)?(?:end of\s+)?(?:fy\s?'?\d{2,4}|fiscal\s+'?\d{2,4}|20\d\d|"
                    r"this (?:financial year|year|fiscal|quarter)|next (?:financial year|fiscal year|year|fiscal|quarter|\w+ years)|"
                    r"the next \w+ (?:years|quarters)|h[12]\b)", re.I)
METRICS = [("capacity", r"capacity|mtpa|million tons?\b|million tonnes?|gw\b|mw\b"), ("revenue_growth", r"revenue|top ?line|sales growth|volume growth"),
           ("margin", r"margin|ebitda"), ("capex", r"capex|capital expenditure|investment of"), ("debt", r"debt|leverage|net debt"),
           ("launch_or_completion", r"launch|commission|complet|start(?:ed)? production|ramp")]


def transcript_announcements(client, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:  # pragma: no cover - network
    url = (f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
           f"&from_date={start.strftime('%d-%m-%Y')}&to_date={end.strftime('%d-%m-%Y')}")
    raw = client.get(url)
    return [a for a in (json.loads(raw) if raw else []) if "transcript" in (a.get("attchmntText") or "").lower()]


def download_text(client, a: dict, root: Path) -> Path | None:  # pragma: no cover - network
    root.mkdir(parents=True, exist_ok=True)
    pdf = root / Path(a["attchmntFile"]).name
    txt = pdf.with_suffix(".txt")
    if txt.exists():
        return txt
    blob = client.get(a["attchmntFile"])
    if not blob or blob[:4] != b"%PDF":
        return None
    pdf.write_bytes(blob)
    subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=False, timeout=60)
    return txt if txt.exists() and txt.stat().st_size > 500 else None


def document_for(a: dict, symbol: str, text: str) -> Document:
    from .sources import _ist

    return Document("NSE filing: earnings-call transcript", int(Tier.REGULATORY), a["attchmntFile"],
                    _ist(a.get("exchdisstime") or a["an_dt"], "%d-%b-%Y %H:%M:%S"), [symbol], "Earnings-call transcript",
                    " ".join(text.split()), "NSE corporate announcements + pdftotext")


TURN = re.compile(r"^\s{0,12}([A-Z][A-Za-z.' ]{2,40}?):\s+(.*)$")


def turns(text: str) -> list[tuple[str, str]]:
    """(speaker, utterance) pairs from 'Name:   text' layout; continuation lines join the current turn."""
    out, cur, buf = [], None, []
    for line in text.splitlines():
        m = TURN.match(line)
        if m and len(m.group(1).split()) <= 5:
            if cur is not None:
                out.append((cur, " ".join(buf)))
            cur, buf = m.group(1).strip(), [m.group(2)]
        elif cur is not None and line.strip():
            buf.append(line.strip())
    if cur is not None:
        out.append((cur, " ".join(buf)))
    return [(s, " ".join(u.split())) for s, u in out]


def answering_speakers(tt: list[tuple[str, str]]) -> set[str]:
    """Management = speakers who answer: mostly statements, few questions. Analysts and the moderator are excluded."""
    stats: dict[str, list[int]] = {}
    for s, u in tt:
        stats.setdefault(s, [0, 0])
        stats[s][0] += 1
        stats[s][1] += "?" in u
    return {s for s, (n, q) in stats.items() if s.lower() not in ("moderator", "operator") and n >= 2 and q / n < 0.3}


def is_full_transcript(text: str) -> bool:
    """Many filings are cover letters pointing to the company website; those carry no transcript text."""
    return len(text) > 8000 and len(turns(text)) >= 10


def promise_candidates(text: str) -> list[dict]:
    tt = turns(text)
    mgmt = answering_speakers(tt)
    out = []
    for speaker, utt in tt:
        if speaker not in mgmt:
            continue                                       # analysts' questions are not promises
        for sent in re.split(r"(?<=[.!?])\s+", utt):
            if len(sent) < 30 or len(sent) > 400 or "?" in sent:
                continue
            if not (FUTURE.search(sent) and ANCHOR.search(sent)):
                continue
            metric = next((k for k, rx in METRICS if re.search(rx, sent, re.I)), "other")
            nums = list(NUMBER.finditer(sent))
            ft = re.search(r"\bfrom\b.*?\bto\b", sent, re.I)          # "from X to Y": the target is Y
            after = [n for n in nums if ft and n.start() >= ft.end()]
            num = after[0] if after else (nums[0] if nums else None)
            if num or metric == "launch_or_completion":       # a measurable target, or a dated completion
                anchors = [a.group(0) for a in ANCHOR.finditer(sent)]
                dated = [a for a in anchors if re.search(r"fy|fiscal|20\d\d|year", a, re.I)]
                out.append({"speaker": speaker, "quote": sent, "metric": metric, "target": num.group(1) if num else "",       # heuristic parse; the quote is the record
                            "anchor": (dated or anchors)[0]})
    return out


def due_date(anchor: str, said: pd.Timestamp) -> str:
    """Best-effort due date from the time anchor (Indian fiscal year ends 31 March)."""
    a = anchor.lower()
    m = re.search(r"(?:fy\s?'?|fiscal\s+'?)(\d{2,4})", a)
    if m:
        y = int(m.group(1))
        y = y + 2000 if y < 100 else y
        return f"{y}-03-31"
    m = re.search(r"(20\d\d)", a)
    if m:
        return f"{m.group(1)}-12-31"
    fy_end = pd.Timestamp(year=said.year + (1 if said.month > 3 else 0), month=3, day=31)
    if "this year" in a or "this fiscal" in a or "this financial" in a:
        return str(fy_end.date())
    if "next year" in a or "next fiscal" in a or "next financial" in a:
        return str((fy_end + pd.DateOffset(years=1)).date())
    m = re.search(r"next (\w+) years", a)
    words = {"two": 2, "three": 3, "four": 4, "five": 5}
    if m and m.group(1) in words:
        return str((said + pd.DateOffset(years=words[m.group(1)])).date())
    return ""
