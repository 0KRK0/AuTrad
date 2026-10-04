"""Company-website (investor-relations) earnings-call transcripts — tier 2, company disclosure.

Some companies file only a cover letter with the exchange ("the transcript has been uploaded at <link>"). This module
collects the transcripts from the companies' own IR pages (one small collector per site) and stamps each one with a
point-in-time availability instant:

  * `available_at` = the exchange timestamp of the company's own NSE intimation that the transcript was uploaded
    (the earliest transcript-related NSE announcement from the call date to call date + 21 days), when one exists;
  * otherwise `available_at` is empty and the document may only be used from `first_seen_at` (the day we fetched it).
    It is never back-dated to the call date.

Collectors (as of Oct 2026): MARUTI (site GraphQL feed), LT (ASP.NET archive, one post-back per fiscal year),
TATASTEEL (static page). HINDUNILVR's site serves files only to interactive browsers (bot protection), so it is not
collected automatically; 10 of its 15 NSE filings since 2022 already carry the full transcript.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .evidence import Document, Tier

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RX = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS}),?\s+(20\d\d)\b|\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d\d)\b", re.I)


@dataclass
class WebTranscript:
    symbol: str
    url: str
    title: str
    fiscal_label: str = ""
    listed_at: str = ""                  # date the site attaches to the item, if any (not used for PIT)
    path: str = ""                       # local pdf
    sha256: str = ""
    first_seen_at: str = ""
    call_date: str = ""                  # from the transcript text
    available_at: str = ""               # exchange intimation timestamp (IST ISO) or ""
    intimation_url: str = ""
    full_text: bool = False


def _session():  # pragma: no cover - network
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    return s


# ------------------------------------------------------------------------------------------------ collectors
def maruti(s) -> list[WebTranscript]:  # pragma: no cover - network
    u = "https://www.marutisuzuki.com/graphql/execute.json/msil-platform/corporateDocumentsList;documentCategory=companyReports"
    items = s.get(u, timeout=60).json()["data"]["corporateDocumentsList"]["items"]
    out = []
    for x in items:
        blob = f"{x.get('title')} {x.get('_path')} {(x.get('description') or {}).get('markdown')}"
        pl = x.get("pdfLink") or {}
        pdf = ("https://www.marutisuzuki.com" + pl["_path"]) if pl.get("_path") else pl.get("_publishUrl")   # publish host 404s
        if pdf and re.search(r"transcript", blob, re.I):
            out.append(WebTranscript("MARUTI", pdf, ((x.get("description") or {}).get("markdown") or x.get("title") or "").strip(),
                                     x.get("yearRange") or "", x.get("date") or ""))
    return out


def lt(s) -> list[WebTranscript]:  # pragma: no cover - network
    u = "https://investors.larsentoubro.com/Transcripts-Archives.aspx"
    t = s.get(u, timeout=60).text
    years = re.findall(r'<option(?: selected="selected")? value="(FY\d{4})"', t)
    out = []
    for fy in years:
        h = {m.group(1): m.group(2) for m in re.finditer(r'<input type="hidden" name="([^"]+)" id="[^"]*" value="([^"]*)"', t)}
        h.update({"__EVENTTARGET": "ddlDtYr", "__EVENTARGUMENT": "", "ddlDtYr": fy})
        page = s.post(u, data=h, timeout=60).text
        for m in re.finditer(r"<i>([^<]+)</i>.*?fnDownloadpdf\('([^']+)'\)", page, re.S):
            out.append(WebTranscript("LT", m.group(2).replace(" ", "%20"), m.group(1).strip(), fy))
        time.sleep(0.5)
    return out


def tatasteel(s) -> list[WebTranscript]:  # pragma: no cover - network
    base = "https://www.tatasteel.com"
    t = s.get(base + "/investors/financial-performance/analyst-call-recording/", timeout=60).text
    out = []
    for m in re.finditer(r'href="(/media/[^"]+\.pdf)"', t):
        if re.search(r"transcript", m.group(1), re.I):
            out.append(WebTranscript("TATASTEEL", base + m.group(1), Path(m.group(1)).stem, tatasteel_label(m.group(1))))
    return out


def tatasteel_label(path: str) -> str:
    """'/media/1/2qfy2025-transcript-v7.pdf' -> '2QFY2025' (four-digit years kept whole)."""
    m = re.search(r"([1-4]qfy(?:20)?\d\d)(?!\d)", path, re.I)
    return m.group(1).upper() if m else ""


COLLECTORS = {"MARUTI": maruti, "LT": lt, "TATASTEEL": tatasteel}


# ------------------------------------------------------------------------------------------------ download + stamp
def download(s, w: WebTranscript, root: Path) -> WebTranscript:  # pragma: no cover - network
    d = Path(root) / w.symbol
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(w.url.encode()).hexdigest()[:16]
    pdf = d / f"{key}.pdf"
    if not pdf.exists():
        content = None
        for i in range(3):
            try:
                r = s.get(w.url, timeout=90)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    content = r.content
                    break
                if r.status_code in (403, 404):
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3.0 * (i + 1))
        if content is None:
            return w
        pdf.write_bytes(content)
        w.first_seen_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    txt = pdf.with_suffix(".txt")
    if not txt.exists():
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=False, timeout=90)
    w.path = str(pdf).replace("\\", "/")
    w.sha256 = hashlib.sha256(pdf.read_bytes()).hexdigest()
    return w


def quarter_end_from_label(label: str) -> pd.Timestamp | None:
    """'4QFY26', 'Q2 FY18', 'Q1FY27', 'Q3 FY 26', 'Q2, FY2025-26' -> end of that Indian fiscal quarter."""
    t = (label or "").upper().replace(" ", "").replace(",", "")
    m = re.search(r"(?:([1-4])Q|Q([1-4]))FY(20\d\d|\d\d)(?!\d)(?:-(\d\d)(?!\d))?", t)
    if not m:
        return None
    q = int(m.group(1) or m.group(2))
    fy = int(m.group(4) or m.group(3)[-2:]) + 2000              # FY2025-26 / FY26 / FY2026 -> fiscal year ending Mar 2026
    if len(m.group(3)) == 4 and not m.group(4):
        fy = int(m.group(3))
    month, year = {1: (6, fy - 1), 2: (9, fy - 1), 3: (12, fy - 1), 4: (3, fy)}[q]
    return pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)


def call_date(text: str) -> pd.Timestamp | None:
    """The call date as printed on the transcript's first pages (first explicit date in the first 4,000 chars that is
    not a period end such as 'quarter ended March 31, 2026')."""
    head = text[:4000]
    for m in DATE_RX.finditer(head):
        if re.search(r"(ended|ending|as on|as of|as at)\s*$", head[max(0, m.start() - 25):m.start()], re.I):
            continue
        try:
            if m.group(1):
                return pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
            return pd.Timestamp(f"{m.group(5)} {m.group(4)} {m.group(6)}")
        except ValueError:
            continue
    return None


def notice_subject(a: dict) -> str:
    t = a.get("attchmntText") or ""
    m = re.search(r"Sub(?:ject)?\s*[.:-]+\s*(.{0,160})", t, re.I | re.S)
    return " ".join((m.group(1) if m else t[:160]).split())


def is_transcript_notice(a: dict) -> bool:
    """An NSE notice *about a transcript* (not the audio/video recording notice, whose text often links to a page
    called 'transcripts')."""
    head = f"{a.get('desc') or ''} | {notice_subject(a)}"
    return bool(re.search(r"transcript", head, re.I)) and not re.search(r"\baudio\b|\brecording\b|\bvideo\b|\bclip\b", head, re.I)


def notice_quarter(a: dict) -> pd.Timestamp | None:
    return quarter_end_from_label(re.sub(r"[\s'’]", "", notice_subject(a)))


def transcript_quarter(w: WebTranscript) -> pd.Timestamp | None:
    return next((q for q in (quarter_end_from_label(x) for x in (w.fiscal_label, w.title, Path(w.url).stem)) if q is not None), None)


def stamp(w: WebTranscript, nse_index: list[dict], window_days: int = 21) -> WebTranscript:
    """Attach the exchange intimation timestamp (PIT) — never the call date itself."""
    from .transcripts import is_full_transcript

    if not w.path:
        return w
    txt = Path(w.path).with_suffix(".txt")
    text = txt.read_text(errors="ignore") if txt.exists() else ""
    w.full_text = is_full_transcript(text)
    cd = call_date(text)
    w.call_date = str(cd.date()) if cd is not None else ""
    if cd is not None:
        lo, hi = cd, cd + pd.Timedelta(days=window_days)
    qe = transcript_quarter(w)
    if cd is None:                                           # undated transcript: use its fiscal-quarter label
        if qe is None:
            return w
        lo, hi = qe, qe + pd.Timedelta(days=75)
        w.call_date = f"undated; quarter {qe.date()}"
    best = None
    w.available_at, w.intimation_url = "", ""
    for a in nse_index:
        if not is_transcript_notice(a):
            continue
        t = pd.to_datetime(a.get("exchdisstime") or a.get("an_dt"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
        if pd.isna(t) or not (lo <= t.normalize() <= hi):
            continue
        nq = notice_quarter(a)
        if qe is not None and nq is not None and nq != qe:
            continue                                         # the notice is about a different quarter's call
        if best is None or t < best[0]:
            best = (t, a.get("attchmntFile", ""))
    if best is not None:
        w.available_at = best[0].tz_localize("Asia/Kolkata").isoformat()
        w.intimation_url = best[1]
    return w


def document_for(w: WebTranscript, text: str) -> Document | None:
    """Tier-2 company disclosure. Without an exchange timestamp the document is published 'as of first fetch'."""
    when = w.available_at or (pd.Timestamp(w.first_seen_at).tz_convert("Asia/Kolkata").isoformat() if w.first_seen_at else "")
    if not when:
        return None
    return Document(f"Company IR website ({w.symbol})", int(Tier.COMPANY_DISCLOSURE), w.url, when, [w.symbol],
                    f"Earnings-call transcript: {w.title}", " ".join(text.split()),
                    "company IR collector + pdftotext" + ("" if w.available_at else " (no exchange timestamp; first-seen only)"))


def save_index(root: Path, items: list[WebTranscript]) -> None:
    p = Path(root) / "index.json"
    p.write_text(json.dumps([asdict(w) for w in items], indent=1))


def load_index(root: Path) -> list[WebTranscript]:
    p = Path(root) / "index.json"
    return [WebTranscript(**x) for x in json.loads(p.read_text())] if p.exists() else []


# ------------------------------------------------------------------------------------------------ cover-letter links
def cover_letter_links(pdf: Path, text: str = "") -> list[str]:
    """Transcript PDF links in an exchange cover letter: the PDF's hyperlink annotations first, then URLs in the text
    re-joined across pdftotext line wraps. Only links that look like a transcript PDF are kept."""
    import urllib.parse

    urls: list[str] = []
    try:
        import pypdf

        for pg in pypdf.PdfReader(str(pdf)).pages:
            for a in pg.get("/Annots") or []:
                act = a.get_object().get("/A")
                uri = act.get_object().get("/URI") if act else None
                if uri:
                    urls.append(str(uri))
    except Exception:  # noqa: BLE001
        pass
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        for m in re.finditer(r"https?://\S+", ln):
            u = m.group(0)
            j = i + 1
            while not re.search(r"\.pdf$", u, re.I) and j < len(lines) and j <= i + 3:
                nxt = lines[j].strip().split(" ")[0] if lines[j].strip() else ""
                if not nxt or not re.fullmatch(r"[\w%./\-()&=?~+]+", nxt):
                    break
                u += nxt
                j += 1
            urls.append(u)
    out = []
    for u in urls:
        if "safelinks.protection.outlook.com" in u:              # unwrap Outlook safe-links
            q = urllib.parse.parse_qs(urllib.parse.urlparse(u).query).get("url")
            u = q[0] if q else u
        u = u.strip().rstrip(".,);]").replace(" ", "%20")
        if re.search(r"\.pdf$", u, re.I) and re.search(r"transcript|earnings[-_ %20]*call|con(?:ference)?[-_ %20]*call", u, re.I) \
                and not re.search(r"agm|annual[-_ %20]*general", u, re.I) and u not in out:
            out.append(u)
    return out
