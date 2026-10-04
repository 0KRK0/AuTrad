"""v0.8 transcripts: NSE-filed transcripts for the whole current NIFTY 50, and company-website transcripts where the
NSE filing is only a cover letter.

    python -m scripts.v08_transcripts nse --budget 160     # resumable (network, your machine)
    python -m scripts.v08_transcripts web --budget 160     # MARUTI, LT, TATASTEEL IR sites (network)
    python -m scripts.v08_transcripts links --budget 160   # follow transcript links in NSE cover letters (network)
    python -m scripts.v08_transcripts audit                # -> research/data_audits/transcripts_coverage_<date>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from intel.company_ir import (COLLECTORS, WebTranscript, _session, cover_letter_links, download, is_transcript_notice, load_index,
                              save_index, stamp)
from intel.transcripts import download_text, is_full_transcript, transcript_announcements
from marketdata.pit import default_universe

TR = Path("data/intel/transcripts")
WEB = Path("data/intel/transcripts_web")
YEARS = range(2022, 2027)
END = pd.Timestamp("2026-10-03")


def universe() -> list[str]:
    u = default_universe()
    return sorted(u.members(END))


def nse(budget: float) -> None:  # pragma: no cover - network
    from marketdata.nse_filings import Client

    c, t0 = Client(pause=0.3), time.time()
    for s in universe():
        idx, meta = TR / s / "index.json", TR / s / "fetched_years.json"
        idx.parent.mkdir(parents=True, exist_ok=True)
        items = json.loads(idx.read_text()) if idx.exists() else []
        done = set(json.loads(meta.read_text())) if meta.exists() else (set(range(2022, 2026)) if idx.exists() else set())
        for y in (range(2016, 2027) if s in COLLECTORS else YEARS):   # older intimations stamp website transcripts
            if y in done:
                continue
            if time.time() - t0 > budget:
                print("budget reached; run again")
                return
            b = pd.Timestamp(f"{y}-12-31") if y < END.year else END
            got = transcript_announcements(c, s.replace("&", "%26"), pd.Timestamp(f"{y}-01-01"), b)
            seen = {a.get("attchmntFile") for a in items}
            items += [a for a in got if a.get("attchmntFile") not in seen]
            done.add(y)
            idx.write_text(json.dumps(items, indent=0))
            meta.write_text(json.dumps(sorted(done)))
    for s in universe():
        for a in json.loads((TR / s / "index.json").read_text()):
            if time.time() - t0 > budget:
                print("budget reached; run again")
                return
            download_text(c, a, TR / s)
    print("nse transcripts complete")


def web(budget: float) -> None:  # pragma: no cover - network
    s, t0 = _session(), time.time()
    old = {w.url: w for w in load_index(WEB)}
    items = []
    for sym, fn in COLLECTORS.items():
        try:
            found = fn(s)
        except Exception as e:  # noqa: BLE001
            print("collector failed", sym, e)
            found = [w for w in old.values() if w.symbol == sym]
        nse_idx = json.loads((TR / sym / "index.json").read_text()) if (TR / sym / "index.json").exists() else []
        for w in found:
            if w.url in {x.url for x in items}:
                continue                                      # the same file listed twice on the site
            if w.url in old:                                  # keep first-seen time and local file
                p = old[w.url]
                w.first_seen_at, w.path, w.sha256 = p.first_seen_at, p.path, p.sha256
            if not w.path and time.time() - t0 < budget:
                w = download(s, w, WEB)
            items.append(stamp(w, nse_idx))
        print(sym, len(found), "listed;", sum(1 for w in items if w.symbol == sym and w.path), "downloaded", flush=True)
        save_index(WEB, items + [w for w in old.values() if w.url not in {x.url for x in items}])
    save_index(WEB, items + [w for w in old.values() if w.url not in {x.url for x in items}])   # keep cover-letter items


def links(budget: float) -> None:  # pragma: no cover - network
    """Link-only NSE filings: download the transcript the cover letter points to. PIT = the cover letter's own NSE
    dissemination time, so these are as point-in-time as the filing itself."""
    s, t0 = _session(), time.time()
    items = load_index(WEB)
    have = {w.url for w in items}
    scan_p = WEB / "scanned_cover_letters.json"
    scanned = set(json.loads(scan_p.read_text())) if scan_p.exists() else set()
    for sym in universe():
        idx = TR / sym / "index.json"
        if not idx.exists():
            continue
        for a in json.loads(idx.read_text()):
            if time.time() - t0 > budget:
                save_index(WEB, items)
                print("budget reached; run again")
                return
            if a["attchmntFile"] in scanned:
                continue
            pdf = TR / sym / Path(a["attchmntFile"]).name
            txt = pdf.with_suffix(".txt")
            if not txt.exists():
                continue
            text = txt.read_text(errors="ignore")
            if is_full_transcript(text):
                continue
            t = pd.to_datetime(a.get("exchdisstime") or a.get("an_dt"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
            if not is_transcript_notice(a):
                scanned.add(a["attchmntFile"])
                continue                                      # recording notices do not date a transcript
            for u in cover_letter_links(pdf, text):
                if u in have:
                    for w0 in items:                          # the earliest cover letter pointing to a file dates it
                        if w0.url == u and w0.intimation_url and pd.notna(t) and (not w0.available_at or
                                                                                  pd.Timestamp(w0.available_at) > t.tz_localize("Asia/Kolkata")):
                            w0.available_at, w0.intimation_url = t.tz_localize("Asia/Kolkata").isoformat(), a["attchmntFile"]
                    continue
                if time.time() - t0 > budget:
                    save_index(WEB, items)
                    print("budget reached; run again")
                    return
                w = WebTranscript(sym, u, Path(u).stem, "", "", available_at=t.tz_localize("Asia/Kolkata").isoformat() if pd.notna(t) else "",
                                  intimation_url=a["attchmntFile"])
                w = download(s, w, WEB)
                if w.path:
                    from intel.company_ir import call_date
                    tx = Path(w.path).with_suffix(".txt")
                    body = tx.read_text(errors="ignore") if tx.exists() else ""
                    w.full_text = is_full_transcript(body)
                    cd = call_date(body)
                    w.call_date = str(cd.date()) if cd is not None else ""
                items.append(w)
                have.add(u)
            scanned.add(a["attchmntFile"])
            scan_p.write_text(json.dumps(sorted(scanned)))
    save_index(WEB, items)
    print("links complete:", sum(1 for w in items if w.intimation_url and w.path and w.symbol not in COLLECTORS), "downloaded via cover letters")


def audit() -> dict:
    import hashlib

    from scripts.intel_v08 import transcript_docs

    rows = []
    webs = load_index(WEB)
    for s in universe():
        idx = TR / s / "index.json"
        ann = [a for a in (json.loads(idx.read_text()) if idx.exists() else [])
               if (pd.to_datetime(a.get("exchdisstime") or a.get("an_dt"), format="%d-%b-%Y %H:%M:%S", errors="coerce")
                   or pd.Timestamp("1900-01-01")) >= pd.Timestamp("2022-01-01")]
        notices = [a for a in ann if is_transcript_notice(a)]
        full, link_only = 0, 0
        for a in ann:
            t = TR / s / (Path(a["attchmntFile"]).stem + ".txt")
            if t.exists() and is_full_transcript(t.read_text(errors="ignore")):
                full += 1
            elif a in notices:
                link_only += 1
        w = [x for x in webs if x.symbol == s and x.path]
        docs = transcript_docs(s)
        calls = {hashlib.sha256(" ".join(text.split()).encode()).hexdigest() for d, text, _ in docs
                 if pd.Timestamp(d.published_at) >= pd.Timestamp("2022-01-01", tz="UTC")}
        rows.append({"symbol": s, "nse_announcements": len(ann), "nse_recording_notices": len(ann) - len(notices),
                     "nse_transcript_notices": len(notices), "nse_full": full, "nse_link_only_transcript_notices": link_only,
                     "web_downloaded": len(w), "web_full_with_exchange_timestamp": sum(1 for x in w if x.full_text and x.available_at),
                     "web_full_first_seen_only": sum(1 for x in w if x.full_text and not x.available_at),
                     "usable_pit_transcripts": len(calls)})
    df = pd.DataFrame(rows)
    res = {"universe": "NIFTY 50 members on " + str(END.date()), "window": "2022-01-01..2026-10-03 (all counts)",
           "companies": int(len(df)), "companies_with_any_transcript_notice": int((df.nse_transcript_notices > 0).sum()),
           "nse_announcements": int(df.nse_announcements.sum()), "nse_recording_notices": int(df.nse_recording_notices.sum()),
           "nse_transcript_notices": int(df.nse_transcript_notices.sum()), "nse_full": int(df.nse_full.sum()),
           "nse_link_only_transcript_notices": int(df.nse_link_only_transcript_notices.sum()),
           "web_downloaded_all_years": int(df.web_downloaded.sum()),
           "web_full_with_exchange_timestamp_all_years": int(df.web_full_with_exchange_timestamp.sum()),
           "web_full_first_seen_only_all_years": int(df.web_full_first_seen_only.sum()),
           "usable_pit_transcripts": int(df.usable_pit_transcripts.sum()),
           "usable_definition": "distinct transcript texts in Company Memory published 2022-01-01 or later: NSE full-text filings plus "
                                "website copies with an exchange transcript-notice timestamp not within 21 days of an NSE full text",
           "gap_companies_under_half_usable": df[df.usable_pit_transcripts < 0.5 * df.nse_transcript_notices].symbol.tolist(),
           "per_company": df.to_dict("records"), "made_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "pit_rule": "NSE filings from NSE dissemination time. Website copies: the earliest NSE *transcript* notice (recording "
                       "notices excluded) from the printed call date to +21 days, or, for undated transcripts, from quarter end to +75 "
                       "days with the notice's quarter tag required to match; cover-letter links take their cover letter's time; "
                       "otherwise first-seen only"}
    p = Path("research/data_audits") / f"transcripts_coverage_{time.strftime('%Y-%m-%d')}.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_company"}, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["nse", "web", "links", "audit"])
    ap.add_argument("--budget", type=float, default=160)
    a = ap.parse_args()
    {"nse": lambda: nse(a.budget), "web": lambda: web(a.budget), "links": lambda: links(a.budget), "audit": audit}[a.cmd]()


if __name__ == "__main__":
    main()
