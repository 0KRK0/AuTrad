"""Company Intelligence agent: understands what a company is and what is happening around it, as of a date, with
every fact traced to a document in the Evidence Ledger. It never trades and never talks to other agents; it answers
the Main Head with a Dossier.

Sections: identity, financial trajectory (PIT filings), results timeline and market reactions, exchange
announcements and scheduled board meetings (when fetched), news headlines (opt-in), management-claim record,
red flags (inferences that cite their evidence), and what is unknown. Facts quote their documents verbatim;
computations quote their inputs (DERIVED); interpretations are INFERENCE; AuTrad's own labels are LABEL.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from intel.claims import ClaimTracker
from intel.evidence import Document, EvidenceLedger, Statement, render
from intel.sources import dataset_extract, filing_documents, filing_quotes, pit_document

from .context import MarketContext


@dataclass
class Dossier:
    symbol: str
    asof: str
    sections: dict[str, list[Statement]] = field(default_factory=dict)
    unknowns: list[str] = field(default_factory=list)
    upcoming: list[Statement] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def statements(self) -> list[Statement]:
        return [s for v in self.sections.values() for s in v] + self.upcoming

    def provenance_coverage(self) -> float:
        """Share of facts and computations that resolve to quoted evidence (labels and inferences excluded)."""
        facts = [s for s in self.statements() if s.kind in ("fact", "derived")]
        return sum(s.status in ("SOURCED", "CORROBORATED", "DERIVED") for s in facts) / len(facts) if facts else 1.0

    def markdown(self, ledger: EvidenceLedger) -> str:
        L = [f"### {self.symbol} — dossier as of {self.asof[:10]}", ""]
        for k, v in self.sections.items():
            if v:
                L.append(f"**{k}**")
                L += [f"- {render(s, ledger)}" for s in v]
                L.append("")
        if self.upcoming:
            L.append("**Scheduled**")
            L += [f"- {render(s, ledger)}" for s in self.upcoming]
            L.append("")
        if self.unknowns:
            L.append("**Unknown / not available**")
            L += [f"- {u}" for u in self.unknowns]
        return "\n".join(L) + "\n"


def asof_instant(asof: pd.Timestamp) -> str:
    return (pd.Timestamp(asof).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)).isoformat()


class CompanyIntelligenceAgent:
    def __init__(self, ctx: MarketContext, ledger: EvidenceLedger, claims: ClaimTracker | None = None,
                 announcements: list[Document] | None = None, news: list[Document] | None = None) -> None:
        self.ctx, self.ev, self.claims = ctx, ledger, claims
        for d in (announcements or []) + (news or []):
            self.ev.add(d)
        self.fetched_announcements = announcements is not None
        self.fetched_news = news is not None

    def _st(self, text: str, quotes: dict, asof: str, kind: str = "fact") -> Statement:
        return self.ev.resolve(Statement(text, list(quotes), quotes, kind), asof)

    def dossier(self, symbol: str, asof: pd.Timestamp) -> Dossier:
        ctx = self.ctx
        a_iso = asof_instant(asof)
        D = Dossier(symbol, a_iso)
        t = ctx.idx(asof)
        j = ctx.P.idx.get(symbol)
        C = ctx.close()
        ident: list[Statement] = []
        # identity: name and index membership from the PIT reference file (third-party reconstruction, tier 6)
        rows = [r for r in ctx.pit_rows.get(symbol, []) if pd.Timestamp(r[0]) <= pd.Timestamp(asof)]
        if rows:
            eff, row = rows[-1]
            if True:
                d = self.ev.add(pit_document(symbol, eff, row))
                ident.append(self._st(f"{ctx.company_names.get(symbol, symbol)} is a NIFTY 50 constituent (snapshot effective {pd.Timestamp(eff).date()}).",
                                      {d: row.split(",")[1]}, a_iso))
        if j is not None:
            ident.append(Statement(f"AuTrad sector bucket: {ctx.P.sector[j]}.", kind="label", status="LABEL"))
            if t >= 252 and np.isfinite(C[t, j]):
                vals = {f"adjusted close {ctx.P.dates[t].date()}": float(C[t, j]), f"adjusted close {ctx.P.dates[t - 21].date()}": float(C[t - 21, j]),
                        f"adjusted close {ctx.P.dates[t - 252].date()}": float(C[t - 252, j])}
                x = self.ev.add(dataset_extract(symbol, ctx.P.dates[t], vals, ctx.dataset_hash))
                qs = [f"{k}: {v:.6g}" for k, v in vals.items()]
                ident.append(self._st(f"Last-month return {C[t, j] / C[t - 21, j] - 1:+.1%}; 12-month return {C[t, j] / C[t - 252, j] - 1:+.1%}.",
                                      {x: qs}, a_iso, "derived"))
        D.sections["Identity and price"] = ident
        # financial trajectory from filings (tier 1)
        f = ctx.filings(symbol, asof)
        fdocs = {d.url: d for d in filing_documents(f, symbol)}
        for d in fdocs.values():
            self.ev.add(d)
        fin: list[Statement] = []
        num: dict[str, tuple[float, dict]] = {}
        if len(f):
            last = f.iloc[-1]
            dl = fdocs[str(last["xbrl"])]
            age = (pd.Timestamp(asof) - pd.Timestamp(last["period_end"])).days
            fin.append(self._st(f"Latest filed quarter ended {pd.Timestamp(last['period_end']).date()} ({age} days before the as-of date).",
                                {dl.doc_id: filing_quotes(dl)["quarter ended"]}, a_iso))
            prev = f[pd.to_datetime(f["period_end"]) == pd.Timestamp(last["period_end"]) - pd.DateOffset(years=1)]
            if len(prev):
                p = prev.iloc[-1]
                dp = fdocs[str(p["xbrl"])]
                ql, qp = filing_quotes(dl), filing_quotes(dp)
                for col, key, name in (("revenue", "revenue", "Revenue"), ("net_profit", "net profit", "Net profit")):
                    a_, b_ = last[col], p[col]
                    if pd.notna(a_) and pd.notna(b_) and b_ > 0 and key in ql and key in qp:
                        q = {dl.doc_id: [ql[key], ql["quarter ended"]], dp.doc_id: [qp[key], qp["quarter ended"]]}
                        fin.append(self._st(f"{name} {a_ / b_ - 1:+.1%} year on year.", q, a_iso, "derived"))
                        num[col] = (a_ / b_ - 1, q)
                if all(k in ql and k in qp for k in ("revenue", "net profit")) and last["revenue"] > 0 and p["revenue"] > 0 \
                        and pd.notna(last["net_profit"]) and pd.notna(p["net_profit"]):
                    m1, m0 = last["net_profit"] / last["revenue"], p["net_profit"] / p["revenue"]
                    q = {dl.doc_id: [ql["revenue"], ql["net profit"]], dp.doc_id: [qp["revenue"], qp["net profit"]]}
                    fin.append(self._st(f"Net margin {m1:.1%} vs {m0:.1%} a year earlier ({(m1 - m0) * 100:+.1f} pp).", q, a_iso, "derived"))
                    num["margin_pp"] = ((m1 - m0) * 100, q)
            if age > 120:
                D.unknowns.append(f"No results filing in the last {age} days in AuTrad's store (data may be stale).")
        else:
            D.unknowns.append("No PIT results filings stored for this company.")
        D.sections["Financial trajectory (NSE filings)"] = fin
        # results timeline: last four reactions (derived from the filing time and exchange-anchored closes)
        ev = ctx.events[(ctx.events["symbol"] == symbol) & (ctx.events["reaction_idx"] <= t)].tail(4)
        tl, last_bad = [], None
        for r in ev.itertuples(index=False):
            i = int(r.reaction_idx)
            fd = f[pd.to_datetime(f["period_end"]) == pd.Timestamp(r.period_end)]
            q: dict = {}
            if len(fd):
                dd = fdocs[str(fd.iloc[0]["xbrl"])]
                q[dd.doc_id] = filing_quotes(dd)["quarter ended"]
            vals = {f"adjusted close {ctx.P.dates[i - 1].date()}": float(C[i - 1, j]), f"adjusted close {ctx.P.dates[i].date()}": float(C[i, j]),
                    f"NIFTY50 {ctx.P.dates[i - 1].date()}": float(ctx.P.bench.iloc[i - 1]), f"NIFTY50 {ctx.P.dates[i].date()}": float(ctx.P.bench.iloc[i])}
            x = self.ev.add(dataset_extract(symbol, ctx.P.dates[i], vals, ctx.dataset_hash))
            q[x] = [f"{k}: {v:.6g}" for k, v in vals.items()]
            bad = getattr(r, "bad", 0) == 1
            s_ = self._st(f"Market-adjusted reaction to quarter {pd.Timestamp(r.period_end).date()} results on {ctx.P.dates[i].date()}: {r.ear:+.1%}"
                          f"{' (bad: lower third of all earlier reactions)' if bad else ''}.", q, a_iso, "derived")
            tl.append(s_)
            last_bad = (s_, i) if bad else last_bad
        D.sections["Results timeline"] = tl
        # announcements / meetings (tier 1, quoted)
        if self.fetched_announcements:
            ann = [d for d in self.ev.as_of(symbol, a_iso) if d.source.startswith("NSE corporate")][-8:]
            D.sections["Exchange announcements (latest)"] = [self._st(f"{d.published_at[:10]} {d.title}: {d.text[:160]}", {d.doc_id: d.text[:160]}, a_iso)
                                                             for d in ann if d.text]
            for d in self.ev.as_of(symbol, a_iso):
                if d.source == "NSE board meetings":
                    try:
                        md = pd.Timestamp(d.title.split(":")[0].replace("Board meeting ", ""))
                    except (ValueError, TypeError):
                        continue
                    if md >= pd.Timestamp(asof) and d.text:
                        D.upcoming.append(self._st(f"Board meeting on {md.date()}: {d.text[:200]}", {d.doc_id: d.text[:200]}, a_iso))
        else:
            D.unknowns.append("Exchange announcements not fetched in this run.")
        if self.fetched_news:
            nw = [d for d in self.ev.as_of(symbol, a_iso) if d.source.startswith("news:")][-6:]
            D.sections["News headlines (claims by others; the headline exists, its content is not verified)"] = [
                self._st(f"{d.source[6:]} ({d.published_at[:10]}): \"{d.text}\" [tier {d.tier}]", {d.doc_id: d.text}, a_iso) for d in nw]
        else:
            D.unknowns.append("News not fetched (opt-in).")
        if self.claims is not None:
            rel = self.claims.reliability(symbol, asof=a_iso)
            if rel["claims"]:
                D.sections["Management claims record"] = [self.ev.resolve(Statement(
                    f"{rel['claims']} tracked claims: {rel['met']} met, {rel['missed']} missed, {rel['pending']} pending; "
                    f"reliability {rel['reliability']:.0%} (Beta posterior, {rel['evidence_weight']} resolved).", rel["evidence"], {}, "inference"), a_iso)]
        D.unknowns.append("No analyst consensus (true earnings surprise unknown); no promoter-pledge or insider-trade feed yet.")
        from intel.sources import XOfficialAPI, official_channels
        if not XOfficialAPI().enabled or official_channels().query("symbol == @symbol").empty:
            D.unknowns.append("No verified official social channel configured for this company.")
        # red flags: interpretations that cite the evidence they rest on
        flags = []
        if "net_profit" in num and num["net_profit"][0] < -0.2:
            flags.append(Statement(f"Profit falling sharply ({num['net_profit'][0]:+.0%} YoY).", list(num["net_profit"][1]), {}, "inference"))
        if "revenue" in num and num["revenue"][0] < 0:
            flags.append(Statement(f"Revenue shrinking ({num['revenue'][0]:+.1%} YoY).", list(num["revenue"][1]), {}, "inference"))
        if "margin_pp" in num and num["margin_pp"][0] < -3:
            flags.append(Statement(f"Margin compression of {num['margin_pp'][0]:.1f} pp.", list(num["margin_pp"][1]), {}, "inference"))
        if last_bad is not None and t - last_bad[1] <= 60:
            flags.append(Statement("Latest results drew a bad market reaction within the last 60 sessions (drift risk).", last_bad[0].evidence, {}, "inference"))
        D.sections["Red flags"] = [self.ev.resolve(x, a_iso) for x in flags]
        D.metrics = {k: v[0] for k, v in num.items()}
        return D
