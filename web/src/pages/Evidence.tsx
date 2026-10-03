import { useEffect, useState } from "react";
import { api } from "../api";
import { Skeleton } from "../components/Skeleton";
import { useDebounce } from "../hooks/useDebounce";

interface Ev { seq: number; source: string; url: string; title: string; sentiment: number; symbols: string[]; ingested_at: string }

export default function Evidence() {
  const [q, setQ] = useState("");
  const symbol = useDebounce(q.trim().toUpperCase(), 300); // one request after typing pauses
  const [items, setItems] = useState<Ev[] | null>(null);

  useEffect(() => {
    let live = true;
    setItems(null);
    const qs = symbol ? `?symbol=${encodeURIComponent(symbol)}` : "";
    api<{ items: Ev[] }>(`/api/v1/evidence${qs}`, {}, 15_000).then((r) => live && setItems(r.items)).catch(() => live && setItems([]));
    return () => { live = false; };
  }, [symbol]);

  return (
    <section>
      <input placeholder="Filter by symbol, e.g. AAPL" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Symbol filter" />
      {items === null ? (
        <Skeleton h={160} />
      ) : items.length === 0 ? (
        <p className="note">No evidence yet. Start the collector to ingest filings, central-bank and news feeds.</p>
      ) : (
        <ul className="evidence">
          {items.map((e) => (
            <li key={e.seq}>
              <span className={e.sentiment >= 0 ? "good" : "bad"}>{e.sentiment.toFixed(2)}</span>{" "}
              <a href={e.url} target="_blank" rel="noopener noreferrer">{e.title}</a>
              <small> · {e.source} · {e.symbols.join(", ")} · {e.ingested_at?.slice(0, 16)}</small>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
