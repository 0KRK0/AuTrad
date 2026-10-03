import { memo, useCallback, useEffect, useState } from "react";
import { api, type Order } from "../api";
import { Skeleton } from "../components/Skeleton";

const Row = memo(function Row({ o }: { o: Order }) {
  const px = o.fills.length ? o.fills.reduce((s, f) => s + f.price * Math.abs(f.qty), 0) / o.fills.reduce((s, f) => s + Math.abs(f.qty), 0) : 0;
  return (
    <tr>
      <td>{o.id}</td><td>{o.symbol}</td><td>{o.qty.toFixed(2)}</td><td>{o.status}</td>
      <td>{px ? px.toFixed(2) : "—"}</td><td>{o.created_at?.slice(0, 19)}</td>
    </tr>
  );
});

export default function Orders() {
  const [items, setItems] = useState<Order[]>([]);
  const [cursor, setCursor] = useState<number | null | undefined>(undefined);
  const [loading, setLoading] = useState(false);

  const more = useCallback(async (c?: number | null) => {
    setLoading(true);
    const q = c ? `&cursor=${c}` : "";
    const page = await api<{ items: Order[]; next_cursor: number | null }>(`/api/v1/orders?limit=50${q}`);
    setItems((prev) => (c ? [...prev, ...page.items] : page.items));
    setCursor(page.next_cursor);
    setLoading(false);
  }, []);

  useEffect(() => void more(), [more]);

  return (
    <section>
      <table>
        <thead><tr><th>ID</th><th>Symbol</th><th>Qty</th><th>Status</th><th>Avg px</th><th>Created</th></tr></thead>
        <tbody>{items.map((o) => <Row key={o.id} o={o} />)}</tbody>
      </table>
      {loading && <Skeleton h={120} />}
      {!loading && items.length === 0 && <p className="note">No orders yet. Paper trading writes orders here.</p>}
      {cursor && !loading && <button onClick={() => more(cursor)}>Load more</button>}
    </section>
  );
}
