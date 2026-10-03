// Tiny fetch client with in-memory response cache + in-flight de-duplication.
const cache = new Map<string, { at: number; data: unknown }>();
const inflight = new Map<string, Promise<unknown>>();
let token = "";

export const setToken = (t: string) => {
  token = t;
  cache.clear();
};

export async function api<T>(path: string, init: RequestInit = {}, ttlMs = 0): Promise<T> {
  const key = `${init.method ?? "GET"} ${path} ${init.body ?? ""}`;
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < ttlMs) return hit.data as T;
  const pending = inflight.get(key);
  if (pending) return pending as Promise<T>;
  const p = fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}`, ...init.headers },
  })
    .then(async (r) => {
      if (!r.ok) throw new Error(`${r.status} ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`);
      return r.json();
    })
    .then((data) => {
      if (ttlMs > 0) cache.set(key, { at: Date.now(), data });
      return data;
    })
    .finally(() => inflight.delete(key));
  inflight.set(key, p);
  return p as Promise<T>;
}

export type Metrics = Record<string, number>;
export interface Backtest {
  metrics: Metrics;
  benchmark: Metrics;
  walk_forward: ({ start: string; end: string } & Metrics)[];
  equity: [string, number][];
  orders: number;
  kill_switch: { active: boolean; reason: string };
  elapsed_ms: number;
  data_source: string;
}
export interface Order {
  id: number;
  symbol: string;
  qty: number;
  status: string;
  created_at: string;
  fills: { qty: number; price: number }[];
}
