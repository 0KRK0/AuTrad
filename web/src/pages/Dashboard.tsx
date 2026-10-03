import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { api, type Backtest } from "../api";
import { Skeleton, SkeletonTiles } from "../components/Skeleton";
import { pct, Stat } from "../components/Stat";

const EquityChart = lazy(() => import("../components/EquityChart"));

export default function Dashboard() {
  const [bt, setBt] = useState<Backtest | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [kill, setKill] = useState<{ active: boolean; reason: string } | null>(null);

  const loadKill = useCallback(() => api<{ active: boolean; reason: string }>("/api/v1/killswitch").then(setKill).catch((e) => setErr(String(e))), []);
  useEffect(() => void loadKill(), [loadKill]);

  const run = useCallback(async () => {
    setBusy(true);
    setErr("");
    try {
      setBt(await api<Backtest>("/api/v1/backtests", { method: "POST", body: JSON.stringify({}) }, 60_000));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const trip = useCallback(async () => {
    if (!confirm("Trip the kill switch? All trading stops until a risk officer resets it.")) return;
    await api("/api/v1/killswitch/trip", { method: "POST", body: JSON.stringify({ reason: "manual" }) }).catch((e) => setErr(String(e)));
    loadKill();
  }, [loadKill]);

  const m = bt?.metrics;
  return (
    <section>
      <div className="row">
        <div className={`status ${kill?.active ? "bad" : "good"}`}>
          Kill switch: {kill === null ? "…" : kill.active ? `ACTIVE (${kill.reason})` : "armed"}
        </div>
        <button className="danger" onClick={trip}>Kill switch</button>
        <button onClick={run} disabled={busy}>{busy ? "Running…" : "Run backtest"}</button>
      </div>
      {err && <p className="error" role="alert">{err}</p>}
      {busy && !bt && <SkeletonTiles n={6} />}
      {m && bt && (
        <>
          <p className="note">{bt.data_source} · {bt.orders} orders · {bt.elapsed_ms} ms</p>
          <div className="tiles">
            <Stat label="Total return" value={pct(m.total_return)} tone={m.total_return >= 0 ? "good" : "bad"} />
            <Stat label="Benchmark" value={pct(bt.benchmark.total_return)} />
            <Stat label="Sharpe" value={m.sharpe.toFixed(2)} />
            <Stat label="Max drawdown" value={pct(m.max_drawdown)} tone="bad" />
            <Stat label="Benchmark max DD" value={pct(bt.benchmark.max_drawdown)} />
            <Stat label="CVaR 95% (daily)" value={pct(m.cvar_95, 2)} />
          </div>
          <Suspense fallback={<Skeleton h={220} />}>
            <EquityChart points={bt.equity} />
          </Suspense>
          <h3>Walk-forward folds</h3>
          <table>
            <thead><tr><th>Period</th><th>Return</th><th>Sharpe</th><th>Max DD</th></tr></thead>
            <tbody>
              {bt.walk_forward.map((f) => (
                <tr key={f.start}>
                  <td>{f.start} → {f.end}</td>
                  <td className={f.total_return >= 0 ? "good" : "bad"}>{pct(f.total_return)}</td>
                  <td>{f.sharpe.toFixed(2)}</td>
                  <td>{pct(f.max_drawdown)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
