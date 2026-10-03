import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { Skeleton } from "../components/Skeleton";
import { pct } from "../components/Stat";

interface Agent { category: string; style: string; speculative: boolean; paused: boolean; max_category: number; max_position: number; stop_loss: number; time_stop_bars: number }
interface Ov { id: number; action: string; symbol: string | null; category: string | null; weight: number; user: string; reason: string; consumed: boolean }

const ACTIONS = ["BUY", "SELL", "TARGET", "FLATTEN", "PIN", "PAUSE_CATEGORY", "RESUME_CATEGORY"] as const;

export default function Agents() {
  const [agents, setAgents] = useState<Agent[] | null>(null);
  const [ovs, setOvs] = useState<Ov[]>([]);
  const [msg, setMsg] = useState("");

  const load = useCallback(async () => {
    const [a, o] = await Promise.all([
      api<{ agents: Agent[] }>("/api/v1/agents"),
      api<{ items: Ov[] }>("/api/v1/overrides"),
    ]);
    setAgents(a.agents);
    setOvs(o.items.slice().reverse());
  }, []);
  useEffect(() => void load().catch((e) => setMsg(String(e))), [load]);

  const submit = useCallback(async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    const action = String(f.get("action"));
    const target = String(f.get("target") ?? "").trim();
    const isCat = action.endsWith("_CATEGORY");
    const body = {
      action,
      reason: String(f.get("reason")),
      weight: Number(f.get("weight") || 0) / 100,
      ...(isCat ? { category: target } : { symbol: target.toUpperCase() }),
    };
    try {
      await api("/api/v1/overrides", { method: "POST", body: JSON.stringify(body) });
      setMsg(`${action} queued`);
      e.currentTarget.reset();
      load();
    } catch (err) {
      setMsg(String(err));
    }
  }, [load]);

  return (
    <section>
      <h3>Your override</h3>
      <form className="override" onSubmit={submit}>
        <select name="action" aria-label="Action">{ACTIONS.map((a) => <option key={a}>{a}</option>)}</select>
        <input name="target" placeholder="Symbol (DOGEINR) or category (crypto_meme)" required aria-label="Symbol or category" />
        <input name="weight" type="number" step="0.1" min="-100" max="100" placeholder="% of equity" aria-label="Weight percent" />
        <input name="reason" placeholder="Why (goes in the audit log)" minLength={3} required aria-label="Reason" />
        <button>Send</button>
      </form>
      <p className="note">Overrides skip the AI's soft brakes, but the kill switch, gross exposure and your 10% per-symbol cap still apply.</p>
      {msg && <p role="status">{msg}</p>}

      <h3>Category agents</h3>
      {agents === null ? <Skeleton h={160} /> : (
        <table>
          <thead><tr><th>Category</th><th>Style</th><th>Budget</th><th>Per symbol</th><th>Stop</th><th>Time stop</th><th>State</th></tr></thead>
          <tbody>{agents.map((a) => (
            <tr key={a.category}>
              <td>{a.category}{a.speculative ? " ⚡" : ""}</td><td>{a.style}</td><td>{pct(a.max_category)}</td>
              <td>{pct(a.max_position)}</td><td>{pct(a.stop_loss)}</td><td>{a.time_stop_bars} bars</td>
              <td className={a.paused ? "bad" : "good"}>{a.paused ? "paused" : "active"}</td>
            </tr>
          ))}</tbody>
        </table>
      )}

      <h3>Recent overrides</h3>
      {ovs.length === 0 ? <p className="note">None yet.</p> : (
        <ul className="evidence">{ovs.map((o) => (
          <li key={o.id}>#{o.id} {o.action} {o.symbol ?? o.category} {o.weight ? pct(o.weight) : ""} · {o.user} · {o.reason} {o.consumed ? "· executed" : ""}</li>
        ))}</ul>
      )}
    </section>
  );
}
