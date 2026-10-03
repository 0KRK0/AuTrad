import { lazy, Suspense, useCallback, useState } from "react";
import { setToken } from "./api";
import { SkeletonTiles } from "./components/Skeleton";

// Route-level code splitting: each page is its own chunk, loaded on first visit.
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Orders = lazy(() => import("./pages/Orders"));
const Evidence = lazy(() => import("./pages/Evidence"));
const Agents = lazy(() => import("./pages/Agents"));

const TABS = { dashboard: "Control plane", agents: "Agents & override", orders: "Orders", evidence: "Evidence" } as const;
type Tab = keyof typeof TABS;

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [authed, setAuthed] = useState(false);
  const onToken = useCallback((e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setToken(String(new FormData(e.currentTarget).get("token") ?? ""));
    setAuthed(true);
  }, []);

  return (
    <div className="shell">
      <header>
        <strong>AuTrad AI</strong>
        <span className="mode">PAPER</span>
        {authed && (
          <nav>
            {(Object.keys(TABS) as Tab[]).map((t) => (
              <button key={t} className={t === tab ? "on" : ""} onClick={() => setTab(t)}>
                {TABS[t]}
              </button>
            ))}
          </nav>
        )}
      </header>
      <main>
        {!authed ? (
          <form className="login" onSubmit={onToken}>
            <label>
              API token <input name="token" type="password" autoComplete="off" required />
            </label>
            <button>Continue</button>
          </form>
        ) : (
          <Suspense fallback={<SkeletonTiles n={4} />}>
            {tab === "dashboard" && <Dashboard />}
            {tab === "agents" && <Agents />}
            {tab === "orders" && <Orders />}
            {tab === "evidence" && <Evidence />}
          </Suspense>
        )}
      </main>
      <footer>Research / paper-trading platform — not a regulated investment service. Past performance does not predict future results.</footer>
    </div>
  );
}
