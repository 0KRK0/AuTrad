import { memo } from "react";

export const Stat = memo(function Stat({ label, value, tone }: { label: string; value: string; tone?: "good" | "bad" }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className={`value ${tone ?? ""}`}>{value}</div>
    </div>
  );
});

export const pct = (x: number | undefined, d = 1) => (x === undefined || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`);
