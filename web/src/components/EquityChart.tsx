import { memo, useMemo } from "react";

// Lazy-loaded chunk: plain SVG, no chart library shipped.
function EquityChart({ points }: { points: [string, number][] }) {
  const { d, min, max, first, last } = useMemo(() => {
    const ys = points.map((p) => p[1]);
    const min = Math.min(...ys), max = Math.max(...ys);
    const W = 720, H = 220;
    const d = points
      .map((p, i) => `${i ? "L" : "M"}${((i / (points.length - 1)) * W).toFixed(1)} ${(H - ((p[1] - min) / (max - min || 1)) * H).toFixed(1)}`)
      .join("");
    return { d, min, max, first: points[0]?.[0], last: points.at(-1)?.[0] };
  }, [points]);
  return (
    <figure className="chart">
      <svg viewBox="0 0 720 220" role="img" aria-label="Equity curve">
        <path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption>
        {first?.slice(0, 10)} → {last?.slice(0, 10)} · range £{min.toFixed(0)}–£{max.toFixed(0)}
      </figcaption>
    </figure>
  );
}
export default memo(EquityChart);
