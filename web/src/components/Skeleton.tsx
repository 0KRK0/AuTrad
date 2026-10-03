import { memo } from "react";

export const Skeleton = memo(function Skeleton({ h = 16, w = "100%" }: { h?: number; w?: number | string }) {
  return <div className="skeleton" style={{ height: h, width: w }} aria-hidden="true" />;
});

export const SkeletonTiles = memo(function SkeletonTiles({ n = 4 }: { n?: number }) {
  return (
    <div className="tiles" aria-busy="true" aria-label="Loading">
      {Array.from({ length: n }, (_, i) => (
        <div className="tile" key={i}>
          <Skeleton h={12} w="50%" />
          <Skeleton h={28} w="70%" />
        </div>
      ))}
    </div>
  );
});
