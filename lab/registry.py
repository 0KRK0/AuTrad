"""Append-only research trial registry. Every configuration ever evaluated is counted, so the
Deflated Sharpe Ratio charges for all of our searching, not just the current run's grid."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT = Path("research") / "trials.jsonl"

EXPERIMENT_FIELDS = [
    "experiment_id", "git_sha", "dataset_version", "data_hash", "exchange_hash", "transforms_version",
    "universe_definition", "point_in_time", "training_period", "validation_period", "test_period", "features",
    "model", "hyperparameters", "configs_tried", "cost_model", "slippage_model", "turnover", "capacity",
    "gross_return", "net_return", "sharpe", "sortino", "max_drawdown", "calmar", "tail_loss", "hit_rate",
    "p_value", "deflated_sharpe", "pbo", "number_of_prior_trials", "decision", "post_test_peek",
]


def git_sha() -> str:
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               timeout=5).stdout.strip()
        return (sha + ("+dirty" if dirty else "")) if sha else "no-git"
    except Exception:  # noqa: BLE001
        return "no-git"

V02_SEED = [
    {"version": "v0.2", "strategy": "argo_agents", "config": c, "note": "v0.2 walk-forward grid"}
    for c in ["daily", "h5_weekly", "h5_weekly_strict", "h20_weekly", "h20_monthly", "h20_monthly_strict",
              "explore_daily_full_sample", "explore_h20_full_sample"]
]


class Registry:
    def __init__(self, path: str | Path = DEFAULT) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w") as f:
                for r in V02_SEED:
                    f.write(json.dumps({"ts": "2026-09-30T22:45:00Z", **r}) + "\n")

    def count(self) -> int:
        """Distinct (strategy, config) pairs ever evaluated. Re-running the same grid is not a new trial.
        Experiment summaries (kind == "experiment") are not trials."""
        seen = set()
        for line in self.path.open():
            if line.strip():
                r = json.loads(line)
                if r.get("kind") in ("experiment", "preregistration", "note"):
                    continue
                seen.add((r.get("strategy"), json.dumps(r.get("config"), sort_keys=True, default=str)))
        return len(seen)

    def experiments(self) -> list[dict]:
        return [r for r in (json.loads(x) for x in self.path.open() if x.strip()) if r.get("kind") == "experiment"]

    SIX_QUESTIONS = ("economic_counterparty", "mechanism_constraint", "our_advantage", "failure_condition")

    @staticmethod
    def _version(r: dict) -> tuple[int, int] | None:
        try:
            major, minor = (int(x) for x in str(r["version"]).lstrip("v").split(".")[:2])
            return major, minor
        except (KeyError, ValueError):
            return None

    @classmethod
    def _needs_mechanism(cls, r: dict) -> bool:
        """From v1.1 every experiment row must state its economic_mechanism (why the edge should exist)."""
        if r.get("kind") != "experiment":
            return False
        v = cls._version(r)                     # required unless the version is explicitly below v1.1
        return v is None or v >= (1, 1)

    @classmethod
    def _needs_six(cls, r: dict) -> bool:
        """From v1.2 the row must also name who pays, the constraint, AuTrad's advantage and what would kill it."""
        if r.get("kind") != "experiment":
            return False
        v = cls._version(r)
        return v is None or v >= (1, 2)

    def add(self, records: list[dict]) -> None:
        for r in records:
            if self._needs_mechanism(r) and not str(r.get("economic_mechanism", "")).strip():
                raise ValueError(f"experiment {r.get('experiment_id')}: economic_mechanism is required from v1.1")
            if self._needs_six(r):
                missing = [k for k in self.SIX_QUESTIONS if not str(r.get(k, "")).strip()]
                if missing:
                    raise ValueError(f"experiment {r.get('experiment_id')}: {', '.join(missing)} required from v1.2")
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.path.open("a") as f:
            for r in records:
                f.write(json.dumps({"ts": ts, **r}, default=str) + "\n")
