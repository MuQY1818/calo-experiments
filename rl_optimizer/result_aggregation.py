"""Deterministic aggregation for completed CALO experiment results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .algorithm_executor import algorithm_display_name
from .durable_io import atomic_write_text
from .experiment_config import display_project_path, resolve_project_path
from .result_store import (
    FINAL_FILENAME,
    PROGRESS_FILENAME,
    ResultStoreError,
    load_json,
    write_json,
)


AGGREGATE_JSON = "aggregate_summary.json"
AGGREGATE_MARKDOWN = "aggregate_summary.md"
DEFAULT_AGGREGATE_DIR = Path("results/dynamic_experiment_aggregate")


class AggregationError(RuntimeError):
    """Indicate invalid or mutually incompatible aggregate inputs."""


@dataclass(frozen=True)
class AggregateOutputs:
    """Paths written by one aggregation operation."""

    json_path: Path
    markdown_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class _ResultSource:
    """One validated completed result file."""

    label: str
    path: Path
    results: dict[str, Any]
    config_fingerprint: str
    calibration_identity: dict[str, Any]
    seed: int
    diagnostic_calibration_disabled: bool


def relative_improvement(candidate: float, baseline: float) -> float:
    """Return percentage improvement with the historical zero convention."""
    baseline_abs = abs(float(baseline))
    if baseline_abs < 1e-12:
        return 0.0 if abs(float(candidate)) < 1e-12 else float("inf")
    return (float(candidate) - float(baseline)) / baseline_abs * 100.0


def _is_small_baseline(baseline: float, threshold: float = 0.05) -> bool:
    """Return whether percentage changes are unstable for a baseline."""
    return abs(float(baseline)) < threshold


def _summarize_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-seed comparisons using the accepted reporting rules."""
    if not comparisons:
        return {
            "total_comparisons": 0,
            "primary_wins": 0,
            "win_rate_pct": 0.0,
            "mean_raw_reward_delta": 0.0,
            "median_relative_improvement_pct": 0.0,
            "best_raw_reward_delta": 0.0,
            "worst_raw_reward_delta": 0.0,
            "small_baseline_cases": 0,
        }
    raw_deltas = np.asarray([row["raw_reward_delta"] for row in comparisons], dtype=float)
    finite_relative = np.asarray(
        [
            row["relative_improvement_pct"]
            for row in comparisons
            if np.isfinite(row["relative_improvement_pct"])
        ],
        dtype=float,
    )
    primary_wins = sum(bool(row["primary_win"]) for row in comparisons)
    total = len(comparisons)
    return {
        "total_comparisons": total,
        "primary_wins": primary_wins,
        "win_rate_pct": primary_wins / total * 100.0,
        "mean_raw_reward_delta": float(np.mean(raw_deltas)),
        "median_relative_improvement_pct": (
            float(np.median(finite_relative)) if finite_relative.size else 0.0
        ),
        "best_raw_reward_delta": float(np.max(raw_deltas)),
        "worst_raw_reward_delta": float(np.min(raw_deltas)),
        "small_baseline_cases": sum(bool(row["small_baseline"]) for row in comparisons),
    }


class ResultAggregator:
    """Validate completed runs and write aggregate JSON and Markdown."""

    def aggregate(
        self,
        result_refs: Iterable[str | Path],
        output_dir: str | Path = DEFAULT_AGGREGATE_DIR,
    ) -> AggregateOutputs:
        """Aggregate one or more completed result files.

        Args:
            result_refs: Result JSON paths or directories containing the final JSON.
            output_dir: Repository-root-relative or absolute output directory.

        Returns:
            Paths and structured payload written by this operation.

        Raises:
            AggregationError: If inputs are absent, malformed, or incompatible.
        """
        sources = [self._load_source(value) for value in result_refs]
        if not sources:
            raise AggregationError("At least one completed result is required.")
        algorithms, cases = self._validate_compatible(sources)
        primary = "ppo" if "ppo" in algorithms else algorithms[0]
        baselines = [algorithm for algorithm in algorithms if algorithm != primary]
        payload: dict[str, Any] = {
            "primary_algorithm": primary,
            "primary_display_name": algorithm_display_name(primary),
            "config_fingerprint": sources[0].config_fingerprint,
            "calibration_identity": sources[0].calibration_identity,
            "diagnostic_calibration_disabled": sources[0].diagnostic_calibration_disabled,
            "seeds": [source.seed for source in sources],
            "sources": [
                {
                    "label": source.label,
                    "path": display_project_path(source.path),
                    "seed": source.seed,
                }
                for source in sources
            ],
        }
        if baselines:
            payload["baselines"] = {
                baseline: self._aggregate_baseline(
                    sources, cases, primary=primary, baseline=baseline
                )
                for baseline in baselines
            }
        else:
            payload["single_algorithm"] = self._aggregate_single(sources, cases, primary)

        output_path = resolve_project_path(output_dir)
        json_path = output_path / AGGREGATE_JSON
        markdown_path = output_path / AGGREGATE_MARKDOWN
        write_json(json_path, payload)
        atomic_write_text(markdown_path, self._render_markdown(payload))
        return AggregateOutputs(json_path, markdown_path, payload)

    @staticmethod
    def _load_source(result_ref: str | Path) -> _ResultSource:
        """Resolve and load one result reference."""
        path = resolve_project_path(result_ref)
        if path.is_dir():
            path = path / FINAL_FILENAME
        if not path.is_file():
            raise AggregationError(f"Completed result not found: {display_project_path(path)}")
        progress_path = path.parent / PROGRESS_FILENAME
        if not progress_path.is_file():
            raise AggregationError(
                f"Aggregate source requires {PROGRESS_FILENAME}: "
                f"{display_project_path(path.parent)}"
            )
        try:
            raw = load_json(path)
            progress = load_json(progress_path)
        except ResultStoreError as exc:
            raise AggregationError(str(exc)) from exc
        if not isinstance(raw, dict) or not raw:
            raise AggregationError(
                f"Result must be a non-empty object: {display_project_path(path)}"
            )
        if not isinstance(progress, dict) or progress.get("status") != "completed":
            raise AggregationError(
                f"Aggregate source is not marked completed: {display_project_path(progress_path)}"
            )
        if progress.get("results_path") != path.name:
            raise AggregationError(
                f"Progress/result path mismatch: {display_project_path(progress_path)}"
            )
        fingerprint = progress.get("config_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise AggregationError(
                f"Missing config fingerprint: {display_project_path(progress_path)}"
            )
        calibration_identity = progress.get("calibration_identity")
        if not isinstance(calibration_identity, dict):
            raise AggregationError(
                f"Missing calibration identity: {display_project_path(progress_path)}"
            )
        seed = progress.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AggregationError(f"Invalid seed: {display_project_path(progress_path)}")
        diagnostic = progress.get("diagnostic_calibration_disabled")
        if not isinstance(diagnostic, bool):
            raise AggregationError(
                f"Missing diagnostic calibration mode: {display_project_path(progress_path)}"
            )
        label = path.parent.name if path.name == FINAL_FILENAME else path.stem
        return _ResultSource(
            label=label,
            path=path,
            results=raw,
            config_fingerprint=fingerprint,
            calibration_identity=calibration_identity,
            seed=seed,
            diagnostic_calibration_disabled=diagnostic,
        )

    def _validate_compatible(
        self, sources: list[_ResultSource]
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Validate matching algorithms, cases, and metric contracts."""
        paths = [source.path.resolve() for source in sources]
        if len(set(paths)) != len(paths):
            raise AggregationError("Aggregate result sources must be unique.")
        fingerprints = {source.config_fingerprint for source in sources}
        if len(fingerprints) != 1:
            raise AggregationError("Aggregate result config fingerprints differ.")
        if any(
            source.calibration_identity != sources[0].calibration_identity for source in sources[1:]
        ):
            raise AggregationError("Aggregate result calibration identities differ.")
        diagnostics = {source.diagnostic_calibration_disabled for source in sources}
        if len(diagnostics) != 1:
            raise AggregationError("Aggregate result calibration modes differ.")
        seeds = [source.seed for source in sources]
        if len(set(seeds)) != len(seeds):
            raise AggregationError("Aggregate result seeds must be unique.")
        algorithms = sorted(str(value) for value in sources[0].results)
        cases = self._case_keys(sources[0].results, algorithms)
        for source in sources:
            current_algorithms = sorted(str(value) for value in source.results)
            if current_algorithms != algorithms:
                raise AggregationError(
                    f"Algorithm set differs in {display_project_path(source.path)}."
                )
            current_cases = self._case_keys(source.results, algorithms)
            if current_cases != cases:
                raise AggregationError(
                    f"Benchmark/workload cases differ in " f"{display_project_path(source.path)}."
                )
            for algorithm in algorithms:
                for benchmark, load_pattern in cases:
                    metrics = source.results[algorithm][benchmark][load_pattern]
                    if not isinstance(metrics, dict) or "mean_reward" not in metrics:
                        raise AggregationError(
                            "Every case must contain a numeric mean_reward: "
                            f"{source.label}/{algorithm}/{benchmark}/{load_pattern}."
                        )
                    try:
                        reward = float(metrics["mean_reward"])
                    except (TypeError, ValueError) as exc:
                        raise AggregationError(
                            "Every case must contain a numeric mean_reward: "
                            f"{source.label}/{algorithm}/{benchmark}/{load_pattern}."
                        ) from exc
                    if not math.isfinite(reward):
                        raise AggregationError(
                            "Every case must contain a finite mean_reward: "
                            f"{source.label}/{algorithm}/{benchmark}/{load_pattern}."
                        )
        return algorithms, cases

    @staticmethod
    def _case_keys(results: dict[str, Any], algorithms: list[str]) -> list[tuple[str, str]]:
        """Return sorted cases and require identical shape for each algorithm."""
        expected: list[tuple[str, str]] | None = None
        for algorithm in algorithms:
            algorithm_results = results.get(algorithm)
            if not isinstance(algorithm_results, dict):
                raise AggregationError(f"Algorithm '{algorithm}' must contain an object.")
            current: list[tuple[str, str]] = []
            for benchmark in sorted(algorithm_results):
                loads = algorithm_results[benchmark]
                if not isinstance(loads, dict):
                    raise AggregationError(
                        f"Benchmark '{benchmark}' under '{algorithm}' must be an object."
                    )
                current.extend((str(benchmark), str(load)) for load in sorted(loads))
            if expected is None:
                expected = current
            elif current != expected:
                raise AggregationError(
                    "All algorithms must contain the same benchmark/workload cases."
                )
        if not expected:
            raise AggregationError("Aggregate inputs contain no completed cases.")
        return expected

    @staticmethod
    def _reward(source: _ResultSource, algorithm: str, benchmark: str, load_pattern: str) -> float:
        """Read one validated reward value."""
        return float(source.results[algorithm][benchmark][load_pattern]["mean_reward"])

    def _aggregate_single(
        self,
        sources: list[_ResultSource],
        cases: list[tuple[str, str]],
        primary: str,
    ) -> dict[str, Any]:
        """Aggregate a protocol that contains one algorithm."""
        rows = []
        all_rewards = []
        for benchmark, load_pattern in cases:
            rewards = np.asarray(
                [self._reward(source, primary, benchmark, load_pattern) for source in sources],
                dtype=float,
            )
            all_rewards.extend(rewards.tolist())
            rows.append(
                {
                    "benchmark": benchmark,
                    "load_pattern": load_pattern,
                    "seed_count": len(sources),
                    "reward_mean": float(np.mean(rewards)),
                    "reward_std": float(np.std(rewards, ddof=0)),
                    "reward_min": float(np.min(rewards)),
                    "reward_max": float(np.max(rewards)),
                }
            )
        overall = np.asarray(all_rewards, dtype=float)
        return {
            "overall": {
                "total_comparisons": len(cases),
                "seed_count": len(sources),
                "mean_reward": float(np.mean(overall)),
                "std_reward": float(np.std(overall, ddof=0)),
                "min_reward": float(np.min(overall)),
                "max_reward": float(np.max(overall)),
            },
            "cases": rows,
        }

    def _aggregate_baseline(
        self,
        sources: list[_ResultSource],
        cases: list[tuple[str, str]],
        *,
        primary: str,
        baseline: str,
    ) -> dict[str, Any]:
        """Aggregate one primary-versus-baseline comparison."""
        rows: list[dict[str, Any]] = []
        all_comparisons: list[dict[str, Any]] = []
        for benchmark, load_pattern in cases:
            seed_rows: list[dict[str, Any]] = []
            for source in sources:
                primary_reward = self._reward(source, primary, benchmark, load_pattern)
                baseline_reward = self._reward(source, baseline, benchmark, load_pattern)
                comparison = {
                    "source": source.label,
                    "benchmark": benchmark,
                    "load_pattern": load_pattern,
                    "primary_reward": primary_reward,
                    "baseline_reward": baseline_reward,
                    "raw_reward_delta": primary_reward - baseline_reward,
                    "relative_improvement_pct": relative_improvement(
                        primary_reward, baseline_reward
                    ),
                    "small_baseline": _is_small_baseline(baseline_reward),
                    "primary_win": primary_reward > baseline_reward,
                }
                seed_rows.append(comparison)
                all_comparisons.append(comparison)
            primary_values = np.asarray(
                [float(row["primary_reward"]) for row in seed_rows], dtype=float
            )
            baseline_values = np.asarray(
                [float(row["baseline_reward"]) for row in seed_rows], dtype=float
            )
            raw_deltas = np.asarray(
                [float(row["raw_reward_delta"]) for row in seed_rows], dtype=float
            )
            finite_relative = np.asarray(
                [
                    float(row["relative_improvement_pct"])
                    for row in seed_rows
                    if np.isfinite(float(row["relative_improvement_pct"]))
                ],
                dtype=float,
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "load_pattern": load_pattern,
                    "seed_count": len(seed_rows),
                    "primary_reward_mean": float(np.mean(primary_values)),
                    "primary_reward_std": float(np.std(primary_values, ddof=0)),
                    "baseline_reward_mean": float(np.mean(baseline_values)),
                    "baseline_reward_std": float(np.std(baseline_values, ddof=0)),
                    "mean_raw_reward_delta": float(np.mean(raw_deltas)),
                    "median_relative_improvement_pct": (
                        float(np.median(finite_relative)) if finite_relative.size else 0.0
                    ),
                    "wins": sum(bool(row["primary_win"]) for row in seed_rows),
                    "small_baseline_cases": sum(bool(row["small_baseline"]) for row in seed_rows),
                }
            )
        return {
            "baseline_display_name": algorithm_display_name(baseline),
            "overall": _summarize_comparisons(all_comparisons),
            "cases": rows,
        }

    @staticmethod
    def _render_markdown(payload: dict[str, Any]) -> str:
        """Render a compact deterministic table from an aggregate payload."""
        primary_name = str(payload["primary_display_name"])
        lines = [
            "# Dynamic Experiment Aggregate Summary",
            "",
            f"Primary algorithm: {primary_name}",
            "",
            "## Sources",
            "",
        ]
        lines.extend(f"- {item['label']}: `{item['path']}`" for item in payload["sources"])
        single = payload.get("single_algorithm")
        if isinstance(single, dict):
            overall = single["overall"]
            lines.extend(
                [
                    "",
                    f"## {primary_name} multi-seed summary",
                    "",
                    "| Benchmark | Workload | Seeds | Mean reward | Std | Min | Max |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in single["cases"]:
                lines.append(
                    f"| {row['benchmark']} | {row['load_pattern']} | "
                    f"{row['seed_count']} | {row['reward_mean']:+.4f} | "
                    f"{row['reward_std']:.4f} | {row['reward_min']:+.4f} | "
                    f"{row['reward_max']:+.4f} |"
                )
            lines.extend(
                [
                    "",
                    f"Overall mean reward: {overall['mean_reward']:+.4f} "
                    f"across {overall['seed_count']} seeds.",
                ]
            )
            return "\n".join(lines) + "\n"

        for baseline, summary in payload["baselines"].items():
            baseline_name = summary["baseline_display_name"]
            lines.extend(
                [
                    "",
                    f"## {primary_name} vs {baseline_name}",
                    "",
                    "| Benchmark | Workload | Seeds | Primary reward | "
                    "Baseline reward | Raw delta | Median pct | Wins |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in summary["cases"]:
                lines.append(
                    f"| {row['benchmark']} | {row['load_pattern']} | "
                    f"{row['seed_count']} | {row['primary_reward_mean']:+.4f} "
                    f"+/- {row['primary_reward_std']:.4f} | "
                    f"{row['baseline_reward_mean']:+.4f} +/- "
                    f"{row['baseline_reward_std']:.4f} | "
                    f"{row['mean_raw_reward_delta']:+.4f} | "
                    f"{row['median_relative_improvement_pct']:+.2f}% | "
                    f"{row['wins']}/{row['seed_count']} |"
                )
            overall = summary["overall"]
            lines.extend(
                [
                    "",
                    f"Mean raw-reward delta: "
                    f"{overall['mean_raw_reward_delta']:+.4f}; "
                    f"win rate: {overall['win_rate_pct']:.1f}% "
                    f"against `{baseline}`.",
                ]
            )
        return "\n".join(lines) + "\n"
