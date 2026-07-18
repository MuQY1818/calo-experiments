"""Durable progress, resume, and final-result storage for CALO experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .durable_io import atomic_write_text
from .experiment_config import ExperimentConfig, display_project_path


FINAL_FILENAME = "dynamic_experiment_results.json"
PARTIAL_FILENAME = "dynamic_experiment_results.partial.json"
PROGRESS_FILENAME = "progress.json"


class ResultStoreError(RuntimeError):
    """Indicate an invalid or incompatible result directory."""


@dataclass
class RunState:
    """In-memory state restored or initialized for one seed."""

    results: dict[str, dict[str, dict[str, dict[str, Any]]]]
    progress: dict[str, Any]
    completed_cases: set[tuple[str, str]]

    @property
    def is_complete(self) -> bool:
        """Return whether all configured cases are complete."""
        return int(self.progress["completed_cases"]) >= int(self.progress["total_cases"])


def _timestamp() -> str:
    """Return a stable local timestamp for progress records."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _json_default(value: Any) -> Any:
    """Convert NumPy values to JSON-compatible Python values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    """Atomically write stable, human-readable JSON."""
    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
    )


def load_json(path: Path) -> Any:
    """Load one JSON file and report corruption with its path."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultStoreError(f"Cannot read result JSON {path}: {exc}") from exc


class ResultStore:
    """Own the on-disk lifecycle for one configured experiment seed."""

    def __init__(self, output_dir: Path, config: ExperimentConfig, seed: int):
        """Create a result owner without mutating the directory yet."""
        self.output_dir = output_dir.resolve()
        self.config = config
        self.seed = int(seed)
        self.final_path = self.output_dir / FINAL_FILENAME
        self.partial_path = self.output_dir / PARTIAL_FILENAME
        self.progress_path = self.output_dir / PROGRESS_FILENAME

    def initialize(self, *, resume: bool) -> RunState:
        """Create a clean run state or restore a compatible partial run."""
        if resume:
            return self._resume()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            path.name
            for path in (self.final_path, self.partial_path, self.progress_path)
            if path.exists()
        ]
        if existing:
            raise ResultStoreError(
                f"Output directory already contains run state {existing}; use --resume or "
                "choose a new --output-dir."
            )
        state = RunState(
            results=self._empty_results(),
            progress=self._base_progress(started_at=_timestamp()),
            completed_cases=set(),
        )
        self.save_progress(state.progress)
        self.save_partial(state.results)
        return state

    def _resume(self) -> RunState:
        """Restore the final or partial payload and verify protocol identity."""
        if not self.progress_path.is_file():
            raise ResultStoreError(
                f"Cannot resume without {PROGRESS_FILENAME} in {self.output_dir}."
            )
        progress = load_json(self.progress_path)
        if not isinstance(progress, dict):
            raise ResultStoreError("progress.json must contain an object.")
        if progress.get("config_fingerprint") != self.config.fingerprint:
            raise ResultStoreError("Resume config fingerprint does not match this run directory.")
        if progress.get("calibration_identity") != self.config.calibration_identity.to_dict():
            raise ResultStoreError("Resume calibration identity does not match this run directory.")
        progress_seed = progress.get("seed")
        if isinstance(progress_seed, bool) or not isinstance(progress_seed, int):
            raise ResultStoreError("Resume progress contains an invalid seed.")
        if progress_seed != self.seed:
            raise ResultStoreError("Resume seed does not match this run directory.")

        source_path = self.final_path if self.final_path.is_file() else self.partial_path
        if not source_path.is_file():
            raise ResultStoreError("Resume requires a final or partial result JSON.")
        if progress.get("status") == "completed" and source_path == self.partial_path:
            raise ResultStoreError("Completed progress requires a final result JSON.")
        raw_results = load_json(source_path)
        results = self._normalize_results(raw_results)
        completed = {case for case in self.ordered_cases() if self.is_case_complete(results, *case)}
        if source_path == self.final_path and len(completed) != len(self.ordered_cases()):
            raise ResultStoreError("Final result JSON does not contain every configured case.")
        restored = self._base_progress(started_at=str(progress.get("started_at", _timestamp())))
        restored.update(
            {
                "status": "running",
                "completed_cases": len(completed),
                "current_stage": "resuming",
                "last_completed_case": self._last_completed(completed),
            }
        )
        self.save_progress(restored)
        if not self.final_path.is_file():
            self.save_partial(results)
        return RunState(results=results, progress=restored, completed_cases=completed)

    def _empty_results(self) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
        """Create the configured algorithm/benchmark result shape."""
        return {
            algorithm: {benchmark: {} for benchmark in self.config.benchmarks}
            for algorithm in self.config.algorithms
        }

    def _normalize_results(
        self,
        raw_results: Any,
    ) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
        """Restrict restored results to the current configured shape."""
        if not isinstance(raw_results, dict):
            raise ResultStoreError("Experiment results must contain an object.")
        normalized = self._empty_results()
        for algorithm in self.config.algorithms:
            algorithm_results = raw_results.get(algorithm, {})
            if not isinstance(algorithm_results, dict):
                continue
            for benchmark in self.config.benchmarks:
                benchmark_results = algorithm_results.get(benchmark, {})
                if isinstance(benchmark_results, dict):
                    normalized[algorithm][benchmark] = dict(benchmark_results)
        return normalized

    def _base_progress(self, *, started_at: str) -> dict[str, Any]:
        """Build the complete progress metadata contract."""
        return {
            "status": "running",
            "started_at": started_at,
            "config_path": display_project_path(self.config.source_path),
            "config_fingerprint": self.config.fingerprint,
            "calibration_identity": self.config.calibration_identity.to_dict(),
            "output_dir": display_project_path(self.output_dir),
            "seed": self.seed,
            "protocol_role": self.config.protocol_role,
            "diagnostic_calibration_disabled": self.config.diagnostic_calibration_disabled,
            "total_cases": len(self.ordered_cases()),
            "completed_cases": 0,
            "current_benchmark": None,
            "current_load_pattern": None,
            "current_algorithm": None,
            "current_stage": "initializing",
            "last_completed_case": None,
        }

    def ordered_cases(self) -> list[tuple[str, str]]:
        """Return the canonical benchmark/workload order."""
        return [
            (benchmark, load_pattern)
            for benchmark in self.config.benchmarks
            for load_pattern in self.config.load_patterns
        ]

    def is_case_complete(
        self,
        results: dict[str, dict[str, dict[str, dict[str, Any]]]],
        benchmark: str,
        load_pattern: str,
    ) -> bool:
        """Return whether every configured algorithm has metrics for a case."""
        return all(
            load_pattern in results.get(algorithm, {}).get(benchmark, {})
            for algorithm in self.config.algorithms
        )

    def _last_completed(self, completed: set[tuple[str, str]]) -> str | None:
        """Return the last completed label in canonical order."""
        labels = [
            f"{benchmark} / {load}"
            for benchmark, load in self.ordered_cases()
            if (benchmark, load) in completed
        ]
        return labels[-1] if labels else None

    def save_progress(self, progress: dict[str, Any]) -> None:
        """Persist progress with a fresh update timestamp."""
        progress["updated_at"] = _timestamp()
        write_json(self.progress_path, progress)

    def save_partial(
        self,
        results: dict[str, dict[str, dict[str, dict[str, Any]]]],
    ) -> None:
        """Persist visible intermediate results."""
        write_json(self.partial_path, results)

    def start_case(
        self,
        progress: dict[str, Any],
        benchmark: str,
        load_pattern: str,
    ) -> None:
        """Mark one case as active."""
        progress.update(
            {
                "current_benchmark": benchmark,
                "current_load_pattern": load_pattern,
                "current_algorithm": None,
                "current_stage": "creating_environment",
            }
        )
        self.save_progress(progress)

    def set_algorithm_stage(
        self,
        progress: dict[str, Any],
        algorithm: str,
        stage: str,
    ) -> None:
        """Update the current algorithm and its execution stage."""
        progress["current_algorithm"] = algorithm
        progress["current_stage"] = stage
        self.save_progress(progress)

    def complete_case(
        self,
        state: RunState,
        benchmark: str,
        load_pattern: str,
    ) -> None:
        """Persist a fully completed benchmark/workload case."""
        state.completed_cases.add((benchmark, load_pattern))
        state.progress.update(
            {
                "completed_cases": len(state.completed_cases),
                "current_algorithm": None,
                "current_stage": "completed_case",
                "last_completed_case": f"{benchmark} / {load_pattern}",
            }
        )
        self.save_partial(state.results)
        self.save_progress(state.progress)

    def finalize(
        self,
        state: RunState,
        *,
        extra_outputs: dict[str, Any] | None = None,
    ) -> Path:
        """Write final results, remove partial state, and mark completion."""
        incomplete = [
            f"{benchmark} / {load_pattern}"
            for benchmark, load_pattern in self.ordered_cases()
            if not self.is_case_complete(state.results, benchmark, load_pattern)
        ]
        if incomplete:
            raise ResultStoreError(
                f"Cannot finalize with {len(incomplete)} incomplete cases; first: "
                f"{incomplete[0]}."
            )
        write_json(self.final_path, state.results)
        if extra_outputs:
            for filename, payload in extra_outputs.items():
                write_json(self.output_dir / filename, payload)
        state.progress.update(
            {
                "status": "completed",
                "completed_cases": len(self.ordered_cases()),
                "current_algorithm": None,
                "current_stage": "completed",
                "results_path": FINAL_FILENAME,
            }
        )
        self.save_progress(state.progress)
        self.partial_path.unlink(missing_ok=True)
        return self.final_path

    def fail(self, progress: dict[str, Any], error: BaseException) -> None:
        """Record a failed run without deleting resumable partial results."""
        progress.update(
            {
                "status": "failed",
                "current_stage": "failed",
                "error": str(error),
            }
        )
        self.save_progress(progress)
