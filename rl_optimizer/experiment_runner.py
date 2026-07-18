"""Experiment, suite, and calibrated smoke orchestration for CALO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Callable, Iterable

import numpy as np

from .algorithm_executor import AlgorithmExecutor, algorithm_display_name
from .environment_factory import AlgorithmEnvironments, EnvironmentFactory
from .experiment_config import (
    ExperimentConfig,
    display_project_path,
    resolve_project_path,
)
from .result_aggregation import AggregateOutputs, ResultAggregator
from .result_store import (
    FINAL_FILENAME,
    PARTIAL_FILENAME,
    PROGRESS_FILENAME,
    ResultStore,
    write_json,
)


DEFAULT_RUN_DIR = Path("outputs/paper_full48/seed_42")
DEFAULT_SUITE_DIR = Path("outputs/paper_full48")
SUITE_MANIFEST = "suite_manifest.json"


@dataclass(frozen=True)
class SuiteOutputs:
    """Completed per-seed and aggregate output paths."""

    seed_results: tuple[Path, ...]
    aggregate: AggregateOutputs
    manifest_path: Path


class ExperimentRunner:
    """Coordinate validated protocols without owning simulator algorithms."""

    def __init__(
        self,
        config: ExperimentConfig,
        emit: Callable[[str], None] | None = None,
    ):
        """Create a runner for one immutable experiment protocol."""
        self.config = config
        self.emit = emit or (lambda message: None)

    def run_seed(
        self,
        output_dir: str | Path = DEFAULT_RUN_DIR,
        *,
        seed: int | None = None,
        resume: bool = False,
    ) -> Path:
        """Run all configured cases for one seed and write final results."""
        resolved_seed = self.config.default_seed() if seed is None else int(seed)
        output_path = resolve_project_path(output_dir)
        store = ResultStore(output_path, self.config, resolved_seed)
        state = store.initialize(resume=resume)
        if state.is_complete:
            self.emit(f"Run already complete: {display_project_path(store.final_path)}")
            return store.finalize(state)

        np.random.seed(resolved_seed)
        random.seed(resolved_seed)
        factory = EnvironmentFactory(self.config)
        total_cases = len(store.ordered_cases())
        self.emit(
            f"Running seed {resolved_seed}: {total_cases} cases, "
            f"{len(self.config.algorithms)} algorithms."
        )
        try:
            for case_index, (benchmark, load_pattern) in enumerate(store.ordered_cases(), start=1):
                if (benchmark, load_pattern) in state.completed_cases:
                    self.emit(
                        f"Skipping completed case {case_index}/{total_cases}: "
                        f"{benchmark} / {load_pattern}"
                    )
                    continue
                store.start_case(state.progress, benchmark, load_pattern)
                self.emit(f"Case {case_index}/{total_cases}: {benchmark} / {load_pattern}")
                for algorithm in self.config.algorithms:
                    store.set_algorithm_stage(state.progress, algorithm, f"preparing_{algorithm}")
                    environments = factory.build(
                        benchmark=benchmark,
                        load_pattern=load_pattern,
                        algorithm=algorithm,
                        seed=resolved_seed,
                    )

                    def update_stage(stage: str, current: str = algorithm) -> None:
                        store.set_algorithm_stage(state.progress, current, stage)

                    executor = AlgorithmExecutor(self.config, progress_callback=update_stage)
                    self.emit(f"  {algorithm_display_name(algorithm)}")
                    try:
                        metrics = executor.execute(
                            algorithm,
                            environments,
                            benchmark=benchmark,
                            load_pattern=load_pattern,
                            seed=resolved_seed,
                        )
                    finally:
                        self._close_environments(environments)
                    state.results[algorithm][benchmark][load_pattern] = metrics
                    store.save_partial(state.results)
                store.complete_case(state, benchmark, load_pattern)
            result_path = store.finalize(state)
        except BaseException as exc:
            store.fail(state.progress, exc)
            raise
        self.emit(f"Results: {display_project_path(result_path)}")
        return result_path

    def run_suite(
        self,
        output_dir: str | Path = DEFAULT_SUITE_DIR,
        *,
        seeds: Iterable[int] | None = None,
        resume: bool = False,
    ) -> SuiteOutputs:
        """Run one protocol for each seed and aggregate completed results."""
        resolved_seeds = (
            tuple(int(seed) for seed in seeds) if seeds is not None else self.config.seeds
        )
        if not resolved_seeds:
            raise ValueError("A suite requires at least one seed.")
        if len(set(resolved_seeds)) != len(resolved_seeds):
            raise ValueError("Suite seeds must be unique.")
        suite_dir = resolve_project_path(output_dir)
        suite_dir.mkdir(parents=True, exist_ok=True)
        result_paths_list = []
        for seed in resolved_seeds:
            seed_dir = suite_dir / f"seed_{seed}"
            seed_resume = resume and any(
                (seed_dir / filename).exists()
                for filename in (FINAL_FILENAME, PARTIAL_FILENAME, PROGRESS_FILENAME)
            )
            result_paths_list.append(self.run_seed(seed_dir, seed=seed, resume=seed_resume))
        result_paths = tuple(result_paths_list)
        aggregate = ResultAggregator().aggregate(result_paths, output_dir=suite_dir / "aggregate")
        manifest_path = suite_dir / SUITE_MANIFEST
        write_json(
            manifest_path,
            {
                "config_path": display_project_path(self.config.source_path),
                "config_fingerprint": self.config.fingerprint,
                "calibration_identity": self.config.calibration_identity.to_dict(),
                "seeds": list(resolved_seeds),
                "seed_count": len(resolved_seeds),
                "per_seed_results": [display_project_path(path) for path in result_paths],
                "aggregate_json": display_project_path(aggregate.json_path),
                "aggregate_markdown": display_project_path(aggregate.markdown_path),
            },
        )
        self.emit(f"Suite manifest: {display_project_path(manifest_path)}")
        return SuiteOutputs(result_paths, aggregate, manifest_path)

    def run_smoke(
        self,
        *,
        steps: int = 5,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Exercise every benchmark/workload through calibrated simulation."""
        if steps <= 0:
            raise ValueError("Smoke steps must be positive.")
        resolved_seed = self.config.default_seed() if seed is None else int(seed)
        factory = EnvironmentFactory(self.config)
        results = []
        for benchmark in self.config.benchmarks:
            for load_pattern in self.config.load_patterns:
                environment = factory.build_smoke(benchmark, load_pattern)
                try:
                    observation, reset_info = environment.reset(
                        seed=resolved_seed,
                        options={"load_pattern": load_pattern},
                    )
                    action_space = environment.base_env.action_space_wrapper
                    action_architectures = factory.action_architectures(environment)
                    action_sequence = self._smoke_actions(action_space, action_architectures)
                    total_reward = 0.0
                    total_arrivals = 0
                    peak_concurrency = 0
                    max_queue_ratio = 0.0
                    max_cold_start_rate = 0.0
                    for step_index in range(steps):
                        action = action_sequence[step_index % len(action_sequence)]
                        _, reward, terminated, truncated, info = environment.step(action)
                        metrics = info["metrics"]
                        total_reward += float(reward)
                        total_arrivals += int(metrics["arrival_count"])
                        peak_concurrency = max(peak_concurrency, int(metrics["peak_concurrency"]))
                        max_queue_ratio = max(max_queue_ratio, float(metrics["queue_ratio"]))
                        max_cold_start_rate = max(
                            max_cold_start_rate,
                            float(metrics["cold_start_rate"]),
                        )
                        if terminated or truncated:
                            break
                    architecture_probes = self._probe_architectures(
                        environment,
                        action_architectures,
                    )
                    result = {
                        "benchmark": benchmark,
                        "load_pattern": load_pattern,
                        "observation_dim": int(np.asarray(observation).size),
                        "action_count": int(action_space.n_actions),
                        "action_architectures": list(action_architectures),
                        "calibration_architectures": list(factory.calibration_architectures()),
                        "architecture_probes": architecture_probes,
                        "workload": self._workload_label(reset_info),
                        "reward_sum": total_reward,
                        "arrival_count": total_arrivals,
                        "peak_concurrency": peak_concurrency,
                        "max_queue_ratio": max_queue_ratio,
                        "max_cold_start_rate": max_cold_start_rate,
                        "diagnostic_calibration_disabled": (
                            self.config.diagnostic_calibration_disabled
                        ),
                    }
                    self._validate_smoke_contract(result)
                    results.append(result)
                    self.emit(self._format_smoke_result(result))
                finally:
                    environment.close()
        expected = len(self.config.benchmarks) * len(self.config.load_patterns)
        if len(results) != expected:
            raise RuntimeError(f"Smoke completed {len(results)} cases; expected {expected}.")
        return results

    @staticmethod
    def _smoke_actions(action_space: Any, architectures: tuple[str, ...]) -> list[int]:
        """Choose stable actions that include every architecture."""
        actions = [
            action_space.get_action_id(512, architecture, 300) for architecture in architectures
        ]
        actions.extend([0, action_space.n_actions - 1, action_space.n_actions // 2])
        return list(dict.fromkeys(int(action) for action in actions))

    @staticmethod
    def _probe_architectures(
        environment: Any,
        architectures: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        """Read warm, cold, and TTL routing for one action per architecture."""
        action_space = environment.base_env.action_space_wrapper
        return {
            architecture: environment.probe_calibration(
                action_id=action_space.get_action_id(512, architecture, 300),
                architecture=architecture,
            )
            for architecture in architectures
        }

    def _validate_smoke_contract(self, result: dict[str, Any]) -> None:
        """Enforce accepted-paper dimensions for the canonical main protocol."""
        if self.config.protocol_role != "main":
            return
        if result["observation_dim"] != 79:
            raise RuntimeError(
                f"Main smoke expected a 79-D observation; received " f"{result['observation_dim']}."
            )
        if result["action_count"] != 48:
            raise RuntimeError(
                f"Main smoke expected 48 actions; received {result['action_count']}."
            )
        expected_architectures = {"x64", "arm64"}
        if set(result["action_architectures"]) != expected_architectures:
            raise RuntimeError("Main smoke action catalog must contain x64 and arm64.")
        if self.config.diagnostic_calibration_disabled:
            if result["calibration_architectures"] != ["heuristic"]:
                raise RuntimeError("Diagnostic smoke must report heuristic calibration.")
            return
        if set(result["calibration_architectures"]) != expected_architectures:
            raise RuntimeError("Main smoke calibration must contain x64 and arm64.")

    @staticmethod
    def _workload_label(reset_info: dict[str, Any]) -> str:
        """Return a stable workload label from Gym reset metadata."""
        profile = reset_info.get("workload_profile", {})
        if reset_info.get("workload_source") == "azure_trace" and isinstance(profile, dict):
            function_key = profile.get("function_key")
            if function_key:
                return str(function_key)
        return str(reset_info.get("load_pattern") or reset_info.get("workload_source", "n/a"))

    @staticmethod
    def _format_smoke_result(result: dict[str, Any]) -> str:
        """Render one concise smoke case line."""
        action_arch = ",".join(result["action_architectures"])
        calibration_arch = ",".join(result["calibration_architectures"])
        return (
            f"[Smoke] {result['benchmark']} / {result['load_pattern']}: "
            f"observation={result['observation_dim']}, "
            f"actions={result['action_count']}, "
            f"action_arch={action_arch}, calibration_arch={calibration_arch}, "
            f"reward_sum={result['reward_sum']:.3f}, "
            f"arrivals={result['arrival_count']}"
        )

    @staticmethod
    def _close_environments(environments: AlgorithmEnvironments) -> None:
        """Close evaluation and distinct shared-training environments."""
        environments.evaluation.close()
        if environments.training is not environments.evaluation:
            environments.training.close()
