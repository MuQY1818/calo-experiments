"""Environment construction for validated CALO experiment protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .codebert_analyzer import CodeBERTAnalyzer
from .dynamic_environment import DynamicLoadEnvironment, MultiBenchmarkDynamicEnv
from .environment import ServerlessFunctionEnv
from .experiment_config import ExperimentConfig, PROJECT_ROOT
from .state_space import StateSpace


PPO_ALGORITHMS = {"ppo", "ppo_flat", "ppo_load_only"}


@dataclass(frozen=True)
class AlgorithmEnvironments:
    """Evaluation environment plus the optional shared training environment."""

    evaluation: DynamicLoadEnvironment
    training: DynamicLoadEnvironment | MultiBenchmarkDynamicEnv


class SmokeTestStateSpace:
    """Lightweight state contract used by calibrated simulator smoke tests."""

    def __init__(self, code_feature_dim: int = StateSpace.DEFAULT_CODE_FEATURE_DIM):
        """Create a zero-valued state with the production dimensional layout."""
        from gymnasium import spaces

        self.code_feature_dim = int(code_feature_dim)
        self.state_dim = (
            self.code_feature_dim
            + StateSpace.LOAD_FEATURE_DIM
            + StateSpace.CATEGORY_DIM
            + StateSpace.HISTORY_DIM
            + StateSpace.CONTEXT_DIM
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        class _LoadMonitor:
            container_ttl = 600.0
            last_request_time: float | None = None

        self.load_monitor = _LoadMonitor()
        self.benchmark_name = ""

    def set_function(self, benchmark_name: str) -> None:
        """Record the benchmark without loading CodeBERT."""
        self.benchmark_name = benchmark_name

    def reset(self) -> None:
        """Reset the minimal dynamic state."""
        self.load_monitor.last_request_time = None

    def set_simulation_time(self, current_time: float) -> None:
        """Accept the production state-space time hook."""
        del current_time

    def update_configuration(
        self,
        memory_mb: int,
        architecture: str,
        timeout_sec: int,
    ) -> None:
        """Accept configuration updates required by the environment."""
        del memory_mb, architecture, timeout_sec

    def extract_state(self) -> np.ndarray:
        """Return one correctly shaped zero observation."""
        return np.zeros(self.state_dim, dtype=np.float32)

    def update_load(self, is_cold_start: bool = False) -> None:
        """Accept one load update."""
        del is_cold_start

    def update_response(self, response_time: float) -> None:
        """Accept one response-time update."""
        del response_time

    def update_utilization(self, cpu: float, memory: float) -> None:
        """Accept one utilization update."""
        del cpu, memory

    def update_performance(
        self,
        latency: float,
        cost: float,
        success: float,
        cold_start: float,
        invocation_count: int = 1,
    ) -> None:
        """Accept one performance update."""
        del latency, cost, success, cold_start, invocation_count

    def record_batch(
        self,
        arrival_times: Iterable[float],
        completion_times: Iterable[float],
        response_times: Iterable[float],
        cold_start_flags: Iterable[bool],
    ) -> None:
        """Record only the final arrival time needed for TTL behavior."""
        del completion_times, response_times, cold_start_flags
        arrivals = list(arrival_times)
        if arrivals:
            self.load_monitor.last_request_time = float(arrivals[-1])


class EnvironmentFactory:
    """Construct algorithm-specific environments from one validated config."""

    def __init__(self, config: ExperimentConfig):
        """Store immutable protocol inputs."""
        self.config = config
        self.environment = config.runtime_section("environment")
        settings = config.data.get("algorithm_settings", {})
        self.algorithm_settings = dict(settings) if isinstance(settings, dict) else {}

    def build(
        self,
        benchmark: str,
        load_pattern: str,
        algorithm: str,
        seed: int,
    ) -> AlgorithmEnvironments:
        """Build fresh evaluation and training environments for one algorithm."""
        evaluation = self._build_dynamic(
            benchmark=benchmark,
            load_pattern=load_pattern,
            algorithm=algorithm,
            state_space=self._build_state_space(algorithm),
            seed=seed,
        )
        training: DynamicLoadEnvironment | MultiBenchmarkDynamicEnv = evaluation
        if algorithm in PPO_ALGORITHMS:
            shared = self._build_shared_training(
                evaluation_benchmark=benchmark,
                algorithm=algorithm,
                seed=seed,
            )
            if shared is not None:
                training = shared
        return AlgorithmEnvironments(evaluation=evaluation, training=training)

    def build_smoke(self, benchmark: str, load_pattern: str) -> DynamicLoadEnvironment:
        """Build a calibrated environment without loading source embeddings."""
        code_dimension = int(
            self.environment.get("code_feature_dim", StateSpace.DEFAULT_CODE_FEATURE_DIM)
        )
        return self._build_dynamic(
            benchmark=benchmark,
            load_pattern=load_pattern,
            algorithm="smoke",
            state_space=SmokeTestStateSpace(code_feature_dim=code_dimension),
        )

    def _algorithm_settings(self, algorithm: str) -> dict[str, Any]:
        """Return settings for one algorithm."""
        value = self.algorithm_settings.get(algorithm, {})
        return dict(value) if isinstance(value, dict) else {}

    def _build_state_space(self, algorithm: str) -> StateSpace:
        """Build the state variant declared for one algorithm."""
        settings = self._algorithm_settings(algorithm)
        disable_code = bool(settings.get("disable_code_features", algorithm == "ppo_load_only"))
        disable_category = bool(
            settings.get("disable_function_category", algorithm == "ppo_load_only")
        )
        code_dimension = int(
            settings.get(
                "code_feature_dim",
                self.environment.get("code_feature_dim", StateSpace.DEFAULT_CODE_FEATURE_DIM),
            )
        )
        analyzer = None
        if not disable_code:
            analyzer = CodeBERTAnalyzer.get_shared(
                policy=self.config.codebert_policy,
                embed_dim=code_dimension,
            )
        return StateSpace(
            codebert_analyzer=analyzer,
            sebs_root=str(PROJECT_ROOT),
            enable_code_features=not disable_code,
            enable_function_category=not disable_category,
            code_feature_dim=code_dimension,
        )

    def _build_base(self, benchmark: str, state_space: Any) -> ServerlessFunctionEnv:
        """Construct the calibrated service environment."""
        return ServerlessFunctionEnv(
            benchmark=benchmark,
            deployment="local",
            enable_real_execution=False,
            normalize_reward=bool(self.environment.get("normalize_reward", False)),
            reward_weights=self.environment.get("reward_weights"),
            reward_penalties=self.environment.get("reward_penalties"),
            success_bonus=float(self.environment.get("success_bonus", 0.5)),
            cost_normalization=str(self.environment.get("cost_normalization", "ratio")),
            calibration_dir=self.environment.get("calibration_dir"),
            calibration_dirs=self.environment.get("calibration_dirs"),
            step_duration_sec=float(self.environment.get("step_duration_sec", 15.0)),
            max_containers=int(self.environment.get("max_containers", 32)),
            container_ttl_sec=float(self.environment.get("container_ttl_sec", 600.0)),
            state_space=state_space,
            action_space_config=self.environment.get("action_space"),
        )

    def _build_dynamic(
        self,
        *,
        benchmark: str,
        load_pattern: str,
        algorithm: str,
        state_space: Any,
        seed: int | None = None,
    ) -> DynamicLoadEnvironment:
        """Wrap one base environment in the accepted workload controller."""
        del algorithm
        dynamic = DynamicLoadEnvironment(
            self._build_base(benchmark, state_space),
            episode_length=int(self.environment["episode_length"]),
            load_change_freq=int(self.environment["load_change_freq"]),
            switch_penalty=float(self.environment["switch_penalty"]),
            adaptation_penalty=float(self.environment.get("adaptation_penalty", 0.05)),
            workload_source=str(self.environment.get("workload_source", "synthetic")),
            azure_profile_path=self.environment.get("azure_profile_path"),
            azure_summary_path=self.environment.get("azure_summary_path"),
            azure_top_k=int(self.environment.get("azure_top_k", 50)),
            azure_arrival_scale=float(self.environment.get("azure_arrival_scale", 1.0)),
            azure_max_arrivals_per_step=self.environment.get("azure_max_arrivals_per_step"),
            azure_profile_selection=str(
                self.environment.get("azure_profile_selection", "benchmark_aware")
            ),
            azure_selection_pool_size=int(self.environment.get("azure_selection_pool_size", 16)),
            azure_target_concurrency=self.environment.get("azure_target_concurrency"),
            calibration_dirs=self.environment.get("calibration_dirs"),
        )
        if seed is not None:
            dynamic.base_env.service_model.rng = np.random.default_rng(seed)
        dynamic.load_pattern = load_pattern
        return dynamic

    def _shared_setup(self, evaluation_benchmark: str) -> tuple[list[str], list[str]] | None:
        """Resolve benchmark and workload lists for shared PPO training."""
        raw = self.config.data.get("shared_training", {})
        shared = dict(raw) if isinstance(raw, dict) else {}
        if not shared.get("enabled", False):
            return None
        benchmarks = [str(item) for item in shared.get("benchmarks", self.config.benchmarks)]
        if shared.get("exclude_eval_benchmark", False):
            benchmarks = [item for item in benchmarks if item != evaluation_benchmark]
        if not benchmarks:
            benchmarks = [evaluation_benchmark]
        load_patterns = [
            str(item) for item in shared.get("load_patterns", self.config.load_patterns)
        ]
        if not load_patterns:
            load_patterns = [self.config.load_patterns[0]]
        return list(dict.fromkeys(benchmarks)), list(dict.fromkeys(load_patterns))

    def _build_shared_training(
        self,
        *,
        evaluation_benchmark: str,
        algorithm: str,
        seed: int,
    ) -> MultiBenchmarkDynamicEnv | None:
        """Build episode-wise shared training when enabled by the protocol."""
        setup = self._shared_setup(evaluation_benchmark)
        if setup is None:
            return None
        benchmarks, load_patterns = setup

        def environment_builder(benchmark: str) -> DynamicLoadEnvironment:
            return self._build_dynamic(
                benchmark=benchmark,
                load_pattern=load_patterns[0],
                algorithm=algorithm,
                state_space=self._build_state_space(algorithm),
                seed=seed,
            )

        return MultiBenchmarkDynamicEnv(
            benchmark_names=benchmarks,
            env_factory=environment_builder,
            load_patterns=load_patterns,
            seed=seed,
        )

    def calibration_architectures(self) -> tuple[str, ...]:
        """Return calibration-routing labels exposed to smoke verification."""
        mapping = self.environment.get("calibration_dirs")
        if isinstance(mapping, dict) and mapping:
            return tuple(sorted(str(key) for key in mapping))
        return ("single",) if self.environment.get("calibration_dir") else ("heuristic",)

    def action_architectures(self, environment: DynamicLoadEnvironment) -> tuple[str, ...]:
        """Return architectures represented in an environment action catalog."""
        action_space = environment.base_env.action_space_wrapper
        return tuple(sorted(set(action_space.ARCHITECTURE_OPTIONS)))
