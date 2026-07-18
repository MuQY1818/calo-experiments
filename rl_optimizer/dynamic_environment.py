"""Dynamic workload wrappers for the CALO simulator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .container_pool import ContainerPool
from .environment import ServerlessFunctionEnv
from .service_model import CalibratedServiceModel
from .sim_clock import SimulationClock
from .workload_sources import (
    AzureTraceWorkloadSource,
    SyntheticWorkloadSource,
    WorkloadStep,
)


DEFAULT_AZURE_PROFILE_PATH = Path(
    "external_data/azure_functions/processed/"
    "azure_functions_2019_topk_minute_profiles.csv"
)


class DynamicLoadEnvironment(gym.Env):
    """Wraps the base environment with time-varying workload windows."""

    SUPPORTED_SYNTHETIC_PATTERNS = ("sine", "spike", "surge", "decay", "random")

    def __init__(
        self,
        base_env: ServerlessFunctionEnv,
        episode_length: int = 50,
        load_change_freq: int = 5,
        switch_penalty: float = 0.05,
        adaptation_penalty: float = 0.05,
        workload_source: str = "synthetic",
        azure_profile_path: str | None = None,
        azure_summary_path: str | None = None,
        azure_top_k: int = 50,
        azure_arrival_scale: float = 1.0,
        azure_max_arrivals_per_step: int | None = None,
        azure_profile_selection: str = "benchmark_aware",
        azure_selection_pool_size: int = 16,
        azure_target_concurrency: float | None = None,
        calibration_dir: str | None = None,
        calibration_dirs: Dict[str, str] | None = None,
    ):
        """Initializes the dynamic workload environment.

        Args:
            base_env: Underlying batch simulator environment.
            episode_length: Max number of control steps per episode.
            load_change_freq: Synthetic workload refresh period in control steps.
            switch_penalty: Penalty applied when the configuration changes.
            adaptation_penalty: Penalty when the agent ignores a recent load shift.
            workload_source: ``synthetic`` or ``azure_trace``.
            azure_profile_path: Optional path to processed Azure workload profiles.
            azure_summary_path: Optional path to Azure summary CSV.
            azure_top_k: Top-K active Azure functions used for replay.
            azure_arrival_scale: Scales Azure replay intensity without changing shape.
            azure_max_arrivals_per_step: Optional cap for replayed arrivals per step.
            azure_profile_selection: Strategy for Azure profile selection.
            azure_selection_pool_size: Candidate pool size for benchmark-aware matching.
            azure_target_concurrency: Optional override for benchmark target concurrency.
            calibration_dir: Optional single calibration directory override.
            calibration_dirs: Optional architecture-specific calibration map.
        """
        super().__init__()

        self.base_env = base_env
        self.episode_length = int(episode_length)
        self.load_change_freq = max(1, int(load_change_freq))
        self.switch_penalty = float(switch_penalty)
        self.adaptation_penalty = float(adaptation_penalty)
        self.default_workload_source = str(workload_source)
        self.workload_source = self.default_workload_source
        self.azure_profile_path = (
            Path(azure_profile_path) if azure_profile_path else DEFAULT_AZURE_PROFILE_PATH
        )
        self.azure_summary_path = Path(azure_summary_path) if azure_summary_path else None
        self.azure_top_k = int(azure_top_k)
        self.azure_arrival_scale = float(azure_arrival_scale)
        self.azure_max_arrivals_per_step = (
            None
            if azure_max_arrivals_per_step is None
            else int(azure_max_arrivals_per_step)
        )
        self.azure_profile_selection = str(azure_profile_selection)
        self.azure_selection_pool_size = max(1, int(azure_selection_pool_size))
        self.azure_target_concurrency = (
            None if azure_target_concurrency is None else float(azure_target_concurrency)
        )
        self.step_duration_sec = float(base_env.step_duration_sec)

        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space
        self.action_space_wrapper = base_env.action_space_wrapper
        self.state_space = base_env.state_space

        self.simulation_clock = SimulationClock()
        if calibration_dirs or calibration_dir:
            self.service_model = CalibratedServiceModel(
                calibration_dir=calibration_dir,
                calibration_dirs=calibration_dirs,
            )
        else:
            self.service_model = base_env.service_model
        self.container_pool = ContainerPool(
            ttl_sec=base_env.container_pool.ttl_sec,
            max_containers=base_env.container_pool.max_containers,
        )
        self.base_env.attach_simulation_backend(
            simulation_clock=self.simulation_clock,
            service_model=self.service_model,
            container_pool=self.container_pool,
        )

        self.rng = np.random.default_rng()
        self.current_step = 0
        self.last_action: Optional[int] = None
        self.load_pattern = "sine"
        self.load_recently_changed = False
        self._workload_driver = None
        self._current_workload = self._default_workload_step()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        """Resets the episode and selects the workload driver."""
        super().reset(seed=seed)
        options = options or {}

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.last_action = None
        self.load_recently_changed = False
        self.load_pattern = self._resolve_load_pattern(options)
        selected_source = self._resolve_workload_source(options)

        self.simulation_clock.reset()
        self.container_pool.reset()
        self._workload_driver = self._build_workload_driver(
            workload_source=selected_source,
            load_pattern=self.load_pattern,
            seed=seed,
        )
        self._current_workload = self._sample_workload(step_index=0)
        self.base_env.set_workload_step(self._current_workload)

        observation, info = self.base_env.reset(seed=seed)
        info["load_pattern"] = self.load_pattern
        info["workload_source"] = selected_source
        profile_info = self._get_workload_driver_info()
        if profile_info:
            info["workload_profile"] = profile_info
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Executes one action under the current workload window."""
        adaptation_penalty = 0.0
        if self.load_recently_changed and self.last_action is not None:
            if int(action) == int(self.last_action):
                adaptation_penalty = self.adaptation_penalty
        self.load_recently_changed = False

        observation, reward, done, truncated, info = self.base_env.step(int(action))

        if self.last_action is not None and int(action) != int(self.last_action):
            reward -= self.switch_penalty
            info["config_switch"] = True
            info["switch_penalty"] = self.switch_penalty
        else:
            info["config_switch"] = False
            info["switch_penalty"] = 0.0

        if adaptation_penalty > 0.0:
            reward -= adaptation_penalty
        info["adaptation_penalty"] = adaptation_penalty

        self.last_action = int(action)
        self.current_step += 1

        next_workload = self._sample_workload(step_index=self.current_step)
        load_changed = self._has_meaningful_change(
            previous=self._current_workload,
            current=next_workload,
            step_index=self.current_step,
        )
        self._current_workload = next_workload
        self.base_env.set_workload_step(next_workload)
        self.load_recently_changed = load_changed

        info["load_changed"] = load_changed
        info["load_pattern"] = self.load_pattern
        info["workload_source"] = self.workload_source
        info["next_workload"] = self.base_env._serialize_workload(next_workload)
        profile_info = self._get_workload_driver_info()
        if profile_info:
            info["workload_profile"] = profile_info

        if self.current_step >= self.episode_length:
            truncated = True

        return observation, reward, done, truncated, info

    def probe_calibration(
        self,
        *,
        action_id: int,
        architecture: str,
    ) -> Dict[str, float | int | str]:
        """Read calibrated values for one explicit action without advancing state."""
        action = int(action_id)
        expected_architecture = str(architecture)
        configuration = self.action_space_wrapper.get_configuration(action)
        if configuration.architecture != expected_architecture:
            raise ValueError(
                f"Action {action} uses {configuration.architecture}, not "
                f"{expected_architecture}."
            )

        model = self.service_model
        rng_state = deepcopy(model.rng.bit_generator.state)
        try:
            warm_runtime_ms = model.sample_warm_runtime_ms(
                self.base_env.benchmark,
                configuration.memory_mb,
                configuration.timeout_sec,
                expected_architecture,
            )
            cold_overhead_ms = model.sample_cold_overhead_ms(
                self.base_env.benchmark,
                configuration.memory_mb,
                configuration.timeout_sec,
                expected_architecture,
            )
            ttl_sec = model.estimate_ttl_sec(
                self.base_env.benchmark,
                configuration.memory_mb,
                configuration.timeout_sec,
                expected_architecture,
            )
        finally:
            model.rng.bit_generator.state = rng_state
        return {
            "action_id": action,
            "architecture": expected_architecture,
            "memory_mb": configuration.memory_mb,
            "timeout_sec": configuration.timeout_sec,
            "warm_runtime_ms": float(warm_runtime_ms),
            "cold_overhead_ms": float(cold_overhead_ms),
            "ttl_sec": float(ttl_sec),
        }

    def _resolve_load_pattern(self, options: Dict) -> str:
        """Returns the current load-pattern label."""
        requested = options.get("load_pattern")
        if requested is None:
            return self.load_pattern
        return str(requested)

    def _resolve_workload_source(self, options: Dict) -> str:
        """Returns the workload source type for the next episode."""
        requested = options.get("workload_source")
        if requested is not None:
            self.workload_source = str(requested)
        elif self.load_pattern == "azure_trace":
            self.workload_source = "azure_trace"
        else:
            self.workload_source = self.default_workload_source
        return self.workload_source

    def _build_workload_driver(
        self,
        workload_source: str,
        load_pattern: str,
        seed: Optional[int],
    ):
        """Constructs the selected workload driver."""
        if workload_source == "azure_trace":
            driver = AzureTraceWorkloadSource(
                profile_path=self.azure_profile_path,
                summary_path=self.azure_summary_path,
                step_duration_sec=self.step_duration_sec,
                top_k=self.azure_top_k,
                arrival_scale=self.azure_arrival_scale,
                max_arrivals_per_step=self.azure_max_arrivals_per_step,
                benchmark_name=self.base_env.benchmark,
                profile_selection=self.azure_profile_selection,
                selection_pool_size=self.azure_selection_pool_size,
                target_concurrency=self.azure_target_concurrency,
                rng=self.rng,
            )
            driver.reset(seed=seed)
            return driver

        synthetic_pattern = (
            load_pattern
            if load_pattern in self.SUPPORTED_SYNTHETIC_PATTERNS
            else "random"
        )
        driver = SyntheticWorkloadSource(
            pattern=synthetic_pattern,
            step_duration_sec=self.step_duration_sec,
            rng=self.rng,
        )
        driver.reset(seed=seed)
        return driver

    def _sample_workload(self, step_index: int) -> WorkloadStep:
        """Samples the workload window for one control step."""
        if self._workload_driver is None:
            return self._default_workload_step()

        if isinstance(self._workload_driver, SyntheticWorkloadSource):
            source_index = step_index // self.load_change_freq
        else:
            source_index = step_index
        return self._workload_driver.next_step(source_index)

    def _has_meaningful_change(
        self,
        previous: WorkloadStep,
        current: WorkloadStep,
        step_index: int,
    ) -> bool:
        """Determines whether the next decision should treat load as changed."""
        if step_index <= 0:
            return False

        if isinstance(self._workload_driver, SyntheticWorkloadSource):
            return step_index % self.load_change_freq == 0

        if previous.profile_key != current.profile_key:
            return True
        if previous.minute_of_day != current.minute_of_day:
            return True
        if previous.arrival_count != current.arrival_count:
            return True
        return not np.isclose(previous.load_value, current.load_value)

    def _default_workload_step(self) -> WorkloadStep:
        """Returns a small default workload used before reset."""
        return WorkloadStep(
            arrival_count=1,
            step_duration_sec=self.step_duration_sec,
            source_name="default",
            load_value=1.0,
            minute_of_day=0,
            mean_invocations_per_minute=4.0,
        )

    def _get_workload_driver_info(self) -> Dict[str, float | str]:
        """Return debug metadata from the active workload driver."""
        if self._workload_driver is None:
            return {}
        getter = getattr(self._workload_driver, "get_selected_profile_info", None)
        if getter is None:
            return {}
        return getter()


class MultiBenchmarkDynamicEnv(gym.Env):
    """Episode-level wrapper that samples across multiple benchmark envs."""

    def __init__(
        self,
        benchmark_names: list[str],
        env_factory: Callable[[str], DynamicLoadEnvironment],
        load_patterns: list[str] | None = None,
        seed: int | None = None,
    ):
        """Initialize one shared training environment across benchmarks.

        Args:
            benchmark_names: Benchmarks sampled at episode boundaries.
            env_factory: Builds a dynamic environment for one benchmark.
            load_patterns: Optional load-pattern candidates sampled per episode.
            seed: Optional RNG seed for benchmark/load-pattern sampling.
        """
        super().__init__()
        if not benchmark_names:
            raise ValueError("benchmark_names must not be empty")

        self.benchmark_names = [str(name) for name in benchmark_names]
        self.env_factory = env_factory
        self.load_patterns = (
            [str(pattern) for pattern in load_patterns]
            if load_patterns
            else ["spike"]
        )
        self.rng = np.random.default_rng(seed)
        self.env_cache: Dict[str, DynamicLoadEnvironment] = {}
        self.current_env: DynamicLoadEnvironment | None = None
        self.current_benchmark = self.benchmark_names[0]
        self.load_pattern = self.load_patterns[0]

        reference_env = self._get_env(self.current_benchmark)
        self.observation_space = reference_env.observation_space
        self.action_space = reference_env.action_space
        self.action_space_wrapper = reference_env.action_space_wrapper

    @property
    def base_env(self) -> ServerlessFunctionEnv:
        """Return the active base env or a cached reference."""
        if self.current_env is not None:
            return self.current_env.base_env
        return self._get_env(self.current_benchmark).base_env

    @property
    def benchmark(self) -> str:
        """Return the benchmark selected for the current episode."""
        return self.current_benchmark

    @property
    def state_space(self):
        """Return the state space of the active benchmark env."""
        return self.base_env.state_space

    def _get_env(self, benchmark_name: str) -> DynamicLoadEnvironment:
        """Return a cached dynamic env for one benchmark."""
        env = self.env_cache.get(benchmark_name)
        if env is None:
            env = self.env_factory(benchmark_name)
            self.env_cache[benchmark_name] = env
        return env

    def _sample_benchmark(self) -> str:
        """Sample one benchmark uniformly for the next episode."""
        index = int(self.rng.integers(0, len(self.benchmark_names)))
        return self.benchmark_names[index]

    def _sample_load_pattern(self) -> str:
        """Sample one load pattern uniformly for the next episode."""
        index = int(self.rng.integers(0, len(self.load_patterns)))
        return self.load_patterns[index]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        """Reset one sampled benchmark environment."""
        super().reset(seed=seed)
        options = dict(options or {})
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        benchmark = str(options.pop("benchmark", self._sample_benchmark()))
        load_pattern = str(options.get("load_pattern", self._sample_load_pattern()))
        env = self._get_env(benchmark)

        self.current_benchmark = benchmark
        self.current_env = env
        self.load_pattern = load_pattern
        reset_seed = (
            None if seed is None else int(self.rng.integers(0, np.iinfo(np.int32).max))
        )
        obs, info = env.reset(
            seed=reset_seed,
            options={**options, "load_pattern": load_pattern},
        )
        info["benchmark"] = benchmark
        info["sampled_benchmark"] = benchmark
        info["load_pattern"] = load_pattern
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Delegate one action to the currently selected benchmark env."""
        if self.current_env is None:
            raise RuntimeError("Call reset() before step() on MultiBenchmarkDynamicEnv")

        obs, reward, done, truncated, info = self.current_env.step(action)
        info["benchmark"] = self.current_benchmark
        info["load_pattern"] = self.load_pattern
        return obs, reward, done, truncated, info

    def close(self) -> None:
        """Close cached child environments."""
        for env in self.env_cache.values():
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        self.env_cache.clear()
        self.current_env = None


class MultiObjectiveEnvironment(gym.Env):
    """Multi-objective wrapper over the base environment."""

    def __init__(
        self,
        base_env: ServerlessFunctionEnv,
        objectives: Dict[str, float] = None,
    ):
        """Initializes the multi-objective wrapper."""
        super().__init__()

        self.base_env = base_env

        if objectives is None:
            objectives = {
                "latency": 0.4,
                "cost": 0.3,
                "carbon": 0.2,
                "sla": 0.1,
            }
        self.objectives = objectives

        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

    def reset(self, seed=None, options=None):
        """Resets the wrapper and delegates to the base environment."""
        return self.base_env.reset(seed=seed, options=options)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Executes an action and reweights the reward across objectives."""
        obs, reward, done, truncated, info = self.base_env.step(action)

        metrics = info["metrics"]

        latency_score = 1.0 - min(metrics["latency"] / 3000.0, 1.0)
        cost_score = 1.0 - min(metrics["cost"] / 0.0001, 1.0)

        config = self.base_env.action_space_wrapper.get_configuration(action)
        carbon_score = 0.8 if config.architecture == "arm64" else 0.5
        sla_score = 1.0 if metrics["latency"] < 500.0 else 0.0

        multi_reward = (
            self.objectives["latency"] * latency_score
            + self.objectives["cost"] * cost_score
            + self.objectives["carbon"] * carbon_score
            + self.objectives["sla"] * sla_score
        )

        del reward
        return obs, multi_reward - 1.0, done, truncated, info
