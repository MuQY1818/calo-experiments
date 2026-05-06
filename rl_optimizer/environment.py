"""SeBS Gymnasium environment for CALO."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_space import ActionSpace
from .container_pool import ContainerPool
from .service_model import CalibratedServiceModel, ExactFeasibilityHint
from .sim_clock import SimulationClock
from .state_space import StateSpace
from .workload_sources import WorkloadStep


class ServerlessFunctionEnv(gym.Env):
    """Gym environment for serverless configuration optimization."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark: str,
        deployment: str = "local",
        config_path: str = "config/example.json",
        sebs_root: str = ".",
        max_steps: int = 100,
        reward_weights: Dict[str, float] | None = None,
        reward_penalties: Dict[str, float] | None = None,
        enable_real_execution: bool = False,
        normalize_reward: bool = True,
        success_bonus: float = 0.5,
        cost_normalization: str = "ratio",
        calibration_dir: str | None = None,
        calibration_dirs: Dict[str, str] | None = None,
        step_duration_sec: float = 15.0,
        max_containers: int = 32,
        container_ttl_sec: float = 600.0,
        state_space: StateSpace | None = None,
        action_space_config: Dict[str, Any] | None = None,
        log_init: bool = True,
    ):
        super().__init__()

        self.benchmark = benchmark
        self.deployment = deployment
        self.config_path = config_path
        self.sebs_root = Path(sebs_root)
        self.max_steps = max_steps
        self.enable_real_execution = enable_real_execution
        self.normalize_reward = normalize_reward
        self.success_bonus = success_bonus
        self.cost_normalization = cost_normalization
        self.step_duration_sec = float(step_duration_sec)

        self.reward_weights = (
            reward_weights if reward_weights is not None else {"latency": 0.5, "cost": 0.5}
        )
        self.reward_penalties = {
            "failure": 3.0,
            "cold_start": 0.2,
            "queue": 0.1,
        }
        if reward_penalties is not None:
            self.reward_penalties.update(
                {
                    key: float(value)
                    for key, value in reward_penalties.items()
                }
            )

        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        self.reward_count = 0

        self.state_space = (
            state_space if state_space is not None else StateSpace(sebs_root=str(sebs_root))
        )
        action_space_config = action_space_config or {}
        self.action_space_wrapper = ActionSpace(
            memory_options=action_space_config.get("memory_options"),
            architecture_options=action_space_config.get("architecture_options"),
            timeout_options=action_space_config.get("timeout_options"),
            preset=action_space_config.get("preset"),
        )
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.state_space.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.action_space_wrapper.n_actions)

        self.current_step = 0
        self.episode_rewards = []
        self.episode_metrics = []

        if (
            getattr(self.state_space, "code_features", None) is None
            or getattr(self.state_space, "function_category", None) is None
        ):
            self.state_space.set_function(benchmark)
        self.baseline_latency = 1000.0
        self.baseline_cost = 0.001

        self.simulation_clock = SimulationClock()
        self.service_model = CalibratedServiceModel(
            calibration_dir=calibration_dir,
            calibration_dirs=calibration_dirs,
        )
        self.container_pool = ContainerPool(
            ttl_sec=container_ttl_sec,
            max_containers=max_containers,
        )
        self._external_simulation_backend = False
        self.current_workload = WorkloadStep(
            arrival_count=1,
            step_duration_sec=self.step_duration_sec,
            source_name="default",
            load_value=1.0,
            minute_of_day=0,
            mean_invocations_per_minute=4.0,
        )

        if log_init:
            print("[Env] Environment initialized")
            print(f"  Benchmark: {benchmark}")
            print(f"  Deployment: {deployment}")
            print(f"  Real execution: {enable_real_execution}")
            print(f"  Observation space: {self.observation_space.shape}")
            print(f"  Action space: {self.action_space.n}")

    def attach_simulation_backend(
        self,
        simulation_clock: SimulationClock | None = None,
        service_model: CalibratedServiceModel | None = None,
        container_pool: ContainerPool | None = None,
    ) -> None:
        """Attach externally managed simulator components."""
        if simulation_clock is not None:
            self.simulation_clock = simulation_clock
        if service_model is not None:
            self.service_model = service_model
        if container_pool is not None:
            self.container_pool = container_pool
        self._external_simulation_backend = True
        self.state_space.load_monitor.container_ttl = self.container_pool.ttl_sec

    def set_workload_step(self, workload_step: WorkloadStep) -> None:
        """Set the workload that will be executed in the next environment step."""
        self.current_workload = workload_step

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        del options

        self.current_step = 0
        self.episode_rewards = []
        self.episode_metrics = []

        self.state_space.reset()
        if not self._external_simulation_backend:
            self.simulation_clock.reset()
            self.container_pool.reset()
        self.state_space.set_simulation_time(self.simulation_clock.now())

        default_action = self.action_space_wrapper.get_default_action()
        config = self.action_space_wrapper.get_configuration(default_action)
        self.state_space.update_configuration(
            config.memory_mb,
            config.architecture,
            config.timeout_sec,
        )

        observation = self.state_space.extract_state()
        info = {
            "step": self.current_step,
            "configuration": config.to_dict(),
            "simulation_time_sec": self.simulation_clock.now(),
            "workload": self._serialize_workload(self.current_workload),
        }
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one control step."""
        self.current_step += 1

        config = self.action_space_wrapper.get_configuration(action)
        self.state_space.update_configuration(
            config.memory_mb,
            config.architecture,
            config.timeout_sec,
        )

        metrics = self._execute_function(config)

        if not metrics.get("load_events_applied", False):
            self.state_space.update_load(is_cold_start=metrics["cold_start"])
            self.state_space.update_response(metrics["latency"] / 1000.0)

        self.state_space.update_utilization(
            metrics.get("cpu_utilization", 0.0),
            metrics.get("memory_utilization", 0.0),
        )
        self.state_space.update_performance(
            latency=metrics.get("p95_latency", metrics["latency"]),
            cost=metrics["cost"],
            success=metrics.get("success_rate", float(metrics["success"])),
            cold_start=metrics.get("cold_start_rate", float(metrics["cold_start"])),
            invocation_count=metrics.get("arrival_count", 1),
        )

        raw_reward = self._compute_reward(metrics)
        reward = (
            self._update_and_normalize_reward(raw_reward)
            if self.normalize_reward
            else raw_reward
        )

        self.state_space.set_simulation_time(
            metrics.get("window_end_sec", self.simulation_clock.now())
        )
        observation = self.state_space.extract_state()

        terminated = metrics.get("success_rate", 1.0) <= 0.0
        truncated = self.current_step >= self.max_steps

        self.episode_rewards.append(reward)
        self.episode_metrics.append(metrics)

        info = {
            "step": self.current_step,
            "action": action,
            "configuration": config.to_dict(),
            "metrics": metrics,
            "raw_reward": raw_reward,
            "normalized_reward": reward,
            "cumulative_reward": sum(self.episode_rewards),
            "simulation_time_sec": self.simulation_clock.now(),
            "workload": self._serialize_workload(self.current_workload),
        }
        return observation, reward, terminated, truncated, info

    def _execute_function(self, config) -> Dict[str, Any]:
        """Execute one environment step."""
        if self.enable_real_execution:
            return self._execute_real(config)
        return self._execute_simulated_batch(config)

    def _execute_real(self, config) -> Dict[str, Any]:
        """Best-effort real execution path through SeBS."""
        cmd = [
            str(self.sebs_root / "sebs.py"),
            "benchmark",
            "invoke",
            self.benchmark,
            "test",
            "--config",
            self.config_path,
            "--deployment",
            self.deployment,
            "--verbose",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.sebs_root),
                capture_output=True,
                text=True,
                timeout=config.timeout_sec + 10,
            )
            if result.returncode != 0:
                print(f"[Error] SeBS execution failed: {result.stderr}")
                return {
                    "latency": 10000.0,
                    "cost": 0.01,
                    "success": False,
                    "cold_start": True,
                    "success_rate": 0.0,
                    "cold_start_rate": 1.0,
                    "timeout_rate": 1.0,
                }

            return self._execute_simulated_batch(config)
        except subprocess.TimeoutExpired:
            print("[Error] Function execution timed out")
            return {
                "latency": config.timeout_sec * 1000.0,
                "cost": self._estimate_cost(config, config.timeout_sec * 1000.0),
                "success": False,
                "cold_start": False,
                "success_rate": 0.0,
                "cold_start_rate": 0.0,
                "timeout_rate": 1.0,
            }
        except Exception as exc:
            print(f"[Error] Execution raised an exception: {exc}")
            return {
                "latency": 10000.0,
                "cost": 0.01,
                "success": False,
                "cold_start": True,
                "success_rate": 0.0,
                "cold_start_rate": 1.0,
                "timeout_rate": 1.0,
            }

    def _execute_simulated_batch(self, config) -> Dict[str, Any]:
        """Simulate one workload window with a calibrated service model."""
        workload = self.current_workload
        window_start_sec = self.simulation_clock.now()
        window_end_sec = window_start_sec + workload.step_duration_sec
        ttl_sec = self.service_model.estimate_ttl_sec(
            self.benchmark,
            config.memory_mb,
            config.timeout_sec,
            config.architecture,
        )
        self.container_pool.set_ttl(ttl_sec)
        self.state_space.load_monitor.container_ttl = ttl_sec

        arrival_count = max(0, int(workload.arrival_count))
        if arrival_count == 0:
            self.simulation_clock.advance(workload.step_duration_sec)
            return {
                "latency": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "cost": 0.0,
                "total_cost": 0.0,
                "success": True,
                "success_rate": 1.0,
                "cold_start": False,
                "cold_start_rate": 0.0,
                "timeout_rate": 0.0,
                "arrival_count": 0,
                "peak_concurrency": 0,
                "queue_ratio": 0.0,
                "cpu_utilization": 0.0,
                "memory_utilization": 0.0,
                "window_start_sec": window_start_sec,
                "window_end_sec": window_end_sec,
                "load_events_applied": True,
            }

        arrival_rate_per_sec = arrival_count / max(workload.step_duration_sec, 1e-6)
        warm_feasibility = self.service_model.get_exact_warm_feasibility(
            benchmark=self.benchmark,
            memory_mb=config.memory_mb,
            timeout_sec=config.timeout_sec,
            architecture=config.architecture,
        )
        if warm_feasibility is not None and not warm_feasibility.is_feasible:
            return self._simulate_infeasible_batch(
                config=config,
                workload=workload,
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                arrival_count=arrival_count,
                feasibility_hint=warm_feasibility,
            )

        burst_feasibility = self.service_model.get_exact_burst_feasibility(
            benchmark=self.benchmark,
            memory_mb=config.memory_mb,
            timeout_sec=config.timeout_sec,
            architecture=config.architecture,
            arrival_rate_per_sec=arrival_rate_per_sec,
        )
        if burst_feasibility is not None and not burst_feasibility.is_feasible:
            return self._simulate_infeasible_batch(
                config=config,
                workload=workload,
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                arrival_count=arrival_count,
                feasibility_hint=burst_feasibility,
            )

        arrival_times_sec = self._sample_arrival_times_sec(
            workload=workload,
            window_start_sec=window_start_sec,
            arrival_count=arrival_count,
        )

        warm_service_times_ms = np.asarray(
            [
                self.service_model.sample_warm_runtime_ms(
                    benchmark=self.benchmark,
                    memory_mb=config.memory_mb,
                    timeout_sec=config.timeout_sec,
                    architecture=config.architecture,
                    arrival_rate_per_sec=arrival_rate_per_sec,
                )
                for _ in range(arrival_count)
            ],
            dtype=np.float32,
        )

        idle_gap_sec = 0.0
        if self.state_space.load_monitor.last_request_time is not None:
            idle_gap_sec = max(
                0.0,
                window_start_sec - self.state_space.load_monitor.last_request_time,
            )
        cold_overheads_ms = np.asarray(
            [
                self.service_model.sample_cold_overhead_ms(
                    benchmark=self.benchmark,
                    memory_mb=config.memory_mb,
                    timeout_sec=config.timeout_sec,
                    architecture=config.architecture,
                    idle_gap_sec=idle_gap_sec,
                )
                for _ in range(arrival_count)
            ],
            dtype=np.float32,
        )

        batch = self.container_pool.simulate_batch(
            arrival_times_sec=arrival_times_sec,
            warm_service_times_ms=warm_service_times_ms,
            cold_overheads_ms=cold_overheads_ms,
        )

        latencies_ms = np.minimum(
            batch["latencies_ms"].astype(np.float32),
            config.timeout_sec * 1000.0,
        )
        completion_times_sec = batch["completion_times_sec"].astype(np.float32)
        cold_flags = batch["cold_flags"].astype(bool)
        queue_flags = batch["queue_flags"].astype(bool)
        timeout_flags = batch["latencies_ms"] > config.timeout_sec * 1000.0
        success_flags = ~timeout_flags

        self.state_space.record_batch(
            arrival_times=arrival_times_sec,
            completion_times=completion_times_sec,
            response_times=latencies_ms / 1000.0,
            cold_start_flags=cold_flags,
        )
        self.simulation_clock.advance(workload.step_duration_sec)

        per_request_costs = np.asarray(
            [self._estimate_cost(config, float(latency_ms)) for latency_ms in latencies_ms],
            dtype=np.float32,
        )

        mean_latency = float(latencies_ms.mean())
        p50_latency = float(np.percentile(latencies_ms, 50))
        p95_latency = float(np.percentile(latencies_ms, 95))
        p99_latency = float(np.percentile(latencies_ms, 99))
        mean_cost = float(per_request_costs.mean())
        total_cost = float(per_request_costs.sum())
        success_rate = float(success_flags.mean())
        cold_start_rate = float(cold_flags.mean())
        timeout_rate = float(timeout_flags.mean())
        queue_ratio = float(queue_flags.mean())

        cpu_utilization = min(
            1.0,
            batch["peak_concurrency"] / max(self.container_pool.max_containers, 1),
        )
        memory_utilization = min(
            1.0,
            (p95_latency / max(config.timeout_sec * 1000.0, 1.0)) * 1.2,
        )

        return {
            "latency": mean_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "cost": mean_cost,
            "total_cost": total_cost,
            "success": success_rate > 0.0,
            "success_rate": success_rate,
            "cold_start": cold_start_rate > 0.0,
            "cold_start_rate": cold_start_rate,
            "timeout_rate": timeout_rate,
            "arrival_count": arrival_count,
            "peak_concurrency": int(batch["peak_concurrency"]),
            "queue_ratio": queue_ratio,
            "arrival_burstiness": self._estimate_within_window_burstiness(workload),
            "cpu_utilization": cpu_utilization,
            "memory_utilization": memory_utilization,
            "window_start_sec": window_start_sec,
            "window_end_sec": window_end_sec,
            "load_events_applied": True,
        }

    def _simulate_infeasible_batch(
        self,
        config,
        workload: WorkloadStep,
        window_start_sec: float,
        window_end_sec: float,
        arrival_count: int,
        feasibility_hint: ExactFeasibilityHint,
    ) -> Dict[str, Any]:
        """Simulate a batch forced to fail by exact calibration evidence."""
        arrival_times_sec = self._sample_arrival_times_sec(
            workload=workload,
            window_start_sec=window_start_sec,
            arrival_count=arrival_count,
        )
        timeout_ms = float(config.timeout_sec * 1000.0)
        forced_service_times_ms = np.full(arrival_count, timeout_ms, dtype=np.float32)
        cold_overheads_ms = np.zeros(arrival_count, dtype=np.float32)

        batch = self.container_pool.simulate_batch(
            arrival_times_sec=arrival_times_sec,
            warm_service_times_ms=forced_service_times_ms,
            cold_overheads_ms=cold_overheads_ms,
        )

        latencies_ms = np.full(arrival_count, timeout_ms, dtype=np.float32)
        completion_times_sec = np.maximum(
            batch["completion_times_sec"].astype(np.float32),
            arrival_times_sec + timeout_ms / 1000.0,
        )
        cold_flags = batch["cold_flags"].astype(bool)
        queue_flags = batch["queue_flags"].astype(bool)
        timeout_flags = np.ones(arrival_count, dtype=bool)
        success_flags = np.zeros(arrival_count, dtype=bool)

        self.state_space.record_batch(
            arrival_times=arrival_times_sec,
            completion_times=completion_times_sec,
            response_times=latencies_ms / 1000.0,
            cold_start_flags=cold_flags,
        )
        self.simulation_clock.advance(workload.step_duration_sec)

        per_request_costs = np.asarray(
            [self._estimate_cost(config, float(latency_ms)) for latency_ms in latencies_ms],
            dtype=np.float32,
        )

        mean_latency = float(latencies_ms.mean())
        p50_latency = float(np.percentile(latencies_ms, 50))
        p95_latency = float(np.percentile(latencies_ms, 95))
        p99_latency = float(np.percentile(latencies_ms, 99))
        mean_cost = float(per_request_costs.mean())
        total_cost = float(per_request_costs.sum())
        success_rate = float(success_flags.mean())
        cold_start_rate = float(cold_flags.mean())
        timeout_rate = float(timeout_flags.mean())
        queue_ratio = float(queue_flags.mean())

        cpu_utilization = min(
            1.0,
            batch["peak_concurrency"] / max(self.container_pool.max_containers, 1),
        )
        memory_utilization = 1.0

        metrics = {
            "latency": mean_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "cost": mean_cost,
            "total_cost": total_cost,
            "success": success_rate > 0.0,
            "success_rate": success_rate,
            "cold_start": cold_start_rate > 0.0,
            "cold_start_rate": cold_start_rate,
            "timeout_rate": timeout_rate,
            "arrival_count": arrival_count,
            "peak_concurrency": int(batch["peak_concurrency"]),
            "queue_ratio": queue_ratio,
            "arrival_burstiness": self._estimate_within_window_burstiness(workload),
            "cpu_utilization": cpu_utilization,
            "memory_utilization": memory_utilization,
            "window_start_sec": window_start_sec,
            "window_end_sec": window_end_sec,
            "load_events_applied": True,
            "infeasible_by_calibration": True,
            "infeasible_source": feasibility_hint.source,
            "infeasible_profile_source": feasibility_hint.profile_source,
        }
        if feasibility_hint.matched_concurrency is not None:
            metrics["infeasible_matched_concurrency"] = float(
                feasibility_hint.matched_concurrency
            )
        return metrics

    def _sample_arrival_times_sec(
        self,
        workload: WorkloadStep,
        window_start_sec: float,
        arrival_count: int,
    ) -> np.ndarray:
        """Sample intra-window arrival timestamps with burstiness-aware clustering."""
        if arrival_count <= 0:
            return np.asarray([], dtype=np.float32)
        if arrival_count == 1:
            return np.asarray([window_start_sec], dtype=np.float32)

        rng = getattr(self, "np_random", np.random.default_rng())
        burstiness = self._estimate_within_window_burstiness(workload)
        duration_sec = max(float(workload.step_duration_sec), 1e-6)

        if burstiness <= 1.05:
            offsets = np.sort(rng.uniform(0.0, duration_sec, size=arrival_count))
            return window_start_sec + offsets.astype(np.float32)

        n_bins = int(np.clip(max(4, arrival_count // 2), 4, 16))
        alpha = float(np.clip(1.4 / burstiness, 0.08, 2.5))
        weights = rng.dirichlet(np.full(n_bins, alpha, dtype=np.float64))
        counts = rng.multinomial(arrival_count, weights)

        offsets = []
        for bin_index, count in enumerate(counts):
            if count <= 0:
                continue
            left = duration_sec * bin_index / n_bins
            right = duration_sec * (bin_index + 1) / n_bins
            offsets.extend(rng.uniform(left, right, size=count))

        if not offsets:
            return np.asarray([window_start_sec] * arrival_count, dtype=np.float32)
        return window_start_sec + np.sort(np.asarray(offsets, dtype=np.float32))

    def _estimate_within_window_burstiness(self, workload: WorkloadStep) -> float:
        """Estimate short-term burstiness from workload statistics."""
        mean_rate = max(float(workload.mean_invocations_per_minute), 1e-6)
        std_rate = max(float(workload.std_invocations_per_minute), 0.0)
        max_rate = max(float(workload.max_invocations_per_minute), mean_rate)
        peak_ratio = max_rate / mean_rate
        dispersion = std_rate / max(np.sqrt(mean_rate), 1.0)
        hint = max(float(getattr(workload, "burstiness_hint", 1.0)), 1.0)
        return float(max(1.0, 0.45 * peak_ratio + 0.30 * dispersion + 0.25 * hint))

    def _estimate_cost(self, config, latency_ms: float) -> float:
        """Estimate Lambda-style cost for one request."""
        gb = config.memory_mb / 1024.0
        seconds = latency_ms / 1000.0
        if config.architecture == "arm64":
            price_per_gb_second = 0.0000133334
        else:
            price_per_gb_second = 0.0000166667
        cost = gb * seconds * price_per_gb_second
        cost += 0.0000002
        return float(cost)

    def _compute_reward(self, metrics: Dict[str, Any]) -> float:
        """Compute the aggregated reward for one workload window."""
        latency_signal = metrics.get("p95_latency", metrics["latency"])
        normalized_latency = latency_signal / self.baseline_latency

        if self.cost_normalization == "log":
            normalized_cost = math.log1p(metrics["cost"] / self.baseline_cost)
        elif self.cost_normalization == "sqrt":
            normalized_cost = math.sqrt(metrics["cost"] / self.baseline_cost)
        else:
            normalized_cost = metrics["cost"] / self.baseline_cost

        reward = -(
            self.reward_weights["latency"] * normalized_latency
            + self.reward_weights["cost"] * normalized_cost
        )

        arrival_count = metrics.get("arrival_count", 1)
        if arrival_count > 0:
            failure_rate = metrics.get(
                "failure_rate",
                1.0 - metrics.get("success_rate", float(metrics.get("success", True))),
            )
            reward += self.success_bonus * metrics.get("success_rate", 1.0)
            reward -= self.reward_penalties["failure"] * failure_rate
            reward -= (
                self.reward_penalties["cold_start"]
                * metrics.get("cold_start_rate", 0.0)
            )
            reward -= self.reward_penalties["queue"] * metrics.get("queue_ratio", 0.0)

        return float(reward)

    def _update_and_normalize_reward(self, reward: float) -> float:
        """Normalize rewards online with Welford statistics."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2

        reward_std = (
            np.sqrt(self.reward_m2 / (self.reward_count - 1))
            if self.reward_count > 1
            else 1.0
        )
        if reward_std > 1e-6:
            return float((reward - self.reward_mean) / reward_std)
        return float(reward - self.reward_mean)

    def get_episode_summary(self) -> Dict:
        """Return a compact episode summary."""
        if not self.episode_metrics:
            return {}

        return {
            "total_steps": self.current_step,
            "total_reward": float(sum(self.episode_rewards)),
            "mean_reward": float(np.mean(self.episode_rewards)),
            "mean_latency": float(np.mean([m["latency"] for m in self.episode_metrics])),
            "mean_p95_latency": float(
                np.mean([m.get("p95_latency", m["latency"]) for m in self.episode_metrics])
            ),
            "mean_cost": float(np.mean([m["cost"] for m in self.episode_metrics])),
            "success_rate": float(
                np.mean([m.get("success_rate", float(m["success"])) for m in self.episode_metrics])
            ),
            "cold_start_rate": float(
                np.mean(
                    [
                        m.get("cold_start_rate", float(m["cold_start"]))
                        for m in self.episode_metrics
                    ]
                )
            ),
            "timeout_rate": float(
                np.mean([m.get("timeout_rate", 0.0) for m in self.episode_metrics])
            ),
        }

    def render(self):
        """No-op render hook."""
        return None

    def close(self):
        """No-op close hook."""
        return None

    def _serialize_workload(self, workload: WorkloadStep) -> Dict[str, Any]:
        """Convert a workload step into a JSON-friendly dict."""
        return {
            "arrival_count": int(workload.arrival_count),
            "step_duration_sec": float(workload.step_duration_sec),
            "source_name": workload.source_name,
            "load_value": float(workload.load_value),
            "minute_of_day": int(workload.minute_of_day),
            "profile_key": workload.profile_key,
            "mean_invocations_per_minute": float(workload.mean_invocations_per_minute),
            "std_invocations_per_minute": float(workload.std_invocations_per_minute),
            "max_invocations_per_minute": float(workload.max_invocations_per_minute),
        }
