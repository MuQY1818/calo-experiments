#!/usr/bin/env python3
"""Run a small real-platform online A/B comparison on OpenWhisk."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_optimizer.baselines import DefaultBaseline
from rl_optimizer.environment import ServerlessFunctionEnv
from rl_optimizer.pure_ppo import PurePPO
from rl_optimizer.workload_sources import WorkloadStep
from run_dynamic_experiment import _build_dynamic_env, _build_state_space_for_algorithm
from sebs.faas.function import ExecutionResult, Trigger
from sebs.sebs import SeBS


SUPPORTED_POLICIES = ("ppo", "ppo_load_only", "bayes_opt_online", "default")
INITIAL_PRIME_SETTLE_MAX_ATTEMPTS = 3
INITIAL_PRIME_SETTLE_SLEEP_SEC = 5.0
INITIAL_PRIME_SETTLE_P95_THRESHOLD_MS = 5000.0


@dataclass
class PolicyRunArtifact:
    """In-memory handle for one trained or fixed policy."""

    policy_name: str
    display_name: str
    env_helper: ServerlessFunctionEnv
    train_time_sec: float = 0.0
    model: Any | None = None
    default_action: int | None = None
    policy_controller: Any | None = None
    shadow_env: Any | None = None
    feasible_actions: set[int] | None = None


@dataclass
class RealDeploymentHandle:
    """Reusable OpenWhisk deployment handle for one online policy replay."""

    sebs_client: SeBS
    deployment_client: Any
    openwhisk_settings: dict[str, Any]
    benchmark_name: str
    function_name: str
    benchmark_handle: Any
    input_config: dict[str, Any]
    trigger: Any | None = None
    current_architecture: str = "x64"
    current_memory_mb: int | None = None
    current_timeout_sec: int | None = None


@dataclass
class ActionStabilizationState:
    """Tracks short-term online action smoothing state."""

    last_effective_action: int | None = None
    steps_since_switch: int = 0
    pending_anchor_upgrade_action: int | None = None
    pending_anchor_upgrade_count: int = 0
    pending_upgrade_action: int | None = None
    pending_upgrade_count: int = 0
    pending_downgrade_action: int | None = None
    pending_downgrade_count: int = 0
    last_step_metrics: dict[str, float] = field(default_factory=dict)
    last_step_raw_reward: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a focused OpenWhisk online deployment study.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/rl_experiments/full_suite.json"),
        help="Experiment config used for reward/action/workload defaults.",
    )
    parser.add_argument(
        "--openwhisk-config",
        type=Path,
        default=Path("config/openwhisk_standalone.example.json"),
        help="Path to the OpenWhisk standalone SeBS config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/review_real_platform_online_ab"),
        help="Directory for raw OpenWhisk outputs and summaries.",
    )
    parser.add_argument(
        "--batch-spec",
        type=Path,
        default=None,
        help=(
            "Optional JSON batch spec. When provided, the script runs multiple "
            "cases under the base output directory and writes one batch summary."
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="411.image-recognition",
        help="Benchmark under test.",
    )
    parser.add_argument(
        "--load-pattern",
        type=str,
        default="spike",
        help="Workload label passed to the dynamic workload source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for workload generation and PPO training.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of control steps in the real-platform replay episode.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=12000,
        help="PPO simulator-training budget before real deployment replay.",
    )
    parser.add_argument(
        "--max-real-invocations-per-step",
        type=int,
        default=8,
        help="Cap real OpenWhisk repetitions per control step.",
    )
    parser.add_argument(
        "--min-real-invocations-per-step",
        type=int,
        default=1,
        help="Minimum repetitions per control step after capping.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ppo", "default"],
        choices=SUPPORTED_POLICIES,
        help="Policies to compare online on the real platform.",
    )
    parser.add_argument(
        "--policy-order-mode",
        type=str,
        default="specified",
        choices=["specified", "rotate_by_case"],
        help=(
            "Execution order for real-platform replay. 'rotate_by_case' "
            "deterministically rotates policy order per case to reduce "
            "first-run platform bias."
        ),
    )
    parser.add_argument(
        "--inter-policy-idle-sec",
        type=float,
        default=0.0,
        help="Optional idle gap between policy replays within one case.",
    )
    parser.add_argument(
        "--real-invoke-mode",
        type=str,
        default="parallel",
        choices=["parallel", "sequential"],
        help="Use direct concurrent trigger replay or the legacy sequential CLI replay.",
    )
    parser.add_argument(
        "--enforce-feasible-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict online replay to actions that were observed as deployable in the "
            "completed fixed-configuration validation sweep."
        ),
    )
    parser.add_argument(
        "--feasible-actions-source",
        type=Path,
        default=Path(
            "results/"
            "review_real_platform_validation_x64_full24_all5_live_20260413/"
            "validation_summary.json"
        ),
        help=(
            "Path to validation_summary.json used to derive the real-platform "
            "deployability mask."
        ),
    )
    parser.add_argument(
        "--stabilize-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a lightweight online action stabilizer during real replay.",
    )
    parser.add_argument(
        "--stabilization-min-dwell-steps",
        type=int,
        default=2,
        help="Minimum number of steps to stay on a config before switching away.",
    )
    parser.add_argument(
        "--stabilization-downgrade-patience",
        type=int,
        default=2,
        help="Required consecutive downgrade proposals before applying a downgrade.",
    )
    parser.add_argument(
        "--stabilization-load-drop-threshold",
        type=float,
        default=0.95,
        help=(
            "Maximum allowed current/previous workload ratio before a downgrade "
            "is considered safe."
        ),
    )
    parser.add_argument(
        "--function-prefix",
        type=str,
        default="review-online",
        help="Prefix for OpenWhisk function names.",
    )
    parser.add_argument(
        "--initial-prime-repetitions",
        type=int,
        default=0,
        help=(
            "Optional uncounted warmup replay before measured steps. "
            "Uses the initial policy action and the given repetition count."
        ),
    )
    parser.add_argument(
        "--initial-prime-lookahead-steps",
        type=int,
        default=0,
        help=(
            "Optional lookahead horizon used to size the initial warmup "
            "against the early burst peak, with --initial-prime-repetitions "
            "still acting as the upper bound."
        ),
    )
    parser.add_argument(
        "--reconfiguration-prime-repetitions",
        type=int,
        default=0,
        help=(
            "Optional uncounted warmup replay after each configuration switch "
            "before measuring the step."
        ),
    )
    parser.add_argument(
        "--default-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep CALO anchored to the provider default until sustained pressure "
            "justifies an upgrade."
        ),
    )
    parser.add_argument(
        "--default-anchor-release-patience",
        type=int,
        default=2,
        help="Required consecutive upgrade proposals before leaving the default action.",
    )
    parser.add_argument(
        "--default-anchor-min-arrivals",
        type=int,
        default=10,
        help="Minimum real arrivals in one step before the default anchor may release.",
    )
    parser.add_argument(
        "--default-anchor-load-ratio-threshold",
        type=float,
        default=1.35,
        help="Minimum current/previous workload ratio before the default anchor may release.",
    )
    parser.add_argument(
        "--default-anchor-min-burstiness",
        type=float,
        default=1.55,
        help="Minimum burstiness hint before the default anchor may release.",
    )
    parser.add_argument(
        "--default-anchor-min-step",
        type=int,
        default=2,
        help="Earliest control step at which CALO may leave the default anchor.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep existing raw outputs instead of recreating the output directory.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load the JSON experiment config."""
    return json.loads(path.read_text())


def load_batch_spec(path: Path) -> dict[str, Any]:
    """Load a JSON batch specification."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Batch spec must be a JSON object.")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Batch spec must contain a non-empty 'cases' list.")
    return payload


def _build_openwhisk_experiment_config(
    openwhisk_settings: dict[str, Any],
    architecture: str,
) -> dict[str, Any]:
    """Create a mutable SeBS experiment config for direct OpenWhisk replay."""
    experiment_config = copy.deepcopy(openwhisk_settings.get("experiments", {}))
    experiment_config["deployment"] = "openwhisk"
    experiment_config["update_code"] = False
    experiment_config["update_storage"] = False
    experiment_config["download_results"] = False
    experiment_config["container_deployment"] = True
    experiment_config["architecture"] = architecture
    return experiment_config


def _sanitize_case_name(raw_name: str) -> str:
    """Convert a case label into a filesystem-friendly directory name."""
    sanitized = []
    for char in raw_name:
        if char.isalnum() or char in ("-", "_"):
            sanitized.append(char)
        else:
            sanitized.append("_")
    collapsed = "".join(sanitized).strip("_")
    return collapsed or "case"


def _resolve_policy_execution_order(
    policies: list[str],
    benchmark: str,
    load_pattern: str,
    seed: int,
    mode: str,
) -> list[str]:
    """Return one deterministic policy replay order for the current case."""
    ordered_policies = [str(policy) for policy in policies]
    if str(mode) != "rotate_by_case" or len(ordered_policies) <= 1:
        return ordered_policies

    case_signature = f"{benchmark}:{load_pattern}:{seed}"
    rotation = sum(ord(char) for char in case_signature) % len(ordered_policies)
    if rotation == 0:
        return ordered_policies
    return ordered_policies[rotation:] + ordered_policies[:rotation]


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite int or float."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_feasible_action_set(
    validation_summary_path: Path,
    benchmark: str,
    action_space,
    architecture: str = "x64",
) -> set[int]:
    """Load deployable actions for one benchmark from fixed-validation results."""
    if not validation_summary_path.exists():
        raise FileNotFoundError(
            f"Feasible action source does not exist: {validation_summary_path}"
        )

    payload = json.loads(validation_summary_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(
            "Expected validation_summary.json to contain a JSON list of rows."
        )

    feasible_actions: set[int] = set()
    for row in payload:
        if str(row.get("benchmark")) != str(benchmark):
            continue
        if str(row.get("architecture")) != str(architecture):
            continue
        if not _is_finite_number(row.get("real_warm_mean_ms")):
            continue

        memory_mb = int(row["memory_mb"])
        timeout_sec = int(row["timeout_sec"])
        action_id = action_space.get_action_id(memory_mb, architecture, timeout_sec)
        feasible_actions.add(int(action_id))

    if not feasible_actions:
        raise ValueError(
            "No feasible actions were loaded for "
            f"{benchmark} from {validation_summary_path}."
        )
    return feasible_actions


def _copy_state_space_runtime(target_state_space, source_state_space) -> None:
    """Copy only dynamic state-space signals while reusing static code features."""
    target_state_space.load_monitor = copy.deepcopy(source_state_space.load_monitor)
    target_state_space.performance_history = copy.deepcopy(
        source_state_space.performance_history
    )
    target_state_space.config_context = copy.deepcopy(
        source_state_space.config_context
    )
    target_state_space.simulation_time_sec = float(
        getattr(source_state_space, "simulation_time_sec", 0.0)
    )
    target_state_space.load_monitor.set_current_time(
        target_state_space.simulation_time_sec
    )


def _clone_dynamic_env_runtime(env):
    """Clone one dynamic env runtime state without re-running code analysis."""
    from rl_optimizer.dynamic_environment import DynamicLoadEnvironment

    source_env = env
    source_base_env = source_env.base_env
    action_space_config = {
        "memory_options": list(source_base_env.action_space_wrapper.MEMORY_OPTIONS),
        "architecture_options": list(
            source_base_env.action_space_wrapper.ARCHITECTURE_OPTIONS
        ),
        "timeout_options": list(source_base_env.action_space_wrapper.TIMEOUT_OPTIONS),
    }

    if hasattr(source_base_env.state_space, "fork_for_env"):
        forked_state_space = source_base_env.state_space.fork_for_env()
        _copy_state_space_runtime(
            target_state_space=forked_state_space,
            source_state_space=source_base_env.state_space,
        )
    else:
        forked_state_space = copy.deepcopy(source_base_env.state_space)

    shadow_base_env = ServerlessFunctionEnv(
        benchmark=source_base_env.benchmark,
        deployment=source_base_env.deployment,
        config_path=source_base_env.config_path,
        sebs_root=str(source_base_env.sebs_root),
        max_steps=source_base_env.max_steps,
        reward_weights=dict(source_base_env.reward_weights),
        reward_penalties=dict(source_base_env.reward_penalties),
        enable_real_execution=source_base_env.enable_real_execution,
        normalize_reward=source_base_env.normalize_reward,
        success_bonus=source_base_env.success_bonus,
        cost_normalization=source_base_env.cost_normalization,
        calibration_dir=getattr(source_base_env.service_model, "calibration_dir", None),
        calibration_dirs=getattr(
            source_base_env.service_model,
            "calibration_dirs",
            None,
        ),
        step_duration_sec=source_base_env.step_duration_sec,
        max_containers=source_base_env.container_pool.max_containers,
        container_ttl_sec=source_base_env.container_pool.ttl_sec,
        state_space=forked_state_space,
        action_space_config=action_space_config,
        log_init=False,
    )
    shadow_env = DynamicLoadEnvironment(
        shadow_base_env,
        episode_length=source_env.episode_length,
        load_change_freq=source_env.load_change_freq,
        switch_penalty=source_env.switch_penalty,
        adaptation_penalty=source_env.adaptation_penalty,
        workload_source=source_env.default_workload_source,
        azure_profile_path=str(source_env.azure_profile_path),
        azure_summary_path=(
            str(source_env.azure_summary_path)
            if source_env.azure_summary_path is not None
            else None
        ),
        azure_top_k=source_env.azure_top_k,
        azure_arrival_scale=source_env.azure_arrival_scale,
        azure_max_arrivals_per_step=source_env.azure_max_arrivals_per_step,
        azure_profile_selection=source_env.azure_profile_selection,
        azure_selection_pool_size=source_env.azure_selection_pool_size,
        azure_target_concurrency=source_env.azure_target_concurrency,
    )
    shadow_env.simulation_clock.set(source_env.simulation_clock.now())
    shadow_env.container_pool._containers = copy.deepcopy(
        source_env.container_pool._containers
    )
    shadow_env.base_env.attach_simulation_backend(
        simulation_clock=shadow_env.simulation_clock,
        service_model=shadow_env.service_model,
        container_pool=shadow_env.container_pool,
    )
    shadow_env.base_env.current_step = int(source_base_env.current_step)
    shadow_env.base_env.reward_mean = float(source_base_env.reward_mean)
    shadow_env.base_env.reward_m2 = float(source_base_env.reward_m2)
    shadow_env.base_env.reward_count = int(source_base_env.reward_count)
    shadow_env.base_env.episode_rewards = list(source_base_env.episode_rewards)
    shadow_env.base_env.episode_metrics = copy.deepcopy(
        source_base_env.episode_metrics
    )
    shadow_env.base_env.current_workload = copy.deepcopy(
        source_base_env.current_workload
    )
    shadow_env.base_env.state_space.set_simulation_time(
        getattr(source_base_env.state_space, "simulation_time_sec", 0.0)
    )
    shadow_env.current_step = int(source_env.current_step)
    shadow_env.last_action = source_env.last_action
    shadow_env.load_pattern = str(source_env.load_pattern)
    shadow_env.load_recently_changed = bool(source_env.load_recently_changed)
    shadow_env.workload_source = str(source_env.workload_source)
    shadow_env._current_workload = copy.deepcopy(source_env._current_workload)
    shadow_env.base_env.set_workload_step(
        copy.deepcopy(source_env.base_env.current_workload)
    )
    shadow_env._workload_driver = copy.deepcopy(source_env._workload_driver)
    shadow_env.rng = copy.deepcopy(source_env.rng)
    return shadow_env


def _sync_shadow_env_runtime(
    shadow_env,
    source_state_space,
    current_workload: WorkloadStep,
    current_step: int,
    current_time_sec: float,
    last_action: int | None,
    load_recently_changed: bool,
) -> None:
    """Align one simulator shadow env with the latest real replay state."""
    _copy_state_space_runtime(
        target_state_space=shadow_env.base_env.state_space,
        source_state_space=source_state_space,
    )
    shadow_env.simulation_clock.set(float(current_time_sec))
    shadow_env.base_env.state_space.set_simulation_time(float(current_time_sec))
    shadow_env.current_step = int(current_step)
    shadow_env.base_env.current_step = int(current_step)
    shadow_env.last_action = None if last_action is None else int(last_action)
    shadow_env.load_recently_changed = bool(load_recently_changed)
    shadow_env._current_workload = copy.deepcopy(current_workload)
    shadow_env.base_env.set_workload_step(copy.deepcopy(current_workload))


class RealReplayBayesOptController:
    """Bayesian optimization controller for real-platform replay."""

    def __init__(
        self,
        shadow_env,
        feasible_actions: set[int] | None = None,
        reoptimize_freq: int = 10,
        n_iterations: int = 20,
        seed: int | None = None,
    ):
        self.shadow_env = shadow_env
        self.action_space = shadow_env.action_space_wrapper
        self.feasible_actions = (
            sorted(int(action) for action in feasible_actions)
            if feasible_actions
            else list(range(self.action_space.n_actions))
        )
        self.reoptimize_freq = max(1, int(reoptimize_freq))
        self.n_iterations = max(1, int(n_iterations))
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.current_best_action: int | None = None
        self.step_count = 0
        self.observations: list[tuple[int, float]] = []
        self.best_reward = float("-inf")

    def reset(self) -> None:
        """Reset per-episode optimizer state."""
        self.current_best_action = None
        self.step_count = 0
        self.observations = []
        self.best_reward = float("-inf")
        self.rng = np.random.default_rng(self.seed)

    def _evaluate_action(self, action: int) -> float:
        """Score one candidate action with a one-step simulator lookahead."""
        probe_env = _clone_dynamic_env_runtime(self.shadow_env)
        _, reward, _, _, info = probe_env.step(int(action))
        return float(info.get("raw_reward", reward))

    def _random_search(self) -> int:
        """Fallback random search when sklearn is unavailable."""
        sampled_actions = list(self.feasible_actions)
        self.rng.shuffle(sampled_actions)
        for action in sampled_actions[: self.n_iterations]:
            reward = self._evaluate_action(int(action))
            self.observations.append((int(action), reward))
            if reward > self.best_reward:
                self.best_reward = reward
                self.current_best_action = int(action)
        return int(self.current_best_action)

    def optimize(self) -> int:
        """Run one discrete BayesOpt round on the synced shadow env."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            return self._random_search()

        candidate_actions = list(self.feasible_actions)
        if not candidate_actions:
            raise ValueError("RealReplayBayesOptController requires feasible actions.")

        initial_budget = min(10, self.n_iterations, len(candidate_actions))
        initial_actions = list(
            self.rng.choice(candidate_actions, size=initial_budget, replace=False)
        )
        observed_actions = set()
        self.observations = []
        self.best_reward = float("-inf")
        self.current_best_action = int(initial_actions[0])

        for action in initial_actions:
            reward = self._evaluate_action(int(action))
            self.observations.append((int(action), reward))
            observed_actions.add(int(action))
            if reward > self.best_reward:
                self.best_reward = reward
                self.current_best_action = int(action)

        if len(observed_actions) == len(candidate_actions) or len(observed_actions) < 2:
            return int(self.current_best_action)

        for _ in range(initial_budget, min(self.n_iterations, len(candidate_actions))):
            X = np.asarray([obs[0] for obs in self.observations], dtype=np.float64).reshape(-1, 1)
            y = np.asarray([obs[1] for obs in self.observations], dtype=np.float64)
            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=self.seed,
            )
            gp.fit(X, y)

            best_ucb = float("-inf")
            next_action: int | None = None
            for action in candidate_actions:
                if int(action) in observed_actions:
                    continue
                mean_reward, sigma = gp.predict(
                    np.asarray([[float(action)]], dtype=np.float64),
                    return_std=True,
                )
                ucb = float(mean_reward[0]) + 2.0 * float(sigma[0])
                if ucb > best_ucb:
                    best_ucb = ucb
                    next_action = int(action)

            if next_action is None:
                break

            reward = self._evaluate_action(next_action)
            self.observations.append((next_action, reward))
            observed_actions.add(next_action)
            if reward > self.best_reward:
                self.best_reward = reward
                self.current_best_action = next_action

        return int(self.current_best_action)

    def select_action(self, obs: np.ndarray) -> int:
        """Select one action for the current real replay step."""
        del obs
        if self.step_count % self.reoptimize_freq == 0 or self.current_best_action is None:
            self.current_best_action = self.optimize()
        self.step_count += 1
        return int(self.current_best_action)


def build_workload_sequence(
    benchmark: str,
    environment_cfg: dict[str, Any],
    load_pattern: str,
    steps: int,
    seed: int,
) -> tuple[list[WorkloadStep], dict[str, Any]]:
    """Generate one deterministic workload sequence shared across policies."""
    reference_env = _build_dynamic_env(
        benchmark=benchmark,
        environment_cfg=environment_cfg,
        load_pattern=load_pattern,
        algorithm="ppo",
        algorithm_settings=None,
    )
    options = {"load_pattern": load_pattern}
    if environment_cfg.get("workload_source") == "azure_trace":
        options["workload_source"] = "azure_trace"
    _, reset_info = reference_env.reset(seed=seed, options=options)

    workload_steps = [copy.deepcopy(reference_env._current_workload)]
    for step_index in range(1, steps):
        workload_steps.append(copy.deepcopy(reference_env._sample_workload(step_index)))
    return workload_steps, reset_info


def build_policy_artifact(
    policy_name: str,
    config: dict[str, Any],
    benchmark: str,
    load_pattern: str,
    seed: int,
    total_timesteps: int,
    enforce_feasible_actions: bool,
    feasible_actions_source: Path,
) -> PolicyRunArtifact:
    """Build and optionally train one policy in the calibrated simulator."""
    environment_cfg = dict(config["environment"])
    algorithm_settings = config.get("algorithm_settings")
    state_space_algorithm = policy_name if policy_name in {"ppo", "ppo_load_only"} else "ppo"
    env_helper = ServerlessFunctionEnv(
        benchmark=benchmark,
        deployment="local",
        enable_real_execution=False,
        normalize_reward=False,
        reward_weights=environment_cfg.get("reward_weights"),
        reward_penalties=environment_cfg.get("reward_penalties"),
        success_bonus=environment_cfg.get("success_bonus", 0.5),
        cost_normalization=environment_cfg.get("cost_normalization", "ratio"),
        calibration_dir=environment_cfg.get("calibration_dir"),
        step_duration_sec=environment_cfg.get("step_duration_sec", 15.0),
        max_containers=environment_cfg.get("max_containers", 32),
        container_ttl_sec=environment_cfg.get("container_ttl_sec", 600.0),
        state_space=_build_state_space_for_algorithm(
            state_space_algorithm,
            environment_cfg=environment_cfg,
            algorithm_settings=algorithm_settings,
        ),
        action_space_config=environment_cfg.get("action_space"),
    )
    feasible_actions = (
        load_feasible_action_set(
            validation_summary_path=feasible_actions_source,
            benchmark=benchmark,
            action_space=env_helper.action_space_wrapper,
            architecture="x64",
        )
        if enforce_feasible_actions
        else None
    )
    default_memory = int(config.get("baseline", {}).get("default_memory", 512))
    default_action = DefaultBaseline(env_helper, memory=default_memory).default_action

    if policy_name == "default":
        return PolicyRunArtifact(
            policy_name=policy_name,
            display_name="Default",
            env_helper=env_helper,
            default_action=default_action,
            feasible_actions=feasible_actions,
        )

    if policy_name == "bayes_opt_online":
        shadow_env = _build_dynamic_env(
            benchmark=benchmark,
            environment_cfg=environment_cfg,
            load_pattern=load_pattern,
            algorithm="ppo",
            algorithm_settings=algorithm_settings,
        )
        shadow_env.reset(seed=seed, options={"load_pattern": load_pattern})
        controller = RealReplayBayesOptController(
            shadow_env=shadow_env,
            feasible_actions=feasible_actions,
            reoptimize_freq=int(config.get("baseline", {}).get("reoptimize_freq", 10)),
            n_iterations=20,
            seed=seed,
        )
        return PolicyRunArtifact(
            policy_name=policy_name,
            display_name="Online BO",
            env_helper=env_helper,
            default_action=default_action,
            policy_controller=controller,
            shadow_env=shadow_env,
            feasible_actions=feasible_actions,
        )

    dynamic_env = _build_dynamic_env(
        benchmark=benchmark,
        environment_cfg=environment_cfg,
        load_pattern=load_pattern,
        algorithm=policy_name,
        algorithm_settings=algorithm_settings,
    )
    dynamic_env.reset(seed=seed, options={"load_pattern": load_pattern})

    training_cfg = dict(config["training"])
    ppo = PurePPO(
        dynamic_env,
        total_timesteps=total_timesteps,
        ppo_kwargs=training_cfg["ppo_kwargs"],
        seed=seed,
    )
    train_time_sec = float(
        ppo.train(progress_label=f"{benchmark} / {load_pattern} [{policy_name}]")
    )
    return PolicyRunArtifact(
        policy_name=policy_name,
        display_name="CALO" if policy_name == "ppo" else "Load-Only CALO",
        env_helper=env_helper,
        train_time_sec=train_time_sec,
        model=ppo.model,
        default_action=default_action,
        feasible_actions=feasible_actions,
    )


def _build_function_name(
    benchmark: str,
    policy_name: str,
    function_prefix: str,
    seed: int,
) -> str:
    """Build one stable function name per policy run."""
    compact_name = benchmark.replace(".", "-")
    return (
        f"{function_prefix}-{compact_name}-{policy_name}-"
        f"seed{seed}-{uuid.uuid4().hex[:8]}"
    )


def _flatten_invocations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten SeBS invocation records across function names."""
    invocations = []
    for function_invocations in payload.get("_invocations", {}).values():
        invocations.extend(function_invocations.values())
    return invocations


def create_real_deployment_handle(
    benchmark: str,
    openwhisk_config: Path,
    output_dir: Path,
    cache_dir: Path,
    function_name: str,
    architecture: str,
) -> RealDeploymentHandle:
    """Initialize one reusable OpenWhisk deployment client for direct replay."""
    os.environ["SEBS_WITH_OPENWHISK"] = "true"
    openwhisk_settings = load_config(openwhisk_config)
    sebs_output_dir = output_dir / "_sebs_runtime"
    sebs_cache_dir = cache_dir / "_sebs_runtime"
    sebs_output_dir.mkdir(parents=True, exist_ok=True)
    sebs_cache_dir.mkdir(parents=True, exist_ok=True)

    sebs_client = SeBS(
        cache_dir=str(sebs_cache_dir),
        output_dir=str(sebs_output_dir),
        verbose=False,
        logging_filename=str(sebs_output_dir / "openwhisk_runtime.log"),
    )
    deployment_client = sebs_client.get_deployment(
        openwhisk_settings,
        logging_filename=str(sebs_output_dir / "openwhisk_deployment.log"),
    )
    deployment_client.disable_rich_output()
    deployment_client.initialize(resource_prefix="review")

    experiment_cfg = sebs_client.get_experiment_config(
        _build_openwhisk_experiment_config(openwhisk_settings, architecture)
    )
    benchmark_handle = sebs_client.get_benchmark(benchmark, deployment_client, experiment_cfg)
    input_config = benchmark_handle.prepare_input(
        deployment_client.system_resources,
        size="test",
        replace_existing=experiment_cfg.update_storage,
    )
    return RealDeploymentHandle(
        sebs_client=sebs_client,
        deployment_client=deployment_client,
        openwhisk_settings=openwhisk_settings,
        benchmark_name=benchmark,
        function_name=function_name,
        benchmark_handle=benchmark_handle,
        input_config=input_config,
        current_architecture=architecture,
    )


def close_real_deployment_handle(handle: RealDeploymentHandle) -> None:
    """Close the direct OpenWhisk replay client."""
    try:
        handle.deployment_client.shutdown()
    finally:
        handle.sebs_client.shutdown()


def ensure_real_function_configuration(
    handle: RealDeploymentHandle,
    memory_mb: int,
    timeout_sec: int,
    architecture: str,
) -> Any:
    """Deploy or update the OpenWhisk function to the requested configuration."""
    if architecture != handle.current_architecture:
        experiment_cfg = handle.sebs_client.get_experiment_config(
            _build_openwhisk_experiment_config(handle.openwhisk_settings, architecture)
        )
        handle.benchmark_handle = handle.sebs_client.get_benchmark(
            handle.benchmark_name,
            handle.deployment_client,
            experiment_cfg,
        )
        handle.input_config = handle.benchmark_handle.prepare_input(
            handle.deployment_client.system_resources,
            size="test",
            replace_existing=experiment_cfg.update_storage,
        )
        handle.current_architecture = architecture
        handle.current_memory_mb = None
        handle.current_timeout_sec = None
        handle.trigger = None

    handle.benchmark_handle.benchmark_config.memory = int(memory_mb)
    handle.benchmark_handle.benchmark_config.timeout = int(timeout_sec)
    function = handle.deployment_client.get_function(
        handle.benchmark_handle,
        handle.function_name,
    )
    triggers = function.triggers(Trigger.TriggerType.LIBRARY)
    if triggers:
        handle.trigger = triggers[0]
    else:
        handle.trigger = handle.deployment_client.create_trigger(
            function,
            Trigger.TriggerType.LIBRARY,
        )
    handle.current_memory_mb = int(memory_mb)
    handle.current_timeout_sec = int(timeout_sec)
    return handle.trigger


def estimate_request_cost_usd(
    memory_mb: int,
    architecture: str,
    latency_ms: float,
) -> float:
    """Estimate request cost using the same pricing rule as the simulator."""
    gb = float(memory_mb) / 1024.0
    seconds = float(latency_ms) / 1000.0
    if architecture == "arm64":
        price_per_gb_second = 0.0000133334
    else:
        price_per_gb_second = 0.0000166667
    return float(gb * seconds * price_per_gb_second + 0.0000002)


def run_real_step_sequential(
    benchmark: str,
    openwhisk_config: Path,
    output_dir: Path,
    cache_dir: Path,
    function_name: str,
    memory_mb: int,
    timeout_sec: int,
    architecture: str,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one real OpenWhisk control step with sequential repetitions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python-venv/bin/python",
        "sebs.py",
        "benchmark",
        "invoke",
        benchmark,
        "test",
        "--trigger",
        "library",
        "--repetitions",
        str(repetitions),
        "--memory",
        str(memory_mb),
        "--timeout",
        str(timeout_sec),
        "--function-name",
        function_name,
        "--config",
        str(openwhisk_config),
        "--deployment",
        "openwhisk",
        "--architecture",
        architecture,
        "--container-deployment",
        "--cache",
        str(cache_dir),
        "--no-update-code",
        "--no-update-storage",
        "--output-dir",
        str(output_dir),
        "--output-file",
        "out.log",
        "--no-preserve-out",
        "--verbose",
    ]

    env = os.environ.copy()
    env["SEBS_WITH_OPENWHISK"] = "true"
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"OpenWhisk online step failed for {benchmark} / {function_name}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    experiments_path = output_dir / "experiments.json"
    if not experiments_path.exists():
        raise RuntimeError(
            f"Missing experiments.json after OpenWhisk step {benchmark} / {function_name}."
        )

    payload = json.loads(experiments_path.read_text())
    invocations = _flatten_invocations(payload)
    parsed_invocations = []
    latencies_ms = []
    cold_flags = []
    success_flags = []
    benchmark_latencies_ms = []

    for invocation in invocations:
        stats = invocation.get("stats", {})
        times = invocation.get("times", {})
        success = not bool(stats.get("failure", False))
        client_ms = float(times.get("client", math.nan)) / 1000.0
        benchmark_ms = float(times.get("benchmark", math.nan)) / 1000.0
        if not np.isfinite(client_ms):
            client_ms = float(timeout_sec) * 1000.0
        if client_ms <= 0.0:
            client_ms = float(timeout_sec) * 1000.0
        latencies_ms.append(client_ms)
        cold_flags.append(bool(stats.get("cold_start", False)))
        success_flags.append(success)
        if np.isfinite(benchmark_ms):
            benchmark_latencies_ms.append(benchmark_ms)
        parsed_invocations.append(
            {
                "request_id": invocation.get("request_id"),
                "success": success,
                "cold_start": bool(stats.get("cold_start", False)),
                "client_ms": client_ms,
                "benchmark_ms": benchmark_ms if np.isfinite(benchmark_ms) else None,
                "memory_used_mb": stats.get("memory_used"),
            }
        )

    latencies_ms_array = np.asarray(latencies_ms, dtype=np.float64)
    cold_flags_array = np.asarray(cold_flags, dtype=bool)
    success_flags_array = np.asarray(success_flags, dtype=bool)
    mean_cost = float(
        np.mean(
            [
                estimate_request_cost_usd(memory_mb, architecture, latency_ms)
                for latency_ms in latencies_ms_array
            ]
        )
    )

    metrics = {
        "latency": float(np.mean(latencies_ms_array)),
        "p50_latency": float(np.percentile(latencies_ms_array, 50)),
        "p95_latency": float(np.percentile(latencies_ms_array, 95)),
        "p99_latency": float(np.percentile(latencies_ms_array, 99)),
        "cost": mean_cost,
        "total_cost": float(mean_cost * len(latencies_ms_array)),
        "success": bool(success_flags_array.any()),
        "success_rate": float(np.mean(success_flags_array)) if success_flags_array.size else 0.0,
        "cold_start": bool(cold_flags_array.any()),
        "cold_start_rate": float(np.mean(cold_flags_array)) if cold_flags_array.size else 0.0,
        "timeout_rate": float(1.0 - np.mean(success_flags_array)) if success_flags_array.size else 1.0,
        "arrival_count": int(len(latencies_ms_array)),
        "peak_concurrency": 1 if latencies_ms_array.size else 0,
        "queue_ratio": 0.0,
        "cpu_utilization": 0.0 if latencies_ms_array.size == 0 else 1.0 / 32.0,
        "memory_utilization": min(
            1.0,
            float(np.percentile(latencies_ms_array, 95))
            / max(float(timeout_sec) * 1000.0, 1.0)
            * 1.2,
        ),
        "benchmark_mean_ms": (
            float(np.mean(benchmark_latencies_ms))
            if benchmark_latencies_ms
            else math.nan
        ),
        "latencies_ms": latencies_ms_array.tolist(),
        "cold_flags": cold_flags_array.astype(int).tolist(),
    }
    return metrics, parsed_invocations


def _execution_result_to_record(
    invocation: ExecutionResult,
    timeout_sec: int,
) -> tuple[dict[str, Any], float, bool, bool, float]:
    """Convert one ExecutionResult into a stable JSON record and metrics fields."""
    success = not bool(invocation.stats.failure)
    client_ms = float(invocation.times.client) / 1000.0
    benchmark_ms = float(invocation.times.benchmark) / 1000.0
    if not np.isfinite(client_ms) or client_ms <= 0.0:
        client_ms = float(timeout_sec) * 1000.0
    if not np.isfinite(benchmark_ms):
        benchmark_ms = math.nan
    record = {
        "request_id": invocation.request_id,
        "success": success,
        "cold_start": bool(invocation.stats.cold_start),
        "client_ms": client_ms,
        "benchmark_ms": benchmark_ms if np.isfinite(benchmark_ms) else None,
        "memory_used_mb": invocation.stats.memory_used,
    }
    return (
        record,
        client_ms,
        bool(invocation.stats.cold_start),
        success,
        benchmark_ms,
    )


def _build_failed_invocation_record(
    error: Exception,
    timeout_sec: int,
) -> tuple[dict[str, Any], float, bool, bool, float]:
    """Create a synthetic failed invocation record for robust aggregation."""
    client_ms = float(timeout_sec) * 1000.0
    record = {
        "request_id": None,
        "success": False,
        "cold_start": False,
        "client_ms": client_ms,
        "benchmark_ms": None,
        "memory_used_mb": None,
        "error": str(error),
    }
    return record, client_ms, False, False, math.nan


def run_real_step_parallel(
    deployment_handle: RealDeploymentHandle,
    memory_mb: int,
    timeout_sec: int,
    architecture: str,
    repetitions: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one real OpenWhisk control step with concurrent trigger replay."""
    trigger = ensure_real_function_configuration(
        deployment_handle,
        memory_mb=memory_mb,
        timeout_sec=timeout_sec,
        architecture=architecture,
    )
    parsed_invocations = []
    latencies_ms = []
    cold_flags = []
    success_flags = []
    benchmark_latencies_ms = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, repetitions)) as executor:
        futures = [
            executor.submit(trigger.sync_invoke, copy.deepcopy(deployment_handle.input_config))
            for _ in range(repetitions)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                record, client_ms, is_cold, is_success, benchmark_ms = _execution_result_to_record(
                    future.result(),
                    timeout_sec=timeout_sec,
                )
            except Exception as exc:
                record, client_ms, is_cold, is_success, benchmark_ms = (
                    _build_failed_invocation_record(exc, timeout_sec=timeout_sec)
                )
            parsed_invocations.append(record)
            latencies_ms.append(client_ms)
            cold_flags.append(is_cold)
            success_flags.append(is_success)
            if np.isfinite(benchmark_ms):
                benchmark_latencies_ms.append(benchmark_ms)

    latencies_ms_array = np.asarray(latencies_ms, dtype=np.float64)
    cold_flags_array = np.asarray(cold_flags, dtype=bool)
    success_flags_array = np.asarray(success_flags, dtype=bool)
    mean_cost = float(
        np.mean(
            [
                estimate_request_cost_usd(memory_mb, architecture, latency_ms)
                for latency_ms in latencies_ms_array
            ]
        )
    )
    peak_concurrency = int(repetitions if repetitions > 0 else 0)
    cpu_utilization = min(1.0, float(peak_concurrency) / 32.0)
    memory_utilization = min(
        1.0,
        float(np.percentile(latencies_ms_array, 95))
        / max(float(timeout_sec) * 1000.0, 1.0)
        * max(1.0, float(peak_concurrency)),
    )
    metrics = {
        "latency": float(np.mean(latencies_ms_array)),
        "p50_latency": float(np.percentile(latencies_ms_array, 50)),
        "p95_latency": float(np.percentile(latencies_ms_array, 95)),
        "p99_latency": float(np.percentile(latencies_ms_array, 99)),
        "cost": mean_cost,
        "total_cost": float(mean_cost * len(latencies_ms_array)),
        "success": bool(success_flags_array.any()),
        "success_rate": float(np.mean(success_flags_array)) if success_flags_array.size else 0.0,
        "cold_start": bool(cold_flags_array.any()),
        "cold_start_rate": float(np.mean(cold_flags_array)) if cold_flags_array.size else 0.0,
        "timeout_rate": float(1.0 - np.mean(success_flags_array)) if success_flags_array.size else 1.0,
        "arrival_count": int(len(latencies_ms_array)),
        "peak_concurrency": peak_concurrency,
        "queue_ratio": 0.0,
        "cpu_utilization": cpu_utilization,
        "memory_utilization": memory_utilization,
        "benchmark_mean_ms": (
            float(np.mean(benchmark_latencies_ms))
            if benchmark_latencies_ms
            else math.nan
        ),
        "latencies_ms": latencies_ms_array.tolist(),
        "cold_flags": cold_flags_array.astype(int).tolist(),
    }
    return metrics, parsed_invocations


def _run_unmeasured_prime_replay(
    *,
    benchmark: str,
    openwhisk_config: Path,
    output_dir: Path,
    cache_dir: Path,
    function_name: str,
    real_invoke_mode: str,
    deployment_handle: RealDeploymentHandle | None,
    memory_mb: int,
    timeout_sec: int,
    architecture: str,
    repetitions: int,
    prime_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one unmeasured warmup replay for a configuration."""
    prime_output_dir = output_dir / prime_label
    prime_output_dir.mkdir(parents=True, exist_ok=True)
    if real_invoke_mode == "parallel":
        if deployment_handle is None:
            raise ValueError("Parallel replay requires an active deployment handle.")
        return run_real_step_parallel(
            deployment_handle=deployment_handle,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            architecture=architecture,
            repetitions=int(repetitions),
        )
    return run_real_step_sequential(
        benchmark=benchmark,
        openwhisk_config=openwhisk_config,
        output_dir=prime_output_dir,
        cache_dir=cache_dir,
        function_name=function_name,
        memory_mb=memory_mb,
        timeout_sec=timeout_sec,
        architecture=architecture,
        repetitions=int(repetitions),
    )


def _policy_action(
    artifact: PolicyRunArtifact,
    observation: np.ndarray,
) -> int:
    """Select one action for the current observation."""
    if artifact.policy_name == "bayes_opt_online":
        if artifact.policy_controller is None:
            raise ValueError("Online BO artifact is missing its controller.")
        return int(
            artifact.policy_controller.select_action(
                np.asarray(observation, dtype=np.float32)
            )
        )
    if artifact.policy_name == "default":
        return int(artifact.default_action)
    if artifact.model is None:
        raise ValueError(f"Policy {artifact.policy_name} is missing a trained model.")
    action, _ = artifact.model.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
    return int(action)


def _nearest_feasible_action(
    env_helper: ServerlessFunctionEnv,
    requested_action: int,
    feasible_actions: set[int],
) -> int:
    """Map one infeasible action to the nearest deployable action."""
    action_space = env_helper.action_space_wrapper
    requested = action_space.get_configuration(int(requested_action))
    feasible_configs = [
        (
            int(action_id),
            action_space.get_configuration(int(action_id)),
        )
        for action_id in feasible_actions
    ]

    same_timeout = [
        (action_id, config)
        for action_id, config in feasible_configs
        if str(config.architecture) == str(requested.architecture)
        and int(config.timeout_sec) == int(requested.timeout_sec)
    ]
    if same_timeout:
        upscale = [
            (action_id, config)
            for action_id, config in same_timeout
            if int(config.memory_mb) >= int(requested.memory_mb)
        ]
        if upscale:
            return int(
                min(
                    upscale,
                    key=lambda item: (
                        int(item[1].memory_mb) - int(requested.memory_mb),
                        int(item[1].timeout_sec),
                    ),
                )[0]
            )
        return int(
            min(
                same_timeout,
                key=lambda item: abs(int(item[1].memory_mb) - int(requested.memory_mb)),
            )[0]
        )

    def _distance(item: tuple[int, Any]) -> tuple[int, int, int, int]:
        _, config = item
        architecture_penalty = 0 if str(config.architecture) == str(requested.architecture) else 1
        timeout_delta = abs(int(config.timeout_sec) - int(requested.timeout_sec))
        upscale_penalty = 0 if int(config.memory_mb) >= int(requested.memory_mb) else 1
        memory_delta = abs(int(config.memory_mb) - int(requested.memory_mb))
        return (
            architecture_penalty,
            timeout_delta,
            upscale_penalty,
            memory_delta,
        )

    return int(min(feasible_configs, key=_distance)[0])


def _coerce_action_to_feasible(
    artifact: PolicyRunArtifact,
    requested_action: int,
) -> tuple[int, dict[str, Any]]:
    """Apply the real-platform deployability mask to one proposed action."""
    if not artifact.feasible_actions:
        return int(requested_action), {
            "mask_enabled": False,
            "requested_action": int(requested_action),
            "applied_action": int(requested_action),
            "remapped": False,
        }

    if int(requested_action) in artifact.feasible_actions:
        return int(requested_action), {
            "mask_enabled": True,
            "requested_action": int(requested_action),
            "applied_action": int(requested_action),
            "remapped": False,
        }

    remapped_action = _nearest_feasible_action(
        env_helper=artifact.env_helper,
        requested_action=int(requested_action),
        feasible_actions=artifact.feasible_actions,
    )
    return remapped_action, {
        "mask_enabled": True,
        "requested_action": int(requested_action),
        "applied_action": int(remapped_action),
        "remapped": True,
        "requested_config": artifact.env_helper.action_space_wrapper.get_configuration(
            int(requested_action)
        ).to_dict(),
        "applied_config": artifact.env_helper.action_space_wrapper.get_configuration(
            int(remapped_action)
        ).to_dict(),
    }


def _action_capacity_rank(
    env_helper: ServerlessFunctionEnv,
    action: int,
) -> tuple[int, int, int]:
    """Return a coarse capacity rank for one discrete action."""
    action_space = env_helper.action_space_wrapper
    config = action_space.get_configuration(int(action))
    memory_idx = action_space.MEMORY_OPTIONS.index(config.memory_mb)
    timeout_idx = action_space.TIMEOUT_OPTIONS.index(config.timeout_sec)
    arch_idx = action_space.ARCHITECTURE_OPTIONS.index(config.architecture)
    return (memory_idx, timeout_idx, arch_idx)


def _workload_ratio_against_previous(
    previous_workload: WorkloadStep | None,
    current_workload: WorkloadStep,
) -> float:
    """Compute a conservative current/previous workload ratio."""
    if previous_workload is None:
        return 1.0
    previous_arrivals = max(int(previous_workload.arrival_count), 1)
    previous_load_value = max(float(previous_workload.load_value), 1e-6)
    previous_mean_rate = max(float(previous_workload.mean_invocations_per_minute), 1e-6)
    arrival_ratio = float(current_workload.arrival_count) / float(previous_arrivals)
    load_ratio = float(current_workload.load_value) / previous_load_value
    mean_rate_ratio = (
        float(current_workload.mean_invocations_per_minute) / previous_mean_rate
    )
    return float(max(arrival_ratio, load_ratio, mean_rate_ratio))


def _describe_action_transition(
    env_helper: ServerlessFunctionEnv,
    current_action: int,
    proposed_action: int,
) -> dict[str, Any]:
    """Summarize how the proposed action differs from the current action."""
    action_space = env_helper.action_space_wrapper
    current_config = action_space.get_configuration(int(current_action))
    proposed_config = action_space.get_configuration(int(proposed_action))
    memory_delta_mb = int(proposed_config.memory_mb) - int(current_config.memory_mb)
    timeout_delta_sec = int(proposed_config.timeout_sec) - int(current_config.timeout_sec)
    architecture_changed = (
        str(proposed_config.architecture) != str(current_config.architecture)
    )
    timeout_only = (
        memory_delta_mb == 0 and timeout_delta_sec != 0 and not architecture_changed
    )
    return {
        "current_config": current_config.to_dict(),
        "proposed_config": proposed_config.to_dict(),
        "memory_delta_mb": memory_delta_mb,
        "timeout_delta_sec": timeout_delta_sec,
        "architecture_changed": architecture_changed,
        "timeout_only": timeout_only,
    }


def _runtime_pressure_summary(
    last_step_metrics: dict[str, float],
    current_timeout_sec: int,
) -> dict[str, float | bool]:
    """Describe whether the previous real step showed resource pressure."""
    if not last_step_metrics:
        return {
            "has_signal": False,
            "runtime_pressure": False,
            "latency_to_timeout_ratio": 0.0,
            "success_rate": 1.0,
            "timeout_rate": 0.0,
            "cold_start_rate": 0.0,
        }

    p95_latency_ms = float(last_step_metrics.get("p95_latency", 0.0))
    success_rate = float(last_step_metrics.get("success_rate", 1.0))
    timeout_rate = float(last_step_metrics.get("timeout_rate", 0.0))
    cold_start_rate = float(last_step_metrics.get("cold_start_rate", 0.0))
    latency_to_timeout_ratio = p95_latency_ms / max(float(current_timeout_sec) * 1000.0, 1.0)
    runtime_pressure = bool(
        timeout_rate > 0.0
        or success_rate < 0.999
        or latency_to_timeout_ratio >= 0.75
    )
    return {
        "has_signal": True,
        "runtime_pressure": runtime_pressure,
        "latency_to_timeout_ratio": float(latency_to_timeout_ratio),
        "success_rate": success_rate,
        "timeout_rate": timeout_rate,
        "cold_start_rate": cold_start_rate,
    }


def _policy_uses_default_anchor(artifact: PolicyRunArtifact) -> bool:
    """Return whether a policy should use the default-anchored guard."""
    return artifact.policy_name in {"ppo", "ppo_load_only"}


def _apply_default_anchor_guard(
    artifact: PolicyRunArtifact,
    proposed_action: int,
    stabilization_state: ActionStabilizationState,
    current_workload: WorkloadStep,
    previous_workload: WorkloadStep | None,
    current_step_index: int,
    default_anchor_enabled: bool,
    default_anchor_release_patience: int,
    default_anchor_min_arrivals: int,
    default_anchor_load_ratio_threshold: float,
    default_anchor_min_burstiness: float,
    default_anchor_min_step: int,
) -> tuple[int, dict[str, Any]]:
    """Guard CALO against leaving the default action too early on real replay."""
    default_action = artifact.default_action
    if default_action is None:
        default_action = artifact.env_helper.action_space_wrapper.get_default_action()
    default_action = int(default_action)

    info: dict[str, Any] = {
        "enabled": bool(default_anchor_enabled and _policy_uses_default_anchor(artifact)),
        "default_action": default_action,
        "proposed_action": int(proposed_action),
        "applied_action": int(proposed_action),
        "decision": "accept",
        "reason": "disabled",
    }
    if not info["enabled"]:
        return int(proposed_action), info

    current_action = stabilization_state.last_effective_action
    if current_action is None:
        current_action = default_action
    current_action = int(current_action)
    info["current_action"] = current_action
    info["anchored"] = bool(current_action == default_action)

    if int(proposed_action) == default_action:
        stabilization_state.pending_anchor_upgrade_action = None
        stabilization_state.pending_anchor_upgrade_count = 0
        info["reason"] = "default_selected"
        return default_action, info

    if current_action != default_action:
        stabilization_state.pending_anchor_upgrade_action = None
        stabilization_state.pending_anchor_upgrade_count = 0
        info["reason"] = "anchor_released"
        return int(proposed_action), info

    workload_ratio = _workload_ratio_against_previous(previous_workload, current_workload)
    runtime_pressure = _runtime_pressure_summary(
        stabilization_state.last_step_metrics,
        current_timeout_sec=int(
            artifact.env_helper.action_space_wrapper.get_configuration(default_action).timeout_sec
        ),
    )
    proposed_rank = _action_capacity_rank(artifact.env_helper, int(proposed_action))
    default_rank = _action_capacity_rank(artifact.env_helper, default_action)
    transition = _describe_action_transition(
        artifact.env_helper,
        current_action=default_action,
        proposed_action=int(proposed_action),
    )
    release_signal = bool(
        current_step_index >= int(default_anchor_min_step)
        and (
            int(current_workload.arrival_count) >= int(default_anchor_min_arrivals)
            or workload_ratio >= float(default_anchor_load_ratio_threshold)
            or float(getattr(current_workload, "burstiness_hint", 1.0))
            >= float(default_anchor_min_burstiness)
            or bool(runtime_pressure["runtime_pressure"])
        )
    )
    info["workload_ratio"] = workload_ratio
    info["runtime_pressure"] = runtime_pressure
    info["release_signal"] = release_signal
    info["transition"] = transition

    if proposed_rank < default_rank:
        stabilization_state.pending_anchor_upgrade_action = None
        stabilization_state.pending_anchor_upgrade_count = 0
        info["decision"] = "hold_default"
        info["reason"] = "subdefault_action_blocked"
        info["applied_action"] = default_action
        return default_action, info

    if transition["timeout_only"] and not bool(runtime_pressure["runtime_pressure"]):
        stabilization_state.pending_anchor_upgrade_action = None
        stabilization_state.pending_anchor_upgrade_count = 0
        info["decision"] = "hold_default"
        info["reason"] = "timeout_only_blocked"
        info["applied_action"] = default_action
        return default_action, info

    if not release_signal:
        stabilization_state.pending_anchor_upgrade_action = None
        stabilization_state.pending_anchor_upgrade_count = 0
        info["decision"] = "hold_default"
        info["reason"] = "release_signal_not_met"
        info["applied_action"] = default_action
        return default_action, info

    if stabilization_state.pending_anchor_upgrade_action == int(proposed_action):
        stabilization_state.pending_anchor_upgrade_count += 1
    else:
        stabilization_state.pending_anchor_upgrade_action = int(proposed_action)
        stabilization_state.pending_anchor_upgrade_count = 1
    info["pending_upgrade_count"] = int(
        stabilization_state.pending_anchor_upgrade_count
    )

    if stabilization_state.pending_anchor_upgrade_count < max(
        int(default_anchor_release_patience), 1
    ):
        info["decision"] = "hold_default"
        info["reason"] = "awaiting_upgrade_confirmation"
        info["applied_action"] = default_action
        return default_action, info

    stabilization_state.pending_anchor_upgrade_action = None
    stabilization_state.pending_anchor_upgrade_count = 0
    info["reason"] = "default_anchor_released"
    return int(proposed_action), info


def _stabilize_action_for_online_replay(
    artifact: PolicyRunArtifact,
    proposed_action: int,
    stabilization_state: ActionStabilizationState,
    current_workload: WorkloadStep,
    previous_workload: WorkloadStep | None,
    stabilize_actions: bool,
    min_dwell_steps: int,
    downgrade_patience: int,
    load_drop_threshold: float,
) -> tuple[int, dict[str, Any]]:
    """Optionally smooth one proposed online action."""
    info: dict[str, Any] = {
        "enabled": bool(stabilize_actions and artifact.policy_name != "default"),
        "proposed_action": int(proposed_action),
        "applied_action": int(proposed_action),
        "decision": "accept",
        "reason": "disabled_or_default",
        "steps_since_switch": int(stabilization_state.steps_since_switch),
        "pending_downgrade_count": int(stabilization_state.pending_downgrade_count),
        "workload_ratio": 1.0,
    }
    if not info["enabled"]:
        return int(proposed_action), info

    last_effective_action = stabilization_state.last_effective_action
    if last_effective_action is None:
        info["reason"] = "first_action"
        return int(proposed_action), info

    if int(proposed_action) == int(last_effective_action):
        stabilization_state.pending_upgrade_action = None
        stabilization_state.pending_upgrade_count = 0
        stabilization_state.pending_downgrade_action = None
        stabilization_state.pending_downgrade_count = 0
        info["reason"] = "same_as_current"
        return int(proposed_action), info

    current_rank = _action_capacity_rank(artifact.env_helper, int(last_effective_action))
    proposed_rank = _action_capacity_rank(artifact.env_helper, int(proposed_action))
    workload_ratio = _workload_ratio_against_previous(previous_workload, current_workload)
    transition = _describe_action_transition(
        artifact.env_helper,
        current_action=int(last_effective_action),
        proposed_action=int(proposed_action),
    )
    runtime_pressure = _runtime_pressure_summary(
        stabilization_state.last_step_metrics,
        current_timeout_sec=int(transition["current_config"]["timeout_sec"]),
    )
    info["workload_ratio"] = workload_ratio
    info["transition"] = transition
    info["runtime_pressure"] = runtime_pressure

    if (
        stabilization_state.steps_since_switch < max(int(min_dwell_steps), 0)
        and proposed_rank > current_rank
        and not bool(runtime_pressure["runtime_pressure"])
    ):
        info["decision"] = "hold"
        info["reason"] = "post_switch_cooldown"
        info["applied_action"] = int(last_effective_action)
        return int(last_effective_action), info

    timeout_only_guard = bool(
        transition["timeout_only"] and not bool(runtime_pressure["runtime_pressure"])
    )
    required_dwell_steps = max(int(min_dwell_steps), 0)
    required_downgrade_patience = max(int(downgrade_patience), 1)
    effective_load_drop_threshold = float(load_drop_threshold)

    if timeout_only_guard:
        required_dwell_steps = max(required_dwell_steps + 2, 4)
        required_downgrade_patience = max(required_downgrade_patience + 1, 3)
        effective_load_drop_threshold = min(effective_load_drop_threshold, 0.80)
        info["timeout_only_guard"] = {
            "enabled": True,
            "required_dwell_steps": required_dwell_steps,
            "required_downgrade_patience": required_downgrade_patience,
            "effective_load_drop_threshold": effective_load_drop_threshold,
        }
    else:
        info["timeout_only_guard"] = {"enabled": False}

    if proposed_rank > current_rank:
        if timeout_only_guard and workload_ratio < 1.35:
            info["decision"] = "hold"
            info["reason"] = "timeout_only_upgrade_guard"
            info["applied_action"] = int(last_effective_action)
            return int(last_effective_action), info
        stabilization_state.pending_upgrade_action = None
        stabilization_state.pending_upgrade_count = 0
        stabilization_state.pending_downgrade_action = None
        stabilization_state.pending_downgrade_count = 0
        info["reason"] = "upgrade_allowed"
        return int(proposed_action), info

    if stabilization_state.steps_since_switch < required_dwell_steps:
        info["decision"] = "hold"
        info["reason"] = (
            "timeout_only_downgrade_dwell" if timeout_only_guard else "min_dwell"
        )
        info["applied_action"] = int(last_effective_action)
        return int(last_effective_action), info

    if stabilization_state.pending_downgrade_action == int(proposed_action):
        stabilization_state.pending_downgrade_count += 1
    else:
        stabilization_state.pending_downgrade_action = int(proposed_action)
        stabilization_state.pending_downgrade_count = 1
    info["pending_downgrade_count"] = int(stabilization_state.pending_downgrade_count)

    if stabilization_state.pending_downgrade_count < required_downgrade_patience:
        info["decision"] = "hold"
        info["reason"] = (
            "timeout_only_downgrade_patience"
            if timeout_only_guard
            else "downgrade_patience"
        )
        info["applied_action"] = int(last_effective_action)
        return int(last_effective_action), info

    if workload_ratio > effective_load_drop_threshold:
        info["decision"] = "hold"
        info["reason"] = (
            "timeout_only_load_drop_guard"
            if timeout_only_guard
            else "insufficient_load_drop"
        )
        info["applied_action"] = int(last_effective_action)
        return int(last_effective_action), info

    stabilization_state.pending_downgrade_action = None
    stabilization_state.pending_downgrade_count = 0
    stabilization_state.pending_upgrade_action = None
    stabilization_state.pending_upgrade_count = 0
    info["reason"] = "downgrade_confirmed"
    return int(proposed_action), info


def _plan_online_action(
    *,
    artifact: PolicyRunArtifact,
    observation: np.ndarray,
    stabilization_state: ActionStabilizationState,
    current_workload: WorkloadStep,
    previous_workload: WorkloadStep | None,
    current_step_index: int,
    default_anchor_enabled: bool,
    default_anchor_release_patience: int,
    default_anchor_min_arrivals: int,
    default_anchor_load_ratio_threshold: float,
    default_anchor_min_burstiness: float,
    default_anchor_min_step: int,
    stabilize_actions: bool,
    stabilization_min_dwell_steps: int,
    stabilization_downgrade_patience: int,
    stabilization_load_drop_threshold: float,
) -> dict[str, Any]:
    """Resolve the effective online action after masking and runtime guards."""
    raw_proposed_action = _policy_action(artifact, observation)
    proposed_action, mask_info = _coerce_action_to_feasible(
        artifact=artifact,
        requested_action=int(raw_proposed_action),
    )
    default_anchor_action, default_anchor_info = _apply_default_anchor_guard(
        artifact=artifact,
        proposed_action=int(proposed_action),
        stabilization_state=stabilization_state,
        current_workload=current_workload,
        previous_workload=previous_workload,
        current_step_index=current_step_index,
        default_anchor_enabled=default_anchor_enabled,
        default_anchor_release_patience=default_anchor_release_patience,
        default_anchor_min_arrivals=default_anchor_min_arrivals,
        default_anchor_load_ratio_threshold=default_anchor_load_ratio_threshold,
        default_anchor_min_burstiness=default_anchor_min_burstiness,
        default_anchor_min_step=default_anchor_min_step,
    )
    action, stabilization_info = _stabilize_action_for_online_replay(
        artifact=artifact,
        proposed_action=int(default_anchor_action),
        stabilization_state=stabilization_state,
        current_workload=current_workload,
        previous_workload=previous_workload,
        stabilize_actions=stabilize_actions,
        min_dwell_steps=stabilization_min_dwell_steps,
        downgrade_patience=stabilization_downgrade_patience,
        load_drop_threshold=stabilization_load_drop_threshold,
    )
    return {
        "raw_proposed_action": int(raw_proposed_action),
        "proposed_action": int(proposed_action),
        "mask_info": mask_info,
        "default_anchor_action": int(default_anchor_action),
        "default_anchor_info": default_anchor_info,
        "action": int(action),
        "stabilization_info": stabilization_info,
    }


def _serialize_workload(workload_step: WorkloadStep) -> dict[str, Any]:
    """Convert one workload step to JSON."""
    return asdict(workload_step)


def _write_step_payload(path: Path, payload: dict[str, Any]) -> None:
    """Persist one JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _mean_or_none(values: list[float]) -> float | None:
    """Return the mean of a non-empty list, else None."""
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _planned_real_repetitions(
    requested_invocations: int,
    min_real_invocations_per_step: int,
    max_real_invocations_per_step: int,
) -> int:
    """Map one workload count to the effective real replay concurrency."""
    if int(requested_invocations) <= 0:
        return 0
    return max(
        int(min_real_invocations_per_step),
        min(int(max_real_invocations_per_step), int(requested_invocations)),
    )


def _prime_requires_settle(metrics: dict[str, Any]) -> bool:
    """Return whether one unmeasured prime left the platform in an unstable state."""
    success_rate = float(metrics.get("success_rate", 1.0))
    p95_latency = float(metrics.get("p95_latency", 0.0))
    return success_rate < 1.0 or p95_latency >= INITIAL_PRIME_SETTLE_P95_THRESHOLD_MS


def _run_settle_probes(
    *,
    benchmark: str,
    openwhisk_config: Path,
    output_dir: Path,
    cache_dir: Path,
    function_name: str,
    real_invoke_mode: str,
    deployment_handle: RealDeploymentHandle | None,
    memory_mb: int,
    timeout_sec: int,
    architecture: str,
    triggering_metrics: dict[str, Any],
    settle_repetitions: int,
    settle_label_prefix: str,
) -> list[dict[str, Any]]:
    """Run unmeasured settle probes until the platform reaches a stable state."""
    settle_probes: list[dict[str, Any]] = []
    if not _prime_requires_settle(triggering_metrics):
        return settle_probes

    effective_repetitions = max(int(settle_repetitions), 1)
    for attempt_index in range(INITIAL_PRIME_SETTLE_MAX_ATTEMPTS):
        settle_metrics, settle_invocations = _run_unmeasured_prime_replay(
            benchmark=benchmark,
            openwhisk_config=openwhisk_config,
            output_dir=output_dir,
            cache_dir=cache_dir,
            function_name=function_name,
            real_invoke_mode=real_invoke_mode,
            deployment_handle=deployment_handle,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            architecture=architecture,
            repetitions=effective_repetitions,
            prime_label=f"{settle_label_prefix}_settle_{attempt_index + 1:02d}",
        )
        settle_record = {
            "attempt": int(attempt_index + 1),
            "repetitions": int(effective_repetitions),
            "metrics": settle_metrics,
            "invocations": settle_invocations,
        }
        settle_probes.append(settle_record)
        if not _prime_requires_settle(settle_metrics):
            break
        time.sleep(INITIAL_PRIME_SETTLE_SLEEP_SEC)
    return settle_probes


def _steady_state_summary(step_records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Summarize steps after the first deployment shock."""
    steady_steps = [row for row in step_records if int(row["step"]) > 0]
    if not steady_steps:
        return {
            "step_count": 0,
            "mean_raw_reward": None,
            "mean_latency_ms": None,
            "mean_p95_latency_ms": None,
            "mean_cost_usd": None,
            "mean_success_rate": None,
            "mean_cold_start_rate": None,
        }
    return {
        "step_count": len(steady_steps),
        "mean_raw_reward": _mean_or_none([float(row["raw_reward"]) for row in steady_steps]),
        "mean_latency_ms": _mean_or_none(
            [float(row["metrics"]["latency"]) for row in steady_steps]
        ),
        "mean_p95_latency_ms": _mean_or_none(
            [float(row["metrics"]["p95_latency"]) for row in steady_steps]
        ),
        "mean_cost_usd": _mean_or_none([float(row["metrics"]["cost"]) for row in steady_steps]),
        "mean_success_rate": _mean_or_none(
            [float(row["metrics"]["success_rate"]) for row in steady_steps]
        ),
        "mean_cold_start_rate": _mean_or_none(
            [float(row["metrics"]["cold_start_rate"]) for row in steady_steps]
        ),
    }


def run_policy_online(
    artifact: PolicyRunArtifact,
    benchmark: str,
    openwhisk_config: Path,
    output_dir: Path,
    cache_dir: Path,
    function_prefix: str,
    workload_steps: list[WorkloadStep],
    environment_cfg: dict[str, Any],
    load_pattern: str,
    seed: int,
    max_real_invocations_per_step: int,
    min_real_invocations_per_step: int,
    keep_raw: bool,
    real_invoke_mode: str,
    stabilize_actions: bool,
    stabilization_min_dwell_steps: int,
    stabilization_downgrade_patience: int,
    stabilization_load_drop_threshold: float,
    initial_prime_repetitions: int,
    initial_prime_lookahead_steps: int,
    reconfiguration_prime_repetitions: int,
    default_anchor: bool,
    default_anchor_release_patience: int,
    default_anchor_min_arrivals: int,
    default_anchor_load_ratio_threshold: float,
    default_anchor_min_burstiness: float,
    default_anchor_min_step: int,
) -> dict[str, Any]:
    """Run one policy online on the real OpenWhisk platform."""
    if output_dir.exists() and not keep_raw:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    observation, _ = artifact.env_helper.reset(seed=seed)
    reset_fn = getattr(artifact.policy_controller, "reset", None)
    if callable(reset_fn):
        reset_fn()
    current_time_sec = 0.0
    last_action: int | None = None
    load_recently_changed = False
    function_name = _build_function_name(
        benchmark=benchmark,
        policy_name=artifact.policy_name,
        function_prefix=function_prefix,
        seed=seed,
    )
    step_records = []
    real_deployment_handle = None
    stabilization_state = ActionStabilizationState()
    prime_record: dict[str, Any] | None = None

    dynamic_reference = _build_dynamic_env(
        benchmark=benchmark,
        environment_cfg=environment_cfg,
        load_pattern=load_pattern,
        algorithm="ppo",
        algorithm_settings=None,
    )
    dynamic_reference.reset(seed=seed, options={"load_pattern": load_pattern})
    if real_invoke_mode == "parallel":
        real_deployment_handle = create_real_deployment_handle(
            benchmark=benchmark,
            openwhisk_config=openwhisk_config,
            output_dir=output_dir,
            cache_dir=cache_dir,
            function_name=function_name,
            architecture="x64",
        )

    try:
        if int(initial_prime_repetitions) > 0:
            first_step_repetitions = _planned_real_repetitions(
                requested_invocations=int(workload_steps[0].arrival_count),
                min_real_invocations_per_step=min_real_invocations_per_step,
                max_real_invocations_per_step=max_real_invocations_per_step,
            )
            lookahead_repetitions = int(first_step_repetitions)
            if int(initial_prime_lookahead_steps) > 0:
                lookahead_horizon = min(
                    len(workload_steps),
                    max(int(initial_prime_lookahead_steps), 1),
                )
                for lookahead_step in workload_steps[:lookahead_horizon]:
                    candidate_repetitions = _planned_real_repetitions(
                        requested_invocations=int(lookahead_step.arrival_count),
                        min_real_invocations_per_step=min_real_invocations_per_step,
                        max_real_invocations_per_step=max_real_invocations_per_step,
                    )
                    lookahead_repetitions = max(
                        int(lookahead_repetitions),
                        int(candidate_repetitions),
                    )
            effective_initial_prime_repetitions = int(initial_prime_repetitions)
            if max(first_step_repetitions, lookahead_repetitions) > 0:
                effective_initial_prime_repetitions = max(
                    1,
                    min(
                        int(initial_prime_repetitions),
                        max(
                            int(first_step_repetitions),
                            int(lookahead_repetitions),
                        ),
                    ),
                )
            if artifact.shadow_env is not None:
                _sync_shadow_env_runtime(
                    shadow_env=artifact.shadow_env,
                    source_state_space=artifact.env_helper.state_space,
                    current_workload=workload_steps[0],
                    current_step=0,
                    current_time_sec=current_time_sec,
                    last_action=last_action,
                    load_recently_changed=load_recently_changed,
                )
            prime_action_plan = _plan_online_action(
                artifact=artifact,
                observation=np.asarray(observation, dtype=np.float32),
                stabilization_state=copy.deepcopy(stabilization_state),
                current_workload=workload_steps[0],
                previous_workload=None,
                current_step_index=0,
                default_anchor_enabled=default_anchor,
                default_anchor_release_patience=default_anchor_release_patience,
                default_anchor_min_arrivals=default_anchor_min_arrivals,
                default_anchor_load_ratio_threshold=default_anchor_load_ratio_threshold,
                default_anchor_min_burstiness=default_anchor_min_burstiness,
                default_anchor_min_step=default_anchor_min_step,
                stabilize_actions=stabilize_actions,
                stabilization_min_dwell_steps=stabilization_min_dwell_steps,
                stabilization_downgrade_patience=stabilization_downgrade_patience,
                stabilization_load_drop_threshold=stabilization_load_drop_threshold,
            )
            prime_action = int(prime_action_plan["action"])
            prime_config = artifact.env_helper.action_space_wrapper.get_configuration(
                prime_action
            )
            prime_output_dir = output_dir / "_prime"
            prime_output_dir.mkdir(parents=True, exist_ok=True)
            if real_invoke_mode == "parallel":
                assert real_deployment_handle is not None
                prime_metrics, prime_invocations = _run_unmeasured_prime_replay(
                    benchmark=benchmark,
                    openwhisk_config=openwhisk_config,
                    output_dir=output_dir,
                    cache_dir=cache_dir,
                    function_name=function_name,
                    real_invoke_mode=real_invoke_mode,
                    deployment_handle=real_deployment_handle,
                    memory_mb=prime_config.memory_mb,
                    timeout_sec=prime_config.timeout_sec,
                    architecture=prime_config.architecture,
                    repetitions=int(effective_initial_prime_repetitions),
                    prime_label="_prime",
                )
            else:
                prime_metrics, prime_invocations = _run_unmeasured_prime_replay(
                    benchmark=benchmark,
                    openwhisk_config=openwhisk_config,
                    output_dir=output_dir,
                    cache_dir=cache_dir,
                    function_name=function_name,
                    real_invoke_mode=real_invoke_mode,
                    deployment_handle=real_deployment_handle,
                    memory_mb=prime_config.memory_mb,
                    timeout_sec=prime_config.timeout_sec,
                    architecture=prime_config.architecture,
                    repetitions=int(effective_initial_prime_repetitions),
                    prime_label="_prime",
                )
            prime_record = {
                "policy": artifact.policy_name,
                "raw_proposed_action": int(prime_action_plan["raw_proposed_action"]),
                "proposed_action": int(prime_action_plan["proposed_action"]),
                "action": prime_action,
                "config": prime_config.to_dict(),
                "config_label": str(prime_config),
                "mask": prime_action_plan["mask_info"],
                "default_anchor": prime_action_plan["default_anchor_info"],
                "stabilization": prime_action_plan["stabilization_info"],
                "prime_repetitions": int(effective_initial_prime_repetitions),
                "requested_prime_repetitions": int(initial_prime_repetitions),
                "first_step_repetitions": int(first_step_repetitions),
                "lookahead_repetitions": int(lookahead_repetitions),
                "initial_prime_lookahead_steps": int(initial_prime_lookahead_steps),
                "metrics": prime_metrics,
                "invocations": prime_invocations,
            }
            settle_probes = _run_settle_probes(
                benchmark=benchmark,
                openwhisk_config=openwhisk_config,
                output_dir=output_dir,
                cache_dir=cache_dir,
                function_name=function_name,
                real_invoke_mode=real_invoke_mode,
                deployment_handle=real_deployment_handle,
                memory_mb=prime_config.memory_mb,
                timeout_sec=prime_config.timeout_sec,
                architecture=prime_config.architecture,
                triggering_metrics=prime_metrics,
                settle_repetitions=max(
                    int(first_step_repetitions),
                    int(lookahead_repetitions),
                    int(effective_initial_prime_repetitions),
                    1,
                ),
                settle_label_prefix="_prime",
            )
            if settle_probes:
                prime_record["settle_probes"] = settle_probes
            _write_step_payload(prime_output_dir / "prime_metrics.json", prime_record)
            print(
                f"[{artifact.display_name}] "
                f"prime_repetitions={int(effective_initial_prime_repetitions)}, "
                f"prime_p95_latency={float(prime_metrics['p95_latency']):.2f} ms, "
                f"prime_cold_start_rate={float(prime_metrics['cold_start_rate']):.3f}",
                flush=True,
            )
            if settle_probes:
                last_settle = settle_probes[-1]["metrics"]
                print(
                    f"[{artifact.display_name}] "
                    f"prime_settle_attempts={len(settle_probes)}, "
                    f"settle_p95_latency={float(last_settle['p95_latency']):.2f} ms, "
                    f"settle_success_rate={float(last_settle['success_rate']):.3f}",
                    flush=True,
                )
            last_action = prime_action
            stabilization_state.last_effective_action = prime_action
            stabilization_state.steps_since_switch = 0
            if callable(reset_fn):
                reset_fn()

        for step_index, workload_step in enumerate(workload_steps):
            if artifact.shadow_env is not None:
                _sync_shadow_env_runtime(
                    shadow_env=artifact.shadow_env,
                    source_state_space=artifact.env_helper.state_space,
                    current_workload=workload_step,
                    current_step=step_index,
                    current_time_sec=current_time_sec,
                    last_action=last_action,
                    load_recently_changed=load_recently_changed,
                )
            previous_workload = workload_steps[step_index - 1] if step_index > 0 else None
            action_plan = _plan_online_action(
                artifact=artifact,
                observation=np.asarray(observation, dtype=np.float32),
                stabilization_state=stabilization_state,
                current_workload=workload_step,
                previous_workload=previous_workload,
                current_step_index=step_index,
                default_anchor_enabled=default_anchor,
                default_anchor_release_patience=default_anchor_release_patience,
                default_anchor_min_arrivals=default_anchor_min_arrivals,
                default_anchor_load_ratio_threshold=default_anchor_load_ratio_threshold,
                default_anchor_min_burstiness=default_anchor_min_burstiness,
                default_anchor_min_step=default_anchor_min_step,
                stabilize_actions=stabilize_actions,
                stabilization_min_dwell_steps=stabilization_min_dwell_steps,
                stabilization_downgrade_patience=stabilization_downgrade_patience,
                stabilization_load_drop_threshold=stabilization_load_drop_threshold,
            )
            raw_proposed_action = int(action_plan["raw_proposed_action"])
            proposed_action = int(action_plan["proposed_action"])
            mask_info = action_plan["mask_info"]
            default_anchor_action = int(action_plan["default_anchor_action"])
            default_anchor_info = action_plan["default_anchor_info"]
            action = int(action_plan["action"])
            stabilization_info = action_plan["stabilization_info"]
            config = artifact.env_helper.action_space_wrapper.get_configuration(action)
            requested_invocations = int(workload_step.arrival_count)
            repetitions = 0
            step_output_dir = output_dir / f"step_{step_index:02d}"
            switch_prime_record: dict[str, Any] | None = None
            needs_reconfiguration_prime = bool(
                int(reconfiguration_prime_repetitions) > 0
                and last_action is not None
                and int(action) != int(last_action)
            )
            if requested_invocations > 0:
                repetitions = _planned_real_repetitions(
                    requested_invocations=requested_invocations,
                    min_real_invocations_per_step=min_real_invocations_per_step,
                    max_real_invocations_per_step=max_real_invocations_per_step,
                )
                if needs_reconfiguration_prime:
                    switch_prime_metrics, switch_prime_invocations = _run_unmeasured_prime_replay(
                        benchmark=benchmark,
                        openwhisk_config=openwhisk_config,
                        output_dir=step_output_dir,
                        cache_dir=cache_dir,
                        function_name=function_name,
                        real_invoke_mode=real_invoke_mode,
                        deployment_handle=real_deployment_handle,
                        memory_mb=config.memory_mb,
                        timeout_sec=config.timeout_sec,
                        architecture=config.architecture,
                        repetitions=int(reconfiguration_prime_repetitions),
                        prime_label="reconfiguration_prime",
                    )
                    switch_prime_settles = _run_settle_probes(
                        benchmark=benchmark,
                        openwhisk_config=openwhisk_config,
                        output_dir=step_output_dir,
                        cache_dir=cache_dir,
                        function_name=function_name,
                        real_invoke_mode=real_invoke_mode,
                        deployment_handle=real_deployment_handle,
                        memory_mb=config.memory_mb,
                        timeout_sec=config.timeout_sec,
                        architecture=config.architecture,
                        triggering_metrics=switch_prime_metrics,
                        settle_repetitions=max(int(repetitions), 1),
                        settle_label_prefix="reconfiguration_prime",
                    )
                    switch_prime_record = {
                        "prime_repetitions": int(reconfiguration_prime_repetitions),
                        "metrics": switch_prime_metrics,
                        "invocations": switch_prime_invocations,
                    }
                    if switch_prime_settles:
                        switch_prime_record["settle_probes"] = switch_prime_settles
                if real_invoke_mode == "parallel":
                    assert real_deployment_handle is not None
                    metrics, invocations = run_real_step_parallel(
                        deployment_handle=real_deployment_handle,
                        memory_mb=config.memory_mb,
                        timeout_sec=config.timeout_sec,
                        architecture=config.architecture,
                        repetitions=repetitions,
                    )
                else:
                    metrics, invocations = run_real_step_sequential(
                        benchmark=benchmark,
                        openwhisk_config=openwhisk_config,
                        output_dir=step_output_dir,
                        cache_dir=cache_dir,
                        function_name=function_name,
                        memory_mb=config.memory_mb,
                        timeout_sec=config.timeout_sec,
                        architecture=config.architecture,
                        repetitions=repetitions,
                    )
            else:
                metrics = {
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
                    "benchmark_mean_ms": math.nan,
                    "latencies_ms": [],
                    "cold_flags": [],
                }
                invocations = []

            if artifact.shadow_env is not None:
                artifact.shadow_env.step(int(action))

            artifact.env_helper.state_space.update_configuration(
                config.memory_mb,
                config.architecture,
                config.timeout_sec,
            )
            if metrics["arrival_count"] > 0:
                arrival_times = np.linspace(
                    current_time_sec,
                    current_time_sec + float(workload_step.step_duration_sec),
                    num=metrics["arrival_count"],
                    endpoint=False,
                    dtype=np.float64,
                )
                response_times_sec = np.asarray(metrics["latencies_ms"], dtype=np.float64) / 1000.0
                completion_times = arrival_times + response_times_sec
                cold_flags = np.asarray(metrics["cold_flags"], dtype=bool)
                artifact.env_helper.state_space.record_batch(
                    arrival_times=arrival_times,
                    completion_times=completion_times,
                    response_times=response_times_sec,
                    cold_start_flags=cold_flags,
                )
            artifact.env_helper.state_space.update_utilization(
                float(metrics["cpu_utilization"]),
                float(metrics["memory_utilization"]),
            )
            artifact.env_helper.state_space.update_performance(
                latency=float(metrics["p95_latency"]),
                cost=float(metrics["cost"]),
                success=float(metrics["success_rate"]),
                cold_start=float(metrics["cold_start_rate"]),
                invocation_count=int(metrics["arrival_count"]),
            )

            raw_reward = float(artifact.env_helper._compute_reward(metrics))
            switch_penalty = 0.0
            adaptation_penalty = 0.0
            if last_action is not None and action != last_action:
                switch_penalty = float(environment_cfg["switch_penalty"])
            elif load_recently_changed and last_action is not None:
                adaptation_penalty = float(environment_cfg.get("adaptation_penalty", 0.0))
            raw_reward -= switch_penalty
            raw_reward -= adaptation_penalty

            current_time_sec += float(workload_step.step_duration_sec)
            artifact.env_helper.state_space.set_simulation_time(current_time_sec)
            next_workload = (
                workload_steps[step_index + 1] if step_index + 1 < len(workload_steps) else None
            )
            if next_workload is not None:
                load_recently_changed = bool(
                    dynamic_reference._has_meaningful_change(
                        previous=workload_step,
                        current=next_workload,
                        step_index=step_index + 1,
                    )
                )
            else:
                load_recently_changed = False

            observation = artifact.env_helper.state_space.extract_state()
            step_record = {
                "step": step_index,
                "policy": artifact.policy_name,
                "action": action,
                "raw_proposed_action": int(raw_proposed_action),
                "proposed_action": int(proposed_action),
                "config": config.to_dict(),
                "config_label": str(config),
                "raw_proposed_config": (
                    artifact.env_helper.action_space_wrapper.get_configuration(
                        int(raw_proposed_action)
                    ).to_dict()
                ),
                "proposed_config": artifact.env_helper.action_space_wrapper.get_configuration(
                    int(proposed_action)
                ).to_dict(),
                "default_anchor_action": int(default_anchor_action),
                "mask": mask_info,
                "requested_arrivals": requested_invocations,
                "executed_repetitions": repetitions,
                "workload": _serialize_workload(workload_step),
                "metrics": metrics,
                "raw_reward": raw_reward,
                "switch_penalty": switch_penalty,
                "adaptation_penalty": adaptation_penalty,
                "load_recently_changed_for_next_step": load_recently_changed,
                "function_name": function_name,
                "default_anchor": default_anchor_info,
                "stabilization": stabilization_info,
                "reconfiguration_prime": switch_prime_record,
                "invocations": invocations,
            }
            step_records.append(step_record)
            _write_step_payload(step_output_dir / "step_metrics.json", step_record)
            last_action = action
            stabilization_state.last_step_metrics = {
                "p95_latency": float(metrics["p95_latency"]),
                "success_rate": float(metrics["success_rate"]),
                "timeout_rate": float(metrics["timeout_rate"]),
                "cold_start_rate": float(metrics["cold_start_rate"]),
            }
            stabilization_state.last_step_raw_reward = float(raw_reward)

            if stabilization_state.last_effective_action is None:
                stabilization_state.last_effective_action = int(action)
                stabilization_state.steps_since_switch = 1
            elif int(action) != int(stabilization_state.last_effective_action):
                stabilization_state.last_effective_action = int(action)
                stabilization_state.steps_since_switch = 1
            else:
                stabilization_state.steps_since_switch += 1
    finally:
        if real_deployment_handle is not None:
            close_real_deployment_handle(real_deployment_handle)

    raw_rewards = [float(row["raw_reward"]) for row in step_records]
    mean_latencies = [float(row["metrics"]["latency"]) for row in step_records]
    mean_p95_latencies = [float(row["metrics"]["p95_latency"]) for row in step_records]
    mean_costs = [float(row["metrics"]["cost"]) for row in step_records]
    success_rates = [float(row["metrics"]["success_rate"]) for row in step_records]
    cold_rates = [float(row["metrics"]["cold_start_rate"]) for row in step_records]

    return {
        "policy": artifact.policy_name,
        "display_name": artifact.display_name,
        "train_time_sec": artifact.train_time_sec,
        "benchmark": benchmark,
        "load_pattern": load_pattern,
        "seed": seed,
        "steps": len(step_records),
        "function_name": function_name,
        "mean_raw_reward": float(np.mean(raw_rewards)),
        "cumulative_raw_reward": float(np.sum(raw_rewards)),
        "mean_latency_ms": float(np.mean(mean_latencies)),
        "mean_p95_latency_ms": float(np.mean(mean_p95_latencies)),
        "mean_cost_usd": float(np.mean(mean_costs)),
        "mean_success_rate": float(np.mean(success_rates)),
        "mean_cold_start_rate": float(np.mean(cold_rates)),
        "feasible_action_count": (
            len(artifact.feasible_actions) if artifact.feasible_actions is not None else None
        ),
        "action_space_size": int(artifact.env_helper.action_space_wrapper.n_actions),
        "steady_state": _steady_state_summary(step_records),
        "prime_record": prime_record,
        "step_records": step_records,
    }


def summarize_pairwise(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute pairwise deltas against PPO when present."""
    if not results:
        return []
    primary = next(
        (result for result in results if str(result["policy"]) == "ppo"),
        results[0],
    )
    comparisons = []
    for baseline in results:
        if baseline is primary:
            continue
        comparisons.append(
            {
                "primary_policy": primary["policy"],
                "baseline_policy": baseline["policy"],
                "mean_raw_reward_delta": (
                    float(primary["mean_raw_reward"]) - float(baseline["mean_raw_reward"])
                ),
                "cumulative_raw_reward_delta": (
                    float(primary["cumulative_raw_reward"])
                    - float(baseline["cumulative_raw_reward"])
                ),
                "mean_p95_latency_delta_ms": (
                    float(primary["mean_p95_latency_ms"])
                    - float(baseline["mean_p95_latency_ms"])
                ),
                "mean_cost_delta_usd": (
                    float(primary["mean_cost_usd"]) - float(baseline["mean_cost_usd"])
                ),
                "mean_success_rate_delta": (
                    float(primary["mean_success_rate"])
                    - float(baseline["mean_success_rate"])
                ),
                "steady_state_mean_raw_reward_delta": (
                    (
                        float(primary["steady_state"]["mean_raw_reward"])
                        if primary["steady_state"]["mean_raw_reward"] is not None
                        else math.nan
                    )
                    - (
                        float(baseline["steady_state"]["mean_raw_reward"])
                        if baseline["steady_state"]["mean_raw_reward"] is not None
                        else math.nan
                    )
                ),
                "steady_state_mean_p95_latency_delta_ms": (
                    (
                        float(primary["steady_state"]["mean_p95_latency_ms"])
                        if primary["steady_state"]["mean_p95_latency_ms"] is not None
                        else math.nan
                    )
                    - (
                        float(baseline["steady_state"]["mean_p95_latency_ms"])
                        if baseline["steady_state"]["mean_p95_latency_ms"] is not None
                        else math.nan
                    )
                ),
            }
        )
    return comparisons


def _maybe_generate_reward_plot(
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate step-wise and cumulative reward plots when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping reward plot export because matplotlib is unavailable.", flush=True)
        return

    policy_colors = {
        "ppo": "#2E5EAA",
        "bayes_opt_online": "#C46A1A",
        "default": "#9C3D3D",
        "ppo_load_only": "#4C8C4A",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    max_step = 0
    for result in summary["results"]:
        policy_name = str(result["policy"])
        display_name = str(result["display_name"])
        step_indices = [int(row["step"]) for row in result["step_records"]]
        raw_rewards = [float(row["raw_reward"]) for row in result["step_records"]]
        cumulative_rewards = np.cumsum(np.asarray(raw_rewards, dtype=np.float64))
        color = policy_colors.get(policy_name, None)
        axes[0].plot(
            step_indices,
            raw_rewards,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=display_name,
            color=color,
        )
        axes[1].plot(
            step_indices,
            cumulative_rewards,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            label=display_name,
            color=color,
        )
        if step_indices:
            max_step = max(max_step, max(step_indices))

    pairwise = summary.get("pairwise", [])
    delta_parts = []
    if pairwise:
        for item in pairwise:
            delta_parts.append(
                f"vs {item['baseline_policy']}={float(item['mean_raw_reward_delta']):+.3f}"
            )
    delta_text = f" | {'; '.join(delta_parts)}" if delta_parts else ""

    axes[0].set_title("Step Raw Reward")
    axes[0].set_xlabel("Control Step")
    axes[0].set_ylabel("Raw Reward")
    axes[1].set_title("Cumulative Raw Reward")
    axes[1].set_xlabel("Control Step")
    axes[1].set_ylabel("Cumulative Reward")
    for ax in axes:
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_xticks(list(range(max_step + 1)))
        ax.legend(frameon=False)

    fig.suptitle(
        f"{summary['benchmark']} | {summary['load_pattern']} | seed={summary['seed']}{delta_text}"
    )
    pdf_path = output_dir / "reward_by_step.pdf"
    png_path = output_dir / "reward_by_step.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward plot to {pdf_path}", flush=True)
    print(f"Saved reward plot to {png_path}", flush=True)


def _summary_row(
    summary: dict[str, Any],
    case_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one compact summary row for a completed case."""
    pairwise = summary.get("pairwise", [])
    primary_vs_baseline = pairwise[0] if pairwise else {}
    pairwise_by_baseline = {
        str(item["baseline_policy"]): {
            "mean_raw_reward_delta": item.get("mean_raw_reward_delta"),
            "steady_state_mean_raw_reward_delta": item.get(
                "steady_state_mean_raw_reward_delta"
            ),
            "mean_p95_latency_delta_ms": item.get("mean_p95_latency_delta_ms"),
            "mean_cost_delta_usd": item.get("mean_cost_delta_usd"),
            "mean_success_rate_delta": item.get("mean_success_rate_delta"),
        }
        for item in pairwise
    }
    row = {
        "case_name": case_name,
        "benchmark": summary.get("benchmark"),
        "load_pattern": summary.get("load_pattern"),
        "seed": summary.get("seed"),
        "steps": summary.get("steps"),
        "primary_policy": primary_vs_baseline.get("primary_policy"),
        "baseline_policy": primary_vs_baseline.get("baseline_policy"),
        "mean_raw_reward_delta": primary_vs_baseline.get("mean_raw_reward_delta"),
        "steady_state_mean_raw_reward_delta": primary_vs_baseline.get(
            "steady_state_mean_raw_reward_delta"
        ),
        "mean_p95_latency_delta_ms": primary_vs_baseline.get(
            "mean_p95_latency_delta_ms"
        ),
        "mean_cost_delta_usd": primary_vs_baseline.get("mean_cost_delta_usd"),
        "mean_success_rate_delta": primary_vs_baseline.get("mean_success_rate_delta"),
        "baseline_policies": list(pairwise_by_baseline.keys()),
        "pairwise_by_baseline": pairwise_by_baseline,
        "output_dir": str(output_dir),
    }
    for baseline_policy, metrics in pairwise_by_baseline.items():
        row[f"{baseline_policy}_mean_raw_reward_delta"] = metrics[
            "mean_raw_reward_delta"
        ]
        row[f"{baseline_policy}_steady_state_mean_raw_reward_delta"] = metrics[
            "steady_state_mean_raw_reward_delta"
        ]
        row[f"{baseline_policy}_mean_p95_latency_delta_ms"] = metrics[
            "mean_p95_latency_delta_ms"
        ]
        row[f"{baseline_policy}_mean_cost_delta_usd"] = metrics[
            "mean_cost_delta_usd"
        ]
        row[f"{baseline_policy}_mean_success_rate_delta"] = metrics[
            "mean_success_rate_delta"
        ]
    return row


def run_single_case(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run one real-platform replay case and return its summary."""
    environment_cfg = dict(config["environment"])
    environment_cfg["episode_length"] = int(args.steps)

    if output_dir.exists() and not args.keep_raw:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workload_steps, reset_info = build_workload_sequence(
        benchmark=args.benchmark,
        environment_cfg=environment_cfg,
        load_pattern=args.load_pattern,
        steps=args.steps,
        seed=args.seed,
    )
    _write_step_payload(
        output_dir / "workload_sequence.json",
        {
            "benchmark": args.benchmark,
            "load_pattern": args.load_pattern,
            "seed": args.seed,
            "reset_info": reset_info,
            "workload_steps": [_serialize_workload(step) for step in workload_steps],
        },
    )

    print("=" * 80)
    print("Real-Platform Online Study")
    print("=" * 80)
    print(f"Benchmark: {args.benchmark}")
    print(f"Load pattern: {args.load_pattern}")
    print(f"Policies: {args.policies}")
    policy_execution_order = _resolve_policy_execution_order(
        policies=list(args.policies),
        benchmark=str(args.benchmark),
        load_pattern=str(args.load_pattern),
        seed=int(args.seed),
        mode=str(args.policy_order_mode),
    )
    print(f"Policy execution order: {policy_execution_order}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print(f"Training timesteps: {args.total_timesteps}")
    print(f"Output dir: {output_dir}")
    print("=" * 80, flush=True)

    artifacts_by_name = {}
    for policy_name in args.policies:
        print(f"\nPreparing policy: {policy_name}", flush=True)
        artifact = build_policy_artifact(
            policy_name=policy_name,
            config=config,
            benchmark=args.benchmark,
            load_pattern=args.load_pattern,
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            enforce_feasible_actions=args.enforce_feasible_actions,
            feasible_actions_source=args.feasible_actions_source,
        )
        artifacts_by_name[str(policy_name)] = artifact
        feasible_count = (
            len(artifact.feasible_actions) if artifact.feasible_actions is not None else None
        )
        if feasible_count is not None:
            print(
                f"[{artifact.display_name}] "
                f"feasible_actions={feasible_count}/"
                f"{artifact.env_helper.action_space_wrapper.n_actions}",
                flush=True,
            )

    results = []
    for artifact_index, policy_name in enumerate(policy_execution_order):
        artifact = artifacts_by_name[str(policy_name)]
        print(f"\nRunning real-platform replay for {artifact.display_name}", flush=True)
        policy_output_dir = output_dir / artifact.policy_name
        cache_dir = output_dir / "cache" / artifact.policy_name
        result = run_policy_online(
            artifact=artifact,
            benchmark=args.benchmark,
            openwhisk_config=args.openwhisk_config,
            output_dir=policy_output_dir,
            cache_dir=cache_dir,
            function_prefix=args.function_prefix,
            workload_steps=workload_steps,
            environment_cfg=environment_cfg,
            load_pattern=args.load_pattern,
            seed=args.seed,
            max_real_invocations_per_step=args.max_real_invocations_per_step,
            min_real_invocations_per_step=args.min_real_invocations_per_step,
            keep_raw=args.keep_raw,
            real_invoke_mode=args.real_invoke_mode,
            stabilize_actions=args.stabilize_actions,
            stabilization_min_dwell_steps=args.stabilization_min_dwell_steps,
            stabilization_downgrade_patience=args.stabilization_downgrade_patience,
            stabilization_load_drop_threshold=args.stabilization_load_drop_threshold,
            initial_prime_repetitions=args.initial_prime_repetitions,
            initial_prime_lookahead_steps=args.initial_prime_lookahead_steps,
            reconfiguration_prime_repetitions=args.reconfiguration_prime_repetitions,
            default_anchor=args.default_anchor,
            default_anchor_release_patience=args.default_anchor_release_patience,
            default_anchor_min_arrivals=args.default_anchor_min_arrivals,
            default_anchor_load_ratio_threshold=args.default_anchor_load_ratio_threshold,
            default_anchor_min_burstiness=args.default_anchor_min_burstiness,
            default_anchor_min_step=args.default_anchor_min_step,
        )
        results.append(result)
        print(
            f"[{artifact.display_name}] "
            f"mean_raw_reward={result['mean_raw_reward']:.4f}, "
            f"cumulative_raw_reward={result['cumulative_raw_reward']:.4f}, "
            f"mean_p95_latency={result['mean_p95_latency_ms']:.2f} ms, "
            f"steady_mean_raw_reward="
            f"{result['steady_state']['mean_raw_reward'] if result['steady_state']['mean_raw_reward'] is not None else float('nan'):.4f}, "
            f"success={result['mean_success_rate']:.3f}",
            flush=True,
        )
        if (
            artifact_index < len(policy_execution_order) - 1
            and float(args.inter_policy_idle_sec) > 0.0
        ):
            print(
                f"Idling for {float(args.inter_policy_idle_sec):.1f}s before the next policy replay.",
                flush=True,
            )
            time.sleep(float(args.inter_policy_idle_sec))

    summary = {
        "benchmark": args.benchmark,
        "load_pattern": args.load_pattern,
        "seed": args.seed,
        "steps": args.steps,
        "policies": args.policies,
        "policy_execution_order": policy_execution_order,
        "total_timesteps": args.total_timesteps,
        "max_real_invocations_per_step": args.max_real_invocations_per_step,
        "real_invoke_mode": args.real_invoke_mode,
        "enforce_feasible_actions": args.enforce_feasible_actions,
        "feasible_actions_source": (
            str(args.feasible_actions_source) if args.enforce_feasible_actions else None
        ),
        "initial_prime_repetitions": args.initial_prime_repetitions,
        "initial_prime_lookahead_steps": args.initial_prime_lookahead_steps,
        "reconfiguration_prime_repetitions": args.reconfiguration_prime_repetitions,
        "policy_order_mode": args.policy_order_mode,
        "inter_policy_idle_sec": args.inter_policy_idle_sec,
        "stabilize_actions": args.stabilize_actions,
        "default_anchor": args.default_anchor,
        "default_anchor_release_patience": args.default_anchor_release_patience,
        "default_anchor_min_arrivals": args.default_anchor_min_arrivals,
        "default_anchor_load_ratio_threshold": args.default_anchor_load_ratio_threshold,
        "default_anchor_min_burstiness": args.default_anchor_min_burstiness,
        "default_anchor_min_step": args.default_anchor_min_step,
        "stabilization_min_dwell_steps": args.stabilization_min_dwell_steps,
        "stabilization_downgrade_patience": args.stabilization_downgrade_patience,
        "stabilization_load_drop_threshold": args.stabilization_load_drop_threshold,
        "results": results,
        "pairwise": summarize_pairwise(results),
    }
    _write_step_payload(output_dir / "summary.json", summary)
    _maybe_generate_reward_plot(summary, output_dir)
    print(f"\nSaved summary to {output_dir / 'summary.json'}", flush=True)
    return summary


def run_batch_cases(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a batch of real-platform replay cases from a JSON spec."""
    batch_spec = load_batch_spec(args.batch_spec)
    base_output_dir = args.output_dir
    if base_output_dir.exists() and not args.keep_raw:
        shutil.rmtree(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    shared_overrides = batch_spec.get("shared_overrides", {})
    if shared_overrides is not None and not isinstance(shared_overrides, dict):
        raise ValueError("'shared_overrides' must be a JSON object when provided.")

    batch_rows = []
    case_summaries = []
    for index, case in enumerate(batch_spec["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"Case #{index} in batch spec must be a JSON object.")
        case_args = copy.deepcopy(args)
        for key, value in shared_overrides.items():
            if hasattr(case_args, key):
                current_value = getattr(case_args, key)
                if isinstance(current_value, Path) and isinstance(value, str):
                    value = Path(value)
                setattr(case_args, key, value)
        for key, value in case.items():
            if key in {"name", "output_subdir"}:
                continue
            if hasattr(case_args, key):
                current_value = getattr(case_args, key)
                if isinstance(current_value, Path) and isinstance(value, str):
                    value = Path(value)
                setattr(case_args, key, value)

        case_name = str(
            case.get(
                "name",
                f"{case_args.benchmark}_{case_args.load_pattern}_seed{case_args.seed}",
            )
        )
        output_subdir = str(case.get("output_subdir", _sanitize_case_name(case_name)))
        case_output_dir = base_output_dir / output_subdir

        print("\n" + "#" * 80, flush=True)
        print(
            f"Batch case {index + 1}/{len(batch_spec['cases'])}: {case_name}",
            flush=True,
        )
        print("#" * 80, flush=True)
        summary = run_single_case(
            args=case_args,
            config=copy.deepcopy(config),
            output_dir=case_output_dir,
        )
        batch_rows.append(_summary_row(summary, case_name=case_name, output_dir=case_output_dir))
        case_summaries.append(
            {
                "case_name": case_name,
                "output_dir": str(case_output_dir),
                "summary": summary,
            }
        )

    batch_summary = {
        "config": str(args.config),
        "openwhisk_config": str(args.openwhisk_config),
        "batch_spec": str(args.batch_spec),
        "shared_overrides": shared_overrides,
        "cases": case_summaries,
        "rows": batch_rows,
    }
    _write_step_payload(base_output_dir / "batch_summary.json", batch_summary)
    print(f"\nSaved batch summary to {base_output_dir / 'batch_summary.json'}", flush=True)
    return batch_summary


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    if args.batch_spec is not None:
        run_batch_cases(args=args, config=config)
        return
    run_single_case(args=args, config=config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
