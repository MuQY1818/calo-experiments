#!/usr/bin/env python3
"""Run dynamic-load experiments that compare CALO and baseline policies."""

import json
import argparse
import copy
import os
import random
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rl_optimizer.environment import ServerlessFunctionEnv
from rl_optimizer.dynamic_environment import (
    DynamicLoadEnvironment,
    MultiBenchmarkDynamicEnv,
)
from rl_optimizer.state_space import StateSpace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


ALGORITHM_DISPLAY_NAMES = {
    'ppo': 'CALO',
    'ppo_flat': 'Flat-Fusion CALO',
    'ppo_load_only': 'Load-Only CALO',
    'bayes_opt_online': 'Bayes Opt',
    'default': 'Provider Default',
    'random': 'Random',
    'greedy': 'Greedy Profiling',
}
PPO_ALGORITHMS = {'ppo', 'ppo_flat', 'ppo_load_only'}
SUPPORTED_DYNAMIC_ALGORITHMS = (
    PPO_ALGORITHMS | {'bayes_opt_online', 'default', 'random', 'greedy'}
)


class _SmokeTestStateSpace:
    """Lightweight state-space stub for fast simulator smoke tests."""

    def __init__(self, code_feature_dim: int = StateSpace.DEFAULT_CODE_FEATURE_DIM):
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
            last_request_time = None

        self.load_monitor = _LoadMonitor()

    def set_function(self, benchmark_name: str):
        self.benchmark_name = benchmark_name

    def reset(self):
        pass

    def set_simulation_time(self, current_time: float):
        del current_time

    def update_configuration(self, memory_mb: int, architecture: str, timeout_sec: int):
        del memory_mb, architecture, timeout_sec

    def extract_state(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float32)

    def update_load(self, is_cold_start: bool = False):
        del is_cold_start

    def update_response(self, response_time: float):
        del response_time

    def update_utilization(self, cpu: float, memory: float):
        del cpu, memory

    def update_performance(
        self,
        latency: float,
        cost: float,
        success: float,
        cold_start: float,
        invocation_count: int = 1,
    ):
        del latency, cost, success, cold_start, invocation_count

    def record_batch(
        self,
        arrival_times,
        completion_times,
        response_times,
        cold_start_flags,
    ):
        del completion_times, response_times, cold_start_flags
        if len(arrival_times):
            self.load_monitor.last_request_time = float(arrival_times[-1])


def _format_smoke_workload_label(reset_info: dict) -> str:
    """Build a workload label that is readable in smoke-test logs."""
    workload_source = str(reset_info.get('workload_source', 'n/a'))
    workload_profile = reset_info.get('workload_profile', {})
    if workload_source == 'azure_trace':
        function_key = workload_profile.get('function_key')
        if function_key:
            return str(function_key)
    load_pattern = reset_info.get('load_pattern')
    if load_pattern:
        return str(load_pattern)
    return workload_source


def _format_smoke_calibration_label(base_env: ServerlessFunctionEnv) -> str:
    """Return a short calibration label for smoke-test logs."""
    calibration_dirs = getattr(base_env.service_model, 'calibration_dirs', None)
    if calibration_dirs:
        parts = [
            f"{architecture}:{Path(directory).name}"
            for architecture, directory in sorted(calibration_dirs.items())
        ]
        return ",".join(parts)
    calibration_dir = getattr(base_env.service_model, 'calibration_dir', None)
    if calibration_dir is None:
        return 'heuristic'
    return Path(calibration_dir).name


def _resolve_environment_calibration(
    environment_cfg: dict,
    override_calibration_dir: str | None,
    disable_calibration: bool,
) -> None:
    """Apply CLI calibration overrides to the environment config in place."""
    if disable_calibration:
        environment_cfg['calibration_dir'] = None
        environment_cfg['calibration_dirs'] = None
        return

    if override_calibration_dir is not None:
        environment_cfg['calibration_dir'] = override_calibration_dir
        environment_cfg['calibration_dirs'] = None


def _format_config_calibration_label(environment_cfg: dict) -> str:
    """Return a readable calibration label for config-level logging."""
    calibration_dirs = environment_cfg.get('calibration_dirs')
    if calibration_dirs:
        return ', '.join(
            f"{architecture}:{Path(directory).name}"
            for architecture, directory in sorted(calibration_dirs.items())
        )
    return environment_cfg.get('calibration_dir') or 'heuristic'


def _json_default(value):
    """Convert NumPy types to plain Python types for JSON serialization."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    """Write a JSON file with stable formatting."""
    with open(path, 'w') as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )


def _timestamp() -> str:
    """Return a wall-clock timestamp string for progress tracking."""
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _save_progress(output_dir: Path, progress: dict) -> None:
    """Persist progress metadata for long-running experiments."""
    progress['updated_at'] = _timestamp()
    _write_json(output_dir / 'progress.json', progress)


def _save_partial_results(output_dir: Path, results: dict) -> None:
    """Persist partial results so mid-run status is visible."""
    _write_json(output_dir / 'dynamic_experiment_results.partial.json', results)


def _load_json(path: Path) -> dict:
    """Load one JSON file from disk."""
    with open(path, 'r') as f:
        return json.load(f)


def _ordered_case_keys(
    benchmarks: list[str],
    load_patterns: list[str],
) -> list[tuple[str, str]]:
    """Return cases in the same order as the main experiment loop."""
    return [
        (benchmark, load_pattern)
        for benchmark in benchmarks
        for load_pattern in load_patterns
    ]


def _normalize_results_shape(
    results: dict,
    algorithms: list[str],
    benchmarks: list[str],
) -> dict:
    """Ensure partially loaded results match the configured layout."""
    normalized = {}
    source_results = results if isinstance(results, dict) else {}
    for algorithm in algorithms:
        algorithm_results = source_results.get(algorithm, {})
        if not isinstance(algorithm_results, dict):
            algorithm_results = {}
        normalized[algorithm] = {}
        for benchmark in benchmarks:
            benchmark_results = algorithm_results.get(benchmark, {})
            normalized[algorithm][benchmark] = (
                dict(benchmark_results)
                if isinstance(benchmark_results, dict)
                else {}
            )
    return normalized


def _is_case_complete(
    results: dict,
    algorithms: list[str],
    benchmark: str,
    load_pattern: str,
) -> bool:
    """Return whether one benchmark/load-pattern pair is fully finished."""
    return all(
        load_pattern in results.get(algorithm, {}).get(benchmark, {})
        for algorithm in algorithms
    )


def _load_resume_state(
    *,
    output_dir: Path,
    algorithms: list[str],
    benchmarks: list[str],
    load_patterns: list[str],
    config_path: str,
    seed: int,
    calibration_label: str,
) -> tuple[dict, dict, set[tuple[str, str]]]:
    """Load existing partial artifacts and build a resumable state."""
    partial_path = output_dir / 'dynamic_experiment_results.partial.json'
    final_path = output_dir / 'dynamic_experiment_results.json'
    progress_path = output_dir / 'progress.json'

    raw_results = {}
    if final_path.exists():
        raw_results = _load_json(final_path)
    elif partial_path.exists():
        raw_results = _load_json(partial_path)

    results = _normalize_results_shape(
        results=raw_results,
        algorithms=algorithms,
        benchmarks=benchmarks,
    )
    existing_progress = (
        _load_json(progress_path)
        if progress_path.exists()
        else {}
    )
    ordered_cases = _ordered_case_keys(benchmarks, load_patterns)
    completed_case_keys = {
        (benchmark, load_pattern)
        for benchmark, load_pattern in ordered_cases
        if _is_case_complete(
            results=results,
            algorithms=algorithms,
            benchmark=benchmark,
            load_pattern=load_pattern,
        )
    }

    last_completed_case = None
    for benchmark, load_pattern in ordered_cases:
        if (benchmark, load_pattern) in completed_case_keys:
            last_completed_case = f"{benchmark} / {load_pattern}"

    progress = {
        'status': 'running',
        'started_at': existing_progress.get('started_at', _timestamp()),
        'config_path': config_path,
        'output_dir': str(output_dir),
        'seed': seed,
        'calibration_dir': calibration_label,
        'total_cases': len(ordered_cases),
        'completed_cases': len(completed_case_keys),
        'current_benchmark': existing_progress.get('current_benchmark'),
        'current_load_pattern': existing_progress.get('current_load_pattern'),
        'current_algorithm': existing_progress.get('current_algorithm'),
        'current_stage': 'resuming',
        'last_completed_case': last_completed_case,
    }
    return results, progress, completed_case_keys


def _relative_improvement(candidate: float, baseline: float) -> float:
    """Compute percentage improvement with safe zero handling."""
    baseline_abs = abs(float(baseline))
    if baseline_abs < 1e-12:
        return 0.0 if abs(float(candidate)) < 1e-12 else float('inf')
    return (float(candidate) - float(baseline)) / baseline_abs * 100.0


def _is_small_baseline(baseline: float, threshold: float = 0.05) -> bool:
    """Return whether one baseline reward is too small for stable percentages."""
    return abs(float(baseline)) < threshold


def _format_relative_improvement(value: float) -> str:
    """Render one relative-improvement value for user-facing logs."""
    if not np.isfinite(value):
        return 'n/a'
    return f'{value:+.2f}%'


def _format_case_comparison(candidate: float, baseline: float) -> str:
    """Format one candidate-versus-baseline comparison for logs."""
    raw_delta = float(candidate) - float(baseline)
    relative = _relative_improvement(candidate, baseline)
    small_baseline_note = ' [small baseline]' if _is_small_baseline(baseline) else ''
    return (
        f"raw_delta={raw_delta:+.4f}, "
        f"relative={_format_relative_improvement(relative)}"
        f"{small_baseline_note}"
    )


def _collect_case_comparisons(
    results: dict,
    primary_algorithm: str,
    baseline_algorithm: str,
) -> list[dict]:
    """Collect per-case comparisons for one algorithm pair."""
    comparisons = []
    primary_results = results.get(primary_algorithm, {})
    baseline_results = results.get(baseline_algorithm, {})

    for benchmark, primary_data in primary_results.items():
        baseline_data = baseline_results.get(benchmark, {})
        for load_pattern, primary_metrics in primary_data.items():
            baseline_metrics = baseline_data.get(load_pattern)
            if baseline_metrics is None:
                continue

            primary_reward = float(primary_metrics['mean_reward'])
            baseline_reward = float(baseline_metrics['mean_reward'])
            comparisons.append({
                'benchmark': benchmark,
                'load_pattern': load_pattern,
                'primary_reward': primary_reward,
                'baseline_reward': baseline_reward,
                'raw_reward_delta': primary_reward - baseline_reward,
                'relative_improvement_pct': _relative_improvement(
                    primary_reward,
                    baseline_reward,
                ),
                'small_baseline': _is_small_baseline(baseline_reward),
                'primary_win': primary_reward > baseline_reward,
            })

    return comparisons


def _summarize_case_comparisons(comparisons: list[dict]) -> dict:
    """Summarize one list of per-case comparisons using robust metrics."""
    if not comparisons:
        return {
            'total_comparisons': 0,
            'primary_wins': 0,
            'win_rate_pct': 0.0,
            'mean_raw_reward_delta': 0.0,
            'median_relative_improvement_pct': 0.0,
            'best_raw_reward_delta': 0.0,
            'worst_raw_reward_delta': 0.0,
            'small_baseline_cases': 0,
        }

    raw_deltas = np.array(
        [row['raw_reward_delta'] for row in comparisons],
        dtype=float,
    )
    finite_relative_improvements = np.array(
        [
            row['relative_improvement_pct']
            for row in comparisons
            if np.isfinite(row['relative_improvement_pct'])
        ],
        dtype=float,
    )
    primary_wins = sum(row['primary_win'] for row in comparisons)
    total_comparisons = len(comparisons)

    return {
        'total_comparisons': total_comparisons,
        'primary_wins': primary_wins,
        'win_rate_pct': primary_wins / total_comparisons * 100.0,
        'mean_raw_reward_delta': float(np.mean(raw_deltas)),
        'median_relative_improvement_pct': float(
            np.median(finite_relative_improvements)
        ) if finite_relative_improvements.size else 0.0,
        'best_raw_reward_delta': float(np.max(raw_deltas)),
        'worst_raw_reward_delta': float(np.min(raw_deltas)),
        'small_baseline_cases': sum(row['small_baseline'] for row in comparisons),
    }


def _summarize_feature_attribution(results: dict, algorithm: str = 'ppo') -> dict | None:
    """Aggregate per-case feature-attribution results into compact summaries."""
    algorithm_results = results.get(algorithm, {})
    if not algorithm_results:
        return None

    group_buckets: dict[str, list[float]] = {}
    feature_buckets: dict[str, dict[str, object]] = {}
    total_cases = 0

    for benchmark, benchmark_data in algorithm_results.items():
        for load_pattern, metrics in benchmark_data.items():
            attribution = metrics.get('feature_attribution')
            if not attribution:
                continue
            total_cases += 1

            for row in attribution.get('group_importance', []):
                name = str(row['name'])
                group_buckets.setdefault(name, []).append(float(row['reward_drop']))

            for row in attribution.get('single_feature_importance', []):
                name = str(row['name'])
                bucket = feature_buckets.setdefault(
                    name,
                    {
                        'name': name,
                        'group': str(row['group']),
                        'reward_drops': [],
                    },
                )
                bucket['reward_drops'].append(float(row['reward_drop']))

    if total_cases == 0:
        return None

    group_summary = [
        {
            'name': name,
            'mean_reward_drop': float(np.mean(values)),
            'std_reward_drop': float(np.std(values)),
            'mean_absolute_reward_shift': float(np.mean(np.abs(values))),
            'cases': len(values),
        }
        for name, values in group_buckets.items()
    ]
    group_summary.sort(
        key=lambda item: item['mean_reward_drop'],
        reverse=True,
    )

    single_feature_summary = []
    for bucket in feature_buckets.values():
        values = np.asarray(bucket['reward_drops'], dtype=float)
        single_feature_summary.append(
            {
                'name': str(bucket['name']),
                'group': str(bucket['group']),
                'mean_reward_drop': float(np.mean(values)),
                'std_reward_drop': float(np.std(values)),
                'mean_absolute_reward_shift': float(np.mean(np.abs(values))),
                'cases': int(values.size),
            }
        )
    single_feature_summary.sort(
        key=lambda item: item['mean_reward_drop'],
        reverse=True,
    )

    return {
        'algorithm': algorithm,
        'cases': total_cases,
        'group_importance_mean_reward_drop': group_summary,
        'single_feature_mean_reward_drop': single_feature_summary,
    }


def _algorithm_display_name(algorithm: str) -> str:
    """Return a readable label for one algorithm identifier."""
    return ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)


def _build_state_space_for_algorithm(
    algorithm: str,
    environment_cfg: dict | None = None,
    algorithm_settings: dict | None = None,
) -> StateSpace:
    """Construct one state-space variant for a specific algorithm."""
    environment_cfg = environment_cfg or {}
    settings = {}
    if algorithm_settings is not None:
        settings = dict(algorithm_settings.get(algorithm, {}))

    disable_code_features = bool(
        settings.get('disable_code_features', algorithm == 'ppo_load_only')
    )
    disable_function_category = bool(
        settings.get('disable_function_category', algorithm == 'ppo_load_only')
    )
    code_feature_dim = int(
        settings.get(
            'code_feature_dim',
            environment_cfg.get(
                'code_feature_dim',
                StateSpace.DEFAULT_CODE_FEATURE_DIM,
            ),
        )
    )

    return StateSpace(
        sebs_root='.',
        enable_code_features=not disable_code_features,
        enable_function_category=not disable_function_category,
        code_feature_dim=code_feature_dim,
    )


def _build_dynamic_env(
    benchmark: str,
    environment_cfg: dict,
    load_pattern: str,
    algorithm: str,
    algorithm_settings: dict | None = None,
):
    """Build one fresh dynamic environment for the selected algorithm."""
    base_env = ServerlessFunctionEnv(
        benchmark=benchmark,
        deployment='local',
        enable_real_execution=False,
        normalize_reward=environment_cfg.get('normalize_reward', False),
        reward_weights=environment_cfg.get('reward_weights'),
        reward_penalties=environment_cfg.get('reward_penalties'),
        success_bonus=environment_cfg.get('success_bonus', 0.5),
        cost_normalization=environment_cfg.get('cost_normalization', 'ratio'),
        calibration_dir=environment_cfg.get('calibration_dir'),
        calibration_dirs=environment_cfg.get('calibration_dirs'),
        step_duration_sec=environment_cfg.get('step_duration_sec', 15.0),
        max_containers=environment_cfg.get('max_containers', 32),
        container_ttl_sec=environment_cfg.get('container_ttl_sec', 600.0),
        state_space=_build_state_space_for_algorithm(
            algorithm,
            environment_cfg=environment_cfg,
            algorithm_settings=algorithm_settings,
        ),
        action_space_config=environment_cfg.get('action_space'),
    )

    dynamic_env = DynamicLoadEnvironment(
        base_env,
        episode_length=environment_cfg['episode_length'],
        load_change_freq=environment_cfg['load_change_freq'],
        switch_penalty=environment_cfg['switch_penalty'],
        adaptation_penalty=environment_cfg.get('adaptation_penalty', 0.05),
        workload_source=environment_cfg.get('workload_source', 'synthetic'),
        azure_profile_path=environment_cfg.get('azure_profile_path'),
        azure_summary_path=environment_cfg.get('azure_summary_path'),
        azure_top_k=environment_cfg.get('azure_top_k', 50),
        azure_arrival_scale=environment_cfg.get('azure_arrival_scale', 1.0),
        azure_max_arrivals_per_step=environment_cfg.get('azure_max_arrivals_per_step'),
        azure_profile_selection=environment_cfg.get('azure_profile_selection', 'benchmark_aware'),
        azure_selection_pool_size=environment_cfg.get('azure_selection_pool_size', 16),
        azure_target_concurrency=environment_cfg.get('azure_target_concurrency'),
        calibration_dirs=environment_cfg.get('calibration_dirs'),
    )
    dynamic_env.load_pattern = load_pattern
    return dynamic_env


def _merge_nested_dict(base: dict, overrides: dict) -> dict:
    """Recursively merge override values into a shallow-copied dict."""
    merged = dict(base)
    for key, value in overrides.items():
        current_value = merged.get(key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            merged[key] = _merge_nested_dict(current_value, value)
        else:
            merged[key] = value
    return merged


def _resolve_algorithm_ppo_kwargs(
    base_ppo_kwargs: dict,
    algorithm: str,
    algorithm_settings: dict | None = None,
) -> dict:
    """Apply algorithm-specific PPO overrides from the experiment config."""
    resolved_kwargs = copy.deepcopy(base_ppo_kwargs)
    settings = {}
    if algorithm_settings is not None:
        settings = dict(algorithm_settings.get(algorithm, {}))

    feature_extractor_name = settings.get('feature_extractor_name')
    if feature_extractor_name is not None:
        resolved_kwargs['feature_extractor_name'] = str(feature_extractor_name)

    ppo_kwargs_overrides = settings.get('ppo_kwargs_overrides')
    if ppo_kwargs_overrides:
        resolved_kwargs = _merge_nested_dict(
            resolved_kwargs,
            ppo_kwargs_overrides,
        )
    return resolved_kwargs


def _resolve_shared_training_setup(
    config: dict,
    environment_cfg: dict,
    eval_benchmark: str,
) -> tuple[list[str], list[str]] | None:
    """Resolve shared-training benchmarks/load patterns for one eval case."""
    shared_cfg = config.get('shared_training', {})
    if not shared_cfg.get('enabled', False):
        return None

    training_benchmarks = [
        str(name) for name in shared_cfg.get('benchmarks', config['benchmarks'])
    ]
    if shared_cfg.get('exclude_eval_benchmark', False):
        training_benchmarks = [
            name for name in training_benchmarks if name != eval_benchmark
        ]
    if not training_benchmarks:
        training_benchmarks = [str(eval_benchmark)]

    load_patterns = [
        str(pattern)
        for pattern in shared_cfg.get('load_patterns', environment_cfg['load_patterns'])
    ]
    if not load_patterns:
        load_patterns = [str(environment_cfg['load_patterns'][0])]

    # Preserve user order while removing duplicates.
    training_benchmarks = list(dict.fromkeys(training_benchmarks))
    load_patterns = list(dict.fromkeys(load_patterns))
    return training_benchmarks, load_patterns


def _build_shared_training_env(
    config: dict,
    environment_cfg: dict,
    eval_benchmark: str,
    algorithm: str,
    algorithm_settings: dict | None = None,
    seed: int | None = None,
):
    """Build a shared-training env that samples benchmarks episode-wise."""
    shared_setup = _resolve_shared_training_setup(
        config=config,
        environment_cfg=environment_cfg,
        eval_benchmark=eval_benchmark,
    )
    if shared_setup is None:
        return None

    training_benchmarks, training_load_patterns = shared_setup

    def _env_factory(shared_benchmark: str) -> DynamicLoadEnvironment:
        return _build_dynamic_env(
            benchmark=shared_benchmark,
            environment_cfg=environment_cfg,
            load_pattern=training_load_patterns[0],
            algorithm=algorithm,
            algorithm_settings=algorithm_settings,
        )

    return MultiBenchmarkDynamicEnv(
        benchmark_names=training_benchmarks,
        env_factory=_env_factory,
        load_patterns=training_load_patterns,
        seed=seed,
    )


def _print_policy_summary(name: str, metrics: dict) -> None:
    """Print one policy summary using raw reward as the primary metric."""
    print(f"[{name}] Summary", flush=True)
    print(
        f"  Raw reward: {metrics['mean_reward']:.4f} ± "
        f"{metrics['std_reward']:.4f}",
        flush=True,
    )
    if 'mean_normalized_reward' in metrics:
        print(
            f"  Normalized reward: {metrics['mean_normalized_reward']:.4f} ± "
            f"{metrics['std_normalized_reward']:.4f}",
            flush=True,
        )
    print(
        f"  Latency: {metrics['mean_latency']:.2f} ± "
        f"{metrics['std_latency']:.2f} ms",
        flush=True,
    )
    print(
        f"  Cost: ${metrics['mean_cost']:.6f} ± "
        f"${metrics['std_cost']:.6f}",
        flush=True,
    )
    if 'mean_success_rate' in metrics:
        print(
            f"  Success rate: {metrics['mean_success_rate']:.3f}",
            flush=True,
        )


def run_dynamic_experiment(
    config_path: str = 'config/rl_experiments/tuning_uploader_recognition.json',
    output_dir: str = 'results_dynamic',
    seed: int | None = None,
    override_calibration_dir: str | None = None,
    disable_calibration: bool = False,
    resume: bool = False,
):
    """Run the CALO versus baseline experiment suite."""
    from rl_optimizer.baselines import (
        DefaultBaseline,
        GreedyBaseline,
        RandomBaseline,
    )
    from rl_optimizer.pure_ppo import PurePPO, OnlineBayesOptBaseline

    with open(config_path, 'r') as f:
        config = json.load(f)
    environment_cfg = config['environment']

    _resolve_environment_calibration(
        environment_cfg=environment_cfg,
        override_calibration_dir=override_calibration_dir,
        disable_calibration=disable_calibration,
    )

    if seed is None:
        seeds = config.get('reproducibility', {}).get('seeds', [])
        seed = int(seeds[0]) if seeds else 42
    algorithms = list(config.get('algorithms', ['ppo', 'bayes_opt_online']))
    feature_attribution_cfg = dict(config.get('feature_attribution', {}))
    unsupported_algorithms = [
        algorithm for algorithm in algorithms
        if algorithm not in SUPPORTED_DYNAMIC_ALGORITHMS
    ]
    if unsupported_algorithms:
        raise ValueError(
            "Unsupported dynamic experiment algorithms: "
            f"{unsupported_algorithms}. "
            f"Supported: {sorted(SUPPORTED_DYNAMIC_ALGORITHMS)}"
        )
    algorithm_settings = config.get('algorithm_settings', {})

    np.random.seed(seed)
    random.seed(seed)

    print("=" * 80)
    print("Dynamic Experiment")
    print("=" * 80)
    print(f"Config: {config['description']}")
    print(f"Benchmarks: {len(config['benchmarks'])}")
    print(f"Load patterns: {environment_cfg['load_patterns']}")
    print(
        "Algorithms: "
        f"{[_algorithm_display_name(algorithm) for algorithm in algorithms]}"
    )
    print(f"Seed: {seed}")
    print(
        "Calibration: "
        f"{_format_config_calibration_label(environment_cfg)}"
    )
    print("=" * 80, flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_label = _format_config_calibration_label(environment_cfg)
    if resume:
        results, progress, completed_case_keys = _load_resume_state(
            output_dir=output_dir,
            algorithms=algorithms,
            benchmarks=config['benchmarks'],
            load_patterns=environment_cfg['load_patterns'],
            config_path=config_path,
            seed=seed,
            calibration_label=calibration_label,
        )
        completed_cases = int(progress['completed_cases'])
        total_cases = int(progress['total_cases'])
        print(
            f"[Resume] Found {completed_cases}/{total_cases} completed cases in "
            f"{output_dir}",
            flush=True,
        )
        if completed_cases >= total_cases:
            print(
                f"[Resume] Experiment already completed in {output_dir}",
                flush=True,
            )
            return
    else:
        results = {algorithm: {} for algorithm in algorithms}
        total_cases = len(config['benchmarks']) * len(
            environment_cfg['load_patterns']
        )
        completed_cases = 0
        completed_case_keys = set()
        progress = {
            'status': 'running',
            'started_at': _timestamp(),
            'config_path': config_path,
            'output_dir': str(output_dir),
            'seed': seed,
            'calibration_dir': calibration_label,
            'total_cases': total_cases,
            'completed_cases': completed_cases,
            'current_benchmark': None,
            'current_load_pattern': None,
            'current_algorithm': None,
            'current_stage': 'initializing',
            'last_completed_case': None,
        }
    _save_progress(output_dir, progress)
    _save_partial_results(output_dir, results)

    case_order = {
        case_key: index
        for index, case_key in enumerate(
            _ordered_case_keys(
                benchmarks=config['benchmarks'],
                load_patterns=environment_cfg['load_patterns'],
            ),
            start=1,
        )
    }

    try:
        for benchmark in config['benchmarks']:
            print(f"\n{'='*80}")
            print(f"Benchmark: {benchmark}")
            print(f"{'='*80}", flush=True)

            benchmark_results = {}
            for algorithm in algorithms:
                benchmark_results[algorithm] = results[algorithm].setdefault(
                    benchmark,
                    {},
                )

            for load_pattern in environment_cfg['load_patterns']:
                case_index = case_order[(benchmark, load_pattern)]
                case_label = f"{benchmark} / {load_pattern}"
                if (benchmark, load_pattern) in completed_case_keys:
                    print(
                        f"[Resume] Skipping completed case "
                        f"{case_index}/{total_cases}: {case_label}",
                        flush=True,
                    )
                    continue
                print(f"\nLoad pattern: {load_pattern}")
                print("-" * 80)
                print(
                    f"[Progress] Case {case_index}/{total_cases}: {case_label}",
                    flush=True,
                )

                progress.update({
                    'current_benchmark': benchmark,
                    'current_load_pattern': load_pattern,
                    'current_algorithm': None,
                    'current_stage': 'creating_environment',
                })
                _save_progress(output_dir, progress)

                for algorithm_index, algorithm in enumerate(algorithms, start=1):
                    display_name = _algorithm_display_name(algorithm)
                    progress.update({
                        'current_algorithm': algorithm,
                        'current_stage': f'preparing_{algorithm}',
                    })
                    _save_progress(output_dir, progress)

                    dynamic_env = _build_dynamic_env(
                        benchmark=benchmark,
                        environment_cfg=environment_cfg,
                        load_pattern=load_pattern,
                        algorithm=algorithm,
                        algorithm_settings=algorithm_settings,
                    )
                    training_env = dynamic_env

                    if algorithm in PPO_ALGORITHMS:
                        shared_training_env = _build_shared_training_env(
                            config=config,
                            environment_cfg=environment_cfg,
                            eval_benchmark=benchmark,
                            algorithm=algorithm,
                            algorithm_settings=algorithm_settings,
                            seed=seed,
                        )
                        if shared_training_env is not None:
                            training_env = shared_training_env
                            training_benchmarks, training_load_patterns = (
                                _resolve_shared_training_setup(
                                    config=config,
                                    environment_cfg=environment_cfg,
                                    eval_benchmark=benchmark,
                                )
                            )
                            print(
                                f"[Shared Train] {display_name}: "
                                f"benchmarks={training_benchmarks}, "
                                f"load_patterns={training_load_patterns}, "
                                f"exclude_eval_benchmark="
                                f"{config.get('shared_training', {}).get('exclude_eval_benchmark', False)}",
                                flush=True,
                            )

                        progress['current_stage'] = f'training_{algorithm}'
                        _save_progress(output_dir, progress)
                        print(
                            f"\n[{algorithm_index}] Training {display_name}...",
                            flush=True,
                        )
                        resolved_ppo_kwargs = _resolve_algorithm_ppo_kwargs(
                            base_ppo_kwargs=config['training']['ppo_kwargs'],
                            algorithm=algorithm,
                            algorithm_settings=algorithm_settings,
                        )
                        ppo = PurePPO(
                            training_env,
                            total_timesteps=config['training']['total_timesteps'],
                            ppo_kwargs=resolved_ppo_kwargs,
                            seed=seed,
                            imitation_warmstart=(
                                config['training'].get('imitation_warmstart')
                            ),
                            validation_env=dynamic_env,
                            checkpoint_selection=(
                                config['training'].get('checkpoint_selection')
                            ),
                        )

                        progress_label = f"{case_label} [{display_name}]"
                        ppo_train_time = ppo.train(progress_label=progress_label)

                        progress['current_stage'] = f'evaluating_{algorithm}'
                        _save_progress(output_dir, progress)
                        print(f"\nEvaluating {display_name}...", flush=True)
                        algorithm_results = ppo.evaluate(
                            n_episodes=config['evaluation']['n_episodes'],
                            progress_label=progress_label,
                            eval_env=dynamic_env,
                        )
                        if ppo.imitation_warmstart_summary is not None:
                            algorithm_results['imitation_warmstart'] = (
                                ppo.imitation_warmstart_summary
                            )
                        if ppo.checkpoint_selection_summary is not None:
                            algorithm_results['checkpoint_selection'] = (
                                ppo.checkpoint_selection_summary
                            )
                        if (
                            feature_attribution_cfg.get('enabled', False)
                            and algorithm in PPO_ALGORITHMS
                        ):
                            target_algorithms = feature_attribution_cfg.get(
                                'algorithms',
                                ['ppo'],
                            )
                            if algorithm in target_algorithms:
                                progress['current_stage'] = (
                                    f'attribution_{algorithm}'
                                )
                                _save_progress(output_dir, progress)
                                print(
                                    f"\nRunning feature attribution for "
                                    f"{display_name}...",
                                    flush=True,
                                )
                                algorithm_results['feature_attribution'] = (
                                    ppo.evaluate_feature_attribution(
                                        n_episodes=int(
                                            feature_attribution_cfg.get(
                                                'n_episodes',
                                                config['evaluation']['n_episodes'],
                                            )
                                        ),
                                        eval_env=dynamic_env,
                                        progress_label=progress_label,
                                        single_feature_groups=tuple(
                                            feature_attribution_cfg.get(
                                                'single_feature_groups',
                                                [
                                                    'load',
                                                    'category',
                                                    'history',
                                                    'context',
                                                ],
                                            )
                                        ),
                                        max_single_features=(
                                            feature_attribution_cfg.get(
                                                'max_single_features'
                                            )
                                        ),
                                    )
                                )
                        algorithm_results['training_time'] = ppo_train_time
                    elif algorithm == 'bayes_opt_online':
                        progress['current_stage'] = f'evaluating_{algorithm}'
                        _save_progress(output_dir, progress)
                        print(
                            f"\n[{algorithm_index}] Evaluating {display_name}...",
                            flush=True,
                        )
                        bayes_opt = OnlineBayesOptBaseline(
                            dynamic_env,
                            reoptimize_freq=config['baseline']['reoptimize_freq'],
                            seed=seed,
                        )
                        algorithm_results = bayes_opt.evaluate(
                            n_episodes=config['evaluation']['n_episodes'],
                            progress_label=f"{case_label} [{display_name}]",
                        )
                    elif algorithm == 'default':
                        progress['current_stage'] = f'evaluating_{algorithm}'
                        _save_progress(output_dir, progress)
                        print(
                            f"\n[{algorithm_index}] Evaluating {display_name}...",
                            flush=True,
                        )
                        default_baseline = DefaultBaseline(
                            dynamic_env,
                            memory=config['baseline'].get('default_memory', 512),
                        )
                        algorithm_results = default_baseline.evaluate(
                            n_episodes=config['evaluation']['n_episodes'],
                            progress_label=f"{case_label} [{display_name}]",
                            seed=seed,
                        )
                    elif algorithm == 'random':
                        progress['current_stage'] = f'evaluating_{algorithm}'
                        _save_progress(output_dir, progress)
                        print(
                            f"\n[{algorithm_index}] Evaluating {display_name}...",
                            flush=True,
                        )
                        random_baseline = RandomBaseline(dynamic_env)
                        algorithm_results = random_baseline.evaluate(
                            n_episodes=config['evaluation']['n_episodes'],
                            progress_label=f"{case_label} [{display_name}]",
                            seed=seed,
                        )
                    elif algorithm == 'greedy':
                        progress['current_stage'] = f'evaluating_{algorithm}'
                        _save_progress(output_dir, progress)
                        print(
                            f"\n[{algorithm_index}] Evaluating {display_name}...",
                            flush=True,
                        )
                        greedy_baseline = GreedyBaseline(dynamic_env)
                        algorithm_results = greedy_baseline.evaluate(
                            n_episodes=config['evaluation']['n_episodes'],
                            progress_label=f"{case_label} [{display_name}]",
                            seed=seed,
                        )
                    else:
                        raise ValueError(f"Unsupported algorithm: {algorithm}")

                    _print_policy_summary(display_name, algorithm_results)
                    benchmark_results[algorithm][load_pattern] = algorithm_results
                    _save_partial_results(output_dir, results)

                if (
                    'ppo' in benchmark_results and
                    'bayes_opt_online' in benchmark_results and
                    load_pattern in benchmark_results['ppo'] and
                    load_pattern in benchmark_results['bayes_opt_online']
                ):
                    print(
                        f"\n[Comparison] CALO vs Bayes Opt: "
                        f"{_format_case_comparison(
                            benchmark_results['ppo'][load_pattern]['mean_reward'],
                            benchmark_results['bayes_opt_online'][load_pattern]['mean_reward'],
                        )}",
                        flush=True,
                    )

                if (
                    'ppo' in benchmark_results and
                    'ppo_load_only' in benchmark_results and
                    load_pattern in benchmark_results['ppo'] and
                    load_pattern in benchmark_results['ppo_load_only']
                ):
                    print(
                        f"[Ablation] CALO vs Load-Only: "
                        f"{_format_case_comparison(
                            benchmark_results['ppo'][load_pattern]['mean_reward'],
                            benchmark_results['ppo_load_only'][load_pattern]['mean_reward'],
                        )}",
                        flush=True,
                    )

                completed_cases += 1
                progress.update({
                    'completed_cases': completed_cases,
                    'current_algorithm': None,
                    'current_stage': 'completed_case',
                    'last_completed_case': case_label,
                })
                _save_partial_results(output_dir, results)
                _save_progress(output_dir, progress)
                print(
                    f"[Progress] Completed {completed_cases}/{total_cases}: {case_label}",
                    flush=True,
                )

        results_path = output_dir / 'dynamic_experiment_results.json'
        _write_json(results_path, results)
        progress.update({
            'status': 'completed',
            'completed_cases': total_cases,
            'current_stage': 'completed',
            'results_path': str(results_path),
        })
        _save_progress(output_dir, progress)

        print(f"\nResults saved to: {results_path}", flush=True)

        feature_attribution_summary = _summarize_feature_attribution(results)
        if feature_attribution_summary is not None:
            attribution_path = output_dir / 'feature_attribution_summary.json'
            _write_json(attribution_path, feature_attribution_summary)
            print(
                f"Feature attribution summary saved to: {attribution_path}",
                flush=True,
            )

        generate_report(results, output_dir)
    except Exception as exc:
        progress.update({
            'status': 'failed',
            'current_stage': 'failed',
            'error': str(exc),
        })
        _save_progress(output_dir, progress)
        raise


def run_simulator_smoke_test(
    config_path: str,
    steps_per_case: int = 5,
    seed: int | None = None,
    override_calibration_dir: str | None = None,
    disable_calibration: bool = False,
):
    """Run a fast simulator smoke test without CALO training."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    environment_cfg = config['environment']
    _resolve_environment_calibration(
        environment_cfg=environment_cfg,
        override_calibration_dir=override_calibration_dir,
        disable_calibration=disable_calibration,
    )

    if seed is None:
        seeds = config.get('reproducibility', {}).get('seeds', [])
        seed = int(seeds[0]) if seeds else 42

    print("=" * 80)
    print("Simulator Smoke Test")
    print("=" * 80)
    print(f"Config: {config['description']}")
    print(f"Steps per case: {steps_per_case}")
    print(f"Seed: {seed}")
    print(
        "Calibration: "
        f"{_format_config_calibration_label(environment_cfg)}"
    )
    print("=" * 80, flush=True)

    for benchmark in config['benchmarks']:
        for load_pattern in config['environment']['load_patterns']:
            base_env = ServerlessFunctionEnv(
                benchmark=benchmark,
                deployment='local',
                enable_real_execution=False,
                normalize_reward=False,
                reward_weights=environment_cfg.get('reward_weights'),
                reward_penalties=environment_cfg.get('reward_penalties'),
                success_bonus=environment_cfg.get('success_bonus', 0.5),
                cost_normalization=environment_cfg.get('cost_normalization', 'ratio'),
                calibration_dir=environment_cfg.get('calibration_dir'),
                calibration_dirs=environment_cfg.get('calibration_dirs'),
                step_duration_sec=environment_cfg.get('step_duration_sec', 15.0),
                max_containers=environment_cfg.get('max_containers', 32),
                container_ttl_sec=environment_cfg.get('container_ttl_sec', 600.0),
                state_space=_SmokeTestStateSpace(
                    code_feature_dim=environment_cfg.get(
                        'code_feature_dim',
                        StateSpace.DEFAULT_CODE_FEATURE_DIM,
                    )
                ),
                action_space_config=environment_cfg.get('action_space'),
            )
            candidate_actions = [0]
            if base_env.action_space_wrapper.n_actions > 1:
                candidate_actions.append(base_env.action_space_wrapper.n_actions - 1)
            candidate_actions.append(base_env.action_space_wrapper.n_actions // 2)
            candidate_actions.append(
                max(0, min(base_env.action_space_wrapper.n_actions - 1, 2))
            )
            action_sequence = list(dict.fromkeys(candidate_actions))

            dynamic_env = DynamicLoadEnvironment(
                base_env,
                episode_length=max(
                    steps_per_case,
                    environment_cfg.get('episode_length', steps_per_case),
                ),
                load_change_freq=environment_cfg['load_change_freq'],
                switch_penalty=environment_cfg['switch_penalty'],
                adaptation_penalty=environment_cfg.get('adaptation_penalty', 0.05),
                workload_source=environment_cfg.get('workload_source', 'synthetic'),
                azure_profile_path=environment_cfg.get('azure_profile_path'),
                azure_summary_path=environment_cfg.get('azure_summary_path'),
                azure_top_k=environment_cfg.get('azure_top_k', 50),
                azure_arrival_scale=environment_cfg.get('azure_arrival_scale', 1.0),
                azure_max_arrivals_per_step=environment_cfg.get('azure_max_arrivals_per_step'),
                azure_profile_selection=environment_cfg.get('azure_profile_selection', 'benchmark_aware'),
                azure_selection_pool_size=environment_cfg.get('azure_selection_pool_size', 16),
                azure_target_concurrency=environment_cfg.get('azure_target_concurrency'),
                calibration_dirs=environment_cfg.get('calibration_dirs'),
            )

            options = {}
            if environment_cfg.get('workload_source') == 'azure_trace':
                options['workload_source'] = 'azure_trace'
            if load_pattern:
                options['load_pattern'] = load_pattern
            _, reset_info = dynamic_env.reset(seed=seed, options=options)

            total_reward = 0.0
            total_arrivals = 0
            max_peak_concurrency = 0
            max_queue_ratio = 0.0
            max_cold_start_rate = 0.0

            for step in range(steps_per_case):
                action = action_sequence[step % len(action_sequence)]
                _, reward, done, truncated, info = dynamic_env.step(action)
                metrics = info['metrics']
                total_reward += float(reward)
                total_arrivals += int(metrics['arrival_count'])
                max_peak_concurrency = max(
                    max_peak_concurrency,
                    int(metrics['peak_concurrency']),
                )
                max_queue_ratio = max(
                    max_queue_ratio,
                    float(metrics['queue_ratio']),
                )
                max_cold_start_rate = max(
                    max_cold_start_rate,
                    float(metrics['cold_start_rate']),
                )
                if done or truncated:
                    break

            workload_label = _format_smoke_workload_label(reset_info)
            calibration_label = _format_smoke_calibration_label(base_env)
            ttl_probe_sec = base_env.service_model.estimate_ttl_sec(
                benchmark=benchmark,
                memory_mb=512,
                timeout_sec=300,
                architecture='x64',
            )
            print(
                f"[Smoke] {benchmark} / {load_pattern}: "
                f"reward_sum={total_reward:.3f}, "
                f"arrivals={total_arrivals}, "
                f"peak_concurrency={max_peak_concurrency}, "
                f"max_queue_ratio={max_queue_ratio:.3f}, "
                f"max_cold_start_rate={max_cold_start_rate:.3f}, "
                f"workload={workload_label}, "
                f"calibration={calibration_label}, "
                f"ttl_512_x64_300={ttl_probe_sec:.0f}s",
                flush=True,
            )


def generate_report(results: dict, output_dir: Path):
    """Generate a compact experiment summary."""
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80, flush=True)

    algorithms = list(results.keys())
    if not algorithms:
        print("No results available.", flush=True)
        return

    primary_algorithm = 'ppo' if 'ppo' in results else algorithms[0]
    baseline_algorithms = [
        algorithm for algorithm in algorithms
        if algorithm != primary_algorithm
    ]

    for baseline_algorithm in baseline_algorithms:
        comparisons = _collect_case_comparisons(
            results=results,
            primary_algorithm=primary_algorithm,
            baseline_algorithm=baseline_algorithm,
        )
        summary = _summarize_case_comparisons(comparisons)

        primary_name = _algorithm_display_name(primary_algorithm)
        baseline_name = _algorithm_display_name(baseline_algorithm)
        print(
            f"\n{primary_name} win rate vs {baseline_name}: "
            f"{summary['primary_wins']}/{summary['total_comparisons']} "
            f"({summary['win_rate_pct']:.1f}%)",
            flush=True,
        )
        print(
            f"Mean raw-reward delta vs {baseline_name}: "
            f"{summary['mean_raw_reward_delta']:+.4f}",
            flush=True,
        )
        print(
            f"Median raw-reward improvement vs {baseline_name}: "
            f"{summary['median_relative_improvement_pct']:+.2f}%",
            flush=True,
        )
        print(
            f"Best raw-reward delta vs {baseline_name}: "
            f"{summary['best_raw_reward_delta']:+.4f}",
            flush=True,
        )
        print(
            f"Worst raw-reward delta vs {baseline_name}: "
            f"{summary['worst_raw_reward_delta']:+.4f}",
            flush=True,
        )
        if summary['small_baseline_cases']:
            print(
                "Note: percentage improvements can be unstable when the baseline "
                "raw reward is close to zero; use the raw delta as the primary metric.",
                flush=True,
            )

    plot_results(results, output_dir)


def plot_results(results: dict, output_dir: Path):
    """Render summary plots for the dynamic experiment results."""
    print("\nGenerating plots...", flush=True)

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    algorithms = list(results.keys())
    if not algorithms:
        return
    primary_algorithm = 'ppo' if 'ppo' in results else algorithms[0]
    comparison_algorithms = [
        algorithm for algorithm in algorithms
        if algorithm != primary_algorithm
    ]
    benchmarks = list(results[algorithms[0]].keys())
    load_patterns = list(results[algorithms[0]][benchmarks[0]].keys())
    cases = [(benchmark, pattern) for benchmark in benchmarks for pattern in load_patterns]
    x = np.arange(len(cases))
    width = 0.8 / max(1, len(algorithms))

    ax = axes[0, 0]
    for algorithm_index, algorithm in enumerate(algorithms):
        metric_values = [
            results[algorithm][benchmark][pattern]['mean_reward']
            for benchmark, pattern in cases
        ]
        offsets = x - 0.4 + width / 2 + algorithm_index * width
        ax.bar(
            offsets,
            metric_values,
            width,
            label=_algorithm_display_name(algorithm),
            alpha=0.8,
        )
    ax.set_xlabel('Case')
    ax.set_ylabel('Raw Reward')
    ax.set_title('Raw Reward by Case')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for algorithm_index, algorithm in enumerate(algorithms):
        metric_values = [
            results[algorithm][benchmark][pattern]['mean_latency']
            for benchmark, pattern in cases
        ]
        offsets = x - 0.4 + width / 2 + algorithm_index * width
        ax.bar(
            offsets,
            metric_values,
            width,
            label=_algorithm_display_name(algorithm),
            alpha=0.8,
        )
    ax.set_xlabel('Case')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency by Case')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for algorithm_index, algorithm in enumerate(algorithms):
        metric_values = [
            results[algorithm][benchmark][pattern]['mean_cost'] * 1e6
            for benchmark, pattern in cases
        ]
        offsets = x - 0.4 + width / 2 + algorithm_index * width
        ax.bar(
            offsets,
            metric_values,
            width,
            label=_algorithm_display_name(algorithm),
            alpha=0.8,
        )
    ax.set_xlabel('Case')
    ax.set_ylabel('Cost (micro USD)')
    ax.set_title('Cost by Case')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if comparison_algorithms:
        improvement_width = 0.8 / max(1, len(comparison_algorithms))
        primary_rewards = [
            results[primary_algorithm][benchmark][pattern]['mean_reward']
            for benchmark, pattern in cases
        ]
        for algorithm_index, algorithm in enumerate(comparison_algorithms):
            baseline_rewards = [
                results[algorithm][benchmark][pattern]['mean_reward']
                for benchmark, pattern in cases
            ]
            deltas = [
                float(primary_rewards[i]) - float(baseline_rewards[i])
                for i in range(len(primary_rewards))
            ]
            offsets = x - 0.4 + improvement_width / 2 + algorithm_index * improvement_width
            ax.bar(
                offsets,
                deltas,
                improvement_width,
                label=(
                    f"{_algorithm_display_name(primary_algorithm)} vs "
                    f"{_algorithm_display_name(algorithm)}"
                ),
                alpha=0.8,
            )
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No baseline available', ha='center', va='center')
    ax.set_xlabel('Case')
    ax.set_ylabel('Raw Reward Delta')
    ax.set_title(
        f"{_algorithm_display_name(primary_algorithm)} Raw-Reward Delta"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'dynamic_experiment_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plots saved to: {fig_path}", flush=True)


def _resolve_result_path(result_ref: str) -> Path:
    """Resolve one aggregate input to a concrete result JSON path."""
    path = Path(result_ref)
    if path.is_dir():
        path = path / 'dynamic_experiment_results.json'
    return path


def _result_label(result_path: Path) -> str:
    """Return one short label for an aggregated result file."""
    if result_path.name == 'dynamic_experiment_results.json':
        return result_path.parent.name
    return result_path.stem


def aggregate_experiment_results(result_refs: list[str], output_dir: str | None = None):
    """Aggregate multiple result JSON files into one final ablation table."""
    result_paths = [_resolve_result_path(result_ref) for result_ref in result_refs]
    missing_paths = [str(path) for path in result_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing result files for aggregation: "
            f"{missing_paths}"
        )

    loaded_results = [
        {
            'label': _result_label(result_path),
            'path': result_path,
            'results': json.loads(result_path.read_text()),
        }
        for result_path in result_paths
    ]

    if output_dir is None:
        output_path = Path('results/dynamic_experiment_aggregate')
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    template_results = loaded_results[0]['results']
    algorithms = list(template_results.keys())
    if not algorithms:
        raise ValueError('No algorithms available in the aggregate inputs.')

    primary_algorithm = 'ppo' if 'ppo' in template_results else algorithms[0]
    baseline_algorithms = [
        algorithm for algorithm in algorithms
        if algorithm != primary_algorithm
    ]
    aggregate_payload = {
        'primary_algorithm': primary_algorithm,
        'primary_display_name': _algorithm_display_name(primary_algorithm),
        'sources': [
            {
                'label': item['label'],
                'path': str(item['path']),
            }
            for item in loaded_results
        ],
    }
    markdown_lines = [
        '# Dynamic Experiment Aggregate Summary',
        '',
        f"Primary algorithm: {_algorithm_display_name(primary_algorithm)}",
        '',
        'Sources:',
    ]
    markdown_lines.extend(
        f"- {item['label']}: `{item['path']}`"
        for item in loaded_results
    )

    print("\n" + "=" * 80)
    print("Aggregate Summary")
    print("=" * 80, flush=True)

    if not baseline_algorithms:
        case_rows = []
        all_rewards = []
        for benchmark, primary_data in template_results[primary_algorithm].items():
            for load_pattern in primary_data.keys():
                seed_rows = []
                for item in loaded_results:
                    run_results = item['results']
                    primary_metrics = run_results[primary_algorithm][benchmark][load_pattern]
                    primary_reward = float(primary_metrics['mean_reward'])
                    seed_rows.append({
                        'source': item['label'],
                        'benchmark': benchmark,
                        'load_pattern': load_pattern,
                        'reward': primary_reward,
                    })
                    all_rewards.append(primary_reward)

                rewards = np.array([row['reward'] for row in seed_rows], dtype=float)
                case_rows.append({
                    'benchmark': benchmark,
                    'load_pattern': load_pattern,
                    'seed_count': len(seed_rows),
                    'reward_mean': float(np.mean(rewards)),
                    'reward_std': float(np.std(rewards, ddof=0)),
                    'reward_min': float(np.min(rewards)),
                    'reward_max': float(np.max(rewards)),
                })

        reward_array = np.array(all_rewards, dtype=float)
        overall_summary = {
            'total_comparisons': int(len(case_rows)),
            'seed_count': int(len(loaded_results)),
            'mean_reward': float(np.mean(reward_array)),
            'std_reward': float(np.std(reward_array, ddof=0)),
            'min_reward': float(np.min(reward_array)),
            'max_reward': float(np.max(reward_array)),
        }
        aggregate_payload['single_algorithm'] = {
            'overall': overall_summary,
            'cases': case_rows,
        }

        print(
            f"\n{_algorithm_display_name(primary_algorithm)} multi-seed summary",
            flush=True,
        )
        print(
            f"  Cases: {overall_summary['total_comparisons']} "
            f"across {overall_summary['seed_count']} seeds",
            flush=True,
        )
        print(
            f"  Mean raw reward: {overall_summary['mean_reward']:+.4f}",
            flush=True,
        )
        print(
            f"  Reward std: {overall_summary['std_reward']:.4f}",
            flush=True,
        )
        print(
            f"  Reward range: [{overall_summary['min_reward']:+.4f}, "
            f"{overall_summary['max_reward']:+.4f}]",
            flush=True,
        )

        markdown_lines.extend([
            '',
            f"## {_algorithm_display_name(primary_algorithm)} multi-seed summary",
            '',
            '| Metric | Value |',
            '| --- | --- |',
            f"| Cases | {overall_summary['total_comparisons']} |",
            f"| Seeds | {overall_summary['seed_count']} |",
            f"| Mean raw reward | {overall_summary['mean_reward']:+.4f} |",
            f"| Reward std | {overall_summary['std_reward']:.4f} |",
            (
                f"| Reward range | "
                f"[{overall_summary['min_reward']:+.4f}, "
                f"{overall_summary['max_reward']:+.4f}] |"
            ),
            '',
            '| Benchmark | Load | Seeds | Mean reward | Std | Min | Max |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
        ])
        for case_row in case_rows:
            markdown_lines.append(
                f"| {case_row['benchmark']} | {case_row['load_pattern']} | "
                f"{case_row['seed_count']} | "
                f"{case_row['reward_mean']:+.4f} | "
                f"{case_row['reward_std']:.4f} | "
                f"{case_row['reward_min']:+.4f} | "
                f"{case_row['reward_max']:+.4f} |"
            )

        json_path = output_path / 'aggregate_summary.json'
        markdown_path = output_path / 'aggregate_summary.md'
        _write_json(json_path, aggregate_payload)
        markdown_path.write_text('\n'.join(markdown_lines) + '\n')

        print(f"\nAggregate summary saved to: {json_path}", flush=True)
        print(f"Aggregate table saved to: {markdown_path}", flush=True)
        return

    aggregate_payload['baselines'] = {}

    for baseline_algorithm in baseline_algorithms:
        baseline_name = _algorithm_display_name(baseline_algorithm)
        case_rows = []
        all_comparisons = []

        for benchmark, primary_data in template_results[primary_algorithm].items():
            for load_pattern in primary_data.keys():
                seed_rows = []
                for item in loaded_results:
                    run_results = item['results']
                    primary_metrics = run_results[primary_algorithm][benchmark][load_pattern]
                    baseline_metrics = run_results[baseline_algorithm][benchmark][load_pattern]
                    primary_reward = float(primary_metrics['mean_reward'])
                    baseline_reward = float(baseline_metrics['mean_reward'])
                    comparison = {
                        'source': item['label'],
                        'benchmark': benchmark,
                        'load_pattern': load_pattern,
                        'primary_reward': primary_reward,
                        'baseline_reward': baseline_reward,
                        'raw_reward_delta': primary_reward - baseline_reward,
                        'relative_improvement_pct': _relative_improvement(
                            primary_reward,
                            baseline_reward,
                        ),
                        'small_baseline': _is_small_baseline(baseline_reward),
                        'primary_win': primary_reward > baseline_reward,
                    }
                    seed_rows.append(comparison)
                    all_comparisons.append(comparison)

                primary_rewards = np.array(
                    [row['primary_reward'] for row in seed_rows],
                    dtype=float,
                )
                baseline_rewards = np.array(
                    [row['baseline_reward'] for row in seed_rows],
                    dtype=float,
                )
                raw_deltas = np.array(
                    [row['raw_reward_delta'] for row in seed_rows],
                    dtype=float,
                )
                finite_relative_improvements = np.array(
                    [
                        row['relative_improvement_pct']
                        for row in seed_rows
                        if np.isfinite(row['relative_improvement_pct'])
                    ],
                    dtype=float,
                )
                case_rows.append({
                    'benchmark': benchmark,
                    'load_pattern': load_pattern,
                    'seed_count': len(seed_rows),
                    'primary_reward_mean': float(np.mean(primary_rewards)),
                    'primary_reward_std': float(np.std(primary_rewards, ddof=0)),
                    'baseline_reward_mean': float(np.mean(baseline_rewards)),
                    'baseline_reward_std': float(np.std(baseline_rewards, ddof=0)),
                    'mean_raw_reward_delta': float(np.mean(raw_deltas)),
                    'median_relative_improvement_pct': float(
                        np.median(finite_relative_improvements)
                    ) if finite_relative_improvements.size else 0.0,
                    'wins': sum(row['primary_win'] for row in seed_rows),
                    'small_baseline_cases': sum(
                        row['small_baseline'] for row in seed_rows
                    ),
                })

        overall_summary = _summarize_case_comparisons(all_comparisons)
        aggregate_payload['baselines'][baseline_algorithm] = {
            'baseline_display_name': baseline_name,
            'overall': overall_summary,
            'cases': case_rows,
        }

        print(
            f"\n{_algorithm_display_name(primary_algorithm)} vs {baseline_name}",
            flush=True,
        )
        print(
            f"  Win rate: {overall_summary['primary_wins']}/"
            f"{overall_summary['total_comparisons']} "
            f"({overall_summary['win_rate_pct']:.1f}%)",
            flush=True,
        )
        print(
            f"  Mean raw-reward delta: "
            f"{overall_summary['mean_raw_reward_delta']:+.4f}",
            flush=True,
        )
        print(
            f"  Median raw-reward improvement: "
            f"{overall_summary['median_relative_improvement_pct']:+.2f}%",
            flush=True,
        )
        if overall_summary['small_baseline_cases']:
            print(
                "  Note: at least one case has a near-zero baseline reward, "
                "so percentage means are intentionally omitted.",
                flush=True,
            )

        markdown_lines.extend([
            '',
            f"## {_algorithm_display_name(primary_algorithm)} vs {baseline_name}",
            '',
            '| Metric | Value |',
            '| --- | --- |',
            (
                f"| Win rate | {overall_summary['primary_wins']}/"
                f"{overall_summary['total_comparisons']} "
                f"({overall_summary['win_rate_pct']:.1f}%) |"
            ),
            (
                f"| Mean raw-reward delta | "
                f"{overall_summary['mean_raw_reward_delta']:+.4f} |"
            ),
            (
                f"| Median raw-reward improvement | "
                f"{overall_summary['median_relative_improvement_pct']:+.2f}% |"
            ),
            '',
            '| Benchmark | Load | Seeds | '
            f"{_algorithm_display_name(primary_algorithm)} reward | "
            f"{baseline_name} reward | Mean raw delta | Median pct | Wins |",
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for case_row in case_rows:
            markdown_lines.append(
                f"| {case_row['benchmark']} | {case_row['load_pattern']} | "
                f"{case_row['seed_count']} | "
                f"{case_row['primary_reward_mean']:+.4f} ± "
                f"{case_row['primary_reward_std']:.4f} | "
                f"{case_row['baseline_reward_mean']:+.4f} ± "
                f"{case_row['baseline_reward_std']:.4f} | "
                f"{case_row['mean_raw_reward_delta']:+.4f} | "
                f"{case_row['median_relative_improvement_pct']:+.2f}% | "
                f"{case_row['wins']}/{case_row['seed_count']} |"
            )

    json_path = output_path / 'aggregate_summary.json'
    markdown_path = output_path / 'aggregate_summary.md'
    _write_json(json_path, aggregate_payload)
    markdown_path.write_text('\n'.join(markdown_lines) + '\n')

    print(f"\nAggregate summary saved to: {json_path}", flush=True)
    print(f"Aggregate table saved to: {markdown_path}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the dynamic-load CALO versus baseline experiment'
    )
    parser.add_argument('--config', type=str,
                       default='config/rl_experiments/tuning_uploader_recognition.json',
                       help='Path to the experiment config file')
    parser.add_argument('--output-dir', type=str,
                       default='results_dynamic',
                       help='Directory used for result artifacts')
    parser.add_argument('--smoke-test', action='store_true',
                       help='Run only a fast simulator smoke test without CALO training')
    parser.add_argument('--smoke-steps', type=int, default=5,
                       help='Number of control steps per case in smoke-test mode')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override the seed defined in the config')
    parser.add_argument('--override-calibration-dir', type=str, default=None,
                       help='Override calibration_dir from the config')
    parser.add_argument('--disable-calibration', action='store_true',
                       help='Ignore calibration_dir and force the heuristic simulator')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume an interrupted run from progress and partial result files',
    )
    parser.add_argument(
        '--aggregate-results',
        nargs='+',
        default=None,
        help=(
            'Aggregate one or more completed result directories or '
            'dynamic_experiment_results.json files'
        ),
    )
    parser.add_argument(
        '--aggregate-output-dir',
        type=str,
        default=None,
        help='Directory used for aggregated summary artifacts',
    )

    args = parser.parse_args()

    if args.aggregate_results:
        aggregate_experiment_results(
            result_refs=args.aggregate_results,
            output_dir=args.aggregate_output_dir,
        )
    elif args.smoke_test:
        run_simulator_smoke_test(
            config_path=args.config,
            steps_per_case=args.smoke_steps,
            seed=args.seed,
            override_calibration_dir=args.override_calibration_dir,
            disable_calibration=args.disable_calibration,
        )
    else:
        run_dynamic_experiment(
            args.config,
            args.output_dir,
            seed=args.seed,
            override_calibration_dir=args.override_calibration_dir,
            disable_calibration=args.disable_calibration,
            resume=args.resume,
        )
