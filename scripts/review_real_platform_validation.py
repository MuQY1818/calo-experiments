#!/usr/bin/env python3
"""Run a small simulator-vs-OpenWhisk validation set for reviewer response."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_optimizer.action_space import ActionSpace
from rl_optimizer.environment import ServerlessFunctionEnv


@dataclass(frozen=True)
class ValidationTarget:
    """One benchmark/configuration pair for real-platform validation."""

    benchmark: str
    memory_mb: int
    timeout_sec: int
    architecture: str

    @property
    def label(self) -> str:
        """Return a compact label for filenames and logs."""
        return (
            f"{self.benchmark.replace('.', '_')}"
            f"_{self.memory_mb}mb_{self.timeout_sec}s_{self.architecture}"
        )

    @property
    def short_name(self) -> str:
        """Return a compact benchmark identifier for function naming."""
        pieces = self.benchmark.split(".", maxsplit=1)
        if len(pieces) == 1:
            return self.benchmark.replace(".", "-")
        return f"{pieces[0]}-{pieces[1].replace('.', '-')}"


@dataclass(frozen=True)
class InvocationRecord:
    """One real-platform invoke result."""

    function_name: str
    group_index: int
    attempt_index: int
    request_id: str
    success: bool
    cold_start: bool
    client_ms: float
    benchmark_ms: float
    selected_role: str
    output_dir: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a small OpenWhisk validation set and compare against the simulator.",
    )
    parser.add_argument(
        "--openwhisk-config",
        type=Path,
        default=Path("config/openwhisk_standalone.example.json"),
        help="Path to the OpenWhisk standalone SeBS config.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="results/openwhisk_calibration_x64_stage2_prime",
        help="Calibration directory used by the simulator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/review_real_platform_validation"),
        help="Directory for raw OpenWhisk runs and summary tables.",
    )
    parser.add_argument(
        "--cold-runs",
        type=int,
        default=1,
        help="Number of cold runs per target. Each cold run uses a fresh function name.",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=4,
        help="Number of warm samples to collect after each successful cold sample.",
    )
    parser.add_argument(
        "--sim-samples",
        type=int,
        default=2048,
        help="Number of simulator samples used to estimate warm/cold latency moments.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw SeBS output directories instead of replacing them per run.",
    )
    parser.add_argument(
        "--warm-gap-sec",
        type=float,
        default=2.0,
        help="Sleep interval between consecutive single-invoke attempts.",
    )
    parser.add_argument(
        "--max-attempts-per-cold-run",
        type=int,
        default=8,
        help="Maximum single-invoke attempts used to collect one cold group.",
    )
    parser.add_argument(
        "--function-prefix",
        type=str,
        default="review-val",
        help="Prefix used for generated OpenWhisk function names.",
    )
    parser.add_argument(
        "--benchmark",
        dest="benchmarks",
        action="append",
        default=[],
        help="Optional benchmark override. May be provided multiple times.",
    )
    parser.add_argument(
        "--target-preset",
        type=str,
        default="review_default",
        help=(
            "Target configuration preset. "
            "`review_default` keeps the original two-config reviewer probe, "
            "while ActionSpace presets such as `calibrated_x64_8` expand to "
            "the corresponding Cartesian product."
        ),
    )
    parser.add_argument(
        "--memory-option",
        dest="memory_options",
        action="append",
        type=int,
        default=[],
        help="Optional memory override. May be provided multiple times.",
    )
    parser.add_argument(
        "--timeout-option",
        dest="timeout_options",
        action="append",
        type=int,
        default=[],
        help="Optional timeout override. May be provided multiple times.",
    )
    parser.add_argument(
        "--architecture-option",
        dest="architecture_options",
        action="append",
        default=[],
        help="Optional architecture override. May be provided multiple times.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume from an existing validation_summary.json in the output directory "
            "and skip targets that already have completed rows."
        ),
    )
    parser.add_argument(
        "--latency-weight",
        type=float,
        default=0.7,
        help="Latency weight used for reward-style ranking summaries.",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.3,
        help="Cost weight used for reward-style ranking summaries.",
    )
    parser.add_argument(
        "--success-bonus",
        type=float,
        default=0.7,
        help="Success bonus used for reward-style ranking summaries.",
    )
    parser.add_argument(
        "--failure-penalty",
        type=float,
        default=3.0,
        help="Failure penalty used for reward-style ranking summaries.",
    )
    parser.add_argument(
        "--cold-penalty",
        type=float,
        default=0.2,
        help="Cold-start penalty used for reward-style ranking summaries.",
    )
    parser.add_argument(
        "--queue-penalty",
        type=float,
        default=0.1,
        help="Queue penalty used for reward-style ranking summaries.",
    )
    parser.add_argument(
        "--cost-normalization",
        type=str,
        choices=["ratio", "log", "sqrt"],
        default="log",
        help="Cost normalization used for reward-style ranking summaries.",
    )
    return parser.parse_args()


def default_targets(
    benchmarks: Iterable[str] | None = None,
    target_preset: str = "review_default",
    memory_options: Iterable[int] | None = None,
    timeout_options: Iterable[int] | None = None,
    architecture_options: Iterable[str] | None = None,
) -> list[ValidationTarget]:
    """Return the requested validation target set."""
    benchmark_list = list(benchmarks) if benchmarks else [
        "110.dynamic-html",
        "120.uploader",
        "411.image-recognition",
    ]

    if (
        target_preset == "review_default"
        and not memory_options
        and not timeout_options
        and not architecture_options
    ):
        targets: list[ValidationTarget] = []
        for benchmark in benchmark_list:
            targets.append(
                ValidationTarget(
                    benchmark=benchmark,
                    memory_mb=512,
                    timeout_sec=120,
                    architecture="x64",
                )
            )
            targets.append(
                ValidationTarget(
                    benchmark=benchmark,
                    memory_mb=1024,
                    timeout_sec=300,
                    architecture="x64",
                )
            )
        return targets

    if target_preset == "review_default":
        preset_memory = [512, 1024]
        preset_timeout = [120, 300]
        preset_architecture = ["x64"]
    else:
        preset = ActionSpace.PRESETS[target_preset]
        preset_memory = list(preset["memory_options"])
        preset_timeout = list(preset["timeout_options"])
        preset_architecture = list(preset["architecture_options"])

    resolved_memory = sorted({int(value) for value in (memory_options or preset_memory)})
    resolved_timeout = sorted({int(value) for value in (timeout_options or preset_timeout)})
    resolved_architecture = []
    seen_architecture = set()
    for value in (architecture_options or preset_architecture):
        current = str(value)
        if current in seen_architecture:
            continue
        resolved_architecture.append(current)
        seen_architecture.add(current)

    targets: list[ValidationTarget] = []
    for benchmark in benchmark_list:
        for memory_mb in resolved_memory:
            for timeout_sec in resolved_timeout:
                for architecture in resolved_architecture:
                    targets.append(
                        ValidationTarget(
                            benchmark=benchmark,
                            memory_mb=memory_mb,
                            timeout_sec=timeout_sec,
                            architecture=architecture,
                        )
                    )
    return targets


def estimate_request_cost_usd(
    memory_mb: int,
    architecture: str,
    latency_ms: float,
) -> float:
    """Estimate per-request cost with the same pricing rule as the simulator."""
    if not np.isfinite(latency_ms):
        return math.nan
    gb = float(memory_mb) / 1024.0
    seconds = float(latency_ms) / 1000.0
    if architecture == "arm64":
        price_per_gb_second = 0.0000133334
    else:
        price_per_gb_second = 0.0000166667
    return float(gb * seconds * price_per_gb_second + 0.0000002)


def compute_static_reward(
    latency_ms: float,
    cost_usd: float,
    success_rate: float,
    cold_start_rate: float,
    latency_weight: float,
    cost_weight: float,
    success_bonus: float,
    failure_penalty: float,
    cold_penalty: float,
    queue_penalty: float,
    cost_normalization: str,
    baseline_latency_ms: float = 1000.0,
    baseline_cost_usd: float = 0.001,
    queue_ratio: float = 0.0,
) -> float:
    """Compute a reward-style utility score for one static configuration."""
    if not np.isfinite(latency_ms) or not np.isfinite(cost_usd):
        return math.nan

    normalized_latency = float(latency_ms) / baseline_latency_ms
    if cost_normalization == "log":
        normalized_cost = math.log1p(float(cost_usd) / baseline_cost_usd)
    elif cost_normalization == "sqrt":
        normalized_cost = math.sqrt(float(cost_usd) / baseline_cost_usd)
    else:
        normalized_cost = float(cost_usd) / baseline_cost_usd

    reward = -(
        latency_weight * normalized_latency
        + cost_weight * normalized_cost
    )
    failure_rate = 1.0 - float(success_rate)
    reward += success_bonus * float(success_rate)
    reward -= failure_penalty * failure_rate
    reward -= cold_penalty * float(cold_start_rate)
    reward -= queue_penalty * float(queue_ratio)
    return float(reward)


def parse_single_invocation(
    path: Path,
    function_name: str,
    group_index: int,
    attempt_index: int,
) -> InvocationRecord:
    """Parse a single-invoke SeBS result into a normalized record."""
    payload = json.loads(path.read_text())
    invocations = []
    for function_invocations in payload.get("_invocations", {}).values():
        invocations.extend(function_invocations.values())

    if len(invocations) != 1:
        raise RuntimeError(
            f"Expected exactly one invocation in {path}, found {len(invocations)}."
        )

    invocation = invocations[0]
    stats = invocation.get("stats", {})
    times = invocation.get("times", {})
    request_id = str(invocation.get("request_id", ""))

    return InvocationRecord(
        function_name=function_name,
        group_index=group_index,
        attempt_index=attempt_index,
        request_id=request_id,
        success=not bool(stats.get("failure", False)),
        cold_start=bool(stats.get("cold_start", False)),
        client_ms=float(times.get("client", math.nan)) / 1000.0,
        benchmark_ms=float(times.get("benchmark", math.nan)) / 1000.0,
        selected_role="other",
        output_dir=str(path.parent),
    )


def summarize_records(records: list[InvocationRecord]) -> dict[str, float]:
    """Aggregate a list of invocation records."""
    client_ms = [
        record.client_ms
        for record in records
        if record.success and np.isfinite(record.client_ms)
    ]
    benchmark_ms = [
        record.benchmark_ms
        for record in records
        if record.success and np.isfinite(record.benchmark_ms)
    ]
    total = len(records)
    failures = sum(1 for record in records if not record.success)
    cold_count = sum(1 for record in records if record.cold_start)
    warm_count = sum(1 for record in records if not record.cold_start)

    return {
        "runs": total,
        "failure_count": failures,
        "success_rate": 0.0 if total == 0 else (total - failures) / total,
        "reported_cold_count": cold_count,
        "reported_warm_count": warm_count,
        "mean_ms": float(np.mean(client_ms)) if client_ms else math.nan,
        "p95_ms": float(np.percentile(client_ms, 95)) if client_ms else math.nan,
        "benchmark_mean_ms": (
            float(np.mean(benchmark_ms)) if benchmark_ms else math.nan
        ),
        "benchmark_p95_ms": (
            float(np.percentile(benchmark_ms, 95)) if benchmark_ms else math.nan
        ),
    }


def estimate_simulator_summary(
    target: ValidationTarget,
    calibration_dir: str,
    sim_samples: int,
) -> dict[str, float]:
    """Estimate simulator warm/cold latency moments for one target."""
    env = ServerlessFunctionEnv(
        benchmark=target.benchmark,
        deployment="local",
        enable_real_execution=False,
        normalize_reward=False,
        calibration_dir=calibration_dir,
        action_space_config={
            "memory_options": [target.memory_mb],
            "architecture_options": [target.architecture],
            "timeout_options": [target.timeout_sec],
        },
    )
    config = env.action_space_wrapper.get_configuration(0)

    warm_ms = np.asarray(
        [
            env.service_model.sample_warm_runtime_ms(
                benchmark=target.benchmark,
                memory_mb=config.memory_mb,
                timeout_sec=config.timeout_sec,
                architecture=config.architecture,
                arrival_rate_per_sec=1.0 / env.step_duration_sec,
            )
            for _ in range(sim_samples)
        ],
        dtype=np.float32,
    )
    cold_overhead_ms = np.asarray(
        [
            env.service_model.sample_cold_overhead_ms(
                benchmark=target.benchmark,
                memory_mb=config.memory_mb,
                timeout_sec=config.timeout_sec,
                architecture=config.architecture,
                idle_gap_sec=max(
                    2.0 * env.service_model.estimate_ttl_sec(
                        benchmark=target.benchmark,
                        memory_mb=config.memory_mb,
                        timeout_sec=config.timeout_sec,
                        architecture=config.architecture,
                    ),
                    900.0,
                ),
            )
            for _ in range(sim_samples)
        ],
        dtype=np.float32,
    )
    cold_ms = warm_ms + cold_overhead_ms

    return {
        "sim_warm_mean_ms": float(np.mean(warm_ms)),
        "sim_warm_p95_ms": float(np.percentile(warm_ms, 95)),
        "sim_cold_mean_ms": float(np.mean(cold_ms)),
        "sim_cold_p95_ms": float(np.percentile(cold_ms, 95)),
        "sim_cold_overhead_ms": float(np.mean(cold_overhead_ms)),
    }


def run_single_invoke(
    target: ValidationTarget,
    openwhisk_config: Path,
    run_output_dir: Path,
    cache_dir: Path,
    function_name: str,
) -> Path:
    """Run one single OpenWhisk invocation and return the experiments.json path."""
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python-venv/bin/python",
        "sebs.py",
        "benchmark",
        "invoke",
        target.benchmark,
        "test",
        "--trigger",
        "library",
        "--repetitions",
        "1",
        "--memory",
        str(target.memory_mb),
        "--timeout",
        str(target.timeout_sec),
        "--function-name",
        function_name,
        "--config",
        str(openwhisk_config),
        "--deployment",
        "openwhisk",
        "--architecture",
        target.architecture,
        "--container-deployment",
        "--cache",
        str(cache_dir),
        "--no-update-code",
        "--no-update-storage",
        "--output-dir",
        str(run_output_dir),
        "--output-file",
        "out.log",
        "--no-preserve-out",
        "--verbose",
    ]

    env = os.environ.copy()
    env["SEBS_WITH_OPENWHISK"] = "true"
    result = subprocess.run(
        cmd,
        cwd=".",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"OpenWhisk validation failed for {target.label} ({function_name}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    experiments_path = run_output_dir / "experiments.json"
    if not experiments_path.exists():
        raise RuntimeError(
            f"OpenWhisk validation did not produce experiments.json for "
            f"{target.label} ({function_name}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return experiments_path


def build_function_name(
    target: ValidationTarget,
    function_prefix: str,
    cold_index: int,
) -> str:
    """Create a unique but stable-per-target function name."""
    suffix = uuid.uuid4().hex[:8]
    return (
        f"{function_prefix}-{target.short_name}-"
        f"{target.memory_mb}m-{target.timeout_sec}s-{cold_index}-{suffix}"
    )


def run_real_target(
    target: ValidationTarget,
    openwhisk_config: Path,
    output_dir: Path,
    cold_runs: int,
    warm_runs: int,
    keep_raw: bool,
    warm_gap_sec: float,
    max_attempts_per_cold_run: int,
    function_prefix: str,
) -> dict[str, object]:
    """Run one real OpenWhisk target and collect actual cold/warm samples."""
    target_dir = output_dir / "raw" / target.label
    cache_dir = output_dir / "cache" / target.label
    if target_dir.exists() and not keep_raw:
        shutil.rmtree(target_dir)
    if cache_dir.exists() and not keep_raw:
        shutil.rmtree(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    attempt_records: list[InvocationRecord] = []
    selected_cold_records: list[InvocationRecord] = []
    selected_warm_records: list[InvocationRecord] = []
    function_names: list[str] = []

    for group_index in range(1, cold_runs + 1):
        function_name = build_function_name(
            target=target,
            function_prefix=function_prefix,
            cold_index=group_index,
        )
        function_names.append(function_name)
        cold_collected = False
        warm_collected = 0

        for attempt_index in range(1, max_attempts_per_cold_run + 1):
            if attempt_index > 1 and warm_gap_sec > 0:
                time.sleep(warm_gap_sec)
            attempt_output_dir = (
                target_dir / f"group_{group_index:02d}" / f"attempt_{attempt_index:02d}"
            )
            attempt_path = run_single_invoke(
                target=target,
                openwhisk_config=openwhisk_config,
                run_output_dir=attempt_output_dir,
                cache_dir=cache_dir,
                function_name=function_name,
            )
            record = parse_single_invocation(
                attempt_path,
                function_name=function_name,
                group_index=group_index,
                attempt_index=attempt_index,
            )
            if record.success and record.cold_start and not cold_collected:
                record = InvocationRecord(
                    **{**record.__dict__, "selected_role": "cold_sample"}
                )
                selected_cold_records.append(record)
                cold_collected = True
            elif record.success and cold_collected and not record.cold_start:
                if warm_collected < warm_runs:
                    record = InvocationRecord(
                        **{**record.__dict__, "selected_role": "warm_sample"}
                    )
                    selected_warm_records.append(record)
                    warm_collected += 1
            attempt_records.append(record)
            if cold_collected and warm_collected >= warm_runs:
                break

    all_attempt_summary = summarize_records(attempt_records)
    cold_summary = summarize_records(selected_cold_records)
    warm_summary = summarize_records(selected_warm_records)
    return {
        "function_names": function_names,
        "records": [record.__dict__ for record in attempt_records],
        "summary": {
            "total_invocations": all_attempt_summary["runs"],
            "failure_count": all_attempt_summary["failure_count"],
            "success_rate": all_attempt_summary["success_rate"],
            "reported_cold_count": all_attempt_summary["reported_cold_count"],
            "reported_warm_count": all_attempt_summary["reported_warm_count"],
            "cold_target_runs": cold_runs,
            "cold_selected_count": cold_summary["runs"],
            "cold_capture_rate": (
                cold_summary["runs"] / cold_runs if cold_runs > 0 else 0.0
            ),
            "real_cold_mean_ms": cold_summary["mean_ms"],
            "real_cold_p95_ms": cold_summary["p95_ms"],
            "real_cold_benchmark_mean_ms": cold_summary["benchmark_mean_ms"],
            "warm_target_runs": cold_runs * warm_runs,
            "warm_selected_count": warm_summary["runs"],
            "warm_capture_rate": (
                warm_summary["runs"] / (cold_runs * warm_runs)
                if (cold_runs * warm_runs) > 0
                else 0.0
            ),
            "real_warm_mean_ms": warm_summary["mean_ms"],
            "real_warm_p95_ms": warm_summary["p95_ms"],
            "real_warm_benchmark_mean_ms": warm_summary["benchmark_mean_ms"],
            "real_cold_overhead_ms": (
                cold_summary["mean_ms"] - warm_summary["mean_ms"]
                if np.isfinite(cold_summary["mean_ms"]) and np.isfinite(warm_summary["mean_ms"])
                else math.nan
            ),
            "sample_collection_complete": bool(
                cold_summary["runs"] == cold_runs
                and warm_summary["runs"] == cold_runs * warm_runs
            ),
        },
    }


def relative_error(predicted: float, observed: float) -> float:
    """Return absolute relative error in percent."""
    if not np.isfinite(predicted) or not np.isfinite(observed):
        return math.nan
    if abs(observed) < 1e-12:
        return math.nan
    return abs(predicted - observed) / abs(observed) * 100.0


def write_outputs(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    """Write CSV, JSON, and LaTeX summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "validation_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    columns = [
        "benchmark",
        "memory_mb",
        "timeout_sec",
        "architecture",
        "cold_selected_count",
        "cold_capture_rate",
        "real_warm_mean_ms",
        "sim_warm_mean_ms",
        "warm_mean_abs_rel_error_pct",
        "warm_selected_count",
        "warm_capture_rate",
        "real_cold_mean_ms",
        "sim_cold_mean_ms",
        "cold_mean_abs_rel_error_pct",
        "success_rate",
    ]
    csv_lines = [",".join(columns)]
    for row in rows:
        csv_lines.append(
            ",".join(str(row.get(column, "")) for column in columns)
        )
    (output_dir / "validation_summary.csv").write_text("\n".join(csv_lines) + "\n")

    latex_lines = [
        "\\begin{tabular}{lrrccccc}",
        "\\toprule",
        (
            "Benchmark & Mem. & Timeout & Cold cap. & Warm cap. "
            "& Real warm & Sim warm & Real cold & Sim cold \\\\"
        ),
        "\\midrule",
    ]
    for row in rows:
        latex_lines.append(
            (
                f"{row['benchmark']} & {row['memory_mb']} & {row['timeout_sec']} "
                f"& {100.0 * float(row['cold_capture_rate']):.0f}\\% "
                f"& {100.0 * float(row['warm_capture_rate']):.0f}\\% "
                f"& {float(row['real_warm_mean_ms']):.1f} "
                f"& {float(row['sim_warm_mean_ms']):.1f} "
                f"& {float(row['real_cold_mean_ms']):.1f} "
                f"& {float(row['sim_cold_mean_ms']):.1f} \\\\"
            )
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "validation_table.tex").write_text("\n".join(latex_lines) + "\n")


def _target_key(
    benchmark: str,
    memory_mb: int,
    timeout_sec: int,
    architecture: str,
) -> tuple[str, int, int, str]:
    """Build a stable key for one validation target."""
    return (
        str(benchmark),
        int(memory_mb),
        int(timeout_sec),
        str(architecture),
    )


def load_existing_rows(output_dir: Path) -> list[dict[str, float | str]]:
    """Load previously completed validation rows from disk if available."""
    summary_path = output_dir / "validation_summary.json"
    if not summary_path.exists():
        return []

    payload = json.loads(summary_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a JSON list in {summary_path}, got {type(payload).__name__}."
        )
    return payload


def build_ranking_summary(
    rows: list[dict[str, float | str]],
) -> dict[str, object]:
    """Summarize configuration-ranking agreement between simulator and reality."""
    metric_specs = [
        {
            "metric": "warm_latency",
            "real_key": "real_warm_mean_ms",
            "sim_key": "sim_warm_mean_ms",
            "higher_is_better": False,
            "required_count_key": "warm_selected_count",
        },
        {
            "metric": "cold_latency",
            "real_key": "real_cold_mean_ms",
            "sim_key": "sim_cold_mean_ms",
            "higher_is_better": False,
            "required_count_key": "cold_selected_count",
        },
        {
            "metric": "warm_reward",
            "real_key": "real_warm_reward",
            "sim_key": "sim_warm_reward",
            "higher_is_better": True,
            "required_count_key": "warm_selected_count",
        },
        {
            "metric": "cold_reward",
            "real_key": "real_cold_reward",
            "sim_key": "sim_cold_reward",
            "higher_is_better": True,
            "required_count_key": "cold_selected_count",
        },
    ]

    grouped_rows: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["benchmark"])].append(row)

    benchmark_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []

    for metric_spec in metric_specs:
        metric_name = metric_spec["metric"]
        metric_rows: list[dict[str, object]] = []
        for benchmark, benchmark_group in sorted(grouped_rows.items()):
            valid_rows = []
            for row in benchmark_group:
                real_value = row.get(metric_spec["real_key"])
                sim_value = row.get(metric_spec["sim_key"])
                if not np.isfinite(real_value) or not np.isfinite(sim_value):
                    continue
                if int(row.get(metric_spec["required_count_key"], 0)) <= 0:
                    continue
                valid_rows.append(row)

            if len(valid_rows) < 2:
                continue

            aligned_rows = sorted(
                valid_rows,
                key=lambda item: (
                    int(item["memory_mb"]),
                    int(item["timeout_sec"]),
                    str(item["architecture"]),
                ),
            )
            real_values = np.asarray(
                [float(item[metric_spec["real_key"]]) for item in aligned_rows],
                dtype=np.float64,
            )
            sim_values = np.asarray(
                [float(item[metric_spec["sim_key"]]) for item in aligned_rows],
                dtype=np.float64,
            )
            rank_result = spearmanr(sim_values, real_values, nan_policy="omit")
            spearman_value = (
                float(rank_result.statistic)
                if np.isfinite(rank_result.statistic)
                else math.nan
            )

            real_sorted = sorted(
                valid_rows,
                key=lambda item: float(item[metric_spec["real_key"]]),
                reverse=bool(metric_spec["higher_is_better"]),
            )
            sim_sorted = sorted(
                valid_rows,
                key=lambda item: float(item[metric_spec["sim_key"]]),
                reverse=bool(metric_spec["higher_is_better"]),
            )
            top_k = min(2, len(valid_rows))
            real_topk = {
                str(item["config_label"])
                for item in real_sorted[:top_k]
            }
            sim_topk = {
                str(item["config_label"])
                for item in sim_sorted[:top_k]
            }

            metric_row = {
                "benchmark": benchmark,
                "metric": metric_name,
                "config_count": len(valid_rows),
                "complete_count": sum(
                    bool(item.get("sample_collection_complete", False))
                    for item in valid_rows
                ),
                "top1_match": (
                    str(real_sorted[0]["config_label"])
                    == str(sim_sorted[0]["config_label"])
                ),
                "top2_overlap": len(real_topk & sim_topk),
                "spearman": spearman_value,
                "real_best": str(real_sorted[0]["config_label"]),
                "sim_best": str(sim_sorted[0]["config_label"]),
                "real_best_value": float(real_sorted[0][metric_spec["real_key"]]),
                "sim_best_value": float(sim_sorted[0][metric_spec["sim_key"]]),
            }
            metric_rows.append(metric_row)
            benchmark_rows.append(metric_row)

        finite_spearman = [
            float(item["spearman"])
            for item in metric_rows
            if np.isfinite(item["spearman"])
        ]
        overall_rows.append(
            {
                "metric": metric_name,
                "benchmark_count": len(metric_rows),
                "top1_match_rate": (
                    float(np.mean([bool(item["top1_match"]) for item in metric_rows]))
                    if metric_rows
                    else math.nan
                ),
                "mean_top2_overlap": (
                    float(np.mean([float(item["top2_overlap"]) for item in metric_rows]))
                    if metric_rows
                    else math.nan
                ),
                "mean_spearman": (
                    float(np.mean(finite_spearman))
                    if finite_spearman
                    else math.nan
                ),
            }
        )

    return {
        "benchmark_rows": benchmark_rows,
        "overall": overall_rows,
    }


def write_ranking_outputs(
    ranking_summary: dict[str, object],
    output_dir: Path,
) -> None:
    """Write ranking-agreement summaries."""
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(ranking_summary, indent=2, ensure_ascii=False)
    )

    benchmark_rows = list(ranking_summary["benchmark_rows"])
    csv_columns = [
        "benchmark",
        "metric",
        "config_count",
        "complete_count",
        "top1_match",
        "top2_overlap",
        "spearman",
        "real_best",
        "sim_best",
        "real_best_value",
        "sim_best_value",
    ]
    csv_path = output_dir / "ranking_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in benchmark_rows:
            writer.writerow({column: row.get(column, "") for column in csv_columns})

    latex_lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Benchmark & Metric & N & Top-1 & Top-2 & Spearman & Real / Sim Best \\\\",
        "\\midrule",
    ]
    for row in benchmark_rows:
        top1_value = "Yes" if bool(row["top1_match"]) else "No"
        spearman_value = row["spearman"]
        spearman_text = (
            f"{float(spearman_value):.3f}"
            if np.isfinite(spearman_value)
            else "--"
        )
        latex_lines.append(
            (
                f"{row['benchmark']} & {row['metric']} & {int(row['config_count'])} "
                f"& {top1_value} & {int(row['top2_overlap'])} "
                f"& {spearman_text} "
                f"& {row['real_best']} / {row['sim_best']} \\\\"
            )
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "ranking_table.tex").write_text("\n".join(latex_lines) + "\n")


def main() -> None:
    """Run the reviewer validation set."""
    args = parse_args()
    targets = default_targets(
        benchmarks=args.benchmarks,
        target_preset=args.target_preset,
        memory_options=args.memory_options,
        timeout_options=args.timeout_options,
        architecture_options=args.architecture_options,
    )
    rows: list[dict[str, float | str]] = (
        load_existing_rows(args.output_dir) if args.resume else []
    )
    completed_keys = {
        _target_key(
            benchmark=str(row["benchmark"]),
            memory_mb=int(row["memory_mb"]),
            timeout_sec=int(row["timeout_sec"]),
            architecture=str(row["architecture"]),
        )
        for row in rows
    }

    print("=" * 80)
    print("Reviewer Real-Platform Validation")
    print("=" * 80)
    print(f"Targets: {len(targets)}")
    print(f"OpenWhisk config: {args.openwhisk_config}")
    print(f"Calibration: {args.calibration_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Cold runs / warm runs: {args.cold_runs} / {args.warm_runs}")
    if args.resume:
        print(f"Resume enabled: {len(rows)} completed targets loaded")
    print(
        "Target preset: "
        f"{args.target_preset} "
        f"(memory={args.memory_options or 'preset'}, "
        f"timeout={args.timeout_options or 'preset'}, "
        f"arch={args.architecture_options or 'preset'})"
    )
    print("=" * 80, flush=True)

    for index, target in enumerate(targets, start=1):
        target_key = _target_key(
            benchmark=target.benchmark,
            memory_mb=target.memory_mb,
            timeout_sec=target.timeout_sec,
            architecture=target.architecture,
        )
        if target_key in completed_keys:
            print(
                f"[{index}/{len(targets)}] {target.label} (resume skip)",
                flush=True,
            )
            continue
        print(f"[{index}/{len(targets)}] {target.label}", flush=True)
        real_result = run_real_target(
            target=target,
            openwhisk_config=args.openwhisk_config,
            output_dir=args.output_dir,
            cold_runs=args.cold_runs,
            warm_runs=args.warm_runs,
            keep_raw=args.keep_raw,
            warm_gap_sec=args.warm_gap_sec,
            max_attempts_per_cold_run=args.max_attempts_per_cold_run,
            function_prefix=args.function_prefix,
        )
        real_summary = real_result["summary"]
        target_records_path = args.output_dir / "raw" / target.label / "records.json"
        target_records_path.write_text(
            json.dumps(real_result["records"], indent=2, ensure_ascii=False)
        )
        sim_summary = estimate_simulator_summary(
            target=target,
            calibration_dir=args.calibration_dir,
            sim_samples=args.sim_samples,
        )
        row: dict[str, float | str] = {
            "benchmark": target.benchmark,
            "memory_mb": target.memory_mb,
            "timeout_sec": target.timeout_sec,
            "architecture": target.architecture,
            "config_label": (
                f"{target.memory_mb}MB/{target.timeout_sec}s/{target.architecture}"
            ),
            "function_names": ",".join(real_result["function_names"]),
            **real_summary,
            **sim_summary,
        }
        row["real_warm_cost_usd"] = estimate_request_cost_usd(
            memory_mb=target.memory_mb,
            architecture=target.architecture,
            latency_ms=float(row["real_warm_mean_ms"]),
        )
        row["real_cold_cost_usd"] = estimate_request_cost_usd(
            memory_mb=target.memory_mb,
            architecture=target.architecture,
            latency_ms=float(row["real_cold_mean_ms"]),
        )
        row["sim_warm_cost_usd"] = estimate_request_cost_usd(
            memory_mb=target.memory_mb,
            architecture=target.architecture,
            latency_ms=float(row["sim_warm_mean_ms"]),
        )
        row["sim_cold_cost_usd"] = estimate_request_cost_usd(
            memory_mb=target.memory_mb,
            architecture=target.architecture,
            latency_ms=float(row["sim_cold_mean_ms"]),
        )
        row["real_warm_reward"] = compute_static_reward(
            latency_ms=float(row["real_warm_mean_ms"]),
            cost_usd=float(row["real_warm_cost_usd"]),
            success_rate=float(row["success_rate"]),
            cold_start_rate=0.0,
            latency_weight=args.latency_weight,
            cost_weight=args.cost_weight,
            success_bonus=args.success_bonus,
            failure_penalty=args.failure_penalty,
            cold_penalty=args.cold_penalty,
            queue_penalty=args.queue_penalty,
            cost_normalization=args.cost_normalization,
        )
        row["real_cold_reward"] = compute_static_reward(
            latency_ms=float(row["real_cold_mean_ms"]),
            cost_usd=float(row["real_cold_cost_usd"]),
            success_rate=float(row["success_rate"]),
            cold_start_rate=1.0,
            latency_weight=args.latency_weight,
            cost_weight=args.cost_weight,
            success_bonus=args.success_bonus,
            failure_penalty=args.failure_penalty,
            cold_penalty=args.cold_penalty,
            queue_penalty=args.queue_penalty,
            cost_normalization=args.cost_normalization,
        )
        row["sim_warm_reward"] = compute_static_reward(
            latency_ms=float(row["sim_warm_mean_ms"]),
            cost_usd=float(row["sim_warm_cost_usd"]),
            success_rate=1.0,
            cold_start_rate=0.0,
            latency_weight=args.latency_weight,
            cost_weight=args.cost_weight,
            success_bonus=args.success_bonus,
            failure_penalty=args.failure_penalty,
            cold_penalty=args.cold_penalty,
            queue_penalty=args.queue_penalty,
            cost_normalization=args.cost_normalization,
        )
        row["sim_cold_reward"] = compute_static_reward(
            latency_ms=float(row["sim_cold_mean_ms"]),
            cost_usd=float(row["sim_cold_cost_usd"]),
            success_rate=1.0,
            cold_start_rate=1.0,
            latency_weight=args.latency_weight,
            cost_weight=args.cost_weight,
            success_bonus=args.success_bonus,
            failure_penalty=args.failure_penalty,
            cold_penalty=args.cold_penalty,
            queue_penalty=args.queue_penalty,
            cost_normalization=args.cost_normalization,
        )
        row["warm_mean_abs_rel_error_pct"] = relative_error(
            float(row["sim_warm_mean_ms"]),
            float(row["real_warm_mean_ms"]),
        )
        row["cold_mean_abs_rel_error_pct"] = relative_error(
            float(row["sim_cold_mean_ms"]),
            float(row["real_cold_mean_ms"]),
        )
        rows.append(row)
        completed_keys.add(target_key)
        write_outputs(rows, args.output_dir)
        write_ranking_outputs(build_ranking_summary(rows), args.output_dir)
        print(
            (
                f"  cold captured={int(row['cold_selected_count'])}/{int(row['cold_target_runs'])}, "
                f"warm captured={int(row['warm_selected_count'])}/{int(row['warm_target_runs'])}, "
                f"invoke success={float(row['success_rate']):.2f}, "
                f"real warm={float(row['real_warm_mean_ms']):.2f} ms, "
                f"real cold={float(row['real_cold_mean_ms']):.2f} ms"
            ),
            flush=True,
        )

    print(f"Saved validation summary to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
