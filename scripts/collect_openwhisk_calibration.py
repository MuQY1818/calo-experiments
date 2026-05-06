#!/usr/bin/env python3
"""Collect OpenWhisk calibration data for the CALO simulator.

This script uses SeBS benchmarks as the workload under test and gathers
calibration data from a real OpenWhisk deployment. It focuses on three phases:

1. warm: steady-state latency measurements under different memory/timeout pairs
2. cold: idle-gap experiments to approximate cold-start behavior
3. burst: concurrent invocations to observe tail latency under load spikes

The script writes one raw invocation CSV and several aggregated profile CSVs that
can later drive a more realistic trace-based simulator.
"""

from __future__ import annotations

import argparse
import copy
import concurrent.futures
import csv
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RAW_COLUMNS = [
    "phase",
    "sample_kind",
    "benchmark",
    "function_name",
    "architecture",
    "memory_mb",
    "timeout_sec",
    "trigger",
    "round_index",
    "sample_index",
    "concurrency",
    "idle_gap_sec",
    "request_id",
    "success",
    "cold_start",
    "client_latency_ms",
    "benchmark_latency_ms",
    "provider_init_ms",
    "provider_exec_ms",
    "billed_time_ms",
    "gb_seconds",
    "memory_used_mb",
    "output_begin_ts",
    "output_end_ts",
    "captured_at_ts",
]

DEDUPE_KEY_COLUMNS = [
    "phase",
    "sample_kind",
    "benchmark",
    "architecture",
    "memory_mb",
    "timeout_sec",
    "round_index",
    "sample_index",
    "concurrency",
    "idle_gap_sec",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect real OpenWhisk calibration data for CALO."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the OpenWhisk calibration JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for calibration.output_dir.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["warm", "cold", "burst"],
        choices=["warm", "cold", "burst"],
        help="Calibration phases to execute.",
    )
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Do not delete an existing output directory before running.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run environment and config preflight checks, then exit.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional benchmark list to override calibration.benchmarks for this run.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the required top-level config sections."""
    required_sections = ["deployment", "experiments", "calibration"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing top-level config section: {section}")

    deployment = config["deployment"]
    if deployment.get("name") != "openwhisk":
        raise ValueError("This script only supports deployment.name == 'openwhisk'.")

    openwhisk_cfg = deployment.get("openwhisk", {})
    storage_cfg = openwhisk_cfg.get("storage")
    if not isinstance(storage_cfg, dict):
        raise ValueError("deployment.openwhisk.storage 缺失，无法为 OpenWhisk 准备 MinIO/ScyllaDB。")
    if "object" not in storage_cfg:
        raise ValueError(
            "deployment.openwhisk.storage.object 缺失，请补齐对象存储配置。"
        )

    experiments_cfg = config["experiments"]
    if not experiments_cfg.get("container_deployment", False):
        raise ValueError("OpenWhisk 校准必须启用 experiments.container_deployment=true。")

    calibration = config["calibration"]
    required_calibration_keys = [
        "benchmarks",
        "input_size",
        "trigger",
        "output_dir",
        "cache_dir",
        "resource_prefix",
        "warm_matrix",
        "cold_start",
        "burst",
    ]
    for key in required_calibration_keys:
        if key not in calibration:
            raise ValueError(f"Missing calibration config key: {key}")

    architectures = calibration.get("architectures")
    if architectures is not None:
        if not isinstance(architectures, list) or not architectures:
            raise ValueError("calibration.architectures 必须是非空列表")
        for architecture in architectures:
            if not isinstance(architecture, str) or not architecture:
                raise ValueError("calibration.architectures 中存在非法架构值")
    elif not config["experiments"].get("architecture"):
        raise ValueError("缺少 experiments.architecture，且未提供 calibration.architectures")

    benchmark_overrides = calibration.get("benchmark_overrides")
    if benchmark_overrides is not None and not isinstance(benchmark_overrides, dict):
        raise ValueError("calibration.benchmark_overrides 必须是 benchmark -> phase config 的字典")

    for key in ["continue_on_error", "skip_measured_on_warmup_failure"]:
        if key in calibration and not isinstance(calibration[key], bool):
            raise ValueError(f"calibration.{key} 必须是布尔值")

    if "max_failures_per_target" in calibration:
        max_failures = calibration["max_failures_per_target"]
        if not isinstance(max_failures, int) or max_failures <= 0:
            raise ValueError("calibration.max_failures_per_target 必须是正整数")

    cold_cfg = calibration["cold_start"]
    cold_mode = cold_cfg.get("mode", "idle_gap")
    if cold_mode not in {"idle_gap", "enforced_update"}:
        raise ValueError("calibration.cold_start.mode 仅支持 idle_gap 或 enforced_update")
    if "post_enforce_sleep_sec" in cold_cfg:
        sleep_sec = cold_cfg["post_enforce_sleep_sec"]
        if not isinstance(sleep_sec, (int, float)) or sleep_sec < 0:
            raise ValueError(
                "calibration.cold_start.post_enforce_sleep_sec 必须是非负数"
            )

    burst_cfg = calibration["burst"]
    if "prime_concurrency_rounds" in burst_cfg:
        prime_rounds = burst_cfg["prime_concurrency_rounds"]
        if not isinstance(prime_rounds, int) or prime_rounds < 0:
            raise ValueError(
                "calibration.burst.prime_concurrency_rounds 必须是非负整数"
            )


def collect_prerequisite_checks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect OpenWhisk runtime prerequisite checks."""
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    systems_path = Path("config/systems.json")
    add_check(
        "SeBS systems config",
        systems_path.exists(),
        str(systems_path) if systems_path.exists() else "缺少 config/systems.json",
    )

    openwhisk_tool_path = Path("tools/openwhisk_preparation.py")
    add_check(
        "OpenWhisk preparation tool",
        openwhisk_tool_path.exists(),
        str(openwhisk_tool_path) if openwhisk_tool_path.exists() else "缺少 tools/openwhisk_preparation.py",
    )

    dockerfile_path = Path("dockerfiles/openwhisk/python/Dockerfile.function")
    add_check(
        "OpenWhisk Python Dockerfile",
        dockerfile_path.exists(),
        str(dockerfile_path) if dockerfile_path.exists() else "缺少 dockerfiles/openwhisk/python/Dockerfile.function",
    )

    docker_exec = shutil.which("docker")
    add_check(
        "Docker CLI",
        docker_exec is not None,
        docker_exec if docker_exec is not None else "PATH 中未找到 docker 可执行文件",
    )

    docker_sdk = None
    try:
        import docker as docker_sdk  # type: ignore

        add_check("Python docker package", True, "已安装")
    except ModuleNotFoundError:
        add_check(
            "Python docker package",
            False,
            "当前环境缺少 docker 包，请在 python-venv 中安装 SeBS 依赖。",
        )

    if docker_sdk is not None:
        try:
            docker_client = docker_sdk.from_env()
            docker_client.ping()
            add_check("Docker daemon", True, "Docker daemon 可访问")
            docker_client.close()
        except Exception as exc:  # pragma: no cover - depends on local daemon state
            add_check("Docker daemon", False, f"Docker daemon 不可访问: {exc}")

    wsk_exec = config["deployment"]["openwhisk"]["wskExec"]
    wsk_path = shutil.which(wsk_exec)
    if wsk_path is None and Path(wsk_exec).exists():
        wsk_path = str(Path(wsk_exec).resolve())
    add_check(
        "wsk executable",
        wsk_path is not None,
        wsk_path if wsk_path is not None else f"找不到 wsk 可执行文件: {wsk_exec}",
    )

    if wsk_path is not None:
        try:
            result = subprocess.run(
                [wsk_exec, "property", "get"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = "\n".join(
                part.strip() for part in [result.stdout or "", result.stderr or ""] if part.strip()
            )
            has_api_host = "whisk api host" in output.lower()
            has_auth = "whisk auth" in output.lower()
            cli_detail = "已检测到 apihost 和 auth 配置" if has_api_host and has_auth else (
                output.splitlines()[0] if output else "未检测到 wsk property 配置"
            )
            add_check(
                "wsk CLI properties",
                has_api_host and has_auth,
                cli_detail,
            )
            reachability_detail = (
                "OpenWhisk API 可访问"
                if result.returncode == 0
                else (output.splitlines()[-1] if output else "无法连接 OpenWhisk API")
            )
            add_check(
                "OpenWhisk API reachability",
                result.returncode == 0,
                reachability_detail,
            )
        except Exception as exc:  # pragma: no cover - depends on local CLI state
            add_check("wsk CLI properties", False, f"无法读取 wsk properties: {exc}")
            add_check("OpenWhisk API reachability", False, f"无法验证 OpenWhisk API: {exc}")

    return checks


def format_prerequisite_report(checks: Sequence[Dict[str, Any]]) -> str:
    """Format the preflight check result as a readable report."""
    lines = ["OpenWhisk preflight checks:"]
    for check in checks:
        state = "PASS" if check["ok"] else "FAIL"
        lines.append(f"- [{state}] {check['name']}: {check['detail']}")
    return "\n".join(lines)


def ensure_prerequisites(config: Dict[str, Any], checks: Sequence[Dict[str, Any]] | None = None) -> None:
    """Check runtime prerequisites before importing SeBS internals."""
    resolved_checks = list(checks) if checks is not None else collect_prerequisite_checks(config)
    failed_checks = [check for check in resolved_checks if not check["ok"]]
    if failed_checks:
        raise RuntimeError(format_prerequisite_report(resolved_checks))


def sanitize_name(name: str) -> str:
    """Sanitize a benchmark name for use as an OpenWhisk action name."""
    return name.replace(".", "-").replace("_", "-")


def percentile(series: pd.Series, ratio: float) -> float:
    """Return a percentile with safe handling of empty series."""
    if series.empty:
        return float("nan")
    return float(series.quantile(ratio))


class OpenWhiskCalibrationCollector:
    """Collects warm, cold, and burst calibration data from OpenWhisk."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        keep_existing_output: bool,
    ):
        self.config = config
        self.calibration_cfg = config["calibration"]
        self.experiments_cfg = config["experiments"]
        self.output_dir = output_dir
        self.keep_existing_output = keep_existing_output
        self.raw_csv_path = self.output_dir / "calibration_invocations.csv"
        self.metadata_path = self.output_dir / "metadata.json"
        self.log_path = self.output_dir / "collector.log"

        self.sebs_client = None
        self.deployment_client = None
        self.experiment_config = None
        self.trigger_type = None

        self._architectures = self._resolve_architectures()
        self._benchmark_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._experiment_config_cache: Dict[str, Any] = {}
        self._successful_invocation_keys: set[tuple[Any, ...]] = set()
        self._successful_invocation_outcomes: Dict[tuple[Any, ...], bool] = {}

    def _resolve_architectures(self) -> List[str]:
        """Return the list of architectures to calibrate."""
        architectures = self.calibration_cfg.get("architectures")
        if architectures:
            return [str(architecture) for architecture in architectures]
        return [str(self.experiments_cfg.get("architecture", "x64"))]

    def _resolve_benchmarks(self) -> List[str]:
        """Return the benchmark list for the current run."""
        return [str(benchmark) for benchmark in self.calibration_cfg["benchmarks"]]

    def _get_phase_config(self, benchmark_name: str, phase_name: str) -> Dict[str, Any]:
        """Return one phase config with optional benchmark-specific overrides."""
        phase_cfg = copy.deepcopy(self.calibration_cfg[phase_name])
        benchmark_overrides = self.calibration_cfg.get("benchmark_overrides", {})
        benchmark_cfg = benchmark_overrides.get(benchmark_name, {})
        phase_override = benchmark_cfg.get(phase_name, {})
        if not phase_override:
            return phase_cfg

        for key, value in phase_override.items():
            phase_cfg[key] = copy.deepcopy(value)
        return phase_cfg

    def _continue_on_error(self) -> bool:
        """Return whether collection should continue after invocation failures."""
        return bool(self.calibration_cfg.get("continue_on_error", True))

    def _skip_measured_on_warmup_failure(self) -> bool:
        """Return whether measured warm samples should be skipped after failed warmups."""
        return bool(self.calibration_cfg.get("skip_measured_on_warmup_failure", True))

    def _max_failures_per_target(self) -> int:
        """Return the maximum consecutive failures before skipping a target."""
        return int(self.calibration_cfg.get("max_failures_per_target", 2))

    def setup(self) -> None:
        """Initialize output paths and SeBS clients."""
        append_output = self.output_dir.exists() and self.keep_existing_output
        if self.output_dir.exists() and not self.keep_existing_output:
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO if self.calibration_cfg.get("verbose", True) else logging.WARNING,
            format="[%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.log_path,
                    mode="a" if append_output else "w",
                    encoding="utf-8",
                ),
            ],
        )

        self._initialize_raw_csv()
        self._load_existing_successful_invocations()
        self._initialize_sebs_clients()
        self._write_metadata()

    def teardown(self) -> None:
        """Shutdown SeBS resources."""
        if self.deployment_client is not None:
            self.deployment_client.shutdown()
        if self.sebs_client is not None:
            self.sebs_client.shutdown()

    def run(self, phases: Sequence[str]) -> None:
        """Run selected calibration phases and write aggregated outputs."""
        if "warm" in phases:
            self.collect_warm_matrix()
        if "cold" in phases:
            self.collect_cold_start_profile()
        if "burst" in phases:
            self.collect_burst_profile()
        self.write_aggregates()

    def _refresh_aggregate_snapshot(self, phase: str, benchmark_name: str) -> None:
        """Refresh aggregate CSVs during long-running collection jobs."""
        try:
            self.write_aggregates()
            logging.info(
                "Refreshed aggregate snapshot after %s phase for %s.",
                phase,
                benchmark_name,
            )
        except Exception as exc:  # pragma: no cover - best-effort checkpointing
            logging.warning(
                "Failed to refresh aggregate snapshot after %s phase for %s: %s",
                phase,
                benchmark_name,
                exc,
            )

    def _initialize_raw_csv(self) -> None:
        """Create the raw CSV file with header."""
        if self.keep_existing_output and self.raw_csv_path.exists():
            if self.raw_csv_path.stat().st_size > 0:
                logging.info("Reusing existing raw CSV at %s", self.raw_csv_path)
                return
        with self.raw_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
            writer.writeheader()

    def _load_existing_successful_invocations(self) -> None:
        """Load successful invocation keys from an existing raw CSV for resume."""
        if not self.keep_existing_output or not self.raw_csv_path.exists():
            return
        if self.raw_csv_path.stat().st_size <= 0:
            return

        raw = pd.read_csv(
            self.raw_csv_path,
            usecols=lambda column: column in {*DEDUPE_KEY_COLUMNS, "success", "cold_start"},
        )
        if raw.empty:
            return

        normalized = self._normalize_raw_dataframe(raw)
        successful = normalized[normalized["success"]].copy()
        if successful.empty:
            return

        self._successful_invocation_keys = {
            self._build_invocation_key(
                phase=str(row.phase),
                sample_kind=str(row.sample_kind),
                benchmark_name=str(row.benchmark),
                architecture=str(row.architecture),
                memory_mb=int(row.memory_mb),
                timeout_sec=int(row.timeout_sec),
                round_index=int(row.round_index),
                sample_index=int(row.sample_index),
                concurrency=int(row.concurrency),
                idle_gap_sec=int(row.idle_gap_sec),
            )
            for row in successful.itertuples(index=False)
        }
        self._successful_invocation_outcomes = {
            self._build_invocation_key(
                phase=str(row.phase),
                sample_kind=str(row.sample_kind),
                benchmark_name=str(row.benchmark),
                architecture=str(row.architecture),
                memory_mb=int(row.memory_mb),
                timeout_sec=int(row.timeout_sec),
                round_index=int(row.round_index),
                sample_index=int(row.sample_index),
                concurrency=int(row.concurrency),
                idle_gap_sec=int(row.idle_gap_sec),
            ): bool(row.cold_start)
            for row in successful.itertuples(index=False)
        }
        logging.info(
            "Loaded %s successful invocation keys from %s for resume.",
            len(self._successful_invocation_keys),
            self.raw_csv_path,
        )

    def _build_invocation_key(
        self,
        phase: str,
        sample_kind: str,
        benchmark_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        round_index: int,
        sample_index: int,
        concurrency: int,
        idle_gap_sec: int,
    ) -> tuple[Any, ...]:
        """Build one dedupe key for a raw invocation row."""
        return (
            phase,
            sample_kind,
            benchmark_name,
            architecture,
            int(memory_mb),
            int(timeout_sec),
            int(round_index),
            int(sample_index),
            int(concurrency),
            int(idle_gap_sec),
        )

    def _has_successful_invocation(
        self,
        phase: str,
        sample_kind: str,
        benchmark_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        round_index: int,
        sample_index: int,
        concurrency: int,
        idle_gap_sec: int,
    ) -> bool:
        """Return whether one invocation key already has a successful sample."""
        key = self._build_invocation_key(
            phase=phase,
            sample_kind=sample_kind,
            benchmark_name=benchmark_name,
            architecture=architecture,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            round_index=round_index,
            sample_index=sample_index,
            concurrency=concurrency,
            idle_gap_sec=idle_gap_sec,
        )
        return key in self._successful_invocation_keys

    def _get_successful_invocation_outcome(
        self,
        phase: str,
        sample_kind: str,
        benchmark_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        round_index: int,
        sample_index: int,
        concurrency: int,
        idle_gap_sec: int,
    ) -> bool | None:
        """Return a cached cold-start outcome for one successful invocation."""
        key = self._build_invocation_key(
            phase=phase,
            sample_kind=sample_kind,
            benchmark_name=benchmark_name,
            architecture=architecture,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            round_index=round_index,
            sample_index=sample_index,
            concurrency=concurrency,
            idle_gap_sec=idle_gap_sec,
        )
        return self._successful_invocation_outcomes.get(key)

    def _all_cold_samples_exist(
        self,
        benchmark_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        idle_gaps_sec: Sequence[int],
        measured_repetitions: int,
    ) -> bool:
        """Return whether the fixed sweep matrix already exists for one target."""
        for idle_gap_sec in idle_gaps_sec:
            for sample_index in range(measured_repetitions):
                if not self._has_successful_invocation(
                    phase="cold",
                    sample_kind="measured",
                    benchmark_name=benchmark_name,
                    architecture=architecture,
                    memory_mb=memory_mb,
                    timeout_sec=timeout_sec,
                    round_index=0,
                    sample_index=sample_index,
                    concurrency=1,
                    idle_gap_sec=int(idle_gap_sec),
                ):
                    return False
        return True

    def _load_existing_idle_gap_outcomes(
        self,
        benchmark_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        idle_gaps_sec: Sequence[int],
    ) -> Dict[int, bool]:
        """Load existing measured cold-start outcomes for one idle-gap target."""
        outcomes: Dict[int, bool] = {}
        for idle_gap_sec in idle_gaps_sec:
            outcome = self._get_successful_invocation_outcome(
                phase="cold",
                sample_kind="measured",
                benchmark_name=benchmark_name,
                architecture=architecture,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                round_index=0,
                sample_index=0,
                concurrency=1,
                idle_gap_sec=int(idle_gap_sec),
            )
            if outcome is not None:
                outcomes[int(idle_gap_sec)] = bool(outcome)
        return outcomes

    def _is_ttl_identified(
        self,
        idle_gaps_sec: Sequence[int],
        outcomes: Dict[int, bool],
    ) -> bool:
        """Return whether current discrete idle-gap observations already identify TTL."""
        if not outcomes:
            return False
        ordered = sorted(int(gap) for gap in idle_gaps_sec)
        measured = {int(gap): bool(outcomes[int(gap)]) for gap in ordered if int(gap) in outcomes}
        if not measured:
            return False

        highest_measured_gap = max(measured)
        if highest_measured_gap == ordered[-1] and measured[highest_measured_gap] is False:
            return True

        first_cold_gap = next((gap for gap in ordered if measured.get(gap) is True), None)
        if first_cold_gap is None:
            return False
        cold_index = ordered.index(first_cold_gap)
        if cold_index == 0:
            return True
        previous_gap = ordered[cold_index - 1]
        return measured.get(previous_gap) is False

    def _select_next_idle_gap_for_ttl_search(
        self,
        idle_gaps_sec: Sequence[int],
        outcomes: Dict[int, bool],
        cold_cfg: Dict[str, Any],
    ) -> int | None:
        """Choose the next idle-gap point for adaptive TTL search."""
        ordered = sorted(int(gap) for gap in idle_gaps_sec)
        if self._is_ttl_identified(ordered, outcomes):
            return None

        measured = {int(gap): bool(value) for gap, value in outcomes.items()}
        unmeasured = [gap for gap in ordered if gap not in measured]
        if not unmeasured:
            return None

        initial_gap_sec = cold_cfg.get("ttl_search_initial_gap_sec")
        if not measured and initial_gap_sec is not None:
            initial_gap_sec = int(initial_gap_sec)
            if initial_gap_sec in unmeasured:
                return initial_gap_sec

        warm_gaps = [gap for gap, is_cold in measured.items() if not is_cold]
        cold_gaps = [gap for gap, is_cold in measured.items() if is_cold]

        if cold_gaps:
            first_cold_gap = min(cold_gaps)
            lower_candidates = [gap for gap in ordered if gap < first_cold_gap and gap not in measured]
            if not lower_candidates:
                return None
            return lower_candidates[len(lower_candidates) // 2]

        if warm_gaps:
            highest_warm_gap = max(warm_gaps)
            upper_candidates = [gap for gap in ordered if gap > highest_warm_gap and gap not in measured]
            if not upper_candidates:
                return None
            if bool(cold_cfg.get("ttl_search_upper_bias", True)):
                return upper_candidates[-1]
            return upper_candidates[len(upper_candidates) // 2]

        return ordered[len(ordered) // 2]

    def _invoke_cold_sample(
        self,
        benchmark_name: str,
        function_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        trigger: Any,
        input_config: Dict[str, Any],
        function: Any,
        benchmark_obj: Any,
        cold_mode: str,
        enforce_sleep_sec: float,
        prime_repetitions: int,
        idle_gap_sec: int,
        sample_index: int,
    ) -> tuple[bool, bool | None]:
        """Run one cold measurement or reuse an existing successful sample."""
        existing_outcome = self._get_successful_invocation_outcome(
            phase="cold",
            sample_kind="measured",
            benchmark_name=benchmark_name,
            architecture=architecture,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            round_index=0,
            sample_index=sample_index,
            concurrency=1,
            idle_gap_sec=idle_gap_sec,
        )
        if existing_outcome is not None:
            logging.info(
                "Skipping existing cold sample for %s at %s/%s/%s idle_gap=%s sample=%s.",
                benchmark_name,
                architecture,
                memory_mb,
                timeout_sec,
                idle_gap_sec,
                sample_index,
            )
            return True, bool(existing_outcome)

        if cold_mode == "enforced_update":
            assert self.deployment_client is not None
            self.deployment_client.enforce_cold_start([function], benchmark_obj)
            if enforce_sleep_sec > 0:
                logging.info(
                    "Cold sample wait: benchmark=%s architecture=%s memory=%s "
                    "timeout=%s mode=%s sleep_sec=%.1f sample=%s",
                    benchmark_name,
                    architecture,
                    memory_mb,
                    timeout_sec,
                    cold_mode,
                    enforce_sleep_sec,
                    sample_index,
                )
                time.sleep(enforce_sleep_sec)
        else:
            for _ in range(prime_repetitions):
                trigger.sync_invoke(input_config)
            if idle_gap_sec > 0:
                logging.info(
                    "Cold sample wait: benchmark=%s architecture=%s memory=%s "
                    "timeout=%s idle_gap=%s sample=%s",
                    benchmark_name,
                    architecture,
                    memory_mb,
                    timeout_sec,
                    idle_gap_sec,
                    sample_index,
                )
                time.sleep(idle_gap_sec)

        result = trigger.sync_invoke(input_config)
        success = self._record_result(
            result=result,
            phase="cold",
            sample_kind="measured",
            benchmark_name=benchmark_name,
            function_name=function_name,
            architecture=architecture,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            round_index=0,
            sample_index=sample_index,
            concurrency=1,
            idle_gap_sec=idle_gap_sec,
        )
        if not success:
            return False, None
        return True, bool(result.stats.cold_start)

    def _initialize_sebs_clients(self) -> None:
        """Initialize SeBS deployment and trigger type."""
        os.environ.setdefault("SEBS_WITH_OPENWHISK", "true")

        from sebs import SeBS
        from sebs.faas.function import Trigger

        cache_dir = self.calibration_cfg["cache_dir"]
        self.sebs_client = SeBS(
            cache_dir=cache_dir,
            output_dir=str(self.output_dir),
            verbose=self.calibration_cfg.get("verbose", True),
            logging_filename=str(self.log_path),
        )
        self.deployment_client = self.sebs_client.get_deployment(
            self.config,
            logging_filename=str(self.log_path),
        )
        self.deployment_client.initialize(
            resource_prefix=self.calibration_cfg.get("resource_prefix")
        )
        self.experiment_config = self.sebs_client.get_experiment_config(self.experiments_cfg)
        self.trigger_type = Trigger.TriggerType.get(self.calibration_cfg["trigger"])

    def _write_metadata(self) -> None:
        """Write metadata for this collection run."""
        metadata = {
            "config": self.config,
            "architectures": self._architectures,
            "hostname": socket.gethostname(),
            "started_at_ts": time.time(),
            "raw_csv": str(self.raw_csv_path),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _get_experiment_config(self, architecture: str):
        """Build or retrieve one SeBS experiment config for an architecture."""
        if architecture in self._experiment_config_cache:
            return self._experiment_config_cache[architecture]

        assert self.sebs_client is not None

        experiment_cfg_dict = json.loads(json.dumps(self.experiments_cfg))
        experiment_cfg_dict["architecture"] = architecture
        experiment_config = self.sebs_client.get_experiment_config(experiment_cfg_dict)
        self._experiment_config_cache[architecture] = experiment_config
        return experiment_config

    def _get_benchmark_context(
        self,
        benchmark_name: str,
        architecture: str,
    ) -> Dict[str, Any]:
        """Build or retrieve cached SeBS objects for one benchmark."""
        cache_key = (benchmark_name, architecture)
        if cache_key in self._benchmark_cache:
            return self._benchmark_cache[cache_key]

        assert self.sebs_client is not None
        assert self.deployment_client is not None

        experiment_config = self._get_experiment_config(architecture)
        benchmark_obj = self.sebs_client.get_benchmark(
            benchmark_name,
            self.deployment_client,
            experiment_config,
            logging_filename=str(self.log_path),
        )
        input_config = benchmark_obj.prepare_input(
            self.deployment_client.system_resources,
            size=self.calibration_cfg["input_size"],
            replace_existing=experiment_config.update_storage,
        )

        runtime = self.experiments_cfg["runtime"]
        function_name = (
            f"calo-calib-{sanitize_name(benchmark_name)}-"
            f"{runtime['language']}-{runtime['version']}-{architecture}"
        )

        context = {
            "benchmark_obj": benchmark_obj,
            "input_config": input_config,
            "function_name": function_name,
        }
        self._benchmark_cache[cache_key] = context
        return context

    def _get_function_and_trigger(
        self,
        benchmark_name: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> Dict[str, Any]:
        """Ensure the action is deployed with the requested configuration."""
        assert self.deployment_client is not None
        assert self.trigger_type is not None

        context = self._get_benchmark_context(benchmark_name, architecture)
        benchmark_obj = context["benchmark_obj"]
        benchmark_obj.benchmark_config.memory = memory_mb
        benchmark_obj.benchmark_config.timeout = timeout_sec

        function = self.deployment_client.get_function(
            benchmark_obj,
            context["function_name"],
        )
        triggers = function.triggers(self.trigger_type)
        trigger = triggers[0] if triggers else self.deployment_client.create_trigger(
            function, self.trigger_type
        )

        return {
            "function": function,
            "trigger": trigger,
            "input_config": context["input_config"],
            "function_name": context["function_name"],
            "benchmark_obj": context["benchmark_obj"],
        }

    def _try_get_function_and_trigger(
        self,
        benchmark_name: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
        phase_name: str,
    ) -> Dict[str, Any] | None:
        """Return a target handle, or continue after target-setup failure."""
        try:
            return self._get_function_and_trigger(
                benchmark_name=benchmark_name,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                architecture=architecture,
            )
        except Exception as exc:  # pylint: disable=broad-except
            message = (
                "Target setup failed for phase=%s benchmark=%s architecture=%s "
                "memory=%s timeout=%s: %s"
            )
            if not self._continue_on_error():
                raise RuntimeError(
                    message
                    % (
                        phase_name,
                        benchmark_name,
                        architecture,
                        memory_mb,
                        timeout_sec,
                        exc,
                    )
                ) from exc
            logging.warning(
                message,
                phase_name,
                benchmark_name,
                architecture,
                memory_mb,
                timeout_sec,
                exc,
            )
            return None

    def collect_warm_matrix(self) -> None:
        """Collect warm latency measurements across memory/timeout settings."""
        for architecture in self._architectures:
            for benchmark_name in self._resolve_benchmarks():
                warm_cfg = self._get_phase_config(benchmark_name, "warm_matrix")
                for timeout_sec in warm_cfg["timeout_sec"]:
                    for memory_mb in warm_cfg["memory_mb"]:
                        logging.info(
                            "Warm phase: benchmark=%s architecture=%s memory=%s timeout=%s",
                            benchmark_name,
                            architecture,
                            memory_mb,
                            timeout_sec,
                        )
                        target = self._try_get_function_and_trigger(
                            benchmark_name,
                            memory_mb,
                            timeout_sec,
                            architecture,
                            "warm",
                        )
                        if target is None:
                            continue
                        trigger = target["trigger"]
                        input_config = target["input_config"]
                        function_name = target["function_name"]

                        warmup_ok = self._invoke_serial(
                            phase="warm",
                            sample_kind="warmup",
                            benchmark_name=benchmark_name,
                            function_name=function_name,
                            architecture=architecture,
                            memory_mb=memory_mb,
                            timeout_sec=timeout_sec,
                            trigger=trigger,
                            input_config=input_config,
                            repetitions=warm_cfg["warmup_repetitions"],
                        )
                        if not warmup_ok and self._skip_measured_on_warmup_failure():
                            logging.warning(
                                "Skipping measured warm samples for %s at %s/%s/%s after warmup failures.",
                                benchmark_name,
                                architecture,
                                memory_mb,
                                timeout_sec,
                            )
                            continue
                        self._invoke_serial(
                            phase="warm",
                            sample_kind="measured",
                            benchmark_name=benchmark_name,
                            function_name=function_name,
                            architecture=architecture,
                            memory_mb=memory_mb,
                            timeout_sec=timeout_sec,
                            trigger=trigger,
                            input_config=input_config,
                            repetitions=warm_cfg["measured_repetitions"],
                        )
                self._refresh_aggregate_snapshot("warm", benchmark_name)

    def collect_cold_start_profile(self) -> None:
        """Collect idle-gap measurements to approximate cold-start behavior."""
        for architecture in self._architectures:
            for benchmark_name in self._resolve_benchmarks():
                cold_cfg = self._get_phase_config(benchmark_name, "cold_start")
                timeout_sec = cold_cfg["timeout_sec"]
                cold_mode = str(cold_cfg.get("mode", "idle_gap"))
                enforce_sleep_sec = float(cold_cfg.get("post_enforce_sleep_sec", 2.0))
                idle_gap_strategy = str(cold_cfg.get("idle_gap_strategy", "sweep"))
                for memory_mb in cold_cfg["memory_mb"]:
                    measured_idle_gaps = (
                        [-1]
                        if cold_mode == "enforced_update"
                        else [int(gap) for gap in cold_cfg["idle_gaps_sec"]]
                    )
                    if (
                        cold_mode == "idle_gap"
                        and idle_gap_strategy == "sweep"
                        and self._all_cold_samples_exist(
                            benchmark_name=benchmark_name,
                            architecture=architecture,
                            memory_mb=memory_mb,
                            timeout_sec=timeout_sec,
                            idle_gaps_sec=measured_idle_gaps,
                            measured_repetitions=int(cold_cfg["measured_repetitions"]),
                        )
                    ):
                        logging.info(
                            "Skipping completed cold target for %s at %s/%s/%s.",
                            benchmark_name,
                            architecture,
                            memory_mb,
                            timeout_sec,
                        )
                        continue
                    if cold_mode == "idle_gap" and idle_gap_strategy == "adaptive_ttl_search":
                        existing_outcomes = self._load_existing_idle_gap_outcomes(
                            benchmark_name=benchmark_name,
                            architecture=architecture,
                            memory_mb=memory_mb,
                            timeout_sec=timeout_sec,
                            idle_gaps_sec=measured_idle_gaps,
                        )
                        if self._is_ttl_identified(measured_idle_gaps, existing_outcomes):
                            logging.info(
                                "Skipping TTL-identified cold target for %s at %s/%s/%s.",
                                benchmark_name,
                                architecture,
                                memory_mb,
                                timeout_sec,
                            )
                            continue

                    logging.info(
                        "Cold phase: benchmark=%s architecture=%s memory=%s timeout=%s",
                        benchmark_name,
                        architecture,
                        memory_mb,
                        timeout_sec,
                    )
                    target = self._try_get_function_and_trigger(
                        benchmark_name,
                        memory_mb,
                        timeout_sec,
                        architecture,
                        "cold",
                    )
                    if target is None:
                        continue
                    trigger = target["trigger"]
                    input_config = target["input_config"]
                    function_name = target["function_name"]
                    function = target["function"]
                    benchmark_obj = target["benchmark_obj"]

                    warmup_ok = self._invoke_serial(
                        phase="cold",
                        sample_kind="warmup",
                        benchmark_name=benchmark_name,
                        function_name=function_name,
                        architecture=architecture,
                        memory_mb=memory_mb,
                        timeout_sec=timeout_sec,
                        trigger=trigger,
                        input_config=input_config,
                        repetitions=cold_cfg["initial_warmup_repetitions"],
                    )
                    if not warmup_ok:
                        logging.warning(
                            "Skipping cold-start measurements for %s at %s/%s/%s because warmup failed.",
                            benchmark_name,
                            architecture,
                            memory_mb,
                            timeout_sec,
                        )
                        continue

                    if cold_mode == "idle_gap" and idle_gap_strategy == "adaptive_ttl_search":
                        observed_outcomes = self._load_existing_idle_gap_outcomes(
                            benchmark_name=benchmark_name,
                            architecture=architecture,
                            memory_mb=memory_mb,
                            timeout_sec=timeout_sec,
                            idle_gaps_sec=measured_idle_gaps,
                        )
                        failure_count = 0
                        while True:
                            next_idle_gap = self._select_next_idle_gap_for_ttl_search(
                                idle_gaps_sec=measured_idle_gaps,
                                outcomes=observed_outcomes,
                                cold_cfg=cold_cfg,
                            )
                            if next_idle_gap is None:
                                break
                            success, cold_start = self._invoke_cold_sample(
                                benchmark_name=benchmark_name,
                                function_name=function_name,
                                architecture=architecture,
                                memory_mb=memory_mb,
                                timeout_sec=timeout_sec,
                                trigger=trigger,
                                input_config=input_config,
                                function=function,
                                benchmark_obj=benchmark_obj,
                                cold_mode=cold_mode,
                                enforce_sleep_sec=enforce_sleep_sec,
                                prime_repetitions=int(cold_cfg["prime_repetitions"]),
                                idle_gap_sec=next_idle_gap,
                                sample_index=0,
                            )
                            failure_count = 0 if success else failure_count + 1
                            if success and cold_start is not None:
                                observed_outcomes[int(next_idle_gap)] = bool(cold_start)
                            if failure_count >= self._max_failures_per_target():
                                logging.warning(
                                    "Stopping adaptive TTL search for %s at %s/%s/%s after %s consecutive failures.",
                                    benchmark_name,
                                    architecture,
                                    memory_mb,
                                    timeout_sec,
                                    failure_count,
                                )
                                break
                    else:
                        for idle_gap_sec in measured_idle_gaps:
                            failure_count = 0
                            for sample_index in range(cold_cfg["measured_repetitions"]):
                                success, _ = self._invoke_cold_sample(
                                    benchmark_name=benchmark_name,
                                    function_name=function_name,
                                    architecture=architecture,
                                    memory_mb=memory_mb,
                                    timeout_sec=timeout_sec,
                                    trigger=trigger,
                                    input_config=input_config,
                                    function=function,
                                    benchmark_obj=benchmark_obj,
                                    cold_mode=cold_mode,
                                    enforce_sleep_sec=enforce_sleep_sec,
                                    prime_repetitions=int(cold_cfg["prime_repetitions"]),
                                    idle_gap_sec=int(idle_gap_sec),
                                    sample_index=sample_index,
                                )
                                failure_count = 0 if success else failure_count + 1
                                if failure_count >= self._max_failures_per_target():
                                    logging.warning(
                                        "Skipping remaining cold samples for %s at %s/%s/%s idle_gap=%s after %s consecutive failures.",
                                        benchmark_name,
                                        architecture,
                                        memory_mb,
                                        timeout_sec,
                                        idle_gap_sec,
                                        failure_count,
                                    )
                                    break
                self._refresh_aggregate_snapshot("cold", benchmark_name)

    def collect_burst_profile(self) -> None:
        """Collect concurrent burst measurements."""
        for architecture in self._architectures:
            for benchmark_name in self._resolve_benchmarks():
                burst_cfg = self._get_phase_config(benchmark_name, "burst")
                timeout_sec = burst_cfg["timeout_sec"]
                for memory_mb in burst_cfg["memory_mb"]:
                    logging.info(
                        "Burst phase: benchmark=%s architecture=%s memory=%s timeout=%s",
                        benchmark_name,
                        architecture,
                        memory_mb,
                        timeout_sec,
                    )
                    target = self._try_get_function_and_trigger(
                        benchmark_name,
                        memory_mb,
                        timeout_sec,
                        architecture,
                        "burst",
                    )
                    if target is None:
                        continue
                    trigger = target["trigger"]
                    input_config = target["input_config"]
                    function_name = target["function_name"]

                    for concurrency in burst_cfg["concurrency"]:
                        failure_count = 0
                        for round_index in range(burst_cfg["rounds"]):
                            if all(
                                self._has_successful_invocation(
                                    phase="burst",
                                    sample_kind="measured",
                                    benchmark_name=benchmark_name,
                                    architecture=architecture,
                                    memory_mb=memory_mb,
                                    timeout_sec=timeout_sec,
                                    round_index=round_index,
                                    sample_index=sample_index,
                                    concurrency=concurrency,
                                    idle_gap_sec=burst_cfg["pre_round_idle_sec"],
                                )
                                for sample_index in range(concurrency)
                            ):
                                logging.info(
                                    "Skipping existing burst round for %s at %s/%s/%s concurrency=%s round=%s.",
                                    benchmark_name,
                                    architecture,
                                    memory_mb,
                                    timeout_sec,
                                    concurrency,
                                    round_index,
                                )
                                continue
                            pre_round_failed = False
                            for warmup_index in range(burst_cfg["pre_round_warmups"]):
                                result = trigger.sync_invoke(input_config)
                                warmup_success = self._record_result(
                                    result=result,
                                    phase="burst",
                                    sample_kind="prewarmup",
                                    benchmark_name=benchmark_name,
                                    function_name=function_name,
                                    architecture=architecture,
                                    memory_mb=memory_mb,
                                    timeout_sec=timeout_sec,
                                    round_index=round_index,
                                    sample_index=warmup_index,
                                    concurrency=concurrency,
                                    idle_gap_sec=burst_cfg["pre_round_idle_sec"],
                                )
                                if not warmup_success:
                                    pre_round_failed = True
                                    break
                            if pre_round_failed:
                                logging.warning(
                                    "Skipping burst round for %s at %s/%s/%s concurrency=%s because pre-round warmup failed.",
                                    benchmark_name,
                                    architecture,
                                    memory_mb,
                                    timeout_sec,
                                    concurrency,
                                )
                                continue
                            if burst_cfg["pre_round_idle_sec"] > 0:
                                time.sleep(burst_cfg["pre_round_idle_sec"])

                            # Prime the container pool right before the measured burst.
                            # This avoids mixing intentional warm-pool preparation with the
                            # optional idle gap that may be used to emulate quiet periods.
                            prime_failed = False
                            for prime_round_index in range(
                                int(burst_cfg.get("prime_concurrency_rounds", 0))
                            ):
                                with concurrent.futures.ThreadPoolExecutor(
                                    max_workers=concurrency
                                ) as executor:
                                    futures = [
                                        executor.submit(trigger.sync_invoke, input_config)
                                        for _ in range(concurrency)
                                    ]
                                    for prime_sample_index, future in enumerate(futures):
                                        result = future.result()
                                        prime_success = self._record_result(
                                            result=result,
                                            phase="burst",
                                            sample_kind="prime",
                                            benchmark_name=benchmark_name,
                                            function_name=function_name,
                                            architecture=architecture,
                                            memory_mb=memory_mb,
                                            timeout_sec=timeout_sec,
                                            round_index=round_index,
                                            sample_index=(
                                                prime_round_index * concurrency
                                                + prime_sample_index
                                            ),
                                            concurrency=concurrency,
                                            idle_gap_sec=burst_cfg["pre_round_idle_sec"],
                                        )
                                        if not prime_success:
                                            prime_failed = True
                            if prime_failed:
                                logging.warning(
                                    "Skipping burst round for %s at %s/%s/%s concurrency=%s because priming failed.",
                                    benchmark_name,
                                    architecture,
                                    memory_mb,
                                    timeout_sec,
                                    concurrency,
                                )
                                continue

                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=concurrency
                            ) as executor:
                                futures = [
                                    executor.submit(trigger.sync_invoke, input_config)
                                    for _ in range(concurrency)
                                ]
                                for sample_index, future in enumerate(futures):
                                    result = future.result()
                                    success = self._record_result(
                                        result=result,
                                        phase="burst",
                                        sample_kind="measured",
                                        benchmark_name=benchmark_name,
                                        function_name=function_name,
                                        architecture=architecture,
                                        memory_mb=memory_mb,
                                        timeout_sec=timeout_sec,
                                        round_index=round_index,
                                        sample_index=sample_index,
                                        concurrency=concurrency,
                                        idle_gap_sec=burst_cfg["pre_round_idle_sec"],
                                    )
                                    failure_count = 0 if success else failure_count + 1
                                if failure_count >= self._max_failures_per_target():
                                    logging.warning(
                                        "Skipping remaining burst rounds for %s at %s/%s/%s concurrency=%s after %s consecutive failures.",
                                        benchmark_name,
                                        architecture,
                                        memory_mb,
                                        timeout_sec,
                                        concurrency,
                                        failure_count,
                                    )
                                    break
                self._refresh_aggregate_snapshot("burst", benchmark_name)

    def _invoke_serial(
        self,
        phase: str,
        sample_kind: str,
        benchmark_name: str,
        function_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        trigger: Any,
        input_config: Dict[str, Any],
        repetitions: int,
    ) -> bool:
        """Run sequential invocations and append rows to the raw CSV."""
        had_success = False
        failure_count = 0
        for sample_index in range(repetitions):
            if self._has_successful_invocation(
                phase=phase,
                sample_kind=sample_kind,
                benchmark_name=benchmark_name,
                architecture=architecture,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                round_index=0,
                sample_index=sample_index,
                concurrency=1,
                idle_gap_sec=0,
            ):
                logging.info(
                    "Skipping existing %s/%s sample for %s at %s/%s/%s sample=%s.",
                    phase,
                    sample_kind,
                    benchmark_name,
                    architecture,
                    memory_mb,
                    timeout_sec,
                    sample_index,
                )
                had_success = True
                failure_count = 0
                continue
            result = trigger.sync_invoke(input_config)
            success = self._record_result(
                result=result,
                phase=phase,
                sample_kind=sample_kind,
                benchmark_name=benchmark_name,
                function_name=function_name,
                architecture=architecture,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                round_index=0,
                sample_index=sample_index,
                concurrency=1,
                idle_gap_sec=0,
            )
            if success:
                had_success = True
                failure_count = 0
            else:
                failure_count += 1
                if failure_count >= self._max_failures_per_target():
                    logging.warning(
                        "Skipping remaining %s samples for %s at %s/%s/%s after %s consecutive failures.",
                        phase,
                        benchmark_name,
                        architecture,
                        memory_mb,
                        timeout_sec,
                        failure_count,
                    )
                    break
        return had_success

    def _record_result(
        self,
        result: Any,
        phase: str,
        sample_kind: str,
        benchmark_name: str,
        function_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        round_index: int,
        sample_index: int,
        concurrency: int,
        idle_gap_sec: int,
    ) -> bool:
        """Persist one invocation result and handle failures consistently."""
        row = self._result_to_row(
            result=result,
            phase=phase,
            sample_kind=sample_kind,
            benchmark_name=benchmark_name,
            function_name=function_name,
            architecture=architecture,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            round_index=round_index,
            sample_index=sample_index,
            concurrency=concurrency,
            idle_gap_sec=idle_gap_sec,
        )
        self._append_row(row)
        latency_ms = row["client_latency_ms"]
        latency_text = (
            f"{float(latency_ms):.3f}"
            if latency_ms is not None and not math.isnan(float(latency_ms))
            else "nan"
        )
        log_message = (
            "Invocation result: benchmark=%s phase=%s sample_kind=%s arch=%s "
            "memory=%s timeout=%s round=%s sample=%s concurrency=%s idle_gap=%s "
            "success=%s cold=%s latency_ms=%s"
        )
        logging.info(
            log_message,
            benchmark_name,
            phase,
            sample_kind,
            architecture,
            memory_mb,
            timeout_sec,
            round_index,
            sample_index,
            concurrency,
            idle_gap_sec,
            row["success"],
            row["cold_start"],
            latency_text,
        )

        if row["success"]:
            return True

        message = (
            f"Invocation failed for benchmark={benchmark_name} phase={phase} "
            f"arch={architecture} memory={memory_mb} timeout={timeout_sec} "
            f"sample={sample_index}"
        )
        if not self._continue_on_error():
            raise RuntimeError(message)
        logging.warning(message)
        return False

    def _append_row(self, row: Dict[str, Any]) -> None:
        """Append one invocation row to the raw CSV."""
        with self.raw_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
            writer.writerow(row)
        if row["success"]:
            key = self._build_invocation_key(
                phase=str(row["phase"]),
                sample_kind=str(row["sample_kind"]),
                benchmark_name=str(row["benchmark"]),
                architecture=str(row["architecture"]),
                memory_mb=int(row["memory_mb"]),
                timeout_sec=int(row["timeout_sec"]),
                round_index=int(row["round_index"]),
                sample_index=int(row["sample_index"]),
                concurrency=int(row["concurrency"]),
                idle_gap_sec=int(row["idle_gap_sec"]),
            )
            self._successful_invocation_keys.add(key)
            self._successful_invocation_outcomes[key] = bool(row["cold_start"])

    def _result_to_row(
        self,
        result: Any,
        phase: str,
        sample_kind: str,
        benchmark_name: str,
        function_name: str,
        architecture: str,
        memory_mb: int,
        timeout_sec: int,
        round_index: int,
        sample_index: int,
        concurrency: int,
        idle_gap_sec: int,
    ) -> Dict[str, Any]:
        """Convert a SeBS ExecutionResult into a flat CSV row."""
        output = result.output if isinstance(result.output, dict) else {}
        billed_time = result.billing.billed_time
        billed_ms = billed_time / 1000.0 if billed_time is not None else math.nan
        memory_used_mb = result.stats.memory_used
        benchmark_latency_ms = (
            math.nan if result.stats.failure else result.times.benchmark / 1000.0
        )
        provider_init_ms = (
            math.nan if result.stats.failure else result.provider_times.initialization / 1000.0
        )
        provider_exec_ms = (
            math.nan if result.stats.failure else result.provider_times.execution / 1000.0
        )
        if result.stats.failure:
            billed_ms = math.nan
        return {
            "phase": phase,
            "sample_kind": sample_kind,
            "benchmark": benchmark_name,
            "function_name": function_name,
            "architecture": architecture,
            "memory_mb": memory_mb,
            "timeout_sec": timeout_sec,
            "trigger": self.calibration_cfg["trigger"],
            "round_index": round_index,
            "sample_index": sample_index,
            "concurrency": concurrency,
            "idle_gap_sec": idle_gap_sec,
            "request_id": result.request_id,
            "success": not result.stats.failure,
            "cold_start": bool(result.stats.cold_start),
            "client_latency_ms": result.times.client / 1000.0,
            "benchmark_latency_ms": benchmark_latency_ms,
            "provider_init_ms": provider_init_ms,
            "provider_exec_ms": provider_exec_ms,
            "billed_time_ms": billed_ms,
            "gb_seconds": result.billing.gb_seconds,
            "memory_used_mb": memory_used_mb,
            "output_begin_ts": output.get("begin"),
            "output_end_ts": output.get("end"),
            "captured_at_ts": time.time(),
        }

    def write_aggregates(self) -> None:
        """Write aggregated calibration profiles from the raw CSV."""
        raw = pd.read_csv(self.raw_csv_path)
        raw = self._normalize_raw_dataframe(raw)
        self._write_warm_profile(raw)
        self._write_cold_profile(raw)
        self._write_burst_profile(raw)

    def _normalize_raw_dataframe(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize CSV-loaded dtypes for downstream aggregation."""
        normalized = raw.copy()
        dedupe_keys = list(DEDUPE_KEY_COLUMNS)
        existing_dedupe_keys = [
            column for column in dedupe_keys if column in normalized.columns
        ]
        if existing_dedupe_keys:
            normalized = normalized.drop_duplicates(
                subset=existing_dedupe_keys,
                keep="last",
            )

        for column in ["success", "cold_start"]:
            if column in normalized.columns:
                normalized[column] = normalized[column].map(
                    lambda value: str(value).strip().lower() in {"true", "1"}
                )

        numeric_columns = [
            "memory_mb",
            "timeout_sec",
            "round_index",
            "sample_index",
            "concurrency",
            "idle_gap_sec",
            "client_latency_ms",
            "benchmark_latency_ms",
            "provider_init_ms",
            "provider_exec_ms",
            "billed_time_ms",
            "gb_seconds",
            "memory_used_mb",
        ]
        for column in numeric_columns:
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized

    def _merge_success_latency_statistics(
        self,
        raw: pd.DataFrame,
        grouped: pd.DataFrame,
        keys: List[str],
        percentile_columns: Dict[str, float],
        extra_aggs: Dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Merge latency statistics computed only from successful invocations."""
        successful = raw[raw["success"]].copy()
        if successful.empty:
            grouped["successful_invocations"] = 0
            grouped["mean_client_latency_ms"] = math.nan
            if extra_aggs:
                for column in extra_aggs:
                    grouped[column] = math.nan
            for column in percentile_columns:
                grouped[column] = math.nan
            return grouped

        agg_kwargs: Dict[str, tuple[str, str]] = {
            "successful_invocations": ("client_latency_ms", "count"),
            "mean_client_latency_ms": ("client_latency_ms", "mean"),
        }
        if extra_aggs:
            for name, source_column in extra_aggs.items():
                agg_kwargs[name] = (source_column, "mean")
        success_grouped = successful.groupby(keys, as_index=False).agg(**agg_kwargs)

        percentile_grouped = successful.groupby(keys)["client_latency_ms"].agg(
            **{
                column_name: (lambda series, ratio=ratio: percentile(series, ratio))
                for column_name, ratio in percentile_columns.items()
            }
        )
        percentile_grouped = percentile_grouped.reset_index()
        grouped = grouped.merge(success_grouped, on=keys, how="left")
        grouped = grouped.merge(percentile_grouped, on=keys, how="left")
        grouped["successful_invocations"] = grouped["successful_invocations"].fillna(0).astype(int)
        return grouped

    def _aggregate_profile_rows(
        self,
        raw: pd.DataFrame,
        keys: List[str],
        percentile_columns: Dict[str, float],
        *,
        grouped_extra_aggs: Dict[str, tuple[str, Any]] | None = None,
        extra_aggs: Dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Aggregate one profile source into grouped rows."""
        if raw.empty:
            return pd.DataFrame(columns=keys)

        agg_kwargs: Dict[str, tuple[str, Any]] = {
            "invocations": ("client_latency_ms", "count"),
            "failed_invocations": ("success", lambda series: int((~series).sum())),
            "success_rate": ("success", "mean"),
        }
        if grouped_extra_aggs:
            agg_kwargs.update(grouped_extra_aggs)

        grouped = raw.groupby(keys, as_index=False).agg(**agg_kwargs)
        return self._merge_success_latency_statistics(
            raw=raw,
            grouped=grouped,
            keys=keys,
            percentile_columns=percentile_columns,
            extra_aggs=extra_aggs,
        )

    def _combine_primary_and_fallback_rows(
        self,
        primary_raw: pd.DataFrame,
        fallback_raw: pd.DataFrame,
        keys: List[str],
        percentile_columns: Dict[str, float],
        *,
        grouped_extra_aggs: Dict[str, tuple[str, Any]] | None = None,
        extra_aggs: Dict[str, str] | None = None,
        fallback_source: str,
    ) -> pd.DataFrame:
        """Prefer measured rows and append fallback-only combinations when needed."""
        primary_grouped = self._aggregate_profile_rows(
            raw=primary_raw,
            keys=keys,
            percentile_columns=percentile_columns,
            grouped_extra_aggs=grouped_extra_aggs,
            extra_aggs=extra_aggs,
        )
        if not primary_grouped.empty:
            primary_grouped["profile_source"] = "measured"

        fallback_candidates = fallback_raw.copy()
        if not primary_grouped.empty and not fallback_candidates.empty:
            existing = primary_grouped[keys].drop_duplicates().assign(_has_primary=True)
            fallback_candidates = fallback_candidates.merge(existing, on=keys, how="left")
            fallback_candidates = fallback_candidates[
                fallback_candidates["_has_primary"].isna()
            ].drop(columns=["_has_primary"])

        fallback_grouped = self._aggregate_profile_rows(
            raw=fallback_candidates,
            keys=keys,
            percentile_columns=percentile_columns,
            grouped_extra_aggs=grouped_extra_aggs,
            extra_aggs=extra_aggs,
        )
        if not fallback_grouped.empty:
            fallback_grouped["profile_source"] = fallback_source

        if primary_grouped.empty:
            combined = fallback_grouped
        elif fallback_grouped.empty:
            combined = primary_grouped
        else:
            combined = pd.concat(
                [primary_grouped, fallback_grouped],
                ignore_index=True,
                sort=False,
            )
        if combined.empty:
            return combined
        return combined.sort_values(keys).reset_index(drop=True)

    def _write_warm_profile(self, raw: pd.DataFrame) -> None:
        """Aggregate warm-phase measurements."""
        warm = raw[(raw["phase"] == "warm") & (raw["sample_kind"] == "measured")].copy()
        warmup_fallback = raw[
            (raw["phase"] == "warm") & (raw["sample_kind"] == "warmup")
        ].copy()
        if warm.empty and warmup_fallback.empty:
            return

        keys = ["benchmark", "architecture", "memory_mb", "timeout_sec"]
        grouped = self._combine_primary_and_fallback_rows(
            primary_raw=warm,
            fallback_raw=warmup_fallback,
            keys=keys,
            percentile_columns={
                "p50_latency_ms": 0.50,
                "p95_latency_ms": 0.95,
                "p99_latency_ms": 0.99,
            },
            grouped_extra_aggs={"cold_rate": ("cold_start", "mean")},
            extra_aggs={"mean_benchmark_latency_ms": "benchmark_latency_ms"},
            fallback_source="warmup_only",
        )
        grouped.to_csv(self.output_dir / "warm_profile.csv", index=False)

    def _write_cold_profile(self, raw: pd.DataFrame) -> None:
        """Aggregate cold-phase idle-gap measurements."""
        cold = raw[(raw["phase"] == "cold") & (raw["sample_kind"] == "measured")].copy()
        cold_warmup_fallback = raw[
            (raw["phase"] == "cold") & (raw["sample_kind"] == "warmup")
        ].copy()
        if cold.empty and cold_warmup_fallback.empty:
            return

        warm_profile_path = self.output_dir / "warm_profile.csv"
        warm = pd.read_csv(warm_profile_path) if warm_profile_path.exists() else pd.DataFrame()

        keys = ["benchmark", "architecture", "memory_mb", "timeout_sec", "idle_gap_sec"]
        grouped = self._combine_primary_and_fallback_rows(
            primary_raw=cold,
            fallback_raw=cold_warmup_fallback,
            keys=keys,
            percentile_columns={
                "p50_latency_ms": 0.50,
                "p95_latency_ms": 0.95,
            },
            grouped_extra_aggs={"cold_probability": ("cold_start", "mean")},
            fallback_source="warmup_only",
        )
        if not warm.empty:
            warm = warm.rename(
                columns={
                    "mean_client_latency_ms": "warm_mean_client_latency_ms",
                    "p95_latency_ms": "warm_p95_latency_ms",
                }
            )
            grouped = self._attach_nearest_warm_reference(
                grouped=grouped,
                warm=warm,
                metric_columns=[
                    "warm_mean_client_latency_ms",
                    "warm_p95_latency_ms",
                ],
            )
            grouped["estimated_cold_overhead_ms"] = (
                grouped["mean_client_latency_ms"] - grouped["warm_mean_client_latency_ms"]
            )
            grouped["estimated_p95_cold_overhead_ms"] = (
                grouped["p95_latency_ms"] - grouped["warm_p95_latency_ms"]
            )
        grouped.to_csv(self.output_dir / "cold_profile.csv", index=False)

    def _write_burst_profile(self, raw: pd.DataFrame) -> None:
        """Aggregate burst-phase concurrent measurements."""
        burst = raw[(raw["phase"] == "burst") & (raw["sample_kind"] == "measured")].copy()
        burst_fallback = raw[
            (raw["phase"] == "burst")
            & (raw["sample_kind"].isin(["prewarmup", "prime"]))
        ].copy()
        if burst.empty and burst_fallback.empty:
            return

        warm_profile_path = self.output_dir / "warm_profile.csv"
        warm = pd.read_csv(warm_profile_path) if warm_profile_path.exists() else pd.DataFrame()

        keys = ["benchmark", "architecture", "memory_mb", "timeout_sec", "concurrency"]
        grouped = self._combine_primary_and_fallback_rows(
            primary_raw=burst,
            fallback_raw=burst_fallback,
            keys=keys,
            percentile_columns={
                "p50_latency_ms": 0.50,
                "p95_latency_ms": 0.95,
                "p99_latency_ms": 0.99,
            },
            grouped_extra_aggs={
                "rounds": ("round_index", "nunique"),
                "cold_rate": ("cold_start", "mean"),
            },
            fallback_source="prewarm_only",
        )

        warm_only = burst[burst["success"] & (~burst["cold_start"])].copy()
        if warm_only.empty:
            grouped["warm_only_successful_invocations"] = 0
            grouped["warm_only_mean_client_latency_ms"] = math.nan
            grouped["warm_only_p50_latency_ms"] = math.nan
            grouped["warm_only_p95_latency_ms"] = math.nan
            grouped["warm_only_p99_latency_ms"] = math.nan
        else:
            warm_only_grouped = self._merge_success_latency_statistics(
                raw=warm_only,
                grouped=grouped[keys].drop_duplicates().copy(),
                keys=keys,
                percentile_columns={
                    "warm_only_p50_latency_ms": 0.50,
                    "warm_only_p95_latency_ms": 0.95,
                    "warm_only_p99_latency_ms": 0.99,
                },
            )
            warm_only_grouped = warm_only_grouped.rename(
                columns={
                    "successful_invocations": "warm_only_successful_invocations",
                    "mean_client_latency_ms": "warm_only_mean_client_latency_ms",
                }
            )
            grouped = grouped.merge(warm_only_grouped, on=keys, how="left")
            grouped["warm_only_successful_invocations"] = (
                grouped["warm_only_successful_invocations"].fillna(0).astype(int)
            )

        if not warm.empty:
            warm = warm.rename(
                columns={
                    "mean_client_latency_ms": "warm_mean_client_latency_ms",
                    "p50_latency_ms": "warm_p50_latency_ms",
                    "p95_latency_ms": "warm_p95_latency_ms",
                    "p99_latency_ms": "warm_p99_latency_ms",
                }
            )
            grouped = self._attach_nearest_warm_reference(
                grouped=grouped,
                warm=warm,
                metric_columns=[
                    "warm_mean_client_latency_ms",
                    "warm_p50_latency_ms",
                    "warm_p95_latency_ms",
                    "warm_p99_latency_ms",
                ],
            )
            slowdown_mean_source = grouped["warm_only_mean_client_latency_ms"].where(
                grouped["warm_only_successful_invocations"] > 0,
                grouped["mean_client_latency_ms"],
            )
            slowdown_p50_source = grouped["warm_only_p50_latency_ms"].where(
                grouped["warm_only_successful_invocations"] > 0,
                grouped["p50_latency_ms"],
            )
            slowdown_p95_source = grouped["warm_only_p95_latency_ms"].where(
                grouped["warm_only_successful_invocations"] > 0,
                grouped["p95_latency_ms"],
            )
            slowdown_p99_source = grouped["warm_only_p99_latency_ms"].where(
                grouped["warm_only_successful_invocations"] > 0,
                grouped["p99_latency_ms"],
            )
            grouped["latency_slowdown_mean"] = (
                slowdown_mean_source
                / grouped["warm_mean_client_latency_ms"].clip(lower=1e-6)
            )
            grouped["latency_slowdown_p50"] = (
                slowdown_p50_source
                / grouped["warm_p50_latency_ms"].clip(lower=1e-6)
            )
            grouped["latency_slowdown_p95"] = (
                slowdown_p95_source
                / grouped["warm_p95_latency_ms"].clip(lower=1e-6)
            )
            grouped["latency_slowdown_p99"] = (
                slowdown_p99_source
                / grouped["warm_p99_latency_ms"].clip(lower=1e-6)
            )
        grouped.to_csv(self.output_dir / "burst_profile.csv", index=False)

    def _attach_nearest_warm_reference(
        self,
        grouped: pd.DataFrame,
        warm: pd.DataFrame,
        metric_columns: list[str],
    ) -> pd.DataFrame:
        """Attach warm metrics by nearest configuration when exact rows are missing."""
        if grouped.empty or warm.empty:
            return grouped

        reference_columns = [
            "benchmark",
            "architecture",
            "memory_mb",
            "timeout_sec",
            *metric_columns,
        ]
        warm_reference = warm[reference_columns].copy()

        attached_rows = []
        for row in grouped.itertuples(index=False):
            candidates = warm_reference[
                (warm_reference["benchmark"] == row.benchmark)
                & (warm_reference["architecture"] == row.architecture)
            ]
            if candidates.empty:
                candidates = warm_reference[warm_reference["benchmark"] == row.benchmark]

            if candidates.empty:
                attached_rows.append({column: np.nan for column in metric_columns})
                continue

            distances = (
                (candidates["memory_mb"] - int(row.memory_mb)).abs()
                + 0.1 * (candidates["timeout_sec"] - int(row.timeout_sec)).abs()
            )
            best_match = candidates.iloc[int(distances.argmin())]
            attached_rows.append(
                {
                    column: best_match.get(column, np.nan)
                    for column in metric_columns
                }
            )

        attached = pd.DataFrame(attached_rows)
        return pd.concat([grouped.reset_index(drop=True), attached], axis=1)


def main() -> None:
    """Program entry point."""
    args = parse_args()
    config = load_json(args.config)
    if args.benchmarks:
        config.setdefault("calibration", {})["benchmarks"] = list(args.benchmarks)
    validate_config(config)
    checks = collect_prerequisite_checks(config)

    if args.check_only:
        print(format_prerequisite_report(checks))
        raise SystemExit(0 if all(check["ok"] for check in checks) else 1)

    ensure_prerequisites(config, checks=checks)

    output_dir = args.output_dir or Path(config["calibration"]["output_dir"])
    collector = OpenWhiskCalibrationCollector(
        config=config,
        output_dir=output_dir,
        keep_existing_output=args.keep_existing_output,
    )

    try:
        collector.setup()
        collector.run(args.phases)
    finally:
        collector.teardown()


if __name__ == "__main__":
    main()
