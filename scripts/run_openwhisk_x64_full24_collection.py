#!/usr/bin/env python3
"""Orchestrate reusable OpenWhisk calibration collection profiles."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / "python-venv" / "bin" / "python"
DEFAULT_X64_OUTPUT_DIR = (
    REPO_ROOT / "results" / "openwhisk_calibration_x64_stage2_prime"
)
DEFAULT_ARM64_OUTPUT_DIR = (
    REPO_ROOT / "results" / "openwhisk_calibration_arm64_stage1"
)


@dataclass(frozen=True)
class Stage:
    """One collection stage."""

    name: str
    config_path: Path
    phases: tuple[str, ...]


@dataclass(frozen=True)
class CollectionProfile:
    """One named collection profile."""

    stages: tuple[Stage, ...]
    default_output_dir: Path
    expected_warm: int
    expected_cold: int
    expected_burst: int


COLLECTION_PROFILES: dict[str, CollectionProfile] = {
    "full24": CollectionProfile(
        stages=(
            Stage(
                name="warm",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_full24_warm.json",
                phases=("warm",),
            ),
            Stage(
                name="timeout60",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_full24_timeout60.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="timeout120_lowmem",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_full24_timeout120_lowmem.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="timeout300_lowmem",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_full24_timeout300_lowmem.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="timeout900",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_full24_timeout900.json",
                phases=("cold", "burst"),
            ),
        ),
        default_output_dir=DEFAULT_X64_OUTPUT_DIR,
        expected_warm=120,
        expected_cold=120,
        expected_burst=120,
    ),
    "missing-retry": CollectionProfile(
        stages=(
            Stage(
                name="warm_retry",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_retry_missing_warm.json",
                phases=("warm",),
            ),
            Stage(
                name="timeout60_retry",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_retry_missing_timeout60.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="timeout120_retry",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_retry_missing_timeout120.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="timeout300_retry",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_retry_missing_timeout300.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="timeout900_retry",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_x64_retry_missing_timeout900.json",
                phases=("cold", "burst"),
            ),
        ),
        default_output_dir=DEFAULT_X64_OUTPUT_DIR,
        expected_warm=120,
        expected_cold=120,
        expected_burst=120,
    ),
    "arm64-main8-warm": CollectionProfile(
        stages=(
            Stage(
                name="main8_warm",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_main8_warm.json",
                phases=("warm",),
            ),
        ),
        default_output_dir=DEFAULT_ARM64_OUTPUT_DIR,
        expected_warm=40,
        expected_cold=0,
        expected_burst=0,
    ),
    "arm64-411-full24": CollectionProfile(
        stages=(
            Stage(
                name="411_full24_warm",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_warm.json",
                phases=("warm",),
            ),
            Stage(
                name="411_timeout60",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout60.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="411_timeout120",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout120.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="411_timeout300",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout300.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="411_timeout900",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout900.json",
                phases=("cold", "burst"),
            ),
        ),
        default_output_dir=DEFAULT_ARM64_OUTPUT_DIR,
        expected_warm=24,
        expected_cold=24,
        expected_burst=24,
    ),
    "arm64-minimal": CollectionProfile(
        stages=(
            Stage(
                name="main8_warm",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_main8_warm.json",
                phases=("warm",),
            ),
            Stage(
                name="411_full24_warm",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_warm.json",
                phases=("warm",),
            ),
            Stage(
                name="411_timeout60",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout60.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="411_timeout120",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout120.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="411_timeout300",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout300.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="411_timeout900",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_411_full24_timeout900.json",
                phases=("cold", "burst"),
            ),
        ),
        default_output_dir=DEFAULT_ARM64_OUTPUT_DIR,
        expected_warm=56,
        expected_cold=24,
        expected_burst=24,
    ),
    "arm64-main4-full24": CollectionProfile(
        stages=(
            Stage(
                name="main4_full24_warm",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_main4_full24_warm.json",
                phases=("warm",),
            ),
            Stage(
                name="main4_timeout60",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_main4_full24_timeout60.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="main4_timeout120",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_main4_full24_timeout120.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="main4_timeout300",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_main4_full24_timeout300.json",
                phases=("cold", "burst"),
            ),
            Stage(
                name="main4_timeout900",
                config_path=REPO_ROOT
                / "config"
                / "calibration"
                / "openwhisk_calibration_arm64_main4_full24_timeout900.json",
                phases=("cold", "burst"),
            ),
        ),
        default_output_dir=DEFAULT_ARM64_OUTPUT_DIR,
        expected_warm=120,
        expected_cold=120,
        expected_burst=120,
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one OpenWhisk calibration collection profile."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(COLLECTION_PROFILES),
        default="full24",
        help="Named stage profile to execute.",
    )
    parser.add_argument(
        "--python-exec",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python interpreter used to invoke collect_openwhisk_calibration.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Coverage/log directory. Defaults to the selected profile output dir.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for orchestrator stage logs. Defaults to profile-specific subdir.",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Stage name to start from. Defaults to the first stage in the profile.",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        help="Stage name to stop after. Defaults to the last stage in the profile.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def slice_stages(
    stages: Sequence[Stage],
    start_at: str | None,
    stop_after: str | None,
) -> Sequence[Stage]:
    """Return the selected contiguous stage slice."""
    stage_names = [stage.name for stage in stages]
    selected_start = start_at or stage_names[0]
    selected_stop = stop_after or stage_names[-1]
    if selected_start not in stage_names:
        raise ValueError(f"Unknown start stage: {selected_start}")
    if selected_stop not in stage_names:
        raise ValueError(f"Unknown stop stage: {selected_stop}")
    start_index = stage_names.index(selected_start)
    stop_index = stage_names.index(selected_stop)
    if start_index > stop_index:
        raise ValueError("--start-at must not come after --stop-after")
    return stages[start_index : stop_index + 1]


def build_command(python_exec: Path, stage: Stage) -> list[str]:
    """Build the child command for one stage."""
    return [
        str(python_exec),
        "scripts/collect_openwhisk_calibration.py",
        "--config",
        str(stage.config_path.relative_to(REPO_ROOT)),
        "--keep-existing-output",
        "--phases",
        *stage.phases,
    ]


def run_stage(command: Sequence[str], log_path: Path) -> int:
    """Run one stage while streaming output to both terminal and log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()
        except KeyboardInterrupt:
            process.terminate()
            raise
        return process.wait()


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV file if present."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_combo_count(rows: Iterable[dict[str, str]]) -> int:
    """Count unique benchmark/configuration tuples."""
    combos = set()
    for row in rows:
        benchmark = row.get("benchmark")
        architecture = row.get("architecture")
        memory_mb = row.get("memory_mb")
        timeout_sec = row.get("timeout_sec")
        if not benchmark or not architecture or not memory_mb or not timeout_sec:
            continue
        combos.add(
            (
                benchmark,
                architecture,
                int(float(memory_mb)),
                int(float(timeout_sec)),
            )
        )
    return len(combos)


def print_coverage(
    output_dir: Path,
    expected_warm: int,
    expected_cold: int,
    expected_burst: int,
) -> None:
    """Print current aggregated coverage for warm/cold/burst tables."""
    warm_rows = _read_csv(output_dir / "warm_profile.csv")
    cold_rows = _read_csv(output_dir / "cold_profile.csv")
    burst_rows = _read_csv(output_dir / "burst_profile.csv")
    warm_count = _unique_combo_count(warm_rows)
    cold_count = _unique_combo_count(cold_rows)
    burst_count = _unique_combo_count(burst_rows)
    print(
        "[SUMMARY] coverage warm/cold/burst = "
        f"{warm_count}/{expected_warm}, {cold_count}/{expected_cold}, "
        f"{burst_count}/{expected_burst}"
    )
    print(
        "[SUMMARY] remaining warm/cold/burst = "
        f"{max(expected_warm - warm_count, 0)}, "
        f"{max(expected_cold - cold_count, 0)}, "
        f"{max(expected_burst - burst_count, 0)}"
    )


def main() -> int:
    """Program entry point."""
    args = parse_args()
    profile = COLLECTION_PROFILES[args.profile]
    stages = slice_stages(profile.stages, args.start_at, args.stop_after)
    output_dir = args.output_dir or profile.default_output_dir
    log_dir = args.log_dir or (output_dir / "orchestrator_logs" / args.profile)

    if not args.python_exec.exists():
        print(f"[ERROR] Python interpreter not found: {args.python_exec}", file=sys.stderr)
        return 1

    print(f"[INFO] Selected profile: {args.profile}")
    print("[INFO] Selected stages:", ", ".join(stage.name for stage in stages))
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Log directory: {log_dir}")
    print_coverage(
        output_dir=output_dir,
        expected_warm=profile.expected_warm,
        expected_cold=profile.expected_cold,
        expected_burst=profile.expected_burst,
    )

    for stage in stages:
        command = build_command(args.python_exec, stage)
        print("")
        print(f"[INFO] Starting stage: {stage.name}")
        print("[INFO] Command:", " ".join(command))
        if args.dry_run:
            continue
        stage_log = log_dir / f"{stage.name}.log"
        return_code = run_stage(command=command, log_path=stage_log)
        print(f"[INFO] Stage {stage.name} finished with code {return_code}")
        print_coverage(
            output_dir=output_dir,
            expected_warm=profile.expected_warm,
            expected_cold=profile.expected_cold,
            expected_burst=profile.expected_burst,
        )
        if return_code != 0:
            print(
                f"[ERROR] Stage {stage.name} failed. See log: {stage_log}",
                file=sys.stderr,
            )
            return return_code

    print(f"[INFO] Collection profile '{args.profile}' completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
