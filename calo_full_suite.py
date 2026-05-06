#!/usr/bin/env python3
"""Run the default CALO simulator suite without generating plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_dynamic_experiment import (
    _load_json,
    _write_json,
    aggregate_experiment_results,
    run_dynamic_experiment,
)


def _default_seeds(config_path: str) -> list[int]:
    """Read the seed list from one experiment config."""
    config = _load_json(Path(config_path))
    seeds = config.get("reproducibility", {}).get("seeds", [42])
    return [int(seed) for seed in seeds]


def _collect_result_paths(output_dir: Path, seeds: list[int]) -> list[str]:
    """Return per-seed result paths in deterministic order."""
    paths = []
    for seed in seeds:
        result_path = output_dir / f"seed_{seed}" / "dynamic_experiment_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing per-seed result file: {result_path}")
        paths.append(str(result_path))
    return paths


def _write_suite_manifest(
    output_dir: Path,
    config_path: str,
    seeds: list[int],
    aggregate_dir: Path,
) -> None:
    """Persist lightweight metadata for the suite run."""
    payload = {
        "config_path": config_path,
        "seeds": seeds,
        "seed_count": len(seeds),
        "per_seed_dirs": [
            str(output_dir / f"seed_{seed}")
            for seed in seeds
        ],
        "aggregate_dir": str(aggregate_dir),
    }
    _write_json(output_dir / "suite_manifest.json", payload)


def run_full_suite(
    config_path: str = "config/rl_experiments/full_suite.json",
    output_dir: str = "outputs/full_suite",
    seeds: list[int] | None = None,
    override_calibration_dir: str | None = None,
    disable_calibration: bool = False,
    resume: bool = False,
) -> None:
    """Run the configured dynamic experiment once per seed and aggregate results."""
    resolved_seeds = seeds if seeds is not None else _default_seeds(config_path)
    suite_output_dir = Path(output_dir)
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("CALO Simulator Suite", flush=True)
    print("=" * 80, flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Output directory: {suite_output_dir}", flush=True)
    print(f"Seeds: {resolved_seeds}", flush=True)
    print("Plot export: disabled", flush=True)

    for seed_index, seed in enumerate(resolved_seeds, start=1):
        seed_output_dir = suite_output_dir / f"seed_{seed}"
        print("\n" + "=" * 80, flush=True)
        print(
            f"Seed {seed_index}/{len(resolved_seeds)}: {seed}",
            flush=True,
        )
        print("=" * 80, flush=True)
        np.random.seed(seed)
        run_dynamic_experiment(
            config_path=config_path,
            output_dir=str(seed_output_dir),
            seed=seed,
            override_calibration_dir=override_calibration_dir,
            disable_calibration=disable_calibration,
            resume=resume,
        )

    aggregate_dir = suite_output_dir / "aggregate"
    aggregate_experiment_results(
        result_refs=_collect_result_paths(suite_output_dir, resolved_seeds),
        output_dir=str(aggregate_dir),
    )
    _write_suite_manifest(
        output_dir=suite_output_dir,
        config_path=config_path,
        seeds=resolved_seeds,
        aggregate_dir=aggregate_dir,
    )


def _parse_seed_list(raw_value: str | None) -> list[int] | None:
    """Parse a comma-separated seed list."""
    if raw_value is None:
        return None
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        return None
    return [int(value) for value in values]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the CALO simulator suite and write JSON/Markdown outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/rl_experiments/full_suite.json",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/full_suite",
        help="Directory for suite outputs.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed override, for example 42,52,62.",
    )
    parser.add_argument(
        "--override-calibration-dir",
        type=str,
        default=None,
        help="Override calibration_dir from the config.",
    )
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help="Force the heuristic simulator instead of calibration tables.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted per-seed runs when partial results exist.",
    )

    args = parser.parse_args()
    run_full_suite(
        config_path=args.config,
        output_dir=args.output_dir,
        seeds=_parse_seed_list(args.seeds),
        override_calibration_dir=args.override_calibration_dir,
        disable_calibration=args.disable_calibration,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
