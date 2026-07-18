"""Public command-line interface for CALO experiment reproduction."""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from . import __version__


PAPER_CONFIG = "config/rl_experiments/paper_full48.json"


def _positive_int(value: str) -> int:
    """Parse one positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _seed_list(value: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated seed list."""
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return parsed


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    """Add shared configuration and diagnostic options."""
    parser.add_argument(
        "--config",
        default=PAPER_CONFIG,
        help="Experiment JSON path, resolved from the repository root.",
    )
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help="Use the heuristic simulator as an explicitly labeled diagnostic.",
    )
    parser.add_argument(
        "--offline-model",
        action="store_true",
        help="Require the pinned CodeBERT revision to exist in the local cache.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free top-level parser."""
    parser = argparse.ArgumentParser(
        prog="calo",
        description="Reproduce and verify the accepted CALO TCC experiments.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    smoke = subparsers.add_parser(
        "smoke", help="Run the calibrated benchmark/workload smoke matrix."
    )
    _add_config_options(smoke)
    smoke.add_argument(
        "--steps",
        type=_positive_int,
        default=5,
        help="Control steps per benchmark/workload case.",
    )
    smoke.add_argument("--seed", type=int, default=None)
    smoke.set_defaults(handler=_dispatch_smoke)

    run = subparsers.add_parser("run", help="Run one configured experiment seed.")
    _add_config_options(run)
    run.add_argument(
        "--output-dir",
        default="outputs/paper_seed_42",
        help="Result directory, resolved from the repository root.",
    )
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--resume", action="store_true", help="Resume compatible partial state.")
    run.set_defaults(handler=_dispatch_run)

    suite = subparsers.add_parser("suite", help="Run configured seeds and aggregate their results.")
    _add_config_options(suite)
    suite.add_argument(
        "--output-dir",
        default="outputs/paper_suite",
        help="Suite directory, resolved from the repository root.",
    )
    suite.add_argument(
        "--seeds",
        type=_seed_list,
        default=None,
        help="Optional comma-separated seed override, for example 42,52,62.",
    )
    suite.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible partial state in each seed directory.",
    )
    suite.set_defaults(handler=_dispatch_suite)

    aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate completed run directories or result JSON files."
    )
    aggregate.add_argument("results", nargs="+", help="Completed result directories or JSON paths.")
    aggregate.add_argument(
        "--output-dir",
        default="results/dynamic_experiment_aggregate",
        help="Directory for aggregate JSON and Markdown.",
    )
    aggregate.set_defaults(handler=_dispatch_aggregate)

    verify = subparsers.add_parser("verify", help="Verify checksums and accepted-paper claims.")
    verify.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Released artifact directory, resolved from the repository root.",
    )
    verify.add_argument(
        "--measure-inference",
        action="store_true",
        help="Append a local 10,000-forward CPU policy-shape measurement.",
    )
    verify.set_defaults(handler=_dispatch_verify)
    return parser


def _load_config(args: argparse.Namespace) -> Any:
    """Load a validated protocol only after a config-backed command is selected."""
    from .experiment_config import ExperimentConfig

    return ExperimentConfig.load(
        args.config,
        disable_calibration=bool(args.disable_calibration),
        offline_model=bool(args.offline_model),
    )


def _dispatch_smoke(args: argparse.Namespace) -> int:
    """Execute the smoke command."""
    from .experiment_runner import ExperimentRunner

    config = _load_config(args)
    results = ExperimentRunner(config, emit=print).run_smoke(steps=args.steps, seed=args.seed)
    calibration = "heuristic diagnostic" if args.disable_calibration else "calibrated"
    print(
        f"Smoke passed: {len(results)} cases; mode={calibration}; "
        f"observation={results[0]['observation_dim']}; "
        f"actions={results[0]['action_count']}."
    )
    return 0


def _dispatch_run(args: argparse.Namespace) -> int:
    """Execute one seed."""
    from .experiment_runner import ExperimentRunner

    config = _load_config(args)
    path = ExperimentRunner(config, emit=print).run_seed(
        args.output_dir, seed=args.seed, resume=args.resume
    )
    print(f"Run completed: {path}")
    return 0


def _dispatch_suite(args: argparse.Namespace) -> int:
    """Execute a multi-seed suite."""
    from .experiment_runner import ExperimentRunner

    config = _load_config(args)
    outputs = ExperimentRunner(config, emit=print).run_suite(
        args.output_dir, seeds=args.seeds, resume=args.resume
    )
    print(f"Suite completed: {outputs.manifest_path}")
    return 0


def _dispatch_aggregate(args: argparse.Namespace) -> int:
    """Aggregate completed result files."""
    from .result_aggregation import ResultAggregator

    outputs = ResultAggregator().aggregate(args.results, args.output_dir)
    print(f"Aggregate JSON: {outputs.json_path}")
    print(f"Aggregate Markdown: {outputs.markdown_path}")
    return 0


def _dispatch_verify(args: argparse.Namespace) -> int:
    """Verify immutable published evidence."""
    from .artifact_verifier import format_verification_report, verify_artifacts

    report = verify_artifacts(
        artifact_dir=args.artifact_dir,
        measure_inference=bool(args.measure_inference),
    )
    print(format_verification_report(report))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute one command, and return a process exit status."""
    parser = build_parser()
    arguments = list(argv) if argv is not None else None
    if arguments is not None and not arguments:
        parser.print_help()
        return 0
    if arguments is None and len(sys.argv) == 1:
        parser.print_help()
        return 0
    args = parser.parse_args(arguments)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"calo: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
