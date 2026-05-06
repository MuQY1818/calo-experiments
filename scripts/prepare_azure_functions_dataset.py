#!/usr/bin/env python3
"""Prepare Azure Functions public dataset artifacts for CALO experiments.

This script scans the Azure Functions 2019 dataset and generates lightweight
artifacts that are easier to reuse in later simulator or workload studies:

1. Function-level summary across all available days.
2. Minute-of-day invocation profiles for the top-K most active functions.
3. A metadata JSON file describing the processed inputs and known caveats.

The Azure Functions 2019 memory files are app-level rather than function-level,
so they are not merged into the function summary in this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List

import numpy as np
import pandas as pd


ID_COLUMNS = ["HashOwner", "HashApp", "HashFunction"]
MINUTE_COLUMNS = [str(index) for index in range(1, 1441)]
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Azure Functions 2019 workload summaries."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("external_data/azure_functions/raw/dataset2019"),
        help="Directory containing the extracted Azure Functions 2019 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("external_data/azure_functions/processed"),
        help="Directory for processed CSV/JSON outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=200,
        help="Number of most active functions to keep for minute-level profiles.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=512,
        help="Chunk size used when scanning wide invocation CSV files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def build_function_key(frame: pd.DataFrame) -> pd.Series:
    """Builds a stable function key from the anonymized identifiers."""
    return (
        frame["HashOwner"].astype(str)
        + "|"
        + frame["HashApp"].astype(str)
        + "|"
        + frame["HashFunction"].astype(str)
    )


def list_matching_files(raw_dir: Path, pattern: str) -> List[Path]:
    """Returns sorted files that match a pattern under the raw directory."""
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {pattern} under {raw_dir}")
    return files


def summarize_invocation_files(
    invocation_files: Iterable[Path], chunksize: int
) -> pd.DataFrame:
    """Aggregates function-level invocation statistics across days."""
    del chunksize
    invocation_files = list(invocation_files)
    aggregates: Dict[str, Dict[str, float]] = {}

    for file_path in invocation_files:
        logging.info("Scanning invocation file %s", file_path.name)
        with file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[:4] != ID_COLUMNS + ["Trigger"] or header[4:] != MINUTE_COLUMNS:
                raise ValueError(f"Unexpected header layout in {file_path}")

            for row in reader:
                minute_values = np.asarray(row[4:], dtype=np.float64)
                total_invocations = float(minute_values.sum())
                mean_invocations = total_invocations / len(MINUTE_COLUMNS)
                std_invocations = float(minute_values.std())
                peak_invocations = float(minute_values.max())
                zero_fraction = float((minute_values == 0).mean())
                active_fraction = 1.0 - zero_fraction
                burstiness = peak_invocations / max(mean_invocations, EPSILON)
                function_key = "|".join(row[:3])

                aggregate = aggregates.setdefault(
                    function_key,
                    {
                        "HashOwner": row[0],
                        "HashApp": row[1],
                        "HashFunction": row[2],
                        "function_key": function_key,
                        "trigger": row[3],
                        "days_observed": 0,
                        "total_invocations": 0.0,
                        "daily_total_sum": 0.0,
                        "daily_total_sq_sum": 0.0,
                        "max_daily_invocations": 0.0,
                        "mean_invocations_sum": 0.0,
                        "std_invocations_sum": 0.0,
                        "peak_invocations_per_minute": 0.0,
                        "zero_fraction_sum": 0.0,
                        "active_fraction_sum": 0.0,
                        "burstiness_sum": 0.0,
                    },
                )

                aggregate["days_observed"] += 1
                aggregate["total_invocations"] += total_invocations
                aggregate["daily_total_sum"] += total_invocations
                aggregate["daily_total_sq_sum"] += total_invocations ** 2
                aggregate["max_daily_invocations"] = max(
                    aggregate["max_daily_invocations"], total_invocations
                )
                aggregate["mean_invocations_sum"] += mean_invocations
                aggregate["std_invocations_sum"] += std_invocations
                aggregate["peak_invocations_per_minute"] = max(
                    aggregate["peak_invocations_per_minute"], peak_invocations
                )
                aggregate["zero_fraction_sum"] += zero_fraction
                aggregate["active_fraction_sum"] += active_fraction
                aggregate["burstiness_sum"] += burstiness

    records: List[Dict[str, float]] = []
    for aggregate in aggregates.values():
        days_observed = max(int(aggregate["days_observed"]), 1)
        avg_invocations_per_day = aggregate["daily_total_sum"] / days_observed
        variance = (
            aggregate["daily_total_sq_sum"] / days_observed
            - avg_invocations_per_day ** 2
        )
        std_invocations_per_day = float(np.sqrt(max(variance, 0.0)))
        records.append(
            {
                "HashOwner": aggregate["HashOwner"],
                "HashApp": aggregate["HashApp"],
                "HashFunction": aggregate["HashFunction"],
                "function_key": aggregate["function_key"],
                "trigger": aggregate["trigger"],
                "days_observed": days_observed,
                "total_invocations": aggregate["total_invocations"],
                "avg_invocations_per_day": avg_invocations_per_day,
                "std_invocations_per_day": std_invocations_per_day,
                "max_daily_invocations": aggregate["max_daily_invocations"],
                "avg_invocations_per_minute": aggregate["mean_invocations_sum"]
                / days_observed,
                "std_invocations_per_minute": aggregate["std_invocations_sum"]
                / days_observed,
                "peak_invocations_per_minute": aggregate["peak_invocations_per_minute"],
                "mean_zero_minute_fraction": aggregate["zero_fraction_sum"]
                / days_observed,
                "mean_active_minute_fraction": aggregate["active_fraction_sum"]
                / days_observed,
                "mean_burstiness": aggregate["burstiness_sum"] / days_observed,
                "daily_invocation_cv": std_invocations_per_day
                / max(avg_invocations_per_day, EPSILON),
                "observed_day_fraction": days_observed / len(invocation_files),
            }
        )

    return pd.DataFrame.from_records(records).sort_values(
        "total_invocations", ascending=False
    ).reset_index(drop=True)


def summarize_duration_files(duration_files: Iterable[Path]) -> pd.DataFrame:
    """Aggregates function duration percentiles across days with count weights."""
    aggregates: Dict[str, Dict[str, float]] = {}

    for file_path in duration_files:
        logging.info("Scanning duration file %s", file_path.name)
        frame = pd.read_csv(file_path)
        frame["function_key"] = build_function_key(frame)
        for row in frame.itertuples(index=False):
            aggregate = aggregates.setdefault(
                row.function_key,
                {
                    "HashOwner": row.HashOwner,
                    "HashApp": row.HashApp,
                    "HashFunction": row.HashFunction,
                    "function_key": row.function_key,
                    "duration_days_observed": 0,
                    "duration_sample_count": 0.0,
                    "min_duration_ms": float("inf"),
                    "max_duration_ms": 0.0,
                    "avg_duration_numerator": 0.0,
                    "p50_duration_numerator": 0.0,
                    "p99_duration_numerator": 0.0,
                },
            )
            aggregate["duration_days_observed"] += 1
            aggregate["duration_sample_count"] += row.Count
            aggregate["min_duration_ms"] = min(aggregate["min_duration_ms"], row.Minimum)
            aggregate["max_duration_ms"] = max(aggregate["max_duration_ms"], row.Maximum)
            aggregate["avg_duration_numerator"] += row.Average * row.Count
            aggregate["p50_duration_numerator"] += row.percentile_Average_50 * row.Count
            aggregate["p99_duration_numerator"] += row.percentile_Average_99 * row.Count

    records: List[Dict[str, float]] = []
    for aggregate in aggregates.values():
        count = max(float(aggregate["duration_sample_count"]), EPSILON)
        records.append(
            {
                "HashOwner": aggregate["HashOwner"],
                "HashApp": aggregate["HashApp"],
                "HashFunction": aggregate["HashFunction"],
                "function_key": aggregate["function_key"],
                "duration_days_observed": int(aggregate["duration_days_observed"]),
                "duration_sample_count": aggregate["duration_sample_count"],
                "min_duration_ms": aggregate["min_duration_ms"],
                "max_duration_ms": aggregate["max_duration_ms"],
                "mean_duration_ms": aggregate["avg_duration_numerator"] / count,
                "p50_duration_ms": aggregate["p50_duration_numerator"] / count,
                "p99_duration_ms": aggregate["p99_duration_numerator"] / count,
            }
        )

    return pd.DataFrame.from_records(records)


def build_topk_profiles(
    invocation_files: Iterable[Path],
    top_functions: pd.DataFrame,
    chunksize: int,
) -> pd.DataFrame:
    """Builds minute-of-day profiles for the top-K most active functions."""
    del chunksize
    top_functions = top_functions.copy()
    top_functions["activity_rank"] = np.arange(1, len(top_functions) + 1)
    top_keys = set(top_functions["function_key"].tolist())
    rank_map = dict(zip(top_functions["function_key"], top_functions["activity_rank"]))

    profile_store: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    metadata_map: Dict[str, Dict[str, str]] = {}

    for file_path in invocation_files:
        logging.info("Building top-K profile from %s", file_path.name)
        with file_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if header[:4] != ID_COLUMNS + ["Trigger"] or header[4:] != MINUTE_COLUMNS:
                raise ValueError(f"Unexpected header layout in {file_path}")

            for row in reader:
                function_key = "|".join(row[:3])
                if function_key not in top_keys:
                    continue

                minute_values = np.asarray(row[4:], dtype=np.float64)
                profile_store[function_key].append(minute_values)
                if function_key not in metadata_map:
                    metadata_map[function_key] = {
                        "HashOwner": row[0],
                        "HashApp": row[1],
                        "HashFunction": row[2],
                        "trigger": row[3],
                    }

    profile_frames: List[pd.DataFrame] = []
    for function_key, arrays in profile_store.items():
        stacked = np.vstack(arrays)
        metadata = metadata_map[function_key]
        frame = pd.DataFrame(
            {
                "function_key": function_key,
                "activity_rank": rank_map[function_key],
                "HashOwner": metadata["HashOwner"],
                "HashApp": metadata["HashApp"],
                "HashFunction": metadata["HashFunction"],
                "trigger": metadata["trigger"],
                "minute_of_day": np.arange(1, len(MINUTE_COLUMNS) + 1),
                "mean_invocations": stacked.mean(axis=0),
                "std_invocations": stacked.std(axis=0),
                "max_invocations": stacked.max(axis=0),
                "active_day_fraction": (stacked > 0).mean(axis=0),
                "days_observed": stacked.shape[0],
            }
        )
        profile_frames.append(frame)

    if not profile_frames:
        return pd.DataFrame()

    return pd.concat(profile_frames, ignore_index=True).sort_values(
        ["activity_rank", "minute_of_day"]
    )


def build_metadata(
    raw_dir: Path,
    output_dir: Path,
    invocation_files: List[Path],
    duration_files: List[Path],
    function_summary: pd.DataFrame,
    top_k: int,
) -> Dict[str, object]:
    """Constructs metadata for the generated artifacts."""
    return {
        "dataset": "Azure Functions Dataset 2019",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "invocation_files": [file_path.name for file_path in invocation_files],
        "duration_files": [file_path.name for file_path in duration_files],
        "function_count": int(len(function_summary)),
        "top_k_requested": int(top_k),
        "top_k_generated": int(min(top_k, len(function_summary))),
        "notes": [
            "Execution durations in Azure Functions 2019 do not include cold-start time.",
            "App memory files are app-level and are not merged into the function-level summary.",
            (
                "The 2021 invocation trace archive may require additional extraction "
                "tooling in this environment."
            ),
        ],
    }


def main() -> None:
    """Runs the dataset preparation workflow."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
    )

    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    invocation_files = list_matching_files(raw_dir, "invocations_per_function_md.anon.d*.csv")
    duration_files = list_matching_files(raw_dir, "function_durations_percentiles.anon.d*.csv")

    logging.info("Processing Azure Functions dataset under %s", raw_dir)
    function_summary = summarize_invocation_files(invocation_files, args.chunksize)
    duration_summary = summarize_duration_files(duration_files)
    merged_summary = function_summary.merge(
        duration_summary,
        on=ID_COLUMNS + ["function_key"],
        how="left",
    )

    top_functions = merged_summary.head(args.top_k).copy()
    top_profiles = build_topk_profiles(invocation_files, top_functions, args.chunksize)

    summary_path = output_dir / "azure_functions_2019_function_summary.csv"
    profiles_path = output_dir / "azure_functions_2019_topk_minute_profiles.csv"
    metadata_path = output_dir / "azure_functions_2019_metadata.json"

    merged_summary.to_csv(summary_path, index=False)
    top_profiles.to_csv(profiles_path, index=False)

    metadata = build_metadata(
        raw_dir=raw_dir,
        output_dir=output_dir,
        invocation_files=invocation_files,
        duration_files=duration_files,
        function_summary=merged_summary,
        top_k=args.top_k,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logging.info("Wrote %s", summary_path)
    logging.info("Wrote %s", profiles_path)
    logging.info("Wrote %s", metadata_path)


if __name__ == "__main__":
    main()
