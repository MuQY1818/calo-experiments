#!/usr/bin/env python3
"""Train LightGBM service surrogates from real OpenWhisk calibration data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LightGBM-based warm/cold/burst service surrogates."
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("results/openwhisk_calibration_x64_stage2_prime"),
        help="Directory containing warm/cold/burst calibration CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store trained LightGBM artifacts. Defaults to <calibration-dir>/lightgbm_service.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout ratio for quick regression metrics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test splitting.",
    )
    return parser.parse_args()


def build_pipeline(categorical_features: Iterable[str], numeric_features: Iterable[str]) -> Pipeline:
    """Create a LightGBM regression pipeline with one-hot encoding."""
    categorical_features = list(categorical_features)
    numeric_features = list(numeric_features)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )
    regressor = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )


def prepare_warm_dataset(calibration_dir: Path) -> pd.DataFrame:
    """Load successful warm-like raw rows plus aggregated p95 references."""
    raw = pd.read_csv(calibration_dir / "calibration_invocations.csv")
    raw = raw[
        raw["success"]
        & (~raw["cold_start"])
        & (raw["concurrency"] == 1)
        & (raw["sample_kind"] == "measured")
        & (raw["phase"].isin(["warm", "cold"]))
    ].copy()
    return raw[["benchmark", "architecture", "memory_mb", "timeout_sec", "client_latency_ms"]].copy()


def prepare_warm_p95_dataset(calibration_dir: Path) -> pd.DataFrame:
    """Load aggregated warm rows for p95 training."""
    warm = pd.read_csv(calibration_dir / "warm_profile.csv")
    return warm[["benchmark", "architecture", "memory_mb", "timeout_sec", "p95_latency_ms"]].copy()


def prepare_cold_dataset(calibration_dir: Path) -> pd.DataFrame:
    """Load successful true cold-start raw rows and derive overhead targets."""
    raw = pd.read_csv(calibration_dir / "calibration_invocations.csv")
    raw = raw[
        raw["success"]
        & raw["cold_start"]
        & (raw["phase"] == "cold")
        & (raw["sample_kind"] == "measured")
    ].copy()
    warm = pd.read_csv(calibration_dir / "warm_profile.csv")[
        ["benchmark", "architecture", "memory_mb", "timeout_sec", "mean_client_latency_ms"]
    ].rename(columns={"mean_client_latency_ms": "warm_mean_client_latency_ms"})
    merged = raw.merge(
        warm,
        on=["benchmark", "architecture", "memory_mb", "timeout_sec"],
        how="left",
    )
    merged["estimated_cold_overhead_ms"] = (
        merged["client_latency_ms"] - merged["warm_mean_client_latency_ms"]
    ).clip(lower=1.0)
    return merged[
        [
            "benchmark",
            "architecture",
            "memory_mb",
            "timeout_sec",
            "estimated_cold_overhead_ms",
        ]
    ].copy()


def prepare_cold_p95_dataset(calibration_dir: Path) -> pd.DataFrame:
    """Load aggregated enforced cold rows for p95 overhead training."""
    cold = pd.read_csv(calibration_dir / "cold_profile.csv")
    cold = cold[cold["idle_gap_sec"] < 0].copy()
    return cold[
        [
            "benchmark",
            "architecture",
            "memory_mb",
            "timeout_sec",
            "estimated_p95_cold_overhead_ms",
        ]
    ].copy()


def prepare_burst_dataset(calibration_dir: Path) -> pd.DataFrame:
    """Load successful burst rows and derive slowdown targets from warm references."""
    raw = pd.read_csv(calibration_dir / "calibration_invocations.csv")
    raw = raw[
        raw["success"]
        & (~raw["cold_start"])
        & (raw["phase"] == "burst")
        & (raw["sample_kind"] == "measured")
    ].copy()
    warm = pd.read_csv(calibration_dir / "warm_profile.csv")[
        ["benchmark", "architecture", "memory_mb", "timeout_sec", "mean_client_latency_ms"]
    ].rename(columns={"mean_client_latency_ms": "warm_mean_client_latency_ms"})
    merged = raw.merge(
        warm,
        on=["benchmark", "architecture", "memory_mb", "timeout_sec"],
        how="left",
    )
    merged["latency_slowdown_mean"] = (
        merged["client_latency_ms"] / merged["warm_mean_client_latency_ms"].clip(lower=1.0)
    ).clip(lower=1.0)
    return merged[
        [
            "benchmark",
            "architecture",
            "memory_mb",
            "timeout_sec",
            "concurrency",
            "latency_slowdown_mean",
        ]
    ].copy()


def prepare_burst_p95_dataset(calibration_dir: Path) -> pd.DataFrame:
    """Load aggregated burst slowdown rows for p95 slowdown training."""
    burst = pd.read_csv(calibration_dir / "burst_profile.csv")
    return burst[
        [
            "benchmark",
            "architecture",
            "memory_mb",
            "timeout_sec",
            "concurrency",
            "latency_slowdown_p95",
        ]
    ].copy()


def fit_and_report(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    log_target: bool,
    test_size: float,
    random_state: int,
) -> tuple[Pipeline, Dict[str, float]]:
    """Fit one LightGBM pipeline and compute quick holdout metrics."""
    frame = dataset.dropna(subset=feature_columns + [target_column]).copy()
    if frame.empty:
        raise ValueError(f"No rows available for target column: {target_column}")

    x = frame[feature_columns].copy()
    y = frame[target_column].astype(float).to_numpy()
    y_model = np.log1p(np.clip(y, 0.0, None)) if log_target else y

    x_train, x_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
        x,
        y_model,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    pipeline = build_pipeline(
        categorical_features=["benchmark", "architecture"],
        numeric_features=[column for column in feature_columns if column not in {"benchmark", "architecture"}],
    )
    pipeline.fit(x_train, y_train)

    y_pred_model = pipeline.predict(x_test)
    y_pred = np.expm1(y_pred_model) if log_target else y_pred_model
    metrics = {
        "rows": float(len(frame)),
        "mae": float(mean_absolute_error(y_true_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_test, y_pred))),
    }
    return pipeline, metrics


def save_model(output_dir: Path, filename: str, model: Pipeline) -> None:
    """Persist one sklearn pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / filename)


def main() -> None:
    """Train all service surrogate models and write artifacts to disk."""
    args = parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.calibration_dir / "lightgbm_service"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    warm = prepare_warm_dataset(args.calibration_dir)
    warm_p95 = prepare_warm_p95_dataset(args.calibration_dir)
    cold = prepare_cold_dataset(args.calibration_dir)
    cold_p95 = prepare_cold_p95_dataset(args.calibration_dir)
    burst = prepare_burst_dataset(args.calibration_dir)
    burst_p95 = prepare_burst_p95_dataset(args.calibration_dir)

    metrics: Dict[str, Dict[str, float]] = {}

    warm_features = ["benchmark", "architecture", "memory_mb", "timeout_sec"]
    warm_mean_model, metrics["warm_mean_model"] = fit_and_report(
        warm,
        warm_features,
        "client_latency_ms",
        log_target=True,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    warm_p95_model, metrics["warm_p95_model"] = fit_and_report(
        warm_p95,
        warm_features,
        "p95_latency_ms",
        log_target=True,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(output_dir, "warm_mean_model.joblib", warm_mean_model)
    save_model(output_dir, "warm_p95_model.joblib", warm_p95_model)

    cold_features = ["benchmark", "architecture", "memory_mb", "timeout_sec"]
    cold_mean_model, metrics["cold_overhead_mean_model"] = fit_and_report(
        cold,
        cold_features,
        "estimated_cold_overhead_ms",
        log_target=True,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    cold_p95_model, metrics["cold_overhead_p95_model"] = fit_and_report(
        cold_p95,
        cold_features,
        "estimated_p95_cold_overhead_ms",
        log_target=True,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(output_dir, "cold_overhead_mean_model.joblib", cold_mean_model)
    save_model(output_dir, "cold_overhead_p95_model.joblib", cold_p95_model)

    burst_features = ["benchmark", "architecture", "memory_mb", "timeout_sec", "concurrency"]
    burst_mean_model, metrics["burst_slowdown_mean_model"] = fit_and_report(
        burst,
        burst_features,
        "latency_slowdown_mean",
        log_target=True,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    burst_p95_model, metrics["burst_slowdown_p95_model"] = fit_and_report(
        burst_p95,
        burst_features,
        "latency_slowdown_p95",
        log_target=True,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    save_model(output_dir, "burst_slowdown_mean_model.joblib", burst_mean_model)
    save_model(output_dir, "burst_slowdown_p95_model.joblib", burst_p95_model)

    metadata = {
        "calibration_dir": str(args.calibration_dir),
        "output_dir": str(output_dir),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "models": metrics,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
