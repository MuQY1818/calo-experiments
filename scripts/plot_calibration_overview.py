#!/usr/bin/env python
"""Generate overview figures for real OpenWhisk calibration data."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_optimizer.service_model import CalibratedServiceModel

BENCHMARK_ORDER = [
    "110.dynamic-html",
    "120.uploader",
    "210.thumbnailer",
    "311.compression",
    "411.image-recognition",
]
BENCHMARK_LABELS = {
    "110.dynamic-html": "Dynamic HTML",
    "120.uploader": "Uploader",
    "210.thumbnailer": "Thumbnailer",
    "311.compression": "Compression",
    "411.image-recognition": "Image Recognition",
}
SHORT_BENCHMARK_LABELS = {
    "110.dynamic-html": "DynHTML",
    "120.uploader": "Uploader",
    "210.thumbnailer": "Thumb",
    "311.compression": "Compress",
    "411.image-recognition": "ImgRec",
}
MEMORY_ORDER = [512, 1024, 2048, 3008]
TIMEOUT_ORDER = [120, 300]
FULL24_MEMORY_ORDER = [128, 256, 512, 1024, 2048, 3008]
FULL24_TIMEOUT_ORDER = [60, 120, 300, 900]
IDLE_GAP_TARGETS = [0, 120, 300, 600, 900]
GROUPED_COLORS = {
    "warm": "#1b9e77",
    "burst": "#7570b3",
    "cold_enforced": "#d95f02",
    "cold_idle_gap": "#66a61e",
}
PRIMING_COLORS = {
    "base": "#d95f02",
    "primed": "#1b9e77",
}
PI_COLORS = {
    "primary": "#f2c84b",
    "secondary": "#7ea95a",
    "deep_green": "#0d6b2f",
    "mid_green": "#3a9e49",
    "cool_neutral": "#b7c4c8",
    "warm_neutral": "#ddd8ca",
    "edge": "#101010",
    "grid": "#dadada",
    "axes_face": "#fffdf8",
    "legend_border": "#c9c4b6",
    "fail": "#ece7dc",
}


def configure_pi_style() -> None:
    """Apply a consistent PI-like plotting style."""
    plt.style.use("default")
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.1,
            "axes.edgecolor": PI_COLORS["edge"],
            "axes.facecolor": PI_COLORS["axes_face"],
            "errorbar.capsize": 4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate calibration overview figures from real OpenWhisk CSV tables."
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("results/openwhisk_calibration_x64_stage2_prime"),
        help="Directory containing warm/cold/burst calibration CSV files.",
    )
    parser.add_argument(
        "--burst-base-dir",
        type=Path,
        default=Path("results/openwhisk_burst_probe_base"),
        help="Directory containing the non-primed burst probe CSV files.",
    )
    parser.add_argument(
        "--burst-primed-dir",
        type=Path,
        default=Path("results/openwhisk_burst_probe_prime"),
        help="Directory containing the primed burst probe CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures. Defaults to <calibration-dir>/figures.",
    )
    parser.add_argument(
        "--arm64-calibration-dir",
        type=Path,
        default=Path("results/openwhisk_calibration_arm64_stage1"),
        help="Optional ARM64 calibration directory used for architecture comparison.",
    )
    parser.add_argument(
        "--comparison-benchmark",
        type=str,
        default="411.image-recognition",
        help="Benchmark used for the x64 vs ARM64 comparison figure.",
    )
    parser.add_argument(
        "--comparison-burst-concurrency",
        type=int,
        default=2,
        help="Burst concurrency used in the architecture comparison figure.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read one CSV file into a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """Save one figure as both PDF and PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def read_csv_frame(path: Path) -> pd.DataFrame:
    """Read one CSV file into a dataframe."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    return pd.read_csv(path)


def build_coverage_counts(calibration_dir: Path) -> Dict[int, Dict[str, int]]:
    """Count grouped profile rows by timeout and phase."""
    counts = {
        timeout: {
            "warm": 0,
            "burst": 0,
            "cold_enforced": 0,
            "cold_idle_gap": 0,
        }
        for timeout in TIMEOUT_ORDER
    }

    warm_rows = read_csv_rows(calibration_dir / "warm_profile.csv")
    for row in warm_rows:
        timeout = int(float(row["timeout_sec"]))
        if timeout in counts:
            counts[timeout]["warm"] += 1

    burst_rows = read_csv_rows(calibration_dir / "burst_profile.csv")
    for row in burst_rows:
        timeout = int(float(row["timeout_sec"]))
        if timeout in counts:
            counts[timeout]["burst"] += 1

    cold_rows = read_csv_rows(calibration_dir / "cold_profile.csv")
    for row in cold_rows:
        timeout = int(float(row["timeout_sec"]))
        if timeout not in counts:
            continue
        idle_gap = int(float(row["idle_gap_sec"]))
        key = "cold_enforced" if idle_gap < 0 else "cold_idle_gap"
        counts[timeout][key] += 1
    return counts


def build_idle_gap_progress(calibration_dir: Path, timeout_sec: int) -> np.ndarray:
    """Count completed idle-gap points from raw invocation logs."""
    rows = read_csv_rows(calibration_dir / "calibration_invocations.csv")
    seen: Dict[tuple[str, int], set[int]] = defaultdict(set)
    for row in rows:
        if row.get("phase") != "cold":
            continue
        if int(float(row["timeout_sec"])) != timeout_sec:
            continue
        if row.get("sample_kind") != "measured":
            continue
        if row.get("success") != "True":
            continue
        idle_gap = int(float(row["idle_gap_sec"]))
        if idle_gap < 0:
            continue
        key = (row["benchmark"], int(float(row["memory_mb"])))
        seen[key].add(idle_gap)

    matrix = np.zeros((len(BENCHMARK_ORDER), len(MEMORY_ORDER)), dtype=float)
    for row_index, benchmark in enumerate(BENCHMARK_ORDER):
        for col_index, memory_mb in enumerate(MEMORY_ORDER):
            matrix[row_index, col_index] = len(seen.get((benchmark, memory_mb), set()))
    return matrix


def build_ttl_matrix(calibration_dir: Path) -> Dict[int, np.ndarray]:
    """Build TTL estimates from the current service-model logic."""
    model = CalibratedServiceModel(calibration_dir=calibration_dir)
    matrices: Dict[int, np.ndarray] = {}
    for timeout_sec in TIMEOUT_ORDER:
        matrix = np.zeros((len(BENCHMARK_ORDER), len(MEMORY_ORDER)), dtype=float)
        for row_index, benchmark in enumerate(BENCHMARK_ORDER):
            for col_index, memory_mb in enumerate(MEMORY_ORDER):
                matrix[row_index, col_index] = model.estimate_ttl_sec(
                    benchmark=benchmark,
                    memory_mb=memory_mb,
                    timeout_sec=timeout_sec,
                    architecture="x64",
                )
        matrices[timeout_sec] = matrix
    return matrices


def load_burst_probe_rates(directory: Path) -> Dict[str, float]:
    """Load burst cold rates from one burst probe directory."""
    rows = read_csv_rows(directory / "burst_profile.csv")
    values: Dict[str, float] = {}
    for row in rows:
        label = (
            f"{SHORT_BENCHMARK_LABELS.get(row['benchmark'], row['benchmark'])}\n"
            f"{int(float(row['memory_mb']))}MB C{int(float(row['concurrency']))}"
        )
        values[label] = float(row["cold_rate"])
    return values


def build_architecture_comparison_rows(
    x64_calibration_dir: Path,
    arm64_calibration_dir: Path,
    benchmark: str,
    burst_concurrency: int,
) -> pd.DataFrame:
    """Build a normalized x64 vs ARM64 comparison table for one benchmark."""
    row_frames: list[pd.DataFrame] = []
    phase_specs = (
        ("warm", x64_calibration_dir / "warm_profile.csv", None),
        ("cold", x64_calibration_dir / "cold_profile.csv", None),
        ("burst", x64_calibration_dir / "burst_profile.csv", burst_concurrency),
        ("warm", arm64_calibration_dir / "warm_profile.csv", None),
        ("cold", arm64_calibration_dir / "cold_profile.csv", None),
        ("burst", arm64_calibration_dir / "burst_profile.csv", burst_concurrency),
    )

    for phase_name, path, concurrency in phase_specs:
        frame = read_csv_frame(path)
        frame = frame[frame["benchmark"] == benchmark].copy()
        if frame.empty:
            continue
        if phase_name == "cold":
            frame = frame[frame["idle_gap_sec"] == -1].copy()
        if phase_name == "burst":
            frame = frame[frame["concurrency"] == burst_concurrency].copy()
        if frame.empty:
            continue
        frame["phase_name"] = phase_name
        if "concurrency" not in frame.columns:
            frame["concurrency"] = 1
        if "idle_gap_sec" not in frame.columns:
            frame["idle_gap_sec"] = np.nan
        if "mean_client_latency_ms" not in frame.columns:
            frame["mean_client_latency_ms"] = np.nan
        if "warm_only_p50_latency_ms" not in frame.columns:
            frame["warm_only_p50_latency_ms"] = np.nan
        if "latency_slowdown_p50" not in frame.columns:
            frame["latency_slowdown_p50"] = np.nan
        row_frames.append(
            frame[
                [
                    "benchmark",
                    "architecture",
                    "memory_mb",
                    "timeout_sec",
                    "phase_name",
                    "concurrency",
                    "idle_gap_sec",
                    "invocations",
                    "failed_invocations",
                    "success_rate",
                    "successful_invocations",
                    "mean_client_latency_ms",
                    "p50_latency_ms",
                    "p95_latency_ms",
                    "warm_only_p50_latency_ms",
                    "latency_slowdown_p50",
                    "profile_source",
                ]
            ].copy()
        )

    if not row_frames:
        return pd.DataFrame()

    combined = pd.concat(row_frames, ignore_index=True)
    combined["memory_mb"] = combined["memory_mb"].astype(int)
    combined["timeout_sec"] = combined["timeout_sec"].astype(int)
    combined["concurrency"] = combined["concurrency"].astype(int)
    combined["successful_invocations"] = (
        pd.to_numeric(combined["successful_invocations"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    combined["success_rate"] = pd.to_numeric(
        combined["success_rate"], errors="coerce"
    ).fillna(0.0)
    combined["p50_latency_ms"] = pd.to_numeric(
        combined["p50_latency_ms"], errors="coerce"
    )
    combined["p95_latency_ms"] = pd.to_numeric(
        combined["p95_latency_ms"], errors="coerce"
    )
    combined["mean_client_latency_ms"] = pd.to_numeric(
        combined["mean_client_latency_ms"], errors="coerce"
    )
    combined["warm_only_p50_latency_ms"] = pd.to_numeric(
        combined["warm_only_p50_latency_ms"], errors="coerce"
    )
    combined["latency_slowdown_p50"] = pd.to_numeric(
        combined["latency_slowdown_p50"], errors="coerce"
    )
    combined["is_feasible"] = combined["successful_invocations"] > 0
    combined = combined.sort_values(
        ["phase_name", "architecture", "timeout_sec", "memory_mb"]
    ).reset_index(drop=True)
    return combined


def write_architecture_comparison_summary(
    output_dir: Path,
    comparison_rows: pd.DataFrame,
    benchmark: str,
) -> None:
    """Write a reusable architecture-comparison CSV summary."""
    if comparison_rows.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = benchmark.replace(".", "_").replace("-", "_")
    comparison_rows.to_csv(
        output_dir / f"architecture_comparison_{slug}.csv",
        index=False,
    )


def _build_architecture_metric_matrix(
    comparison_rows: pd.DataFrame,
    phase_name: str,
    metric_column: str,
    label_kind: str,
) -> tuple[np.ndarray, list[list[str]], list[str]]:
    """Build one architecture-timeout metric matrix with fail annotations."""
    phase_rows = comparison_rows[comparison_rows["phase_name"] == phase_name].copy()
    row_keys = [
        (architecture, timeout_sec)
        for architecture in ("x64", "arm64")
        for timeout_sec in FULL24_TIMEOUT_ORDER
    ]
    matrix = np.full((len(row_keys), len(FULL24_MEMORY_ORDER)), np.nan, dtype=float)
    labels: list[list[str]] = [
        ["" for _ in FULL24_MEMORY_ORDER] for _ in row_keys
    ]
    row_labels = [f"{architecture} / {timeout_sec}s" for architecture, timeout_sec in row_keys]

    for row_index, (architecture, timeout_sec) in enumerate(row_keys):
        scoped = phase_rows[
            (phase_rows["architecture"] == architecture)
            & (phase_rows["timeout_sec"] == timeout_sec)
        ]
        for col_index, memory_mb in enumerate(FULL24_MEMORY_ORDER):
            match = scoped[scoped["memory_mb"] == memory_mb]
            if match.empty:
                labels[row_index][col_index] = "NA"
                continue
            record = match.iloc[0]
            if metric_column == "success_rate":
                value = float(record["success_rate"])
                matrix[row_index, col_index] = value
                labels[row_index][col_index] = (
                    "fail" if value <= 0.0 else f"{int(round(100.0 * value))}%"
                )
                continue

            value = record.get(metric_column, np.nan)
            if bool(record["is_feasible"]) and not np.isnan(value):
                matrix[row_index, col_index] = float(value)
                if label_kind == "ms":
                    labels[row_index][col_index] = f"{int(round(float(value)))}"
                elif label_kind == "x":
                    labels[row_index][col_index] = f"{float(value):.1f}x"
                else:
                    labels[row_index][col_index] = f"{float(value):.2f}"
            else:
                labels[row_index][col_index] = "fail"
    return matrix, labels, row_labels


def _plot_architecture_metric_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: list[list[str]],
    row_labels: list[str],
    title: str,
    colorbar_label: str,
    cmap_name: str,
    threshold_ratio: float = 0.58,
) -> None:
    """Plot one architecture metric heatmap with fail annotations."""
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(PI_COLORS["fail"])
    valid_values = masked.compressed()
    if valid_values.size == 0:
        norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        norm = Normalize(
            vmin=float(np.nanmin(valid_values)),
            vmax=float(np.nanmax(valid_values)),
        )
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(FULL24_MEMORY_ORDER)))
    ax.set_xticklabels([f"{memory}MB" for memory in FULL24_MEMORY_ORDER])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    ax.set_xlabel("Memory Tier")

    threshold = (
        norm.vmin + threshold_ratio * (norm.vmax - norm.vmin)
        if norm.vmax > norm.vmin
        else norm.vmin
    )
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            label = labels[row_index][col_index]
            value = matrix[row_index, col_index]
            if np.isnan(value):
                color = PI_COLORS["edge"]
            else:
                color = "white" if value >= threshold else PI_COLORS["edge"]
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
                fontweight="semibold",
            )

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["left"].set_color(PI_COLORS["edge"])
    ax.spines["bottom"].set_color(PI_COLORS["edge"])
    ax.tick_params(length=0)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(colorbar_label)


def plot_architecture_comparison(
    output_dir: Path,
    comparison_rows: pd.DataFrame,
    benchmark: str,
    burst_concurrency: int,
) -> None:
    """Plot architecture comparison heatmaps for one benchmark."""
    if comparison_rows.empty:
        return

    configure_pi_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), sharey=True)
    warm_success_matrix, warm_success_labels, row_labels = _build_architecture_metric_matrix(
        comparison_rows=comparison_rows,
        phase_name="warm",
        metric_column="success_rate",
        label_kind="pct",
    )
    burst_success_matrix, burst_success_labels, _ = _build_architecture_metric_matrix(
        comparison_rows=comparison_rows,
        phase_name="burst",
        metric_column="success_rate",
        label_kind="pct",
    )
    warm_latency_matrix, warm_latency_labels, _ = _build_architecture_metric_matrix(
        comparison_rows=comparison_rows,
        phase_name="warm",
        metric_column="p50_latency_ms",
        label_kind="ms",
    )
    burst_slowdown_matrix, burst_slowdown_labels, _ = _build_architecture_metric_matrix(
        comparison_rows=comparison_rows,
        phase_name="burst",
        metric_column="latency_slowdown_p50",
        label_kind="x",
    )

    _plot_architecture_metric_panel(
        axes[0, 0],
        warm_success_matrix,
        warm_success_labels,
        row_labels,
        title="Warm Feasibility",
        colorbar_label="Warm Success Rate",
        cmap_name="YlGn",
        threshold_ratio=0.55,
    )
    _plot_architecture_metric_panel(
        axes[0, 1],
        burst_success_matrix,
        burst_success_labels,
        row_labels,
        title=f"Burst Feasibility (C{burst_concurrency})",
        colorbar_label="Burst Success Rate",
        cmap_name="YlGn",
        threshold_ratio=0.55,
    )
    _plot_architecture_metric_panel(
        axes[1, 0],
        warm_latency_matrix,
        warm_latency_labels,
        row_labels,
        title="Warm p50 Latency",
        colorbar_label="Warm p50 (ms)",
        cmap_name="YlGnBu",
    )
    _plot_architecture_metric_panel(
        axes[1, 1],
        burst_slowdown_matrix,
        burst_slowdown_labels,
        row_labels,
        title=f"Burst Slowdown p50 (C{burst_concurrency})",
        colorbar_label="Burst Slowdown",
        cmap_name="YlOrBr",
        threshold_ratio=0.60,
    )
    axes[0, 0].set_ylabel("Architecture / Timeout")
    axes[1, 0].set_ylabel("Architecture / Timeout")
    for axis in axes[0]:
        axis.tick_params(axis="x", labelbottom=False)
    for axis in axes.flat:
        axis.set_xlabel("Memory Tier")

    fig.subplots_adjust(wspace=0.22, hspace=0.20, bottom=0.11)
    slug = benchmark.replace(".", "_").replace("-", "_")
    save_figure(fig, output_dir, f"architecture_comparison_{slug}")


def annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    labels: Sequence[Sequence[str]],
    threshold: float,
    fmt_color_low: str = "#0f172a",
    fmt_color_high: str = "white",
) -> None:
    """Add centered text annotations to one heatmap."""
    rows, cols = matrix.shape
    for row_index in range(rows):
        for col_index in range(cols):
            value = matrix[row_index, col_index]
            color = fmt_color_high if value >= threshold else fmt_color_low
            ax.text(
                col_index,
                row_index,
                labels[row_index][col_index],
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                fontweight="semibold",
            )


def plot_coverage_by_timeout(output_dir: Path, counts: Dict[int, Dict[str, int]]) -> None:
    """Plot grouped row counts by timeout."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    categories = ["warm", "burst", "cold_enforced", "cold_idle_gap"]
    labels = {
        "warm": "Warm",
        "burst": "Burst",
        "cold_enforced": "Cold Enforced",
        "cold_idle_gap": "Cold Idle-Gap",
    }
    x = np.arange(len(TIMEOUT_ORDER))
    width = 0.17

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for index, category in enumerate(categories):
        offsets = (index - (len(categories) - 1) / 2.0) * width * 1.2
        values = [counts[timeout][category] for timeout in TIMEOUT_ORDER]
        bars = ax.bar(
            x + offsets,
            values,
            width=width,
            label=labels[category],
            color=GROUPED_COLORS[category],
            alpha=0.88,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.5,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Timeout {timeout}s" for timeout in TIMEOUT_ORDER])
    ax.set_ylabel("Grouped Profile Rows")
    ax.set_title("Calibration Coverage by Timeout")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.78))
    fig.subplots_adjust(top=0.88, right=0.80)
    save_figure(fig, output_dir, "calibration_profiles_by_timeout")


def plot_idle_gap_progress(output_dir: Path, matrix: np.ndarray, timeout_sec: int) -> None:
    """Plot current idle-gap progress from raw cold invocations."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=len(IDLE_GAP_TARGETS))
    ax.set_xticks(np.arange(len(MEMORY_ORDER)))
    ax.set_xticklabels([f"{memory}MB" for memory in MEMORY_ORDER])
    ax.set_yticks(np.arange(len(BENCHMARK_ORDER)))
    ax.set_yticklabels([BENCHMARK_LABELS[bench] for bench in BENCHMARK_ORDER])
    ax.set_xlabel("Memory Tier")
    ax.set_ylabel("Benchmark")
    ax.set_title(f"Idle-Gap Progress for Timeout {timeout_sec}s")

    annotation_labels = [
        [f"{int(value)}/{len(IDLE_GAP_TARGETS)}" for value in row]
        for row in matrix
    ]
    annotate_heatmap(ax, matrix, annotation_labels, threshold=3.2)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Completed Idle-Gap Points")
    fig.tight_layout()
    save_figure(fig, output_dir, f"idle_gap_progress_timeout{timeout_sec}")


def plot_ttl_status(output_dir: Path, ttl_matrices: Dict[int, np.ndarray]) -> None:
    """Plot current TTL estimates from the aggregated calibration tables."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(1, len(TIMEOUT_ORDER), figsize=(10.0, 4.5), sharey=True)
    if len(TIMEOUT_ORDER) == 1:
        axes = [axes]

    for axis, timeout_sec in zip(axes, TIMEOUT_ORDER):
        matrix = ttl_matrices[timeout_sec]
        image = axis.imshow(matrix, cmap="viridis", vmin=600, vmax=900)
        axis.set_xticks(np.arange(len(MEMORY_ORDER)))
        axis.set_xticklabels([f"{memory}MB" for memory in MEMORY_ORDER])
        axis.set_yticks(np.arange(len(BENCHMARK_ORDER)))
        axis.set_yticklabels([BENCHMARK_LABELS[bench] for bench in BENCHMARK_ORDER])
        axis.set_xlabel("Memory Tier")
        axis.set_title(f"Timeout {timeout_sec}s")

        labels = [
            [f"{int(value)}" for value in row]
            for row in matrix
        ]
        annotate_heatmap(axis, matrix, labels, threshold=760.0)

    axes[0].set_ylabel("Benchmark")
    fig.suptitle("Estimated TTL from Current Aggregated Profiles", y=0.98)
    fig.subplots_adjust(top=0.82, right=0.88, wspace=0.26)
    cbar_ax = fig.add_axes([0.90, 0.14, 0.022, 0.66])
    colorbar = fig.colorbar(image, cax=cbar_ax)
    colorbar.set_label("Estimated TTL (s)")
    save_figure(fig, output_dir, "ttl_status_x64")


def plot_burst_priming_effect(
    output_dir: Path,
    base_rates: Dict[str, float],
    primed_rates: Dict[str, float],
) -> None:
    """Plot cold-rate reduction from burst priming."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    labels = sorted(set(base_rates) | set(primed_rates))
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    base_values = [base_rates.get(label, np.nan) for label in labels]
    primed_values = [primed_rates.get(label, np.nan) for label in labels]
    base_bars = ax.bar(
        x - width / 2.0,
        base_values,
        width=width,
        color=PRIMING_COLORS["base"],
        alpha=0.88,
        label="No Priming",
    )
    primed_bars = ax.bar(
        x + width / 2.0,
        primed_values,
        width=width,
        color=PRIMING_COLORS["primed"],
        alpha=0.88,
        label="Primed",
    )

    for bars in (base_bars, primed_bars):
        for bar in bars:
            height = float(bar.get_height())
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.012,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=16, ha="right")
    ax.set_ylabel("Measured Cold Rate")
    ax.set_ylim(0.0, 0.32)
    ax.set_title("Effect of Burst Priming on Measured Cold Rate")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir, "burst_priming_effect")


def main() -> None:
    """Entrypoint for calibration-overview plotting."""
    args = parse_args()
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.calibration_dir / "figures"
    )

    coverage_counts = build_coverage_counts(args.calibration_dir)
    idle_gap_progress = build_idle_gap_progress(args.calibration_dir, timeout_sec=120)
    ttl_matrices = build_ttl_matrix(args.calibration_dir)
    base_rates = load_burst_probe_rates(args.burst_base_dir)
    primed_rates = load_burst_probe_rates(args.burst_primed_dir)

    plot_coverage_by_timeout(output_dir, coverage_counts)
    plot_idle_gap_progress(output_dir, idle_gap_progress, timeout_sec=120)
    plot_ttl_status(output_dir, ttl_matrices)
    plot_burst_priming_effect(output_dir, base_rates, primed_rates)

    if args.arm64_calibration_dir.exists():
        comparison_rows = build_architecture_comparison_rows(
            x64_calibration_dir=args.calibration_dir,
            arm64_calibration_dir=args.arm64_calibration_dir,
            benchmark=args.comparison_benchmark,
            burst_concurrency=args.comparison_burst_concurrency,
        )
        write_architecture_comparison_summary(
            output_dir=output_dir,
            comparison_rows=comparison_rows,
            benchmark=args.comparison_benchmark,
        )
        plot_architecture_comparison(
            output_dir=output_dir,
            comparison_rows=comparison_rows,
            benchmark=args.comparison_benchmark,
            burst_concurrency=args.comparison_burst_concurrency,
        )


if __name__ == "__main__":
    main()
