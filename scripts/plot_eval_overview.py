#!/usr/bin/env python
"""Generate evaluation overview figures for the CALO paper."""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

DEFAULT_RESULT_PATTERN = "results/full_suite_stage2_prime_seed*/dynamic_experiment_results.json"
DEFAULT_OUTPUT_DIR = Path(
    "Code_Aware_and_Load_Aware_Resource_Configuration_for_Serverless_Functions_via_Deep_Reinforcement_Learning__1_/figures"
)
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

BENCH_ORDER = [
    "110.dynamic-html",
    "120.uploader",
    "210.thumbnailer",
    "311.compression",
    "411.image-recognition",
]
BENCH_LABELS = {
    "110.dynamic-html": "Dynamic HTML",
    "120.uploader": "Uploader",
    "210.thumbnailer": "Thumbnailer",
    "311.compression": "Compression",
    "411.image-recognition": "Image Recognition",
}
ALGO_LABELS = {
    "ppo": "CALO",
    "ppo_load_only": "Load-Only CALO",
    "greedy": "Greedy Profiling",
    "bayes_opt_online": "Online BO",
    "default": "Provider Default",
    "random": "Random",
}
ALGO_ORDER = [
    "ppo",
    "ppo_load_only",
    "bayes_opt_online",
    "greedy",
    "default",
    "random",
]

PLOT_COLORS = {
    "ppo": "#f2c84b",
    "ppo_load_only": "#7ea95a",
    "bayes_opt_online": "#0d6b2f",
    "greedy": "#3a9e49",
    "default": "#b7c4c8",
    "random": "#ddd8ca",
}
PATTERN_ORDER = ["sine", "spike", "decay", "random"]
PATTERN_LABELS = {
    "sine": "Periodic",
    "spike": "Spike",
    "decay": "Decay",
    "random": "Random Walk",
}
EDGE_COLOR = "#101010"
GRID_COLOR = "#dadada"
FACE_COLOR = "#fffdf8"
LEGEND_EDGE = "#c9c4b6"
ATTR_GROUP_COLORS = {
    "load": "#0d6b2f",
    "history": "#3a9e49",
    "code": "#f2c84b",
    "context": "#b7c4c8",
    "category": "#ddd8ca",
}
ATTR_GROUP_LABELS = {
    "load": "Load",
    "history": "History",
    "code": "Code",
    "context": "Context",
    "category": "Category",
}
PAPER_MAIN_ALGOS = [
    "ppo",
    "bayes_opt_online",
    "greedy",
    "default",
]
ATTR_FEATURE_LABELS = {
    "history_latency_variance": "Latency variance",
    "burst_indicator": "Burst indicator",
    "peak_concurrency": "Peak concurrency",
    "history_mean_latency": "Mean latency",
    "history_mean_cost": "Mean cost",
    "current_qps": "Current QPS",
    "history_success_rate": "Success rate",
    "cold_start_probability": "Cold-start prob.",
    "container_warmth": "Container warmth",
    "avg_response_time": "Avg. response time",
}
REAL_PLATFORM_CASE_ORDER = [
    ("spike", 42),
    ("spike", 52),
    ("spike", 62),
    ("decay", 42),
    ("decay", 52),
]
REAL_PLATFORM_CASE_STYLES = {
    ("spike", 42): {
        "label": "Spike 42",
        "color": "#0d6b2f",
        "linestyle": "-",
        "marker": "o",
    },
    ("spike", 52): {
        "label": "Spike 52",
        "color": "#3a9e49",
        "linestyle": "-",
        "marker": "s",
    },
    ("spike", 62): {
        "label": "Spike 62",
        "color": "#7ea95a",
        "linestyle": "-",
        "marker": "^",
    },
    ("decay", 42): {
        "label": "Decay 42",
        "color": "#8c6239",
        "linestyle": "--",
        "marker": "D",
    },
    ("decay", 52): {
        "label": "Decay 52",
        "color": "#c9932d",
        "linestyle": "--",
        "marker": "P",
    },
}


def _apply_pi_style() -> None:
    """Apply a paper-style theme inspired by PI figures."""
    plt.style.use("default")
    mpl.rcdefaults()
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
            "axes.edgecolor": EDGE_COLOR,
            "axes.grid": False,
            "errorbar.capsize": 4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _lighten(color: str, amount: float = 0.45) -> str:
    """Blend one color with white."""
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb + (1.0 - rgb) * amount
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def _ordered_subset(preferred: Sequence[str], available: Sequence[str]) -> List[str]:
    """Return available entries in preferred order with unknowns appended."""
    available_set = set(available)
    ordered = [item for item in preferred if item in available_set]
    ordered.extend(sorted(item for item in available if item not in ordered))
    return ordered


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply the common PI-like axis styling."""
    ax.set_facecolor(FACE_COLOR)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, linestyle="-", alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(EDGE_COLOR)
    ax.spines["bottom"].set_color(EDGE_COLOR)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(axis="both", which="both", length=0, colors=EDGE_COLOR)


def _style_legend(legend: plt.Legend | None) -> None:
    """Apply a framed white legend similar to the reference figures."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(0.9)
    frame.set_alpha(0.96)


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save one vector figure and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def _resolve_file_inputs(
    path_args: Sequence[str], directory_filename: str | None = None
) -> List[str]:
    """Resolve CLI path arguments that may be files, directories, or globs."""
    resolved: List[str] = []
    for path_arg in path_args:
        candidate = Path(path_arg)
        if candidate.is_dir():
            if directory_filename is None:
                raise FileNotFoundError(
                    f"Directory input requires a default filename: {path_arg}"
                )
            default_file = candidate / directory_filename
            if not default_file.is_file():
                raise FileNotFoundError(f"Missing {directory_filename} in {path_arg}")
            resolved.append(str(default_file))
            continue
        matches = sorted(glob.glob(path_arg))
        if matches:
            resolved.extend(matches)
            continue
        if candidate.is_file():
            resolved.append(str(candidate))
            continue
        raise FileNotFoundError(f"Input path not found: {path_arg}")
    return list(dict.fromkeys(resolved))


def _present_bench_order(
    aggregated: Dict[str, Dict[str, Dict[str, float]]],
) -> List[str]:
    """Return the benchmark order for the currently loaded results."""
    available = {
        benchmark
        for algo_data in aggregated.values()
        for benchmark in algo_data.keys()
    }
    return _ordered_subset(BENCH_ORDER, list(available))


def _present_pattern_order(
    per_algo_pattern: Dict[Tuple[str, str], Dict[str, List[float]]],
) -> List[str]:
    """Return the workload-pattern order for the current results."""
    available = {pattern for _, pattern in per_algo_pattern.keys()}
    return _ordered_subset(PATTERN_ORDER, list(available))


def _present_algo_order(available_algos: Sequence[str]) -> List[str]:
    """Return the algorithm order for the current results."""
    return _ordered_subset(ALGO_ORDER, list(available_algos))


def _present_paper_algo_order(available_algos: Sequence[str]) -> List[str]:
    """Return the main-paper algorithm order without extreme-tail baselines."""
    available = set(available_algos)
    paper_algos = [algo for algo in PAPER_MAIN_ALGOS if algo in available]
    if paper_algos:
        return paper_algos
    return _present_algo_order(available_algos)


def load_seed_results(
    patterns: Union[str, Sequence[str]],
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, List[float]]],
    Dict[Tuple[str, str], Dict[str, List[float]]],
    Dict[str, Dict[str, List[float]]],
]:
    per_algo_bench: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {
            "reward": [],
            "success": [],
            "latency": [],
            "cost": [],
        }
    )
    per_algo_pattern: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(
        lambda: {
            "reward": [],
            "success": [],
            "latency": [],
            "cost": [],
        }
    )
    per_algo_global: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {
            "reward": [],
            "success": [],
            "latency": [],
            "cost": [],
        }
    )

    if isinstance(patterns, str):
        pattern_list = [patterns]
    else:
        pattern_list = list(patterns)

    seed_paths: List[str] = []
    for pattern in pattern_list:
        seed_paths.extend(glob.glob(pattern))
    seed_paths = sorted(set(seed_paths))
    if not seed_paths:
        raise FileNotFoundError(
            f"No result files found for patterns: {', '.join(pattern_list)}"
        )

    for path in seed_paths:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for algo, benches in data.items():
            for bench, patterns in benches.items():
                reward_vals = [metrics["mean_reward"] for metrics in patterns.values()]
                success_vals = [metrics["mean_success_rate"] for metrics in patterns.values()]
                latency_vals = [metrics["mean_latency"] for metrics in patterns.values()]
                cost_vals = [metrics["mean_cost"] for metrics in patterns.values()]

                per_algo_bench[(algo, bench)]["reward"].append(float(np.mean(reward_vals)))
                per_algo_bench[(algo, bench)]["success"].append(float(np.mean(success_vals)))
                per_algo_bench[(algo, bench)]["latency"].append(float(np.mean(latency_vals)))
                per_algo_bench[(algo, bench)]["cost"].append(float(np.mean(cost_vals)))

                for pattern_name, metrics in patterns.items():
                    per_algo_pattern[(algo, pattern_name)]["reward"].append(float(metrics["mean_reward"]))
                    per_algo_pattern[(algo, pattern_name)]["success"].append(float(metrics["mean_success_rate"]))
                    per_algo_pattern[(algo, pattern_name)]["latency"].append(float(metrics["mean_latency"]))
                    per_algo_pattern[(algo, pattern_name)]["cost"].append(float(metrics["mean_cost"]))

                    per_algo_global[algo]["reward"].append(float(metrics["mean_reward"]))
                    per_algo_global[algo]["success"].append(float(metrics["mean_success_rate"]))
                    per_algo_global[algo]["latency"].append(float(metrics["mean_latency"]))
                    per_algo_global[algo]["cost"].append(float(metrics["mean_cost"]))

    return per_algo_bench, per_algo_pattern, per_algo_global


def aggregate_nested(
    per_algo_key: Dict[Tuple[str, str], Dict[str, List[float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (algo, key), metrics in per_algo_key.items():
        aggregated[algo][key] = {
            metric: float(np.mean(values)) for metric, values in metrics.items()
        }
    return aggregated


def aggregate_global(per_algo: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    for algo, metrics in per_algo.items():
        aggregated[algo] = {metric: float(np.mean(values)) for metric, values in metrics.items()}
    return aggregated


def compute_cvar(values: List[float], alpha: float = 0.1) -> float:
    """Compute Conditional Value-at-Risk at the given alpha."""
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0:
        return float("nan")
    cutoff = max(1, int(np.ceil(alpha * arr.size)))
    return float(np.mean(arr[:cutoff]))


def plot_reward_by_benchmark(aggregated: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    _apply_pi_style()

    width = 0.12
    group_spacing = 1.2
    bench_order = _present_bench_order(aggregated)
    algo_order = _present_paper_algo_order(list(aggregated.keys()))
    x = np.arange(len(bench_order)) * group_spacing

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for idx, algo in enumerate(algo_order):
        if algo not in aggregated:
            continue
        rewards = [aggregated[algo][bench]["reward"] for bench in bench_order]
        offset = (idx - (len(algo_order) - 1) / 2) * width * 1.35
        ax.bar(
            x + offset,
            rewards,
            width=width,
            label=ALGO_LABELS.get(algo, algo.title()),
            color=PLOT_COLORS.get(algo),
            edgecolor=EDGE_COLOR,
            linewidth=1.0,
            alpha=0.96,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABELS.get(b, b) for b in bench_order], rotation=16, ha="right")
    ax.set_ylabel("Mean Reward (higher is better)")
    _style_axes(ax)
    legend = ax.legend(
        frameon=True,
        ncol=1,
        loc="upper right",
        handlelength=1.4,
        borderpad=0.45,
        labelspacing=0.35,
    )
    _style_legend(legend)
    fig.tight_layout()
    _save_figure(fig, "fig_reward_by_benchmark.pdf")


def plot_success_reliability(per_algo_global: Dict[str, Dict[str, List[float]]]) -> None:
    _apply_pi_style()

    fig, ax = plt.subplots(figsize=(5.7, 3.7))
    algo_order = _present_paper_algo_order(list(per_algo_global.keys()))
    categories = []
    values = []
    errors = []
    colors = []
    for algo in algo_order:
        if algo not in per_algo_global:
            continue
        categories.append(ALGO_LABELS.get(algo, algo.title()))
        success_values = np.asarray(per_algo_global[algo]["success"], dtype=float) * 100.0
        values.append(float(np.mean(success_values)))
        errors.append(float(np.std(success_values)))
        colors.append(PLOT_COLORS.get(algo))

    bars = ax.bar(
        categories,
        values,
        yerr=errors,
        color=colors,
        alpha=0.97,
        edgecolor=EDGE_COLOR,
        linewidth=1.2,
        error_kw={
            "elinewidth": 1.0,
            "ecolor": EDGE_COLOR,
            "capthick": 1.0,
        },
    )

    ax.set_ylabel("Success Rate (%)")
    ymin = max(0.0, min(v - e for v, e in zip(values, errors)) - 3.0)
    ymax = min(100.0, max(v + e for v, e in zip(values, errors)) + 4.0)
    ax.set_ylim(ymin, ymax)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_yticks(np.linspace(ymin, ymax, 5))
    _style_axes(ax)

    for patch, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}%",
            xy=(patch.get_x() + patch.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    _save_figure(fig, "fig_success_rate.pdf")


def plot_workload_patterns(
    per_algo_pattern: Dict[Tuple[str, str], Dict[str, List[float]]],
) -> None:
    _apply_pi_style()

    fig, ax = plt.subplots(figsize=(5.9, 3.7))
    pattern_order = _present_pattern_order(per_algo_pattern)
    available_algos = {algo for algo, _ in per_algo_pattern.keys()}
    algo_order = _present_paper_algo_order(list(available_algos))
    x = np.arange(len(pattern_order))
    for algo in algo_order:
        means = []
        stds = []
        for pattern in pattern_order:
            metrics = per_algo_pattern.get((algo, pattern))
            if metrics is None:
                continue
            reward_values = np.asarray(metrics["reward"], dtype=float)
            means.append(float(np.mean(reward_values)))
            stds.append(float(np.std(reward_values)))
        if not means:
            continue
        color = PLOT_COLORS.get(algo)
        means_array = np.asarray(means, dtype=float)
        std_array = np.asarray(stds, dtype=float)
        ax.plot(
            x,
            means_array,
            marker="o",
            linewidth=2.2 if algo == "ppo" else 1.8,
            markersize=5.0,
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=0.9,
            color=color,
            label=ALGO_LABELS.get(algo, algo.title()),
            zorder=3 if algo == "ppo" else 2,
        )
        if algo == "ppo":
            ax.fill_between(
                x,
                means_array - std_array,
                means_array + std_array,
                color=_lighten(color, 0.35),
                alpha=0.28,
                zorder=1,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABELS.get(p, p) for p in pattern_order])
    ax.set_ylabel("Mean Reward (higher is better)")
    _style_axes(ax)
    legend = ax.legend(
        frameon=True,
        loc="lower left",
        ncol=1,
        borderpad=0.45,
        labelspacing=0.35,
    )
    _style_legend(legend)
    fig.tight_layout()
    _save_figure(fig, "fig_workload_reward.pdf")


def plot_reward_robustness(per_algo_global: Dict[str, Dict[str, List[float]]]) -> None:
    _apply_pi_style()

    stats = {}
    for algo, metrics in per_algo_global.items():
        rewards = np.asarray(metrics["reward"], dtype=float)
        stats[algo] = {
            "median": float(np.median(rewards)),
            "cvar": compute_cvar(rewards, alpha=0.1),
        }

    algos = _present_paper_algo_order(list(stats.keys()))
    x = np.arange(len(algos), dtype=float) * 1.08
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.4, 2.95))
    medians = [stats[algo]["median"] for algo in algos]
    cvars = [stats[algo]["cvar"] for algo in algos]
    bars_median = ax.bar(
        x - width / 2,
        medians,
        width=width,
        color=[PLOT_COLORS.get(algo) for algo in algos],
        alpha=0.96,
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        label="Median",
    )
    bars_cvar = ax.bar(
        x + width / 2,
        cvars,
        width=width,
        color=[_lighten(PLOT_COLORS.get(algo), 0.45) for algo in algos],
        alpha=0.96,
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        label="CVaR$_{10}$",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [ALGO_LABELS.get(algo, algo.title()) for algo in algos],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Reward (higher is better)")
    y_min = min(medians + cvars)
    y_max = max(medians + cvars)
    span = y_max - y_min
    ax.set_ylim(y_min - 0.12 * span - 0.05, y_max + 0.22 * span + 0.05)
    legend = ax.legend(
        frameon=True,
        loc="upper right",
        ncol=2,
        borderpad=0.45,
        labelspacing=0.35,
        columnspacing=0.9,
        handlelength=1.4,
    )
    _style_legend(legend)
    _style_axes(ax)
    ax.margins(x=0.14)
    for patch, value in zip(bars_median, medians):
        x_shift = -4
        if value >= 0 and value < 0.05:
            offset = 10
        else:
            offset = 5 if value >= 0 else -9
        va = "bottom" if value >= 0 else "top"
        ax.annotate(
            f"{value:.2f}",
            xy=(patch.get_x() + patch.get_width() / 2, value),
            xytext=(x_shift, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7.1,
            zorder=5,
        )
    for patch, value in zip(bars_cvar, cvars):
        x_shift = 4
        if value >= 0 and value < 0.05:
            offset = 8
        else:
            offset = 5 if value >= 0 else -9
        va = "bottom" if value >= 0 else "top"
        ax.annotate(
            f"{value:.2f}",
            xy=(patch.get_x() + patch.get_width() / 2, value),
            xytext=(x_shift, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7.1,
            zorder=5,
        )
    fig.tight_layout()
    _save_figure(fig, "fig_reward_robustness.pdf")


def plot_latency_cost_frontier(global_metrics: Dict[str, Dict[str, float]]) -> None:
    _apply_pi_style()

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    algo_order = _present_paper_algo_order(list(global_metrics.keys()))
    for algo in algo_order:
        if algo not in global_metrics:
            continue
        latency = global_metrics[algo]["latency"]
        cost = global_metrics[algo]["cost"] * 1e5  # scale to 1e-5 USD
        ax.scatter(
            cost,
            latency,
            s=86 if algo == "ppo" else 72,
            color=PLOT_COLORS.get(algo),
            edgecolors=EDGE_COLOR,
            linewidth=1.0,
            label=ALGO_LABELS.get(algo, algo.title()),
        )

    ax.set_xlabel("Cost (×10$^{-5}$ USD / invocation)")
    ax.set_ylabel("Mean Latency (ms)")
    _style_axes(ax, grid_axis="both")
    legend = ax.legend(
        frameon=True,
        loc="upper right",
        borderpad=0.45,
        labelspacing=0.35,
    )
    _style_legend(legend)
    fig.tight_layout()
    _save_figure(fig, "fig_latency_cost_tradeoff.pdf")


def _extract_attribution_reward_shift(item: Dict[str, object]) -> float:
    """Return a stable absolute reward-shift scalar from mixed schema versions."""
    if "mean_absolute_reward_shift" in item:
        return abs(float(item["mean_absolute_reward_shift"]))
    if "absolute_reward_shift" in item:
        return abs(float(item["absolute_reward_shift"]))
    if "reward_drop" in item:
        return abs(float(item["reward_drop"]))
    return abs(float(item.get("mean_reward_drop", 0.0)))


def load_feature_attribution(
    result_paths: Sequence[str],
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Load and aggregate feature-attribution results from one or more JSON files."""
    group_buckets: Dict[str, List[float]] = defaultdict(list)
    feature_buckets: Dict[str, List[float]] = defaultdict(list)

    for result_path in _resolve_file_inputs(
        result_paths,
        directory_filename="dynamic_experiment_results.json",
    ):
        with open(result_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        ppo_results = data.get("ppo", {})
        for benchmark_data in ppo_results.values():
            for metrics in benchmark_data.values():
                attribution = metrics.get("feature_attribution")
                if attribution is None:
                    continue

                for item in attribution.get("group_importance", []):
                    group_buckets[str(item["name"])].append(
                        _extract_attribution_reward_shift(item)
                    )

                for item in attribution.get("single_feature_importance", []):
                    group = str(item.get("group", ""))
                    name = str(item["name"])
                    if group not in {"load", "history"}:
                        continue
                    if name in {"day_of_week", "hour_of_day"}:
                        continue
                    feature_buckets[name].append(
                        _extract_attribution_reward_shift(item)
                    )

    group_summary = sorted(
        [
            (group, float(np.mean(values)))
            for group, values in group_buckets.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    feature_summary = sorted(
        [
            (feature, float(np.mean(values)))
            for feature, values in feature_buckets.items()
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return group_summary, feature_summary


def plot_feature_attribution(result_paths: Sequence[str], top_k: int = 6) -> None:
    """Plot compact feature-attribution results for the paper."""
    _apply_pi_style()
    group_summary, feature_summary = load_feature_attribution(result_paths)
    if not group_summary or not feature_summary:
        raise ValueError(
            "No feature attribution found in the provided result files."
        )

    feature_summary = feature_summary[:top_k]

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(6.6, 3.2),
        gridspec_kw={"width_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )

    ax = axes[0]
    group_names = [ATTR_GROUP_LABELS.get(name, name.title()) for name, _ in group_summary]
    group_values = [value for _, value in group_summary]
    group_colors = [ATTR_GROUP_COLORS.get(name, "#888888") for name, _ in group_summary]
    y = np.arange(len(group_names))
    bars = ax.barh(
        y,
        group_values,
        color=group_colors,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        height=0.62,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(group_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean abs. reward shift")
    ax.set_title("State Groups", pad=6)
    _style_axes(ax)
    for bar, value in zip(bars, group_values):
        ax.annotate(
            f"{value:.4f}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color=EDGE_COLOR,
        )

    ax = axes[1]
    feature_names = [ATTR_FEATURE_LABELS.get(name, name) for name, _ in feature_summary]
    feature_values = [value for _, value in feature_summary]
    y = np.arange(len(feature_names))
    bars = ax.barh(
        y,
        feature_values,
        color=_lighten(PLOT_COLORS["ppo"], 0.15),
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        height=0.62,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean abs. reward shift")
    ax.set_title("Top Runtime Signals", pad=6)
    _style_axes(ax)
    for bar, value in zip(bars, feature_values):
        ax.annotate(
            f"{value:.4f}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.3,
            color=EDGE_COLOR,
        )

    _save_figure(fig, "fig_feature_attribution.pdf")


def _real_platform_case_sort_key(case: Dict[str, object]) -> Tuple[int, str, int]:
    """Sort real-platform replay cases in a stable paper-friendly order."""
    key = (str(case["load_pattern"]), int(case["seed"]))
    if key in REAL_PLATFORM_CASE_ORDER:
        return (REAL_PLATFORM_CASE_ORDER.index(key), key[0], key[1])
    return (len(REAL_PLATFORM_CASE_ORDER), key[0], key[1])


def load_real_platform_reward_deltas(
    summary_paths: Sequence[str],
) -> List[Dict[str, object]]:
    """Load per-step and cumulative reward deltas from replay summaries."""
    cases: List[Dict[str, object]] = []
    for summary_path in _resolve_file_inputs(
        summary_paths,
        directory_filename="summary.json",
    ):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

        results = {
            entry["policy"]: entry
            for entry in summary.get("results", [])
        }
        if "ppo" not in results or "default" not in results:
            raise ValueError(
                "Expected both 'ppo' and 'default' policies in "
                f"{summary_path}, found {sorted(results.keys())}."
            )

        ppo_steps = results["ppo"].get("step_records", [])
        default_steps = results["default"].get("step_records", [])
        if not ppo_steps or len(ppo_steps) != len(default_steps):
            raise ValueError(
                f"Invalid step records in {summary_path}: "
                f"{len(ppo_steps)} PPO steps vs {len(default_steps)} default steps."
            )

        step_deltas = [
            float(ppo_record["raw_reward"]) - float(default_record["raw_reward"])
            for ppo_record, default_record in zip(ppo_steps, default_steps)
        ]
        cumulative_deltas = list(np.cumsum(np.asarray(step_deltas, dtype=float)))

        load_pattern = str(summary["load_pattern"])
        seed = int(summary["seed"])
        style = REAL_PLATFORM_CASE_STYLES.get(
            (load_pattern, seed),
            {
                "label": f"{load_pattern.title()} {seed}",
                "color": "#4c4c4c",
                "linestyle": "-",
                "marker": "o",
            },
        )
        cases.append(
            {
                "benchmark": str(summary["benchmark"]),
                "load_pattern": load_pattern,
                "seed": seed,
                "step_indices": list(range(len(step_deltas))),
                "step_deltas": step_deltas,
                "cumulative_deltas": cumulative_deltas,
                "label": style["label"],
                "color": style["color"],
                "linestyle": style["linestyle"],
                "marker": style["marker"],
            }
        )
    return sorted(cases, key=_real_platform_case_sort_key)


def plot_real_platform_reward_delta(summary_paths: Sequence[str]) -> None:
    """Plot step-wise and cumulative reward deltas for focused replay cases."""
    _apply_pi_style()
    cases = load_real_platform_reward_deltas(summary_paths)
    if not cases:
        raise ValueError("No real-platform replay summaries were provided.")

    fig, axes = plt.subplots(
        nrows=2,
        figsize=(3.45, 4.75),
        sharex=True,
    )
    line_handles = []
    for case in cases:
        kwargs = {
            "color": str(case["color"]),
            "linestyle": str(case["linestyle"]),
            "marker": str(case["marker"]),
            "linewidth": 1.8,
            "markersize": 4.2,
            "markeredgecolor": EDGE_COLOR,
            "markeredgewidth": 0.7,
            "label": str(case["label"]),
            "zorder": 3,
        }
        (line,) = axes[0].plot(
            case["step_indices"],
            case["step_deltas"],
            **kwargs,
        )
        axes[1].plot(
            case["step_indices"],
            case["cumulative_deltas"],
            **kwargs,
        )
        line_handles.append(line)

    for ax in axes:
        _style_axes(ax)
        ax.axhline(0.0, color=EDGE_COLOR, linewidth=0.9, linestyle=":", alpha=0.85)
        ax.set_xticks(cases[0]["step_indices"])

    axes[0].set_ylabel("Step Delta")
    axes[1].set_ylabel("Cumulative Delta")
    axes[1].set_xlabel("Control Step")
    axes[0].set_title("Per-Step Raw Reward Delta", pad=6)
    axes[1].set_title("Cumulative Raw Reward Delta", pad=6)

    legend = fig.legend(
        handles=line_handles,
        labels=[str(case["label"]) for case in cases],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        borderpad=0.38,
        labelspacing=0.35,
        columnspacing=0.9,
        handlelength=1.8,
    )
    _style_legend(legend)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save_figure(fig, "7_real_platform_reward_delta.pdf")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    if not args.skip_overview:
        result_patterns = [derive_pattern(pattern) for pattern in args.results_globs]
        _per_algo_bench, per_algo_pattern, per_algo_global = load_seed_results(
            result_patterns
        )
        global_metrics = aggregate_global(per_algo_global)
        plot_reward_robustness(per_algo_global)
        plot_success_reliability(per_algo_global)
        plot_workload_patterns(per_algo_pattern)
        plot_latency_cost_frontier(global_metrics)
    if args.feature_attribution is not None:
        plot_feature_attribution(
            result_paths=args.feature_attribution,
            top_k=args.feature_top_k,
        )
    if args.real_platform_summaries:
        plot_real_platform_reward_delta(args.real_platform_summaries)
    print("Saved evaluation overview figures to", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CALO evaluation figures")
    parser.add_argument(
        "--results",
        dest="results_globs",
        nargs="+",
        default=[DEFAULT_RESULT_PATTERN],
        help=(
            "One or more glob patterns, directories, or files containing JSON "
            "result data (shell globs are supported)."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated figures will be saved",
    )
    parser.add_argument(
        "--skip-overview",
        dest="skip_overview",
        action="store_true",
        help="Skip the broad-comparison overview figures.",
    )
    parser.add_argument(
        "--feature-attribution",
        dest="feature_attribution",
        nargs="+",
        default=None,
        help=(
            "Optional dynamic_experiment_results.json files or result directories "
            "that contain feature_attribution results for plotting one compact "
            "attribution figure."
        ),
    )
    parser.add_argument(
        "--feature-top-k",
        dest="feature_top_k",
        type=int,
        default=6,
        help="Number of runtime features to show in the attribution panel.",
    )
    parser.add_argument(
        "--real-platform-summaries",
        dest="real_platform_summaries",
        nargs="+",
        default=None,
        help=(
            "Optional summary.json files or result directories produced by "
            "review_real_platform_online_ab.py for plotting the focused "
            "real-platform reward-delta figure."
        ),
    )
    return parser.parse_args()


def derive_pattern(path_arg: str) -> str:
    candidate = Path(path_arg)
    if candidate.is_dir():
        return str(candidate / "*.json")
    return path_arg


if __name__ == "__main__":
    main()
