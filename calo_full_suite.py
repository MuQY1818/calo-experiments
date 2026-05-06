#!/usr/bin/env python3
"""
CALO 全量评估脚本
5个 benchmark × 4 种负载 × 5 个算法 × 3 个 seed
生成 6 组统计图表支撑主文分析
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.legend import Legend

ALGO_ORDER = [
    "ppo",
    "ppo_load_only",
    "bayes_opt_online",
    "greedy",
    "default",
    "random",
]
ALGO_LABELS = {
    "ppo": "CALO",
    "ppo_load_only": "Load-Only CALO",
    "bayes_opt_online": "Online BO",
    "greedy": "Greedy Profiling",
    "default": "Provider Default",
    "random": "Random",
}
BENCH_LABELS = {
    "110.dynamic-html": "Dynamic HTML",
    "120.uploader": "Uploader",
    "210.thumbnailer": "Thumbnailer",
    "311.compression": "Compression",
    "411.image-recognition": "Image Recognition",
}
PATTERN_LABELS = {
    "sine": "Periodic",
    "spike": "Spike",
    "decay": "Decay",
    "random": "Random Walk",
}
PLOT_COLORS = {
    "ppo": "#f2c84b",
    "ppo_load_only": "#7ea95a",
    "bayes_opt_online": "#0d6b2f",
    "greedy": "#3a9e49",
    "default": "#b7c4c8",
    "random": "#ddd8ca",
}
PAPER_MAIN_ALGOS = [
    "ppo",
    "bayes_opt_online",
    "greedy",
    "default",
]
EDGE_COLOR = "#101010"
GRID_COLOR = "#dadada"
FACE_COLOR = "#fffdf8"
LEGEND_EDGE = "#c9c4b6"


def _build_state_space(environment_cfg: dict):
    """Construct one state space that matches the configured code dimension."""
    from rl_optimizer.state_space import StateSpace

    return StateSpace(
        sebs_root='.',
        code_feature_dim=environment_cfg.get(
            'code_feature_dim',
            StateSpace.DEFAULT_CODE_FEATURE_DIM,
        ),
    )


def _apply_pi_style() -> None:
    """Apply a restrained PI-like plotting style."""
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
            "legend.fontsize": 9.2,
            "axes.linewidth": 1.1,
            "axes.edgecolor": EDGE_COLOR,
            "axes.grid": False,
            "errorbar.capsize": 4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _ordered_subset(preferred: Sequence[str], available: Sequence[str]) -> List[str]:
    """Return values in preferred order and append unknown entries."""
    available_set = set(available)
    ordered = [item for item in preferred if item in available_set]
    ordered.extend(sorted(item for item in available if item not in ordered))
    return ordered


def _present_algo_order(algorithms: Sequence[str]) -> List[str]:
    """Return the plotting order for algorithms."""
    return _ordered_subset(ALGO_ORDER, list(algorithms))


def _present_paper_algo_order(algorithms: Sequence[str]) -> List[str]:
    """Return the main-paper plotting order without extreme-tail baselines."""
    available = set(algorithms)
    paper_algos = [algo for algo in PAPER_MAIN_ALGOS if algo in available]
    if paper_algos:
        return paper_algos
    return _present_algo_order(algorithms)


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    """Apply the common axis styling."""
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


def _style_legend(legend: Legend | None) -> None:
    """Apply a compact framed legend."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_linewidth(0.9)
    frame.set_alpha(0.96)


def _save_figure(fig: plt.Figure, path_stem: Path, with_png: bool = True) -> None:
    """Save one figure in vector form and optionally emit a PNG preview."""
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    if with_png:
        fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _lighten(color: str, amount: float = 0.45) -> str:
    """Blend a color toward white."""
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb + (1.0 - rgb) * amount
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def run_full_suite(
    config_path: str = 'config/rl_experiments/full_suite.json',
    output_dir: str = 'results_full_suite',
    debug: bool = False,
):
    """
    运行 CALO 全量组合实验
    """
    # 延迟导入训练相关依赖，便于仅绘制图表时无需安装重型框架
    from rl_optimizer.environment import ServerlessFunctionEnv
    from rl_optimizer.dynamic_environment import DynamicLoadEnvironment
    from rl_optimizer.pure_ppo import PurePPO, OnlineBayesOptBaseline
    from rl_optimizer.baselines import DefaultBaseline, RandomBaseline, GreedyBaseline

    with open(config_path, 'r') as f:
        config = json.load(f)
    environment_cfg = config['environment']

    print("=" * 80)
    print("CALO 全量组合实验")
    print("=" * 80)
    print(f"配置: {config['description']}")
    print(f"Benchmarks: {len(config['benchmarks'])} 个")
    print(f"负载模式: {len(config['environment']['load_patterns'])} 种")
    print(f"算法: {len(config['algorithms'])} 个")
    print(f"Seeds: {len(config['reproducibility']['seeds'])} 个")
    total_experiments = (len(config['benchmarks']) *
                        len(config['environment']['load_patterns']) *
                        len(config['algorithms']) *
                        len(config['reproducibility']['seeds']))
    print(f"总实验数: {total_experiments}")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for seed_idx, seed in enumerate(config['reproducibility']['seeds']):
        print(f"\n{'='*80}")
        print(f"Seed {seed_idx+1}/{len(config['reproducibility']['seeds'])}: {seed}")
        print(f"{'='*80}")

        np.random.seed(seed)
        seed_results = {}

        for algorithm in config['algorithms']:
            seed_results[algorithm] = {}

        for benchmark in config['benchmarks']:
            print(f"\n{'='*80}")
            print(f"Benchmark: {benchmark}")
            print(f"{'='*80}")

            for load_pattern in config['environment']['load_patterns']:
                print(f"\n负载模式: {load_pattern}")
                print("-" * 80)

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
                    step_duration_sec=environment_cfg.get('step_duration_sec', 15.0),
                    max_containers=environment_cfg.get('max_containers', 32),
                    container_ttl_sec=environment_cfg.get('container_ttl_sec', 600.0),
                    state_space=_build_state_space(environment_cfg),
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
                )

                # 设置负载模式
                dynamic_env.load_pattern = load_pattern

                # 运行所有算法
                for algorithm in config['algorithms']:
                    print(f"\n[{algorithm.upper()}]")

                    start_time = time.time()

                    if algorithm == 'ppo':
                        agent = PurePPO(
                            dynamic_env,
                            total_timesteps=config['training']['total_timesteps'],
                            ppo_kwargs=config['training']['ppo_kwargs'],
                        )

                        # 训练并保存曲线图
                        training_plot_path = (output_dir / 'training_curves' /
                                            f'seed{seed}_{benchmark}_{load_pattern}_ppo.png')
                        training_plot_path.parent.mkdir(parents=True, exist_ok=True)

                        agent.train(save_plots_to=str(training_plot_path))
                        results = agent.evaluate(n_episodes=config['evaluation']['n_episodes'])

                    elif algorithm == 'bayes_opt_online':
                        agent = OnlineBayesOptBaseline(
                            dynamic_env,
                            reoptimize_freq=config['baseline']['reoptimize_freq'],
                        )
                        results = agent.evaluate(n_episodes=config['evaluation']['n_episodes'])

                    elif algorithm == 'default':
                        agent = DefaultBaseline(
                            dynamic_env,
                            memory=config['baseline']['default_memory'],
                        )
                        results = agent.evaluate(n_episodes=config['evaluation']['n_episodes'])

                    elif algorithm == 'random':
                        agent = RandomBaseline(dynamic_env)
                        results = agent.evaluate(n_episodes=config['evaluation']['n_episodes'])

                    elif algorithm == 'greedy':
                        agent = GreedyBaseline(dynamic_env)
                        results = agent.evaluate(n_episodes=config['evaluation']['n_episodes'])

                    else:
                        raise ValueError(f"Unknown algorithm: {algorithm}")

                    elapsed_time = time.time() - start_time
                    results['wall_time'] = elapsed_time

                    print(f"  奖励: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
                    print(f"  延迟: {results['mean_latency']:.2f} ± {results['std_latency']:.2f} ms")
                    print(f"  成本: ${results['mean_cost']:.6f} ± ${results['std_cost']:.6f}")
                    print(f"  用时: {elapsed_time:.2f}秒")

                    # 保存结果
                    if benchmark not in seed_results[algorithm]:
                        seed_results[algorithm][benchmark] = {}
                    seed_results[algorithm][benchmark][load_pattern] = results

        all_results[f'seed_{seed}'] = seed_results

        # 保存中间结果
        results_path = output_dir / f'results_seed_{seed}.json'
        with open(results_path, 'w') as f:
            json.dump(seed_results, f, indent=2)
        print(f"\nSeed {seed} 结果已保存到: {results_path}")

    # 合并所有seed的结果
    final_results = aggregate_results(all_results, config)

    # 保存最终结果
    final_path = output_dir / 'full_suite_results.json'
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n最终结果已保存到: {final_path}")

    # 生成报告和图表
    generate_suite_report(final_results, config, output_dir)


def aggregate_results(all_results: Dict, config: Dict) -> Dict:
    """
    聚合多个seed的结果
    """
    algorithms = config['algorithms']
    benchmarks = config['benchmarks']
    load_patterns = config['environment']['load_patterns']

    aggregated = {}

    for algorithm in algorithms:
        aggregated[algorithm] = {}

        for benchmark in benchmarks:
            aggregated[algorithm][benchmark] = {}

            for load_pattern in load_patterns:
                # 收集所有seed的结果
                seed_rewards = []
                seed_latencies = []
                seed_costs = []
                seed_success_rates = []

                for seed_key in all_results.keys():
                    seed_data = all_results[seed_key][algorithm][benchmark][load_pattern]
                    seed_rewards.append(seed_data['mean_reward'])
                    seed_latencies.append(seed_data['mean_latency'])
                    seed_costs.append(seed_data['mean_cost'])
                    seed_success_rates.append(seed_data.get('mean_success_rate', 1.0))

                # 计算跨seed的统计量
                aggregated[algorithm][benchmark][load_pattern] = {
                    'mean_reward': float(np.mean(seed_rewards)),
                    'std_reward': float(np.std(seed_rewards)),
                    'mean_latency': float(np.mean(seed_latencies)),
                    'std_latency': float(np.std(seed_latencies)),
                    'mean_cost': float(np.mean(seed_costs)),
                    'std_cost': float(np.std(seed_costs)),
                    'mean_success_rate': float(np.mean(seed_success_rates)),
                    'std_success_rate': float(np.std(seed_success_rates)),
                }

    return aggregated


def generate_suite_report(results: Dict, config: Dict, output_dir: Path):
    """
    生成综合评估报告和图表
    """
    print("\n" + "=" * 80)
    print("生成综合评估报告和图表")
    print("=" * 80)

    algorithms = config['algorithms']
    benchmarks = config['benchmarks']
    load_patterns = config['environment']['load_patterns']

    # 创建图表目录
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # 1. 整体性能对比柱状图
    plot_overall_performance(results, algorithms, benchmarks, load_patterns, fig_dir)

    # 2. 负载模式对比图
    plot_workload_pattern_comparison(
        results,
        algorithms,
        benchmarks,
        load_patterns,
        fig_dir,
    )

    # 3. Benchmark分组柱状图
    plot_benchmark_comparison(results, algorithms, benchmarks, load_patterns, fig_dir)

    # 4. 帕累托前沿散点图
    plot_pareto_frontier(results, algorithms, benchmarks, load_patterns, fig_dir)

    # 5. 性能热力图
    plot_heatmap(results, algorithms, benchmarks, load_patterns, fig_dir)

    # 6. 统计显著性分析
    plot_statistical_analysis(results, algorithms, benchmarks, load_patterns, fig_dir)

    print(f"\n所有图表已保存到: {fig_dir}")


def plot_overall_performance(results, algorithms, benchmarks, load_patterns, fig_dir):
    """图表1: 整体性能对比"""
    print("\n[1/6] 生成整体性能对比图...")

    _apply_pi_style()
    algo_order = _present_paper_algo_order(algorithms)
    fig, ax = plt.subplots(figsize=(5.9, 3.7))

    # 计算每个算法的平均奖励
    avg_rewards = {}
    for algorithm in algo_order:
        rewards = []
        for benchmark in benchmarks:
            for pattern in load_patterns:
                rewards.append(results[algorithm][benchmark][pattern]['mean_reward'])
        avg_rewards[algorithm] = np.mean(rewards)

    x = np.arange(len(algo_order))
    reward_values = [avg_rewards[alg] for alg in algo_order]
    bars = ax.bar(
        x,
        reward_values,
        color=[PLOT_COLORS.get(alg, "#cccccc") for alg in algo_order],
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        alpha=0.97,
    )

    ax.set_ylabel('Mean Reward')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [ALGO_LABELS.get(alg, alg.upper()) for alg in algo_order],
        rotation=16,
        ha='right',
    )
    y_min = min(reward_values)
    y_max = max(reward_values)
    span = max(y_max - y_min, 0.1)
    ax.set_ylim(y_min - 0.18 * span - 0.04, y_max + 0.10 * span + 0.03)
    _style_axes(ax)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        offset = 4 if height >= 0 else -10
        va = 'bottom' if height >= 0 else 'top'
        ax.annotate(
            f'{height:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, offset),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=8.8,
        )

    plt.tight_layout()
    _save_figure(fig, fig_dir / '1_overall_performance')


def plot_workload_pattern_comparison(
    results,
    algorithms,
    benchmarks,
    load_patterns,
    fig_dir,
):
    """图表2: 不同负载模式下的 reward 对比"""
    print("[2/6] 生成负载模式对比图...")

    _apply_pi_style()
    algo_order = _present_paper_algo_order(algorithms)
    fig, ax = plt.subplots(figsize=(5.9, 3.7))

    x = np.arange(len(load_patterns))
    markers = {
        'ppo': 'o',
        'bayes_opt_online': 's',
        'greedy': '^',
        'default': 'D',
        'random': 'v',
        'ppo_load_only': 'P',
    }
    linestyles = {
        'ppo': '-',
        'bayes_opt_online': '-',
        'greedy': '--',
        'default': '-.',
        'random': ':',
        'ppo_load_only': '--',
    }

    for algorithm in algo_order:
        means = []
        for pattern in load_patterns:
            reward_values = [
                results[algorithm][benchmark][pattern]['mean_reward']
                for benchmark in benchmarks
            ]
            means.append(float(np.mean(reward_values)))

        means_array = np.asarray(means, dtype=float)
        color = PLOT_COLORS.get(algorithm, "#cccccc")
        ax.plot(
            x,
            means_array,
            marker=markers.get(algorithm, 'o'),
            markersize=5.0,
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=0.8,
            linewidth=2.3 if algorithm == 'ppo' else 1.8,
            linestyle=linestyles.get(algorithm, '-'),
            color=color,
            label=ALGO_LABELS.get(algorithm, algorithm.upper()),
            zorder=3 if algorithm == 'ppo' else 2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PATTERN_LABELS.get(p, p.capitalize()) for p in load_patterns])
    ax.set_ylabel('Mean Reward')
    _style_axes(ax)
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=True,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.8,
        columnspacing=0.9,
    )
    _style_legend(legend)

    plt.tight_layout()
    _save_figure(fig, fig_dir / '2_workload_pattern_comparison')


def plot_benchmark_comparison(results, algorithms, benchmarks, load_patterns, fig_dir):
    """图表3: Benchmark分组对比"""
    print("[3/6] 生成Benchmark分组对比图...")

    _apply_pi_style()
    algo_order = _present_paper_algo_order(algorithms)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    # 计算每个benchmark的平均奖励
    benchmark_rewards = {alg: [] for alg in algo_order}

    for benchmark in benchmarks:
        for algorithm in algo_order:
            rewards = []
            for pattern in load_patterns:
                rewards.append(results[algorithm][benchmark][pattern]['mean_reward'])
            benchmark_rewards[algorithm].append(np.mean(rewards))

    x = np.arange(len(benchmarks))
    width = 0.13

    for i, algorithm in enumerate(algo_order):
        offset = (i - (len(algo_order) - 1) / 2.0) * width * 1.3
        ax.bar(
            x + offset,
            benchmark_rewards[algorithm],
            width,
            label=ALGO_LABELS.get(algorithm, algorithm.upper()),
            color=PLOT_COLORS.get(algorithm, "#cccccc"),
            edgecolor=EDGE_COLOR,
            linewidth=1.0,
            alpha=0.97,
        )

    ax.set_ylabel('Mean Reward')
    ax.set_xticks(x)
    ax.set_xticklabels(
        [BENCH_LABELS.get(bench, bench) for bench in benchmarks],
        rotation=16,
        ha='right',
    )
    _style_axes(ax)
    legend = ax.legend(
        ncol=1,
        loc='lower right',
        frameon=True,
        borderpad=0.45,
        labelspacing=0.35,
    )
    _style_legend(legend)

    plt.tight_layout()
    _save_figure(fig, fig_dir / '3_benchmark_comparison')


def plot_pareto_frontier(results, algorithms, benchmarks, load_patterns, fig_dir):
    """图表4: 帕累托前沿（延迟-成本）"""
    print("[4/6] 生成帕累托前沿图...")

    _apply_pi_style()
    algo_order = _present_paper_algo_order(algorithms)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    markers = {
        'ppo': 'o',
        'bayes_opt_online': 's',
        'greedy': '^',
        'default': 'D',
        'random': 'v',
        'ppo_load_only': 'P',
    }

    for algorithm in algo_order:
        latencies = []
        costs = []

        for benchmark in benchmarks:
            for pattern in load_patterns:
                latencies.append(results[algorithm][benchmark][pattern]['mean_latency'])
                costs.append(results[algorithm][benchmark][pattern]['mean_cost'] * 1e6)

        latencies_array = np.asarray(latencies, dtype=float)
        costs_array = np.asarray(costs, dtype=float)
        color = PLOT_COLORS.get(algorithm, "#cccccc")

        ax.scatter(
            latencies_array,
            costs_array,
            s=52 if algorithm == 'ppo' else 44,
            alpha=0.62,
            facecolors=_lighten(color, 0.18),
            color=color,
            marker=markers.get(algorithm, 'o'),
            edgecolors=EDGE_COLOR,
            linewidth=0.7,
            zorder=2,
        )
        ax.scatter(
            float(np.mean(latencies_array)),
            float(np.mean(costs_array)),
            s=140 if algorithm == 'ppo' else 112,
            alpha=0.98,
            color=color,
            marker=markers.get(algorithm, 'o'),
            label=ALGO_LABELS.get(algorithm, algorithm.upper()),
            edgecolors=EDGE_COLOR,
            linewidth=1.1,
            zorder=4,
        )

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cost (μUSD)')
    _style_axes(ax, grid_axis='both')
    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=True,
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.6,
        columnspacing=0.9,
    )
    _style_legend(legend)

    plt.tight_layout()
    _save_figure(fig, fig_dir / '4_pareto_frontier')


def plot_heatmap(results, algorithms, benchmarks, load_patterns, fig_dir):
    """图表5: 性能热力图"""
    print("[5/6] 生成性能热力图...")

    _apply_pi_style()
    fig, axes = plt.subplots(
        1,
        len(algorithms),
        figsize=(len(algorithms) * 3.0, 4.6),
        gridspec_kw={"wspace": 0.22},
    )

    all_vals = []
    for algorithm in algorithms:
        for benchmark in benchmarks:
            for pattern in load_patterns:
                all_vals.append(results[algorithm][benchmark][pattern]['mean_reward'])
    max_abs = max(abs(np.min(all_vals)), abs(np.max(all_vals)), 0.01)
    vmin, vmax = -max_abs, max_abs
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = sns.diverging_palette(145, 15, as_cmap=True)

    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx] if len(algorithms) > 1 else axes

        # 创建热力图数据
        heatmap_data = np.zeros((len(benchmarks), len(load_patterns)))

        for i, benchmark in enumerate(benchmarks):
            for j, pattern in enumerate(load_patterns):
                heatmap_data[i, j] = results[algorithm][benchmark][pattern]['mean_reward']

        # 绘制热力图（使用 seaborn 统一排版）
        im = sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar=False,
            square=True,
            linewidths=0.4,
            linecolor="#f0f0f0",
        )

        ax.set_xticks(np.arange(len(load_patterns)) + 0.5)
        ax.set_yticks(np.arange(len(benchmarks)) + 0.5)
        ax.set_facecolor(FACE_COLOR)
        ax.set_xticklabels(
            [PATTERN_LABELS.get(p, p.capitalize()) for p in load_patterns],
            rotation=28,
            ha='right',
            fontsize=8,
        )
        ylabels = [BENCH_LABELS.get(b, b) for b in benchmarks]
        if idx == 0:
            ax.set_yticklabels(ylabels, fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
        else:
            ax.set_yticklabels([])
        ax.set_title(ALGO_LABELS.get(algorithm, algorithm.upper()), fontsize=10.2, pad=6)
        ax.tick_params(axis='both', which='both', length=0, colors=EDGE_COLOR)

    # 统一颜色条，放置在右侧单独轴，避免与子图重叠
    cbar_ax = fig.add_axes([0.91, 0.2, 0.015, 0.6])
    fig.colorbar(im.collections[0], cax=cbar_ax)
    cbar_ax.set_title('Reward', fontsize=8, pad=6)
    cbar_ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.12, right=0.9, bottom=0.18, top=0.9, wspace=0.25)
    _save_figure(fig, fig_dir / '5_performance_heatmap')


def plot_statistical_analysis(results, algorithms, benchmarks, load_patterns, fig_dir):
    """图表6: 统计显著性分析（箱线图）"""
    print("[6/6] 生成统计显著性分析图...")

    _apply_pi_style()
    algo_order = _present_paper_algo_order(algorithms)
    fig, ax = plt.subplots(figsize=(5.9, 3.8))

    # 收集所有场景的奖励数据
    data = {alg: [] for alg in algo_order}

    for algorithm in algo_order:
        for benchmark in benchmarks:
            for pattern in load_patterns:
                data[algorithm].append(results[algorithm][benchmark][pattern]['mean_reward'])

    # 绘制箱线图
    positions = np.arange(len(algo_order))
    bp = ax.boxplot(
        [data[alg] for alg in algo_order],
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        meanprops={
            'marker': 'D',
            'markerfacecolor': 'white',
            'markeredgecolor': EDGE_COLOR,
            'markersize': 5,
        },
        medianprops={'color': EDGE_COLOR, 'linewidth': 1.5},
        whiskerprops={'color': EDGE_COLOR, 'linewidth': 1.1},
        capprops={'color': EDGE_COLOR, 'linewidth': 1.1},
        flierprops={
            'marker': 'o',
            'markerfacecolor': '#ffffff',
            'markeredgecolor': EDGE_COLOR,
            'markersize': 4,
            'alpha': 0.75,
        },
    )

    for patch, algorithm in zip(bp['boxes'], algo_order):
        patch.set_facecolor(_lighten(PLOT_COLORS.get(algorithm, "#cccccc"), 0.12))
        patch.set_edgecolor(EDGE_COLOR)
        patch.set_linewidth(1.1)
        patch.set_alpha(0.97)

    ax.axhline(0.0, color=LEGEND_EDGE, linewidth=0.9, linestyle='--', zorder=0)
    ax.set_ylabel('Reward Distribution')
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [ALGO_LABELS.get(alg, alg.upper()) for alg in algo_order],
        rotation=16,
        ha='right',
    )
    _style_axes(ax)

    plt.tight_layout()
    _save_figure(fig, fig_dir / '6_statistical_analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行 CALO 全量组合实验')
    parser.add_argument(
        '--config',
        type=str,
        default='config/rl_experiments/full_suite.json',
        help='配置文件路径',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_full_suite',
        help='输出目录',
    )
    parser.add_argument(
        '--results-json',
        type=str,
        default='',
        help='仅重绘图表时使用的聚合结果 JSON 路径',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='调试模式',
    )

    args = parser.parse_args()

    if args.results_json:
        with open(args.config, 'r', encoding='utf-8') as handle:
            config = json.load(handle)
        with open(args.results_json, 'r', encoding='utf-8') as handle:
            results = json.load(handle)
        generate_suite_report(results, config, Path(args.output_dir))
    else:
        run_full_suite(args.config, args.output_dir, args.debug)
