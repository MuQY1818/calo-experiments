"""
结果分析脚本
分析泛化性测试的实验结果
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict


def load_results(results_path: str) -> Dict:
    """加载实验结果"""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_algorithm_performance(results: Dict, algorithm: str) -> Dict:
    """分析单个算法的性能"""
    algo_results = results.get(algorithm, {})

    summary = {
        'benchmarks': {},
        'overall': {}
    }

    all_rewards = []
    all_latencies = []
    all_costs = []
    all_success_rates = []

    for benchmark, runs in algo_results.items():
        if len(runs) == 0:
            continue

        rewards = [r['mean_reward'] for r in runs]
        latencies = [r['mean_latency'] for r in runs]
        costs = [r['mean_cost'] for r in runs]
        success_rates = [r['mean_success_rate'] for r in runs]

        summary['benchmarks'][benchmark] = {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'mean_latency': float(np.mean(latencies)),
            'std_latency': float(np.std(latencies)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'mean_success_rate': float(np.mean(success_rates)),
            'n_runs': len(runs)
        }

        all_rewards.extend(rewards)
        all_latencies.extend(latencies)
        all_costs.extend(costs)
        all_success_rates.extend(success_rates)

    if len(all_rewards) > 0:
        summary['overall'] = {
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'mean_latency': float(np.mean(all_latencies)),
            'std_latency': float(np.std(all_latencies)),
            'mean_cost': float(np.mean(all_costs)),
            'std_cost': float(np.std(all_costs)),
            'mean_success_rate': float(np.mean(all_success_rates)),
            'n_benchmarks': len(summary['benchmarks']),
            'total_runs': len(all_rewards)
        }

    return summary


def compare_algorithms(results: Dict, algo1: str, algo2: str) -> Dict:
    """对比两个算法"""
    summary1 = analyze_algorithm_performance(results, algo1)
    summary2 = analyze_algorithm_performance(results, algo2)

    comparison = {
        'benchmark_level': {},
        'overall': {}
    }

    # 对比每个benchmark
    for benchmark in summary1['benchmarks']:
        if benchmark in summary2['benchmarks']:
            s1 = summary1['benchmarks'][benchmark]
            s2 = summary2['benchmarks'][benchmark]

            improvement = ((s1['mean_reward'] - s2['mean_reward']) / abs(s2['mean_reward'])) * 100

            comparison['benchmark_level'][benchmark] = {
                f'{algo1}_reward': s1['mean_reward'],
                f'{algo2}_reward': s2['mean_reward'],
                'improvement_pct': improvement,
                f'{algo1}_latency': s1['mean_latency'],
                f'{algo2}_latency': s2['mean_latency'],
                f'{algo1}_success_rate': s1['mean_success_rate'],
                f'{algo2}_success_rate': s2['mean_success_rate']
            }

    # 总体对比
    if 'overall' in summary1 and 'overall' in summary2:
        o1 = summary1['overall']
        o2 = summary2['overall']

        overall_improvement = ((o1['mean_reward'] - o2['mean_reward']) / abs(o2['mean_reward'])) * 100

        comparison['overall'] = {
            f'{algo1}_reward': o1['mean_reward'],
            f'{algo2}_reward': o2['mean_reward'],
            'improvement_pct': overall_improvement,
            f'{algo1}_std': o1['std_reward'],
            f'{algo2}_std': o2['std_reward'],
            'variance_reduction_pct': ((o2['std_reward'] - o1['std_reward']) / o2['std_reward']) * 100
        }

    return comparison


def print_summary(results_path: str):
    """打印实验总结"""
    results = load_results(results_path)

    print("=" * 70)
    print("泛化性测试结果分析")
    print("=" * 70)

    # 分析SE-PPO
    se_ppo_summary = analyze_algorithm_performance(results, 'se_ppo')
    print("\n【SE-PPO性能】")
    print(f"\n总体:")
    if 'overall' in se_ppo_summary:
        o = se_ppo_summary['overall']
        print(f"  平均奖励: {o['mean_reward']:.4f} ± {o['std_reward']:.4f}")
        print(f"  平均延迟: {o['mean_latency']:.2f} ± {o['std_latency']:.2f} ms")
        print(f"  成功率: {o['mean_success_rate']:.2%}")
        print(f"  测试数: {o['total_runs']} runs on {o['n_benchmarks']} benchmarks")

    print("\n各Benchmark:")
    for benchmark, stats in se_ppo_summary['benchmarks'].items():
        print(f"  {benchmark}:")
        print(f"    奖励: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
        print(f"    延迟: {stats['mean_latency']:.2f} ms")
        print(f"    成功率: {stats['mean_success_rate']:.2%}")

    # 分析Random
    random_summary = analyze_algorithm_performance(results, 'random')
    print("\n【Random Baseline】")
    print(f"\n总体:")
    if 'overall' in random_summary:
        o = random_summary['overall']
        print(f"  平均奖励: {o['mean_reward']:.4f} ± {o['std_reward']:.4f}")
        print(f"  平均延迟: {o['mean_latency']:.2f} ± {o['std_latency']:.2f} ms")
        print(f"  成功率: {o['mean_success_rate']:.2%}")

    # 对比
    comparison = compare_algorithms(results, 'se_ppo', 'random')
    print("\n【SE-PPO vs Random对比】")
    print(f"\n总体:")
    if 'overall' in comparison:
        o = comparison['overall']
        print(f"  SE-PPO奖励: {o['se_ppo_reward']:.4f}")
        print(f"  Random奖励: {o['random_reward']:.4f}")
        print(f"  改进: {o['improvement_pct']:.2f}%")
        print(f"  方差降低: {o['variance_reduction_pct']:.2f}%")

    print("\n各Benchmark对比:")
    for benchmark, stats in comparison['benchmark_level'].items():
        print(f"  {benchmark}:")
        print(f"    SE-PPO: {stats['se_ppo_reward']:.4f}")
        print(f"    Random: {stats['random_reward']:.4f}")
        print(f"    改进: {stats['improvement_pct']:.2f}%")

    print("\n" + "=" * 70)

    # 保存汇总
    output_dir = Path(results_path).parent
    summary_output = {
        'se_ppo': se_ppo_summary,
        'random': random_summary,
        'comparison': comparison
    }

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary_output, f, indent=2)

    print(f"\n分析结果已保存到: {output_dir / 'analysis_summary.json'}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = 'results_generalization/logs/experiment_results.json'

    print_summary(results_path)
