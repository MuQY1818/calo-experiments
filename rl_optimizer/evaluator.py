"""
Evaluation and Visualization Script
评估模型性能并生成可视化图表
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .environment import ServerlessFunctionEnv
from stable_baselines3 import PPO


class Evaluator:
    """评估器"""

    def __init__(self, results_dir: str = 'results'):
        """
        初始化评估器

        Args:
            results_dir: 结果目录
        """
        self.results_dir = Path(results_dir)
        self.logs_dir = self.results_dir / 'logs'
        self.models_dir = self.results_dir / 'models'
        self.figures_dir = self.results_dir / 'figures'

        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # 加载结果
        self.results = self._load_results()

        print(f"[Evaluator] 初始化完成")
        print(f"  结果目录: {self.results_dir}")

    def _load_results(self) -> Dict:
        """加载实验结果"""
        results_file = self.logs_dir / 'experiment_results.json'

        if not results_file.exists():
            print(f"[Warning] 结果文件不存在: {results_file}")
            return {}

        with open(results_file, 'r') as f:
            results = json.load(f)

        print(f"[Load] 加载了 {len(results)} 个算法的结果")
        return results

    def compare_algorithms(self, metric: str = 'mean_reward') -> Dict:
        """
        对比不同算法的性能

        Args:
            metric: 对比的指标 ['mean_reward', 'mean_latency', 'mean_cost']

        Returns:
            对比结果
        """
        if not self.results:
            print("[Error] 没有可用的结果")
            return {}

        print(f"\n{'='*60}")
        print(f"算法性能对比 ({metric})")
        print(f"{'='*60}")

        comparison = {}

        for algorithm in self.results:
            comparison[algorithm] = {}

            for benchmark in self.results[algorithm]:
                if len(self.results[algorithm][benchmark]) == 0:
                    continue

                values = [r[metric] for r in self.results[algorithm][benchmark]]
                comparison[algorithm][benchmark] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values,
                }

        # 打印表格
        print(f"\n{'':<15}", end='')
        benchmarks = list(next(iter(comparison.values())).keys())
        for bm in benchmarks:
            print(f"{bm:<20}", end='')
        print()
        print("-" * (15 + 20 * len(benchmarks)))

        for algorithm in comparison:
            print(f"{algorithm:<15}", end='')
            for benchmark in benchmarks:
                if benchmark in comparison[algorithm]:
                    mean = comparison[algorithm][benchmark]['mean']
                    std = comparison[algorithm][benchmark]['std']
                    print(f"{mean:>10.4f}±{std:<8.4f}", end='')
                else:
                    print(f"{'N/A':<20}", end='')
            print()

        return comparison

    def plot_algorithm_comparison(self, metric: str = 'mean_reward'):
        """
        绘制算法对比图

        Args:
            metric: 对比指标
        """
        comparison = self.compare_algorithms(metric)

        if not comparison:
            return

        # 准备数据
        algorithms = list(comparison.keys())
        benchmarks = list(next(iter(comparison.values())).keys())

        data = []
        for algorithm in algorithms:
            for benchmark in benchmarks:
                if benchmark in comparison[algorithm]:
                    data.append({
                        'algorithm': algorithm,
                        'benchmark': benchmark,
                        'value': comparison[algorithm][benchmark]['mean'],
                        'std': comparison[algorithm][benchmark]['std'],
                    })

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))

        # 分组柱状图
        x = np.arange(len(benchmarks))
        width = 0.8 / len(algorithms)

        for i, algorithm in enumerate(algorithms):
            means = []
            stds = []
            for benchmark in benchmarks:
                if benchmark in comparison[algorithm]:
                    means.append(comparison[algorithm][benchmark]['mean'])
                    stds.append(comparison[algorithm][benchmark]['std'])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(
                x + i * width,
                means,
                width,
                yerr=stds,
                label=algorithm,
                capsize=3
            )

        ax.set_xlabel('Benchmark')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Algorithm Comparison ({metric})')
        ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        output_file = self.figures_dir / f'comparison_{metric}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n[Save] 图表已保存: {output_file}")

        plt.close()

    def analyze_sample_efficiency(self):
        """分析样本效率"""
        if 'se_ppo' not in self.results:
            print("[Warning] 没有SE-PPO的结果")
            return

        print(f"\n{'='*60}")
        print("样本效率分析")
        print(f"{'='*60}")

        for benchmark in self.results['se_ppo']:
            if len(self.results['se_ppo'][benchmark]) == 0:
                continue

            sample_efficiencies = []
            real_interactions = []
            surrogate_interactions = []

            for result in self.results['se_ppo'][benchmark]:
                if 'sample_efficiency' in result:
                    sample_efficiencies.append(result['sample_efficiency'])
                if 'real_interactions' in result:
                    real_interactions.append(result['real_interactions'])
                if 'surrogate_interactions' in result:
                    surrogate_interactions.append(result['surrogate_interactions'])

            print(f"\n{benchmark}:")
            print(f"  样本效率: {np.mean(sample_efficiencies):.2f}x ± {np.std(sample_efficiencies):.2f}x")
            print(f"  真实交互: {np.mean(real_interactions):.0f} ± {np.std(real_interactions):.0f}")
            print(f"  代理交互: {np.mean(surrogate_interactions):.0f} ± {np.std(surrogate_interactions):.0f}")

    def generate_report(self, output_file: str = None):
        """
        生成完整报告

        Args:
            output_file: 输出文件路径（Markdown格式）
        """
        if output_file is None:
            output_file = self.results_dir / 'logs' / 'evaluation_report.md'

        lines = []
        lines.append("# Experimental Results Report\n")
        lines.append(f"\n## Overview\n")
        lines.append(f"- Algorithms: {list(self.results.keys())}\n")

        if self.results:
            lines.append(f"- Benchmarks: {list(next(iter(self.results.values())).keys())}\n")

        lines.append(f"\n## Performance Comparison\n")

        for metric in ['mean_reward', 'mean_latency', 'mean_cost']:
            lines.append(f"\n### {metric.replace('_', ' ').title()}\n")

            comparison = self.compare_algorithms(metric)

            if comparison:
                # 创建Markdown表格
                algorithms = list(comparison.keys())
                benchmarks = list(next(iter(comparison.values())).keys())

                # 表头
                lines.append(f"| Algorithm | " + " | ".join(benchmarks) + " |\n")
                lines.append(f"| --- | " + " | ".join(['---'] * len(benchmarks)) + " |\n")

                # 数据行
                for algorithm in algorithms:
                    row = f"| {algorithm} |"
                    for benchmark in benchmarks:
                        if benchmark in comparison[algorithm]:
                            mean = comparison[algorithm][benchmark]['mean']
                            std = comparison[algorithm][benchmark]['std']
                            row += f" {mean:.4f}±{std:.4f} |"
                        else:
                            row += " N/A |"
                    lines.append(row + "\n")

        lines.append(f"\n## Sample Efficiency (SE-PPO)\n")

        if 'se_ppo' in self.results:
            for benchmark in self.results['se_ppo']:
                if len(self.results['se_ppo'][benchmark]) == 0:
                    continue

                sample_efficiencies = [
                    r.get('sample_efficiency', 0)
                    for r in self.results['se_ppo'][benchmark]
                ]

                lines.append(f"- {benchmark}: {np.mean(sample_efficiencies):.2f}x\n")

        # 保存报告
        with open(output_file, 'w') as f:
            f.writelines(lines)

        print(f"\n[Save] 报告已保存: {output_file}")

    def plot_all_metrics(self):
        """绘制所有指标的对比图"""
        for metric in ['mean_reward', 'mean_latency', 'mean_cost']:
            try:
                self.plot_algorithm_comparison(metric)
            except Exception as e:
                print(f"[Error] 绘制 {metric} 失败: {e}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='SE-PPO评估脚本')

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='结果目录'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='mean_reward',
        choices=['mean_reward', 'mean_latency', 'mean_cost'],
        help='对比指标'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='生成图表'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='生成报告'
    )

    args = parser.parse_args()

    # 创建评估器
    evaluator = Evaluator(results_dir=args.results_dir)

    # 对比算法
    evaluator.compare_algorithms(metric=args.metric)

    # 分析样本效率
    evaluator.analyze_sample_efficiency()

    # 生成图表
    if args.plot:
        evaluator.plot_all_metrics()

    # 生成报告
    if args.report:
        evaluator.generate_report()


if __name__ == '__main__':
    main()
