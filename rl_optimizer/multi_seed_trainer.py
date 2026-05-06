"""
Multi-Seed Trainer
运行多个随机种子，选择最佳模型，提高训练稳定性
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from .trainer import Trainer


class MultiSeedTrainer:
    """多seed训练器，选择最佳模型"""

    def __init__(
        self,
        config_path: str,
        output_dir: str = 'results_multi_seed',
        n_seeds: int = 5,
        selection_metric: str = 'mean_reward',
        verbose: int = 1
    ):
        """
        初始化多seed训练器

        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            n_seeds: 训练的随机种子数量
            selection_metric: 选择最佳模型的指标 (mean_reward, mean_latency, mean_cost)
            verbose: 日志级别
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.n_seeds = n_seeds
        self.selection_metric = selection_metric
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[MultiSeedTrainer] 初始化完成")
        print(f"  输出目录: {self.output_dir}")
        print(f"  随机种子数: {n_seeds}")
        print(f"  选择指标: {selection_metric}")

    def train_all_seeds(
        self,
        algorithm: str,
        benchmark: str
    ) -> Dict:
        """
        用多个seed训练同一个算法

        Args:
            algorithm: 算法名称
            benchmark: benchmark名称

        Returns:
            所有seed的结果 + 最佳seed信息
        """
        print(f"\n{'='*60}")
        print(f"Multi-Seed训练: {algorithm} on {benchmark}")
        print(f"{'='*60}")

        results = []
        trainer = Trainer(
            config_path=self.config_path,
            output_dir=str(self.output_dir),
            verbose=self.verbose
        )

        for seed in range(self.n_seeds):
            print(f"\n[Seed {seed}/{self.n_seeds-1}] 开始训练...")

            try:
                result = trainer.train_algorithm(algorithm, benchmark, seed)
                results.append(result)

                # 打印当前结果
                print(f"[Seed {seed}] 完成")
                print(f"  {self.selection_metric}: {result[self.selection_metric]:.4f}")

            except Exception as e:
                print(f"[Error] Seed {seed} 失败: {e}")
                continue

        if len(results) == 0:
            raise RuntimeError("所有seed都失败了！")

        # 选择最佳seed
        best_result = self._select_best(results)

        # 汇总统计
        summary = self._compute_summary(results, best_result)

        return summary

    def _select_best(self, results: List[Dict]) -> Dict:
        """
        根据selection_metric选择最佳结果

        对于reward: 越大越好
        对于latency/cost: 越小越好
        """
        if self.selection_metric == 'mean_reward':
            # reward越大（越接近0）越好
            best_idx = np.argmax([r['mean_reward'] for r in results])
        elif self.selection_metric in ['mean_latency', 'mean_cost']:
            # latency/cost越小越好
            best_idx = np.argmin([r[self.selection_metric] for r in results])
        else:
            raise ValueError(f"未知的指标: {self.selection_metric}")

        return results[best_idx]

    def _compute_summary(self, results: List[Dict], best_result: Dict) -> Dict:
        """计算多seed的统计摘要"""
        metrics = ['mean_reward', 'mean_latency', 'mean_cost', 'mean_success_rate']

        summary = {
            'n_seeds': len(results),
            'best_seed': best_result['seed'],
            'best_result': best_result,
            'all_results': results,
            'statistics': {}
        }

        # 计算每个指标的统计
        for metric in metrics:
            if metric in results[0]:
                values = [r[metric] for r in results]
                summary['statistics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'best': best_result[metric]
                }

        return summary

    def print_summary(self, summary: Dict):
        """打印汇总信息"""
        print(f"\n{'='*60}")
        print("Multi-Seed训练汇总")
        print(f"{'='*60}")

        print(f"\n训练Seeds数: {summary['n_seeds']}")
        print(f"最佳Seed: {summary['best_seed']}")

        print("\n各指标统计:")
        for metric, stats in summary['statistics'].items():
            print(f"\n{metric}:")
            print(f"  平均: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  最小: {stats['min']:.4f}")
            print(f"  最大: {stats['max']:.4f}")
            print(f"  最佳: {stats['best']:.4f}")

        # 计算稳定性指标
        reward_cv = (
            summary['statistics']['mean_reward']['std'] /
            abs(summary['statistics']['mean_reward']['mean'])
        )
        print(f"\n稳定性指标 (Coefficient of Variation):")
        print(f"  Reward CV: {reward_cv:.2%}")

    def save_summary(self, summary: Dict, filename: str = 'multi_seed_summary.json'):
        """保存汇总结果"""
        output_file = self.output_dir / filename

        # 转换numpy类型为Python类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_summary = convert_to_serializable(summary)

        with open(output_file, 'w') as f:
            json.dump(serializable_summary, f, indent=2)

        print(f"\n[Save] 汇总结果已保存到: {output_file}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Multi-Seed SE-PPO训练')

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results_multi_seed',
        help='输出目录'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='se_ppo',
        help='算法名称'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Benchmark名称'
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=5,
        help='随机种子数量'
    )
    parser.add_argument(
        '--selection-metric',
        type=str,
        default='mean_reward',
        choices=['mean_reward', 'mean_latency', 'mean_cost'],
        help='选择最佳模型的指标'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='日志级别'
    )

    args = parser.parse_args()

    # 创建多seed训练器
    trainer = MultiSeedTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
        n_seeds=args.n_seeds,
        selection_metric=args.selection_metric,
        verbose=args.verbose
    )

    # 训练所有seeds
    summary = trainer.train_all_seeds(args.algorithm, args.benchmark)

    # 打印汇总
    trainer.print_summary(summary)

    # 保存结果
    trainer.save_summary(summary)


if __name__ == '__main__':
    main()
