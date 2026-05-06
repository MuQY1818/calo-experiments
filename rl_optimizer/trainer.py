"""
Training Script for SE-PPO and Baselines
完整的训练脚本，支持多个benchmark和算法
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

from .environment import ServerlessFunctionEnv
from .surrogate_model import GaussianProcessSurrogate
from .se_ppo import SampleEfficientPPO
from .baselines import (
    DefaultPolicy, RandomPolicy, GreedyPolicy,
    RuleBasedPolicy, GridSearchPolicy, BayesianOptimizationPolicy
)


class Trainer:
    """训练器"""

    def __init__(
        self,
        config_path: str = None,
        output_dir: str = 'results',
        verbose: int = 1
    ):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            verbose: 日志级别
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)

        print(f"[Trainer] 初始化完成")
        print(f"  输出目录: {self.output_dir}")
        print(f"  Benchmark数量: {len(self.config['benchmarks'])}")

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'benchmarks': [
                '110.dynamic-html',
                '120.uploader',
                '210.thumbnailer',
                '311.compression',
                '411.image-recognition',
                '501.graph-pagerank',
                '502.graph-mst',
            ],
            'algorithms': ['se_ppo', 'default', 'random', 'greedy', 'rule_based'],
            'training': {
                'total_timesteps': 2000,
                'real_ratio': 0.2,
                'surrogate_update_freq': 100,
                'max_steps_per_episode': 50,
            },
            'evaluation': {
                'n_episodes': 10,
            },
            'environment': {
                'deployment': 'local',
                'enable_real_execution': False,
            }
        }

    def train_algorithm(
        self,
        algorithm: str,
        benchmark: str,
        seed: int = 42
    ) -> Dict:
        """
        训练单个算法

        Args:
            algorithm: 算法名称
            benchmark: benchmark名称
            seed: 随机种子

        Returns:
            训练结果
        """
        print(f"\n{'='*60}")
        print(f"训练: {algorithm} on {benchmark}")
        print(f"{'='*60}")

        # 创建环境
        env = ServerlessFunctionEnv(
            benchmark=benchmark,
            deployment=self.config['environment']['deployment'],
            enable_real_execution=self.config['environment']['enable_real_execution'],
            max_steps=self.config['training']['max_steps_per_episode'],
            action_space_config=self.config['environment'].get('action_space'),
        )

        # 设置随机种子
        np.random.seed(seed)

        start_time = time.time()

        if algorithm == 'se_ppo':
            result = self._train_se_ppo(env, benchmark, seed)
        elif algorithm == 'default':
            result = self._evaluate_baseline(DefaultPolicy(env), "Default")
        elif algorithm == 'random':
            result = self._evaluate_baseline(RandomPolicy(env), "Random")
        elif algorithm == 'greedy':
            result = self._evaluate_baseline(GreedyPolicy(env), "Greedy")
        elif algorithm == 'rule_based':
            result = self._evaluate_baseline(RuleBasedPolicy(env), "RuleBased")
        elif algorithm == 'grid_search':
            result = self._train_grid_search(env, benchmark)
        elif algorithm == 'bayes_opt':
            result = self._train_bayes_opt(env, benchmark)
        else:
            raise ValueError(f"未知算法: {algorithm}")

        training_time = time.time() - start_time

        result['training_time'] = training_time
        result['algorithm'] = algorithm
        result['benchmark'] = benchmark
        result['seed'] = seed

        print(f"\n训练完成，耗时: {training_time:.2f}秒")
        print(f"平均奖励: {result.get('mean_reward', 0):.4f}")
        print(f"平均延迟: {result.get('mean_latency', 0):.2f} ms")
        print(f"平均成本: ${result.get('mean_cost', 0):.6f}")

        return result

    def _train_se_ppo(self, env: ServerlessFunctionEnv, benchmark: str, seed: int) -> Dict:
        """训练SE-PPO"""
        surrogate = GaussianProcessSurrogate(
            state_dim=79,
            action_dim=env.action_space_wrapper.n_actions,
            use_sklearn=True,
        )

        # 获取PPO超参数（如果配置中有）
        ppo_kwargs = self.config['training'].get('ppo_kwargs', None)

        agent = SampleEfficientPPO(
            env=env,
            surrogate_model=surrogate,
            total_timesteps=self.config['training']['total_timesteps'],
            real_ratio=self.config['training']['real_ratio'],
            surrogate_update_freq=self.config['training']['surrogate_update_freq'],
            ppo_kwargs=ppo_kwargs,
            verbose=self.verbose,
        )

        # 训练
        training_stats = agent.train()

        # 评估
        eval_results = agent.evaluate(n_episodes=self.config['evaluation']['n_episodes'])

        # 保存模型
        model_path = self.output_dir / 'models' / f'se_ppo_{benchmark}_{seed}.zip'
        agent.save(str(model_path))

        # 合并结果
        result = {
            **eval_results,
            'training_stats': training_stats,
            'real_interactions': training_stats['real_interactions'],
            'surrogate_interactions': training_stats['surrogate_interactions'],
            'sample_efficiency': training_stats['surrogate_interactions'] / max(training_stats['real_interactions'], 1),
        }

        return result

    def _evaluate_baseline(self, policy, name: str) -> Dict:
        """评估baseline策略"""
        print(f"评估 {name}...")
        result = policy.evaluate(n_episodes=self.config['evaluation']['n_episodes'])
        return result

    def _train_grid_search(self, env: ServerlessFunctionEnv, benchmark: str) -> Dict:
        """训练Grid Search"""
        grid_search = GridSearchPolicy(env)
        grid_search.search(steps_per_config=5)
        result = grid_search.evaluate(n_episodes=self.config['evaluation']['n_episodes'])
        return result

    def _train_bayes_opt(self, env: ServerlessFunctionEnv, benchmark: str) -> Dict:
        """训练Bayesian Optimization"""
        bayes_opt = BayesianOptimizationPolicy(env)
        bayes_opt.optimize(n_iterations=50)
        result = bayes_opt.evaluate(n_episodes=self.config['evaluation']['n_episodes'])
        return result

    def run_experiments(
        self,
        algorithms: List[str] = None,
        benchmarks: List[str] = None,
        n_seeds: int = 3
    ) -> Dict:
        """
        运行完整实验

        Args:
            algorithms: 算法列表（None表示所有）
            benchmarks: benchmark列表（None表示所有）
            n_seeds: 每个配置运行的随机种子数

        Returns:
            所有实验结果
        """
        if algorithms is None:
            algorithms = self.config['algorithms']
        if benchmarks is None:
            benchmarks = self.config['benchmarks']

        print(f"\n{'='*60}")
        print(f"开始实验")
        print(f"{'='*60}")
        print(f"算法: {algorithms}")
        print(f"Benchmark: {benchmarks}")
        print(f"随机种子数: {n_seeds}")
        print(f"总实验数: {len(algorithms) * len(benchmarks) * n_seeds}")

        all_results = {}

        for algorithm in algorithms:
            all_results[algorithm] = {}

            for benchmark in benchmarks:
                all_results[algorithm][benchmark] = []

                for seed in range(n_seeds):
                    try:
                        result = self.train_algorithm(algorithm, benchmark, seed)
                        all_results[algorithm][benchmark].append(result)

                        # 保存中间结果
                        self._save_results(all_results)

                    except Exception as e:
                        print(f"\n[Error] {algorithm} on {benchmark} (seed={seed}) 失败: {e}")
                        import traceback
                        traceback.print_exc()

        # 保存最终结果
        self._save_results(all_results)

        # 打印汇总
        self._print_summary(all_results)

        return all_results

    def _save_results(self, results: Dict):
        """保存结果到JSON文件"""
        output_file = self.output_dir / 'logs' / 'experiment_results.json'

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

        serializable_results = convert_to_serializable(results)

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n[Save] 结果已保存到: {output_file}")

    def _print_summary(self, results: Dict):
        """打印实验汇总"""
        print(f"\n{'='*60}")
        print("实验汇总")
        print(f"{'='*60}")

        for algorithm in results:
            print(f"\n{algorithm}:")

            for benchmark in results[algorithm]:
                if len(results[algorithm][benchmark]) == 0:
                    continue

                rewards = [r['mean_reward'] for r in results[algorithm][benchmark]]
                latencies = [r['mean_latency'] for r in results[algorithm][benchmark]]
                costs = [r['mean_cost'] for r in results[algorithm][benchmark]]

                print(f"  {benchmark}:")
                print(f"    奖励: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
                print(f"    延迟: {np.mean(latencies):.2f} ± {np.std(latencies):.2f} ms")
                print(f"    成本: ${np.mean(costs):.6f} ± ${np.std(costs):.6f}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='SE-PPO训练脚本')

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='输出目录'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default=None,
        help='算法名称（不指定则运行所有）'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        default=None,
        help='Benchmark名称（不指定则运行所有）'
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=3,
        help='随机种子数'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='日志级别'
    )

    args = parser.parse_args()

    # 创建训练器
    trainer = Trainer(
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=args.verbose
    )

    # 运行实验
    algorithms = [args.algorithm] if args.algorithm else None
    benchmarks = [args.benchmark] if args.benchmark else None

    trainer.run_experiments(
        algorithms=algorithms,
        benchmarks=benchmarks,
        n_seeds=args.n_seeds
    )


if __name__ == '__main__':
    main()
