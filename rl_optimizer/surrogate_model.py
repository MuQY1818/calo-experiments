"""
Gaussian Process Surrogate Model
高斯过程代理模型，用于减少真实环境调用次数
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings


class GaussianProcessSurrogate:
    """
    高斯过程代理模型

    用途：
    - 学习(state, action) -> (reward, latency, cost)的映射
    - 提供预测的不确定性估计
    - 减少真实环境调用，提高样本效率
    """

    def __init__(
        self,
        state_dim: int = 79,
        action_dim: int = 48,
        kernel: str = 'rbf',
        noise_level: float = 0.1,
        use_sklearn: bool = True,
    ):
        """
        初始化高斯过程代理模型

        Args:
            state_dim: 状态维度
            action_dim: 动作维度（离散动作数量）
            kernel: 核函数类型 ['rbf', 'matern']
            noise_level: 观测噪声水平
            use_sklearn: 是否使用sklearn（True）还是GPy（False）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.kernel = kernel
        self.noise_level = noise_level
        self.use_sklearn = use_sklearn

        # 训练数据
        self.X_train = []  # (state, action)对
        self.y_reward_train = []  # 奖励
        self.y_latency_train = []  # 延迟
        self.y_cost_train = []  # 成本

        # GP模型（延迟初始化）
        self.gp_reward = None
        self.gp_latency = None
        self.gp_cost = None

        # 是否已拟合
        self.is_fitted = False

        print(f"[SurrogateGP] 初始化完成")
        print(f"  状态维度: {state_dim}")
        print(f"  动作数量: {action_dim}")
        print(f"  核函数: {kernel}")
        print(f"  使用: {'sklearn' if use_sklearn else 'GPy'}")

    def add_sample(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        latency: float,
        cost: float
    ):
        """
        添加一个训练样本

        Args:
            state: 状态向量（79维）
            action: 动作ID（0-47）
            reward: 奖励
            latency: 延迟（ms）
            cost: 成本（美元）
        """
        # 构造输入特征：state + one-hot(action)
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0

        x = np.concatenate([state, action_onehot])

        self.X_train.append(x)
        self.y_reward_train.append(reward)
        self.y_latency_train.append(latency)
        self.y_cost_train.append(cost)

        # 添加新样本后需要重新拟合
        self.is_fitted = False

    def fit(self):
        """拟合高斯过程模型"""
        if len(self.X_train) < 5:
            print(f"[Warning] 训练样本太少({len(self.X_train)})，建议至少5个样本")
            return

        X = np.array(self.X_train)
        y_reward = np.array(self.y_reward_train).reshape(-1, 1)
        y_latency = np.array(self.y_latency_train).reshape(-1, 1)
        y_cost = np.array(self.y_cost_train).reshape(-1, 1)

        if self.use_sklearn:
            self._fit_sklearn(X, y_reward, y_latency, y_cost)
        else:
            self._fit_gpy(X, y_reward, y_latency, y_cost)

        self.is_fitted = True
        print(f"[SurrogateGP] 模型已拟合，样本数: {len(self.X_train)}")

    def _fit_sklearn(self, X, y_reward, y_latency, y_cost):
        """使用sklearn的GaussianProcessRegressor"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
        except ImportError:
            print("[Error] sklearn未安装，请运行: pip install scikit-learn")
            return

        # 选择核函数
        if self.kernel == 'rbf':
            kernel = C(1.0) * RBF(length_scale=1.0)
        elif self.kernel == 'matern':
            kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5)
        else:
            kernel = C(1.0) * RBF(length_scale=1.0)

        # 拟合三个GP模型
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.gp_reward = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level ** 2,
                n_restarts_optimizer=2,
                normalize_y=True,
            )
            self.gp_reward.fit(X, y_reward.ravel())

            self.gp_latency = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level ** 2,
                n_restarts_optimizer=2,
                normalize_y=True,
            )
            self.gp_latency.fit(X, y_latency.ravel())

            self.gp_cost = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.noise_level ** 2,
                n_restarts_optimizer=2,
                normalize_y=True,
            )
            self.gp_cost.fit(X, y_cost.ravel())

    def _fit_gpy(self, X, y_reward, y_latency, y_cost):
        """使用GPy"""
        try:
            import GPy
        except ImportError:
            print("[Error] GPy未安装，请运行: pip install GPy")
            return

        # 选择核函数
        input_dim = X.shape[1]
        if self.kernel == 'rbf':
            kernel = GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)
        elif self.kernel == 'matern':
            kernel = GPy.kern.Matern52(input_dim=input_dim, variance=1.0, lengthscale=1.0)
        else:
            kernel = GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)

        # 拟合三个GP模型
        self.gp_reward = GPy.models.GPRegression(X, y_reward, kernel)
        self.gp_reward.Gaussian_noise.variance = self.noise_level ** 2
        self.gp_reward.optimize(messages=False)

        self.gp_latency = GPy.models.GPRegression(X, y_latency, kernel.copy())
        self.gp_latency.Gaussian_noise.variance = self.noise_level ** 2
        self.gp_latency.optimize(messages=False)

        self.gp_cost = GPy.models.GPRegression(X, y_cost, kernel.copy())
        self.gp_cost.Gaussian_noise.variance = self.noise_level ** 2
        self.gp_cost.optimize(messages=False)

    def predict(
        self,
        state: np.ndarray,
        action: int,
        return_std: bool = True
    ) -> Tuple[float, float, float, Optional[Tuple]]:
        """
        预测给定(state, action)的性能指标

        Args:
            state: 状态向量
            action: 动作ID
            return_std: 是否返回标准差

        Returns:
            如果return_std=True:
                (reward_mean, latency_mean, cost_mean,
                 (reward_std, latency_std, cost_std))
            否则:
                (reward_mean, latency_mean, cost_mean, None)
        """
        if not self.is_fitted:
            # 未拟合则返回默认值
            if return_std:
                return (0.0, 1000.0, 0.001, (1.0, 500.0, 0.0005))
            else:
                return (0.0, 1000.0, 0.001, None)

        # 构造输入
        action_onehot = np.zeros(self.action_dim)
        action_onehot[action] = 1.0
        x = np.concatenate([state, action_onehot]).reshape(1, -1)

        # 预测
        if self.use_sklearn:
            reward_mean, reward_std = self.gp_reward.predict(x, return_std=True)
            latency_mean, latency_std = self.gp_latency.predict(x, return_std=True)
            cost_mean, cost_std = self.gp_cost.predict(x, return_std=True)

            reward_mean = reward_mean[0]
            latency_mean = latency_mean[0]
            cost_mean = cost_mean[0]
            reward_std = reward_std[0]
            latency_std = latency_std[0]
            cost_std = cost_std[0]
        else:
            # GPy
            reward_pred, reward_var = self.gp_reward.predict(x)
            latency_pred, latency_var = self.gp_latency.predict(x)
            cost_pred, cost_var = self.gp_cost.predict(x)

            reward_mean = reward_pred[0, 0]
            latency_mean = latency_pred[0, 0]
            cost_mean = cost_pred[0, 0]
            reward_std = np.sqrt(reward_var[0, 0])
            latency_std = np.sqrt(latency_var[0, 0])
            cost_std = np.sqrt(cost_var[0, 0])

        if return_std:
            return (reward_mean, latency_mean, cost_mean,
                    (reward_std, latency_std, cost_std))
        else:
            return (reward_mean, latency_mean, cost_mean, None)

    def predict_batch(
        self,
        state: np.ndarray,
        actions: List[int],
        return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple]]:
        """
        批量预测多个动作的性能指标

        Args:
            state: 状态向量
            actions: 动作ID列表
            return_std: 是否返回标准差

        Returns:
            (rewards, latencies, costs, stds)
            stds是可选的元组 (reward_stds, latency_stds, cost_stds)
        """
        rewards = []
        latencies = []
        costs = []
        reward_stds = []
        latency_stds = []
        cost_stds = []

        for action in actions:
            result = self.predict(state, action, return_std=return_std)
            rewards.append(result[0])
            latencies.append(result[1])
            costs.append(result[2])

            if return_std:
                reward_stds.append(result[3][0])
                latency_stds.append(result[3][1])
                cost_stds.append(result[3][2])

        rewards = np.array(rewards)
        latencies = np.array(latencies)
        costs = np.array(costs)

        if return_std:
            stds = (np.array(reward_stds), np.array(latency_stds), np.array(cost_stds))
            return rewards, latencies, costs, stds
        else:
            return rewards, latencies, costs, None

    def get_acquisition_scores(
        self,
        state: np.ndarray,
        actions: Optional[List[int]] = None,
        strategy: str = 'ucb',
        kappa: float = 2.0
    ) -> np.ndarray:
        """
        计算获取函数分数（用于选择探索的动作）

        Args:
            state: 当前状态
            actions: 候选动作列表（None表示所有动作）
            strategy: 获取策略 ['ucb', 'ei', 'pi']
            kappa: UCB的探索系数

        Returns:
            获取分数数组
        """
        if actions is None:
            actions = list(range(self.action_dim))

        rewards, latencies, costs, stds = self.predict_batch(
            state, actions, return_std=True
        )

        if strategy == 'ucb':
            # Upper Confidence Bound
            scores = rewards + kappa * stds[0]

        elif strategy == 'ei':
            # Expected Improvement（简化版，假设当前最优是训练集最大值）
            if len(self.y_reward_train) > 0:
                best_reward = max(self.y_reward_train)
                improvement = rewards - best_reward
                scores = improvement + stds[0]
            else:
                scores = rewards

        elif strategy == 'pi':
            # Probability of Improvement
            if len(self.y_reward_train) > 0:
                best_reward = max(self.y_reward_train)
                z = (rewards - best_reward) / (stds[0] + 1e-8)
                from scipy.stats import norm
                scores = norm.cdf(z)
            else:
                scores = rewards

        else:
            scores = rewards

        return scores

    def clear(self):
        """清空训练数据"""
        self.X_train = []
        self.y_reward_train = []
        self.y_latency_train = []
        self.y_cost_train = []
        self.is_fitted = False

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'n_samples': len(self.X_train),
            'is_fitted': self.is_fitted,
            'mean_reward': np.mean(self.y_reward_train) if len(self.y_reward_train) > 0 else 0.0,
            'mean_latency': np.mean(self.y_latency_train) if len(self.y_latency_train) > 0 else 0.0,
            'mean_cost': np.mean(self.y_cost_train) if len(self.y_cost_train) > 0 else 0.0,
        }
