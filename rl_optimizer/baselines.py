"""Baseline methods used for simulator comparisons."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from .environment import ServerlessFunctionEnv


class BayesianOptimizationPolicy:
    """Bayesian optimization baseline over the discrete action space."""

    def __init__(self, env: ServerlessFunctionEnv):
        self.env = env
        self.action_space = env.action_space_wrapper
        self.observations = []
        self.best_action = None
        self.best_reward = float("-inf")

    def reset(self) -> None:
        """Reset optimizer state between independent evaluation episodes."""
        self.observations = []
        self.best_action = None
        self.best_reward = float("-inf")

    def optimize(self, n_iterations: int = 50) -> int:
        """
        Run Bayesian optimization.

        Args:
            n_iterations: Number of optimization iterations.

        Returns:
            Best action id.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
        except ImportError:
            print("[Warning] sklearn is unavailable; falling back to random search.")
            return self._random_search(n_iterations)

        print(
            f"\n[BayesOpt] Starting Bayesian optimization " f"for {n_iterations} iterations...",
            flush=True,
        )

        # Initial random exploration.
        for _ in range(min(10, n_iterations)):
            action = np.random.randint(0, self.action_space.n_actions)
            reward = self._evaluate_action(action)
            self.observations.append((action, reward))

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_action = action

        # Main Bayesian optimization loop.
        for i in range(10, n_iterations):
            # Fit the Gaussian-process surrogate.
            X = np.array([obs[0] for obs in self.observations]).reshape(-1, 1)
            y = np.array([obs[1] for obs in self.observations])

            gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
            )
            gp.fit(X, y)

            # Use an upper-confidence-bound acquisition rule.
            best_ucb = float("-inf")
            next_action = None

            for action in range(self.action_space.n_actions):
                if action in [obs[0] for obs in self.observations]:
                    continue

                X_test = np.array([[action]])
                mu, sigma = gp.predict(X_test, return_std=True)

                kappa = 2.0
                ucb = mu[0] + kappa * sigma[0]

                if ucb > best_ucb:
                    best_ucb = ucb
                    next_action = action

            # Stop if every action has already been evaluated.
            if next_action is None:
                break

            # Evaluate the new action.
            reward = self._evaluate_action(next_action)
            self.observations.append((next_action, reward))

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_action = next_action

            if (i + 1) % 10 == 0:
                print(
                    f"  Iteration {i + 1}/{n_iterations}, "
                    f"current best reward: {self.best_reward:.4f}",
                    flush=True,
                )

        print("\n[BayesOpt] Optimization completed!", flush=True)
        print(f"  Best action: {self.best_action}", flush=True)
        print(f"  Best reward: {self.best_reward:.4f}", flush=True)

        return self.best_action

    def _evaluate_action(self, action: int) -> float:
        """Score one action with a single environment step."""
        obs, _ = self.env.reset()
        obs, reward, _, _, _ = self.env.step(action)
        return reward

    def _random_search(self, n_iterations: int) -> int:
        """Random-search fallback."""
        print(f"\n[RandomSearch] Sampling {n_iterations} random actions...", flush=True)

        for i in range(n_iterations):
            action = np.random.randint(0, self.action_space.n_actions)
            reward = self._evaluate_action(action)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_action = action

        return self.best_action

    def evaluate(self, n_episodes: int = 10) -> Dict:
        """Evaluate the best BayesOpt action."""
        if self.best_action is None:
            print("[Error] Run optimize() before evaluate().", flush=True)
            return {}

        all_rewards = []
        all_latencies = []
        all_costs = []
        all_success_rates = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            truncated = False

            episode_rewards = []
            episode_latencies = []
            episode_costs = []
            episode_successes = []

            while not (done or truncated):
                obs, reward, done, truncated, info = self.env.step(self.best_action)

                episode_rewards.append(float(info.get("raw_reward", reward)))
                episode_latencies.append(info["metrics"]["latency"])
                episode_costs.append(info["metrics"]["cost"])
                episode_successes.append(float(info["metrics"]["success"]))

            all_rewards.append(np.mean(episode_rewards))
            all_latencies.append(np.mean(episode_latencies))
            all_costs.append(np.mean(episode_costs))
            all_success_rates.append(np.mean(episode_successes))

        return {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_latency": np.mean(all_latencies),
            "std_latency": np.std(all_latencies),
            "mean_cost": np.mean(all_costs),
            "std_cost": np.std(all_costs),
            "mean_success_rate": np.mean(all_success_rates),
        }


class DefaultBaseline:
    """Default-configuration baseline for dynamic environments."""

    def __init__(self, env, memory: int = 1024):
        """
        Args:
            env: Dynamic environment.
            memory: Default memory configuration.
        """
        self.env = env
        self.action_space = env.action_space_wrapper
        self.default_action = self.action_space.get_action_id(memory, "x64", 120)

    def evaluate(
        self,
        n_episodes: int = 10,
        progress_label: str | None = None,
        seed: int | None = None,
    ) -> Dict:
        """Evaluate the default configuration."""
        all_rewards = []
        all_latencies = []
        all_costs = []
        all_success_rates = []
        progress_label = progress_label or "Default"
        start_time = time.time()

        for episode in range(n_episodes):
            episode_seed = None if seed is None else seed + episode
            obs, _ = self.env.reset(seed=episode_seed)
            done = False
            truncated = False

            episode_rewards = []
            episode_latencies = []
            episode_costs = []
            episode_successes = []

            while not (done or truncated):
                obs, reward, done, truncated, info = self.env.step(self.default_action)

                episode_rewards.append(float(info.get("raw_reward", reward)))
                episode_latencies.append(info["metrics"]["latency"])
                episode_costs.append(info["metrics"]["cost"])
                episode_successes.append(float(info["metrics"]["success"]))

            all_rewards.append(np.mean(episode_rewards))
            all_latencies.append(np.mean(episode_latencies))
            all_costs.append(np.mean(episode_costs))
            all_success_rates.append(np.mean(episode_successes))
            print(
                f"[Default Eval] {progress_label}: "
                f"{episode + 1}/{n_episodes}, "
                f"raw_reward={all_rewards[-1]:.4f}, "
                f"latency={all_latencies[-1]:.2f} ms, "
                f"success={all_success_rates[-1]:.3f}, "
                f"elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_latency": float(np.mean(all_latencies)),
            "std_latency": float(np.std(all_latencies)),
            "mean_cost": float(np.mean(all_costs)),
            "std_cost": float(np.std(all_costs)),
            "mean_success_rate": float(np.mean(all_success_rates)),
        }


class RandomBaseline:
    """Random baseline for dynamic environments."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space_wrapper

    def evaluate(
        self,
        n_episodes: int = 10,
        progress_label: str | None = None,
        seed: int | None = None,
    ) -> Dict:
        """Evaluate randomly sampled actions."""
        all_rewards = []
        all_latencies = []
        all_costs = []
        all_success_rates = []
        progress_label = progress_label or "Random"
        start_time = time.time()

        for episode in range(n_episodes):
            episode_seed = None if seed is None else seed + episode
            if episode_seed is not None:
                np.random.seed(episode_seed)
            obs, _ = self.env.reset(seed=episode_seed)
            done = False
            truncated = False

            episode_rewards = []
            episode_latencies = []
            episode_costs = []
            episode_successes = []

            while not (done or truncated):
                action = self.action_space.sample()
                obs, reward, done, truncated, info = self.env.step(action)

                episode_rewards.append(float(info.get("raw_reward", reward)))
                episode_latencies.append(info["metrics"]["latency"])
                episode_costs.append(info["metrics"]["cost"])
                episode_successes.append(float(info["metrics"]["success"]))

            all_rewards.append(np.mean(episode_rewards))
            all_latencies.append(np.mean(episode_latencies))
            all_costs.append(np.mean(episode_costs))
            all_success_rates.append(np.mean(episode_successes))
            print(
                f"[Random Eval] {progress_label}: "
                f"{episode + 1}/{n_episodes}, "
                f"raw_reward={all_rewards[-1]:.4f}, "
                f"latency={all_latencies[-1]:.2f} ms, "
                f"success={all_success_rates[-1]:.3f}, "
                f"elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_latency": float(np.mean(all_latencies)),
            "std_latency": float(np.std(all_latencies)),
            "mean_cost": float(np.mean(all_costs)),
            "std_cost": float(np.std(all_costs)),
            "mean_success_rate": float(np.mean(all_success_rates)),
        }


class GreedyBaseline:
    """Greedy baseline for dynamic environments."""

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space_wrapper

    def evaluate(
        self,
        n_episodes: int = 10,
        progress_label: str | None = None,
        seed: int | None = None,
    ) -> Dict:
        """Evaluate a simple category-based greedy policy."""
        all_rewards = []
        all_latencies = []
        all_costs = []
        all_success_rates = []
        progress_label = progress_label or "Greedy"
        start_time = time.time()

        for episode in range(n_episodes):
            episode_seed = None if seed is None else seed + episode
            obs, _ = self.env.reset(seed=episode_seed)
            done = False
            truncated = False

            episode_rewards = []
            episode_latencies = []
            episode_costs = []
            episode_successes = []

            while not (done or truncated):
                # Greedy rule: pick a configuration from the function category.
                state = self.env.base_env.state_space.extract_state()
                category = np.argmax(state[42:47])

                if category == 0:  # Webapps
                    action = self.action_space.get_action_id(512, "x64", 120)
                elif category == 1:  # Multimedia
                    action = self.action_space.get_action_id(2048, "x64", 300)
                elif category == 2:  # Utilities
                    action = self.action_space.get_action_id(1024, "x64", 120)
                elif category == 3:  # Inference
                    action = self.action_space.get_action_id(3008, "x64", 900)
                else:  # Scientific
                    action = self.action_space.get_action_id(1024, "arm64", 300)

                obs, reward, done, truncated, info = self.env.step(action)

                episode_rewards.append(float(info.get("raw_reward", reward)))
                episode_latencies.append(info["metrics"]["latency"])
                episode_costs.append(info["metrics"]["cost"])
                episode_successes.append(float(info["metrics"]["success"]))

            all_rewards.append(np.mean(episode_rewards))
            all_latencies.append(np.mean(episode_latencies))
            all_costs.append(np.mean(episode_costs))
            all_success_rates.append(np.mean(episode_successes))
            print(
                f"[Greedy Eval] {progress_label}: "
                f"{episode + 1}/{n_episodes}, "
                f"raw_reward={all_rewards[-1]:.4f}, "
                f"latency={all_latencies[-1]:.2f} ms, "
                f"success={all_success_rates[-1]:.3f}, "
                f"elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

        return {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "mean_latency": float(np.mean(all_latencies)),
            "std_latency": float(np.std(all_latencies)),
            "mean_cost": float(np.mean(all_costs)),
            "std_cost": float(np.std(all_costs)),
            "mean_success_rate": float(np.mean(all_success_rates)),
        }
