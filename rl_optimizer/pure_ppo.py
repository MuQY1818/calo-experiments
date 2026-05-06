"""Pure PPO implementation for dynamic-load simulator comparisons."""

import copy
from collections import Counter
import time
from typing import Any, Callable, Dict

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from torch import nn

from .state_space import StateSpace


class CALOFeaturesExtractor(BaseFeaturesExtractor):
    """Group-aware feature extractor for code/load/context state vectors."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        code_hidden_dim: int = 32,
        load_hidden_dim: int = 16,
        context_hidden_dim: int = 32,
        code_dim: int = StateSpace.DEFAULT_CODE_FEATURE_DIM,
        load_dim: int = StateSpace.LOAD_FEATURE_DIM,
        category_dim: int = StateSpace.CATEGORY_DIM,
        history_dim: int = StateSpace.HISTORY_DIM,
        context_dim: int = StateSpace.CONTEXT_DIM,
    ):
        """Build grouped MLP branches over the CALO state."""
        super().__init__(observation_space, features_dim)

        context_input_dim = category_dim + history_dim + context_dim
        total_dim = code_dim + load_dim + context_input_dim
        observed_dim = int(np.prod(observation_space.shape))
        if observed_dim != total_dim:
            raise ValueError(
                f"Observation dimension {observed_dim} does not match "
                f"grouped layout {total_dim}"
            )

        self.code_slice = slice(0, code_dim)
        self.load_slice = slice(code_dim, code_dim + load_dim)
        self.context_slice = slice(code_dim + load_dim, total_dim)

        self.code_branch = nn.Sequential(
            nn.Linear(code_dim, 64),
            nn.ReLU(),
            nn.Linear(64, code_hidden_dim),
            nn.ReLU(),
        )
        self.load_branch = nn.Sequential(
            nn.Linear(load_dim, 32),
            nn.ReLU(),
            nn.Linear(32, load_hidden_dim),
            nn.ReLU(),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(context_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, context_hidden_dim),
            nn.ReLU(),
        )
        fusion_input_dim = code_hidden_dim + load_hidden_dim + context_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Encode grouped state features and fuse them for PPO."""
        code_features = observations[:, self.code_slice]
        load_features = observations[:, self.load_slice]
        context_features = observations[:, self.context_slice]

        code_embedding = self.code_branch(code_features)
        load_embedding = self.load_branch(load_features)
        context_embedding = self.context_branch(context_features)
        fused = th.cat(
            [code_embedding, load_embedding, context_embedding],
            dim=1,
        )
        return self.fusion(fused)


class FlatFusionFeaturesExtractor(BaseFeaturesExtractor):
    """Flat MLP extractor over the full concatenated CALO state."""

    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        hidden_dims: tuple[int, int] = (128, 64),
    ):
        """Build a flat fusion MLP without grouped state branches."""
        super().__init__(observation_space, features_dim)

        observed_dim = int(np.prod(observation_space.shape))
        first_hidden_dim, second_hidden_dim = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(observed_dim, first_hidden_dim),
            nn.ReLU(),
            nn.Linear(first_hidden_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Linear(second_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Encode the full observation vector with one shared MLP."""
        return self.network(observations)


def resolve_features_extractor_class(
    extractor_name: str | None,
) -> type[BaseFeaturesExtractor]:
    """Resolve one configured extractor name to the implementation class."""
    normalized_name = str(extractor_name or 'grouped').strip().lower()
    if normalized_name in {'grouped', 'calo', 'grouped_fusion'}:
        return CALOFeaturesExtractor
    if normalized_name in {'flat', 'flat_fusion', 'mlp'}:
        return FlatFusionFeaturesExtractor
    raise ValueError(f"Unsupported feature extractor: {extractor_name}")


class TrainingMonitorCallback(BaseCallback):
    """Track PPO training metrics and print periodic progress updates."""

    def __init__(
        self,
        eval_freq: int = 1000,
        verbose: int = 0,
        report_freq_steps: int = 1000,
        progress_label: str = 'PPO',
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.report_freq_steps = max(1, report_freq_steps)
        self.progress_label = progress_label

        # Training statistics
        self.timesteps = []
        self.episode_rewards = []
        self.episode_lengths = []

        # PPO loss statistics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

        # Temporary variables
        self.episode_reward = 0
        self.episode_length = 0
        self.training_start_time = 0.0
        self.total_timesteps = 0
        self.next_report_timestep = self.report_freq_steps
        self.last_reported_timestep = 0

    def reset_progress(
        self,
        total_timesteps: int | None = None,
        progress_label: str | None = None,
    ) -> None:
        """Reset callback state before a new training run."""
        self.timesteps = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.episode_reward = 0
        self.episode_length = 0
        self.training_start_time = 0.0
        self.last_reported_timestep = 0
        if total_timesteps is not None:
            self.total_timesteps = int(total_timesteps)
        if progress_label is not None:
            self.progress_label = progress_label
        self.next_report_timestep = self.report_freq_steps

    def _report_progress(self, force: bool = False) -> None:
        """Print a flush-safe progress line during training."""
        if not force and self.num_timesteps < self.next_report_timestep:
            return
        if self.num_timesteps == self.last_reported_timestep:
            return

        total_timesteps = max(1, self.total_timesteps)
        progress_pct = min(100.0, self.num_timesteps * 100.0 / total_timesteps)
        elapsed_sec = max(0.0, time.time() - self.training_start_time)
        steps_per_sec = self.num_timesteps / elapsed_sec if elapsed_sec > 0 else 0.0
        recent_rewards = self.episode_rewards[-5:]
        recent_mean_reward = (
            float(np.mean(recent_rewards)) if recent_rewards else 0.0
        )

        print(
            f"[PPO Train] {self.progress_label}: "
            f"{self.num_timesteps}/{total_timesteps} "
            f"({progress_pct:5.1f}%), "
            f"episodes={len(self.episode_rewards)}, "
            f"recent_reward={recent_mean_reward:.4f}, "
            f"elapsed={elapsed_sec:.1f}s, "
            f"speed={steps_per_sec:.1f} steps/s",
            flush=True,
        )

        self.last_reported_timestep = self.num_timesteps
        while self.next_report_timestep <= self.num_timesteps:
            self.next_report_timestep += self.report_freq_steps

    def _on_step(self) -> bool:
        """Called on every environment step."""
        # Accumulate episode reward.
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1

        # Record per-episode statistics when one episode ends.
        if self.locals['dones'][0]:
            self.timesteps.append(self.num_timesteps)
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)

            self.episode_reward = 0
            self.episode_length = 0

        self._report_progress()

        return True

    def _on_rollout_end(self) -> None:
        """Hook for rollout-end metrics."""
        pass

    def _on_training_start(self) -> None:
        """Prepare the callback before training starts."""
        self.training_start_time = time.time()
        self.total_timesteps = int(
            self.locals.get(
                'total_timesteps',
                getattr(self.model, '_total_timesteps', self.total_timesteps),
            )
        )
        self.next_report_timestep = min(
            max(1, self.report_freq_steps),
            max(1, self.total_timesteps),
        )
        self.last_reported_timestep = 0
        print(
            f"[PPO Train] {self.progress_label}: "
            f"start total_timesteps={self.total_timesteps}",
            flush=True,
        )

    def _on_training_end(self) -> None:
        """Finalize progress reporting at the end of training."""
        self._report_progress(force=True)

class ValidationCheckpointCallback(BaseCallback):
    """Keep the best deterministic PPO checkpoint on a validation env."""

    def __init__(
        self,
        owner: "PurePPO",
        eval_env,
        eval_freq_steps: int = 2048,
        n_episodes: int = 3,
        metric: str = 'mean_reward',
        min_timestep: int = 0,
        progress_label: str = 'PPO',
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.owner = owner
        self.eval_env = eval_env
        self.eval_freq_steps = max(1, int(eval_freq_steps))
        self.n_episodes = max(1, int(n_episodes))
        self.metric = str(metric)
        self.min_timestep = max(0, int(min_timestep))
        self.progress_label = progress_label
        self.next_eval_timestep = self.eval_freq_steps
        self.best_metric = float('-inf')
        self.best_timestep = 0
        self.best_policy_state: Dict[str, th.Tensor] | None = None
        self.evaluation_history: list[Dict[str, float]] = []

    def _clone_policy_state(self) -> Dict[str, th.Tensor]:
        """Clone the current policy parameters to CPU memory."""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.policy.state_dict().items()
        }

    def _run_validation(self, force: bool = False) -> None:
        """Run one deterministic validation pass if due."""
        if not force and self.num_timesteps < self.next_eval_timestep:
            return

        metrics, _ = self.owner._run_policy_episodes(
            n_episodes=self.n_episodes,
            eval_env=self.eval_env,
            progress_label=f"{self.progress_label} [validation]",
            observation_transform=None,
            collect_observations=False,
            emit_progress=False,
            log_prefix='PPO Val',
        )
        metric_value = float(metrics[self.metric])
        eligible = bool(self.num_timesteps >= self.min_timestep)
        record = {
            'timestep': int(self.num_timesteps),
            'mean_reward': float(metrics['mean_reward']),
            'mean_latency': float(metrics['mean_latency']),
            'mean_cost': float(metrics['mean_cost']),
            'mean_success_rate': float(metrics['mean_success_rate']),
            'selected_metric': metric_value,
            'eligible': eligible,
        }
        self.evaluation_history.append(record)
        print(
            f"[PPO Val] {self.progress_label}: "
            f"t={self.num_timesteps}, "
            f"reward={metrics['mean_reward']:.4f}, "
            f"latency={metrics['mean_latency']:.2f} ms, "
            f"cost={metrics['mean_cost']:.6f}, "
            f"eligible={eligible}",
            flush=True,
        )

        if eligible and metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_timestep = int(self.num_timesteps)
            self.best_policy_state = self._clone_policy_state()
            print(
                f"[PPO Val] {self.progress_label}: "
                f"new_best t={self.best_timestep}, "
                f"{self.metric}={self.best_metric:.4f}",
                flush=True,
            )

        while self.next_eval_timestep <= self.num_timesteps:
            self.next_eval_timestep += self.eval_freq_steps

    def _on_training_start(self) -> None:
        """Reset validation schedule at training start."""
        self.next_eval_timestep = self.eval_freq_steps
        self.best_metric = float('-inf')
        self.best_timestep = 0
        self.best_policy_state = None
        self.evaluation_history = []

    def _on_step(self) -> bool:
        """Trigger validation when enough timesteps have passed."""
        self._run_validation(force=False)
        return True

    def _on_training_end(self) -> None:
        """Ensure the final checkpoint is evaluated."""
        self._run_validation(force=True)

    def restore_best_policy(self) -> None:
        """Restore the best validation checkpoint if one was recorded."""
        if self.best_policy_state is None:
            return
        self.model.policy.load_state_dict(self.best_policy_state, strict=True)

    def build_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of validation selection."""
        return {
            'enabled': True,
            'metric': self.metric,
            'eval_freq_steps': int(self.eval_freq_steps),
            'n_episodes': int(self.n_episodes),
            'min_timestep': int(self.min_timestep),
            'best_metric': float(self.best_metric),
            'best_timestep': int(self.best_timestep),
            'evaluations': self.evaluation_history,
        }


class PurePPO:
    """Pure PPO baseline without a surrogate model."""

    @staticmethod
    def _resolve_state_space(env):
        """Best-effort lookup of the state space across wrapped env variants."""
        direct_state_space = getattr(env, 'state_space', None)
        if direct_state_space is not None:
            return direct_state_space

        unwrapped = getattr(env, 'unwrapped', None)
        if unwrapped is not None:
            direct_state_space = getattr(unwrapped, 'state_space', None)
            if direct_state_space is not None:
                return direct_state_space

            base_env = getattr(unwrapped, 'base_env', None)
            if base_env is not None:
                direct_state_space = getattr(base_env, 'state_space', None)
                if direct_state_space is not None:
                    return direct_state_space

        base_env = getattr(env, 'base_env', None)
        if base_env is not None:
            return getattr(base_env, 'state_space', None)

        return None

    def __init__(
        self,
        env,
        total_timesteps: int = 50000,
        ppo_kwargs: Dict = None,
        seed: int | None = None,
        imitation_warmstart: Dict[str, Any] | None = None,
        validation_env=None,
        checkpoint_selection: Dict[str, Any] | None = None,
    ):
        """
        Initialize PPO.

        Args:
            env: Training environment.
            total_timesteps: Number of training timesteps.
            ppo_kwargs: PPO hyperparameters.
        """
        self.env = env
        self.total_timesteps = total_timesteps
        self.seed = seed
        self.imitation_warmstart = dict(imitation_warmstart or {})
        self.imitation_warmstart_summary: Dict[str, Any] | None = None
        self.validation_env = validation_env
        self.checkpoint_selection = dict(checkpoint_selection or {})
        self.checkpoint_selection_summary: Dict[str, Any] | None = None
        self.state_space = self._resolve_state_space(env)

        # Default PPO hyperparameters.
        if ppo_kwargs is None:
            ppo_kwargs = {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'verbose': 1,
            }
        else:
            ppo_kwargs = dict(ppo_kwargs)
        self.max_grad_norm = float(ppo_kwargs.get('max_grad_norm', 0.5))
        feature_extractor_name = ppo_kwargs.pop(
            'feature_extractor_name',
            'grouped',
        )

        code_feature_dim = getattr(
            self.state_space,
            'code_feature_dim',
            StateSpace.DEFAULT_CODE_FEATURE_DIM,
        )

        policy_kwargs = dict(ppo_kwargs.pop('policy_kwargs', {}))
        features_extractor_class = policy_kwargs.get('features_extractor_class')
        if features_extractor_class is None:
            features_extractor_class = resolve_features_extractor_class(
                feature_extractor_name
            )
            policy_kwargs['features_extractor_class'] = features_extractor_class

        features_extractor_kwargs = dict(
            policy_kwargs.get('features_extractor_kwargs', {})
        )
        features_extractor_kwargs.setdefault('features_dim', 128)
        if features_extractor_class is CALOFeaturesExtractor:
            features_extractor_kwargs.setdefault('code_dim', code_feature_dim)
            features_extractor_kwargs.setdefault(
                'load_dim',
                StateSpace.LOAD_FEATURE_DIM,
            )
            features_extractor_kwargs.setdefault(
                'category_dim',
                StateSpace.CATEGORY_DIM,
            )
            features_extractor_kwargs.setdefault(
                'history_dim',
                StateSpace.HISTORY_DIM,
            )
            features_extractor_kwargs.setdefault(
                'context_dim',
                StateSpace.CONTEXT_DIM,
            )
        policy_kwargs['features_extractor_kwargs'] = features_extractor_kwargs
        policy_kwargs.setdefault(
            'net_arch',
            {'pi': [64, 64], 'vf': [64, 64]},
        )
        ppo_kwargs['policy_kwargs'] = policy_kwargs

        if self.seed is not None:
            set_random_seed(self.seed)

        # Create the PPO model.
        self.model = PPO('MlpPolicy', env, device='cpu', seed=self.seed, **ppo_kwargs)

        # Training monitor callback.
        self.monitor_callback = TrainingMonitorCallback(
            report_freq_steps=max(1, self.total_timesteps // 10),
        )

        # Training statistics.
        self.training_stats = {
            'timesteps': [],
            'mean_reward': [],
            'mean_latency': [],
            'mean_cost': [],
            'episode_lengths': [],
        }

    def _get_category_slice(self) -> slice:
        """Return the category slice for the current state dimensionality."""
        code_feature_dim = getattr(
            self.state_space,
            'code_feature_dim',
            StateSpace.DEFAULT_CODE_FEATURE_DIM,
        )
        return StateSpace.get_group_slices(
            code_feature_dim=code_feature_dim,
        )['category']

    def _select_greedy_teacher_action(self, observation: np.ndarray) -> int:
        """Map one observation to the greedy baseline action."""
        action_space = getattr(self.env, 'action_space_wrapper', None)
        if action_space is None:
            raise ValueError(
                "imitation_warmstart requires env.action_space_wrapper"
            )

        state = np.asarray(observation, dtype=np.float32)
        category_slice = self._get_category_slice()
        category_features = state[category_slice]
        category = int(np.argmax(category_features)) if category_features.size else 0

        if category == 0:  # Webapps
            return int(action_space.get_action_id(512, 'x64', 120))
        if category == 1:  # Multimedia
            return int(action_space.get_action_id(2048, 'x64', 300))
        if category == 2:  # Utilities
            return int(action_space.get_action_id(1024, 'x64', 120))
        if category == 3:  # Inference
            return int(action_space.get_action_id(3008, 'x64', 900))
        return int(action_space.get_action_id(1024, 'arm64', 300))

    def _resolve_oracle_shortlist_actions(self) -> list[int]:
        """Resolve the candidate pool used by the oracle warm-start teacher."""
        action_space = getattr(self.env, 'action_space_wrapper', None)
        if action_space is None:
            raise ValueError(
                "oracle_shortlist warm-start requires env.action_space_wrapper"
            )

        resolved_actions: list[int] = []
        candidate_actions = self.imitation_warmstart.get('candidate_actions')
        if candidate_actions is not None:
            for action in candidate_actions:
                action_id = int(action)
                action_space.get_configuration(action_id)
                resolved_actions.append(action_id)

        candidate_configs = self.imitation_warmstart.get('candidate_configs')
        if candidate_configs is not None:
            for config in candidate_configs:
                if not isinstance(config, dict):
                    raise ValueError(
                        "imitation_warmstart.candidate_configs must contain dicts"
                    )
                resolved_actions.append(
                    int(
                        action_space.get_action_id(
                            memory_mb=int(config['memory_mb']),
                            architecture=str(config['architecture']),
                            timeout_sec=int(config['timeout_sec']),
                        )
                    )
                )

        if not resolved_actions:
            resolved_actions = list(range(int(action_space.n_actions)))

        deduplicated_actions = list(dict.fromkeys(resolved_actions))
        return deduplicated_actions

    def _rank_fixed_policy_candidates(
        self,
        candidate_actions: list[int],
        n_episodes: int,
        progress_label: str,
    ) -> list[Dict[str, Any]]:
        """Rank candidate actions by fixed-policy rollout reward."""
        if not candidate_actions:
            raise ValueError("bootstrap shortlist requires candidate actions")
        if n_episodes <= 0:
            raise ValueError(
                "bootstrap shortlist requires positive n_episodes"
            )

        env = self.env
        action_space = getattr(env, 'action_space_wrapper', None)
        if action_space is None:
            raise ValueError(
                "bootstrap shortlist requires env.action_space_wrapper"
            )

        load_pattern = getattr(env, 'load_pattern', None)
        rankings = []
        print(
            f"[PPO WarmStart] {progress_label}: "
            f"bootstrap shortlist from {len(candidate_actions)} actions, "
            f"eval_episodes={n_episodes}",
            flush=True,
        )

        for action_id in candidate_actions:
            episode_rewards = []
            episode_latencies = []
            episode_costs = []
            for episode in range(n_episodes):
                episode_seed = (
                    None
                    if self.seed is None
                    else self.seed + 20_000 + int(action_id) * 100 + episode
                )
                if load_pattern:
                    obs, _ = env.reset(
                        seed=episode_seed,
                        options={'load_pattern': load_pattern},
                    )
                else:
                    obs, _ = env.reset(seed=episode_seed)

                done = False
                truncated = False
                step_rewards = []
                step_latencies = []
                step_costs = []
                while not (done or truncated):
                    obs, reward, done, truncated, info = env.step(int(action_id))
                    step_rewards.append(float(info.get('raw_reward', reward)))
                    step_latencies.append(float(info['metrics']['latency']))
                    step_costs.append(float(info['metrics']['cost']))

                episode_rewards.append(
                    float(np.mean(step_rewards)) if step_rewards else 0.0
                )
                episode_latencies.append(
                    float(np.mean(step_latencies)) if step_latencies else 0.0
                )
                episode_costs.append(
                    float(np.mean(step_costs)) if step_costs else 0.0
                )

            configuration = action_space.get_configuration(int(action_id))
            rankings.append({
                'action_id': int(action_id),
                'configuration': configuration.to_dict(),
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'mean_latency': float(np.mean(episode_latencies)),
                'mean_cost': float(np.mean(episode_costs)),
            })

        rankings.sort(
            key=lambda item: (
                -item['mean_reward'],
                item['mean_latency'],
                item['mean_cost'],
            )
        )
        return rankings

    def _clone_dynamic_env_runtime(self, env):
        """Clone the current dynamic environment state without copying CodeBERT."""
        from .dynamic_environment import DynamicLoadEnvironment
        from .environment import ServerlessFunctionEnv

        source_env = env
        if hasattr(env, 'current_env') and getattr(env, 'current_env', None) is not None:
            source_env = env.current_env

        if not hasattr(source_env, 'base_env'):
            return copy.deepcopy(source_env)

        source_base_env = source_env.base_env
        action_space_config = {
            'memory_options': list(source_base_env.action_space_wrapper.MEMORY_OPTIONS),
            'architecture_options': list(
                source_base_env.action_space_wrapper.ARCHITECTURE_OPTIONS
            ),
            'timeout_options': list(
                source_base_env.action_space_wrapper.TIMEOUT_OPTIONS
            ),
        }
        if hasattr(source_base_env.state_space, 'fork_for_env'):
            forked_state_space = source_base_env.state_space.fork_for_env()
            forked_state_space.load_monitor = copy.deepcopy(
                source_base_env.state_space.load_monitor
            )
            forked_state_space.performance_history = copy.deepcopy(
                source_base_env.state_space.performance_history
            )
            forked_state_space.config_context = copy.deepcopy(
                source_base_env.state_space.config_context
            )
            forked_state_space.simulation_time_sec = float(
                getattr(source_base_env.state_space, 'simulation_time_sec', 0.0)
            )
        else:
            forked_state_space = copy.deepcopy(source_base_env.state_space)

        shadow_base_env = ServerlessFunctionEnv(
            benchmark=source_base_env.benchmark,
            deployment=source_base_env.deployment,
            config_path=source_base_env.config_path,
            sebs_root=str(source_base_env.sebs_root),
            max_steps=source_base_env.max_steps,
            reward_weights=dict(source_base_env.reward_weights),
            reward_penalties=dict(source_base_env.reward_penalties),
            enable_real_execution=source_base_env.enable_real_execution,
            normalize_reward=source_base_env.normalize_reward,
            success_bonus=source_base_env.success_bonus,
            cost_normalization=source_base_env.cost_normalization,
            calibration_dir=getattr(source_base_env.service_model, 'calibration_dir', None),
            calibration_dirs=getattr(
                source_base_env.service_model,
                'calibration_dirs',
                None,
            ),
            step_duration_sec=source_base_env.step_duration_sec,
            max_containers=source_base_env.container_pool.max_containers,
            container_ttl_sec=source_base_env.container_pool.ttl_sec,
            state_space=forked_state_space,
            action_space_config=action_space_config,
            log_init=False,
        )
        shadow_env = DynamicLoadEnvironment(
            shadow_base_env,
            episode_length=source_env.episode_length,
            load_change_freq=source_env.load_change_freq,
            switch_penalty=source_env.switch_penalty,
            adaptation_penalty=source_env.adaptation_penalty,
            workload_source=source_env.default_workload_source,
            azure_profile_path=str(source_env.azure_profile_path),
            azure_summary_path=(
                str(source_env.azure_summary_path)
                if source_env.azure_summary_path is not None
                else None
            ),
            azure_top_k=source_env.azure_top_k,
            azure_arrival_scale=source_env.azure_arrival_scale,
            azure_max_arrivals_per_step=source_env.azure_max_arrivals_per_step,
            azure_profile_selection=source_env.azure_profile_selection,
            azure_selection_pool_size=source_env.azure_selection_pool_size,
            azure_target_concurrency=source_env.azure_target_concurrency,
        )
        shadow_env.simulation_clock.set(source_env.simulation_clock.now())
        shadow_env.container_pool._containers = copy.deepcopy(
            source_env.container_pool._containers
        )
        shadow_env.base_env.attach_simulation_backend(
            simulation_clock=shadow_env.simulation_clock,
            service_model=shadow_env.service_model,
            container_pool=shadow_env.container_pool,
        )
        shadow_env.base_env.current_step = int(source_base_env.current_step)
        shadow_env.base_env.reward_mean = float(source_base_env.reward_mean)
        shadow_env.base_env.reward_m2 = float(source_base_env.reward_m2)
        shadow_env.base_env.reward_count = int(source_base_env.reward_count)
        shadow_env.base_env.episode_rewards = list(source_base_env.episode_rewards)
        shadow_env.base_env.episode_metrics = copy.deepcopy(
            source_base_env.episode_metrics
        )
        shadow_env.base_env.current_workload = copy.deepcopy(
            source_base_env.current_workload
        )
        shadow_env.base_env.state_space.set_simulation_time(
            getattr(source_base_env.state_space, 'simulation_time_sec', 0.0)
        )
        shadow_env.current_step = int(source_env.current_step)
        shadow_env.last_action = source_env.last_action
        shadow_env.load_pattern = str(source_env.load_pattern)
        shadow_env.load_recently_changed = bool(source_env.load_recently_changed)
        shadow_env.workload_source = str(source_env.workload_source)
        shadow_env._current_workload = copy.deepcopy(source_env._current_workload)
        shadow_env.base_env.set_workload_step(copy.deepcopy(source_env.base_env.current_workload))
        shadow_env._workload_driver = copy.deepcopy(source_env._workload_driver)
        shadow_env.rng = copy.deepcopy(source_env.rng)
        return shadow_env

    def _select_oracle_shortlist_action(
        self,
        env,
        candidate_actions: list[int],
    ) -> tuple[int, Dict[str, float]]:
        """Select the best one-step action from a shortlist by simulator lookahead."""
        if not candidate_actions:
            raise ValueError("oracle_shortlist requires at least one candidate action")

        best_action = int(candidate_actions[0])
        best_reward = float('-inf')
        best_latency = float('inf')
        best_cost = float('inf')

        for action in candidate_actions:
            probe_env = self._clone_dynamic_env_runtime(env)
            _, probe_reward, _, _, probe_info = probe_env.step(int(action))
            raw_reward = float(probe_info.get('raw_reward', probe_reward))
            latency = float(probe_info['metrics'].get('latency', np.inf))
            cost = float(probe_info['metrics'].get('cost', np.inf))
            if (
                raw_reward > best_reward
                or (
                    np.isclose(raw_reward, best_reward)
                    and (
                        latency < best_latency
                        or (
                            np.isclose(latency, best_latency)
                            and cost < best_cost
                        )
                    )
                )
            ):
                best_action = int(action)
                best_reward = raw_reward
                best_latency = latency
                best_cost = cost

        return best_action, {
            'best_immediate_reward': float(best_reward),
            'best_immediate_latency': float(best_latency),
            'best_immediate_cost': float(best_cost),
            'candidate_count': int(len(candidate_actions)),
        }

    def _resolve_warmstart_teacher(
        self,
    ) -> tuple[Callable[[np.ndarray], int], Dict[str, Any]]:
        """Resolve the teacher used for supervised warm start."""
        action_space = getattr(self.env, 'action_space_wrapper', None)
        if action_space is None:
            raise ValueError(
                "imitation_warmstart requires env.action_space_wrapper"
            )

        mode = str(self.imitation_warmstart.get('mode', 'fixed_config'))
        if mode == 'fixed_action':
            expert_action = int(self.imitation_warmstart['expert_action'])
            resolved_config = action_space.get_configuration(expert_action)
            return (
                lambda _observation: int(expert_action),
                {
                    'mode': mode,
                    'action_id': int(expert_action),
                    'configuration': resolved_config.to_dict(),
                },
            )

        if mode == 'fixed_config':
            config = self.imitation_warmstart.get('expert_config')
            if not isinstance(config, dict):
                raise ValueError(
                    "imitation_warmstart.expert_config must be provided "
                    "when mode='fixed_config'"
                )
            expert_action = action_space.get_action_id(
                memory_mb=int(config['memory_mb']),
                architecture=str(config['architecture']),
                timeout_sec=int(config['timeout_sec']),
            )
            resolved_config = action_space.get_configuration(expert_action)
            return (
                lambda _observation: int(expert_action),
                {
                    'mode': mode,
                    'action_id': int(expert_action),
                    'configuration': resolved_config.to_dict(),
                },
            )

        if mode == 'greedy_policy':
            return self._select_greedy_teacher_action, {
                'mode': mode,
                'policy': 'greedy_category',
            }

        if mode == 'oracle_shortlist':
            candidate_actions = self._resolve_oracle_shortlist_actions()
            return None, {
                'mode': mode,
                'policy': 'one_step_oracle_shortlist',
                'candidate_actions': candidate_actions,
                'candidate_configurations': [
                    action_space.get_configuration(action_id).to_dict()
                    for action_id in candidate_actions
                ],
            }

        raise ValueError(
            "Unsupported imitation_warmstart.mode: "
            f"{mode}. Supported: ['fixed_action', 'fixed_config', "
            "'greedy_policy', 'oracle_shortlist']"
        )

    def _collect_imitation_dataset(
        self,
        teacher_policy: Callable[[np.ndarray], int],
        n_episodes: int,
        progress_label: str,
    ) -> Dict[str, Any]:
        """Collect state-action pairs from one teacher policy."""
        observations = []
        actions = []
        episode_mean_rewards = []
        env = self.env
        load_pattern = getattr(env, 'load_pattern', None)

        for episode in range(n_episodes):
            episode_seed = None if self.seed is None else self.seed + 10_000 + episode
            if load_pattern:
                obs, _ = env.reset(
                    seed=episode_seed,
                    options={'load_pattern': load_pattern},
                )
            else:
                obs, _ = env.reset(seed=episode_seed)

            done = False
            truncated = False
            step_rewards = []
            while not (done or truncated):
                teacher_action = int(teacher_policy(obs))
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(teacher_action)
                obs, reward, done, truncated, info = env.step(teacher_action)
                step_rewards.append(float(info.get('raw_reward', reward)))

            episode_mean_reward = (
                float(np.mean(step_rewards)) if step_rewards else 0.0
            )
            episode_mean_rewards.append(episode_mean_reward)
            print(
                f"[PPO WarmStart] {progress_label}: "
                f"collect {episode + 1}/{n_episodes}, "
                f"episode_reward={episode_mean_reward:.4f}",
                flush=True,
            )

        if not observations:
            raise ValueError("Warm-start rollout produced no observations.")

        return {
            'observations': np.asarray(observations, dtype=np.float32),
            'actions': np.asarray(actions, dtype=np.int64),
            'episode_mean_rewards': episode_mean_rewards,
        }

    def _collect_oracle_shortlist_dataset(
        self,
        candidate_actions: list[int],
        n_episodes: int,
        progress_label: str,
    ) -> Dict[str, Any]:
        """Collect state-action pairs from a one-step oracle over a shortlist."""
        observations = []
        actions = []
        episode_mean_rewards = []
        oracle_probe_evaluations = 0
        env = self.env
        load_pattern = getattr(env, 'load_pattern', None)

        for episode in range(n_episodes):
            episode_seed = None if self.seed is None else self.seed + 10_000 + episode
            if load_pattern:
                obs, _ = env.reset(
                    seed=episode_seed,
                    options={'load_pattern': load_pattern},
                )
            else:
                obs, _ = env.reset(seed=episode_seed)

            done = False
            truncated = False
            step_rewards = []
            while not (done or truncated):
                teacher_action, probe_summary = self._select_oracle_shortlist_action(
                    env=env,
                    candidate_actions=candidate_actions,
                )
                oracle_probe_evaluations += int(probe_summary['candidate_count'])
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(int(teacher_action))
                obs, reward, done, truncated, info = env.step(int(teacher_action))
                step_rewards.append(float(info.get('raw_reward', reward)))

            episode_mean_reward = (
                float(np.mean(step_rewards)) if step_rewards else 0.0
            )
            episode_mean_rewards.append(episode_mean_reward)
            print(
                f"[PPO WarmStart] {progress_label}: "
                f"collect {episode + 1}/{n_episodes}, "
                f"episode_reward={episode_mean_reward:.4f}, "
                f"oracle_candidates={len(candidate_actions)}",
                flush=True,
            )

        if not observations:
            raise ValueError("Oracle warm-start rollout produced no observations.")

        return {
            'observations': np.asarray(observations, dtype=np.float32),
            'actions': np.asarray(actions, dtype=np.int64),
            'episode_mean_rewards': episode_mean_rewards,
            'oracle_probe_evaluations': int(oracle_probe_evaluations),
        }

    def _summarize_teacher_actions(
        self,
        action_batch: np.ndarray,
        top_k: int = 5,
    ) -> list[Dict[str, Any]]:
        """Summarize the most common teacher actions in one dataset."""
        action_space = getattr(self.env, 'action_space_wrapper', None)
        if action_space is None:
            return []

        counts = Counter(int(action) for action in action_batch.tolist())
        total = max(1, sum(counts.values()))
        summary = []
        for action_id, count in counts.most_common(max(1, top_k)):
            config = action_space.get_configuration(int(action_id))
            summary.append({
                'action_id': int(action_id),
                'count': int(count),
                'fraction': float(count / total),
                'configuration': config.to_dict(),
            })
        return summary

    def _measure_imitation_accuracy(
        self,
        observation_batch: np.ndarray,
        action_batch: np.ndarray,
    ) -> float:
        """Measure deterministic policy agreement on one imitation batch."""
        with th.no_grad():
            obs_tensor = th.as_tensor(
                observation_batch,
                dtype=th.float32,
                device=self.model.device,
            )
            action_tensor = th.as_tensor(
                action_batch,
                dtype=th.long,
                device=self.model.device,
            )
            predicted_actions, _, _ = self.model.policy(
                obs_tensor,
                deterministic=True,
            )
            predicted_actions = predicted_actions.view(-1).long()
            matches = predicted_actions.eq(action_tensor).float()
        return float(matches.mean().item())

    def _run_imitation_warmstart(
        self,
        progress_label: str,
    ) -> Dict[str, Any] | None:
        """Run a short supervised pretraining stage before PPO updates."""
        if not self.imitation_warmstart.get('enabled', False):
            return None

        n_episodes = int(self.imitation_warmstart.get('n_episodes', 8))
        n_epochs = int(self.imitation_warmstart.get('epochs', 5))
        batch_size = int(self.imitation_warmstart.get('batch_size', 256))
        if n_episodes <= 0 or n_epochs <= 0 or batch_size <= 0:
            raise ValueError(
                "imitation_warmstart requires positive n_episodes, epochs, "
                "and batch_size"
            )

        teacher_policy, teacher_summary = self._resolve_warmstart_teacher()
        if teacher_summary['mode'] == 'oracle_shortlist':
            candidate_actions = list(teacher_summary['candidate_actions'])
            bootstrap_top_k = int(
                self.imitation_warmstart.get('bootstrap_top_k', 0)
            )
            bootstrap_eval_episodes = int(
                self.imitation_warmstart.get('bootstrap_eval_episodes', 1)
            )
            if bootstrap_top_k > 0 and len(candidate_actions) > bootstrap_top_k:
                ranked_candidates = self._rank_fixed_policy_candidates(
                    candidate_actions=candidate_actions,
                    n_episodes=bootstrap_eval_episodes,
                    progress_label=progress_label,
                )
                selected_candidates = ranked_candidates[
                    :min(bootstrap_top_k, len(ranked_candidates))
                ]
                selected_actions = [
                    int(item['action_id']) for item in selected_candidates
                ]
                teacher_summary['candidate_pool_actions'] = candidate_actions
                teacher_summary['candidate_pool_configurations'] = list(
                    teacher_summary['candidate_configurations']
                )
                teacher_summary['candidate_actions'] = selected_actions
                teacher_summary['candidate_configurations'] = [
                    item['configuration'] for item in selected_candidates
                ]
                teacher_summary['policy'] = (
                    'one_step_oracle_shortlist_bootstrap'
                )
                teacher_summary['bootstrap_shortlist'] = {
                    'enabled': True,
                    'candidate_pool_size': int(len(candidate_actions)),
                    'eval_episodes': int(bootstrap_eval_episodes),
                    'top_k': int(len(selected_actions)),
                    'ranked_candidates': ranked_candidates[
                        :max(len(selected_actions), 8)
                    ],
                }
                candidate_actions = selected_actions
                shortlist_summary = ", ".join(
                    f"{item['action_id']}:{item['configuration']['memory_mb']}"
                    f"/{item['configuration']['architecture']}"
                    f"/{item['configuration']['timeout_sec']}"
                    for item in selected_candidates
                )
                print(
                    f"[PPO WarmStart] {progress_label}: "
                    f"bootstrap selected {shortlist_summary}",
                    flush=True,
                )
            collection = self._collect_oracle_shortlist_dataset(
                candidate_actions=candidate_actions,
                n_episodes=n_episodes,
                progress_label=progress_label,
            )
        else:
            if teacher_policy is None:
                raise ValueError(
                    "Warm-start teacher policy must not be None for this mode"
                )
            collection = self._collect_imitation_dataset(
                teacher_policy=teacher_policy,
                n_episodes=n_episodes,
                progress_label=progress_label,
            )
        observations = collection['observations']
        actions = collection['actions']

        dataset_size = int(observations.shape[0])
        optimizer = self.model.policy.optimizer
        losses = []
        epoch_summaries = []
        warmstart_start_time = time.time()
        self.model.policy.set_training_mode(True)

        for epoch in range(n_epochs):
            permutation = np.random.permutation(dataset_size)
            epoch_losses = []
            for batch_start in range(0, dataset_size, batch_size):
                batch_indices = permutation[batch_start: batch_start + batch_size]
                obs_tensor = th.as_tensor(
                    observations[batch_indices],
                    dtype=th.float32,
                    device=self.model.device,
                )
                action_tensor = th.as_tensor(
                    actions[batch_indices],
                    dtype=th.long,
                    device=self.model.device,
                )
                _, log_prob, _ = self.model.policy.evaluate_actions(
                    obs_tensor,
                    action_tensor,
                )
                loss = -log_prob.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.model.policy.parameters(),
                    self.max_grad_norm,
                )
                optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                epoch_losses.append(loss_value)
                losses.append(loss_value)

            epoch_accuracy = self._measure_imitation_accuracy(
                observation_batch=observations,
                action_batch=actions,
            )
            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            epoch_summaries.append({
                'epoch': int(epoch + 1),
                'mean_loss': epoch_loss,
                'policy_accuracy': epoch_accuracy,
            })
            print(
                f"[PPO WarmStart] {progress_label}: "
                f"epoch {epoch + 1}/{n_epochs}, "
                f"loss={epoch_loss:.4f}, "
                f"policy_accuracy={epoch_accuracy:.4f}",
                flush=True,
            )

        self.model.policy.set_training_mode(False)
        total_time = time.time() - warmstart_start_time
        summary = {
            'enabled': True,
            'dataset_steps': dataset_size,
            'collection_episodes': int(n_episodes),
            'epochs': int(n_epochs),
            'batch_size': int(batch_size),
            'teacher': teacher_summary,
            'teacher_mean_episode_reward': float(
                np.mean(collection['episode_mean_rewards'])
            ),
            'teacher_std_episode_reward': float(
                np.std(collection['episode_mean_rewards'])
            ),
            'teacher_action_support': int(len(set(actions.tolist()))),
            'teacher_top_actions': self._summarize_teacher_actions(actions),
            'warmstart_time_sec': float(total_time),
            'mean_loss': float(np.mean(losses)) if losses else 0.0,
            'final_loss': float(losses[-1]) if losses else 0.0,
            'final_policy_accuracy': float(
                epoch_summaries[-1]['policy_accuracy']
            ) if epoch_summaries else 0.0,
            'epoch_summaries': epoch_summaries,
        }
        if 'action_id' in teacher_summary:
            summary['expert'] = teacher_summary
            summary['expert_mean_episode_reward'] = (
                summary['teacher_mean_episode_reward']
            )
            summary['expert_std_episode_reward'] = (
                summary['teacher_std_episode_reward']
            )
        if 'oracle_probe_evaluations' in collection:
            summary['oracle_probe_evaluations'] = int(
                collection['oracle_probe_evaluations']
            )
        print(
            f"[PPO WarmStart] {progress_label}: "
            f"done dataset_steps={dataset_size}, "
            f"teacher_mode={teacher_summary['mode']}, "
            f"action_support={summary['teacher_action_support']}, "
            f"warmstart_time={total_time:.2f}s",
            flush=True,
        )
        return summary

    def train(
        self,
        callback=None,
        progress_label: str | None = None,
    ):
        """
        Train the PPO model.

        Args:
            callback: Additional callback or callback list.
            progress_label: Label used in progress logs.
        """
        start_time = time.time()
        self.monitor_callback.reset_progress(
            total_timesteps=self.total_timesteps,
            progress_label=progress_label,
        )
        self.imitation_warmstart_summary = self._run_imitation_warmstart(
            progress_label=progress_label or 'PPO',
        )

        # Merge callbacks.
        callbacks = [self.monitor_callback]
        checkpoint_callback = None
        if self.checkpoint_selection.get('enabled', False):
            if self.validation_env is None:
                raise ValueError(
                    "checkpoint_selection requires validation_env"
                )
            checkpoint_callback = ValidationCheckpointCallback(
                owner=self,
                eval_env=self.validation_env,
                eval_freq_steps=int(
                    self.checkpoint_selection.get('eval_freq_steps', 2048)
                ),
                n_episodes=int(
                    self.checkpoint_selection.get('n_episodes', 3)
                ),
                metric=str(
                    self.checkpoint_selection.get('metric', 'mean_reward')
                ),
                min_timestep=int(
                    self.checkpoint_selection.get('min_timestep', 0)
                ),
                progress_label=progress_label or 'PPO',
            )
            callbacks.append(checkpoint_callback)
        if callback is not None:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)

        # Train.
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            progress_bar=False,
        )

        training_time = time.time() - start_time
        if checkpoint_callback is not None:
            checkpoint_callback.restore_best_policy()
            self.checkpoint_selection_summary = (
                checkpoint_callback.build_summary()
            )
            print(
                f"[PPO Val] {progress_label or 'PPO'}: "
                f"restored_best_checkpoint at "
                f"t={self.checkpoint_selection_summary['best_timestep']}, "
                f"{self.checkpoint_selection_summary['metric']}="
                f"{self.checkpoint_selection_summary['best_metric']:.4f}",
                flush=True,
            )
        else:
            self.checkpoint_selection_summary = None

        recent_rewards = self.monitor_callback.episode_rewards[-10:]
        recent_mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        print(f"\nTraining completed in {training_time:.2f}s", flush=True)
        print(
            f"Episodes finished: {len(self.monitor_callback.episode_rewards)}",
            flush=True,
        )
        print(f"Final mean training reward: {recent_mean_reward:.4f}", flush=True)

        return training_time

    def _run_policy_episodes(
        self,
        n_episodes: int,
        eval_env=None,
        progress_label: str | None = None,
        observation_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        collect_observations: bool = False,
        emit_progress: bool = True,
        log_prefix: str = 'PPO Eval',
    ) -> tuple[Dict[str, float], np.ndarray | None]:
        """Run deterministic policy episodes with an optional observation transform."""
        all_rewards = []
        all_normalized_rewards = []
        all_latencies = []
        all_costs = []
        all_success_rates = []
        all_episode_lengths = []
        collected_observations = []
        progress_label = progress_label or 'PPO'
        start_time = time.time()
        eval_env = eval_env if eval_env is not None else self.env

        load_pattern = None
        if hasattr(eval_env, 'load_pattern'):
            load_pattern = eval_env.load_pattern

        for episode in range(n_episodes):
            episode_seed = None if self.seed is None else self.seed + episode
            if load_pattern:
                obs, _ = eval_env.reset(
                    seed=episode_seed,
                    options={'load_pattern': load_pattern},
                )
            else:
                obs, _ = eval_env.reset(seed=episode_seed)
            done = False
            truncated = False

            episode_rewards = []
            episode_normalized_rewards = []
            episode_latencies = []
            episode_costs = []
            episode_successes = []
            steps = 0

            while not (done or truncated):
                model_obs = np.asarray(obs, dtype=np.float32)
                if collect_observations:
                    collected_observations.append(np.array(model_obs, copy=True))
                if observation_transform is not None:
                    model_obs = np.asarray(
                        observation_transform(model_obs),
                        dtype=np.float32,
                    )

                action, _ = self.model.predict(model_obs, deterministic=True)
                action = int(action)
                obs, reward, done, truncated, info = eval_env.step(action)

                episode_rewards.append(float(info.get('raw_reward', reward)))
                episode_normalized_rewards.append(
                    float(info.get('normalized_reward', reward))
                )
                episode_latencies.append(info['metrics']['latency'])
                episode_costs.append(info['metrics']['cost'])
                episode_successes.append(float(info['metrics']['success']))
                steps += 1

            all_rewards.append(np.mean(episode_rewards))
            all_normalized_rewards.append(np.mean(episode_normalized_rewards))
            all_latencies.append(np.mean(episode_latencies))
            all_costs.append(np.mean(episode_costs))
            all_success_rates.append(np.mean(episode_successes))
            all_episode_lengths.append(steps)

            if emit_progress:
                print(
                    f"[{log_prefix}] {progress_label}: "
                    f"{episode + 1}/{n_episodes}, "
                    f"raw_reward={all_rewards[-1]:.4f}, "
                    f"normalized_reward={all_normalized_rewards[-1]:.4f}, "
                    f"latency={all_latencies[-1]:.2f} ms, "
                    f"success={all_success_rates[-1]:.3f}, "
                    f"elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )

        metrics = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_normalized_reward': np.mean(all_normalized_rewards),
            'std_normalized_reward': np.std(all_normalized_rewards),
            'mean_latency': np.mean(all_latencies),
            'std_latency': np.std(all_latencies),
            'mean_cost': np.mean(all_costs),
            'std_cost': np.std(all_costs),
            'mean_success_rate': np.mean(all_success_rates),
            'mean_episode_length': np.mean(all_episode_lengths),
        }
        if not collect_observations:
            return metrics, None
        if collected_observations:
            return metrics, np.asarray(collected_observations, dtype=np.float32)
        return metrics, np.zeros((0, self.env.observation_space.shape[0]), dtype=np.float32)

    def evaluate(
        self,
        n_episodes: int = 30,
        progress_label: str | None = None,
        eval_env=None,
    ) -> Dict:
        """Evaluate the PPO policy."""
        metrics, _ = self._run_policy_episodes(
            n_episodes=n_episodes,
            eval_env=eval_env,
            progress_label=progress_label,
            observation_transform=None,
            collect_observations=False,
            emit_progress=True,
            log_prefix='PPO Eval',
        )
        return metrics

    def evaluate_feature_attribution(
        self,
        n_episodes: int = 6,
        eval_env=None,
        progress_label: str | None = None,
        single_feature_groups: tuple[str, ...] = (
            'load',
            'category',
            'history',
            'context',
        ),
        max_single_features: int | None = None,
    ) -> Dict[str, Any]:
        """Measure reward sensitivity to feature masking at inference time."""
        progress_label = progress_label or 'PPO Attribution'
        eval_env = eval_env if eval_env is not None else self.env
        state_space = self._resolve_state_space(eval_env)
        code_feature_dim = getattr(
            state_space,
            'code_feature_dim',
            StateSpace.DEFAULT_CODE_FEATURE_DIM,
        )
        feature_layout = StateSpace.get_feature_layout(
            code_feature_dim=code_feature_dim
        )
        group_slices = StateSpace.get_group_slices(
            code_feature_dim=code_feature_dim
        )

        baseline_metrics, reference_observations = self._run_policy_episodes(
            n_episodes=n_episodes,
            eval_env=eval_env,
            progress_label=f"{progress_label} [baseline]",
            collect_observations=True,
            emit_progress=False,
            log_prefix='PPO Attr',
        )
        if reference_observations is None or reference_observations.size == 0:
            raise ValueError("No reference observations collected for attribution.")
        reference_vector = np.mean(reference_observations, axis=0)

        def _build_mask_transform(indices: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
            def _transform(obs: np.ndarray) -> np.ndarray:
                masked = np.array(obs, copy=True)
                masked[indices] = reference_vector[indices]
                return masked
            return _transform

        def _summarize_row(
            name: str,
            group: str,
            indices: np.ndarray,
            metrics: Dict[str, float],
        ) -> Dict[str, Any]:
            signed_reward_delta = float(
                metrics['mean_reward'] - baseline_metrics['mean_reward']
            )
            return {
                'name': name,
                'group': group,
                'n_dims': int(indices.size),
                'masked_mean_reward': float(metrics['mean_reward']),
                'masked_mean_latency': float(metrics['mean_latency']),
                'masked_mean_cost': float(metrics['mean_cost']),
                'masked_mean_success_rate': float(metrics['mean_success_rate']),
                'reward_drop': float(
                    baseline_metrics['mean_reward'] - metrics['mean_reward']
                ),
                'signed_reward_delta': signed_reward_delta,
                'absolute_reward_shift': abs(signed_reward_delta),
                'latency_delta_ms': float(
                    metrics['mean_latency'] - baseline_metrics['mean_latency']
                ),
                'cost_delta': float(
                    metrics['mean_cost'] - baseline_metrics['mean_cost']
                ),
                'success_rate_delta': float(
                    metrics['mean_success_rate']
                    - baseline_metrics['mean_success_rate']
                ),
            }

        group_importance = []
        for group_name, group_slice in group_slices.items():
            indices = np.arange(group_slice.start, group_slice.stop, dtype=int)
            masked_metrics, _ = self._run_policy_episodes(
                n_episodes=n_episodes,
                eval_env=eval_env,
                progress_label=f"{progress_label} [{group_name}]",
                observation_transform=_build_mask_transform(indices),
                collect_observations=False,
                emit_progress=False,
                log_prefix='PPO Attr',
            )
            row = _summarize_row(
                name=group_name,
                group=group_name,
                indices=indices,
                metrics=masked_metrics,
            )
            group_importance.append(row)
            print(
                f"[PPO Attr] {progress_label}: group={group_name}, "
                f"reward_drop={row['reward_drop']:+.4f}, "
                f"latency_delta_ms={row['latency_delta_ms']:+.2f}, "
                f"success_rate_delta={row['success_rate_delta']:+.4f}",
                flush=True,
            )

        group_importance.sort(key=lambda item: item['reward_drop'], reverse=True)

        interpretable_features = [
            feature
            for feature in feature_layout
            if feature['interpretable'] and feature['group'] in single_feature_groups
        ]
        if max_single_features is not None:
            interpretable_features = interpretable_features[:max_single_features]

        single_feature_importance = []
        for feature in interpretable_features:
            indices = np.asarray([int(feature['index'])], dtype=int)
            masked_metrics, _ = self._run_policy_episodes(
                n_episodes=n_episodes,
                eval_env=eval_env,
                progress_label=f"{progress_label} [{feature['name']}]",
                observation_transform=_build_mask_transform(indices),
                collect_observations=False,
                emit_progress=False,
                log_prefix='PPO Attr',
            )
            single_feature_importance.append(
                _summarize_row(
                    name=str(feature['name']),
                    group=str(feature['group']),
                    indices=indices,
                    metrics=masked_metrics,
                )
            )

        single_feature_importance.sort(
            key=lambda item: item['reward_drop'],
            reverse=True,
        )

        return {
            'method': 'counterfactual_mean_masking',
            'n_episodes': int(n_episodes),
            'single_feature_groups': list(single_feature_groups),
            'reference': {
                'mask_value': 'empirical_mean_over_unmasked_rollouts',
                'reference_observations': int(reference_observations.shape[0]),
            },
            'baseline': {
                'mean_reward': float(baseline_metrics['mean_reward']),
                'mean_latency': float(baseline_metrics['mean_latency']),
                'mean_cost': float(baseline_metrics['mean_cost']),
                'mean_success_rate': float(
                    baseline_metrics['mean_success_rate']
                ),
            },
            'group_importance': group_importance,
            'single_feature_importance': single_feature_importance,
        }

    def save(self, path: str):
        """Save the PPO model."""
        self.model.save(path)

    def load(self, path: str):
        """Load the PPO model."""
        self.model = PPO.load(path, env=self.env)


class OnlineBayesOptBaseline:
    """Online BayesOpt baseline used for dynamic-load comparisons."""

    def __init__(self, env, reoptimize_freq: int = 10, seed: int | None = None):
        """
        Initialize the online BayesOpt baseline.

        Args:
            env: Evaluation environment.
            reoptimize_freq: Re-optimization interval in control steps.
        """
        self.env = env
        self.reoptimize_freq = reoptimize_freq
        self.seed = seed
        self.current_best_action = None
        self.step_count = 0
        self.progress_label = 'BayesOpt'
        self.current_episode = 0
        self.total_episodes = 0
        self.optimization_env = self._build_shadow_env(env)

        # Import here to avoid circular imports at module load time.
        from .baselines import BayesianOptimizationPolicy
        self.bayes_opt = BayesianOptimizationPolicy(self.optimization_env)

    def _build_shadow_env(self, env):
        """Create an isolated environment for BayesOpt probes.

        BayesOpt internally calls ``reset()`` to score candidate actions. If it
        reuses the live evaluation environment, those resets restart the current
        episode and can prevent truncation from ever being reached.
        """
        from .dynamic_environment import DynamicLoadEnvironment
        from .environment import ServerlessFunctionEnv

        if not hasattr(env, 'base_env'):
            return env

        source_env = env.base_env
        action_space_config = {
            'memory_options': list(source_env.action_space_wrapper.MEMORY_OPTIONS),
            'architecture_options': list(
                source_env.action_space_wrapper.ARCHITECTURE_OPTIONS
            ),
            'timeout_options': list(source_env.action_space_wrapper.TIMEOUT_OPTIONS),
        }
        shadow_base_env = ServerlessFunctionEnv(
            benchmark=source_env.benchmark,
            deployment=source_env.deployment,
            config_path=source_env.config_path,
            sebs_root=str(source_env.sebs_root),
            max_steps=source_env.max_steps,
            reward_weights=dict(source_env.reward_weights),
            reward_penalties=dict(source_env.reward_penalties),
            enable_real_execution=source_env.enable_real_execution,
            normalize_reward=source_env.normalize_reward,
            success_bonus=source_env.success_bonus,
            cost_normalization=source_env.cost_normalization,
            calibration_dir=getattr(source_env.service_model, 'calibration_dir', None),
            calibration_dirs=getattr(source_env.service_model, 'calibration_dirs', None),
            step_duration_sec=source_env.step_duration_sec,
            max_containers=source_env.container_pool.max_containers,
            container_ttl_sec=source_env.container_pool.ttl_sec,
            state_space=source_env.state_space.fork_for_env(),
            action_space_config=action_space_config,
            log_init=False,
        )
        shadow_env = DynamicLoadEnvironment(
            shadow_base_env,
            episode_length=env.episode_length,
            load_change_freq=env.load_change_freq,
            switch_penalty=env.switch_penalty,
            adaptation_penalty=env.adaptation_penalty,
            workload_source=env.default_workload_source,
            azure_profile_path=str(env.azure_profile_path),
            azure_summary_path=(
                str(env.azure_summary_path)
                if env.azure_summary_path is not None
                else None
            ),
            azure_top_k=env.azure_top_k,
            azure_arrival_scale=env.azure_arrival_scale,
            azure_max_arrivals_per_step=env.azure_max_arrivals_per_step,
            azure_profile_selection=env.azure_profile_selection,
            azure_selection_pool_size=env.azure_selection_pool_size,
            azure_target_concurrency=env.azure_target_concurrency,
        )
        shadow_env.load_pattern = getattr(env, 'load_pattern', shadow_env.load_pattern)
        return shadow_env

    def reset(self):
        """Reset per-episode BayesOpt state."""
        self.step_count = 0
        self.current_best_action = None
        self.bayes_opt.reset()
        if hasattr(self.optimization_env, 'load_pattern') and hasattr(self.env, 'load_pattern'):
            self.optimization_env.load_pattern = self.env.load_pattern

    def select_action(self, obs: np.ndarray) -> int:
        """Select the next action."""
        del obs
        # Re-optimize every N control steps.
        if (self.step_count % self.reoptimize_freq == 0 or
            self.current_best_action is None):
            print(
                f"[OnlineBayesOpt] {self.progress_label}: "
                f"episode {self.current_episode}/{self.total_episodes}, "
                f"step {self.step_count}, reoptimize...",
                flush=True,
            )
            self.current_best_action = self.bayes_opt.optimize(n_iterations=20)

        self.step_count += 1
        return self.current_best_action

    def evaluate(
        self,
        n_episodes: int = 30,
        progress_label: str | None = None,
    ) -> Dict:
        """Evaluate the online BayesOpt policy."""
        all_rewards = []
        all_normalized_rewards = []
        all_latencies = []
        all_costs = []
        all_success_rates = []
        progress_label = progress_label or 'BayesOpt'
        self.progress_label = progress_label
        self.total_episodes = n_episodes
        start_time = time.time()
        load_pattern = None
        if hasattr(self.env, 'load_pattern'):
            load_pattern = self.env.load_pattern

        for episode in range(n_episodes):
            episode_seed = None if self.seed is None else self.seed + episode
            if episode_seed is not None:
                np.random.seed(episode_seed)
            self.current_episode = episode + 1
            self.reset()
            if load_pattern:
                obs, _ = self.env.reset(
                    seed=episode_seed,
                    options={'load_pattern': load_pattern},
                )
            else:
                obs, _ = self.env.reset(seed=episode_seed)
            done = False
            truncated = False

            episode_rewards = []
            episode_normalized_rewards = []
            episode_latencies = []
            episode_costs = []
            episode_successes = []

            while not (done or truncated):
                action = self.select_action(obs)
                obs, reward, done, truncated, info = self.env.step(action)

                episode_rewards.append(float(info.get('raw_reward', reward)))
                episode_normalized_rewards.append(
                    float(info.get('normalized_reward', reward))
                )
                episode_latencies.append(info['metrics']['latency'])
                episode_costs.append(info['metrics']['cost'])
                episode_successes.append(float(info['metrics']['success']))

            all_rewards.append(np.mean(episode_rewards))
            all_normalized_rewards.append(np.mean(episode_normalized_rewards))
            all_latencies.append(np.mean(episode_latencies))
            all_costs.append(np.mean(episode_costs))
            all_success_rates.append(np.mean(episode_successes))
            print(
                f"[BayesOpt Eval] {progress_label}: "
                f"{episode + 1}/{n_episodes}, "
                f"raw_reward={all_rewards[-1]:.4f}, "
                f"normalized_reward={all_normalized_rewards[-1]:.4f}, "
                f"latency={all_latencies[-1]:.2f} ms, "
                f"success={all_success_rates[-1]:.3f}, "
                f"elapsed={time.time() - start_time:.1f}s",
                flush=True,
            )

        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_normalized_reward': np.mean(all_normalized_rewards),
            'std_normalized_reward': np.std(all_normalized_rewards),
            'mean_latency': np.mean(all_latencies),
            'std_latency': np.std(all_latencies),
            'mean_cost': np.mean(all_costs),
            'std_cost': np.std(all_costs),
            'mean_success_rate': np.mean(all_success_rates),
        }
