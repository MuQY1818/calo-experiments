"""Single-algorithm execution for CALO experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .environment_factory import AlgorithmEnvironments, PPO_ALGORITHMS
from .experiment_config import ExperimentConfig


ALGORITHM_DISPLAY_NAMES = {
    "ppo": "CALO",
    "ppo_flat": "Flat-Fusion CALO",
    "ppo_load_only": "Load-Only CALO",
    "bayes_opt_online": "Online BO",
    "default": "Provider Default",
    "random": "Random",
    "greedy": "Greedy Profiling",
}


def algorithm_display_name(algorithm: str) -> str:
    """Return the public label for an algorithm identifier."""
    return ALGORITHM_DISPLAY_NAMES.get(algorithm, algorithm)


def _merge_nested(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge PPO overrides without mutating the base config."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_nested(current, value)
        else:
            merged[key] = value
    return merged


class AlgorithmExecutor:
    """Train or evaluate exactly one configured algorithm."""

    def __init__(
        self,
        config: ExperimentConfig,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """Store protocol settings and an optional stage callback."""
        self.config = config
        self.training = config.runtime_section("training")
        self.evaluation = config.runtime_section("evaluation")
        self.baseline = config.runtime_section("baseline")
        raw_settings = config.data.get("algorithm_settings", {})
        self.algorithm_settings = dict(raw_settings) if isinstance(raw_settings, dict) else {}
        raw_attribution = config.data.get("feature_attribution", {})
        self.feature_attribution = (
            dict(raw_attribution) if isinstance(raw_attribution, dict) else {}
        )
        self.progress_callback = progress_callback or (lambda stage: None)

    def execute(
        self,
        algorithm: str,
        environments: AlgorithmEnvironments,
        *,
        benchmark: str,
        load_pattern: str,
        seed: int,
    ) -> dict[str, Any]:
        """Execute one algorithm and return normalized metric fields."""
        label = f"{benchmark} / {load_pattern} [{algorithm_display_name(algorithm)}]"
        if algorithm in PPO_ALGORITHMS:
            return self._execute_ppo(algorithm, environments, seed=seed, label=label)
        if algorithm == "bayes_opt_online":
            return self._execute_online_bo(environments, seed=seed, label=label)
        if algorithm == "default":
            return self._execute_default(environments, seed=seed, label=label)
        if algorithm == "random":
            return self._execute_random(environments, seed=seed, label=label)
        if algorithm == "greedy":
            return self._execute_greedy(environments, seed=seed, label=label)
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _settings_for(self, algorithm: str) -> dict[str, Any]:
        """Return algorithm-specific config overrides."""
        raw = self.algorithm_settings.get(algorithm, {})
        return dict(raw) if isinstance(raw, dict) else {}

    def _ppo_kwargs(self, algorithm: str) -> dict[str, Any]:
        """Resolve the accepted PPO config plus ablation overrides."""
        kwargs = deepcopy(dict(self.training["ppo_kwargs"]))
        settings = self._settings_for(algorithm)
        extractor = settings.get("feature_extractor_name")
        if extractor is not None:
            kwargs["feature_extractor_name"] = str(extractor)
        overrides = settings.get("ppo_kwargs_overrides")
        if isinstance(overrides, dict):
            kwargs = _merge_nested(kwargs, overrides)
        return kwargs

    def _execute_ppo(
        self,
        algorithm: str,
        environments: AlgorithmEnvironments,
        *,
        seed: int,
        label: str,
    ) -> dict[str, Any]:
        """Train and evaluate the CALO policy family."""
        from .pure_ppo import PurePPO

        self.progress_callback(f"training_{algorithm}")
        policy = PurePPO(
            environments.training,
            total_timesteps=int(self.training["total_timesteps"]),
            ppo_kwargs=self._ppo_kwargs(algorithm),
            seed=seed,
            imitation_warmstart=self.training.get("imitation_warmstart"),
            validation_env=environments.evaluation,
            checkpoint_selection=self.training.get("checkpoint_selection"),
        )
        training_time = policy.train(progress_label=label)
        self.progress_callback(f"evaluating_{algorithm}")
        metrics = policy.evaluate(
            n_episodes=int(self.evaluation["n_episodes"]),
            progress_label=label,
            eval_env=environments.evaluation,
        )
        if policy.imitation_warmstart_summary is not None:
            metrics["imitation_warmstart"] = policy.imitation_warmstart_summary
        if policy.checkpoint_selection_summary is not None:
            metrics["checkpoint_selection"] = policy.checkpoint_selection_summary
        if self._attribution_enabled(algorithm):
            self.progress_callback(f"attribution_{algorithm}")
            metrics["feature_attribution"] = policy.evaluate_feature_attribution(
                n_episodes=int(
                    self.feature_attribution.get(
                        "n_episodes",
                        self.evaluation["n_episodes"],
                    )
                ),
                eval_env=environments.evaluation,
                progress_label=label,
                single_feature_groups=tuple(
                    self.feature_attribution.get(
                        "single_feature_groups",
                        ["load", "category", "history", "context"],
                    )
                ),
                max_single_features=self.feature_attribution.get("max_single_features"),
            )
        metrics["training_time"] = float(training_time)
        return dict(metrics)

    def _attribution_enabled(self, algorithm: str) -> bool:
        """Return whether attribution is configured for this policy."""
        if not self.feature_attribution.get("enabled", False):
            return False
        targets = self.feature_attribution.get("algorithms", ["ppo"])
        return algorithm in targets

    def _execute_online_bo(
        self,
        environments: AlgorithmEnvironments,
        *,
        seed: int,
        label: str,
    ) -> dict[str, Any]:
        """Evaluate the stateful online Bayesian optimization baseline."""
        from .pure_ppo import OnlineBayesOptBaseline

        self.progress_callback("evaluating_bayes_opt_online")
        baseline = OnlineBayesOptBaseline(
            environments.evaluation,
            reoptimize_freq=int(self.baseline["reoptimize_freq"]),
            seed=seed,
        )
        return dict(
            baseline.evaluate(
                n_episodes=int(self.evaluation["n_episodes"]),
                progress_label=label,
            )
        )

    def _execute_default(
        self,
        environments: AlgorithmEnvironments,
        *,
        seed: int,
        label: str,
    ) -> dict[str, Any]:
        """Evaluate the provider-default resource configuration."""
        from .baselines import DefaultBaseline

        self.progress_callback("evaluating_default")
        baseline = DefaultBaseline(
            environments.evaluation,
            memory=int(self.baseline.get("default_memory", 512)),
        )
        return dict(
            baseline.evaluate(
                n_episodes=int(self.evaluation["n_episodes"]),
                progress_label=label,
                seed=seed,
            )
        )

    def _execute_random(
        self,
        environments: AlgorithmEnvironments,
        *,
        seed: int,
        label: str,
    ) -> dict[str, Any]:
        """Evaluate uniform random configuration selection."""
        from .baselines import RandomBaseline

        self.progress_callback("evaluating_random")
        return dict(
            RandomBaseline(environments.evaluation).evaluate(
                n_episodes=int(self.evaluation["n_episodes"]),
                progress_label=label,
                seed=seed,
            )
        )

    def _execute_greedy(
        self,
        environments: AlgorithmEnvironments,
        *,
        seed: int,
        label: str,
    ) -> dict[str, Any]:
        """Evaluate greedy profiling."""
        from .baselines import GreedyBaseline

        self.progress_callback("evaluating_greedy")
        return dict(
            GreedyBaseline(environments.evaluation).evaluate(
                n_episodes=int(self.evaluation["n_episodes"]),
                progress_label=label,
                seed=seed,
            )
        )
