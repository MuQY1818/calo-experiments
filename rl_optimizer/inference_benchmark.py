"""Optional CPU-only forward-pass measurement for the CALO policy shape."""

from __future__ import annotations

from importlib import metadata
import platform
import sys
import time
from typing import Any, Callable

import numpy as np


OBSERVATION_DIM = 79
ACTION_COUNT = 48
INFERENCE_WARMUP_PASSES = 1_000
INFERENCE_FORWARD_PASSES = 10_000


class InferenceBenchmarkError(RuntimeError):
    """Indicate that the local policy-shape benchmark cannot be executed."""


class InferenceBenchmark:
    """Construct and measure the accepted CALO actor-critic shape on CPU."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ):
        """Create a benchmark with an injectable monotonic clock for tests."""
        self.clock_ns = clock_ns

    def measure(self) -> dict[str, Any]:
        """Warm the policy and time exactly 10,000 single-state forwards."""
        try:
            import torch
            from gymnasium import spaces
            from stable_baselines3.common.policies import ActorCriticPolicy

            from .pure_ppo import CALOFeaturesExtractor
            from .state_space import StateSpace
        except ImportError as exc:
            raise InferenceBenchmarkError(
                "Inference measurement requires the installed project runtime "
                "dependencies. Reinstall with `python -m pip install -e .`."
            ) from exc

        state_dimension = (
            StateSpace.DEFAULT_CODE_FEATURE_DIM
            + StateSpace.LOAD_FEATURE_DIM
            + StateSpace.CATEGORY_DIM
            + StateSpace.HISTORY_DIM
            + StateSpace.CONTEXT_DIM
        )
        if state_dimension != OBSERVATION_DIM:
            raise InferenceBenchmarkError(
                f"CALO state layout is {state_dimension}-D; expected {OBSERVATION_DIM}-D."
            )

        torch.manual_seed(42)
        observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        action_space: Any = spaces.Discrete(ACTION_COUNT)
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            lr_schedule=lambda _: 0.0,
            features_extractor_class=CALOFeaturesExtractor,
            features_extractor_kwargs={
                "features_dim": 128,
                "code_dim": StateSpace.DEFAULT_CODE_FEATURE_DIM,
                "load_dim": StateSpace.LOAD_FEATURE_DIM,
                "category_dim": StateSpace.CATEGORY_DIM,
                "history_dim": StateSpace.HISTORY_DIM,
                "context_dim": StateSpace.CONTEXT_DIM,
            },
            net_arch={"pi": [64, 64], "vf": [64, 64]},
        ).to(torch.device("cpu"))
        policy.eval()
        observation = torch.zeros((1, OBSERVATION_DIM), dtype=torch.float32, device="cpu")

        timings_ms = np.empty(INFERENCE_FORWARD_PASSES, dtype=np.float64)
        with torch.inference_mode():
            for _ in range(INFERENCE_WARMUP_PASSES):
                policy(observation, deterministic=True)
            for index in range(INFERENCE_FORWARD_PASSES):
                started_ns = self.clock_ns()
                policy(observation, deterministic=True)
                timings_ms[index] = (self.clock_ns() - started_ns) / 1_000_000.0

        return {
            "scope": "local_cpu_randomly_initialized_policy_shape",
            "paper_claim_comparison": False,
            "device": "cpu",
            "observation_dim": OBSERVATION_DIM,
            "action_count": ACTION_COUNT,
            "features_extractor": "CALOFeaturesExtractor",
            "features_dim": 128,
            "actor_layers": [64, 64],
            "critic_layers": [64, 64],
            "warmup_passes": INFERENCE_WARMUP_PASSES,
            "forward_passes": INFERENCE_FORWARD_PASSES,
            "median_ms": float(np.median(timings_ms)),
            "p95_ms": float(np.percentile(timings_ms, 95)),
            "runtime": {
                "os": platform.system(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor() or "unknown",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "stable_baselines3": _package_version("stable-baselines3"),
                "numpy": np.__version__,
                "python_implementation": platform.python_implementation(),
                "byte_order": sys.byteorder,
            },
        }


def _package_version(package: str) -> str:
    """Return an installed package version without importing the package."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def measure_inference() -> dict[str, Any]:
    """Run the public, fixed-size local CPU inference measurement."""
    return InferenceBenchmark().measure()
