"""Domain-contract tests for the published CALO simulator."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from rl_optimizer.action_space import ActionSpace
from rl_optimizer.codebert_analyzer import (
    CODEBERT_MODEL,
    CODEBERT_REVISION,
    CodeBERTAnalyzer,
)
from rl_optimizer.container_pool import ContainerPool
from rl_optimizer.environment import ServerlessFunctionEnv
from rl_optimizer.environment_factory import EnvironmentFactory
from rl_optimizer.experiment_config import ExperimentConfig
from rl_optimizer.service_model import CalibratedServiceModel
from rl_optimizer.state_space import StateSpace
from rl_optimizer.workload_sources import SyntheticWorkloadSource, WorkloadStep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_ROOT = PROJECT_ROOT / "artifacts" / "calibration"
PAPER_CONFIG = PROJECT_ROOT / "config" / "rl_experiments" / "paper_full48.json"


class _FakeWarmSurrogate:
    """Minimal surrogate used to observe fallback routing."""

    is_ready = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict_warm(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> tuple[float, float]:
        """Record one prediction request and return fixed percentiles."""
        self.calls.append(
            {
                "benchmark": benchmark,
                "memory_mb": memory_mb,
                "timeout_sec": timeout_sec,
                "architecture": architecture,
            }
        )
        return 321.0, 456.0


class ActionSpaceContractTest(unittest.TestCase):
    """Verify the frozen action catalogs and their encodings."""

    def test_full_and_calibrated_presets(self) -> None:
        full = ActionSpace(preset="full_48")
        calibrated = ActionSpace(preset="calibrated_x64_8")

        self.assertEqual(full.n_actions, 48)
        self.assertEqual(full.MEMORY_OPTIONS, [128, 256, 512, 1024, 2048, 3008])
        self.assertEqual(full.ARCHITECTURE_OPTIONS, ["x64", "arm64"])
        self.assertEqual(full.TIMEOUT_OPTIONS, [60, 120, 300, 900])
        self.assertEqual(calibrated.n_actions, 8)
        self.assertEqual(calibrated.MEMORY_OPTIONS, [512, 1024, 2048, 3008])
        self.assertEqual(calibrated.ARCHITECTURE_OPTIONS, ["x64"])
        self.assertEqual(calibrated.TIMEOUT_OPTIONS, [120, 300])

    def test_encode_decode_and_constraint_mask(self) -> None:
        action_space = ActionSpace(preset="full_48")
        action = action_space.get_action_id(2048, "arm64", 300)

        encoded = action_space.encode_action(action)
        self.assertEqual(encoded.shape, (12,))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(float(encoded.sum()), 3.0)
        self.assertEqual(action_space.decode_action(encoded), action)

        mask = action_space.get_action_mask(
            {
                "max_memory": 512,
                "allowed_arch": ["arm64"],
                "max_timeout": 120,
            }
        )
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(int(mask.sum()), 6)
        for action_id in np.flatnonzero(mask):
            config = action_space.get_configuration(int(action_id))
            self.assertLessEqual(config.memory_mb, 512)
            self.assertEqual(config.architecture, "arm64")
            self.assertLessEqual(config.timeout_sec, 120)


class StateAndWorkloadContractTest(unittest.TestCase):
    """Verify the accepted state layout and deterministic workloads."""

    def test_default_state_has_frozen_79_dimension_layout(self) -> None:
        state_space = StateSpace(enable_code_features=False)
        with redirect_stdout(io.StringIO()):
            state_space.set_function("411.image-recognition")

        state = state_space.extract_state()
        layout = state_space.get_feature_layout()
        groups = Counter(str(item["group"]) for item in layout)
        slices = state_space.get_group_slices()

        self.assertEqual(state_space.state_dim, 79)
        self.assertEqual(state.shape, (79,))
        self.assertEqual(len(layout), 79)
        self.assertEqual([item["index"] for item in layout], list(range(79)))
        self.assertEqual(
            groups,
            {"code": 32, "load": 10, "category": 5, "history": 5, "context": 27},
        )
        self.assertEqual(slices["code"], slice(0, 32))
        self.assertEqual(slices["load"], slice(32, 42))
        self.assertEqual(slices["category"], slice(42, 47))
        self.assertEqual(slices["history"], slice(47, 52))
        self.assertEqual(slices["context"], slice(52, 79))
        np.testing.assert_array_equal(state[slices["code"]], np.zeros(32))
        np.testing.assert_array_equal(
            state[slices["category"]],
            np.asarray([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        )

    def test_synthetic_workload_seed_is_deterministic(self) -> None:
        first = SyntheticWorkloadSource(pattern="random", step_duration_sec=15.0)
        second = SyntheticWorkloadSource(pattern="random", step_duration_sec=15.0)
        different = SyntheticWorkloadSource(pattern="random", step_duration_sec=15.0)
        first.reset(seed=52)
        second.reset(seed=52)
        different.reset(seed=62)

        first_sequence = [first.next_step(index) for index in range(12)]
        second_sequence = [second.next_step(index) for index in range(12)]
        different_sequence = [different.next_step(index) for index in range(12)]

        self.assertEqual(first_sequence, second_sequence)
        self.assertNotEqual(first_sequence, different_sequence)
        self.assertTrue(all(step.source_name == "synthetic:random" for step in first_sequence))


class CodeBERTCacheContractTest(unittest.TestCase):
    """Verify PCA caches are bound to the accepted model and source corpus."""

    @staticmethod
    def _build_analyzer(state_path: Path, embed_dim: int = 2) -> CodeBERTAnalyzer:
        """Build an unloaded analyzer suitable for synthetic cache tests."""
        analyzer = CodeBERTAnalyzer.__new__(CodeBERTAnalyzer)
        analyzer.model_name = CODEBERT_MODEL
        analyzer.revision = CODEBERT_REVISION
        analyzer.embed_dim = embed_dim
        analyzer.pca_state_path = state_path
        analyzer.pca_components = None
        analyzer.pca_mean = None
        analyzer.pca_explained_variance_ratio = None
        analyzer.pca_corpus_fingerprint = None
        analyzer.pca_model_id = "unfitted"
        return analyzer

    @staticmethod
    def _fingerprint(value: str) -> str:
        """Return a synthetic corpus fingerprint in the production format."""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def test_valid_pca_state_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "pca.pkl"
            corpus_fingerprint = self._fingerprint("accepted-corpus")
            source = self._build_analyzer(state_path)
            source.fit_pca(
                np.asarray(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [2.0, 4.0, 6.0, 8.0],
                        [4.0, 3.0, 2.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                corpus_fingerprint=corpus_fingerprint,
            )
            source.save_pca_state()
            assert source.pca_components is not None
            assert source.pca_mean is not None
            expected_components = source.pca_components.copy()
            expected_mean = source.pca_mean.copy()

            target = self._build_analyzer(state_path)
            loaded = target.load_pca_state(expected_corpus_fingerprint=corpus_fingerprint)

        self.assertTrue(loaded)
        np.testing.assert_array_equal(target.pca_components, expected_components)
        np.testing.assert_array_equal(target.pca_mean, expected_mean)
        self.assertEqual(target.pca_model_id, source.pca_model_id)
        self.assertEqual(target.pca_corpus_fingerprint, corpus_fingerprint)

    def test_pca_state_identity_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "pca.pkl"
            corpus_fingerprint = self._fingerprint("accepted-corpus")
            source = self._build_analyzer(state_path)
            source.fit_pca(
                np.eye(3, 4, dtype=np.float32),
                corpus_fingerprint=corpus_fingerprint,
            )
            source.save_pca_state()
            stored_state = source._load_pickle(state_path)
            self.assertIsInstance(stored_state, dict)
            assert isinstance(stored_state, dict)

            mismatch_cases: dict[str, dict[str, object]] = {
                "model identity": {"model_name": "untrusted-model"},
                "model revision": {"revision": "untrusted-revision"},
                "embedding dimension": {"embed_dim": 64},
                "corpus fingerprint": {"corpus_fingerprint": self._fingerprint("different-corpus")},
            }
            for label, updates in mismatch_cases.items():
                with self.subTest(label=label):
                    state: dict[str, object] = dict(stored_state)
                    state.update(updates)
                    source._save_pickle(state_path, state)
                    target = self._build_analyzer(state_path)

                    self.assertFalse(
                        target.load_pca_state(expected_corpus_fingerprint=corpus_fingerprint)
                    )
                    self.assertIsNone(target.pca_components)
                    self.assertEqual(target.pca_model_id, "unfitted")

    def test_old_or_malformed_pca_state_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "pca.pkl"
            corpus_fingerprint = self._fingerprint("accepted-corpus")
            target = self._build_analyzer(state_path)
            old_state = {
                "model_name": CODEBERT_MODEL,
                "revision": CODEBERT_REVISION,
                "embed_dim": target.embed_dim,
                "corpus_fingerprint": corpus_fingerprint,
            }
            target._save_pickle(state_path, old_state)
            self.assertFalse(target.load_pca_state())

            malformed_state = {
                **old_state,
                "schema_version": target._PCA_STATE_VERSION,
                "model_id": "0" * 12,
                "mean": np.zeros(4, dtype=np.float32),
            }
            target._save_pickle(state_path, malformed_state)
            self.assertFalse(target.load_pca_state())

            state_path.write_bytes(b"not-a-pickle")
            self.assertFalse(target.load_pca_state())
            self.assertIsNone(target.pca_components)


class ContainerPoolContractTest(unittest.TestCase):
    """Verify warm reuse, TTL eviction, and queueing semantics."""

    def test_warm_reuse_and_ttl_expiry(self) -> None:
        pool = ContainerPool(ttl_sec=10.0, max_containers=1)

        cold = pool.simulate_batch([0.0], [100.0], [50.0])
        warm = pool.simulate_batch([1.0], [100.0], [50.0])
        expired = pool.simulate_batch([12.0], [100.0], [50.0])

        self.assertEqual(cold["cold_flags"].tolist(), [True])
        self.assertEqual(cold["latencies_ms"].tolist(), [150.0])
        self.assertEqual(warm["cold_flags"].tolist(), [False])
        self.assertEqual(warm["latencies_ms"].tolist(), [100.0])
        self.assertEqual(expired["cold_flags"].tolist(), [True])
        self.assertEqual(expired["latencies_ms"].tolist(), [150.0])

    def test_full_pool_queues_overlapping_arrivals(self) -> None:
        pool = ContainerPool(ttl_sec=10.0, max_containers=1)

        batch = pool.simulate_batch(
            arrival_times_sec=[0.0, 0.0],
            warm_service_times_ms=[100.0, 100.0],
            cold_overheads_ms=[0.0, 0.0],
        )

        self.assertEqual(batch["cold_flags"].tolist(), [True, False])
        self.assertEqual(batch["queue_flags"].tolist(), [False, True])
        self.assertAlmostEqual(float(batch["wait_times_ms"][1]), 100.0, places=4)
        self.assertAlmostEqual(float(batch["latencies_ms"][1]), 200.0, places=4)
        self.assertEqual(batch["container_count"], 1)


class CalibrationContractTest(unittest.TestCase):
    """Verify architecture routing and calibrated fallback boundaries."""

    def _build_model(self) -> CalibratedServiceModel:
        """Build a profile-backed model without loading bundled joblib files."""
        return CalibratedServiceModel(
            calibration_dirs={
                "x64": CALIBRATION_ROOT / "x64",
                "arm64": CALIBRATION_ROOT / "arm64",
            },
            lightgbm_model_dir=PROJECT_ROOT / "artifacts" / "missing-model",
            rng=np.random.default_rng(42),
        )

    def test_dual_architecture_profiles_route_independently(self) -> None:
        model = self._build_model()
        with mock.patch.object(
            model,
            "_sample_from_percentiles",
            side_effect=lambda p50_ms, p95_ms, mean_ms: mean_ms,
        ):
            x64_runtime = model.sample_warm_runtime_ms("110.dynamic-html", 128, 60, "x64")
            arm64_runtime = model.sample_warm_runtime_ms("110.dynamic-html", 128, 60, "arm64")

        self.assertAlmostEqual(x64_runtime, 39.573, places=3)
        self.assertAlmostEqual(arm64_runtime, 29.216, places=3)
        self.assertNotEqual(x64_runtime, arm64_runtime)

    def test_calibration_probe_is_action_explicit_and_read_only(self) -> None:
        config = ExperimentConfig.load(PAPER_CONFIG)
        environment = EnvironmentFactory(config).build_smoke("110.dynamic-html", "sine")
        self.addCleanup(environment.close)
        action_space = environment.action_space_wrapper
        action_id = action_space.get_action_id(512, "x64", 300)
        rng_state = deepcopy(environment.service_model.rng.bit_generator.state)
        current_step = environment.current_step
        current_time = environment.simulation_clock.now()

        probe = environment.probe_calibration(
            action_id=action_id,
            architecture="x64",
        )

        self.assertEqual(probe["action_id"], action_id)
        self.assertEqual(probe["architecture"], "x64")
        self.assertEqual(probe["memory_mb"], 512)
        self.assertEqual(probe["timeout_sec"], 300)
        self.assertEqual(environment.service_model.rng.bit_generator.state, rng_state)
        self.assertEqual(environment.current_step, current_step)
        self.assertEqual(environment.simulation_clock.now(), current_time)
        with self.assertRaisesRegex(ValueError, "uses x64, not arm64"):
            environment.probe_calibration(
                action_id=action_id,
                architecture="arm64",
            )

    def test_exact_profile_precedes_surrogate_and_gap_uses_surrogate(self) -> None:
        model = self._build_model()
        surrogate = _FakeWarmSurrogate()
        model._service_surrogates = {"x64": surrogate}

        with mock.patch.object(
            model,
            "_sample_from_percentiles",
            side_effect=lambda p50_ms, p95_ms, mean_ms: mean_ms,
        ):
            exact_runtime = model.sample_warm_runtime_ms("110.dynamic-html", 128, 60, "x64")
            self.assertEqual(surrogate.calls, [])
            surrogate_runtime = model.sample_warm_runtime_ms("110.dynamic-html", 384, 60, "x64")

        self.assertAlmostEqual(exact_runtime, 39.573, places=3)
        self.assertEqual(surrogate_runtime, 321.0)
        self.assertEqual(len(surrogate.calls), 1)
        self.assertEqual(surrogate.calls[0]["memory_mb"], 384)
        self.assertEqual(surrogate.calls[0]["architecture"], "x64")

    def test_exact_infeasible_profile_stops_environment_step(self) -> None:
        model = self._build_model()
        model._service_surrogates = {"x64": _FakeWarmSurrogate()}
        state_space = StateSpace(enable_code_features=False)
        with redirect_stdout(io.StringIO()):
            environment = ServerlessFunctionEnv(
                benchmark="411.image-recognition",
                max_steps=2,
                normalize_reward=False,
                state_space=state_space,
                action_space_config={"preset": "full_48"},
                log_init=False,
            )
        self.addCleanup(environment.close)
        environment.attach_simulation_backend(service_model=model)
        environment.set_workload_step(
            WorkloadStep(
                arrival_count=2,
                step_duration_sec=1.0,
                source_name="test",
                load_value=1.0,
                minute_of_day=0,
            )
        )
        environment.reset(seed=42)
        action = environment.action_space_wrapper.get_action_id(128, "x64", 60)

        _, _, terminated, truncated, info = environment.step(action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["metrics"]["infeasible_by_calibration"])
        self.assertEqual(info["metrics"]["infeasible_source"], "warm")
        self.assertEqual(info["metrics"]["success_rate"], 0.0)
        self.assertEqual(info["metrics"]["timeout_rate"], 1.0)


class GymContractTest(unittest.TestCase):
    """Verify reset and step follow the Gymnasium API contract."""

    def test_reset_and_step_contract(self) -> None:
        state_space = StateSpace(enable_code_features=False)
        with redirect_stdout(io.StringIO()):
            environment = ServerlessFunctionEnv(
                benchmark="110.dynamic-html",
                max_steps=1,
                normalize_reward=False,
                state_space=state_space,
                action_space_config={"preset": "full_48"},
                log_init=False,
            )
        self.addCleanup(environment.close)

        observation, reset_info = environment.reset(seed=42)
        action = environment.action_space_wrapper.get_default_action()
        result = environment.step(action)

        self.assertTrue(environment.observation_space.contains(observation))
        self.assertEqual(reset_info["step"], 0)
        self.assertIn("configuration", reset_info)
        self.assertEqual(len(result), 5)
        next_observation, reward, terminated, truncated, step_info = result
        self.assertTrue(environment.observation_space.contains(next_observation))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(step_info["action"], action)
        self.assertIn("metrics", step_info)
        self.assertEqual(environment.action_space.n, 48)


if __name__ == "__main__":
    unittest.main()
