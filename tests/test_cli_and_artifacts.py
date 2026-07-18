"""CLI, result lifecycle, aggregation, and release-artifact tests."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

from rl_optimizer.artifact_verifier import verify_artifacts
from rl_optimizer.codebert_analyzer import (
    CODEBERT_MODEL,
    CODEBERT_REVISION,
    CodeBERTAnalyzer,
    CodeBERTLoadPolicy,
    ModelUnavailableError,
)
from rl_optimizer.durable_io import (
    DurabilityUncertainError,
    DurableWriteError,
    atomic_write_text,
)
from rl_optimizer.experiment_config import ExperimentConfig
from rl_optimizer.experiment_runner import ExperimentRunner
from rl_optimizer.result_aggregation import AggregationError, ResultAggregator
from rl_optimizer.result_store import (
    FINAL_FILENAME,
    PARTIAL_FILENAME,
    PROGRESS_FILENAME,
    ResultStore,
    ResultStoreError,
    load_json,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = PROJECT_ROOT / "config" / "rl_experiments" / "paper_full48.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
CONFIG_NAMES = {
    "paper_full48.json",
    "supplementary_embedding_16.json",
    "supplementary_embedding_32.json",
    "supplementary_embedding_64.json",
    "supplementary_feature_attribution.json",
    "supplementary_load_only.json",
    "supplementary_reward_balanced.json",
    "supplementary_reward_default.json",
    "supplementary_reward_heavy.json",
    "supplementary_reward_light.json",
}


def _run_calo(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the public CLI in an isolated interpreter."""
    return subprocess.run(
        [sys.executable, "-m", "rl_optimizer.cli", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _result_payload(seed_index: int) -> dict[str, object]:
    """Build a deterministic two-algorithm, 20-case result fixture."""
    benchmarks = [
        "110.dynamic-html",
        "120.uploader",
        "210.thumbnailer",
        "311.compression",
        "411.image-recognition",
    ]
    workloads = ["sine", "spike", "decay", "random"]
    payload: dict[str, object] = {"ppo": {}, "bayes_opt_online": {}}
    for algorithm in payload:
        algorithm_results: dict[str, object] = {}
        for benchmark_index, benchmark in enumerate(benchmarks):
            workload_results = {}
            for workload_index, workload in enumerate(workloads):
                baseline = 0.2 + benchmark_index * 0.02 + workload_index * 0.003
                baseline += seed_index * 0.01
                reward = baseline + 0.1 if algorithm == "ppo" else baseline
                workload_results[workload] = {
                    "mean_reward": reward,
                    "mean_latency": 100.0 - reward,
                    "mean_cost": reward * 1e-6,
                }
            algorithm_results[benchmark] = workload_results
        payload[algorithm] = algorithm_results
    return payload


def _write_result_fixtures(root: Path) -> list[Path]:
    """Write three completed result fixtures and return their directories."""
    config = ExperimentConfig.load(PAPER_CONFIG)
    directories = []
    for seed_index, seed in enumerate((42, 52, 62)):
        directory = root / f"seed_{seed}"
        write_json(directory / FINAL_FILENAME, _result_payload(seed_index))
        write_json(
            directory / PROGRESS_FILENAME,
            {
                "status": "completed",
                "config_fingerprint": config.fingerprint,
                "calibration_identity": config.calibration_identity.to_dict(),
                "diagnostic_calibration_disabled": False,
                "results_path": FINAL_FILENAME,
                "seed": seed,
            },
        )
        directories.append(directory)
    return directories


def _write_reduced_run_config(root: Path) -> Path:
    """Write a two-case supplementary protocol for executable run tests."""
    data = json.loads(PAPER_CONFIG.read_text(encoding="utf-8"))
    data["description"] = "Reduced run fixture"
    data["protocol_role"] = "supplementary"
    data["protocol_summary"] = {
        "variant": "reduced_fixture",
        "action_preset": "full_48",
        "architecture_scope": "x64+arm64",
        "evidence_source": "executable test fixture",
    }
    data["benchmarks"] = ["110.dynamic-html"]
    data["algorithms"] = ["default"]
    data["algorithm_settings"] = {
        "default": {
            "disable_code_features": True,
            "disable_function_category": True,
        }
    }
    data["environment"]["episode_length"] = 2
    data["environment"]["load_change_freq"] = 1
    data["environment"]["load_patterns"] = ["sine", "spike"]
    data["training"]["total_timesteps"] = 1
    data["evaluation"]["n_episodes"] = 1
    data["reproducibility"]["seeds"] = [7]
    path = root / "reduced_run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class CliIsolationTest(unittest.TestCase):
    """Verify CLI parsing and configuration failures stay side-effect free."""

    def test_no_arguments_prints_help_without_model_or_ppo_imports(self) -> None:
        script = textwrap.dedent(
            """
            import contextlib
            import io
            import json
            import sys

            from rl_optimizer.cli import main

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main([])
            forbidden = [
                name
                for name in (
                    "rl_optimizer.codebert_analyzer",
                    "rl_optimizer.experiment_runner",
                    "stable_baselines3",
                    "transformers",
                )
                if name in sys.modules
            ]
            print(json.dumps({
                "status": status,
                "help": output.getvalue(),
                "forbidden": forbidden,
            }))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], 0)
        self.assertIn("usage: calo", report["help"])
        self.assertEqual(report["forbidden"], [])

    def test_missing_config_reports_clear_cli_error(self) -> None:
        completed = _run_calo(
            "smoke",
            "--config",
            "/definitely/missing/calo-config.json",
            "--steps",
            "1",
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Experiment config not found", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_missing_calibration_fails_before_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "paper_full48.json"
            data = json.loads(PAPER_CONFIG.read_text(encoding="utf-8"))
            missing_root = Path(temporary) / "missing-calibration"
            data["environment"]["calibration_dirs"] = {
                "x64": str(missing_root / "x64"),
                "arm64": str(missing_root / "arm64"),
            }
            config_path.write_text(json.dumps(data), encoding="utf-8")

            completed = _run_calo(
                "smoke",
                "--config",
                str(config_path),
                "--steps",
                "1",
            )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("Calibration for x64 is incomplete", completed.stderr)
        self.assertIn("Restore the published artifacts", completed.stderr)

    def test_offline_cache_miss_never_attempts_download(self) -> None:
        policy = CodeBERTLoadPolicy(
            model_name=CODEBERT_MODEL,
            revision=CODEBERT_REVISION,
            cache_dir=PROJECT_ROOT / ".cache" / "test-missing-codebert",
            offline=True,
        )
        analyzer = CodeBERTAnalyzer.__new__(CodeBERTAnalyzer)
        analyzer.policy = policy
        analyzer.model_name = policy.model_name
        analyzer.revision = policy.revision
        component = mock.Mock()
        component.from_pretrained.side_effect = OSError("cache miss")

        with self.assertRaisesRegex(
            ModelUnavailableError,
            "absent from the local Hugging Face cache",
        ) as raised:
            analyzer._load_pretrained_component(component, "tokenizer")

        component.from_pretrained.assert_called_once_with(
            CODEBERT_MODEL,
            revision=CODEBERT_REVISION,
            local_files_only=True,
        )
        self.assertIn("Disable `--offline-model`", str(raised.exception))


class DurableIOTest(unittest.TestCase):
    """Verify atomic-write failure boundaries exposed to experiment callers."""

    def test_failure_before_replace_preserves_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result.txt"
            target.write_text("old\n", encoding="utf-8")

            with (
                mock.patch("rl_optimizer.durable_io.os.fsync", side_effect=OSError("disk")),
                self.assertRaises(DurableWriteError) as raised,
            ):
                atomic_write_text(target, "new\n")

            self.assertFalse(raised.exception.target_installed)
            self.assertFalse(raised.exception.durability_uncertain)
            self.assertEqual(target.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_parent_sync_failure_reports_installed_target_as_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result.txt"
            target.write_text("old\n", encoding="utf-8")

            with (
                mock.patch(
                    "rl_optimizer.durable_io.os.fsync",
                    side_effect=[None, OSError("directory")],
                ),
                self.assertRaises(DurabilityUncertainError) as raised,
            ):
                atomic_write_text(target, "new\n")

            self.assertTrue(raised.exception.target_installed)
            self.assertTrue(raised.exception.durability_uncertain)
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")


class ResultLifecycleTest(unittest.TestCase):
    """Verify resumable state identity and final-file cleanup."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ExperimentConfig.load(PAPER_CONFIG)

    def test_resume_restores_cases_and_rejects_fingerprint_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            store = ResultStore(output_dir, self.config, seed=42)
            state = store.initialize(resume=False)
            benchmark, workload = store.ordered_cases()[0]
            for algorithm in self.config.algorithms:
                state.results[algorithm][benchmark][workload] = {"mean_reward": 0.5}
            store.complete_case(state, benchmark, workload)

            resumed = ResultStore(output_dir, self.config, seed=42).initialize(resume=True)
            self.assertEqual(resumed.completed_cases, {(benchmark, workload)})
            self.assertEqual(resumed.progress["config_fingerprint"], self.config.fingerprint)

            changed_data = json.loads(json.dumps(self.config.source_data))
            changed_data["training"]["total_timesteps"] += 1
            changed_config = replace(self.config, source_data=changed_data)
            with self.assertRaisesRegex(ResultStoreError, "fingerprint does not match"):
                ResultStore(output_dir, changed_config, seed=42).initialize(resume=True)

    def test_fingerprint_excludes_operational_paths_and_offline_mode(self) -> None:
        source_data = json.loads(json.dumps(self.config.source_data))
        source_data["description"] = "Different public description"
        source_data["environment"]["calibration_dirs"] = {
            "x64": "/host-a/calibration/x64",
            "arm64": "/host-a/calibration/arm64",
        }
        source_data["reproducibility"]["codebert"]["cache_dir"] = "/host-a/model-cache"
        operational = replace(
            self.config,
            source_data=source_data,
            codebert_policy=replace(self.config.codebert_policy, offline=True),
        )

        self.assertEqual(operational.fingerprint, self.config.fingerprint)
        self.assertNotIn(
            str(PROJECT_ROOT),
            json.dumps(self.config.calibration_identity.to_dict(), sort_keys=True),
        )

    def test_calibration_mode_changes_fingerprint_and_blocks_cross_resume(self) -> None:
        diagnostic = ExperimentConfig.load(PAPER_CONFIG, disable_calibration=True)
        self.assertNotEqual(self.config.fingerprint, diagnostic.fingerprint)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (source, target) in enumerate(
                ((self.config, diagnostic), (diagnostic, self.config))
            ):
                output_dir = root / f"run_{index}"
                ResultStore(output_dir, source, seed=42).initialize(resume=False)
                with self.assertRaisesRegex(ResultStoreError, "fingerprint does not match"):
                    ResultStore(output_dir, target, seed=42).initialize(resume=True)

    def test_suite_resume_detects_any_existing_run_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "seed_42" / PROGRESS_FILENAME, {})
            write_json(root / "seed_62" / FINAL_FILENAME, {})
            calls = []

            def fake_run_seed(output_dir: Path, *, seed: int, resume: bool) -> Path:
                calls.append((seed, resume))
                return Path(output_dir) / FINAL_FILENAME

            aggregate = mock.Mock(
                json_path=root / "aggregate" / "aggregate_summary.json",
                markdown_path=root / "aggregate" / "aggregate_summary.md",
            )
            runner = ExperimentRunner(self.config, emit=lambda _: None)
            with (
                mock.patch.object(runner, "run_seed", side_effect=fake_run_seed),
                mock.patch.object(ResultAggregator, "aggregate", return_value=aggregate),
            ):
                runner.run_suite(root, seeds=(42, 52, 62), resume=True)

            self.assertEqual(calls, [(42, True), (52, False), (62, True)])

    def test_resume_rejects_missing_or_contradictory_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent = root / "absent"
            with self.assertRaisesRegex(ResultStoreError, "without progress.json"):
                ResultStore(absent, self.config, seed=42).initialize(resume=True)
            self.assertFalse(absent.exists())

            progress_only = root / "progress_only"
            write_json(
                progress_only / PROGRESS_FILENAME,
                {
                    "status": "running",
                    "config_fingerprint": self.config.fingerprint,
                    "calibration_identity": self.config.calibration_identity.to_dict(),
                    "seed": 42,
                },
            )
            with self.assertRaisesRegex(ResultStoreError, "final or partial"):
                ResultStore(progress_only, self.config, seed=42).initialize(resume=True)

            completed_partial = root / "completed_partial"
            write_json(completed_partial / PARTIAL_FILENAME, {})
            write_json(
                completed_partial / PROGRESS_FILENAME,
                {
                    "status": "completed",
                    "config_fingerprint": self.config.fingerprint,
                    "calibration_identity": self.config.calibration_identity.to_dict(),
                    "seed": 42,
                },
            )
            with self.assertRaisesRegex(ResultStoreError, "requires a final"):
                ResultStore(completed_partial, self.config, seed=42).initialize(resume=True)

    def test_finalize_removes_partial_and_preserves_final_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "run"
            store = ResultStore(output_dir, self.config, seed=52)
            state = store.initialize(resume=False)
            with self.assertRaisesRegex(ResultStoreError, "incomplete cases"):
                store.finalize(state)
            for benchmark, workload in store.ordered_cases():
                for algorithm in self.config.algorithms:
                    state.results[algorithm][benchmark][workload] = {"mean_reward": 0.0}

            final_path = store.finalize(state)

            self.assertEqual(final_path.name, FINAL_FILENAME)
            self.assertTrue(final_path.is_file())
            self.assertFalse((output_dir / PARTIAL_FILENAME).exists())
            progress = load_json(output_dir / PROGRESS_FILENAME)
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(progress["config_fingerprint"], self.config.fingerprint)
            self.assertEqual(
                progress["calibration_identity"],
                self.config.calibration_identity.to_dict(),
            )

    def test_reduced_cli_run_and_partial_resume_match_clean_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = _write_reduced_run_config(root)
            config = ExperimentConfig.load(config_path)
            clean_dir = root / "clean"

            clean = _run_calo(
                "run",
                "--config",
                str(config_path),
                "--output-dir",
                str(clean_dir),
                "--seed",
                "7",
            )
            self.assertEqual(clean.returncode, 0, clean.stderr)
            clean_results = load_json(clean_dir / FINAL_FILENAME)
            self.assertFalse((clean_dir / PARTIAL_FILENAME).exists())
            self.assertEqual(load_json(clean_dir / PROGRESS_FILENAME)["status"], "completed")

            resumed_dir = root / "resumed"
            store = ResultStore(resumed_dir, config, seed=7)
            state = store.initialize(resume=False)
            first_benchmark, first_workload = store.ordered_cases()[0]
            state.results["default"][first_benchmark][first_workload] = clean_results["default"][
                first_benchmark
            ][first_workload]
            store.complete_case(state, first_benchmark, first_workload)

            resumed = _run_calo(
                "run",
                "--config",
                str(config_path),
                "--output-dir",
                str(resumed_dir),
                "--seed",
                "7",
                "--resume",
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("Skipping completed case 1/2", resumed.stdout)
            self.assertEqual(load_json(resumed_dir / FINAL_FILENAME), clean_results)
            self.assertFalse((resumed_dir / PARTIAL_FILENAME).exists())


class AggregationTest(unittest.TestCase):
    """Verify deterministic three-seed aggregation over all 20 cases."""

    def test_three_fixture_aggregate_is_stable_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = _write_result_fixtures(root)
            first = ResultAggregator().aggregate(fixtures, root / "aggregate_first")
            second = ResultAggregator().aggregate(fixtures, root / "aggregate_second")

            self.assertEqual(first.json_path.read_bytes(), second.json_path.read_bytes())
            self.assertEqual(first.markdown_path.read_bytes(), second.markdown_path.read_bytes())
            baseline = first.payload["baselines"]["bayes_opt_online"]
            self.assertEqual(len(baseline["cases"]), 20)
            self.assertEqual(baseline["overall"]["total_comparisons"], 60)
            self.assertEqual(baseline["overall"]["primary_wins"], 60)
            self.assertAlmostEqual(baseline["overall"]["mean_raw_reward_delta"], 0.1)
            markdown = first.markdown_path.read_text(encoding="utf-8")
            self.assertEqual(markdown.count("| 110.dynamic-html |"), 4)

    def test_aggregate_cli_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = _write_result_fixtures(root)
            output_dir = root / "cli_aggregate"

            completed = _run_calo(
                "aggregate",
                *(str(path) for path in fixtures),
                "--output-dir",
                str(output_dir),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Aggregate JSON:", completed.stdout)
            self.assertTrue((output_dir / "aggregate_summary.json").is_file())
            self.assertTrue((output_dir / "aggregate_summary.md").is_file())

    def test_rejects_duplicate_sources_and_incompatible_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = _write_result_fixtures(root)
            aggregator = ResultAggregator()
            with self.assertRaisesRegex(AggregationError, "sources must be unique"):
                aggregator.aggregate([fixtures[0], fixtures[0]], root / "duplicate_source")

            progress_path = fixtures[1] / PROGRESS_FILENAME
            progress = load_json(progress_path)
            progress["config_fingerprint"] = "different-fingerprint"
            write_json(progress_path, progress)
            with self.assertRaisesRegex(AggregationError, "fingerprints differ"):
                aggregator.aggregate(fixtures, root / "fingerprint_mismatch")

            progress["config_fingerprint"] = load_json(fixtures[0] / PROGRESS_FILENAME)[
                "config_fingerprint"
            ]
            accepted_identity = load_json(fixtures[0] / PROGRESS_FILENAME)["calibration_identity"]
            progress["calibration_identity"] = {
                "mode": "calibrated",
                "architectures": {"x64": "different-content"},
            }
            write_json(progress_path, progress)
            with self.assertRaisesRegex(AggregationError, "identities differ"):
                aggregator.aggregate(fixtures, root / "calibration_identity_mismatch")

            progress["calibration_identity"] = accepted_identity
            progress["diagnostic_calibration_disabled"] = True
            write_json(progress_path, progress)
            with self.assertRaisesRegex(AggregationError, "calibration modes differ"):
                aggregator.aggregate(fixtures, root / "calibration_mode_mismatch")

            progress["diagnostic_calibration_disabled"] = False
            progress["seed"] = 42
            write_json(progress_path, progress)
            with self.assertRaisesRegex(AggregationError, "seeds must be unique"):
                aggregator.aggregate(fixtures, root / "duplicate_seed")

    def test_rejects_non_finite_reward(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = _write_result_fixtures(root)
            payload = load_json(fixtures[0] / FINAL_FILENAME)
            payload["ppo"]["110.dynamic-html"]["sine"]["mean_reward"] = float("nan")
            write_json(fixtures[0] / FINAL_FILENAME, payload)

            with self.assertRaisesRegex(AggregationError, "finite mean_reward"):
                ResultAggregator().aggregate(fixtures, root / "non_finite")


class PublishedProtocolTest(unittest.TestCase):
    """Verify every published protocol and the calibrated smoke matrix."""

    def test_all_ten_configs_pass_strict_preflight(self) -> None:
        config_dir = PROJECT_ROOT / "config" / "rl_experiments"
        paths = sorted(config_dir.glob("*.json"))

        self.assertEqual({path.name for path in paths}, CONFIG_NAMES)
        self.assertEqual(len(paths), 10)
        for path in paths:
            with self.subTest(config=path.name):
                config = ExperimentConfig.load(path)
                self.assertFalse(config.diagnostic_calibration_disabled)
                self.assertIn(config.protocol_role, {"main", "supplementary"})
                environment = config.runtime_section("environment")
                calibration_dirs = environment.get("calibration_dirs")
                calibration_dir = environment.get("calibration_dir")
                self.assertNotEqual(bool(calibration_dirs), bool(calibration_dir))
                if calibration_dirs:
                    self.assertEqual(set(calibration_dirs), {"x64", "arm64"})
                    roots = calibration_dirs.values()
                else:
                    roots = [calibration_dir]
                for calibration_dir in roots:
                    self.assertTrue(Path(calibration_dir).is_dir())

    def test_main_config_runs_all_twenty_calibrated_smoke_cases(self) -> None:
        config = ExperimentConfig.load(PAPER_CONFIG)
        with redirect_stdout(io.StringIO()):
            results = ExperimentRunner(config, emit=lambda _: None).run_smoke(steps=1)

        self.assertEqual(len(results), 20)
        self.assertEqual(
            {(row["benchmark"], row["load_pattern"]) for row in results},
            {
                (benchmark, workload)
                for benchmark in config.benchmarks
                for workload in config.load_patterns
            },
        )
        for row in results:
            self.assertEqual(row["observation_dim"], 79)
            self.assertEqual(row["action_count"], 48)
            self.assertEqual(set(row["action_architectures"]), {"x64", "arm64"})
            self.assertEqual(set(row["calibration_architectures"]), {"x64", "arm64"})
            self.assertEqual(set(row["architecture_probes"]), {"x64", "arm64"})


class ArtifactVerificationTest(unittest.TestCase):
    """Verify the immutable release bundle through Python and the public CLI."""

    def test_bundled_artifacts_and_frozen_claims_pass(self) -> None:
        report = verify_artifacts(ARTIFACT_DIR)

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["artifact_version"], "1.0.0")
        self.assertEqual(report["checks"]["model_contract"]["state_dimension"], 79)
        self.assertEqual(report["checks"]["model_contract"]["action_count"], 48)
        self.assertEqual(report["checks"]["focused_replay"]["case_count"], 8)
        self.assertEqual(report["checks"]["supplementary"]["aggregate_count"], 9)

    def test_verify_cli_reports_pass_without_inference_measurement(self) -> None:
        completed = _run_calo("verify", "--artifact-dir", str(ARTIFACT_DIR))

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CALO artifact verification: PASS", completed.stdout)
        self.assertIn("model_contract: PASS", completed.stdout)
        self.assertNotIn("local_inference", completed.stdout)


if __name__ == "__main__":
    unittest.main()
