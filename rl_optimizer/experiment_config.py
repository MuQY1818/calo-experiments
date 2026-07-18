"""Validated experiment configuration for the CALO command-line interface."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .action_space import ActionSpace
from .codebert_analyzer import (
    CODEBERT_MODEL,
    CODEBERT_REVISION,
    CodeBERTLoadPolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = Path("config/rl_experiments/paper_full48.json")
CALIBRATION_FILES = (
    "warm_profile.csv",
    "cold_profile.csv",
    "burst_profile.csv",
    "lightgbm_service/metadata.json",
    "lightgbm_service/warm_mean_model.joblib",
    "lightgbm_service/warm_p95_model.joblib",
    "lightgbm_service/cold_overhead_mean_model.joblib",
    "lightgbm_service/cold_overhead_p95_model.joblib",
    "lightgbm_service/burst_slowdown_mean_model.joblib",
    "lightgbm_service/burst_slowdown_p95_model.joblib",
)
REQUIRED_TOP_LEVEL = (
    "description",
    "protocol_role",
    "protocol_summary",
    "benchmarks",
    "algorithms",
    "environment",
    "training",
    "evaluation",
    "baseline",
    "reproducibility",
)
SUPPORTED_ALGORITHMS = {
    "ppo",
    "ppo_flat",
    "ppo_load_only",
    "bayes_opt_online",
    "default",
    "random",
    "greedy",
}
NON_SEMANTIC_CONFIG_KEYS = {
    "cache_first",
    "description",
    "evidence_source",
    "offline",
}


class ConfigurationError(RuntimeError):
    """Indicate an invalid or incomplete experiment configuration."""


@dataclass(frozen=True)
class CalibrationIdentity:
    """Path-independent identity of the calibration used by one protocol."""

    mode: str
    architecture_digests: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation stored with run results."""
        return {
            "mode": self.mode,
            "architectures": dict(self.architecture_digests),
        }


def _scientific_projection(value: Any) -> Any:
    """Remove host-local and descriptive fields from protocol identity."""
    if isinstance(value, Mapping):
        projected = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key in NON_SEMANTIC_CONFIG_KEYS:
                continue
            if key.endswith(("_cache", "_dir", "_dirs", "_path")):
                continue
            projected[key] = _scientific_projection(item)
        return projected
    if isinstance(value, list):
        return [_scientific_projection(item) for item in value]
    return value


def _calibration_digest(root: Path) -> str:
    """Hash every calibration input by relative name and content."""
    digest = hashlib.sha256()
    for relative_name in CALIBRATION_FILES:
        path = root / relative_name
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a user path relative to the repository root.

    Args:
        path: Absolute path or repository-root-relative path.

    Returns:
        Normalized absolute path.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def display_project_path(path: Path) -> str:
    """Return a stable project-relative label when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _mapping(value: Any, field: str) -> dict[str, Any]:
    """Return a plain mapping or raise a configuration error."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"Configuration field '{field}' must be an object.")
    return {str(key): item for key, item in value.items()}


def _string_list(value: Any, field: str) -> list[str]:
    """Return a non-empty list of strings."""
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"Configuration field '{field}' must be a non-empty list.")
    normalized = [str(item) for item in value]
    if any(not item for item in normalized):
        raise ConfigurationError(f"Configuration field '{field}' contains an empty value.")
    return normalized


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable validated view of one CALO experiment protocol."""

    source_path: Path
    source_data: dict[str, Any]
    data: dict[str, Any]
    calibration_identity: CalibrationIdentity
    codebert_policy: CodeBERTLoadPolicy
    diagnostic_calibration_disabled: bool = False

    @classmethod
    def load(
        cls,
        path: str | Path = PAPER_CONFIG,
        *,
        override_calibration_dir: str | Path | None = None,
        disable_calibration: bool = False,
        offline_model: bool = False,
    ) -> "ExperimentConfig":
        """Load, resolve, and validate one JSON experiment config.

        Args:
            path: Config path, resolved from the project root when relative.
            override_calibration_dir: Optional single-architecture diagnostic override.
            disable_calibration: Explicitly use the heuristic service model.
            offline_model: Require the pinned CodeBERT revision in local cache.

        Returns:
            Validated experiment configuration.

        Raises:
            ConfigurationError: If the JSON, schema, or required files are invalid.
        """
        source_path = resolve_project_path(path)
        if not source_path.is_file():
            raise ConfigurationError(
                f"Experiment config not found: {display_project_path(source_path)}"
            )
        try:
            parsed = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Cannot read experiment config {source_path}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigurationError("The experiment config root must be a JSON object.")

        source_data = deepcopy(parsed)
        runtime_data = deepcopy(parsed)
        cls._validate_schema(runtime_data, source_path)
        cls._resolve_calibration(
            runtime_data,
            override_calibration_dir=override_calibration_dir,
            disable_calibration=disable_calibration,
        )
        calibration_identity = cls._build_calibration_identity(
            runtime_data,
            disabled=disable_calibration,
        )
        codebert_policy = cls._resolve_codebert(runtime_data, offline_model=offline_model)
        return cls(
            source_path=source_path,
            source_data=source_data,
            data=runtime_data,
            calibration_identity=calibration_identity,
            codebert_policy=codebert_policy,
            diagnostic_calibration_disabled=disable_calibration,
        )

    @staticmethod
    def _validate_schema(data: dict[str, Any], source_path: Path) -> None:
        """Validate protocol metadata and runtime fields."""
        missing = [field for field in REQUIRED_TOP_LEVEL if field not in data]
        if missing:
            raise ConfigurationError(f"Missing required config fields: {missing}")

        role = str(data["protocol_role"])
        if role not in {"main", "supplementary"}:
            raise ConfigurationError("protocol_role must be 'main' or 'supplementary'.")
        if role == "main" and source_path.name != PAPER_CONFIG.name:
            raise ConfigurationError("Only paper_full48.json may declare protocol_role='main'.")

        summary = _mapping(data["protocol_summary"], "protocol_summary")
        for field in ("variant", "action_preset", "architecture_scope", "evidence_source"):
            if not summary.get(field):
                raise ConfigurationError(f"protocol_summary.{field} is required.")

        benchmarks = _string_list(data["benchmarks"], "benchmarks")
        algorithms = _string_list(data["algorithms"], "algorithms")
        unsupported = sorted(set(algorithms) - SUPPORTED_ALGORITHMS)
        if unsupported:
            raise ConfigurationError(f"Unsupported algorithms: {unsupported}")

        environment = _mapping(data["environment"], "environment")
        load_patterns = _string_list(environment.get("load_patterns"), "environment.load_patterns")
        action_config = _mapping(environment.get("action_space"), "environment.action_space")
        preset = str(action_config.get("preset", ""))
        if preset not in ActionSpace.PRESETS:
            raise ConfigurationError(
                f"Unknown action-space preset '{preset}'. Available: {sorted(ActionSpace.PRESETS)}"
            )
        if summary["action_preset"] != preset:
            raise ConfigurationError(
                "protocol_summary.action_preset must match environment.action_space.preset."
            )

        training = _mapping(data["training"], "training")
        if int(training.get("total_timesteps", 0)) <= 0:
            raise ConfigurationError("training.total_timesteps must be positive.")
        _mapping(training.get("ppo_kwargs"), "training.ppo_kwargs")
        evaluation = _mapping(data["evaluation"], "evaluation")
        if int(evaluation.get("n_episodes", 0)) <= 0:
            raise ConfigurationError("evaluation.n_episodes must be positive.")
        seeds = _mapping(data["reproducibility"], "reproducibility").get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ConfigurationError("reproducibility.seeds must be a non-empty list.")
        try:
            normalized_seeds = [int(seed) for seed in seeds]
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("reproducibility.seeds must contain integers.") from exc
        data["reproducibility"]["seeds"] = normalized_seeds

        if environment.get("workload_source") == "azure_trace" or any(
            key.startswith("azure_") for key in environment
        ):
            raise ConfigurationError(
                "Published configs cannot depend on the unbundled Azure trace data."
            )
        if role == "main":
            ExperimentConfig._validate_main_protocol(
                data,
                benchmarks=benchmarks,
                load_patterns=load_patterns,
                preset=preset,
            )

    @staticmethod
    def _validate_main_protocol(
        data: dict[str, Any],
        *,
        benchmarks: list[str],
        load_patterns: list[str],
        preset: str,
    ) -> None:
        """Enforce the accepted-paper protocol as one canonical config."""
        expected_benchmarks = [
            "110.dynamic-html",
            "120.uploader",
            "210.thumbnailer",
            "311.compression",
            "411.image-recognition",
        ]
        if benchmarks != expected_benchmarks:
            raise ConfigurationError("The main protocol must contain the accepted five benchmarks.")
        if load_patterns != ["sine", "spike", "decay", "random"]:
            raise ConfigurationError("The main protocol must contain the four accepted workloads.")
        if preset != "full_48":
            raise ConfigurationError("The main protocol requires the full_48 action preset.")
        if int(data["environment"].get("code_feature_dim", 32)) != 32:
            raise ConfigurationError("The main protocol requires a 32-D code embedding.")
        if int(data["training"].get("total_timesteps", 0)) != 32768:
            raise ConfigurationError("The main protocol requires 32768 PPO timesteps.")
        if not data["training"].get("imitation_warmstart", {}).get("enabled"):
            raise ConfigurationError("The main protocol requires bootstrap shortlist warm-start.")
        if not data["training"].get("checkpoint_selection", {}).get("enabled"):
            raise ConfigurationError("The main protocol requires checkpoint selection.")
        if data["reproducibility"]["seeds"] != [42, 52, 62]:
            raise ConfigurationError("The main protocol seeds must be 42, 52, and 62.")

    @staticmethod
    def _resolve_calibration(
        data: dict[str, Any],
        *,
        override_calibration_dir: str | Path | None,
        disable_calibration: bool,
    ) -> None:
        """Resolve calibration paths and enforce their public-file contract."""
        environment = data["environment"]
        if disable_calibration:
            environment["calibration_dir"] = None
            environment["calibration_dirs"] = None
            return

        if override_calibration_dir is not None:
            calibration_root = resolve_project_path(override_calibration_dir)
            ExperimentConfig._validate_calibration_root(calibration_root, "override")
            environment["calibration_dir"] = str(calibration_root)
            environment["calibration_dirs"] = None
            return

        calibration_dirs = environment.get("calibration_dirs")
        calibration_dir = environment.get("calibration_dir")
        if calibration_dirs:
            mapping = _mapping(calibration_dirs, "environment.calibration_dirs")
            if data["protocol_role"] == "main" and set(mapping) != {"x64", "arm64"}:
                raise ConfigurationError(
                    "The main protocol requires x64 and arm64 calibration dirs."
                )
            resolved: dict[str, str] = {}
            for architecture, path in mapping.items():
                root = resolve_project_path(str(path))
                ExperimentConfig._validate_calibration_root(root, architecture)
                resolved[architecture] = str(root)
            environment["calibration_dirs"] = resolved
            environment["calibration_dir"] = None
            return
        if calibration_dir:
            root = resolve_project_path(str(calibration_dir))
            ExperimentConfig._validate_calibration_root(root, "single")
            environment["calibration_dir"] = str(root)
            environment["calibration_dirs"] = None
            return
        raise ConfigurationError(
            "Calibration is required. Restore the bundled artifacts or use "
            "--disable-calibration only for an explicitly labeled diagnostic run."
        )

    @staticmethod
    def _validate_calibration_root(root: Path, architecture: str) -> None:
        """Require every profile and surrogate used by the service model."""
        missing = [name for name in CALIBRATION_FILES if not (root / name).is_file()]
        if missing:
            raise ConfigurationError(
                f"Calibration for {architecture} is incomplete at {display_project_path(root)}; "
                f"missing: {missing}. Restore the published artifacts before running."
            )

    @staticmethod
    def _build_calibration_identity(
        data: dict[str, Any],
        *,
        disabled: bool,
    ) -> CalibrationIdentity:
        """Build a scientific identity from calibration contents, never paths."""
        if disabled:
            return CalibrationIdentity(mode="heuristic", architecture_digests=())
        environment = data["environment"]
        calibration_dirs = environment.get("calibration_dirs")
        if isinstance(calibration_dirs, Mapping) and calibration_dirs:
            roots = {str(key): Path(value) for key, value in calibration_dirs.items()}
        else:
            scope = str(data["protocol_summary"].get("architecture_scope", "single"))
            architecture = scope if "+" not in scope else "single"
            roots = {architecture: Path(environment["calibration_dir"])}
        return CalibrationIdentity(
            mode="calibrated",
            architecture_digests=tuple(
                sorted(
                    (architecture, _calibration_digest(root))
                    for architecture, root in roots.items()
                )
            ),
        )

    @staticmethod
    def _resolve_codebert(data: dict[str, Any], *, offline_model: bool) -> CodeBERTLoadPolicy:
        """Build the immutable CodeBERT load policy from reproducibility metadata."""
        reproducibility = data["reproducibility"]
        raw_policy = _mapping(reproducibility.get("codebert"), "reproducibility.codebert")
        model_name = str(raw_policy.get("model_name", ""))
        revision = str(raw_policy.get("revision", ""))
        if model_name != CODEBERT_MODEL or revision != CODEBERT_REVISION:
            raise ConfigurationError(
                f"CodeBERT must be pinned to {CODEBERT_MODEL}@{CODEBERT_REVISION}."
            )
        cache_dir = resolve_project_path(
            str(raw_policy.get("cache_dir", ".cache/codebert_embeddings"))
        )
        policy = CodeBERTLoadPolicy(
            model_name=model_name,
            revision=revision,
            cache_dir=cache_dir,
            offline=bool(offline_model),
        )
        policy.validate()
        reproducibility["codebert"] = {
            **raw_policy,
            "model_name": model_name,
            "revision": revision,
            "cache_dir": str(cache_dir),
            "offline": bool(offline_model),
        }
        return policy

    @property
    def description(self) -> str:
        """Return the user-facing protocol description."""
        return str(self.data["description"])

    @property
    def protocol_role(self) -> str:
        """Return main or supplementary."""
        return str(self.data["protocol_role"])

    @property
    def benchmarks(self) -> tuple[str, ...]:
        """Return benchmark order."""
        return tuple(str(item) for item in self.data["benchmarks"])

    @property
    def algorithms(self) -> tuple[str, ...]:
        """Return algorithm order."""
        return tuple(str(item) for item in self.data["algorithms"])

    @property
    def load_patterns(self) -> tuple[str, ...]:
        """Return workload-family order."""
        return tuple(str(item) for item in self.data["environment"]["load_patterns"])

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return configured random seeds."""
        return tuple(int(item) for item in self.data["reproducibility"]["seeds"])

    @property
    def fingerprint(self) -> str:
        """Return a path-independent digest of scientific run semantics."""
        identity = {
            "protocol": _scientific_projection(self.source_data),
            "calibration": self.calibration_identity.to_dict(),
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def runtime_section(self, name: str) -> dict[str, Any]:
        """Return a defensive copy of one runtime section."""
        return deepcopy(_mapping(self.data[name], name))

    def default_seed(self) -> int:
        """Return the first configured seed."""
        return self.seeds[0]
