"""Pinned CodeBERT model loading and source embedding operations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any, ClassVar

import numpy as np


CODEBERT_MODEL = "microsoft/codebert-base"
CODEBERT_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"


class ModelUnavailableError(RuntimeError):
    """Indicate that the pinned CodeBERT files cannot be loaded."""


@dataclass(frozen=True)
class CodeBERTLoadPolicy:
    """Define the immutable CodeBERT identity and cache/network behavior."""

    model_name: str = CODEBERT_MODEL
    revision: str = CODEBERT_REVISION
    cache_dir: Path = Path(".cache/codebert_embeddings")
    offline: bool = False

    def validate(self) -> None:
        """Reject model identities that differ from the accepted artifact."""
        if self.model_name != CODEBERT_MODEL:
            raise ValueError(f"CodeBERT model must be {CODEBERT_MODEL}")
        if self.revision != CODEBERT_REVISION:
            raise ValueError(f"CodeBERT revision must be {CODEBERT_REVISION}")


class CodeBERTAnalyzer:
    """Extract and cache reduced embeddings from the pinned CodeBERT model."""

    _PCA_STATE_VERSION: ClassVar[int] = 1
    _SHARED_ANALYZERS: ClassVar[dict[tuple[str, str, str, int, bool], "CodeBERTAnalyzer"]] = {}

    def __init__(
        self,
        policy: CodeBERTLoadPolicy | None = None,
        embed_dim: int = 32,
    ):
        """Load the pinned model according to an explicit cache policy."""
        self.policy = policy or CodeBERTLoadPolicy()
        self.policy.validate()
        self.model_name = self.policy.model_name
        self.revision = self.policy.revision
        self.cache_dir = self.policy.cache_dir
        self.embed_dim = int(embed_dim)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        identity = hashlib.sha256(f"{self.model_name}@{self.revision}".encode("utf-8")).hexdigest()[
            :16
        ]
        self.model_cache_dir = self.cache_dir / f"model_{identity}"
        self.raw_cache_dir = self.model_cache_dir / "raw"
        self.reduced_cache_dir = self.model_cache_dir / f"reduced_{self.embed_dim}d"
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.reduced_cache_dir.mkdir(parents=True, exist_ok=True)
        self.pca_state_path = self.model_cache_dir / f"pca_{self.embed_dim}d.pkl"

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModelUnavailableError(
                "CodeBERT requires torch and transformers; install the project in its "
                "Python 3.11 environment with `python -m pip install -e .`."
            ) from exc

        print(f"[CodeBERT] Loading {self.model_name}@{self.revision}", flush=True)
        self.torch = torch
        self.tokenizer = self._load_pretrained_component(AutoTokenizer, "tokenizer")
        self.model = self._load_pretrained_component(AutoModel, "model")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.pca_components: np.ndarray | None = None
        self.pca_mean: np.ndarray | None = None
        self.pca_explained_variance_ratio: np.ndarray | None = None
        self.pca_corpus_fingerprint: str | None = None
        self.pca_model_id = "unfitted"
        if self.pca_state_path.exists():
            self.load_pca_state()

    @classmethod
    def get_shared(
        cls,
        model_name: str = CODEBERT_MODEL,
        cache_dir: str | Path = ".cache/codebert_embeddings",
        embed_dim: int = 32,
        revision: str = CODEBERT_REVISION,
        offline: bool = False,
        policy: CodeBERTLoadPolicy | None = None,
    ) -> "CodeBERTAnalyzer":
        """Return one process-local analyzer for an exact load policy."""
        resolved_policy = policy or CodeBERTLoadPolicy(
            model_name=model_name,
            revision=revision,
            cache_dir=Path(cache_dir),
            offline=offline,
        )
        resolved_policy.validate()
        key = (
            resolved_policy.model_name,
            resolved_policy.revision,
            str(resolved_policy.cache_dir.resolve()),
            int(embed_dim),
            resolved_policy.offline,
        )
        if key not in cls._SHARED_ANALYZERS:
            cls._SHARED_ANALYZERS[key] = cls(resolved_policy, embed_dim=embed_dim)
        return cls._SHARED_ANALYZERS[key]

    def _load_pretrained_component(self, auto_class: Any, component_name: str) -> Any:
        """Use the exact cached revision first, then permit one normal download."""
        arguments = {"revision": self.revision, "local_files_only": True}
        try:
            component = auto_class.from_pretrained(self.model_name, **arguments)
            print(f"[CodeBERT] Loaded {component_name} from local cache", flush=True)
            return component
        except Exception as local_error:
            if self.policy.offline:
                raise ModelUnavailableError(self._offline_error(component_name)) from local_error

        try:
            return auto_class.from_pretrained(self.model_name, revision=self.revision)
        except Exception as download_error:
            raise ModelUnavailableError(
                f"Unable to prepare pinned CodeBERT {component_name}. Connect once and run "
                "`calo run --config config/rl_experiments/paper_full48.json "
                "--output-dir outputs/model-preparation --seed 42`, then retry with "
                "`--offline-model`."
            ) from download_error

    def _offline_error(self, component_name: str) -> str:
        """Return actionable preparation guidance for an offline cache miss."""
        return (
            f"Pinned CodeBERT {component_name} is absent from the local Hugging Face cache. "
            "Disable `--offline-model` for one connected run to download "
            f"{self.model_name}@{self.revision}, then retry offline."
        )

    @staticmethod
    def _get_cache_key(code: str) -> str:
        """Return a stable content hash for source code."""
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_pickle(cache_path: Path) -> Any | None:
        """Load a pickle object when present."""
        if not cache_path.exists():
            return None
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def _save_pickle(cache_path: Path, value: Any) -> None:
        """Store one pickle object."""
        with cache_path.open("wb") as handle:
            pickle.dump(value, handle)

    def _load_raw_from_cache(self, cache_key: str) -> np.ndarray | None:
        return self._load_pickle(self.raw_cache_dir / f"{cache_key}.pkl")

    def _save_raw_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        self._save_pickle(self.raw_cache_dir / f"{cache_key}.pkl", embedding)

    def _load_reduced_from_cache(self, cache_key: str) -> np.ndarray | None:
        path = self.reduced_cache_dir / f"{self.pca_model_id}_{cache_key}.pkl"
        return self._load_pickle(path)

    def _save_reduced_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        path = self.reduced_cache_dir / f"{self.pca_model_id}_{cache_key}.pkl"
        self._save_pickle(path, embedding)

    def extract_raw_embedding(self, code: str) -> np.ndarray:
        """Extract the raw CLS embedding for a source snippet."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        device_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**device_inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze(0)

    def extract_raw_from_file(self, file_path: str, use_cache: bool = True) -> np.ndarray:
        """Extract or load a raw embedding from one source file."""
        code = Path(file_path).read_text(encoding="utf-8")
        cache_key = self._get_cache_key(code)
        cached = self._load_raw_from_cache(cache_key) if use_cache else None
        if cached is not None:
            return cached
        embedding = self.extract_raw_embedding(code)
        if use_cache:
            self._save_raw_to_cache(cache_key, embedding)
        return embedding

    @staticmethod
    def normalize_features(embedding: np.ndarray) -> np.ndarray:
        """Apply L2 normalization to a reduced embedding."""
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        return vector if norm <= 1e-12 else vector / norm

    def fit_pca(self, embeddings: np.ndarray, *, corpus_fingerprint: str) -> None:
        """Fit the deterministic NumPy-SVD projection used by CALO."""
        if not corpus_fingerprint:
            raise ValueError("PCA corpus fingerprint must be non-empty")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"PCA input must be 2D, received {matrix.shape}")
        sample_count, feature_count = matrix.shape
        component_count = min(self.embed_dim, sample_count, feature_count)
        if component_count <= 0:
            raise ValueError("PCA requires at least one sample")

        self.pca_mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = matrix - self.pca_mean
        _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
        self.pca_components = vectors[:component_count].astype(np.float32)
        if sample_count > 1:
            variance = (singular_values**2) / (sample_count - 1)
            ratios = variance[:component_count] / max(float(variance.sum()), 1e-12)
            self.pca_explained_variance_ratio = ratios.astype(np.float32)
        else:
            self.pca_explained_variance_ratio = np.ones(component_count, dtype=np.float32)
        state = self.pca_components.tobytes() + self.pca_mean.tobytes()
        self.pca_model_id = hashlib.sha256(state).hexdigest()[:12]
        self.pca_corpus_fingerprint = corpus_fingerprint

    def save_pca_state(self, output_path: str | Path | None = None) -> None:
        """Persist the fitted PCA projection."""
        if self.pca_components is None or self.pca_mean is None:
            raise ValueError("PCA is not fitted yet")
        if self.pca_corpus_fingerprint is None:
            raise ValueError("PCA corpus fingerprint is not set")
        path = Path(output_path) if output_path is not None else self.pca_state_path
        self._save_pickle(
            path,
            {
                "schema_version": self._PCA_STATE_VERSION,
                "components": self.pca_components,
                "mean": self.pca_mean,
                "explained_variance_ratio": self.pca_explained_variance_ratio,
                "model_id": self.pca_model_id,
                "model_name": self.model_name,
                "revision": self.revision,
                "embed_dim": self.embed_dim,
                "corpus_fingerprint": self.pca_corpus_fingerprint,
            },
        )

    def load_pca_state(
        self,
        input_path: str | Path | None = None,
        *,
        expected_corpus_fingerprint: str | None = None,
    ) -> bool:
        """Load a compatible fitted PCA projection without accepting stale state."""
        path = Path(input_path) if input_path is not None else self.pca_state_path
        try:
            state = self._load_pickle(path)
        except Exception:
            return False
        if not isinstance(state, dict):
            return False
        corpus_fingerprint = state.get("corpus_fingerprint")
        if (
            type(state.get("schema_version")) is not int
            or state.get("schema_version") != self._PCA_STATE_VERSION
            or state.get("model_name") != self.model_name
            or state.get("revision") != self.revision
            or type(state.get("embed_dim")) is not int
            or state.get("embed_dim") != self.embed_dim
            or not isinstance(corpus_fingerprint, str)
            or not corpus_fingerprint
        ):
            return False
        if (
            expected_corpus_fingerprint is not None
            and corpus_fingerprint != expected_corpus_fingerprint
        ):
            return False

        components_value = state.get("components")
        mean_value = state.get("mean")
        if not self._is_numeric_array(components_value) or not self._is_numeric_array(mean_value):
            return False
        components = np.asarray(components_value, dtype=np.float32)
        mean = np.asarray(mean_value, dtype=np.float32)
        if (
            components.ndim != 2
            or mean.ndim != 1
            or components.shape[0] < 1
            or components.shape[0] > self.embed_dim
            or components.shape[1] < 1
            or components.shape[1] != mean.shape[0]
            or not np.all(np.isfinite(components))
            or not np.all(np.isfinite(mean))
        ):
            return False
        model_id = hashlib.sha256(components.tobytes() + mean.tobytes()).hexdigest()[:12]
        if state.get("model_id") != model_id:
            return False

        ratios = state.get("explained_variance_ratio")
        ratio_array: np.ndarray | None = None
        if ratios is not None:
            if not self._is_numeric_array(ratios):
                return False
            ratio_array = np.asarray(ratios, dtype=np.float32)
            if (
                ratio_array.ndim != 1
                or ratio_array.shape[0] != components.shape[0]
                or not np.all(np.isfinite(ratio_array))
                or np.any(ratio_array < 0.0)
            ):
                return False

        self.pca_components = components
        self.pca_mean = mean
        self.pca_explained_variance_ratio = ratio_array
        self.pca_model_id = model_id
        self.pca_corpus_fingerprint = corpus_fingerprint
        return True

    @staticmethod
    def _is_numeric_array(value: Any) -> bool:
        """Return whether a cache value is a plain numeric NumPy array."""
        return isinstance(value, np.ndarray) and value.dtype.kind in "fiu"

    def project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Project one raw embedding to the configured dimension."""
        vector = np.asarray(embedding, dtype=np.float32)
        if self.pca_components is None or self.pca_mean is None:
            return self.normalize_features(vector[: self.embed_dim])
        reduced = (vector - self.pca_mean) @ self.pca_components.T
        if reduced.shape[0] < self.embed_dim:
            reduced = np.pad(reduced, (0, self.embed_dim - reduced.shape[0]))
        return self.normalize_features(reduced.astype(np.float32))

    def extract_features(self, code: str, use_cache: bool = True) -> np.ndarray:
        """Extract a reduced embedding for source code."""
        cache_key = self._get_cache_key(code)
        cached = self._load_reduced_from_cache(cache_key) if use_cache else None
        if cached is not None:
            return cached
        raw = self._load_raw_from_cache(cache_key) if use_cache else None
        if raw is None:
            raw = self.extract_raw_embedding(code)
            if use_cache:
                self._save_raw_to_cache(cache_key, raw)
        reduced = self.project_embedding(raw)
        if use_cache:
            self._save_reduced_to_cache(cache_key, reduced)
        return reduced

    def extract_from_file(self, file_path: str, use_cache: bool = True) -> np.ndarray:
        """Extract a reduced embedding from a source file."""
        code = Path(file_path).read_text(encoding="utf-8")
        return self.extract_features(code, use_cache=use_cache)
