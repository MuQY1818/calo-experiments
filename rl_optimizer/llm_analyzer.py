"""CodeBERT-based source-code feature extraction for CALO."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import pickle
from typing import Dict, Optional

import numpy as np


class CodeBERTAnalyzer:
    """Extract and cache reduced CodeBERT embeddings for function source code."""

    _SHARED_ANALYZERS: Dict[tuple[str, str, int], "CodeBERTAnalyzer"] = {}

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        cache_dir: str = ".cache/codebert_embeddings",
        embed_dim: int = 32,
    ):
        """Initialize the analyzer and load the Hugging Face model."""
        print(f"[CodeBERT] Loading model: {model_name}", flush=True)

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_dir = self.cache_dir / "raw"
        self.reduced_cache_dir = self.cache_dir / f"reduced_{embed_dim}d"
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.reduced_cache_dir.mkdir(parents=True, exist_ok=True)
        self.pca_state_path = self.cache_dir / f"pca_{embed_dim}d.pkl"
        self.embed_dim = int(embed_dim)

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CodeBERTAnalyzer requires torch and transformers. "
                "Install the project requirements in an isolated environment."
            ) from exc

        self.torch = torch
        self.tokenizer = self._load_pretrained_component(
            AutoTokenizer,
            model_name,
            component_name="tokenizer",
        )
        self.model = self._load_pretrained_component(
            AutoModel,
            model_name,
            component_name="model",
        )

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = self.torch.device(
            "cuda" if self.torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(f"[CodeBERT] Model ready on device: {self.device}", flush=True)

        self.pca_components = None
        self.pca_mean = None
        self.pca_explained_variance_ratio = None
        self.pca_model_id = "unfitted"
        if self.pca_state_path.exists():
            self.load_pca_state(self.pca_state_path)

    @classmethod
    def get_shared(
        cls,
        model_name: str = "microsoft/codebert-base",
        cache_dir: str = ".cache/codebert_embeddings",
        embed_dim: int = 32,
    ) -> "CodeBERTAnalyzer":
        """Return one process-local analyzer for a model/cache/dimension tuple."""
        key = (str(model_name), str(cache_dir), int(embed_dim))
        analyzer = cls._SHARED_ANALYZERS.get(key)
        if analyzer is None:
            analyzer = cls(
                model_name=model_name,
                cache_dir=cache_dir,
                embed_dim=embed_dim,
            )
            cls._SHARED_ANALYZERS[key] = analyzer
        return analyzer

    def _load_pretrained_component(
        self,
        auto_class,
        model_name: str,
        component_name: str,
    ):
        """Load one Hugging Face component with a cache-first strategy."""
        try:
            with self._offline_hf_hub():
                component = auto_class.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
            print(
                f"[CodeBERT] Loaded {component_name} from local cache",
                flush=True,
            )
            return component
        except Exception as local_error:
            print(
                f"[CodeBERT] Local cache load failed for {component_name}; "
                "falling back to the default Hugging Face resolver",
                flush=True,
            )
            try:
                return auto_class.from_pretrained(model_name)
            except Exception:
                raise local_error

    @contextmanager
    def _offline_hf_hub(self):
        """Temporarily force offline Hugging Face resolution."""
        previous_values = {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
        }
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            yield
        finally:
            for key, value in previous_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _get_cache_key(self, code: str) -> str:
        """Return a stable content hash for source code."""
        return hashlib.md5(code.encode("utf-8")).hexdigest()

    def _load_pickle(self, cache_path: Path):
        """Load a pickle object if it exists."""
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_pickle(self, cache_path: Path, obj) -> None:
        """Store one pickle object."""
        with open(cache_path, "wb") as f:
            pickle.dump(obj, f)

    def _load_raw_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load a raw 768-dimensional embedding from cache."""
        return self._load_pickle(self.raw_cache_dir / f"{cache_key}.pkl")

    def _save_raw_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save a raw 768-dimensional embedding to cache."""
        self._save_pickle(self.raw_cache_dir / f"{cache_key}.pkl", embedding)

    def _load_reduced_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load a reduced embedding from cache."""
        return self._load_pickle(
            self.reduced_cache_dir / f"{self.pca_model_id}_{cache_key}.pkl"
        )

    def _save_reduced_to_cache(self, cache_key: str, embedding: np.ndarray) -> None:
        """Save a reduced embedding to cache."""
        self._save_pickle(
            self.reduced_cache_dir / f"{self.pca_model_id}_{cache_key}.pkl",
            embedding,
        )

    def extract_raw_embedding(self, code: str) -> np.ndarray:
        """Extract the raw CodeBERT CLS embedding for a source snippet."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding.squeeze(0)

    def extract_raw_from_file(
        self,
        file_path: str,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Extract a raw CodeBERT embedding from a source file."""
        code = Path(file_path).read_text(encoding="utf-8")
        cache_key = self._get_cache_key(code)

        if use_cache:
            cached = self._load_raw_from_cache(cache_key)
            if cached is not None:
                return cached

        embedding = self.extract_raw_embedding(code)
        if use_cache:
            self._save_raw_to_cache(cache_key, embedding)
        return embedding

    def normalize_features(self, embedding: np.ndarray) -> np.ndarray:
        """Apply L2 normalization to a reduced embedding."""
        embedding = np.asarray(embedding, dtype=np.float32)
        l2_norm = float(np.linalg.norm(embedding))
        if l2_norm <= 1e-12:
            return embedding
        return embedding / l2_norm

    def fit_pca(self, embeddings: np.ndarray) -> None:
        """Fit a deterministic PCA projection with NumPy SVD."""
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"PCA input must be a 2D matrix, received shape {matrix.shape}"
            )

        n_samples, n_features = matrix.shape
        max_components = min(n_samples, n_features)
        n_components = min(self.embed_dim, max_components)
        if n_components <= 0:
            raise ValueError("PCA requires at least one sample")

        self.pca_mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = matrix - self.pca_mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        self.pca_components = vt[:n_components].astype(np.float32)

        if n_samples > 1:
            explained_variance = (singular_values ** 2) / (n_samples - 1)
            total_variance = float(explained_variance.sum())
            ratios = explained_variance[:n_components] / max(total_variance, 1e-12)
            self.pca_explained_variance_ratio = ratios.astype(np.float32)
        else:
            self.pca_explained_variance_ratio = np.ones(
                n_components,
                dtype=np.float32,
            )

        state_bytes = self.pca_components.tobytes() + self.pca_mean.tobytes()
        self.pca_model_id = hashlib.md5(state_bytes).hexdigest()[:12]

    def save_pca_state(self, output_path: str | Path | None = None) -> None:
        """Save the fitted PCA projection."""
        if self.pca_components is None or self.pca_mean is None:
            raise ValueError("PCA is not fitted yet")
        path = Path(output_path) if output_path is not None else self.pca_state_path
        self._save_pickle(
            path,
            {
                "components": self.pca_components,
                "mean": self.pca_mean,
                "explained_variance_ratio": self.pca_explained_variance_ratio,
                "model_id": self.pca_model_id,
                "embed_dim": self.embed_dim,
            },
        )

    def load_pca_state(self, input_path: str | Path | None = None) -> bool:
        """Load a fitted PCA projection when available."""
        path = Path(input_path) if input_path is not None else self.pca_state_path
        state = self._load_pickle(path)
        if state is None:
            return False

        self.pca_components = np.asarray(state["components"], dtype=np.float32)
        self.pca_mean = np.asarray(state["mean"], dtype=np.float32)
        variance_ratio = state.get("explained_variance_ratio")
        self.pca_explained_variance_ratio = (
            None
            if variance_ratio is None
            else np.asarray(variance_ratio, dtype=np.float32)
        )
        self.pca_model_id = str(state.get("model_id", "loaded_pca"))
        return True

    def project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Project a raw embedding to the configured reduced dimension."""
        vector = np.asarray(embedding, dtype=np.float32)
        if self.pca_components is None or self.pca_mean is None:
            if self.embed_dim >= vector.shape[0]:
                return self.normalize_features(vector)
            return self.normalize_features(vector[:self.embed_dim])

        reduced = (vector - self.pca_mean) @ self.pca_components.T
        if reduced.shape[0] < self.embed_dim:
            padding = np.zeros(self.embed_dim - reduced.shape[0], dtype=np.float32)
            reduced = np.concatenate([reduced.astype(np.float32), padding], axis=0)
        return self.normalize_features(reduced.astype(np.float32))

    def extract_features(self, code: str, use_cache: bool = True) -> np.ndarray:
        """Extract a reduced embedding for source code."""
        cache_key = self._get_cache_key(code)

        if use_cache:
            cached = self._load_reduced_from_cache(cache_key)
            if cached is not None:
                return cached

        raw_embedding = self._load_raw_from_cache(cache_key) if use_cache else None
        if raw_embedding is None:
            raw_embedding = self.extract_raw_embedding(code)
            if use_cache:
                self._save_raw_to_cache(cache_key, raw_embedding)

        reduced_embedding = self.project_embedding(raw_embedding)
        if use_cache:
            self._save_reduced_to_cache(cache_key, reduced_embedding)
        return reduced_embedding

    def extract_from_file(
        self,
        file_path: str,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Extract a reduced embedding from a source file."""
        code = Path(file_path).read_text(encoding="utf-8")
        return self.extract_features(code, use_cache)


class BenchmarkFeatureExtractor:
    """Extract CodeBERT features for SeBS-style benchmark directories."""

    DEFAULT_BENCHMARKS = [
        "110.dynamic-html",
        "120.uploader",
        "130.crud-api",
        "210.thumbnailer",
        "220.video-processing",
        "311.compression",
        "411.image-recognition",
        "501.graph-pagerank",
        "502.graph-mst",
        "503.graph-bfs",
        "504.dna-visualisation",
    ]

    def __init__(
        self,
        sebs_root: str = ".",
        analyzer: Optional[CodeBERTAnalyzer] = None,
    ):
        """Initialize a benchmark-level feature extractor."""
        self.sebs_root = Path(sebs_root)
        self.benchmarks_dir = self.sebs_root / "benchmarks"
        self.analyzer = analyzer or CodeBERTAnalyzer.get_shared()
        self.benchmark_list = list(self.DEFAULT_BENCHMARKS)
        self.pca_state_path = self.analyzer.pca_state_path

    def _find_main_function(self, benchmark_path: Path) -> Optional[Path]:
        """Return the main Python function file for one benchmark."""
        possible_files = [
            "handler.py",
            "function.py",
            "index.py",
            "__init__.py",
        ]

        python_dir = benchmark_path / "python"
        if python_dir.exists():
            for filename in possible_files:
                path = python_dir / filename
                if path.exists():
                    return path

        for filename in possible_files:
            path = benchmark_path / filename
            if path.exists():
                return path
        return None

    def extract_single_benchmark(self, benchmark_name: str) -> Optional[np.ndarray]:
        """Extract reduced source-code features for one benchmark."""
        benchmark_path = self._get_benchmark_path(benchmark_name)
        if benchmark_path is None:
            return None

        main_file = self._find_main_function(benchmark_path)
        if main_file is None:
            print(
                f"[Error] Main function file not found: {benchmark_name}",
                flush=True,
            )
            return None

        print(f"[Extract] {benchmark_name}: {main_file}", flush=True)
        try:
            if self._supports_pca_pipeline():
                self._ensure_pca_ready()
                raw_embedding = self.analyzer.extract_raw_from_file(str(main_file))
                features = self.analyzer.project_embedding(raw_embedding)
                code = main_file.read_text(encoding="utf-8")
                cache_key = self.analyzer._get_cache_key(code)
                self.analyzer._save_reduced_to_cache(cache_key, features)
            else:
                features = self.analyzer.extract_from_file(str(main_file))
            print(
                f"[Success] {benchmark_name}: feature shape {features.shape}",
                flush=True,
            )
            return features
        except Exception as exc:
            print(f"[Error] {benchmark_name}: {exc}", flush=True)
            return None

    def extract_all_benchmarks(self, use_pca: bool = True) -> Dict[str, np.ndarray]:
        """Extract features for all available benchmarks in ``benchmark_list``."""
        print("=" * 60, flush=True)
        print("Extracting benchmark CodeBERT features", flush=True)
        print("=" * 60, flush=True)

        raw_embeddings = self._collect_raw_embeddings()
        if not raw_embeddings:
            print("[Error] No benchmark features were extracted", flush=True)
            return {}

        names = list(raw_embeddings.keys())
        raw_matrix = np.array([raw_embeddings[name] for name in names])
        print(f"Raw benchmark matrix: {raw_matrix.shape}", flush=True)

        if use_pca and self._supports_pca_pipeline():
            pca_corpus = self._collect_pca_corpus_embeddings()
            pca_matrix = np.array(pca_corpus)
            print(
                f"[PCA] Fitting projection from {pca_matrix.shape[0]} samples "
                f"to {self.analyzer.embed_dim} dimensions",
                flush=True,
            )
            self.analyzer.fit_pca(pca_matrix)
            self.analyzer.save_pca_state(self.pca_state_path)
            explained_var = (
                float(self.analyzer.pca_explained_variance_ratio.sum())
                if self.analyzer.pca_explained_variance_ratio is not None
                else 0.0
            )
            print(f"[PCA] Cumulative explained variance: {explained_var:.4f}")
            reduced_matrix = np.vstack(
                [self.analyzer.project_embedding(raw_embeddings[name]) for name in names]
            )
        else:
            print("[PCA] Falling back to truncation plus L2 normalization", flush=True)
            reduced_matrix = np.vstack(
                [
                    self._normalize_vector(raw_embeddings[name][:self.analyzer.embed_dim])
                    for name in names
                ]
            )

        features = {name: reduced_matrix[index] for index, name in enumerate(names)}
        self._cache_reduced_features(features)
        print(
            f"[Done] Extracted {len(features)}/{len(self.benchmark_list)} benchmarks",
            flush=True,
        )
        return features

    def _get_benchmark_path(self, benchmark_name: str) -> Optional[Path]:
        """Resolve the directory for a SeBS benchmark name."""
        category = benchmark_name.split(".")[0]
        if category.startswith("0"):
            category_dir = "000.microbenchmarks"
        elif category.startswith("1"):
            category_dir = "100.webapps"
        elif category.startswith("2"):
            category_dir = "200.multimedia"
        elif category.startswith("3"):
            category_dir = "300.utilities"
        elif category.startswith("4"):
            category_dir = "400.inference"
        elif category.startswith("5"):
            category_dir = "500.scientific"
        else:
            print(
                f"[Error] Unrecognized benchmark category: {benchmark_name}",
                flush=True,
            )
            return None

        benchmark_path = self.benchmarks_dir / category_dir / benchmark_name
        if not benchmark_path.exists():
            print(
                f"[Error] Benchmark path does not exist: {benchmark_path}",
                flush=True,
            )
            return None
        return benchmark_path

    def _supports_pca_pipeline(self) -> bool:
        """Return whether the analyzer exposes raw-embedding PCA utilities."""
        required_methods = [
            "extract_raw_from_file",
            "project_embedding",
            "fit_pca",
            "save_pca_state",
            "load_pca_state",
            "normalize_features",
        ]
        return all(hasattr(self.analyzer, method) for method in required_methods)

    def _collect_raw_embeddings(self) -> Dict[str, np.ndarray]:
        """Collect raw CodeBERT embeddings for available benchmarks."""
        print("[Stage 1] Extracting raw CodeBERT embeddings", flush=True)
        raw_embeddings = {}

        for benchmark_name in self.benchmark_list:
            benchmark_path = self._get_benchmark_path(benchmark_name)
            if benchmark_path is None:
                continue

            main_file = self._find_main_function(benchmark_path)
            if main_file is None:
                print(
                    f"[Error] Main function file not found: {benchmark_name}",
                    flush=True,
                )
                continue

            print(f"[Extract] {benchmark_name}: {main_file}", flush=True)
            try:
                if self._supports_pca_pipeline():
                    raw_embedding = self.analyzer.extract_raw_from_file(str(main_file))
                else:
                    raw_embedding = self.analyzer.extract_from_file(str(main_file))
                raw_embeddings[benchmark_name] = raw_embedding
                print(
                    f"[Success] {benchmark_name}: raw shape {raw_embedding.shape}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[Error] {benchmark_name}: {exc}", flush=True)

        return raw_embeddings

    def _collect_pca_corpus_embeddings(self) -> list[np.ndarray]:
        """Collect raw embeddings over benchmark Python files for PCA fitting."""
        print("[PCA] Collecting source-code corpus", flush=True)
        corpus_embeddings = []
        seen_cache_keys = set()

        for source_file in self._iter_pca_corpus_files():
            try:
                code = source_file.read_text(encoding="utf-8")
                for snippet in self._build_pca_snippets(code):
                    cache_key = self.analyzer._get_cache_key(snippet)
                    if cache_key in seen_cache_keys:
                        continue
                    embedding = self._extract_raw_embedding_from_code(snippet)
                    corpus_embeddings.append(embedding)
                    seen_cache_keys.add(cache_key)
            except Exception as exc:
                print(
                    f"[Warning] Skipping PCA corpus file {source_file}: {exc}",
                    flush=True,
                )

        if not corpus_embeddings:
            raise ValueError("No PCA corpus embeddings could be collected")
        print(f"[PCA] Corpus samples: {len(corpus_embeddings)}", flush=True)
        return corpus_embeddings

    def _iter_pca_corpus_files(self) -> list[Path]:
        """Return Python source files used to fit the PCA projection."""
        files = []
        for path in sorted(self.benchmarks_dir.rglob("*.py")):
            relative_parts = path.relative_to(self.benchmarks_dir).parts
            if relative_parts and relative_parts[0] == "wrappers":
                continue
            files.append(path)
        return files

    def _build_pca_snippets(self, code: str) -> list[str]:
        """Split a source file into deduplicated snippets for PCA fitting."""
        stripped = code.strip()
        if not stripped:
            return []

        snippets = [stripped]
        lines = stripped.splitlines()
        if len(lines) <= 48:
            return snippets

        chunk_size = 48
        stride = 24
        for start in range(0, len(lines), stride):
            chunk = "\n".join(lines[start:start + chunk_size]).strip()
            if len(chunk.splitlines()) < 12:
                continue
            snippets.append(chunk)
            if start + chunk_size >= len(lines):
                break

        deduplicated = []
        seen = set()
        for snippet in snippets:
            if snippet in seen:
                continue
            seen.add(snippet)
            deduplicated.append(snippet)
        return deduplicated

    def _extract_raw_embedding_from_code(self, code: str) -> np.ndarray:
        """Extract and cache a raw embedding for one source snippet."""
        cache_key = self.analyzer._get_cache_key(code)
        cached = self.analyzer._load_raw_from_cache(cache_key)
        if cached is not None:
            return cached
        embedding = self.analyzer.extract_raw_embedding(code)
        self.analyzer._save_raw_to_cache(cache_key, embedding)
        return embedding

    def _ensure_pca_ready(self) -> None:
        """Fit or load PCA before extracting an individual benchmark feature."""
        if not self._supports_pca_pipeline():
            return
        if self.analyzer.pca_components is not None:
            return
        if self.analyzer.load_pca_state(self.pca_state_path):
            print(f"[PCA] Loaded cached state: {self.pca_state_path}", flush=True)
            return

        print("[PCA] No cached state found; fitting from benchmark corpus", flush=True)
        corpus_embeddings = self._collect_pca_corpus_embeddings()
        self.analyzer.fit_pca(np.array(corpus_embeddings))
        self.analyzer.save_pca_state(self.pca_state_path)
        explained = (
            float(self.analyzer.pca_explained_variance_ratio.sum())
            if self.analyzer.pca_explained_variance_ratio is not None
            else 0.0
        )
        print(f"[PCA] Fit complete, explained variance: {explained:.4f}", flush=True)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Apply analyzer normalization or local L2 normalization."""
        if hasattr(self.analyzer, "normalize_features"):
            return self.analyzer.normalize_features(vector)
        array = np.asarray(vector, dtype=np.float32)
        l2_norm = float(np.linalg.norm(array))
        if l2_norm <= 1e-12:
            return array
        return array / l2_norm

    def _cache_reduced_features(self, features: Dict[str, np.ndarray]) -> None:
        """Write reduced embeddings into the analyzer cache."""
        print("[Cache] Updating reduced feature cache", flush=True)
        for name, vector in features.items():
            benchmark_path = self._get_benchmark_path(name)
            if benchmark_path is None:
                continue
            main_file = self._find_main_function(benchmark_path)
            if main_file is None:
                continue
            code = main_file.read_text(encoding="utf-8")
            cache_key = self.analyzer._get_cache_key(code)
            if self._supports_pca_pipeline():
                self.analyzer._save_reduced_to_cache(cache_key, vector)
            print(f"  {name}: cached", flush=True)

    def save_features(
        self,
        features_dict: Dict[str, np.ndarray],
        output_path: str,
    ) -> None:
        """Save a feature dictionary to a pickle file."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "wb") as f:
            pickle.dump(features_dict, f)
        print(f"[Save] Features saved to: {output}", flush=True)

    def load_features(self, input_path: str) -> Dict[str, np.ndarray]:
        """Load a feature dictionary from a pickle file."""
        with open(input_path, "rb") as f:
            features_dict = pickle.load(f)
        print(
            f"[Load] Loaded {len(features_dict)} feature vectors from {input_path}",
            flush=True,
        )
        return features_dict


def main() -> None:
    """CLI entry point for feature extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract CodeBERT features for CALO benchmark functions."
    )
    parser.add_argument("--benchmark", type=str, help="Extract one benchmark.")
    parser.add_argument(
        "--extract-all",
        action="store_true",
        help="Extract every benchmark in the default list.",
    )
    parser.add_argument(
        "--sebs-root",
        type=str,
        default=".",
        help="Repository root that contains the benchmarks directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".cache/codebert_embeddings/all_features.pkl",
        help="Output pickle path for --extract-all.",
    )

    args = parser.parse_args()
    extractor = BenchmarkFeatureExtractor(sebs_root=args.sebs_root)

    if args.benchmark:
        features = extractor.extract_single_benchmark(args.benchmark)
        if features is not None:
            print(f"\nFeature vector ({features.shape[0]} dimensions):")
            print(features)
    elif args.extract_all:
        features = extractor.extract_all_benchmarks()
        extractor.save_features(features, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
