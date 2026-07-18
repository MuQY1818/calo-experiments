"""Benchmark discovery and source-feature persistence for CALO."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle

import numpy as np

from .codebert_analyzer import CodeBERTAnalyzer


class BenchmarkFeatureExtractor:
    """Extract CodeBERT features for SeBS-style benchmark directories."""

    DEFAULT_BENCHMARKS = [
        "110.dynamic-html",
        "120.uploader",
        "210.thumbnailer",
        "311.compression",
        "411.image-recognition",
    ]

    def __init__(
        self,
        sebs_root: str | Path = ".",
        analyzer: CodeBERTAnalyzer | None = None,
    ):
        """Initialize benchmark discovery under a project root."""
        self.sebs_root = Path(sebs_root)
        self.benchmarks_dir = self.sebs_root / "benchmarks"
        self.analyzer = analyzer or CodeBERTAnalyzer.get_shared()
        self.benchmark_list = list(self.DEFAULT_BENCHMARKS)
        self.pca_state_path = self.analyzer.pca_state_path

    def extract_single_benchmark(self, benchmark_name: str) -> np.ndarray | None:
        """Extract reduced source-code features for one benchmark."""
        benchmark_path = self._get_benchmark_path(benchmark_name)
        if benchmark_path is None:
            return None
        main_file = self._find_main_function(benchmark_path)
        if main_file is None:
            print(f"[Error] Main function file not found: {benchmark_name}", flush=True)
            return None
        try:
            self._ensure_pca_ready()
            raw = self.analyzer.extract_raw_from_file(str(main_file))
            features = self.analyzer.project_embedding(raw)
            cache_key = self.analyzer._get_cache_key(main_file.read_text(encoding="utf-8"))
            self.analyzer._save_reduced_to_cache(cache_key, features)
            return features
        except Exception as exc:
            print(f"[Error] {benchmark_name}: {exc}", flush=True)
            return None

    def extract_all_benchmarks(self, use_pca: bool = True) -> dict[str, np.ndarray]:
        """Extract features for every available benchmark in the configured list."""
        raw_embeddings = self._collect_raw_embeddings()
        if not raw_embeddings:
            return {}
        names = list(raw_embeddings)
        if use_pca:
            corpus_fingerprint = self._pca_corpus_fingerprint()
            corpus = np.asarray(self._collect_pca_corpus_embeddings())
            self.analyzer.fit_pca(
                corpus,
                corpus_fingerprint=corpus_fingerprint,
            )
            self.analyzer.save_pca_state()
            reduced = np.vstack(
                [self.analyzer.project_embedding(raw_embeddings[name]) for name in names]
            )
        else:
            reduced = np.vstack(
                [
                    self.analyzer.normalize_features(
                        raw_embeddings[name][: self.analyzer.embed_dim]
                    )
                    for name in names
                ]
            )
        features = {name: reduced[index] for index, name in enumerate(names)}
        self._cache_reduced_features(features)
        return features

    @staticmethod
    def _find_main_function(benchmark_path: Path) -> Path | None:
        """Find the conventional Python entry file for one benchmark."""
        filenames = ("handler.py", "function.py", "index.py", "__init__.py")
        for parent in (benchmark_path / "python", benchmark_path):
            for filename in filenames:
                candidate = parent / filename
                if candidate.exists():
                    return candidate
        return None

    def _get_benchmark_path(self, benchmark_name: str) -> Path | None:
        """Resolve the category directory for one benchmark name."""
        category_dirs = {
            "0": "000.microbenchmarks",
            "1": "100.webapps",
            "2": "200.multimedia",
            "3": "300.utilities",
            "4": "400.inference",
            "5": "500.scientific",
        }
        category = category_dirs.get(benchmark_name[0])
        if category is None:
            return None
        benchmark_path = self.benchmarks_dir / category / benchmark_name
        return benchmark_path if benchmark_path.exists() else None

    def _collect_raw_embeddings(self) -> dict[str, np.ndarray]:
        """Collect raw embeddings for available benchmark entry files."""
        embeddings: dict[str, np.ndarray] = {}
        for benchmark_name in self.benchmark_list:
            benchmark_path = self._get_benchmark_path(benchmark_name)
            main_file = None if benchmark_path is None else self._find_main_function(benchmark_path)
            if main_file is None:
                continue
            try:
                embeddings[benchmark_name] = self.analyzer.extract_raw_from_file(str(main_file))
            except Exception as exc:
                print(f"[Error] {benchmark_name}: {exc}", flush=True)
        return embeddings

    def _collect_pca_corpus_embeddings(self) -> list[np.ndarray]:
        """Build the source-snippet corpus used to fit the PCA projection."""
        embeddings: list[np.ndarray] = []
        seen: set[str] = set()
        for source_file in self._iter_pca_corpus_files():
            try:
                code = source_file.read_text(encoding="utf-8")
                for snippet in self._build_pca_snippets(code):
                    cache_key = self.analyzer._get_cache_key(snippet)
                    if cache_key in seen:
                        continue
                    embeddings.append(self._extract_raw_embedding_from_code(snippet))
                    seen.add(cache_key)
            except Exception as exc:
                print(f"[Warning] Skipping PCA source {source_file}: {exc}", flush=True)
        if not embeddings:
            raise ValueError("No PCA corpus embeddings could be collected")
        return embeddings

    def _iter_pca_corpus_files(self) -> list[Path]:
        """List benchmark Python files eligible for PCA fitting."""
        return [
            path
            for path in sorted(self.benchmarks_dir.rglob("*.py"))
            if path.relative_to(self.benchmarks_dir).parts[0] != "wrappers"
        ]

    def _pca_corpus_fingerprint(self) -> str:
        """Return a stable identity for the ordered benchmark source corpus."""
        digest = hashlib.sha256()
        snippet_count = 0
        for source_file in self._iter_pca_corpus_files():
            try:
                code = source_file.read_text(encoding="utf-8")
            except OSError:
                continue
            relative = source_file.relative_to(self.benchmarks_dir).as_posix()
            for snippet in self._build_pca_snippets(code):
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(self.analyzer._get_cache_key(snippet).encode("ascii"))
                digest.update(b"\0")
                snippet_count += 1
        if snippet_count == 0:
            raise ValueError("No PCA corpus snippets could be identified")
        return digest.hexdigest()

    @staticmethod
    def _build_pca_snippets(code: str) -> list[str]:
        """Split one source file into deterministic overlapping snippets."""
        stripped = code.strip()
        if not stripped:
            return []
        snippets = [stripped]
        lines = stripped.splitlines()
        if len(lines) > 48:
            for start in range(0, len(lines), 24):
                chunk = "\n".join(lines[start : start + 48]).strip()
                if len(chunk.splitlines()) >= 12:
                    snippets.append(chunk)
                if start + 48 >= len(lines):
                    break
        return list(dict.fromkeys(snippets))

    def _extract_raw_embedding_from_code(self, code: str) -> np.ndarray:
        """Extract and cache one raw source embedding."""
        cache_key = self.analyzer._get_cache_key(code)
        cached = self.analyzer._load_raw_from_cache(cache_key)
        if cached is not None:
            return cached
        embedding = self.analyzer.extract_raw_embedding(code)
        self.analyzer._save_raw_to_cache(cache_key, embedding)
        return embedding

    def _ensure_pca_ready(self) -> None:
        """Load or fit PCA before projecting an individual benchmark."""
        corpus_fingerprint = self._pca_corpus_fingerprint()
        if (
            self.analyzer.pca_components is not None
            and self.analyzer.pca_corpus_fingerprint == corpus_fingerprint
        ):
            return
        if self.analyzer.load_pca_state(expected_corpus_fingerprint=corpus_fingerprint):
            return
        self.analyzer.fit_pca(
            np.asarray(self._collect_pca_corpus_embeddings()),
            corpus_fingerprint=corpus_fingerprint,
        )
        self.analyzer.save_pca_state()

    def _cache_reduced_features(self, features: dict[str, np.ndarray]) -> None:
        """Cache reduced vectors under source-content keys."""
        for benchmark_name, vector in features.items():
            benchmark_path = self._get_benchmark_path(benchmark_name)
            main_file = None if benchmark_path is None else self._find_main_function(benchmark_path)
            if main_file is None:
                continue
            cache_key = self.analyzer._get_cache_key(main_file.read_text(encoding="utf-8"))
            self.analyzer._save_reduced_to_cache(cache_key, vector)

    @staticmethod
    def save_features(features: dict[str, np.ndarray], output_path: str | Path) -> None:
        """Persist a benchmark-to-feature mapping."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(features, handle)

    @staticmethod
    def load_features(input_path: str | Path) -> dict[str, np.ndarray]:
        """Load a benchmark-to-feature mapping."""
        with Path(input_path).open("rb") as handle:
            return pickle.load(handle)
