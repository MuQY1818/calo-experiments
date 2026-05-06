"""LLM-based function analyzer built on CodeBERT."""

import os
import pickle
import hashlib
from contextlib import contextmanager
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class CodeBERTAnalyzer:
    """
    使用CodeBERT提取函数代码的语义特征
    """

    _SHARED_ANALYZERS: Dict[tuple[str, str, int], "CodeBERTAnalyzer"] = {}

    def __init__(
        self,
        model_name='microsoft/codebert-base',
        cache_dir='.cache/codebert_embeddings',
        embed_dim=32
    ):
        """
        初始化CodeBERT分析器

        Args:
            model_name: HuggingFace模型名称
            cache_dir: embedding缓存目录
            embed_dim: 降维后的embedding维度
        """
        print(f"[CodeBERT] Loading model: {model_name}")

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cache_dir = self.cache_dir / "raw"
        self.reduced_cache_dir = self.cache_dir / f"reduced_{embed_dim}d"
        self.raw_cache_dir.mkdir(parents=True, exist_ok=True)
        self.reduced_cache_dir.mkdir(parents=True, exist_ok=True)
        self.pca_state_path = self.cache_dir / f"pca_{embed_dim}d.pkl"

        self.embed_dim = embed_dim

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CodeBERTAnalyzer requires torch and transformers. "
                "Install them in the active environment before use."
            ) from exc

        self.torch = torch

        self.tokenizer = self._load_pretrained_component(
            AutoTokenizer,
            model_name,
            component_name='tokenizer',
        )
        self.model = self._load_pretrained_component(
            AutoModel,
            model_name,
            component_name='model',
        )

        # 设置为评估模式，冻结参数
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 如果有GPU就用GPU
        self.device = self.torch.device(
            'cuda' if self.torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        print(f"[CodeBERT] Model ready on device: {self.device}")

        # PCA状态（延迟加载）
        self.pca_components = None
        self.pca_mean = None
        self.pca_explained_variance_ratio = None
        self.pca_model_id = "unfitted"
        if self.pca_state_path.exists():
            self.load_pca_state(self.pca_state_path)

    @classmethod
    def get_shared(
        cls,
        model_name: str = 'microsoft/codebert-base',
        cache_dir: str = '.cache/codebert_embeddings',
        embed_dim: int = 32,
    ) -> "CodeBERTAnalyzer":
        """Return one shared analyzer instance for the current process."""
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
        """Load one Hugging Face component with a local-first strategy."""
        try:
            with self._offline_hf_hub():
                component = auto_class.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
            print(f"[CodeBERT] Loaded {component_name} from local cache")
            return component
        except Exception as local_error:
            print(
                f"[CodeBERT] Local cache load failed for {component_name}; "
                "falling back to default resolution"
            )
            try:
                return auto_class.from_pretrained(model_name)
            except Exception:
                raise local_error

    @contextmanager
    def _offline_hf_hub(self):
        """Temporarily force offline Hugging Face resolution for cache-only loads."""
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
        """生成代码的哈希键"""
        return hashlib.md5(code.encode('utf-8')).hexdigest()

    def _load_pickle(self, cache_path: Path) -> Optional[np.ndarray]:
        """从缓存文件加载对象"""
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _save_pickle(self, cache_path: Path, obj):
        """将对象保存到缓存文件"""
        with open(cache_path, 'wb') as f:
            pickle.dump(obj, f)

    def _load_raw_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """加载原始768维embedding缓存"""
        return self._load_pickle(self.raw_cache_dir / f"{cache_key}.pkl")

    def _save_raw_to_cache(self, cache_key: str, embedding: np.ndarray):
        """保存原始768维embedding缓存"""
        self._save_pickle(self.raw_cache_dir / f"{cache_key}.pkl", embedding)

    def _load_reduced_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """加载降维后的特征缓存"""
        return self._load_pickle(
            self.reduced_cache_dir / f"{self.pca_model_id}_{cache_key}.pkl"
        )

    def _save_reduced_to_cache(self, cache_key: str, embedding: np.ndarray):
        """保存降维后的特征缓存"""
        self._save_pickle(
            self.reduced_cache_dir / f"{self.pca_model_id}_{cache_key}.pkl",
            embedding,
        )

    def extract_raw_embedding(self, code: str) -> np.ndarray:
        """
        提取原始的768维embedding

        Args:
            code: 函数源代码

        Returns:
            768维numpy数组
        """
        # Tokenize代码
        inputs = self.tokenizer(
            code,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        )

        # 移动到正确的设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 提取embedding
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS] token的表示
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding.squeeze(0)  # (768,)

    def extract_raw_from_file(self, file_path: str, use_cache: bool = True) -> np.ndarray:
        """从文件提取原始768维embedding。"""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
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
        """对降维后的特征做 L2 归一化。"""
        embedding = np.asarray(embedding, dtype=np.float32)
        l2_norm = float(np.linalg.norm(embedding))
        if l2_norm <= 1e-12:
            return embedding
        return embedding / l2_norm

    def fit_pca(self, embeddings: np.ndarray):
        """基于一批原始 embedding 拟合 PCA。"""
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(
                f"PCA input must be a 2D matrix, received shape {matrix.shape}"
            )

        n_samples, n_features = matrix.shape
        max_components = min(n_samples, n_features)
        n_components = min(self.embed_dim, max_components)
        if n_components <= 0:
            raise ValueError("PCA requires at least one valid sample")

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
            self.pca_explained_variance_ratio = np.ones(n_components, dtype=np.float32)

        state_bytes = self.pca_components.tobytes() + self.pca_mean.tobytes()
        self.pca_model_id = hashlib.md5(state_bytes).hexdigest()[:12]

    def save_pca_state(self, output_path: str | Path | None = None):
        """保存 PCA 状态。"""
        if self.pca_components is None or self.pca_mean is None:
            raise ValueError("PCA is not fitted yet and cannot be saved")
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

    def load_pca_state(self, input_path: str | Path | None = None):
        """加载 PCA 状态。"""
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
        """用已拟合的 PCA 将768维 embedding 投影到目标维度。"""
        vector = np.asarray(embedding, dtype=np.float32)
        if self.pca_components is None or self.pca_mean is None:
            if self.embed_dim >= vector.shape[0]:
                return self.normalize_features(vector)
            truncated = vector[:self.embed_dim]
            return self.normalize_features(truncated)

        reduced = (vector - self.pca_mean) @ self.pca_components.T
        if reduced.shape[0] < self.embed_dim:
            padding = np.zeros(self.embed_dim - reduced.shape[0], dtype=np.float32)
            reduced = np.concatenate([reduced.astype(np.float32), padding], axis=0)
        return self.normalize_features(reduced.astype(np.float32))

    def extract_features(self, code: str, use_cache: bool = True) -> np.ndarray:
        """
        提取降维后的特征向量

        Args:
            code: 函数源代码
            use_cache: 是否使用缓存

        Returns:
            降维后的特征向量
        """
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

    def extract_from_file(self, file_path: str, use_cache: bool = True) -> np.ndarray:
        """
        从文件中提取特征

        Args:
            file_path: 源代码文件路径
            use_cache: 是否使用缓存

        Returns:
            特征向量
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        return self.extract_features(code, use_cache)


class BenchmarkFeatureExtractor:
    """
    为SeBS的所有benchmark提取特征
    """

    def __init__(self, sebs_root: str = '.', analyzer: Optional[CodeBERTAnalyzer] = None):
        """
        Args:
            sebs_root: SeBS项目根目录
            analyzer: CodeBERT分析器（可选，不提供则创建新的）
        """
        self.sebs_root = Path(sebs_root)
        self.benchmarks_dir = self.sebs_root / 'benchmarks'

        if analyzer is None:
            self.analyzer = CodeBERTAnalyzer.get_shared()
        else:
            self.analyzer = analyzer

        # Benchmark列表
        self.benchmark_list = [
            '110.dynamic-html',
            '120.uploader',
            '130.crud-api',
            '210.thumbnailer',
            '220.video-processing',
            '311.compression',
            '411.image-recognition',
            '501.graph-pagerank',
            '502.graph-mst',
            '503.graph-bfs',
            '504.dna-visualisation',
        ]

        self.pca_state_path = self.analyzer.pca_state_path

    def _find_main_function(self, benchmark_path: Path) -> Optional[Path]:
        """
        查找benchmark的主函数文件

        Args:
            benchmark_path: benchmark目录路径

        Returns:
            主函数文件路径，如果找不到返回None
        """
        # 可能的文件名
        possible_files = [
            'handler.py',
            'function.py',
            'index.py',
            '__init__.py',
        ]

        # 首先尝试python目录
        python_dir = benchmark_path / 'python'
        if python_dir.exists():
            for fname in possible_files:
                fpath = python_dir / fname
                if fpath.exists():
                    return fpath

        # 如果python目录没有，尝试benchmark根目录
        for fname in possible_files:
            fpath = benchmark_path / fname
            if fpath.exists():
                return fpath

        return None

    def extract_single_benchmark(self, benchmark_name: str) -> Optional[np.ndarray]:
        """
        提取单个benchmark的特征

        Args:
            benchmark_name: benchmark名称，如 '110.dynamic-html'

        Returns:
            特征向量，如果失败返回None
        """
        # 构建路径
        category = benchmark_name.split('.')[0]
        if category.startswith('0'):
            category_dir = '000.microbenchmarks'
        elif category.startswith('1'):
            category_dir = '100.webapps'
        elif category.startswith('2'):
            category_dir = '200.multimedia'
        elif category.startswith('3'):
            category_dir = '300.utilities'
        elif category.startswith('4'):
            category_dir = '400.inference'
        elif category.startswith('5'):
            category_dir = '500.scientific'
        else:
            print(f"[Error] Unrecognized benchmark category: {benchmark_name}")
            return None

        benchmark_path = self.benchmarks_dir / category_dir / benchmark_name

        if not benchmark_path.exists():
            print(f"[Error] Benchmark path does not exist: {benchmark_path}")
            return None

        # 查找主函数文件
        main_file = self._find_main_function(benchmark_path)
        if main_file is None:
            print(f"[Error] Main function file not found: {benchmark_name}")
            return None

        print(f"[Extract] {benchmark_name}: {main_file}")

        # 提取特征
        try:
            if self._supports_pca_pipeline():
                self._ensure_pca_ready()
                raw_embedding = self.analyzer.extract_raw_from_file(str(main_file))
                features = self.analyzer.project_embedding(raw_embedding)
                with open(main_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                cache_key = self.analyzer._get_cache_key(code)
                self.analyzer._save_reduced_to_cache(cache_key, features)
            else:
                features = self.analyzer.extract_from_file(str(main_file))
            print(f"[Success] {benchmark_name}: feature shape {features.shape}")
            return features
        except Exception as e:
            print(f"[Error] {benchmark_name}: {e}")
            return None

    def extract_all_benchmarks(self, use_pca: bool = True) -> Dict[str, np.ndarray]:
        """
        提取所有benchmark的特征（改进版：使用PCA统一降维）

        Args:
            use_pca: 是否使用PCA降维（推荐True以获得更好的区分度）

        Returns:
            字典: {benchmark_name: features}
        """
        print("=" * 60)
        print("开始提取所有benchmark的特征 (两阶段PCA方法)")
        print("=" * 60)

        raw_embeddings_dict = self._collect_raw_embeddings()
        if len(raw_embeddings_dict) == 0:
            print("[Error] 没有成功提取任何特征")
            return {}

        names = list(raw_embeddings_dict.keys())
        raw_embeddings_matrix = np.array([raw_embeddings_dict[name] for name in names])
        print(f"目标benchmark特征矩阵: {raw_embeddings_matrix.shape}")

        if use_pca and self._supports_pca_pipeline():
            pca_corpus = self._collect_pca_corpus_embeddings()
            pca_matrix = np.array(pca_corpus)
            print(
                f"\n[阶段2] 基于 {pca_matrix.shape[0]} 个语料样本拟合真实PCA并投影到"
                f"{self.analyzer.embed_dim}维..."
            )
            self.analyzer.fit_pca(pca_matrix)
            self.analyzer.save_pca_state(self.pca_state_path)
            explained_var = (
                float(self.analyzer.pca_explained_variance_ratio.sum())
                if self.analyzer.pca_explained_variance_ratio is not None
                else 0.0
            )
            print(f"PCA降维完成: 768维 → {self.analyzer.embed_dim}维")
            print(f"保留方差比例: {explained_var:.4f}")
            normalized_embeddings = np.vstack(
                [self.analyzer.project_embedding(raw_embeddings_dict[name]) for name in names]
            )
        else:
            print(f"\n[阶段2] 回退到简单截断 + L2 归一化")
            normalized_embeddings = np.vstack(
                [
                    self._normalize_vector(raw_embeddings_dict[name][:self.analyzer.embed_dim])
                    for name in names
                ]
            )

        # 构建特征字典
        features_dict = {name: normalized_embeddings[i] for i, name in enumerate(names)}

        # 更新缓存（用新的特征覆盖旧的）
        print("\n[更新缓存]")
        for name, features in features_dict.items():
            benchmark_path = self._get_benchmark_path(name)
            if benchmark_path is None:
                continue
            main_file = self._find_main_function(benchmark_path)
            if main_file is None:
                continue
            with open(main_file, 'r', encoding='utf-8') as f:
                code = f.read()
            cache_key = self.analyzer._get_cache_key(code)
            if self._supports_pca_pipeline():
                self.analyzer._save_reduced_to_cache(cache_key, features)
            print(f"  {name}: 已缓存")

        print("\n" + "=" * 60)
        print(f"提取完成: {len(features_dict)}/{len(self.benchmark_list)} 个benchmark")
        print("=" * 60)

        return features_dict

    def _get_benchmark_path(self, benchmark_name: str) -> Optional[Path]:
        """获取benchmark的路径"""
        category = benchmark_name.split('.')[0]
        if category.startswith('0'):
            category_dir = '000.microbenchmarks'
        elif category.startswith('1'):
            category_dir = '100.webapps'
        elif category.startswith('2'):
            category_dir = '200.multimedia'
        elif category.startswith('3'):
            category_dir = '300.utilities'
        elif category.startswith('4'):
            category_dir = '400.inference'
        elif category.startswith('5'):
            category_dir = '500.scientific'
        else:
            print(f"[Error] Unrecognized benchmark category: {benchmark_name}")
            return None

        benchmark_path = self.benchmarks_dir / category_dir / benchmark_name

        if not benchmark_path.exists():
            print(f"[Error] Benchmark path does not exist: {benchmark_path}")
            return None

        return benchmark_path

    def _supports_pca_pipeline(self) -> bool:
        """检查分析器是否支持原始embedding + PCA 投影流程。"""
        required_methods = [
            'extract_raw_from_file',
            'project_embedding',
            'fit_pca',
            'save_pca_state',
            'load_pca_state',
            'normalize_features',
        ]
        return all(hasattr(self.analyzer, method) for method in required_methods)

    def _collect_raw_embeddings(self) -> Dict[str, np.ndarray]:
        """收集全部 benchmark 的原始 768 维 embedding。"""
        print("\n[阶段1] 提取原始768维embedding...")
        raw_embeddings_dict = {}

        for benchmark_name in self.benchmark_list:
            benchmark_path = self._get_benchmark_path(benchmark_name)
            if benchmark_path is None:
                continue

            main_file = self._find_main_function(benchmark_path)
            if main_file is None:
                print(f"[Error] 找不到主函数文件: {benchmark_name}")
                continue

            print(f"[Extract] {benchmark_name}: {main_file}")
            try:
                if self._supports_pca_pipeline():
                    raw_embedding = self.analyzer.extract_raw_from_file(str(main_file))
                else:
                    raw_embedding = self.analyzer.extract_from_file(str(main_file))
                raw_embeddings_dict[benchmark_name] = raw_embedding
                print(f"[Success] {benchmark_name}: 原始特征 {raw_embedding.shape}")
            except Exception as e:
                print(f"[Error] {benchmark_name}: {e}")

        return raw_embeddings_dict

    def _collect_pca_corpus_embeddings(self) -> list[np.ndarray]:
        """收集用于拟合 PCA 的 benchmark Python 语料 embedding。"""
        print("\n[PCA语料] 收集 benchmark 目录中的 Python 源文件...")
        corpus_embeddings = []
        seen_cache_keys = set()

        for source_file in self._iter_pca_corpus_files():
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                snippets = self._build_pca_snippets(code)
                for snippet in snippets:
                    cache_key = self.analyzer._get_cache_key(snippet)
                    if cache_key in seen_cache_keys:
                        continue
                    embedding = self._extract_raw_embedding_from_code(snippet)
                    corpus_embeddings.append(embedding)
                    seen_cache_keys.add(cache_key)
            except Exception as e:
                print(f"[Warn] 跳过 PCA 语料文件 {source_file}: {e}")

        if not corpus_embeddings:
            raise ValueError("无法收集 PCA 语料 embedding")
        print(f"[PCA语料] 有效样本数: {len(corpus_embeddings)}")
        return corpus_embeddings

    def _iter_pca_corpus_files(self) -> list[Path]:
        """返回用于 PCA 拟合的源码文件列表。"""
        files = []
        for path in sorted(self.benchmarks_dir.rglob("*.py")):
            relative_parts = path.relative_to(self.benchmarks_dir).parts
            if relative_parts and relative_parts[0] == "wrappers":
                continue
            files.append(path)
        return files

    def _build_pca_snippets(self, code: str) -> list[str]:
        """将源码切成适合 PCA 语料的去重片段。"""
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
        """提取一段源码的原始 embedding，并写入原始缓存。"""
        cache_key = self.analyzer._get_cache_key(code)
        cached = self.analyzer._load_raw_from_cache(cache_key)
        if cached is not None:
            return cached
        embedding = self.analyzer.extract_raw_embedding(code)
        self.analyzer._save_raw_to_cache(cache_key, embedding)
        return embedding

    def _ensure_pca_ready(self):
        """确保单 benchmark 提取时已有可用 PCA 状态。"""
        if not self._supports_pca_pipeline():
            return
        if self.analyzer.pca_components is not None:
            return
        if self.analyzer.load_pca_state(self.pca_state_path):
            print(f"[PCA] 已加载缓存状态: {self.pca_state_path}")
            return

        print("[PCA] 未找到缓存状态，开始基于 benchmark 语料拟合 PCA...")
        corpus_embeddings = self._collect_pca_corpus_embeddings()
        matrix = np.array(corpus_embeddings)
        self.analyzer.fit_pca(matrix)
        self.analyzer.save_pca_state(self.pca_state_path)
        explained = (
            float(self.analyzer.pca_explained_variance_ratio.sum())
            if self.analyzer.pca_explained_variance_ratio is not None
            else 0.0
        )
        print(f"[PCA] 拟合完成，累计解释方差: {explained:.4f}")

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """在 analyzer 不提供 normalize_features 时做本地 L2 归一化。"""
        if hasattr(self.analyzer, "normalize_features"):
            return self.analyzer.normalize_features(vector)
        array = np.asarray(vector, dtype=np.float32)
        l2_norm = float(np.linalg.norm(array))
        if l2_norm <= 1e-12:
            return array
        return array / l2_norm

    def save_features(self, features_dict: Dict[str, np.ndarray], output_path: str):
        """
        保存所有特征到文件

        Args:
            features_dict: 特征字典
            output_path: 输出文件路径
        """
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)

        print(f"[Save] 特征已保存到: {output_path}")

    def load_features(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        从文件加载特征

        Args:
            input_path: 输入文件路径

        Returns:
            特征字典
        """
        with open(input_path, 'rb') as f:
            features_dict = pickle.load(f)

        print(f"[Load] 从 {input_path} 加载了 {len(features_dict)} 个特征")
        return features_dict


def visualize_embeddings(features_dict: Dict[str, np.ndarray], output_path: str = 'results/figures/embeddings_tsne.png'):
    """
    可视化embeddings（使用t-SNE降维到2D）

    Args:
        features_dict: 特征字典
        output_path: 输出图片路径
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] 需要安装 matplotlib: pip install matplotlib")
        return

    # 准备数据
    names = list(features_dict.keys())
    features = np.array([features_dict[name] for name in names])

    # 提取类别信息（用于着色）
    categories = []
    category_names = {
        '1': 'Webapps',
        '2': 'Multimedia',
        '3': 'Utilities',
        '4': 'Inference',
        '5': 'Scientific',
    }

    for name in names:
        first_digit = name[0]
        categories.append(category_names.get(first_digit, 'Unknown'))

    # t-SNE降维
    print("[Visualize] 使用t-SNE降维到2D...")
    # perplexity需要小于样本数量，建议为样本数的1/3左右
    perplexity = min(30, len(features) - 1)
    perplexity = max(5, perplexity // 3)  # 至少为5
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    # 绘图
    plt.figure(figsize=(12, 8))

    # 为每个类别分配颜色
    unique_categories = list(set(categories))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
    category_to_color = {cat: colors[i] for i, cat in enumerate(unique_categories)}

    for i, name in enumerate(names):
        cat = categories[i]
        plt.scatter(
            features_2d[i, 0],
            features_2d[i, 1],
            color=category_to_color[cat],
            label=cat if cat not in [categories[j] for j in range(i)] else '',
            s=100,
            alpha=0.7
        )
        plt.annotate(
            name,
            (features_2d[i, 0], features_2d[i, 1]),
            fontsize=8,
            alpha=0.8
        )

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('CodeBERT Embeddings of Serverless Functions (t-SNE Visualization)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"[Save] 可视化图已保存到: {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='提取Serverless函数的CodeBERT特征')
    parser.add_argument('--benchmark', type=str, help='提取单个benchmark')
    parser.add_argument('--extract-all', action='store_true', help='提取所有benchmark')
    parser.add_argument('--visualize', action='store_true', help='可视化embeddings')
    parser.add_argument('--sebs-root', type=str, default='.', help='SeBS项目根目录')
    parser.add_argument('--output', type=str, default='.cache/codebert_embeddings/all_features.pkl', help='输出文件路径')

    args = parser.parse_args()

    # 创建特征提取器
    extractor = BenchmarkFeatureExtractor(sebs_root=args.sebs_root)

    if args.benchmark:
        # 提取单个benchmark
        features = extractor.extract_single_benchmark(args.benchmark)
        if features is not None:
            print(f"\n特征向量 ({features.shape[0]}维):")
            print(features)

    elif args.extract_all:
        # 提取所有benchmark
        features_dict = extractor.extract_all_benchmarks()

        # 保存
        extractor.save_features(features_dict, args.output)

        # 可视化
        if args.visualize:
            visualize_embeddings(features_dict)

    else:
        parser.print_help()
