"""State-space definition for the CALO RL optimizer."""

import numpy as np
from typing import Dict, Optional, List, Iterable

from .benchmark_features import BenchmarkFeatureExtractor
from .codebert_analyzer import CodeBERTAnalyzer
from .load_monitor import LoadMonitor
from .state_components import ConfigurationContext, FunctionCategory, PerformanceHistory


class StateSpace:
    """State space used by the CALO agent."""

    DEFAULT_CODE_FEATURE_DIM = 32
    LOAD_FEATURE_DIM = 10
    CATEGORY_DIM = 5
    HISTORY_DIM = 5
    CONTEXT_DIM = 27
    LOAD_FEATURE_NAMES = (
        "current_qps",
        "qps_trend",
        "burst_indicator",
        "peak_concurrency",
        "cold_start_probability",
        "avg_response_time",
        "hour_of_day",
        "day_of_week",
        "load_volatility",
        "container_warmth",
    )
    CATEGORY_FEATURE_NAMES = (
        "category_webapps",
        "category_multimedia",
        "category_utilities",
        "category_inference",
        "category_scientific",
    )
    HISTORY_FEATURE_NAMES = (
        "history_mean_latency",
        "history_mean_cost",
        "history_success_rate",
        "history_cold_start_rate",
        "history_latency_variance",
    )
    CONTEXT_FEATURE_NAMES = (
        "memory_128mb",
        "memory_256mb",
        "memory_512mb",
        "memory_1024mb",
        "memory_2048mb",
        "memory_3008mb",
        "arch_x64",
        "arch_arm64",
        "timeout_60s",
        "timeout_120s",
        "timeout_300s",
        "timeout_900s",
        "cpu_utilization",
        "memory_utilization",
        "invocation_volume",
        "failure_rate",
        "reserved_context_00",
        "reserved_context_01",
        "reserved_context_02",
        "reserved_context_03",
        "reserved_context_04",
        "reserved_context_05",
        "reserved_context_06",
        "reserved_context_07",
        "reserved_context_08",
        "reserved_context_09",
        "reserved_context_10",
    )

    def __init__(
        self,
        codebert_analyzer: Optional[CodeBERTAnalyzer] = None,
        sebs_root: str = ".",
        enable_code_features: bool = True,
        enable_function_category: bool = True,
        code_feature_dim: int = DEFAULT_CODE_FEATURE_DIM,
    ):
        """
        Initialize the state space.

        Args:
            codebert_analyzer: Optional CodeBERT analyzer.
            sebs_root: SeBS project root.
            enable_code_features: Whether to include CodeBERT features.
            enable_function_category: Whether to include benchmark-category features.
            code_feature_dim: Dimensionality of the reduced code embedding.
        """
        self.sebs_root = str(sebs_root)
        self.enable_code_features = bool(enable_code_features)
        self.enable_function_category = bool(enable_function_category)
        self.code_feature_dim = int(code_feature_dim)
        if self.code_feature_dim <= 0:
            raise ValueError("code_feature_dim must be positive")

        self.total_dim = (
            self.code_feature_dim
            + self.LOAD_FEATURE_DIM
            + self.CATEGORY_DIM
            + self.HISTORY_DIM
            + self.CONTEXT_DIM
        )

        # LLM-based code feature extractor.
        self.codebert_analyzer = codebert_analyzer
        self.feature_extractor: Optional[BenchmarkFeatureExtractor] = None
        if self.enable_code_features:
            if self.codebert_analyzer is None:
                self.codebert_analyzer = CodeBERTAnalyzer.get_shared(
                    embed_dim=self.code_feature_dim
                )
            self.feature_extractor = BenchmarkFeatureExtractor(
                sebs_root=sebs_root, analyzer=self.codebert_analyzer
            )

        # State-space subcomponents.
        self.load_monitor = LoadMonitor()
        self.performance_history = PerformanceHistory()
        self.config_context = ConfigurationContext()

        # Cached code features and metadata.
        self.code_features: Optional[np.ndarray] = None
        self.function_category: Optional[int] = None
        self.simulation_time_sec = 0.0

    def set_function(self, benchmark_name: str) -> None:
        """
        Set the benchmark function to optimize.

        Args:
            benchmark_name: Benchmark name such as '110.dynamic-html'.
        """
        # Extract code features, unless this is a load-only ablation.
        if self.enable_code_features:
            if self.feature_extractor is None:
                if self.codebert_analyzer is None:
                    self.codebert_analyzer = CodeBERTAnalyzer.get_shared(
                        embed_dim=self.code_feature_dim
                    )
                self.feature_extractor = BenchmarkFeatureExtractor(
                    sebs_root=self.sebs_root,
                    analyzer=self.codebert_analyzer,
                )
            extracted = self.feature_extractor.extract_single_benchmark(benchmark_name)
            if extracted is None:
                raise ValueError(f"Failed to extract code features for benchmark: {benchmark_name}")
            self.code_features = np.asarray(extracted, dtype=np.float32)
        else:
            self.code_features = np.zeros(self.code_feature_dim, dtype=np.float32)

        # Infer the function category.
        self.function_category = FunctionCategory.from_benchmark_name(benchmark_name)

        print(f"[StateSpace] Function set: {benchmark_name}")
        print(
            "  - Code features: "
            f"{self.code_features.shape} "
            f"(enabled={self.enable_code_features})"
        )
        print(f"  - Function category enabled: {self.enable_function_category}")
        print(f"  - Category: {self.function_category}")

    def set_function_from_code(self, code: str, category: int) -> None:
        """
        Set function features directly from source code.

        Args:
            code: Function source code.
            category: Function category from ``FunctionCategory``.
        """
        if self.enable_code_features:
            if self.codebert_analyzer is None:
                self.codebert_analyzer = CodeBERTAnalyzer.get_shared(
                    embed_dim=self.code_feature_dim
                )
            self.code_features = self.codebert_analyzer.extract_features(code)
        else:
            self.code_features = np.zeros(self.code_feature_dim, dtype=np.float32)
        self.function_category = category

    def fork_for_env(self) -> "StateSpace":
        """Create an isolated state-space copy that reuses static code analysis."""
        forked = StateSpace(
            codebert_analyzer=self.codebert_analyzer,
            sebs_root=self.sebs_root,
            enable_code_features=self.enable_code_features,
            enable_function_category=self.enable_function_category,
            code_feature_dim=self.code_feature_dim,
        )
        forked.code_features = (
            None if self.code_features is None else np.array(self.code_features, copy=True)
        )
        forked.function_category = self.function_category
        forked.simulation_time_sec = self.simulation_time_sec
        forked.load_monitor.container_ttl = self.load_monitor.container_ttl
        forked.load_monitor.set_current_time(self.simulation_time_sec)
        return forked

    def extract_state(self) -> np.ndarray:
        """
        Extract the current state vector.

        Returns:
            One-dimensional NumPy array with ``state_dim`` entries.
        """
        if self.code_features is None:
            raise ValueError("Call set_function() before extract_state().")
        if self.function_category is None:
            raise ValueError("Function category is not initialized.")

        # 1. LLM code features.
        llm_features = self.code_features

        # 2. Load features.
        load_features = self.load_monitor.extract_features()

        # 3. Function category.
        if self.enable_function_category:
            category_features = FunctionCategory.to_onehot(self.function_category)
        else:
            category_features = np.zeros(self.CATEGORY_DIM, dtype=np.float32)

        # 4. Historical performance.
        history_features = self.performance_history.extract_features()

        # 5. Context features.
        context_features = self.config_context.extract_features()

        # Concatenate all state components.
        state = np.concatenate(
            [llm_features, load_features, category_features, history_features, context_features]
        )

        assert (
            state.shape[0] == self.total_dim
        ), f"Unexpected state dimension: {state.shape[0]} != {self.total_dim}"

        return state

    def get_state_breakdown(self) -> Dict[str, np.ndarray]:
        """
        Return a detailed state breakdown for debugging.

        Returns:
            Dictionary of state sub-vectors.
        """
        if self.code_features is None or self.function_category is None:
            raise ValueError("Call set_function() before get_state_breakdown().")
        return {
            "llm_features": self.code_features,
            "load_features": self.load_monitor.extract_features(),
            "category_features": (
                FunctionCategory.to_onehot(self.function_category)
                if self.enable_function_category
                else np.zeros(self.CATEGORY_DIM, dtype=np.float32)
            ),
            "history_features": self.performance_history.extract_features(),
            "context_features": self.config_context.extract_features(),
        }

    def update_load(
        self,
        is_cold_start: bool = False,
        timestamp: Optional[float] = None,
    ) -> None:
        """Update load-monitor statistics."""
        self.load_monitor.record_request(is_cold_start, timestamp=timestamp)

    def update_response(
        self,
        response_time: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Update observed response-time statistics."""
        self.load_monitor.record_response(response_time, timestamp=timestamp)

    def record_batch(
        self,
        arrival_times: Iterable[float],
        completion_times: Iterable[float],
        response_times: Iterable[float],
        cold_start_flags: Iterable[bool],
    ) -> None:
        """Replay one workload batch into the load monitor."""
        self.load_monitor.record_batch(
            arrival_times=arrival_times,
            completion_times=completion_times,
            response_times=response_times,
            cold_start_flags=cold_start_flags,
        )

    def update_performance(
        self,
        latency: float,
        cost: float,
        success: float | bool,
        cold_start: float | bool,
        invocation_count: int = 1,
    ) -> None:
        """Update performance history and invocation counters."""
        self.performance_history.record(latency, cost, success, cold_start)
        self.config_context.record_invocation(success, count=invocation_count)

    def update_configuration(
        self,
        memory_mb: int,
        architecture: str,
        timeout_sec: int,
    ) -> None:
        """Update the current resource configuration."""
        self.config_context.set_configuration(memory_mb, architecture, timeout_sec)

    def update_utilization(self, cpu: float, memory: float) -> None:
        """Update resource-utilization signals."""
        self.config_context.update_utilization(cpu, memory)

    def set_simulation_time(self, current_time_sec: float) -> None:
        """Update the simulated time seen by the load monitor."""
        self.simulation_time_sec = float(current_time_sec)
        self.load_monitor.set_current_time(self.simulation_time_sec)

    def reset(self) -> None:
        """Reset all dynamic state while keeping cached code features."""
        self.load_monitor.reset()
        self.performance_history.reset()
        self.config_context.reset()
        self.simulation_time_sec = 0.0

    @property
    def state_dim(self) -> int:
        """Return the state-space dimensionality."""
        return self.total_dim

    @classmethod
    def get_group_slices(
        cls,
        code_feature_dim: int = DEFAULT_CODE_FEATURE_DIM,
    ) -> Dict[str, slice]:
        """Return the slice occupied by each state group."""
        code_end = int(code_feature_dim)
        load_end = code_end + cls.LOAD_FEATURE_DIM
        category_end = load_end + cls.CATEGORY_DIM
        history_end = category_end + cls.HISTORY_DIM
        context_end = history_end + cls.CONTEXT_DIM
        return {
            "code": slice(0, code_end),
            "load": slice(code_end, load_end),
            "category": slice(load_end, category_end),
            "history": slice(category_end, history_end),
            "context": slice(history_end, context_end),
        }

    @classmethod
    def get_feature_layout(
        cls,
        code_feature_dim: int = DEFAULT_CODE_FEATURE_DIM,
    ) -> List[Dict[str, object]]:
        """Return names and metadata for each state dimension."""
        layout: List[Dict[str, object]] = []
        group_slices = cls.get_group_slices(code_feature_dim=code_feature_dim)

        for index in range(group_slices["code"].start, group_slices["code"].stop):
            layout.append(
                {
                    "index": index,
                    "group": "code",
                    "name": f"code_embedding_{index:02d}",
                    "interpretable": False,
                }
            )

        offset = group_slices["load"].start
        for local_index, name in enumerate(cls.LOAD_FEATURE_NAMES):
            layout.append(
                {
                    "index": offset + local_index,
                    "group": "load",
                    "name": name,
                    "interpretable": True,
                }
            )

        offset = group_slices["category"].start
        for local_index, name in enumerate(cls.CATEGORY_FEATURE_NAMES):
            layout.append(
                {
                    "index": offset + local_index,
                    "group": "category",
                    "name": name,
                    "interpretable": True,
                }
            )

        offset = group_slices["history"].start
        for local_index, name in enumerate(cls.HISTORY_FEATURE_NAMES):
            layout.append(
                {
                    "index": offset + local_index,
                    "group": "history",
                    "name": name,
                    "interpretable": True,
                }
            )

        offset = group_slices["context"].start
        for local_index, name in enumerate(cls.CONTEXT_FEATURE_NAMES):
            layout.append(
                {
                    "index": offset + local_index,
                    "group": "context",
                    "name": name,
                    "interpretable": not name.startswith("reserved_"),
                }
            )

        expected_dim = (
            int(code_feature_dim)
            + cls.LOAD_FEATURE_DIM
            + cls.CATEGORY_DIM
            + cls.HISTORY_DIM
            + cls.CONTEXT_DIM
        )
        if len(layout) != expected_dim:
            raise ValueError(
                f"Feature layout length {len(layout)} does not match "
                f"expected dim {expected_dim}"
            )
        return layout
