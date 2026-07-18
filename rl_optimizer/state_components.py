"""Feature components composed by the CALO state-space coordinator."""

from collections import deque

import numpy as np


class FunctionCategory:
    """Encode benchmark categories used by the CALO state."""

    WEBAPPS = 0
    MULTIMEDIA = 1
    UTILITIES = 2
    INFERENCE = 3
    SCIENTIFIC = 4

    @staticmethod
    def from_benchmark_name(name: str) -> int:
        """Infer the function category from a benchmark name."""
        categories = {
            "1": FunctionCategory.WEBAPPS,
            "2": FunctionCategory.MULTIMEDIA,
            "3": FunctionCategory.UTILITIES,
            "4": FunctionCategory.INFERENCE,
            "5": FunctionCategory.SCIENTIFIC,
        }
        return categories.get(name[0], FunctionCategory.WEBAPPS)

    @staticmethod
    def to_onehot(category: int) -> np.ndarray:
        """Convert a category identifier to a five-element one-hot vector."""
        onehot = np.zeros(5, dtype=np.float32)
        onehot[category] = 1.0
        return onehot


class PerformanceHistory:
    """Maintain rolling latency, cost, success, and cold-start history."""

    def __init__(self, max_history: int = 100):
        """Initialize bounded performance buffers.

        Args:
            max_history: Maximum number of records retained per signal.
        """
        self.max_history = max_history
        self.latencies: deque[float] = deque(maxlen=max_history)
        self.costs: deque[float] = deque(maxlen=max_history)
        self.successes: deque[float] = deque(maxlen=max_history)
        self.cold_starts: deque[float] = deque(maxlen=max_history)

    def record(
        self,
        latency: float,
        cost: float,
        success: float | bool,
        cold_start: float | bool,
    ) -> None:
        """Record one execution result."""
        self.latencies.append(latency)
        self.costs.append(cost)
        self.successes.append(float(success))
        self.cold_starts.append(float(cold_start))

    def extract_features(self) -> np.ndarray:
        """Return the five-dimensional normalized history vector."""
        features = np.zeros(5, dtype=np.float32)
        if not self.latencies:
            return features

        features[0] = min(1.0, float(np.mean(self.latencies)) / 10000.0)
        features[1] = min(1.0, float(np.mean(self.costs)) / 0.01)
        features[2] = float(np.mean(self.successes))
        features[3] = float(np.mean(self.cold_starts))
        if len(self.latencies) > 1:
            features[4] = min(1.0, float(np.var(self.latencies)) / (1000.0**2))
        return features

    def reset(self) -> None:
        """Clear all stored performance observations."""
        self.latencies.clear()
        self.costs.clear()
        self.successes.clear()
        self.cold_starts.clear()


class ConfigurationContext:
    """Track the active configuration and execution counters."""

    def __init__(self):
        """Initialize the provider-default context."""
        self.memory_mb = 128
        self.architecture = "x64"
        self.timeout_sec = 60
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.total_invocations = 0
        self.failed_invocations = 0

    def set_configuration(self, memory_mb: int, architecture: str, timeout_sec: int) -> None:
        """Set the active resource configuration."""
        self.memory_mb = memory_mb
        self.architecture = architecture
        self.timeout_sec = timeout_sec

    def update_utilization(self, cpu: float, memory: float) -> None:
        """Update simulated resource-utilization signals."""
        self.cpu_utilization = cpu
        self.memory_utilization = memory

    def record_invocation(self, success: float | bool, count: int = 1) -> None:
        """Record one invocation batch."""
        count = max(0, int(count))
        if count == 0:
            return
        self.total_invocations += count
        success_rate = float(np.clip(float(success), 0.0, 1.0))
        failed = int(round(count * (1.0 - success_rate)))
        self.failed_invocations += min(count, max(0, failed))

    def extract_features(self) -> np.ndarray:
        """Return the 27-dimensional normalized context vector."""
        features = np.zeros(27, dtype=np.float32)
        memory_options = [128, 256, 512, 1024, 2048, 3008]
        memory_index = (
            memory_options.index(self.memory_mb)
            if self.memory_mb in memory_options
            else int(np.argmin([abs(self.memory_mb - value) for value in memory_options]))
        )
        features[memory_index] = 1.0
        features[6 if self.architecture == "x64" else 7] = 1.0

        timeout_options = [60, 120, 300, 900]
        timeout_index = (
            timeout_options.index(self.timeout_sec)
            if self.timeout_sec in timeout_options
            else int(np.argmin([abs(self.timeout_sec - value) for value in timeout_options]))
        )
        features[8 + timeout_index] = 1.0
        features[12] = np.clip(self.cpu_utilization, 0.0, 1.0)
        features[13] = np.clip(self.memory_utilization, 0.0, 1.0)
        if self.total_invocations > 0:
            features[14] = min(1.0, np.log10(self.total_invocations) / 5.0)
            features[15] = self.failed_invocations / self.total_invocations
        return features

    def reset(self) -> None:
        """Reset dynamic counters while keeping the active configuration."""
        self.total_invocations = 0
        self.failed_invocations = 0
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
