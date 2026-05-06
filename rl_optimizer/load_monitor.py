"""Load monitor for serverless functions."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


class LoadMonitor:
    """Tracks request-level events and derives a 10-D load vector."""

    def __init__(self, window_size: int = 300):
        self.window_size = int(window_size)
        self.request_timestamps: Deque[float] = deque(maxlen=100000)
        self.response_records: Deque[Tuple[float, float]] = deque(maxlen=100000)
        self.concurrency_samples: Deque[Tuple[float, int]] = deque(maxlen=100000)
        self.concurrent_requests = 0
        self.cold_start_count = 0
        self.warm_start_count = 0
        self.last_request_time: Optional[float] = None
        self.container_ttl = 600.0
        self._current_time: Optional[float] = None

    def _now(self) -> float:
        """Return the current timestamp for feature extraction."""
        if self._current_time is not None:
            return float(self._current_time)
        return float(time.time())

    def set_current_time(self, timestamp: float) -> None:
        """Set the current simulated time."""
        self._current_time = float(timestamp)

    def advance_time(self, delta_sec: float) -> float:
        """Advance the current simulated time."""
        if self._current_time is None:
            self._current_time = float(time.time())
        self._current_time += float(delta_sec)
        return float(self._current_time)

    def _trim_window(self, current_time: float) -> None:
        """Trim response and concurrency history to the monitoring window."""
        response_cutoff = current_time - self.window_size
        while self.response_records and self.response_records[0][0] < response_cutoff:
            self.response_records.popleft()
        while self.concurrency_samples and self.concurrency_samples[0][0] < response_cutoff:
            self.concurrency_samples.popleft()

    def record_request(
        self,
        is_cold_start: bool = False,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record one request arrival."""
        now = float(timestamp) if timestamp is not None else self._now()
        self.set_current_time(now)
        self.request_timestamps.append(now)
        self.concurrent_requests += 1
        self.concurrency_samples.append((now, self.concurrent_requests))

        if is_cold_start:
            self.cold_start_count += 1
        else:
            self.warm_start_count += 1

        self.last_request_time = now
        self._trim_window(now)

    def record_response(self, response_time: float, timestamp: Optional[float] = None) -> None:
        """Record one response completion."""
        now = float(timestamp) if timestamp is not None else self._now()
        self.set_current_time(now)
        self.response_records.append((now, float(response_time)))
        self.concurrent_requests = max(0, self.concurrent_requests - 1)
        self.concurrency_samples.append((now, self.concurrent_requests))
        self._trim_window(now)

    def record_batch(
        self,
        arrival_times: Iterable[float],
        completion_times: Iterable[float],
        response_times: Iterable[float],
        cold_start_flags: Iterable[bool],
    ) -> None:
        """Replay a batch of request/response events into the monitor."""
        events: List[Tuple[float, str, float | bool]] = []
        for arrival_time, cold_flag in zip(arrival_times, cold_start_flags):
            events.append((float(arrival_time), "request", bool(cold_flag)))
        for completion_time, response_time in zip(completion_times, response_times):
            events.append((float(completion_time), "response", float(response_time)))

        events.sort(key=lambda item: (item[0], 0 if item[1] == "request" else 1))
        for timestamp, event_type, payload in events:
            if event_type == "request":
                self.record_request(is_cold_start=bool(payload), timestamp=timestamp)
            else:
                self.record_response(response_time=float(payload), timestamp=timestamp)

    def get_current_qps(self) -> float:
        """Compute the QPS over the last second."""
        now = self._now()
        recent_requests = [ts for ts in self.request_timestamps if now - ts <= 1.0]
        return float(len(recent_requests))

    def get_qps_trend(self) -> float:
        """Compute a simple load trend over the monitoring window."""
        now = self._now()
        window_start = now - self.window_size
        recent_requests = [ts for ts in self.request_timestamps if ts >= window_start]
        if len(recent_requests) < 2:
            return 0.0

        bucket_size = 10.0
        buckets: Dict[int, int] = {}
        for timestamp in recent_requests:
            bucket_index = int((timestamp - window_start) / bucket_size)
            buckets[bucket_index] = buckets.get(bucket_index, 0) + 1

        x = np.asarray(list(buckets.keys()), dtype=np.float32)
        y = np.asarray([buckets[key] / bucket_size for key in x], dtype=np.float32)
        if len(x) < 2:
            return 0.0

        x_mean = float(x.mean())
        y_mean = float(y.mean())
        numerator = float(((x - x_mean) * (y - y_mean)).sum())
        denominator = float(((x - x_mean) ** 2).sum())
        if denominator < 1e-10:
            return 0.0
        return numerator / denominator

    def detect_burst(self) -> float:
        """Detect whether recent arrivals represent a burst."""
        now = self._now()
        recent_10s = [ts for ts in self.request_timestamps if now - ts <= 10.0]
        recent_60s = [ts for ts in self.request_timestamps if now - ts <= 60.0]
        qps_10s = len(recent_10s) / 10.0
        qps_60s = len(recent_60s) / 60.0
        if qps_60s < 0.1:
            return 0.0
        burst_ratio = qps_10s / qps_60s
        return float(min(1.0, max(0.0, (burst_ratio - 1.0) / 2.0)))

    def get_cold_start_probability(self) -> float:
        """Estimate the probability that the next request will cold start."""
        total_starts = self.cold_start_count + self.warm_start_count
        if total_starts == 0:
            return 0.5

        base_prob = self.cold_start_count / total_starts
        if self.last_request_time is None:
            return float(base_prob)

        time_since_last = self._now() - self.last_request_time
        if time_since_last >= self.container_ttl:
            return 1.0
        time_factor = max(0.0, time_since_last / max(self.container_ttl, 1e-6))
        return float(min(1.0, base_prob * 0.7 + time_factor * 0.3))

    def get_avg_response_time(self) -> float:
        """Return the recent mean response time in seconds."""
        if not self.response_records:
            return 0.0
        values = [record[1] for record in self.response_records]
        return float(np.mean(values))

    def get_recent_peak_concurrency(self) -> int:
        """Return the recent peak concurrency inside the monitoring window."""
        if not self.concurrency_samples:
            return self.concurrent_requests
        return int(max(sample[1] for sample in self.concurrency_samples))

    def get_load_volatility(self) -> float:
        """Compute a normalized load volatility score."""
        now = self._now()
        window_start = now - 60.0
        buckets: Dict[int, int] = {}
        for timestamp in self.request_timestamps:
            if timestamp >= window_start:
                bucket_index = int(timestamp - window_start)
                buckets[bucket_index] = buckets.get(bucket_index, 0) + 1

        if len(buckets) < 2:
            return 0.0

        counts = np.asarray(list(buckets.values()), dtype=np.float32)
        mean = float(counts.mean())
        if mean < 0.1:
            return 0.0
        cv = float(counts.std() / mean)
        return float(min(1.0, cv / 2.0))

    def estimate_container_warmth(self) -> float:
        """Estimate how warm the container pool currently is."""
        if self.last_request_time is None:
            return 0.0
        time_since_last = self._now() - self.last_request_time
        if time_since_last >= self.container_ttl:
            return 0.0
        warmth = 1.0 - (time_since_last / max(self.container_ttl, 1e-6))
        return float(max(0.0, min(1.0, warmth)))

    def extract_features(self) -> np.ndarray:
        """Extract the 10-D load feature vector."""
        now_dt = datetime.utcfromtimestamp(self._now())
        features = np.zeros(10, dtype=np.float32)
        features[0] = min(1.0, self.get_current_qps() / 100.0)
        features[1] = np.clip(self.get_qps_trend() / 10.0, -1.0, 1.0)
        features[2] = self.detect_burst()
        features[3] = min(1.0, self.get_recent_peak_concurrency() / 50.0)
        features[4] = self.get_cold_start_probability()
        features[5] = min(1.0, self.get_avg_response_time() / 10.0)
        features[6] = now_dt.hour / 24.0
        features[7] = now_dt.weekday() / 7.0
        features[8] = self.get_load_volatility()
        features[9] = self.estimate_container_warmth()
        return features

    def get_stats(self) -> Dict:
        """Return human-readable load statistics."""
        return {
            "current_qps": self.get_current_qps(),
            "qps_trend": self.get_qps_trend(),
            "burst_detected": self.detect_burst() > 0.5,
            "concurrent_requests": self.concurrent_requests,
            "peak_concurrency": self.get_recent_peak_concurrency(),
            "cold_start_prob": self.get_cold_start_probability(),
            "avg_response_time": self.get_avg_response_time(),
            "load_volatility": self.get_load_volatility(),
            "container_warmth": self.estimate_container_warmth(),
            "total_requests": len(self.request_timestamps),
            "cold_starts": self.cold_start_count,
            "warm_starts": self.warm_start_count,
        }

    def reset(self) -> None:
        """Reset all tracked events."""
        self.request_timestamps.clear()
        self.response_records.clear()
        self.concurrency_samples.clear()
        self.concurrent_requests = 0
        self.cold_start_count = 0
        self.warm_start_count = 0
        self.last_request_time = None
        self._current_time = None


class MultiLoadMonitor:
    """Manages one LoadMonitor per function."""

    def __init__(self):
        self.monitors: Dict[str, LoadMonitor] = {}

    def get_monitor(self, function_name: str) -> LoadMonitor:
        """Get the monitor for one function."""
        if function_name not in self.monitors:
            self.monitors[function_name] = LoadMonitor()
        return self.monitors[function_name]

    def record_request(
        self,
        function_name: str,
        is_cold_start: bool = False,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record one request for the selected function."""
        monitor = self.get_monitor(function_name)
        monitor.record_request(is_cold_start=is_cold_start, timestamp=timestamp)

    def record_response(
        self,
        function_name: str,
        response_time: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record one response for the selected function."""
        monitor = self.get_monitor(function_name)
        monitor.record_response(response_time=response_time, timestamp=timestamp)

    def extract_features(self, function_name: str) -> np.ndarray:
        """Extract load features for one function."""
        monitor = self.get_monitor(function_name)
        return monitor.extract_features()

    def get_all_stats(self) -> Dict[str, Dict]:
        """Return monitor statistics for all functions."""
        return {name: monitor.get_stats() for name, monitor in self.monitors.items()}
