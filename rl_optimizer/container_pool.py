"""Container-pool model for the CALO simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class ContainerState:
    """State of one warm container."""

    available_at_sec: float
    last_used_at_sec: float


class ContainerPool:
    """Simple warm-container pool with TTL eviction and queueing."""

    def __init__(
        self,
        ttl_sec: float = 600.0,
        max_containers: int = 32,
    ):
        self.ttl_sec = float(ttl_sec)
        self.max_containers = int(max_containers)
        self._containers: List[ContainerState] = []

    def reset(self) -> None:
        """Reset all containers."""
        self._containers.clear()

    def set_ttl(self, ttl_sec: float) -> None:
        """Update the pool TTL."""
        self.ttl_sec = float(ttl_sec)

    def _evict_expired(self, current_time_sec: float) -> None:
        """Evict containers that stayed idle longer than TTL."""
        alive = []
        for container in self._containers:
            idle_time = current_time_sec - max(
                container.last_used_at_sec, container.available_at_sec
            )
            if idle_time < self.ttl_sec:
                alive.append(container)
        self._containers = alive

    def simulate_batch(
        self,
        arrival_times_sec: Sequence[float],
        warm_service_times_ms: Sequence[float],
        cold_overheads_ms: Sequence[float],
    ) -> Dict[str, np.ndarray | float | int]:
        """Simulate one batch of arrivals through the container pool."""
        if len(arrival_times_sec) == 0:
            empty = np.asarray([], dtype=np.float32)
            return {
                "latencies_ms": empty,
                "wait_times_ms": empty,
                "completion_times_sec": empty,
                "cold_flags": empty.astype(bool),
                "queue_flags": empty.astype(bool),
                "peak_concurrency": 0,
                "container_count": len(self._containers),
            }

        latencies_ms = []
        wait_times_ms = []
        completion_times_sec = []
        cold_flags = []
        queue_flags = []

        event_deltas: List[tuple[float, int]] = []

        for arrival_time_sec, warm_service_ms, cold_overhead_ms in zip(
            arrival_times_sec, warm_service_times_ms, cold_overheads_ms
        ):
            self._evict_expired(arrival_time_sec)

            idle_candidates = [
                container
                for container in self._containers
                if container.available_at_sec <= arrival_time_sec
            ]

            cold_start = False
            queue_delay_ms = 0.0

            if idle_candidates:
                container = min(idle_candidates, key=lambda item: item.available_at_sec)
                start_time_sec = arrival_time_sec
            elif len(self._containers) < self.max_containers:
                cold_start = True
                container = ContainerState(
                    available_at_sec=arrival_time_sec,
                    last_used_at_sec=arrival_time_sec,
                )
                self._containers.append(container)
                start_time_sec = arrival_time_sec
            else:
                container = min(self._containers, key=lambda item: item.available_at_sec)
                start_time_sec = container.available_at_sec
                queue_delay_ms = max(0.0, (start_time_sec - arrival_time_sec) * 1000.0)

            total_service_ms = float(warm_service_ms)
            if cold_start:
                total_service_ms += float(cold_overhead_ms)

            completion_time_sec = start_time_sec + total_service_ms / 1000.0
            latency_ms = queue_delay_ms + total_service_ms

            container.available_at_sec = completion_time_sec
            container.last_used_at_sec = completion_time_sec

            latencies_ms.append(latency_ms)
            wait_times_ms.append(queue_delay_ms)
            completion_times_sec.append(completion_time_sec)
            cold_flags.append(cold_start)
            queue_flags.append(queue_delay_ms > 0.0)

            event_deltas.append((arrival_time_sec, 1))
            event_deltas.append((completion_time_sec, -1))

        event_deltas.sort(key=lambda item: (item[0], -item[1]))
        active = 0
        peak_concurrency = 0
        for _, delta in event_deltas:
            active += delta
            peak_concurrency = max(peak_concurrency, active)

        return {
            "latencies_ms": np.asarray(latencies_ms, dtype=np.float32),
            "wait_times_ms": np.asarray(wait_times_ms, dtype=np.float32),
            "completion_times_sec": np.asarray(completion_times_sec, dtype=np.float32),
            "cold_flags": np.asarray(cold_flags, dtype=bool),
            "queue_flags": np.asarray(queue_flags, dtype=bool),
            "peak_concurrency": int(peak_concurrency),
            "container_count": len(self._containers),
        }
