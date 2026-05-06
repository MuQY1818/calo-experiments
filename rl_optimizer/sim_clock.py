"""Simulation clock used by the CALO simulator."""

from __future__ import annotations


class SimulationClock:
    """Tracks simulated time in seconds."""

    def __init__(self, start_time_sec: float = 0.0):
        self._start_time_sec = float(start_time_sec)
        self._current_time_sec = float(start_time_sec)

    def reset(self, start_time_sec: float | None = None) -> None:
        """Reset the simulated clock."""
        if start_time_sec is not None:
            self._start_time_sec = float(start_time_sec)
        self._current_time_sec = float(self._start_time_sec)

    def now(self) -> float:
        """Return the current simulated time in seconds."""
        return self._current_time_sec

    def set(self, current_time_sec: float) -> None:
        """Set the current simulated time."""
        self._current_time_sec = float(current_time_sec)

    def advance(self, delta_sec: float) -> float:
        """Advance the simulated time and return the new timestamp."""
        self._current_time_sec += float(delta_sec)
        return self._current_time_sec
