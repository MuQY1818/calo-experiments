"""Workload sources for the CALO simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class WorkloadStep:
    """Workload metadata for one simulator step."""

    arrival_count: int
    step_duration_sec: float
    source_name: str
    load_value: float
    minute_of_day: int
    profile_key: str = ""
    mean_invocations_per_minute: float = 0.0
    std_invocations_per_minute: float = 0.0
    max_invocations_per_minute: float = 0.0
    burstiness_hint: float = 1.0


@dataclass(frozen=True)
class BenchmarkWorkloadSpec:
    """Target workload characteristics used for Azure profile matching."""

    target_duration_ms: float
    target_burstiness: float
    target_zero_fraction: float
    target_active_fraction: float
    target_concurrency: float
    preferred_trigger: str | None = None


DEFAULT_BENCHMARK_WORKLOAD_SPECS: Dict[str, BenchmarkWorkloadSpec] = {
    "110.dynamic-html": BenchmarkWorkloadSpec(
        target_duration_ms=220.0,
        target_burstiness=1.20,
        target_zero_fraction=0.10,
        target_active_fraction=0.90,
        target_concurrency=0.75,
        preferred_trigger="http",
    ),
    "120.uploader": BenchmarkWorkloadSpec(
        target_duration_ms=350.0,
        target_burstiness=1.45,
        target_zero_fraction=0.22,
        target_active_fraction=0.78,
        target_concurrency=0.60,
        preferred_trigger="http",
    ),
    "210.thumbnailer": BenchmarkWorkloadSpec(
        target_duration_ms=850.0,
        target_burstiness=1.60,
        target_zero_fraction=0.35,
        target_active_fraction=0.65,
        target_concurrency=0.45,
        preferred_trigger="queue",
    ),
    "311.compression": BenchmarkWorkloadSpec(
        target_duration_ms=450.0,
        target_burstiness=1.55,
        target_zero_fraction=0.25,
        target_active_fraction=0.75,
        target_concurrency=0.50,
        preferred_trigger="queue",
    ),
    "411.image-recognition": BenchmarkWorkloadSpec(
        target_duration_ms=2000.0,
        target_burstiness=2.00,
        target_zero_fraction=0.45,
        target_active_fraction=0.55,
        target_concurrency=0.30,
        preferred_trigger="queue",
    ),
}


CATEGORY_FALLBACK_SPECS: Dict[str, BenchmarkWorkloadSpec] = {
    "1": BenchmarkWorkloadSpec(250.0, 1.25, 0.12, 0.88, 0.70, "http"),
    "2": BenchmarkWorkloadSpec(700.0, 1.50, 0.30, 0.70, 0.50, "queue"),
    "3": BenchmarkWorkloadSpec(450.0, 1.55, 0.24, 0.76, 0.55, "queue"),
    "4": BenchmarkWorkloadSpec(1800.0, 1.95, 0.42, 0.58, 0.32, "queue"),
    "5": BenchmarkWorkloadSpec(900.0, 1.70, 0.35, 0.65, 0.40, "queue"),
}


class SyntheticWorkloadSource:
    """Synthetic workload source used as a fallback."""

    def __init__(
        self,
        pattern: str,
        step_duration_sec: float = 15.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.pattern = pattern
        self.step_duration_sec = float(step_duration_sec)
        self.rng = rng if rng is not None else np.random.default_rng()

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal random state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _load_multiplier(self, step_index: int) -> float:
        """Return the synthetic load multiplier for one step."""
        if self.pattern == "sine":
            period = 20
            return float(1.1 + np.sin(2.0 * np.pi * step_index / period))
        if self.pattern == "surge":
            surge_profile = (0.35, 0.45, 0.85, 1.30, 2.40, 0.75, 1.80, 2.90)
            phase_index = step_index % len(surge_profile)
            return float(
                max(0.08, surge_profile[phase_index] + self.rng.uniform(-0.08, 0.08))
            )
        if self.pattern == "spike":
            if step_index % 15 == 0:
                return float(2.5 + self.rng.uniform(-0.3, 0.3))
            return float(0.45 + self.rng.uniform(-0.1, 0.1))
        if self.pattern == "decay":
            return float(max(0.1, 2.2 - step_index / 18.0) + self.rng.uniform(-0.1, 0.1))
        if self.pattern == "random":
            return float(np.clip(self.rng.lognormal(0.0, 0.55), 0.05, 3.0))
        return 1.0

    def next_step(self, step_index: int) -> WorkloadStep:
        """Generate one synthetic workload step."""
        load_multiplier = max(0.05, self._load_multiplier(step_index))
        mean_arrivals = max(0.1, load_multiplier * 10.0)
        arrival_count = int(self.rng.poisson(mean_arrivals))
        minute_of_day = int((step_index * self.step_duration_sec) // 60) % 1440
        burstiness_hint = 1.0
        if self.pattern == "spike":
            burstiness_hint = float(max(1.8, load_multiplier))
        elif self.pattern == "surge":
            burstiness_hint = float(max(1.6, 0.85 * load_multiplier))
        elif self.pattern == "random":
            burstiness_hint = 1.5
        elif self.pattern == "decay":
            burstiness_hint = 1.25
        elif self.pattern == "sine":
            burstiness_hint = 1.1
        return WorkloadStep(
            arrival_count=arrival_count,
            step_duration_sec=self.step_duration_sec,
            source_name=f"synthetic:{self.pattern}",
            load_value=load_multiplier,
            minute_of_day=minute_of_day,
            mean_invocations_per_minute=mean_arrivals * 60.0 / self.step_duration_sec,
            std_invocations_per_minute=0.0,
            max_invocations_per_minute=mean_arrivals * 60.0 / self.step_duration_sec,
            burstiness_hint=burstiness_hint,
        )


class AzureTraceWorkloadSource:
    """Minute-level workload replay based on Azure Functions profiles."""

    def __init__(
        self,
        profile_path: str | Path,
        summary_path: str | Path | None = None,
        step_duration_sec: float = 15.0,
        top_k: int = 50,
        arrival_scale: float = 1.0,
        max_arrivals_per_step: Optional[int] = None,
        benchmark_name: str | None = None,
        profile_selection: str = "benchmark_aware",
        selection_pool_size: int = 16,
        target_concurrency: float | None = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.profile_path = Path(profile_path)
        self.summary_path = (
            Path(summary_path)
            if summary_path is not None
            else self.profile_path.with_name("azure_functions_2019_function_summary.csv")
        )
        self.step_duration_sec = float(step_duration_sec)
        self.top_k = int(top_k)
        self.arrival_scale = float(arrival_scale)
        self.max_arrivals_per_step = (
            None if max_arrivals_per_step is None else int(max_arrivals_per_step)
        )
        self.benchmark_name = benchmark_name
        self.profile_selection = profile_selection
        self.selection_pool_size = max(1, int(selection_pool_size))
        self.target_concurrency = (
            None if target_concurrency is None else float(target_concurrency)
        )
        self.rng = rng if rng is not None else np.random.default_rng()
        self._profiles = self._load_profiles()
        self._summary = self._load_summary()
        self._selected_profile = None
        self._selected_profile_key = ""
        self._selected_profile_scale = self.arrival_scale
        self._selected_profile_info: Dict[str, float | str] = {}

    def _load_profiles(self) -> Dict[str, pd.DataFrame]:
        """Load Azure minute-level profiles grouped by function key."""
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Azure profile CSV not found: {self.profile_path}")

        frame = pd.read_csv(
            self.profile_path,
            usecols=[
                "function_key",
                "activity_rank",
                "minute_of_day",
                "mean_invocations",
                "std_invocations",
                "max_invocations",
            ],
        )
        frame = frame[frame["activity_rank"] <= self.top_k].copy()
        if frame.empty:
            raise ValueError(
                f"No Azure workload profile matched top_k={self.top_k} in {self.profile_path}"
            )

        profiles: Dict[str, pd.DataFrame] = {}
        for function_key, group in frame.groupby("function_key"):
            profiles[function_key] = group.sort_values("minute_of_day").reset_index(drop=True)
        return profiles

    def _load_summary(self) -> Optional[pd.DataFrame]:
        """Load function-level Azure summary metrics for profile selection."""
        if not self.summary_path.exists():
            return None

        frame = pd.read_csv(
            self.summary_path,
            usecols=[
                "function_key",
                "trigger",
                "avg_invocations_per_minute",
                "mean_burstiness",
                "mean_zero_minute_fraction",
                "mean_active_minute_fraction",
                "mean_duration_ms",
                "p50_duration_ms",
                "p99_duration_ms",
            ],
        )
        frame = frame[frame["function_key"].isin(self._profiles.keys())].copy()
        if frame.empty:
            return None
        return frame

    def _resolve_benchmark_spec(self) -> BenchmarkWorkloadSpec:
        """Resolve the benchmark-specific target workload shape."""
        if self.benchmark_name in DEFAULT_BENCHMARK_WORKLOAD_SPECS:
            spec = DEFAULT_BENCHMARK_WORKLOAD_SPECS[self.benchmark_name]
        elif self.benchmark_name:
            spec = CATEGORY_FALLBACK_SPECS.get(
                self.benchmark_name[0],
                BenchmarkWorkloadSpec(500.0, 1.5, 0.25, 0.75, 0.5, None),
            )
        else:
            spec = BenchmarkWorkloadSpec(500.0, 1.5, 0.25, 0.75, 0.5, None)

        if self.target_concurrency is None:
            return spec
        return BenchmarkWorkloadSpec(
            target_duration_ms=spec.target_duration_ms,
            target_burstiness=spec.target_burstiness,
            target_zero_fraction=spec.target_zero_fraction,
            target_active_fraction=spec.target_active_fraction,
            target_concurrency=self.target_concurrency,
            preferred_trigger=spec.preferred_trigger,
        )

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the sampled profile."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._selected_profile = None
        self._selected_profile_key = ""
        self._selected_profile_scale = self.arrival_scale
        self._selected_profile_info = {}

    def _random_profile_choice(self) -> str:
        """Randomly choose one profile key."""
        keys = list(self._profiles.keys())
        return str(self.rng.choice(keys))

    def _compute_profile_distance(
        self,
        summary_row: pd.Series,
        spec: BenchmarkWorkloadSpec,
    ) -> float:
        """Compute a similarity distance between one Azure profile and benchmark needs."""
        duration_distance = abs(
            np.log1p(float(summary_row["mean_duration_ms"]))
            - np.log1p(spec.target_duration_ms)
        )
        burst_distance = abs(float(summary_row["mean_burstiness"]) - spec.target_burstiness)
        zero_distance = abs(
            float(summary_row["mean_zero_minute_fraction"]) - spec.target_zero_fraction
        )
        active_distance = abs(
            float(summary_row["mean_active_minute_fraction"]) - spec.target_active_fraction
        )
        trigger_penalty = 0.0
        if spec.preferred_trigger is not None:
            trigger_penalty = 0.0 if summary_row["trigger"] == spec.preferred_trigger else 0.75
        return float(
            1.5 * duration_distance
            + 1.0 * burst_distance
            + 0.8 * zero_distance
            + 0.8 * active_distance
            + trigger_penalty
        )

    def _compute_selection_scale(
        self,
        profile_key: str,
        spec: BenchmarkWorkloadSpec,
    ) -> float:
        """Compute benchmark-aware arrival scaling for the selected profile."""
        if self._summary is None:
            return self.arrival_scale

        row = self._summary[self._summary["function_key"] == profile_key]
        if row.empty:
            return self.arrival_scale

        observed_rate = float(row.iloc[0]["avg_invocations_per_minute"])
        observed_rate = max(observed_rate, 1e-6)
        target_rate = (
            spec.target_concurrency * 60000.0 / max(spec.target_duration_ms, 1.0)
        )
        return float(self.arrival_scale * target_rate / observed_rate)

    def _select_profile_key(self) -> str:
        """Select one Azure profile key for replay."""
        if self.profile_selection != "benchmark_aware" or self._summary is None:
            return self._random_profile_choice()

        spec = self._resolve_benchmark_spec()
        candidates = self._summary.copy()
        if spec.preferred_trigger is not None:
            trigger_candidates = candidates[candidates["trigger"] == spec.preferred_trigger]
            if len(trigger_candidates) >= min(8, len(candidates)):
                candidates = trigger_candidates

        candidates = candidates.assign(
            selection_distance=candidates.apply(
                lambda row: self._compute_profile_distance(row, spec),
                axis=1,
            )
        )
        nearest = candidates.nsmallest(self.selection_pool_size, "selection_distance")
        weights = 1.0 / (nearest["selection_distance"].to_numpy(dtype=np.float64) + 1e-6)
        weights = weights / weights.sum()
        return str(
            self.rng.choice(
                nearest["function_key"].to_list(),
                p=weights,
            )
        )

    def _select_profile(self) -> pd.DataFrame:
        """Select one Azure function profile for replay."""
        if self._selected_profile is None:
            chosen_key = self._select_profile_key()
            self._selected_profile = self._profiles[chosen_key]
            self._selected_profile_key = chosen_key
            spec = self._resolve_benchmark_spec()
            self._selected_profile_scale = self._compute_selection_scale(chosen_key, spec)
            self._selected_profile_info = {
                "function_key": chosen_key,
                "benchmark_name": self.benchmark_name or "",
                "profile_selection": self.profile_selection,
                "effective_arrival_scale": float(self._selected_profile_scale),
                "target_concurrency": float(spec.target_concurrency),
            }
            if self._summary is not None:
                row = self._summary[self._summary["function_key"] == chosen_key]
                if not row.empty:
                    selected = row.iloc[0]
                    self._selected_profile_info.update(
                        {
                            "trigger": str(selected["trigger"]),
                            "avg_invocations_per_minute": float(
                                selected["avg_invocations_per_minute"]
                            ),
                            "mean_burstiness": float(selected["mean_burstiness"]),
                            "mean_duration_ms": float(selected["mean_duration_ms"]),
                            "mean_zero_minute_fraction": float(
                                selected["mean_zero_minute_fraction"]
                            ),
                        }
                    )
        return self._selected_profile

    def _sample_arrival_count(
        self,
        mean_per_step: float,
        std_per_step: float,
        cap_per_step: int | None = None,
    ) -> int:
        """Sample arrivals with over-dispersion when the trace variance demands it."""
        mean_per_step = max(0.0, float(mean_per_step))
        std_per_step = max(0.0, float(std_per_step))
        if mean_per_step <= 1e-8:
            return 0

        variance = std_per_step ** 2
        if variance > mean_per_step + 1e-6:
            dispersion = (mean_per_step ** 2) / max(variance - mean_per_step, 1e-6)
            dispersion = max(dispersion, 1e-3)
            success_prob = dispersion / (dispersion + mean_per_step)
            arrival_count = int(
                self.rng.negative_binomial(
                    n=dispersion,
                    p=np.clip(success_prob, 1e-6, 1.0 - 1e-6),
                )
            )
        else:
            arrival_count = int(self.rng.poisson(max(mean_per_step, 1e-3)))

        if cap_per_step is not None:
            arrival_count = min(arrival_count, int(cap_per_step))
        return max(arrival_count, 0)

    def next_step(self, step_index: int) -> WorkloadStep:
        """Generate one workload step from the selected Azure profile."""
        profile = self._select_profile()
        minute_of_day = int((step_index * self.step_duration_sec) // 60) % 1440 + 1
        row = profile.iloc[(minute_of_day - 1) % len(profile)]

        mean_per_step = max(
            0.0,
            float(row["mean_invocations"])
            * self._selected_profile_scale
            * self.step_duration_sec
            / 60.0,
        )
        std_per_step = max(
            0.0,
            float(row["std_invocations"])
            * self._selected_profile_scale
            * self.step_duration_sec
            / 60.0,
        )
        profile_max_per_step = max(
            1,
            int(
                np.ceil(
                    float(row["max_invocations"])
                    * self._selected_profile_scale
                    * self.step_duration_sec
                    / 60.0
                )
            ),
        )
        cap_per_step = profile_max_per_step
        if self.max_arrivals_per_step is not None:
            cap_per_step = min(cap_per_step, self.max_arrivals_per_step)
        arrival_count = self._sample_arrival_count(
            mean_per_step=mean_per_step,
            std_per_step=std_per_step,
            cap_per_step=cap_per_step,
        )

        effective_mean_per_step = mean_per_step
        if cap_per_step is not None:
            effective_mean_per_step = min(mean_per_step, float(cap_per_step))
        effective_mean_per_minute = effective_mean_per_step * 60.0 / self.step_duration_sec
        effective_max_per_minute = float(row["max_invocations"]) * self._selected_profile_scale
        if cap_per_step is not None:
            max_per_minute_cap = cap_per_step * 60.0 / self.step_duration_sec
            effective_max_per_minute = min(effective_max_per_minute, max_per_minute_cap)
        std_per_minute = std_per_step * 60.0 / self.step_duration_sec
        denominator = max(effective_mean_per_minute, 1e-6)
        burstiness_hint = max(
            1.0,
            0.6 * (effective_max_per_minute / denominator)
            + 0.4 * (std_per_minute / max(np.sqrt(denominator), 1.0)),
        )

        return WorkloadStep(
            arrival_count=arrival_count,
            step_duration_sec=self.step_duration_sec,
            source_name=f"azure_trace:{self.profile_selection}",
            load_value=effective_mean_per_step,
            minute_of_day=minute_of_day - 1,
            profile_key=self._selected_profile_key,
            mean_invocations_per_minute=effective_mean_per_minute,
            std_invocations_per_minute=std_per_minute,
            max_invocations_per_minute=effective_max_per_minute,
            burstiness_hint=float(burstiness_hint),
        )

    def get_selected_profile_info(self) -> Dict[str, float | str]:
        """Return metadata of the currently selected Azure profile."""
        return dict(self._selected_profile_info)
