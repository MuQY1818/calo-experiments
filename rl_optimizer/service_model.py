"""Calibrated service-time model for the CALO simulator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .lightgbm_service_surrogate import LightGBMServiceSurrogate


DEFAULT_BASE_LATENCIES_MS = {
    "110": 200.0,
    "120": 300.0,
    "130": 250.0,
    "210": 800.0,
    "220": 1500.0,
    "311": 400.0,
    "411": 2000.0,
    "501": 500.0,
    "502": 600.0,
    "503": 400.0,
    "504": 1000.0,
}


@dataclass(frozen=True)
class ExactFeasibilityHint:
    """Feasibility evidence extracted from an exact calibration profile."""

    source: str
    success_rate: float
    successful_invocations: int
    profile_source: str
    matched_concurrency: float | None = None

    @property
    def is_feasible(self) -> bool:
        """Return whether the exact calibrated row indicates a runnable config."""
        return self.successful_invocations > 0 and self.success_rate > 0.0


class CalibratedServiceModel:
    """Samples warm and cold execution times from calibration tables."""

    def __init__(
        self,
        calibration_dir: str | Path | None = None,
        calibration_dirs: Mapping[str, str | Path] | None = None,
        lightgbm_model_dir: str | Path | None = None,
        lightgbm_model_dirs: Mapping[str, str | Path] | None = None,
        rng: Optional[np.random.Generator] = None,
        default_ttl_sec: float = 600.0,
    ):
        self.calibration_dirs = self._resolve_calibration_dirs(
            calibration_dir=calibration_dir,
            calibration_dirs=calibration_dirs,
        )
        self.calibration_dir = self.calibration_dirs.get("*")
        if self.calibration_dir is None and len(self.calibration_dirs) == 1:
            self.calibration_dir = next(iter(self.calibration_dirs.values()))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.default_ttl_sec = float(default_ttl_sec)
        self._warm_profiles = self._load_profile("warm_profile.csv")
        self._cold_profiles = self._load_profile("cold_profile.csv")
        self._burst_profiles = self._load_profile("burst_profile.csv")
        self._ttl_lookup = self._build_ttl_lookup()
        self._service_surrogates = self._build_service_surrogates(
            lightgbm_model_dir=lightgbm_model_dir,
            lightgbm_model_dirs=lightgbm_model_dirs,
        )
        self._service_surrogate = self._service_surrogates.get("*")
        if self._service_surrogate is None:
            self._service_surrogate = next(
                iter(self._service_surrogates.values()),
                LightGBMServiceSurrogate(None),
            )

    def _resolve_calibration_dirs(
        self,
        calibration_dir: str | Path | None,
        calibration_dirs: Mapping[str, str | Path] | None,
    ) -> Dict[str, Path]:
        """Normalize calibration directory inputs to a stable mapping."""
        if calibration_dirs:
            return {
                str(architecture): Path(path)
                for architecture, path in calibration_dirs.items()
                if path
            }
        if calibration_dir is None:
            return {}
        return {"*": Path(calibration_dir)}

    def _build_service_surrogates(
        self,
        lightgbm_model_dir: str | Path | None,
        lightgbm_model_dirs: Mapping[str, str | Path] | None,
    ) -> Dict[str, LightGBMServiceSurrogate]:
        """Build architecture-aware LightGBM surrogate loaders."""
        if lightgbm_model_dirs:
            surrogates: Dict[str, LightGBMServiceSurrogate] = {}
            for architecture, model_dir in lightgbm_model_dirs.items():
                surrogate = LightGBMServiceSurrogate(model_dir)
                if surrogate.is_ready:
                    surrogates[str(architecture)] = surrogate
            return surrogates

        if lightgbm_model_dir is not None:
            surrogate = LightGBMServiceSurrogate(lightgbm_model_dir)
            return {"*": surrogate} if surrogate.is_ready else {}

        surrogates = {}
        for architecture, calibration_root in self.calibration_dirs.items():
            default_model_dir = calibration_root / "lightgbm_service"
            if not default_model_dir.exists():
                continue
            surrogate = LightGBMServiceSurrogate(default_model_dir)
            if surrogate.is_ready:
                surrogates[architecture] = surrogate
        return surrogates

    def _load_profile(self, filename: str) -> Optional[pd.DataFrame]:
        """Load one calibration table if it exists."""
        if not self.calibration_dirs:
            return None

        frames = []
        for architecture, calibration_root in self.calibration_dirs.items():
            path = calibration_root / filename
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if "architecture" not in frame.columns and architecture != "*":
                frame["architecture"] = architecture
            frame["calibration_source_dir"] = str(calibration_root)
            frames.append(frame)

        if not frames:
            return None
        return pd.concat(frames, ignore_index=True, sort=False)

    def _select_service_surrogate(
        self,
        architecture: str,
    ) -> LightGBMServiceSurrogate | None:
        """Return the most appropriate surrogate for one architecture."""
        candidate = self._service_surrogates.get(architecture)
        if candidate is not None and candidate.is_ready:
            return candidate
        candidate = self._service_surrogates.get("*")
        if candidate is not None and candidate.is_ready:
            return candidate
        for candidate in self._service_surrogates.values():
            if candidate.is_ready:
                return candidate
        return None

    def _filter_usable_profiles(self, profiles: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Drop failure-only profile rows before latency sampling."""
        if profiles is None or profiles.empty:
            return profiles
        usable = profiles.copy()
        if "successful_invocations" in usable.columns:
            successful = pd.to_numeric(
                usable["successful_invocations"],
                errors="coerce",
            ).fillna(0)
            usable = usable[successful > 0]
        return usable

    def _coerce_profile_success_rate(self, profile: pd.Series) -> float:
        """Read one profile success-rate field with NaN-safe numeric coercion."""
        numeric = pd.to_numeric(
            pd.Series([profile.get("success_rate", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(numeric):
            return 0.0
        return float(numeric)

    def _coerce_profile_successful_invocations(self, profile: pd.Series) -> int:
        """Read one profile successful-invocation count with a conservative fallback."""
        if "successful_invocations" not in profile.index:
            return int(self._coerce_profile_success_rate(profile) > 0.0)
        numeric = pd.to_numeric(
            pd.Series([profile.get("successful_invocations", np.nan)]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(numeric):
            return 0
        return max(0, int(round(float(numeric))))

    def _build_feasibility_hint(
        self,
        profile: pd.Series,
        source: str,
        matched_concurrency: float | None = None,
    ) -> ExactFeasibilityHint:
        """Convert one exact calibration row into a feasibility hint."""
        profile_source = str(profile.get("profile_source", "unknown"))
        return ExactFeasibilityHint(
            source=source,
            success_rate=self._coerce_profile_success_rate(profile),
            successful_invocations=self._coerce_profile_successful_invocations(profile),
            profile_source=profile_source,
            matched_concurrency=matched_concurrency,
        )

    def get_exact_warm_feasibility(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> Optional[ExactFeasibilityHint]:
        """Return exact warm-profile feasibility evidence when available."""
        profile = self._lookup_exact_warm_profile_raw(
            benchmark=benchmark,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            architecture=architecture,
        )
        if profile is None:
            return None
        return self._build_feasibility_hint(profile=profile, source="warm")

    def get_exact_burst_feasibility(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
        arrival_rate_per_sec: float,
    ) -> Optional[ExactFeasibilityHint]:
        """Return exact burst-profile feasibility evidence for the current load."""
        if arrival_rate_per_sec <= 0.0:
            return None
        reference_mean_ms = self._estimate_reference_warm_mean_ms(
            benchmark=benchmark,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            architecture=architecture,
        )
        expected_concurrency = max(
            1.0,
            arrival_rate_per_sec * max(reference_mean_ms, 1.0) / 1000.0,
        )
        profile = self._lookup_exact_burst_profile_raw(
            benchmark=benchmark,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            expected_concurrency=expected_concurrency,
            architecture=architecture,
        )
        if profile is None:
            return None
        matched_concurrency = float(profile.get("concurrency", expected_concurrency))
        return self._build_feasibility_hint(
            profile=profile,
            source="burst",
            matched_concurrency=matched_concurrency,
        )

    def _build_ttl_lookup(self) -> Dict[tuple[str, str, int, int], float]:
        """Infer a TTL threshold from cold-profile measurements."""
        ttl_lookup: Dict[tuple[str, str, int, int], float] = {}
        cold_profiles = self._filter_usable_profiles(self._cold_profiles)
        if cold_profiles is None or cold_profiles.empty:
            return ttl_lookup

        group_keys = ["benchmark", "memory_mb", "timeout_sec"]
        if "architecture" in cold_profiles.columns:
            group_keys.insert(1, "architecture")

        for _, full_group in cold_profiles.groupby(group_keys):
            group = full_group[full_group["idle_gap_sec"] >= 0].sort_values("idle_gap_sec")
            positive_gap_group = group[group["idle_gap_sec"] > 0]
            threshold = None
            for row in positive_gap_group.itertuples(index=False):
                if getattr(row, "cold_probability", 0.0) >= 0.5:
                    threshold = float(row.idle_gap_sec)
                    break
            if threshold is None:
                if positive_gap_group.empty:
                    threshold = self.default_ttl_sec
                else:
                    max_observed_gap = float(positive_gap_group["idle_gap_sec"].max())
                    threshold = max(self.default_ttl_sec, max_observed_gap + 1.0)
            architecture = (
                str(full_group.iloc[0]["architecture"])
                if "architecture" in full_group.columns
                else "*"
            )
            key = (
                str(full_group.iloc[0]["benchmark"]),
                architecture,
                int(full_group.iloc[0]["memory_mb"]),
                int(full_group.iloc[0]["timeout_sec"]),
            )
            ttl_lookup[key] = threshold
        return ttl_lookup

    def estimate_ttl_sec(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str | None = None,
    ) -> float:
        """Estimate container TTL for one benchmark/configuration tuple."""
        arch_key = architecture if architecture is not None else "*"
        specific_key = (benchmark, arch_key, int(memory_mb), int(timeout_sec))
        fallback_key = (benchmark, "*", int(memory_mb), int(timeout_sec))
        if specific_key in self._ttl_lookup:
            return float(self._ttl_lookup[specific_key])
        if fallback_key in self._ttl_lookup:
            return float(self._ttl_lookup[fallback_key])

        nearest_ttl = self._lookup_nearest_ttl_sec(
            benchmark=benchmark,
            memory_mb=int(memory_mb),
            timeout_sec=int(timeout_sec),
            architecture=architecture,
        )
        if nearest_ttl is not None:
            return float(nearest_ttl)
        return float(self.default_ttl_sec)

    def _lookup_nearest_ttl_sec(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str | None,
    ) -> Optional[float]:
        """Find the closest calibrated TTL when an exact key is unavailable."""
        if not self._ttl_lookup:
            return None

        arch_key = architecture if architecture is not None else "*"
        candidates: list[tuple[float, float]] = []
        for (bench_key, arch_value, mem_value, timeout_value), ttl_sec in self._ttl_lookup.items():
            if bench_key != benchmark:
                continue
            if architecture is not None and arch_value not in {arch_key, "*"}:
                continue
            distance = abs(mem_value - memory_mb) + 0.1 * abs(timeout_value - timeout_sec)
            if arch_value == "*":
                distance += 0.01
            candidates.append((distance, float(ttl_sec)))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def sample_warm_runtime_ms(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
        arrival_rate_per_sec: float | None = None,
    ) -> float:
        """Sample one warm runtime in milliseconds."""
        profile = self._lookup_exact_warm_profile(benchmark, memory_mb, timeout_sec, architecture)
        if profile is not None:
            sample = self._sample_from_percentiles(
                p50_ms=float(profile["p50_latency_ms"]),
                p95_ms=float(profile["p95_latency_ms"]),
                mean_ms=float(profile.get("mean_client_latency_ms", profile["p50_latency_ms"])),
            )
        else:
            surrogate = self._select_service_surrogate(architecture)
            surrogate_prediction = None
            if surrogate is not None:
                surrogate_prediction = surrogate.predict_warm(
                    benchmark=benchmark,
                    memory_mb=memory_mb,
                    timeout_sec=timeout_sec,
                    architecture=architecture,
                )
            if surrogate_prediction is not None:
                mean_ms, p95_ms = surrogate_prediction
                sample = self._sample_from_percentiles(
                    p50_ms=max(mean_ms * 0.9, 1.0),
                    p95_ms=max(p95_ms, mean_ms),
                    mean_ms=mean_ms,
                )
            else:
                profile = self._lookup_warm_profile(benchmark, memory_mb, timeout_sec, architecture)
                if profile is not None:
                    sample = self._sample_from_percentiles(
                        p50_ms=float(profile["p50_latency_ms"]),
                        p95_ms=float(profile["p95_latency_ms"]),
                        mean_ms=float(profile.get("mean_client_latency_ms", profile["p50_latency_ms"])),
                    )
                else:
                    prefix = benchmark[:3]
                    base_latency_ms = DEFAULT_BASE_LATENCIES_MS.get(prefix, 500.0)
                    memory_factor = (128.0 / float(memory_mb)) ** 0.5
                    arch_factor = 0.95 if architecture == "arm64" else 1.0
                    noise = self.rng.uniform(0.85, 1.15)
                    sample = base_latency_ms * memory_factor * arch_factor * noise

        if arrival_rate_per_sec is not None and arrival_rate_per_sec > 0.0:
            sample = self._apply_burst_adjustment(
                runtime_ms=float(sample),
                benchmark=benchmark,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                architecture=architecture,
                arrival_rate_per_sec=float(arrival_rate_per_sec),
            )
        return float(max(1.0, min(sample, timeout_sec * 1000.0)))

    def sample_cold_overhead_ms(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
        idle_gap_sec: float = 0.0,
    ) -> float:
        """Sample one cold-start overhead in milliseconds."""
        profile = self._lookup_exact_enforced_cold_profile(
            benchmark,
            memory_mb,
            timeout_sec,
            architecture,
        )
        if profile is None:
            profile = self._lookup_exact_cold_profile(
                benchmark,
                memory_mb,
                timeout_sec,
                idle_gap_sec,
                architecture,
            )
        if profile is not None and not math.isnan(
            float(profile.get("estimated_cold_overhead_ms", math.nan))
        ):
            mean_ms = max(1.0, float(profile["estimated_cold_overhead_ms"]))
            p95_ms = float(profile.get("estimated_p95_cold_overhead_ms", mean_ms * 1.5))
            sample = self._sample_from_percentiles(
                p50_ms=max(mean_ms * 0.85, 1.0),
                p95_ms=max(p95_ms, mean_ms),
                mean_ms=mean_ms,
            )
            return float(max(1.0, sample))

        surrogate = self._select_service_surrogate(architecture)
        surrogate_prediction = None
        if surrogate is not None:
            surrogate_prediction = surrogate.predict_cold_overhead(
                benchmark=benchmark,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                architecture=architecture,
            )
        if surrogate_prediction is not None:
            mean_ms, p95_ms = surrogate_prediction
            sample = self._sample_from_percentiles(
                p50_ms=max(mean_ms * 0.85, 1.0),
                p95_ms=max(p95_ms, mean_ms),
                mean_ms=mean_ms,
            )
            return float(max(1.0, sample))

        profile = self._lookup_enforced_cold_profile(
            benchmark,
            memory_mb,
            timeout_sec,
            architecture,
        )
        if profile is None:
            profile = self._lookup_cold_profile(
                benchmark,
                memory_mb,
                timeout_sec,
                idle_gap_sec,
                architecture,
            )
        if profile is not None and not math.isnan(
            float(profile.get("estimated_cold_overhead_ms", math.nan))
        ):
            mean_ms = max(1.0, float(profile["estimated_cold_overhead_ms"]))
            p95_ms = float(profile.get("estimated_p95_cold_overhead_ms", mean_ms * 1.5))
            sample = self._sample_from_percentiles(
                p50_ms=max(mean_ms * 0.85, 1.0),
                p95_ms=max(p95_ms, mean_ms),
                mean_ms=mean_ms,
            )
            return float(max(1.0, sample))

        return float(self.rng.uniform(500.0, 1500.0))

    def _lookup_exact_warm_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find one exact warm profile row."""
        profile = self._lookup_exact_warm_profile_raw(
            benchmark=benchmark,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            architecture=architecture,
        )
        if profile is None:
            return None
        if self._coerce_profile_successful_invocations(profile) <= 0:
            return None
        return profile

    def _lookup_exact_warm_profile_raw(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find one exact warm profile row without dropping failure-only evidence."""
        if self._warm_profiles is None or self._warm_profiles.empty:
            return None
        group = self._warm_profiles[
            (self._warm_profiles["benchmark"] == benchmark)
            & (self._warm_profiles["memory_mb"] == memory_mb)
            & (self._warm_profiles["timeout_sec"] == timeout_sec)
        ]
        group = self._filter_architecture(
            group,
            architecture,
            allow_fallback=False,
        )
        if group.empty:
            return None
        return group.iloc[0]

    def _lookup_warm_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find the closest warm profile row."""
        if self._warm_profiles is None or self._warm_profiles.empty:
            return None
        group = self._warm_profiles[self._warm_profiles["benchmark"] == benchmark]
        group = self._filter_architecture(group, architecture)
        group = self._filter_usable_profiles(group)
        if group.empty:
            return None
        distances = (
            (group["memory_mb"] - memory_mb).abs()
            + 0.1 * (group["timeout_sec"] - timeout_sec).abs()
        )
        return group.iloc[int(distances.argmin())]

    def _lookup_exact_burst_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        expected_concurrency: float,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find one burst row for an exact config and nearest available concurrency."""
        profile = self._lookup_exact_burst_profile_raw(
            benchmark=benchmark,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            expected_concurrency=expected_concurrency,
            architecture=architecture,
        )
        if profile is None:
            return None
        if self._coerce_profile_successful_invocations(profile) <= 0:
            return None
        return profile

    def _lookup_exact_burst_profile_raw(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        expected_concurrency: float,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find one exact burst row without dropping failure-only evidence."""
        if self._burst_profiles is None or self._burst_profiles.empty:
            return None
        group = self._burst_profiles[
            (self._burst_profiles["benchmark"] == benchmark)
            & (self._burst_profiles["memory_mb"] == memory_mb)
            & (self._burst_profiles["timeout_sec"] == timeout_sec)
        ]
        group = self._filter_architecture(
            group,
            architecture,
            allow_fallback=False,
        )
        if group.empty:
            return None
        distances = (group["concurrency"] - expected_concurrency).abs()
        return group.iloc[int(distances.argmin())]

    def _lookup_burst_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        expected_concurrency: float,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find the closest burst-profile row for the expected concurrency."""
        if self._burst_profiles is None or self._burst_profiles.empty:
            return None
        group = self._burst_profiles[self._burst_profiles["benchmark"] == benchmark]
        group = self._filter_architecture(group, architecture)
        group = self._filter_usable_profiles(group)
        if group.empty:
            return None
        distances = (
            (group["memory_mb"] - memory_mb).abs()
            + 0.1 * (group["timeout_sec"] - timeout_sec).abs()
            + 0.5 * (group["concurrency"] - expected_concurrency).abs()
        )
        return group.iloc[int(distances.argmin())]

    def _lookup_cold_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        idle_gap_sec: float,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find the closest cold-profile row."""
        if self._cold_profiles is None or self._cold_profiles.empty:
            return None
        group = self._cold_profiles[self._cold_profiles["benchmark"] == benchmark]
        group = self._filter_architecture(group, architecture)
        group = self._filter_usable_profiles(group)
        if group.empty:
            return None
        distances = (
            (group["memory_mb"] - memory_mb).abs()
            + 0.1 * (group["timeout_sec"] - timeout_sec).abs()
            + 0.01 * (group["idle_gap_sec"] - idle_gap_sec).abs()
        )
        return group.iloc[int(distances.argmin())]

    def _lookup_exact_cold_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        idle_gap_sec: float,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find one exact cold-profile row."""
        if self._cold_profiles is None or self._cold_profiles.empty:
            return None
        group = self._cold_profiles[
            (self._cold_profiles["benchmark"] == benchmark)
            & (self._cold_profiles["memory_mb"] == memory_mb)
            & (self._cold_profiles["timeout_sec"] == timeout_sec)
            & (self._cold_profiles["idle_gap_sec"] == idle_gap_sec)
        ]
        group = self._filter_architecture(group, architecture)
        group = self._filter_usable_profiles(group)
        if group.empty:
            return None
        return group.iloc[0]

    def _lookup_enforced_cold_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find the closest enforced cold-start row.

        The collector writes enforced cold starts with ``idle_gap_sec = -1``.
        These rows represent true cold-start initialization overhead and should
        be preferred over ordinary idle-gap rows when sampling cold latency.
        """
        if self._cold_profiles is None or self._cold_profiles.empty:
            return None
        group = self._cold_profiles[
            (self._cold_profiles["benchmark"] == benchmark)
            & (self._cold_profiles["idle_gap_sec"] < 0)
        ]
        group = self._filter_architecture(group, architecture)
        group = self._filter_usable_profiles(group)
        if group.empty:
            return None
        distances = (
            (group["memory_mb"] - memory_mb).abs()
            + 0.1 * (group["timeout_sec"] - timeout_sec).abs()
        )
        return group.iloc[int(distances.argmin())]

    def _lookup_exact_enforced_cold_profile(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> Optional[pd.Series]:
        """Find one exact enforced cold row."""
        if self._cold_profiles is None or self._cold_profiles.empty:
            return None
        group = self._cold_profiles[
            (self._cold_profiles["benchmark"] == benchmark)
            & (self._cold_profiles["memory_mb"] == memory_mb)
            & (self._cold_profiles["timeout_sec"] == timeout_sec)
            & (self._cold_profiles["idle_gap_sec"] < 0)
        ]
        group = self._filter_architecture(group, architecture)
        group = self._filter_usable_profiles(group)
        if group.empty:
            return None
        return group.iloc[0]

    def _sample_from_percentiles(
        self,
        p50_ms: float,
        p95_ms: float,
        mean_ms: float,
    ) -> float:
        """Sample from a log-normal distribution fitted from percentiles."""
        p50_ms = max(p50_ms, 1.0)
        p95_ms = max(p95_ms, p50_ms)
        mean_ms = max(mean_ms, p50_ms)

        mu = math.log(p50_ms)
        sigma = max((math.log(p95_ms) - mu) / 1.64485, 1e-3)
        sample = float(self.rng.lognormal(mean=mu, sigma=sigma))
        return float(max(1.0, 0.6 * sample + 0.4 * mean_ms))

    def _filter_architecture(
        self,
        group: pd.DataFrame,
        architecture: str,
        allow_fallback: bool = True,
    ) -> pd.DataFrame:
        """Filter a calibration table by architecture when the column exists."""
        if "architecture" not in group.columns:
            return group
        filtered = group[group["architecture"] == architecture]
        if not filtered.empty or not allow_fallback:
            return filtered
        return group

    def _estimate_reference_warm_mean_ms(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> float:
        """Estimate the mean warm runtime without burst pressure."""
        profile = self._lookup_exact_warm_profile(benchmark, memory_mb, timeout_sec, architecture)
        if profile is not None:
            return float(
                profile.get("mean_client_latency_ms", profile.get("p50_latency_ms", 1.0))
            )

        surrogate = self._select_service_surrogate(architecture)
        surrogate_prediction = None
        if surrogate is not None:
            surrogate_prediction = surrogate.predict_warm(
                benchmark=benchmark,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                architecture=architecture,
            )
        if surrogate_prediction is not None:
            return float(surrogate_prediction[0])

        profile = self._lookup_warm_profile(benchmark, memory_mb, timeout_sec, architecture)
        if profile is not None:
            return float(
                profile.get("mean_client_latency_ms", profile.get("p50_latency_ms", 1.0))
            )

        prefix = benchmark[:3]
        base_latency_ms = DEFAULT_BASE_LATENCIES_MS.get(prefix, 500.0)
        memory_factor = (128.0 / float(memory_mb)) ** 0.5
        arch_factor = 0.95 if architecture == "arm64" else 1.0
        return float(base_latency_ms * memory_factor * arch_factor)

    def _apply_burst_adjustment(
        self,
        runtime_ms: float,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
        arrival_rate_per_sec: float,
    ) -> float:
        """Adjust one warm runtime sample with concurrency-sensitive slowdown."""
        reference_mean_ms = self._estimate_reference_warm_mean_ms(
            benchmark=benchmark,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            architecture=architecture,
        )
        expected_concurrency = max(
            1.0,
            arrival_rate_per_sec * max(reference_mean_ms, 1.0) / 1000.0,
        )
        burst_profile = self._lookup_exact_burst_profile(
            benchmark=benchmark,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            expected_concurrency=expected_concurrency,
            architecture=architecture,
        )

        if burst_profile is None:
            surrogate = self._select_service_surrogate(architecture)
            surrogate_prediction = None
            if surrogate is not None:
                surrogate_prediction = surrogate.predict_burst_slowdown(
                    benchmark=benchmark,
                    memory_mb=memory_mb,
                    timeout_sec=timeout_sec,
                    architecture=architecture,
                    expected_concurrency=expected_concurrency,
                )
            if surrogate_prediction is not None:
                mean_multiplier, tail_multiplier = surrogate_prediction
                sigma = max(
                    0.03,
                    min(0.45, (math.log(tail_multiplier) - math.log(mean_multiplier)) / 1.64485),
                )
                slowdown = float(self.rng.lognormal(mean=math.log(mean_multiplier), sigma=sigma))
                return float(runtime_ms * slowdown)

            burst_profile = self._lookup_burst_profile(
                benchmark=benchmark,
                memory_mb=memory_mb,
                timeout_sec=timeout_sec,
                expected_concurrency=expected_concurrency,
                architecture=architecture,
            )
            if burst_profile is None:
                heuristic_multiplier = 1.0 + min(1.2, max(0.0, expected_concurrency - 1.0) * 0.03)
                return float(runtime_ms * heuristic_multiplier)

        matched_concurrency = float(burst_profile.get("concurrency", expected_concurrency))
        if expected_concurrency < max(1.5, 0.5 * matched_concurrency):
            heuristic_multiplier = 1.0 + min(0.3, max(0.0, expected_concurrency - 1.0) * 0.03)
            return float(runtime_ms * heuristic_multiplier)

        warm_profile = self._lookup_exact_warm_profile(benchmark, memory_mb, timeout_sec, architecture)
        if warm_profile is not None:
            warm_mean = float(
                warm_profile.get("mean_client_latency_ms", warm_profile["p50_latency_ms"])
            )
            warm_p95 = float(warm_profile.get("p95_latency_ms", warm_mean))
        else:
            warm_mean = reference_mean_ms
            warm_p95 = reference_mean_ms * 1.2

        mean_multiplier = float(
            burst_profile.get(
                "latency_slowdown_mean",
                float(burst_profile["mean_client_latency_ms"]) / max(warm_mean, 1.0),
            )
        )
        tail_multiplier = float(
            burst_profile.get(
                "latency_slowdown_p95",
                float(burst_profile.get("p95_latency_ms", burst_profile["mean_client_latency_ms"]))
                / max(warm_p95, 1.0),
            )
        )

        mean_multiplier = float(np.clip(mean_multiplier, 1.0, 4.0))
        tail_multiplier = float(np.clip(tail_multiplier, mean_multiplier, 6.0))

        sigma = max(
            0.03,
            min(0.45, (math.log(tail_multiplier) - math.log(mean_multiplier)) / 1.64485),
        )
        slowdown = float(self.rng.lognormal(mean=math.log(mean_multiplier), sigma=sigma))
        return float(runtime_ms * slowdown)
