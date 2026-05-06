"""LightGBM-backed service surrogate for CALO simulator backends."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


class LightGBMServiceSurrogate:
    """Load and query fitted LightGBM regressors for service-model fallback."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.metadata: Dict[str, Any] = {}
        self.warm_mean_model = None
        self.warm_p95_model = None
        self.cold_mean_model = None
        self.cold_p95_model = None
        self.burst_mean_model = None
        self.burst_p95_model = None

        if self.model_dir is not None:
            self._load()

    @property
    def is_ready(self) -> bool:
        """Return whether at least one surrogate model is available."""
        return any(
            model is not None
            for model in [
                self.warm_mean_model,
                self.warm_p95_model,
                self.cold_mean_model,
                self.cold_p95_model,
                self.burst_mean_model,
                self.burst_p95_model,
            ]
        )

    def _load(self) -> None:
        """Load trained model artifacts from disk."""
        assert self.model_dir is not None
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.warm_mean_model = self._safe_load("warm_mean_model.joblib")
        self.warm_p95_model = self._safe_load("warm_p95_model.joblib")
        self.cold_mean_model = self._safe_load("cold_overhead_mean_model.joblib")
        self.cold_p95_model = self._safe_load("cold_overhead_p95_model.joblib")
        self.burst_mean_model = self._safe_load("burst_slowdown_mean_model.joblib")
        self.burst_p95_model = self._safe_load("burst_slowdown_p95_model.joblib")

    def _safe_load(self, filename: str):
        """Load one joblib artifact if it exists."""
        assert self.model_dir is not None
        path = self.model_dir / filename
        if not path.exists():
            return None
        return joblib.load(path)

    def _predict_scalar(self, model: Any, payload: Dict[str, Any], exp_transform: bool) -> float | None:
        """Predict one scalar value from a joblib-loaded sklearn pipeline."""
        if model is None:
            return None
        frame = pd.DataFrame([payload])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                category=UserWarning,
            )
            prediction = float(np.asarray(model.predict(frame), dtype=float).reshape(-1)[0])
        if exp_transform:
            prediction = math.expm1(prediction)
        return float(max(prediction, 1e-6))

    def predict_warm(self, benchmark: str, memory_mb: int, timeout_sec: int, architecture: str) -> tuple[float, float] | None:
        """Predict warm mean and p95 latency in milliseconds."""
        payload = {
            "benchmark": benchmark,
            "architecture": architecture,
            "memory_mb": int(memory_mb),
            "timeout_sec": int(timeout_sec),
        }
        mean_ms = self._predict_scalar(self.warm_mean_model, payload, exp_transform=True)
        p95_ms = self._predict_scalar(self.warm_p95_model, payload, exp_transform=True)
        if mean_ms is None or p95_ms is None:
            return None
        return float(mean_ms), float(max(p95_ms, mean_ms))

    def predict_cold_overhead(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
    ) -> tuple[float, float] | None:
        """Predict cold-start overhead mean and p95 in milliseconds."""
        payload = {
            "benchmark": benchmark,
            "architecture": architecture,
            "memory_mb": int(memory_mb),
            "timeout_sec": int(timeout_sec),
        }
        mean_ms = self._predict_scalar(self.cold_mean_model, payload, exp_transform=True)
        p95_ms = self._predict_scalar(self.cold_p95_model, payload, exp_transform=True)
        if mean_ms is None or p95_ms is None:
            return None
        return float(mean_ms), float(max(p95_ms, mean_ms))

    def predict_burst_slowdown(
        self,
        benchmark: str,
        memory_mb: int,
        timeout_sec: int,
        architecture: str,
        expected_concurrency: float,
    ) -> tuple[float, float] | None:
        """Predict burst slowdown mean and p95 multipliers."""
        payload = {
            "benchmark": benchmark,
            "architecture": architecture,
            "memory_mb": int(memory_mb),
            "timeout_sec": int(timeout_sec),
            "concurrency": float(expected_concurrency),
        }
        mean_scale = self._predict_scalar(self.burst_mean_model, payload, exp_transform=True)
        p95_scale = self._predict_scalar(self.burst_p95_model, payload, exp_transform=True)
        if mean_scale is None or p95_scale is None:
            return None
        mean_scale = float(np.clip(mean_scale, 1.0, 10.0))
        p95_scale = float(np.clip(max(p95_scale, mean_scale), mean_scale, 12.0))
        return mean_scale, p95_scale
