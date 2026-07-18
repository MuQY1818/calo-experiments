"""Pure semantic checks for the accepted-paper CALO evidence.

This module accepts a :class:`ManifestInventory`, never a filesystem root.  A
claim source therefore cannot escape the verified manifest through a glob,
``Path.read_text`` call, or an unlisted sidecar file.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import statistics
from typing import Any, Dict, Iterable, Mapping

from .artifact_verifier import (
    ArtifactVerificationError,
    ManifestInventory,
    _require,
    canonical_json_bytes,
)


BENCHMARKS = (
    "110.dynamic-html",
    "120.uploader",
    "210.thumbnailer",
    "311.compression",
    "411.image-recognition",
)
WORKLOADS = ("sine", "spike", "decay", "random")
ATTRIBUTION_GROUPS = {"code": 32, "load": 10, "category": 5, "history": 5, "context": 27}
ATTRIBUTION_FILTERED_GROUPS = {"load", "history"}
ATTRIBUTION_EXCLUDED_FEATURES = {"day_of_week", "hour_of_day"}
SUPPLEMENTARY_PATHS = {
    "supplementary/embedding_16/aggregate_summary.json",
    "supplementary/embedding_32/aggregate_summary.json",
    "supplementary/embedding_64/aggregate_summary.json",
    "supplementary/feature_attribution/aggregate_summary.json",
    "supplementary/load_only/aggregate_summary.json",
    "supplementary/reward_sensitivity_default/aggregate_summary.json",
    "supplementary/reward_sensitivity_heavy/aggregate_summary.json",
    "supplementary/reward_sensitivity_light/aggregate_summary.json",
    "supplementary/reward_sensitivity_weight5050/aggregate_summary.json",
}
REPLAY_POLICY_NAMES = {"ppo", "default", "bayes_opt_online"}
REPLAY_STEP_KEYS = {
    "arrival_count",
    "step_duration_sec",
    "source_name",
    "load_value",
    "minute_of_day",
    "profile_key",
    "mean_invocations_per_minute",
    "std_invocations_per_minute",
    "max_invocations_per_minute",
    "burstiness_hint",
}
FIXED_MEMORY = (128, 256, 512, 1024, 2048, 3008)
FIXED_TIMEOUT = (60, 120, 300, 900)


def _close(actual: float, expected: float, tolerance: float, label: str) -> None:
    """Compare finite paper values with an absolute tolerance."""
    _require(math.isfinite(actual), f"Non-finite computed value: {label}")
    _require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{label} mismatch: expected {expected:.12g}, computed {actual:.12g}",
    )


def _number(value: Any, label: str) -> float:
    """Return a finite numeric value, rejecting bool and null."""
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _object(value: Any, label: str) -> dict[str, Any]:
    """Return a JSON object or raise a stable artifact error."""
    if not isinstance(value, dict):
        raise ArtifactVerificationError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    """Return a JSON array or raise a stable artifact error."""
    if not isinstance(value, list):
        raise ArtifactVerificationError(f"{label} must be an array")
    return value


def _source(inventory: ManifestInventory, path: str) -> dict[str, Any]:
    """Read one explicitly listed JSON source."""
    return _object(inventory.require_json(path), path)


def _digest(value: Any) -> str:
    """Hash a canonical JSON value."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _algorithm_cases(data: Mapping[str, Any], algorithm: str) -> list[Mapping[str, Any]]:
    """Flatten one algorithm's benchmark/workload matrix."""
    raw = _object(data.get(algorithm), f"algorithm results: {algorithm}")
    _require(set(raw) == set(BENCHMARKS), f"Unexpected benchmark set for {algorithm}")
    cases: list[Mapping[str, Any]] = []
    for benchmark in BENCHMARKS:
        loads = _object(raw[benchmark], f"{algorithm}/{benchmark}")
        _require(set(loads) == set(WORKLOADS), f"Malformed {algorithm} matrix")
        for workload in WORKLOADS:
            case = _object(loads[workload], f"{algorithm}/{benchmark}/{workload}")
            cases.append(case)
    return cases


def _algorithm_metrics(
    cases: Iterable[Mapping[str, Any]], tail_fraction: float
) -> Dict[str, float]:
    """Compute reward, latency, cost, and lower-tail metrics."""
    case_list = list(cases)
    _require(bool(case_list), "Cannot aggregate an empty algorithm result")
    rewards = [float(_number(case["mean_reward"], "mean_reward")) for case in case_list]
    tail_count = max(1, int(len(rewards) * tail_fraction))
    return {
        "mean_reward": statistics.fmean(rewards),
        "median_reward": statistics.median(rewards),
        "cvar10_reward": statistics.fmean(sorted(rewards)[:tail_count]),
        "mean_latency_ms": statistics.fmean(
            float(_number(case["mean_latency"], "mean_latency")) for case in case_list
        ),
        "mean_cost_usd": statistics.fmean(
            float(_number(case["mean_cost"], "mean_cost")) for case in case_list
        ),
    }


def _verify_model_contract(claims: Mapping[str, Any]) -> Dict[str, int]:
    """Recompute the frozen 79-dimensional, 48-action contract."""
    contract = _object(claims["model_contract"], "model_contract")
    components = _object(contract["state_components"], "state_components")
    state_dimension = sum(
        int(_number(value, f"state component {name}")) for name, value in components.items()
    )
    catalog = _object(contract["action_catalog"], "action_catalog")
    _require(bool(catalog), "action_catalog must be non-empty")
    action_count = math.prod(len(options) for options in catalog.values())
    _require(
        state_dimension == int(contract["expected_state_dimension"]), "State dimension mismatch"
    )
    _require(action_count == int(contract["expected_action_count"]), "Action count mismatch")
    _require(state_dimension == 79 and action_count == 48, "Published CALO contract is not 79/48")
    return {"state_dimension": state_dimension, "action_count": action_count}


def _verify_main_results(
    inventory: ManifestInventory,
    claim: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    """Recompute the main five-benchmark reward/latency/cost claims."""
    results = _source(inventory, claim["source"])
    computed: Dict[str, Dict[str, float]] = {}
    for algorithm, expected in claim["algorithms"].items():
        cases = _algorithm_cases(results, algorithm)
        _require(
            len(cases) == int(claim["case_count_per_algorithm"]),
            f"Unexpected {algorithm} case count",
        )
        metrics = _algorithm_metrics(cases, float(claim["tail_fraction"]))
        for metric, value in expected.items():
            _close(metrics[metric], float(value), tolerance, f"{algorithm}.{metric}")
        computed[algorithm] = metrics
    deltas = []
    for seed_claim in claim["seed_reward_deltas"]:
        seed_results = _source(inventory, seed_claim["source"])
        ppo = statistics.fmean(
            float(_number(case["mean_reward"], "mean_reward"))
            for case in _algorithm_cases(seed_results, "ppo")
        )
        online = statistics.fmean(
            float(_number(case["mean_reward"], "mean_reward"))
            for case in _algorithm_cases(seed_results, "bayes_opt_online")
        )
        delta = ppo - online
        _close(
            delta,
            float(seed_claim["expected"]),
            tolerance,
            f"seed {seed_claim['seed']} reward delta",
        )
        deltas.append(delta)
    return {"algorithms": computed, "seed_reward_deltas": deltas}


def _csv_rows(payload: bytes, path: str) -> list[dict[str, str]]:
    """Decode one manifest-listed CSV payload."""
    try:
        return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ArtifactVerificationError(f"Invalid calibration CSV {path}: {error}") from error


def _verify_calibration(inventory: ManifestInventory) -> Dict[str, Any]:
    """Validate both architecture profile matrices and six-model metadata."""
    details: Dict[str, Any] = {}
    for architecture in ("x64", "arm64"):
        directory = f"calibration/{architecture}"
        metadata = _source(inventory, f"{directory}/metadata.json")
        _require(
            metadata.get("architecture") == architecture,
            f"Calibration architecture mismatch: {architecture}",
        )
        profile_counts: Dict[str, int] = {}
        for profile, profile_metadata in metadata["profiles"].items():
            path = f"{directory}/{profile_metadata['file']}"
            rows = _csv_rows(inventory.require_bytes(path), path)
            _require(
                len(rows) == int(profile_metadata["row_count"]),
                f"Unexpected {architecture} {profile} row count",
            )
            _require(
                all(row.get("architecture") == architecture for row in rows),
                f"Wrong architecture in {path}",
            )
            profile_counts[profile] = len(rows)
        model_dir = f"{directory}/lightgbm_service"
        model_metadata = _source(inventory, f"{model_dir}/metadata.json")
        _require(
            model_metadata.get("architecture") == architecture,
            f"Model architecture mismatch: {architecture}",
        )
        models = _object(model_metadata.get("models"), f"{architecture} surrogate models")
        _require(len(models) == 6, f"Expected six {architecture} surrogate models")
        for model in models.values():
            path = f"{model_dir}/{model['file']}"
            _require(path in inventory.files, f"Missing model: {path}")
        details[architecture] = {"profile_rows": profile_counts, "surrogate_models": len(models)}
    return details


def _verify_fixed_validation(
    inventory: ManifestInventory,
    claim: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    """Validate the fixed 120-target grid, infeasible boundary, and ranking."""
    data = _source(inventory, claim["source"])
    targets = _array(data.get("targets"), "Fixed validation targets")
    expected_grid = {
        (benchmark, memory, timeout)
        for benchmark in BENCHMARKS
        for memory in FIXED_MEMORY
        for timeout in FIXED_TIMEOUT
    }
    actual_grid = {
        (row["benchmark"], int(row["memory_mb"]), int(row["timeout_sec"])) for row in targets
    }
    _require(actual_grid == expected_grid, "Fixed validation grid mismatch")
    _require(len(targets) == int(claim["target_count"]) == 120, "Fixed target count mismatch")
    deployable = [row for row in targets if row.get("sample_collection_complete") is True]
    _require(
        len(deployable) == int(claim["deployable_count"]) == 112, "Fixed deployable count mismatch"
    )
    _require(
        all(row.get("architecture") == "x64" for row in targets),
        "Fixed validation architecture mismatch",
    )
    distribution = {
        benchmark: sum(
            row["sample_collection_complete"] is True
            for row in targets
            if row["benchmark"] == benchmark
        )
        for benchmark in BENCHMARKS
    }
    _require(
        distribution
        == {
            benchmark: (16 if benchmark == "411.image-recognition" else 24)
            for benchmark in BENCHMARKS
        },
        "Deployable benchmark distribution mismatch",
    )
    infeasible = {
        (row["benchmark"], int(row["memory_mb"]), int(row["timeout_sec"]))
        for row in targets
        if row["sample_collection_complete"] is False
    }
    expected_infeasible = {
        ("411.image-recognition", memory, timeout)
        for memory in (128, 256)
        for timeout in FIXED_TIMEOUT
    }
    _require(infeasible == expected_infeasible, "Fixed infeasible boundary mismatch")
    for row in targets:
        complete = row["sample_collection_complete"] is True
        _require(
            row["config_label"] == f"{row['memory_mb']}MB/{row['timeout_sec']}s/x64",
            "Fixed config label mismatch",
        )
        for key in ("sim_warm_mean_ms", "sim_warm_p95_ms", "sim_cold_mean_ms", "sim_cold_p95_ms"):
            _number(row.get(key), f"fixed {key}")
        for key in (
            "real_warm_mean_ms",
            "real_warm_p95_ms",
            "real_cold_mean_ms",
            "real_cold_p95_ms",
            "warm_mean_abs_rel_error_pct",
            "cold_mean_abs_rel_error_pct",
        ):
            if complete:
                _number(row.get(key), f"fixed {key}")
            else:
                _require(row.get(key) is None, f"Infeasible fixed {key} must be null")
    warm_error = statistics.median(
        float(_number(row["warm_mean_abs_rel_error_pct"], "warm error")) for row in deployable
    )
    cold_error = statistics.median(
        float(_number(row["cold_mean_abs_rel_error_pct"], "cold error")) for row in deployable
    )
    _close(
        warm_error,
        float(claim["median_warm_abs_rel_error_pct"]),
        tolerance,
        "fixed median warm error",
    )
    _close(
        cold_error,
        float(claim["median_cold_abs_rel_error_pct"]),
        tolerance,
        "fixed median cold error",
    )
    ranking_path = claim["ranking_source"]
    ranking = _source(inventory, ranking_path)
    expected_digest = claim.get("ranking_sha256")
    if expected_digest:
        _require(
            hashlib.sha256(inventory.require_bytes(ranking_path)).hexdigest() == expected_digest,
            "Fixed ranking digest mismatch",
        )
    rows = _array(ranking.get("benchmark_rows"), "Fixed ranking rows")
    _require(len(rows) == 20, "Fixed ranking row count mismatch")
    _require(
        {(row.get("benchmark"), row.get("metric")) for row in rows}
        == {
            (b, m)
            for b in BENCHMARKS
            for m in ("warm_latency", "cold_latency", "warm_reward", "cold_reward")
        },
        "Fixed ranking key set mismatch",
    )
    for row in rows:
        expected_count = 16 if row["benchmark"] == "411.image-recognition" else 24
        _require(
            int(row["config_count"]) == expected_count
            and int(row["complete_count"]) == expected_count,
            "Fixed ranking coverage mismatch",
        )
        _require(isinstance(row.get("top1_match"), bool), "ranking top1_match must be boolean")
        for key in ("top2_overlap", "spearman"):
            _number(row.get(key), f"ranking {key}")
    overall = _array(ranking.get("overall"), "Fixed ranking overall")
    _require(len(overall) == 4, "Fixed ranking overall count mismatch")
    for actual, expected in zip(overall, claim["ranking_overall"]):
        _require(actual.get("metric") == expected["metric"], "Fixed ranking metric order mismatch")
        for key, value in expected.items():
            if key == "metric":
                continue
            _close(
                float(_number(actual.get(key), f"ranking {key}")),
                float(value),
                tolerance,
                f"ranking {actual['metric']}.{key}",
            )
    return {
        "target_count": len(targets),
        "deployable_count": len(deployable),
        "median_warm_abs_rel_error_pct": warm_error,
        "median_cold_abs_rel_error_pct": cold_error,
        "ranking_rows": len(rows),
    }


def _pairwise(summary: Mapping[str, Any], baseline: str) -> Mapping[str, Any]:
    """Return one ppo-to-baseline replay row."""
    matches = [
        row
        for row in summary.get("pairwise", [])
        if row.get("primary_policy") == "ppo" and row.get("baseline_policy") == baseline
    ]
    _require(len(matches) == 1, f"Missing replay comparison against {baseline}")
    return matches[0]


def _verify_replay_case(
    inventory: ManifestInventory, expected: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate one replay summary, protocol, workload, and content identities."""
    summary = _source(inventory, expected["summary"])
    workload = _source(inventory, expected["workload"])
    _require(
        hashlib.sha256(inventory.require_bytes(expected["summary"])).hexdigest()
        == expected["summary_sha256"],
        "Replay summary identity mismatch",
    )
    _require(
        summary.get("case_id") == expected["case_id"] == workload.get("case_id"),
        "Replay case identity mismatch",
    )
    for document, label in ((summary, "summary"), (workload, "workload")):
        _require(
            document.get("benchmark") == expected["benchmark"], f"Replay {label} benchmark mismatch"
        )
        _require(
            document.get("load_pattern") == expected["load_pattern"],
            f"Replay {label} workload mismatch",
        )
        _require(
            int(_number(document.get("seed"), f"Replay {label} seed")) == int(expected["seed"]),
            f"Replay {label} seed mismatch",
        )
    _require(
        int(_number(summary.get("steps"), "Replay summary steps"))
        == int(expected["steps"])
        == len(workload.get("workload_steps", [])),
        "Replay step count mismatch",
    )
    _require(
        _digest(summary["protocol"]) == expected["protocol_sha256"],
        "Replay protocol identity mismatch",
    )
    _require(_digest(workload) == expected["workload_sha256"], "Replay workload identity mismatch")
    _require(
        set(row.get("policy") for row in summary.get("results", [])) == REPLAY_POLICY_NAMES,
        "Replay policy set mismatch",
    )
    _require(
        {
            (row.get("primary_policy"), row.get("baseline_policy"))
            for row in summary.get("pairwise", [])
        }
        == {("ppo", "default"), ("ppo", "bayes_opt_online")},
        "Replay pairwise set mismatch",
    )
    steps = _array(workload.get("workload_steps"), "Replay workload steps")
    for step in steps:
        _require(set(step) == REPLAY_STEP_KEYS, "Replay workload step schema mismatch")
        _require(
            step["source_name"] == f"synthetic:{expected['load_pattern']}",
            "Replay workload source mismatch",
        )
        for key, value in step.items():
            if key not in {"source_name", "profile_key"}:
                _number(value, f"replay step {key}")
    for row in summary["results"]:
        for key in (
            "mean_raw_reward",
            "mean_latency_ms",
            "mean_p95_latency_ms",
            "mean_cost_usd",
            "mean_success_rate",
            "mean_cold_start_rate",
        ):
            _number(row.get(key), f"replay {key}")
    return summary


def _verify_replay(
    inventory: ManifestInventory,
    claim: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    """Verify the exact eight replay cases and recompute aggregate values."""
    expected_cases = claim["cases"]
    expected_summary_paths = {case["summary"] for case in expected_cases}
    expected_workload_paths = {case["workload"] for case in expected_cases}
    actual_summary_paths = {
        path
        for path in inventory.files
        if path.startswith("replay/") and path.endswith("/summary.json")
    }
    actual_workload_paths = {
        path
        for path in inventory.files
        if path.startswith("replay/") and path.endswith("/workload_sequence.json")
    }
    _require(
        actual_summary_paths == expected_summary_paths
        and actual_workload_paths == expected_workload_paths,
        "Replay manifest case set mismatch",
    )
    summaries = [_verify_replay_case(inventory, case) for case in expected_cases]

    def mean_pairwise(baseline: str, metric: str) -> float:
        return statistics.fmean(
            float(_number(_pairwise(summary, baseline)[metric], f"replay {metric}"))
            for summary in summaries
        )

    computed = {
        "mean_raw_reward_delta_vs_default": mean_pairwise("default", "mean_raw_reward_delta"),
        "mean_raw_reward_delta_vs_online_bo": mean_pairwise(
            "bayes_opt_online", "mean_raw_reward_delta"
        ),
        "steady_state_delta_vs_default": mean_pairwise(
            "default", "steady_state_mean_raw_reward_delta"
        ),
        "steady_state_delta_vs_online_bo": mean_pairwise(
            "bayes_opt_online", "steady_state_mean_raw_reward_delta"
        ),
        "mean_p95_reduction_vs_online_bo_ms": -mean_pairwise(
            "bayes_opt_online", "mean_p95_latency_delta_ms"
        ),
    }
    for metric, value in computed.items():
        _close(value, float(claim[metric]), tolerance, f"replay {metric}")
    return {"case_count": len(summaries), **computed}


def _verify_attribution(
    inventory: ManifestInventory,
    claim: Mapping[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    """Verify nine compact attribution cases and independently recompute aggregates."""
    data = _source(inventory, claim["source"])
    _require(data.get("schema_version") == 2, "Unsupported attribution schema")
    _require(data.get("primary_algorithm") == "ppo", "Attribution algorithm mismatch")
    cases = _array(data.get("cases"), "Attribution cases")
    _require(len(cases) == 9, "Attribution must contain nine cases")
    expected_cases = {
        (row["case_id"], row["benchmark"], int(row["seed"])) for row in claim["cases"]
    }
    actual_cases = {
        (row.get("case_id"), row.get("benchmark"), int(row.get("seed"))) for row in cases
    }
    _require(actual_cases == expected_cases, "Attribution case set mismatch")
    group_values: dict[str, list[float]] = {name: [] for name in ATTRIBUTION_GROUPS}
    feature_values: dict[str, list[float]] = {}
    for case in cases:
        _require(case.get("load_pattern") == "spike", "Attribution workload must be spike")
        groups = _array(case.get("group_importance"), "Attribution groups")
        singles = _array(case.get("single_feature_importance"), "Attribution features")
        _require(len(groups) == 5, "Attribution group closure mismatch")
        _require(len(singles) == 36, "Attribution single-feature closure mismatch")
        _require(
            {row.get("name") for row in groups} == set(ATTRIBUTION_GROUPS),
            "Attribution group names mismatch",
        )
        for row in groups:
            name = row["name"]
            _require(
                int(row["n_dims"]) == ATTRIBUTION_GROUPS[name],
                f"Attribution dimension mismatch: {name}",
            )
            drop = float(_number(row.get("reward_drop"), f"attribution {name} reward_drop"))
            shift = float(
                _number(
                    row.get("absolute_reward_shift"), f"attribution {name} absolute_reward_shift"
                )
            )
            _close(shift, abs(drop), tolerance, f"attribution {name} absolute shift")
            group_values[name].append(shift)
        names: set[str] = set()
        for row in singles:
            name = str(row["name"])
            _require(name not in names, f"Duplicate attribution feature: {name}")
            names.add(name)
            _require(int(row["n_dims"]) == 1, f"Attribution single dimension mismatch: {name}")
            drop = float(_number(row.get("reward_drop"), f"attribution {name} reward_drop"))
            shift = float(
                _number(
                    row.get("absolute_reward_shift"), f"attribution {name} absolute_reward_shift"
                )
            )
            _close(shift, abs(drop), tolerance, f"attribution {name} absolute shift")
            if (
                row.get("group") in ATTRIBUTION_FILTERED_GROUPS
                and name not in ATTRIBUTION_EXCLUDED_FEATURES
            ):
                feature_values.setdefault(name, []).append(shift)
    expected_groups = claim["expected_group_means"]
    computed_groups = {name: statistics.fmean(values) for name, values in group_values.items()}
    for name, value in expected_groups.items():
        _close(computed_groups[name], float(value), tolerance, f"attribution group {name}")
    aggregate_groups = data["aggregate"]["group_importance"]
    _require(
        {row["name"] for row in aggregate_groups} == set(ATTRIBUTION_GROUPS),
        "Attribution aggregate groups mismatch",
    )
    for row in aggregate_groups:
        _close(
            float(row["mean_absolute_reward_shift"]),
            computed_groups[row["name"]],
            tolerance,
            f"attribution aggregate group {row['name']}",
        )
    computed_features = {name: statistics.fmean(values) for name, values in feature_values.items()}
    ordered_features = sorted(computed_features.items(), key=lambda item: (-item[1], item[0]))
    expected_features = claim["expected_single_means"]
    for name, value in expected_features.items():
        _close(computed_features[name], float(value), tolerance, f"attribution feature {name}")
    aggregate_features = data["aggregate"]["single_feature_importance"]
    _require(
        [row["name"] for row in aggregate_features] == [name for name, _ in ordered_features],
        "Attribution aggregate feature order mismatch",
    )
    for row in aggregate_features:
        _close(
            float(row["mean_absolute_reward_shift"]),
            computed_features[row["name"]],
            tolerance,
            f"attribution aggregate feature {row['name']}",
        )
    top_k = int(claim["top_k"])
    return {
        "case_count": len(cases),
        "group_means": computed_groups,
        "top_features": [
            {"name": name, "mean_absolute_reward_shift": value}
            for name, value in ordered_features[:top_k]
        ],
    }


def _verify_supplementary(
    inventory: ManifestInventory, claims: Mapping[str, Any]
) -> Dict[str, Any]:
    """Check the exact nine supplementary aggregate paths."""
    actual = {
        path
        for path in inventory.files
        if path.startswith("supplementary/") and path.endswith("/aggregate_summary.json")
    }
    _require(actual == SUPPLEMENTARY_PATHS, "Supplementary aggregate inventory mismatch")
    for path in sorted(actual - {claims["attribution"]["source"]}):
        data = _source(inventory, path)
        _require(
            data.get("primary_algorithm") == "ppo", f"Malformed supplementary aggregate: {path}"
        )
    return {"aggregate_count": len(actual)}


def verify_paper_claims(inventory: ManifestInventory, claims: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify every semantic paper claim using only manifest-listed bytes."""
    _require(claims.get("schema_version") == 1, "Unsupported claims schema")
    tolerance = float(claims["tolerances"]["floating_point_absolute"])
    try:
        checks = {
            "model_contract": _verify_model_contract(claims),
            "calibration": _verify_calibration(inventory),
            "main_results": _verify_main_results(inventory, claims["main_comparison"], tolerance),
            "fixed_validation": _verify_fixed_validation(
                inventory, claims["fixed_validation"], tolerance
            ),
            "focused_replay": _verify_replay(inventory, claims["focused_replay"], tolerance),
            "attribution": _verify_attribution(inventory, claims["attribution"], tolerance),
            "supplementary": _verify_supplementary(inventory, claims),
        }
    except (KeyError, TypeError, IndexError) as error:
        raise ArtifactVerificationError(f"Malformed paper claim schema: {error}") from error
    return checks


__all__ = ["verify_paper_claims"]
