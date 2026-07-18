"""Integrity and accepted-paper claim verification for CALO artifacts.

The verifier deliberately has no dependency on the simulator, PyTorch, or
CodeBERT.  It reads the release tree once, validates the SHA-256 inventory, and
passes only those verified bytes to :mod:`paper_claims`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "SHA256SUMS"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024
TEXT_SUFFIXES = frozenset({".csv", ".json", ".md", ".txt", ".yaml", ".yml"})
FORBIDDEN_SUFFIXES = frozenset(
    {".log", ".partial", ".pdf", ".png", ".jpg", ".jpeg", ".svg", ".pkl", ".pt"}
)
JOBLIB_NAMES = frozenset(
    f"calibration/{architecture}/lightgbm_service/{filename}"
    for architecture in ("x64", "arm64")
    for filename in (
        "warm_mean_model.joblib",
        "warm_p95_model.joblib",
        "cold_overhead_mean_model.joblib",
        "cold_overhead_p95_model.joblib",
        "burst_slowdown_mean_model.joblib",
        "burst_slowdown_p95_model.joblib",
    )
)

_PRIVATE_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    (
        "Azure account key",
        re.compile(
            rb"(?i)(?:AccountKey|AZURE_STORAGE_ACCOUNT_KEY|azure_storage_account_key)\s*[:=]"
        ),
    ),
    (
        "credential field",
        re.compile(
            rb"(?i)[\"']?(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret|"
            rb"private[_-]?key|wsk[_-]?auth|secret)[\"']?\s*[:=]"
        ),
    ),
    (
        "absolute host path",
        re.compile(
            rb"(?<![A-Za-z0-9_.-])/(?:home|root|Users|tmp|var|etc|opt|mnt|data|srv)/"
            rb"[A-Za-z0-9._~+:/-]+"
        ),
    ),
    (
        "absolute Windows path",
        re.compile(rb"\b[A-Za-z]:\\[A-Za-z0-9._~+\\ /-]+"),
    ),
    (
        "private IPv4 host",
        re.compile(
            rb"(?<![0-9])(?:10\.(?:[0-9]{1,3}\.){2}[0-9]{1,3}|"
            rb"192\.168\.(?:[0-9]{1,3}\.)[0-9]{1,3}|"
            rb"172\.(?:1[6-9]|2[0-9]|3[01])\.(?:[0-9]{1,3}\.)[0-9]{1,3})"
            rb"(?![0-9])"
        ),
    ),
    (
        "private host field",
        re.compile(
            rb"(?i)[\"']?(?:host|hostname|host_name)[\"']?\s*[:=]\s*"
            rb"[\"'][A-Za-z0-9][A-Za-z0-9.-]{2,}[\"']"
        ),
    ),
)


class ArtifactVerificationError(RuntimeError):
    """Raised when a released artifact violates an integrity or claim check."""


def _require(condition: bool, message: str) -> None:
    """Raise a verification error when a condition is false."""
    if not condition:
        raise ArtifactVerificationError(message)


def _resolve_artifact_dir(artifact_dir: str | Path) -> Path:
    """Resolve an artifact directory and reject a symlinked root."""
    path = Path(artifact_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    _require(not path.is_symlink(), f"Artifact directory cannot be a symbolic link: {path}")
    path = path.resolve()
    _require(path.is_dir(), f"Artifact directory does not exist: {path}")
    return path


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently choosing one."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    """Reject the non-standard NaN/Infinity JSON extensions."""
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _require_finite_tree(value: Any, location: str = "$", seen: set[int] | None = None) -> None:
    """Require finite numeric leaves and JSON-compatible container types."""
    seen = set() if seen is None else seen
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"Non-finite JSON number at {location}")
        return
    if isinstance(value, (list, tuple)):
        marker = id(value)
        _require(marker not in seen, f"Cyclic JSON value at {location}")
        seen.add(marker)
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{location}[{index}]", seen)
        seen.remove(marker)
        return
    if isinstance(value, dict):
        marker = id(value)
        _require(marker not in seen, f"Cyclic JSON value at {location}")
        seen.add(marker)
        for key, item in value.items():
            _require(isinstance(key, str), f"Non-string JSON key at {location}")
            _require_finite_tree(item, f"{location}.{key}", seen)
        seen.remove(marker)
        return
    raise ArtifactVerificationError(f"Unsupported JSON value at {location}: {type(value).__name__}")


def strict_json_loads(payload: bytes | str) -> Any:
    """Decode UTF-8 JSON with duplicate-key and finite-number checks."""
    try:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError(f"Invalid strict JSON: {error}") from error
    _require_finite_tree(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value canonically for content identity checks."""
    _require_finite_tree(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactVerificationError(f"Cannot canonicalize JSON: {error}") from error


def _safe_relative(relative_path: str) -> str:
    """Validate and normalize one manifest-relative POSIX path."""
    _require(
        bool(relative_path) and "\\" not in relative_path,
        f"Unsafe artifact path: {relative_path}",
    )
    pure = PurePosixPath(relative_path)
    _require(
        relative_path == pure.as_posix()
        and not pure.is_absolute()
        and "." not in pure.parts
        and ".." not in pure.parts,
        f"Unsafe artifact path: {relative_path}",
    )
    return relative_path


@dataclass(frozen=True)
class ManifestInventory:
    """Read-once bytes validated against the complete SHA-256 manifest."""

    root: Path
    files: Mapping[str, bytes]
    manifest_bytes: bytes
    entries: Mapping[str, str]

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "ManifestInventory":
        """Read each release file once and validate exact manifest coverage."""
        root = _resolve_artifact_dir(artifact_dir)
        files: dict[str, bytes] = {}
        total_bytes = 0
        manifest_bytes: bytes | None = None
        for directory, directories, names in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            for name in directories:
                child = directory_path / name
                _require(not child.is_symlink(), f"Symbolic links are forbidden: {child}")
            for name in names:
                child = directory_path / name
                _require(not child.is_symlink(), f"Symbolic links are forbidden: {child}")
                relative = child.relative_to(root).as_posix()
                _safe_relative(relative)
                _require(child.is_file(), f"Release entry is not a regular file: {relative}")
                size = child.stat().st_size
                _require(size <= MAX_FILE_BYTES, f"Artifact exceeds 10 MiB: {relative}")
                payload = child.read_bytes()
                _require(len(payload) == size, f"Artifact changed while reading: {relative}")
                if relative == MANIFEST_NAME:
                    manifest_bytes = payload
                else:
                    files[relative] = payload
                    total_bytes += len(payload)
        _require(
            manifest_bytes is not None,
            f"Missing integrity manifest: {root / MANIFEST_NAME}",
        )
        assert manifest_bytes is not None
        _require(
            len(manifest_bytes) <= MAX_FILE_BYTES,
            f"Artifact exceeds 10 MiB: {MANIFEST_NAME}",
        )
        _require(
            total_bytes + len(manifest_bytes) <= MAX_BUNDLE_BYTES,
            f"Artifact bundle exceeds 32 MiB: {total_bytes + len(manifest_bytes)} bytes",
        )

        try:
            manifest_text = manifest_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactVerificationError("SHA256SUMS is not valid UTF-8") from error
        entries: dict[str, str] = {}
        for line_number, line in enumerate(manifest_text.splitlines(), start=1):
            if not line.strip():
                continue
            digest, separator, relative = line.partition("  ")
            _require(
                separator == "  " and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"Malformed manifest entry on line {line_number}",
            )
            relative = _safe_relative(relative)
            _require(relative != MANIFEST_NAME, "Manifest must not list itself")
            _require(relative not in entries, f"Duplicate manifest path: {relative}")
            entries[relative] = digest
        _require(
            set(entries) == set(files),
            "Manifest coverage mismatch: "
            f"missing={sorted(set(files) - set(entries))}, "
            f"extra={sorted(set(entries) - set(files))}",
        )
        for relative, expected in entries.items():
            actual = hashlib.sha256(files[relative]).hexdigest()
            _require(actual == expected, f"SHA-256 mismatch: {relative}")
        return cls(
            root=root,
            files=MappingProxyType(files),
            manifest_bytes=manifest_bytes,
            entries=MappingProxyType(entries),
        )

    def require_bytes(self, relative_path: str) -> bytes:
        """Return bytes only when the path is explicitly manifest-listed."""
        relative = _safe_relative(relative_path)
        _require(relative in self.files, f"Manifest-listed artifact is missing: {relative}")
        return self.files[relative]

    def require_json(self, relative_path: str) -> Any:
        """Load one manifest-listed JSON payload strictly."""
        _require(relative_path.endswith(".json"), f"Expected JSON artifact: {relative_path}")
        return strict_json_loads(self.require_bytes(relative_path))

    def all_payloads(self) -> Iterable[tuple[str, bytes]]:
        """Yield every release payload, including the manifest itself."""
        yield from self.files.items()
        yield MANIFEST_NAME, self.manifest_bytes

    def summary(self) -> Dict[str, int]:
        """Return manifest inventory counts without exposing mutable bytes."""
        return {"file_count": len(self.files), "total_bytes": sum(map(len, self.files.values()))}


def _load_json(path: Path) -> Any:
    """Load one JSON file strictly for backwards-compatible internal callers."""
    try:
        return strict_json_loads(path.read_bytes())
    except OSError as error:
        raise ArtifactVerificationError(f"Cannot read JSON file {path}: {error}") from error


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactVerificationError(f"Cannot read artifact file {path}: {error}") from error


def _check_manifest(root: Path) -> Dict[str, int]:
    """Verify a manifest and return its inventory summary."""
    return ManifestInventory.load(root).summary()


def _scan_private_bytes(relative: str, payload: bytes) -> None:
    """Reject credentials, host information, or absolute paths in any bytes."""
    for label, pattern in _PRIVATE_PATTERNS:
        _require(pattern.search(payload) is None, f"Potential {label} found in {relative}")


def _check_release_policy(inventory: ManifestInventory) -> Dict[str, int]:
    """Check size, suffix, binary, UTF-8, JSON, and privacy release policy."""
    payloads = list(inventory.all_payloads())
    joblibs = {relative for relative, _ in payloads if relative.endswith(".joblib")}
    _require(joblibs == JOBLIB_NAMES, f"Unexpected joblib inventory: {sorted(joblibs)}")
    total_bytes = sum(len(payload) for _, payload in payloads)
    _require(
        total_bytes <= MAX_BUNDLE_BYTES, f"Artifact bundle exceeds 32 MiB: {total_bytes} bytes"
    )
    for relative, payload in payloads:
        _require(
            "partial" not in Path(relative).name.lower(), f"Partial output released: {relative}"
        )
        suffix = Path(relative).suffix.lower()
        _require(suffix not in FORBIDDEN_SUFFIXES, f"Forbidden release file type: {relative}")
        if relative != MANIFEST_NAME:
            _require(
                suffix in TEXT_SUFFIXES or relative in JOBLIB_NAMES,
                f"Unapproved binary or file type: {relative}",
            )
        _scan_private_bytes(relative, payload)
        if suffix in TEXT_SUFFIXES:
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ArtifactVerificationError(
                    f"Text artifact is not UTF-8: {relative}"
                ) from error
            if suffix == ".json":
                strict_json_loads(payload)
    return {"file_count": len(inventory.files), "total_bytes": total_bytes}


def verify_artifacts(
    artifact_dir: str | Path = "artifacts",
    measure_inference: bool = False,
) -> Dict[str, Any]:
    """Verify the released artifact bundle and accepted-paper claims.

    The default path imports no torch, transformers, or CodeBERT modules.  The
    optional local timing report is loaded only after all frozen claims pass.
    """
    inventory = ManifestInventory.load(artifact_dir)
    release = _check_release_policy(inventory)
    claims = inventory.require_json("claims.json")
    _require(isinstance(claims, dict), "claims.json must contain an object")
    from .paper_claims import verify_paper_claims

    checks = {
        "manifest": inventory.summary(),
        "release_policy": release,
        **verify_paper_claims(inventory, claims),
    }
    report: Dict[str, Any] = {
        "status": "PASS",
        "artifact_version": claims["artifact_version"],
        "artifact_dir": str(inventory.root),
        "checks": checks,
    }
    if measure_inference:
        from .inference_benchmark import measure_inference as run_inference

        report["local_inference"] = run_inference()
    return report


def format_verification_report(report: Mapping[str, Any]) -> str:
    """Format a concise human-readable artifact verification report."""
    lines = [
        f"CALO artifact verification: {report['status']}",
        f"Artifact version: {report['artifact_version']}",
    ]
    for name, details in report["checks"].items():
        lines.append(f"{name}: PASS ({json.dumps(details, sort_keys=True)})")
    if "local_inference" in report:
        lines.append(
            "local_inference: REPORTED "
            f"({json.dumps(report['local_inference'], sort_keys=True)})"
        )
    return "\n".join(lines)
