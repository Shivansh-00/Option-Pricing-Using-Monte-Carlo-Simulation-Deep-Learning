from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import importlib.metadata as metadata

logger = logging.getLogger(__name__)


@dataclass
class DependencyCheckReport:
    checked: int
    mismatches: list[str]
    missing: list[str]

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.missing


@dataclass
class CompatibilityResult:
    ok: bool
    reasons: list[str]


def _safe_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def get_runtime_fingerprint() -> dict[str, str]:
    fp: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "scikit-learn", "scipy", "pandas"):
        version = _safe_version(package)
        if version:
            fp[package] = version
    return fp


def validate_dependency_lock(requirements_file: str | Path) -> DependencyCheckReport:
    req_path = Path(requirements_file)
    if not req_path.exists():
        return DependencyCheckReport(checked=0, mismatches=[], missing=[f"lock file not found: {req_path}"])

    mismatches: list[str] = []
    missing: list[str] = []
    checked = 0

    for raw_line in req_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue

        package, expected = [part.strip() for part in line.split("==", 1)]
        checked += 1
        installed = _safe_version(package)
        if installed is None:
            missing.append(f"{package} expected {expected}, installed MISSING")
            continue
        if installed != expected:
            mismatches.append(f"{package} expected {expected}, installed {installed}")

    return DependencyCheckReport(checked=checked, mismatches=mismatches, missing=missing)


def check_model_runtime_compatibility(
    artifact_name: str,
    expected_runtime: dict[str, Any] | None,
    strict: bool = True,
) -> CompatibilityResult:
    if not expected_runtime:
        reason = f"{artifact_name}: missing runtime_fingerprint metadata"
        return CompatibilityResult(ok=not strict, reasons=[reason])

    current = get_runtime_fingerprint()
    reasons: list[str] = []

    for key in ("python", "numpy", "scikit-learn", "scipy", "pandas"):
        expected = expected_runtime.get(key)
        if expected is None:
            continue
        got = current.get(key)
        if got != expected:
            reasons.append(f"{artifact_name}: {key} expected {expected}, runtime {got}")

    return CompatibilityResult(ok=not reasons, reasons=reasons)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
