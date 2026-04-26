#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.runtime_lockstep import (  # noqa: E402
    validate_dependency_lock,
    check_model_runtime_compatibility,
    get_runtime_fingerprint,
)


def _check_vol_engine(models_dir: Path) -> list[str]:
    meta_path = models_dir / "vol_engine_meta.json"
    if not meta_path.exists():
        return [f"vol_engine: metadata missing ({meta_path})"]

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"vol_engine: metadata parse failed: {exc}"]

    result = check_model_runtime_compatibility(
        "vol_engine",
        meta.get("runtime_fingerprint"),
        strict=True,
    )
    return result.reasons


def _check_hybrid_lstm(models_dir: Path) -> list[str]:
    model_path = models_dir / "hybrid_lstm.npz"
    meta_path = models_dir / "hybrid_lstm_meta.json"

    if not model_path.exists():
        return [f"hybrid_lstm: model missing ({model_path})"]
    if not meta_path.exists():
        return [f"hybrid_lstm: metadata missing ({meta_path})"]

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"hybrid_lstm: metadata parse failed: {exc}"]

    result = check_model_runtime_compatibility(
        "hybrid_lstm",
        meta.get("runtime_fingerprint"),
        strict=True,
    )
    return result.reasons


def _check_pinns(models_dir: Path) -> list[str]:
    model_path = models_dir / "pinns_model.pkl"
    if not model_path.exists():
        return [f"pinns_model: model missing ({model_path})"]

    try:
        with open(model_path, "rb") as f:
            state = pickle.load(f)
    except Exception as exc:
        return [f"pinns_model: load failed: {exc}"]

    if not isinstance(state, dict):
        return ["pinns_model: invalid state format"]

    result = check_model_runtime_compatibility(
        "pinns_model",
        state.get("runtime_fingerprint"),
        strict=True,
    )
    return result.reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Check lockstep dependency and model compatibility.")
    parser.add_argument(
        "--models-dir",
        default=str(ROOT / "backend" / "models"),
        help="Path to model artifact directory.",
    )
    parser.add_argument(
        "--lock-file",
        default=str(ROOT / "backend" / "requirements.lock.txt"),
        help="Path to lock file with pinned dependencies.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON report.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    lock_file = Path(args.lock_file)

    dep = validate_dependency_lock(lock_file)

    model_reasons: list[str] = []
    model_reasons.extend(_check_vol_engine(models_dir))
    model_reasons.extend(_check_hybrid_lstm(models_dir))
    model_reasons.extend(_check_pinns(models_dir))

    report = {
        "runtime_fingerprint": get_runtime_fingerprint(),
        "dependency_lock": {
            "file": str(lock_file),
            "checked": dep.checked,
            "mismatches": dep.mismatches,
            "missing": dep.missing,
            "ok": dep.ok,
        },
        "model_compatibility": {
            "models_dir": str(models_dir),
            "issues": model_reasons,
            "ok": len(model_reasons) == 0,
        },
        "overall_ok": dep.ok and len(model_reasons) == 0,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== Lockstep Dependency Check ===")
        print(f"Lock file: {lock_file}")
        print(f"Checked pinned deps: {dep.checked}")
        if dep.ok:
            print("Dependency status: OK")
        else:
            print("Dependency status: FAILED")
            for item in dep.mismatches + dep.missing:
                print(f"  - {item}")

        print("\n=== Model Compatibility Check ===")
        print(f"Models dir: {models_dir}")
        if not model_reasons:
            print("Model compatibility: OK")
        else:
            print("Model compatibility: FAILED")
            for item in model_reasons:
                print(f"  - {item}")

        print("\nOverall:", "OK" if report["overall_ok"] else "FAILED")

    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
