#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.rag.llm_client import LLMClient  # noqa: E402


TEXT_FILE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".json",
    ".jsonc",
    ".md",
    ".yml",
    ".yaml",
}

CLAUDE_OPUS_MODEL = "claude-opus-4.7"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv-1",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
}

INVALID_PATTERNS = {
    "effort_high": re.compile(r'"effort"\s*:\s*"high"', re.IGNORECASE),
    "output_config_effort": re.compile(r"output_config\s*[:=].*effort", re.IGNORECASE),
    "reasoning_effort": re.compile(r"reasoning_effort", re.IGNORECASE),
}

SELF_PATH = Path(__file__).resolve()


def _should_scan(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in TEXT_FILE_SUFFIXES
        and path.resolve() != SELF_PATH
        and not any(part in SKIP_DIR_NAMES for part in path.parts)
    )


def _scan_repo() -> list[str]:
    issues: list[str] = []
    for path in ROOT.rglob("*"):
        if not _should_scan(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for name, pattern in INVALID_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                issues.append(f"{path.relative_to(ROOT)}:{line}: matched {name}")
    return issues


def _check_sanitizer() -> list[str]:
    client = LLMClient(api_key="test-key", model=CLAUDE_OPUS_MODEL)
    payload = client._sanitize_payload(
        {
            "model": CLAUDE_OPUS_MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 128,
            "temperature": 0.2,
            "effort": "high",
            "output_config": {
                "effort": "high",
                "format": "json",
            },
        }
    )

    issues: list[str] = []
    if payload.get("effort") is not None:
        issues.append("sanitizer failed: top-level effort was not removed")
    if payload.get("output_config") is not None:
        issues.append("sanitizer failed: output_config was not removed")
    if payload.get("model") != CLAUDE_OPUS_MODEL:
        issues.append("sanitizer failed: model was unexpectedly changed")
    return issues


def main() -> int:
    repo_issues = _scan_repo()
    sanitizer_issues = _check_sanitizer()

    print("=== Claude Opus 4.7 Compatibility Check ===")
    if repo_issues:
        print("Repo scan: FAILED")
        for issue in repo_issues:
            print(f"  - {issue}")
    else:
        print("Repo scan: OK")

    if sanitizer_issues:
        print("Sanitizer check: FAILED")
        for issue in sanitizer_issues:
            print(f"  - {issue}")
    else:
        print("Sanitizer check: OK")

    overall_ok = not repo_issues and not sanitizer_issues
    print("Overall:", "OK" if overall_ok else "FAILED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())