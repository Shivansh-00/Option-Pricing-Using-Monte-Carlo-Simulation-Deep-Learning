from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    backend_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(root_dir / ".env")
    load_dotenv(backend_env)


def _parse_csv(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    return [item.strip() for item in value.split(",") if item.strip()]


_load_env()


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "Intelligent Option Pricing Platform")
    environment: str = os.getenv("ENV", "dev")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    cors_origins: list[str] = field(
        default_factory=lambda: _parse_csv(
            os.getenv("CORS_ORIGINS"),
            ["http://localhost:8080", "http://127.0.0.1:8080"],
        )
    )
    frontend_dir: str = os.getenv("FRONTEND_DIR", "frontend")
    model_dir: str = os.getenv("MODEL_DIR", "models")


settings = Settings()
