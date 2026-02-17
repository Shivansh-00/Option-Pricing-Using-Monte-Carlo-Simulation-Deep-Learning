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

    # --- Google Gemini LLM for RAG ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_max_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "512"))
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.4"))

    # --- RAG Pipeline ---
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    rag_min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.01"))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "600"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    rag_cache_max: int = int(os.getenv("RAG_CACHE_MAX", "128"))
    rag_response_ttl: int = int(os.getenv("RAG_RESPONSE_TTL", "600"))
    rag_max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

    # --- RAG Embedding ---
    rag_embedding_backend: str = os.getenv("RAG_EMBEDDING_BACKEND", "auto")
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    rag_tfidf_max_features: int = int(os.getenv("RAG_TFIDF_MAX_FEATURES", "8000"))

    # --- LLM Rate Limiting & Circuit Breaker ---
    llm_max_rpm: int = int(os.getenv("LLM_MAX_RPM", "30"))
    llm_max_tpm: int = int(os.getenv("LLM_MAX_TPM", "100000"))
    llm_cb_threshold: int = int(os.getenv("LLM_CB_THRESHOLD", "5"))
    llm_cb_recovery: int = int(os.getenv("LLM_CB_RECOVERY", "60"))


settings = Settings()
