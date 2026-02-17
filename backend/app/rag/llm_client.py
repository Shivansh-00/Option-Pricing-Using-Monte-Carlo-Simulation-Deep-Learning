"""
Enterprise LLM Client
======================
Production-grade LLM integration with:
- Google Gemini REST API (primary)
- Retry with exponential backoff
- Rate limiting (token bucket)
- Circuit breaker pattern
- Response validation
- Token budget management
- Performance telemetry
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

# Approximate chars-per-token ratio for English text
_CHARS_PER_TOKEN = 4


# ── Token Budget Manager ─────────────────────────────────────────────────


class TokenBudget:
    """Estimate and manage token budgets for LLM calls."""

    def __init__(self, max_output_tokens: int = 512) -> None:
        self.max_output = max_output_tokens
        # Gemini model context windows
        self._model_limits: dict[str, int] = {
            "gemini-2.0-flash": 1_048_576,
            "gemini-1.5-flash": 1_048_576,
            "gemini-1.5-pro": 2_097_152,
            "gemini-pro": 32_768,
        }

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation."""
        return max(len(text) // _CHARS_PER_TOKEN, 1)

    def get_context_limit(self, model: str) -> int:
        """Get the model's context window size."""
        return self._model_limits.get(model, 32_768)

    def compute_available_context(
        self,
        model: str,
        system_prompt: str,
        user_prefix: str,
    ) -> int:
        """
        Compute how many tokens are available for context
        after accounting for system prompt, prefix, and output.
        """
        limit = self.get_context_limit(model)
        used = (
            self.estimate_tokens(system_prompt)
            + self.estimate_tokens(user_prefix)
            + self.max_output
            + 100  # Safety margin
        )
        return max(limit - used, 500)

    def truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        max_chars = max_tokens * _CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        # Truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.7:
            return truncated[:last_period + 1]
        return truncated


# ── Rate Limiter (Token Bucket) ───────────────────────────────────────────


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(
        self,
        max_requests_per_minute: int = 30,
        max_tokens_per_minute: int = 100_000,
    ) -> None:
        self._max_rpm = max_requests_per_minute
        self._max_tpm = max_tokens_per_minute
        self._request_tokens = float(max_requests_per_minute)
        self._token_tokens = float(max_tokens_per_minute)
        self._last_refill = time.time()
        self._lock = Lock()

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill
        self._request_tokens = min(
            self._max_rpm,
            self._request_tokens + elapsed * self._max_rpm / 60,
        )
        self._token_tokens = min(
            self._max_tpm,
            self._token_tokens + elapsed * self._max_tpm / 60,
        )
        self._last_refill = now

    def acquire(self, estimated_tokens: int = 1000) -> float:
        """
        Acquire permission to make an API call.
        Returns the wait time in seconds (0 if no wait needed).
        """
        with self._lock:
            self._refill()
            if self._request_tokens >= 1 and self._token_tokens >= estimated_tokens:
                self._request_tokens -= 1
                self._token_tokens -= estimated_tokens
                return 0.0

            # Calculate wait time
            wait_request = max(0, (1 - self._request_tokens) * 60 / self._max_rpm)
            wait_tokens = max(
                0,
                (estimated_tokens - self._token_tokens) * 60 / self._max_tpm,
            )
            return max(wait_request, wait_tokens)


# ── Circuit Breaker ───────────────────────────────────────────────────────


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 2

    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _state: str = field(default="closed", init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def can_execute(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = "half_open"
                    self._half_open_calls = 0
                    logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                    return True
                return False
            if self._state == "half_open":
                return self._half_open_calls < self.half_open_max_calls

        return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._half_open_calls += 1
                if self._half_open_calls >= self.half_open_max_calls:
                    self._state = "closed"
                    self._failure_count = 0
                    logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            elif self._state == "closed":
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == "half_open":
                self._state = "open"
                logger.warning("Circuit breaker: HALF_OPEN -> OPEN")
            elif self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(
                    "Circuit breaker: CLOSED -> OPEN (failures=%d)",
                    self._failure_count,
                )

    @property
    def state(self) -> str:
        return self._state

    @property
    def stats(self) -> dict:
        return {
            "state": self._state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout,
        }


# ── LLM Client ───────────────────────────────────────────────────────────


class LLMClient:
    """
    Enterprise LLM client for Gemini API.

    Features:
    - Retry with exponential backoff
    - Rate limiting
    - Circuit breaker
    - Token budget management
    - Performance telemetry
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or settings.gemini_api_key
        self._model = model or settings.gemini_model
        self._temperature = temperature if temperature is not None else settings.gemini_temperature
        self._max_tokens = max_tokens or settings.gemini_max_tokens
        self._max_retries = max_retries
        self._timeout = timeout

        self._budget = TokenBudget(self._max_tokens)
        self._rate_limiter = RateLimiter(
            max_requests_per_minute=int(os.getenv("LLM_MAX_RPM", "30")),
            max_tokens_per_minute=int(os.getenv("LLM_MAX_TPM", "100000")),
        )
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv("LLM_CB_THRESHOLD", "5")),
            recovery_timeout=float(os.getenv("LLM_CB_RECOVERY", "60")),
        )

        # Telemetry
        self._total_calls = 0
        self._total_errors = 0
        self._total_latency = 0.0
        self._total_tokens_est = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Parameters
        ----------
        system_prompt : str
            System instructions.
        user_prompt : str
            User message with context and question.
        temperature : float, optional
            Override default temperature.
        max_tokens : int, optional
            Override default max tokens.

        Returns
        -------
        str
            Generated response text.

        Raises
        ------
        LLMError
            If generation fails after all retries.
        """
        if not self._api_key:
            raise LLMError(
                "GEMINI_API_KEY not set. Configure via environment variable.",
                error_type="config",
            )

        # Circuit breaker check
        if not self._circuit_breaker.can_execute():
            raise LLMError(
                f"Circuit breaker OPEN — LLM calls disabled for "
                f"{self._circuit_breaker.recovery_timeout}s after "
                f"{self._circuit_breaker.failure_threshold} failures.",
                error_type="circuit_breaker",
            )

        # Rate limiting
        est_tokens = self._budget.estimate_tokens(user_prompt + system_prompt)
        wait_time = self._rate_limiter.acquire(est_tokens)
        if wait_time > 0:
            logger.info("Rate limited, waiting %.1fs", wait_time)
            time.sleep(min(wait_time, 10))

        # Token budget management
        available = self._budget.compute_available_context(
            self._model, system_prompt, "",
        )
        if self._budget.estimate_tokens(user_prompt) > available:
            user_prompt = self._budget.truncate_to_budget(user_prompt, available)
            logger.info("User prompt truncated to fit context window")

        temp = temperature if temperature is not None else self._temperature
        max_tok = max_tokens or self._max_tokens

        url = _GEMINI_URL.format(model=self._model, key=self._api_key)
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_tok,
            },
        }

        t0 = time.time()
        last_error: Exception | None = None

        with httpx.Client(timeout=self._timeout) as client:
            for attempt in range(self._max_retries):
                try:
                    resp = client.post(url, json=payload)

                    if resp.status_code == 429:
                        delay = min(
                            int(resp.headers.get("retry-after", str(2 ** attempt))),
                            30,
                        )
                        logger.info(
                            "Rate limited (429), retrying in %ds (attempt %d/%d)",
                            delay, attempt + 1, self._max_retries,
                        )
                        time.sleep(delay)
                        continue

                    if resp.status_code >= 500:
                        delay = min(2 ** attempt, 15)
                        logger.warning(
                            "Server error %d, retrying in %ds (attempt %d/%d)",
                            resp.status_code, delay, attempt + 1, self._max_retries,
                        )
                        time.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = resp.json()

                    # Extract response text
                    text = (
                        data.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                        .strip()
                    )

                    if not text:
                        raise LLMError(
                            "Empty response from Gemini API",
                            error_type="empty_response",
                        )

                    # Success
                    latency = time.time() - t0
                    self._total_calls += 1
                    self._total_latency += latency
                    self._total_tokens_est += est_tokens
                    self._circuit_breaker.record_success()

                    logger.info(
                        "LLM response: model=%s, latency=%.0fms, est_tokens=%d",
                        self._model, latency * 1000, est_tokens,
                    )
                    return text

                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code in (400, 401, 403):
                        # Non-retryable
                        self._circuit_breaker.record_failure()
                        raise LLMError(
                            f"Gemini API error {exc.response.status_code}: "
                            f"{exc.response.text[:200]}",
                            error_type="api_error",
                        ) from exc

                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    last_error = exc
                    delay = min(2 ** attempt, 15)
                    logger.warning(
                        "Connection error: %s, retrying in %ds (attempt %d/%d)",
                        type(exc).__name__, delay, attempt + 1, self._max_retries,
                    )
                    time.sleep(delay)

                except Exception as exc:
                    last_error = exc
                    logger.error("Unexpected LLM error: %s", exc)
                    break

        # All retries exhausted
        self._total_errors += 1
        self._circuit_breaker.record_failure()
        raise LLMError(
            f"Gemini API exhausted all {self._max_retries} retries. "
            f"Last error: {last_error}",
            error_type="exhausted",
        )

    @property
    def stats(self) -> dict:
        avg_latency = (
            (self._total_latency / self._total_calls * 1000)
            if self._total_calls > 0 else 0.0
        )
        return {
            "model": self._model,
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
            "error_rate": (
                round(self._total_errors / max(self._total_calls + self._total_errors, 1), 3)
            ),
            "avg_latency_ms": round(avg_latency, 1),
            "total_tokens_estimated": self._total_tokens_est,
            "circuit_breaker": self._circuit_breaker.stats,
        }

    @property
    def model(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def max_tokens(self) -> int:
        return self._max_tokens


# ── Error Types ───────────────────────────────────────────────────────────


class LLMError(Exception):
    """Custom exception for LLM errors with type classification."""

    def __init__(self, message: str, error_type: str = "unknown") -> None:
        super().__init__(message)
        self.error_type = error_type


# ── Singleton ─────────────────────────────────────────────────────────────

_CLIENT: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
