"""
OptiQuant — JWT Authentication System
======================================
Features:
  • bcrypt password hashing (12 rounds, auto-salt)
  • PBKDF2-HMAC-SHA256 fallback for legacy passwords
  • HS256 JWT access + refresh tokens with rotation
  • Neon PostgreSQL persistent user store (connection-pooled)
  • Token blacklist (logout / refresh rotation)
  • Rate-limit tracking per IP
  • Password strength validation
  • get_current_user FastAPI dependency for route protection
  • updated_at auto-trigger on row changes
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .database import get_cursor, is_available

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(64))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MIN", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
RATE_LIMIT_WINDOW = 300  # 5 minutes
RATE_LIMIT_MAX = 20      # max attempts per window
MIN_PASSWORD_LENGTH = 8

# ---------------------------------------------------------------------------
# In-memory auth caches (avoid 2x Neon HTTP round-trips per request)
# ---------------------------------------------------------------------------
_AUTH_CACHE_TTL = 60  # seconds
_blacklist_cache: dict[str, tuple[bool, float]] = {}
_user_cache: dict[str, tuple[dict | None, float]] = {}
_cache_lock = threading.Lock()


def _is_blacklisted_cached(jti: str) -> bool:
    now = time.time()
    with _cache_lock:
        entry = _blacklist_cache.get(jti)
        if entry and now - entry[1] < _AUTH_CACHE_TTL:
            return entry[0]
    result = _is_blacklisted(jti)
    with _cache_lock:
        _blacklist_cache[jti] = (result, now)
    return result


def _fetch_user_row_cached(username: str):
    now = time.time()
    with _cache_lock:
        entry = _user_cache.get(username)
        if entry and now - entry[1] < _AUTH_CACHE_TTL:
            return entry[0]
    result = _fetch_user_row(username)
    with _cache_lock:
        _user_cache[username] = (result, now)
    return result

# ---------------------------------------------------------------------------
# Password hashing — PBKDF2-HMAC-SHA256 (no C deps needed)
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash password using bcrypt (12 rounds, auto-salt)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash. Supports bcrypt and legacy PBKDF2."""
    try:
        # bcrypt hashes start with $2b$ or $2a$
        if stored_hash.startswith(("$2b$", "$2a$")):
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
        # Legacy PBKDF2-HMAC-SHA256 fallback
        salt, dk_hex = stored_hash.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return secrets.compare_digest(dk.hex(), dk_hex)
    except Exception:
        return False


def _upgrade_password_hash(user_id: int, password: str) -> None:
    """Re-hash a legacy PBKDF2 password to bcrypt on successful login."""
    new_hash = _hash_password(password)
    with get_cursor() as cur:
        cur.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (new_hash, user_id),
        )


# ---------------------------------------------------------------------------
# Password strength
# ---------------------------------------------------------------------------

def validate_password_strength(password: str) -> str | None:
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if not re.search(r"[A-Z]", password):
        return "Must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Must contain at least one digit."
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?`~]", password):
        return "Must contain at least one special character (!@#$%...)."
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UserRecord:
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str
    last_login: str | None


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MIN * 60


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _check_rate_limit(ip: str) -> None:
    now = time.time()
    with get_cursor() as cur:
        cur.execute(
            "SELECT attempts, window_start FROM rate_limits WHERE ip = %s", (ip,)
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "INSERT INTO rate_limits (ip, attempts, window_start) VALUES (%s, 1, %s)",
                (ip, now),
            )
            return
        if now - row["window_start"] > RATE_LIMIT_WINDOW:
            cur.execute(
                "UPDATE rate_limits SET attempts = 1, window_start = %s WHERE ip = %s",
                (now, ip),
            )
            return
        if row["attempts"] >= RATE_LIMIT_MAX:
            remaining = int(RATE_LIMIT_WINDOW - (now - row["window_start"]))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many attempts. Try again in {remaining}s.",
            )
        cur.execute(
            "UPDATE rate_limits SET attempts = attempts + 1 WHERE ip = %s", (ip,)
        )


# ---------------------------------------------------------------------------
# JWT creation
# ---------------------------------------------------------------------------

def _create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    now = datetime.now(timezone.utc)
    payload.update({
        "iat": now,
        "exp": now + expires_delta,
        "jti": secrets.token_urlsafe(16),
    })
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _create_token_pair(user: UserRecord) -> TokenPair:
    access = _create_token(
        {"sub": user.username, "role": user.role, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN),
    )
    refresh = _create_token(
        {"sub": user.username, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )
    return TokenPair(access_token=access, refresh_token=refresh)


# ---------------------------------------------------------------------------
# Token blacklist
# ---------------------------------------------------------------------------

def _blacklist_token(token: str) -> None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti", "")
        exp = payload.get("exp", 0)
        exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
        with get_cursor() as cur:
            cur.execute(
                "INSERT INTO token_blacklist (jti, expires_at) VALUES (%s, %s) "
                "ON CONFLICT (jti) DO NOTHING",
                (jti, exp_dt),
            )
        # Invalidate cache entry so revocation takes effect immediately
        with _cache_lock:
            _blacklist_cache[jti] = (True, time.time())
    except jwt.PyJWTError:
        pass


def _is_blacklisted(jti: str) -> bool:
    with get_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM token_blacklist WHERE jti = %s", (jti,)
        )
        return cur.fetchone() is not None


def cleanup_blacklist() -> None:
    now = datetime.now(timezone.utc)
    with get_cursor() as cur:
        cur.execute(
            "DELETE FROM token_blacklist WHERE expires_at < %s",
            (now,),
        )


# ---------------------------------------------------------------------------
# Core: signup / login / refresh / logout / profile
# ---------------------------------------------------------------------------

def _require_db() -> None:
    if not is_available():
        raise HTTPException(503, "Database unavailable — please try again later.")

def signup(
    username: str, email: str, password: str, full_name: str = ""
) -> TokenPair:
    _require_db()
    username = username.strip().lower()
    email = email.strip().lower()

    if not username or len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if not re.match(r"^[a-z0-9_]+$", username):
        raise HTTPException(400, "Username: only a-z, 0-9, underscore.")
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email):
        raise HTTPException(400, "Invalid email address.")
    err = validate_password_strength(password)
    if err:
        raise HTTPException(400, err)

    hashed = _hash_password(password)

    # Check for existing user first (avoids transaction abort on conflict)
    with get_cursor() as cur:
        cur.execute(
            "SELECT username, email FROM users WHERE username = %s OR email = %s",
            (username, email),
        )
        existing = cur.fetchone()
        if existing:
            if existing["username"] == username:
                raise HTTPException(409, "Username already taken.")
            raise HTTPException(409, "Email already registered.")

        cur.execute(
            "INSERT INTO users (username, email, password, full_name) "
            "VALUES (%s, %s, %s, %s) RETURNING *",
            (username, email, hashed, full_name),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(500, "Failed to create user.")

    user = _row_to_user(row)
    logger.info("New user registered: %s", username)
    return _create_token_pair(user)


def login(username_or_email: str, password: str, ip: str = "0.0.0.0") -> TokenPair:
    _require_db()
    _check_rate_limit(ip)
    identifier = username_or_email.strip().lower()

    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM users WHERE username = %s OR email = %s",
            (identifier, identifier),
        )
        row = cur.fetchone()

    if not row or not _verify_password(password, row["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials.",
        )
    if not row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled."
        )

    # Transparently upgrade legacy PBKDF2 hashes to bcrypt on login
    if not row["password"].startswith(("$2b$", "$2a$")):
        _upgrade_password_hash(row["id"], password)

    with get_cursor() as cur:
        cur.execute(
            "UPDATE users SET last_login = NOW() WHERE id = %s",
            (row["id"],),
        )

    user = _row_to_user(row)
    logger.info("User logged in: %s", user.username)
    return _create_token_pair(user)


def refresh_tokens(refresh_token: str) -> TokenPair:
    _require_db()
    try:
        payload = jwt.decode(
            refresh_token, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Refresh token expired. Please login again.")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid refresh token.")

    if payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid token type.")
    if _is_blacklisted(payload.get("jti", "")):
        raise HTTPException(401, "Token has been revoked.")

    username = payload.get("sub", "")
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM users WHERE username = %s", (username,)
        )
        row = cur.fetchone()
    if not row or not row["is_active"]:
        raise HTTPException(401, "User not found or disabled.")

    _blacklist_token(refresh_token)
    user = _row_to_user(row)
    return _create_token_pair(user)


def logout(token: str) -> None:
    _require_db()
    _blacklist_token(token)


def get_user_profile(username: str) -> UserRecord:
    _require_db()
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM users WHERE username = %s", (username,)
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "User not found.")
    return _row_to_user(row)


def _row_to_user(row: dict[str, Any]) -> UserRecord:
    return UserRecord(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        full_name=row["full_name"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"] or row["created_at"]),
        last_login=str(row["last_login"]) if row["last_login"] else None,
    )


# ---------------------------------------------------------------------------
# FastAPI dependency — protects routes
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserRecord:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired. Please login again.")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token.")

    if payload.get("type") != "access":
        raise HTTPException(401, "Invalid token type.")

    username = payload.get("sub", "")
    # Run both DB lookups concurrently (each is ~2s to Neon) with caching
    blacklisted, row = await asyncio.gather(
        asyncio.to_thread(_is_blacklisted_cached, payload.get("jti", "")),
        asyncio.to_thread(_fetch_user_row_cached, username),
    )
    if blacklisted:
        raise HTTPException(401, "Token has been revoked.")
    if not row or not row["is_active"]:
        raise HTTPException(401, "User not found or disabled.")
    return _row_to_user(row)


def _fetch_user_row(username: str):
    """Synchronous helper for PostgreSQL lookup (called via to_thread)."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM users WHERE username = %s", (username,)
        )
        return cur.fetchone()
