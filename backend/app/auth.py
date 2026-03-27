"""
OptiQuant — JWT Authentication System
======================================
Features:
  • bcrypt password hashing (12 rounds, auto-salt)
  • PBKDF2-HMAC-SHA256 fallback for legacy passwords
  • HS256 JWT access + refresh tokens with rotation
  • SQLite persistent user store (WAL mode, indexed)
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
import platform
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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

# Persistent DB path — stored in backend/data/ (not temp dir)
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DB_PATH = Path(os.getenv("AUTH_DB_PATH", str(_DATA_DIR / "optiquant_auth.db")))

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
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET password = ?, updated_at = datetime('now') WHERE id = ?",
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
# SQLite user store
# ---------------------------------------------------------------------------

def _init_db() -> None:
    """Create tables, indexes, triggers, and run schema migrations."""
    for attempt in range(5):
        try:
            with _get_conn() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS users (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        username    TEXT    UNIQUE NOT NULL,
                        email       TEXT    UNIQUE NOT NULL,
                        password    TEXT    NOT NULL,
                        full_name   TEXT    DEFAULT '',
                        role        TEXT    DEFAULT 'user',
                        is_active   INTEGER DEFAULT 1,
                        created_at  TEXT    DEFAULT (datetime('now')),
                        updated_at  TEXT    DEFAULT (datetime('now')),
                        last_login  TEXT
                    );

                    -- Explicit indexes for fast lookups
                    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                    CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
                    CREATE INDEX IF NOT EXISTS idx_users_active   ON users(is_active);

                    -- Auto-update updated_at on any row change
                    CREATE TRIGGER IF NOT EXISTS trg_users_updated_at
                    AFTER UPDATE ON users
                    FOR EACH ROW
                    BEGIN
                        UPDATE users SET updated_at = datetime('now')
                        WHERE id = OLD.id AND updated_at = OLD.updated_at;
                    END;

                    CREATE TABLE IF NOT EXISTS token_blacklist (
                        jti         TEXT    PRIMARY KEY,
                        expires_at  TEXT    NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS rate_limits (
                        ip          TEXT    PRIMARY KEY,
                        attempts    INTEGER DEFAULT 0,
                        window_start REAL   NOT NULL
                    );
                """)
                # Schema migration: add updated_at to existing DBs
                _migrate_schema(conn)
            return
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < 4:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add missing columns to existing databases (idempotent)."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "updated_at" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN updated_at TEXT DEFAULT (datetime('now'))")
        conn.execute("UPDATE users SET updated_at = created_at WHERE updated_at IS NULL")
        logger.info("Migrated: added updated_at column to users table")


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


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
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT attempts, window_start FROM rate_limits WHERE ip = ?", (ip,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO rate_limits (ip, attempts, window_start) VALUES (?, 1, ?)",
                (ip, now),
            )
            return
        if now - row["window_start"] > RATE_LIMIT_WINDOW:
            conn.execute(
                "UPDATE rate_limits SET attempts = 1, window_start = ? WHERE ip = ?",
                (now, ip),
            )
            return
        if row["attempts"] >= RATE_LIMIT_MAX:
            remaining = int(RATE_LIMIT_WINDOW - (now - row["window_start"]))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many attempts. Try again in {remaining}s.",
            )
        conn.execute(
            "UPDATE rate_limits SET attempts = attempts + 1 WHERE ip = ?", (ip,)
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
        exp = payload.get("exp", "")
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO token_blacklist (jti, expires_at) VALUES (?, ?)",
                (jti, str(exp)),
            )
    except jwt.PyJWTError:
        pass


def _is_blacklisted(jti: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM token_blacklist WHERE jti = ?", (jti,)
        ).fetchone()
        return row is not None


def cleanup_blacklist() -> None:
    now = int(datetime.now(timezone.utc).timestamp())
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM token_blacklist WHERE CAST(expires_at AS INTEGER) < ?",
            (now,),
        )


# ---------------------------------------------------------------------------
# Core: signup / login / refresh / logout / profile
# ---------------------------------------------------------------------------

def signup(
    username: str, email: str, password: str, full_name: str = ""
) -> TokenPair:
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
    with _get_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, email, password, full_name) "
                "VALUES (?, ?, ?, ?)",
                (username, email, hashed, full_name),
            )
        except sqlite3.IntegrityError:
            existing = conn.execute(
                "SELECT username, email FROM users WHERE username = ? OR email = ?",
                (username, email),
            ).fetchone()
            if existing and existing["username"] == username:
                raise HTTPException(409, "Username already taken.")
            raise HTTPException(409, "Email already registered.")

        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

    user = _row_to_user(row)
    logger.info("New user registered: %s", username)
    return _create_token_pair(user)


def login(username_or_email: str, password: str, ip: str = "0.0.0.0") -> TokenPair:
    _check_rate_limit(ip)
    identifier = username_or_email.strip().lower()

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()

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

    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?",
            (row["id"],),
        )

    user = _row_to_user(row)
    logger.info("User logged in: %s", user.username)
    return _create_token_pair(user)


def refresh_tokens(refresh_token: str) -> TokenPair:
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
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row or not row["is_active"]:
        raise HTTPException(401, "User not found or disabled.")

    _blacklist_token(refresh_token)
    user = _row_to_user(row)
    return _create_token_pair(user)


def logout(token: str) -> None:
    _blacklist_token(token)


def get_user_profile(username: str) -> UserRecord:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "User not found.")
    return _row_to_user(row)


def _row_to_user(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        full_name=row["full_name"],
        role=row["role"],
        is_active=bool(row["is_active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"] or row["created_at"],
        last_login=row["last_login"],
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
    if _is_blacklisted(payload.get("jti", "")):
        raise HTTPException(401, "Token has been revoked.")

    username = payload.get("sub", "")
    row = await asyncio.to_thread(_fetch_user_row, username)
    if not row or not row["is_active"]:
        raise HTTPException(401, "User not found or disabled.")
    return _row_to_user(row)


def _fetch_user_row(username: str):
    """Synchronous helper for SQLite lookup (called via to_thread)."""
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


# ---------------------------------------------------------------------------
# Init on import
# ---------------------------------------------------------------------------
_init_db()
