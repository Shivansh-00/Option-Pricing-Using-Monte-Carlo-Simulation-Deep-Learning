"""
OptiQuant — Neon PostgreSQL Database Module
============================================
Two transports, chosen automatically at startup:

1. **psycopg2 TCP pool** — used when port 5432 is reachable
   (fastest; real connection pool).
2. **Neon HTTP SQL API** — used when port 5432 is blocked
   (works over HTTPS port 443 on any network).

Callers only use  ``get_cursor()``  /  ``get_conn()``  /  ``is_available()``
and never need to know which transport is active.
"""
from __future__ import annotations

import json as _json
import logging
import os
import re
import socket
from contextlib import contextmanager
from typing import Any, Generator
from urllib.parse import urlparse

import requests as _requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)
_JSON_CONTENT_TYPE = "application/json"

# ---------------------------------------------------------------------------
# Transport flag
# ---------------------------------------------------------------------------
_MODE: str = "none"          # "pool" | "http" | "none"

# == psycopg2 pool objects ==
_pool = None                 # psycopg2.pool.ThreadedConnectionPool | None

# == HTTP transport objects ==
_HTTP_ENDPOINT: str = ""     # e.g. https://ep-…pooler.…neon.tech/sql
_HTTP_CONN_STR: str = ""     # full connection string sent as header
_HTTP_SESSION: _requests.Session | None = None  # reuses TLS connections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it to your Neon PostgreSQL connection string."
        )
    return url


def _port_open(host: str, port: int, timeout: float = 4.0) -> bool:
    """Return True if *host:port* accepts a TCP connection."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except Exception:
        return False


def _build_http_vars(url: str):
    """From a postgres:// URL derive the HTTPS endpoint + conn-header."""
    parsed = urlparse(url)
    # Ensure pooler hostname for HTTP endpoint
    host = parsed.hostname or ""
    if "-pooler" not in host:
        host = host.replace(".c-", "-pooler.c-", 1)
    endpoint = f"https://{host}/sql"
    # Connection string: force pooler host, standard port, sslmode=require
    conn_str = re.sub(r"@[^/]+", f"@{host}", url)
    conn_str = re.sub(r":\d+/", "/", conn_str)       # strip any explicit port
    if "sslmode" not in conn_str:
        sep = "&" if "?" in conn_str else "?"
        conn_str += f"{sep}sslmode=require"
    return endpoint, conn_str


# ---------------------------------------------------------------------------
# Initialization / shutdown
# ---------------------------------------------------------------------------

def init_pool(minconn: int = 1, maxconn: int = 10) -> bool:
    """Try TCP pool first; fall back to HTTP transport. Returns True on success."""
    global _pool, _MODE, _HTTP_ENDPOINT, _HTTP_CONN_STR
    if _MODE != "none":
        return True

    url = _get_database_url()

    # --- Attempt 1: direct TCP (psycopg2) ---
    parsed = urlparse(url)
    pg_host = parsed.hostname or ""
    pg_port = parsed.port or 5432
    if _port_open(pg_host, pg_port):
        try:
            import psycopg2.pool  # type: ignore[import-untyped]  # noqa: local import — not needed if HTTP mode
            sep = "&" if "?" in url else "?"
            tcp_url = url if "connect_timeout" in url else f"{url}{sep}connect_timeout=10"
            _pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, tcp_url)
            _MODE = "pool"
            logger.info("Neon PostgreSQL TCP pool initialised (min=%d, max=%d)", minconn, maxconn)
            return True
        except Exception as exc:
            logger.warning("TCP pool failed: %s — trying HTTP transport", exc)

    # --- Attempt 2: Neon HTTP SQL API (HTTPS, port 443) ---
    try:
        endpoint, conn_str = _build_http_vars(url)
        resp = _requests.post(
            endpoint,
            headers={"Content-Type": _JSON_CONTENT_TYPE,
                     "Neon-Connection-String": conn_str},
            json={"query": "SELECT 1"},
            timeout=10,
        )
        resp.raise_for_status()
        _HTTP_ENDPOINT = endpoint
        _HTTP_CONN_STR = conn_str
        _HTTP_SESSION = _requests.Session()
        _HTTP_SESSION.headers.update({
            "Content-Type": _JSON_CONTENT_TYPE,
            "Neon-Connection-String": conn_str,
        })
        _MODE = "http"
        logger.info("Neon HTTP SQL transport initialised (%s)", endpoint)
        return True
    except Exception as exc:
        logger.warning("HTTP transport also failed: %s", exc)

    logger.warning("App will run with limited functionality (no persistent storage)")
    return False


def close_pool() -> None:
    """Close all connections. Called at app shutdown."""
    global _pool, _MODE, _HTTP_SESSION
    if _MODE == "pool" and _pool is not None:
        _pool.closeall()
        _pool = None
    if _HTTP_SESSION is not None:
        _HTTP_SESSION.close()
        _HTTP_SESSION = None
    _MODE = "none"
    logger.info("Database connection pool closed")


def is_available() -> bool:
    return _MODE != "none"


# ---------------------------------------------------------------------------
# TCP-pool context managers (only used when _MODE == "pool")
# ---------------------------------------------------------------------------

def _conn_is_alive(conn) -> bool:
    cur = None
    try:
        if conn.closed:
            return False
        cur = conn.cursor()
        cur.execute("SELECT 1")
        return True
    except Exception:
        return False
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        try:
            conn.rollback()
        except Exception:
            pass


@contextmanager
def _tcp_conn():
    """Get a live connection from the psycopg2 pool."""
    conn = _pool.getconn()
    if not _conn_is_alive(conn):
        logger.debug("Stale TCP connection — reconnecting")
        try:
            conn.close()
        except Exception:
            pass
        _pool.putconn(conn, close=True)
        conn = _pool.getconn()
    try:
        conn.rollback()
    except Exception:
        pass
    if getattr(conn, "autocommit", False):
        conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            _pool.putconn(conn)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HTTP cursor shim  (drop-in for callers that do cur.execute / fetchone / …)
# ---------------------------------------------------------------------------

class _HttpCursor:
    """Minimal DB-API-2-style cursor backed by the Neon HTTP SQL endpoint."""

    def __init__(self):
        self._rows: list[dict] = []
        self._pos: int = 0
        self._batch: list[tuple[str, list]] = []   # for multi-statement txns

    # -- execute / executemany -------------------------------------------------

    def execute(self, query: str, params: tuple | list | None = None):
        # Convert %s positional params → $1, $2, … (Neon HTTP uses $N style)
        converted, plist = self._convert_params(query, params)
        session = _HTTP_SESSION or _requests
        resp = session.post(
            _HTTP_ENDPOINT,
            headers={
                "Content-Type": _JSON_CONTENT_TYPE,
                "Neon-Connection-String": _HTTP_CONN_STR,
            },
            json={"query": converted, "params": plist},
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        self._rows = body.get("rows", [])
        self._pos = 0

    # -- fetch -----------------------------------------------------------------

    def fetchone(self) -> dict | None:
        if self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            return row
        return None

    def fetchall(self) -> list[dict]:
        remaining = self._rows[self._pos:]
        self._pos = len(self._rows)
        return remaining

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _convert_params(query: str, params: tuple | list | None):
        """Replace ``%s`` placeholders with ``$1, $2, …`` for Neon HTTP."""
        if params is None:
            return query, []
        plist = list(params)
        parts = query.split("%s")
        converted = parts[0] + "".join(
            f"${index}{part}"
            for index, part in enumerate(parts[1:], start=1)
        )
        # Serialise non-primitive values (e.g. dicts → JSON strings)
        out: list[Any] = []
        for v in plist:
            if isinstance(v, dict):
                out.append(_json.dumps(v))
            else:
                out.append(v)
        return converted, out

    def close(self):
        """HTTP cursors are stateless, so there is nothing to release."""
        pass


class _HttpConnection:
    """Thin wrapper so ``with get_conn() as conn: conn.cursor(…)`` works."""

    def cursor(self, _cursor_factory=None, **_kw):
        return _HttpCursor()

    def commit(self):
        """HTTP transport executes statements eagerly, so commit is a no-op."""
        pass

    def rollback(self):
        """HTTP transport has no open transaction state to roll back."""
        pass


# ---------------------------------------------------------------------------
# Public API  (transport-agnostic)
# ---------------------------------------------------------------------------

@contextmanager
def get_conn() -> Generator:
    """Return a connection (real psycopg2 conn *or* HTTP shim)."""
    if _MODE == "pool":
        with _tcp_conn() as conn:
            yield conn
    elif _MODE == "http":
        yield _HttpConnection()
    else:
        raise RuntimeError("Database pool is not initialized")


@contextmanager
def get_cursor(cursor_factory=None) -> Generator:
    """Return a dict-cursor inside a managed connection."""
    if _MODE == "pool":
        import psycopg2.extras  # type: ignore[import-untyped]
        factory = cursor_factory or psycopg2.extras.RealDictCursor
        with _tcp_conn() as conn:
            cur = conn.cursor(cursor_factory=factory)
            try:
                yield cur
            finally:
                cur.close()
    elif _MODE == "http":
        cur = _HttpCursor()
        try:
            yield cur
        finally:
            cur.close()
    else:
        raise RuntimeError("Database pool is not initialized")


# ---------------------------------------------------------------------------
# Schema initialization (idempotent — safe to call on every startup)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id          SERIAL PRIMARY KEY,
    username    VARCHAR(64)  UNIQUE NOT NULL,
    email       VARCHAR(255) UNIQUE NOT NULL,
    password    TEXT         NOT NULL,
    full_name   VARCHAR(255) DEFAULT '',
    role        VARCHAR(32)  DEFAULT 'user',
    is_active   BOOLEAN      DEFAULT TRUE,
    created_at  TIMESTAMPTZ  DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  DEFAULT NOW(),
    last_login  TIMESTAMPTZ
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active   ON users(is_active);

-- Token blacklist for JWT revocation
CREATE TABLE IF NOT EXISTS token_blacklist (
    jti         VARCHAR(64)  PRIMARY KEY,
    expires_at  TIMESTAMPTZ  NOT NULL
);

-- Rate limiting per IP
CREATE TABLE IF NOT EXISTS rate_limits (
    ip           VARCHAR(45) PRIMARY KEY,
    attempts     INTEGER     DEFAULT 0,
    window_start DOUBLE PRECISION NOT NULL
);

-- Event log for audit trail and model monitoring
CREATE TABLE IF NOT EXISTS event_log (
    id          SERIAL PRIMARY KEY,
    event       VARCHAR(128) NOT NULL,
    payload     JSONB        DEFAULT '{}',
    created_at  TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_event_log_event      ON event_log(event);
CREATE INDEX IF NOT EXISTS idx_event_log_created_at  ON event_log(created_at);

-- Pricing history for analytics
CREATE TABLE IF NOT EXISTS pricing_history (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER REFERENCES users(id) ON DELETE SET NULL,
    model           VARCHAR(64) NOT NULL,
    option_type     VARCHAR(8)  NOT NULL,
    spot            DOUBLE PRECISION,
    strike          DOUBLE PRECISION,
    expiry          DOUBLE PRECISION,
    rate            DOUBLE PRECISION,
    volatility      DOUBLE PRECISION,
    computed_price   DOUBLE PRECISION,
    greeks          JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pricing_history_user ON pricing_history(user_id);
CREATE INDEX IF NOT EXISTS idx_pricing_history_model ON pricing_history(model);
"""

# Trigger function for auto-updating updated_at (PostgreSQL syntax)
# Split into individual statements so the HTTP transport can run them one-by-one
# (Neon HTTP SQL API rejects multi-statement prepared statements).
_TRIGGER_FN_SQL = """
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
"""

_TRIGGER_ATTACH_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_users_updated_at'
    ) THEN
        CREATE TRIGGER trg_users_updated_at
        BEFORE UPDATE ON users
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$
"""

# Legacy combined string kept for TCP mode (psycopg2 handles multi-statement)
_TRIGGER_SQL = _TRIGGER_FN_SQL + ";\n" + _TRIGGER_ATTACH_SQL + ";"


def _split_simple_sql(sql: str) -> list[str]:
    """Split SQL on semicolons (only for SQL *without* $$ blocks)."""
    stmts = [s.strip() for s in sql.split(";")]
    return [s for s in stmts if s and not all(
        ln.strip() == "" or ln.strip().startswith("--") for ln in s.splitlines()
    )]


def init_db() -> bool:
    """Create all tables, indexes, and triggers. Idempotent."""
    if not is_available():
        logger.warning("Skipping schema init — database not connected")
        return False

    if _MODE == "pool":
        # psycopg2 handles multi-statement natively
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(_SCHEMA_SQL)
            cur.execute(_TRIGGER_SQL)
            cur.close()
    else:
        # HTTP: one statement per request
        cur = _HttpCursor()
        for stmt in _split_simple_sql(_SCHEMA_SQL):
            cur.execute(stmt)
        cur.execute(_TRIGGER_FN_SQL)
        cur.execute(_TRIGGER_ATTACH_SQL)
        cur.close()

    logger.info("Database schema initialized successfully")
    return True
