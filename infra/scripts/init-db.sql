-- ══════════════════════════════════════════════════════════════
--  OptiQuant — PostgreSQL Initialization Script
--  Replaces SQLite for production auth + event logging
-- ══════════════════════════════════════════════════════════════

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ── Users Table (replaces SQLite auth) ──────────────────────
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username    VARCHAR(100) UNIQUE NOT NULL,
    email       VARCHAR(255) UNIQUE,
    password    TEXT NOT NULL,
    role        VARCHAR(20) DEFAULT 'user',
    is_active   BOOLEAN DEFAULT true,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    last_login  TIMESTAMPTZ
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- ── Event Log (audit trail) ────────────────────────────────
CREATE TABLE IF NOT EXISTS event_log (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type  VARCHAR(50) NOT NULL,
    endpoint    VARCHAR(255),
    payload     JSONB,
    ip_address  INET,
    user_agent  TEXT,
    duration_ms FLOAT,
    status_code INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_event_log_user_id ON event_log(user_id);
CREATE INDEX idx_event_log_type ON event_log(event_type);
CREATE INDEX idx_event_log_created ON event_log(created_at DESC);

-- ── Pricing History ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pricing_history (
    id              BIGSERIAL PRIMARY KEY,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    model_type      VARCHAR(50) NOT NULL,
    option_type     VARCHAR(10) NOT NULL,
    spot_price      DOUBLE PRECISION NOT NULL,
    strike_price    DOUBLE PRECISION NOT NULL,
    risk_free_rate  DOUBLE PRECISION NOT NULL,
    volatility      DOUBLE PRECISION NOT NULL,
    time_to_expiry  DOUBLE PRECISION NOT NULL,
    dividend_yield  DOUBLE PRECISION DEFAULT 0,
    result_price    DOUBLE PRECISION,
    confidence_interval JSONB,
    greeks          JSONB,
    compute_time_ms FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pricing_user ON pricing_history(user_id);
CREATE INDEX idx_pricing_model ON pricing_history(model_type);
CREATE INDEX idx_pricing_created ON pricing_history(created_at DESC);

-- ── Model Performance Tracking ──────────────────────────────
CREATE TABLE IF NOT EXISTS model_metrics (
    id              BIGSERIAL PRIMARY KEY,
    model_name      VARCHAR(100) NOT NULL,
    metric_name     VARCHAR(50) NOT NULL,
    metric_value    DOUBLE PRECISION NOT NULL,
    metadata        JSONB,
    recorded_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_model_metrics_name ON model_metrics(model_name, metric_name);
CREATE INDEX idx_model_metrics_recorded ON model_metrics(recorded_at DESC);

-- ── Alerts & Signals ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id              BIGSERIAL PRIMARY KEY,
    alert_type      VARCHAR(50) NOT NULL,
    severity        VARCHAR(20) DEFAULT 'info',
    symbol          VARCHAR(20),
    message         TEXT NOT NULL,
    metadata        JSONB,
    acknowledged    BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_created ON alerts(created_at DESC);

-- ── Create default admin user (password: admin123, CHANGE IN PRODUCTION) ──
-- Password hash is PBKDF2-HMAC-SHA256 - generate a new one for production
INSERT INTO users (username, email, role)
VALUES ('admin', 'admin@optiquant.ai', 'admin')
ON CONFLICT (username) DO NOTHING;

-- ── Grants ──────────────────────────────────────────────────
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO optiquant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO optiquant;
