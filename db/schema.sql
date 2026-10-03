-- AuTrad AI schema. PostgreSQL 16 + TimescaleDB.
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------- Operational ----------
CREATE TABLE IF NOT EXISTS portfolios (
  id           bigserial PRIMARY KEY,
  name         text NOT NULL UNIQUE,
  mode         text NOT NULL DEFAULT 'PAPER' CHECK (mode IN ('RESEARCH','PAPER','LIMITED_LIVE','LIVE')),
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS orders (
  id           bigserial PRIMARY KEY,
  client_id    text NOT NULL UNIQUE,                 -- idempotency key
  portfolio_id bigint REFERENCES portfolios(id),
  symbol       text NOT NULL,
  qty          double precision NOT NULL,
  type         text NOT NULL DEFAULT 'MARKET',
  limit_price  double precision,
  status       text NOT NULL DEFAULT 'NEW',
  created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS orders_portfolio_created_idx ON orders (portfolio_id, created_at DESC);
CREATE INDEX IF NOT EXISTS orders_symbol_created_idx    ON orders (symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS orders_open_idx              ON orders (status) WHERE status IN ('NEW','PARTIAL');

CREATE TABLE IF NOT EXISTS fills (
  id          bigserial PRIMARY KEY,
  order_id    bigint NOT NULL REFERENCES orders(id),
  qty         double precision NOT NULL,
  price       double precision NOT NULL,
  commission  double precision NOT NULL DEFAULT 0,
  filled_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS fills_order_idx ON fills (order_id);   -- makes the batched ANY($1) lookup an index scan

CREATE TABLE IF NOT EXISTS model_registry (
  id            bigserial PRIMARY KEY,
  name          text NOT NULL,
  version       text NOT NULL,
  git_sha       text NOT NULL,
  dataset_hash  text NOT NULL,
  train_start   date, train_end date,
  metrics       jsonb NOT NULL DEFAULT '{}',
  stage         text NOT NULL DEFAULT 'RESEARCH' CHECK (stage IN ('RESEARCH','PAPER','LIMITED_LIVE','LIVE','RETIRED')),
  approved_by   text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (name, version)
);

-- Append-only, hash-chained audit trail (UPDATE/DELETE revoked from the app role)
CREATE TABLE IF NOT EXISTS decision_log (
  seq        bigserial PRIMARY KEY,
  hash       char(64) NOT NULL UNIQUE,
  prev_hash  char(64) NOT NULL,
  kind       text NOT NULL,
  symbol     text,
  payload    jsonb NOT NULL,
  ts         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS decision_log_symbol_ts_idx ON decision_log (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS evidence (
  seq           bigserial,
  id            text NOT NULL,
  source        text NOT NULL,
  url           text NOT NULL,
  title         text NOT NULL,
  summary       text,
  symbols       text[] NOT NULL DEFAULT '{}',
  sentiment     real NOT NULL DEFAULT 0,
  simhash       bigint NOT NULL,
  published_at  timestamptz NOT NULL,
  ingested_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (seq, ingested_at)
);
SELECT create_hypertable('evidence', 'ingested_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS evidence_symbols_gin ON evidence USING gin (symbols);
CREATE UNIQUE INDEX IF NOT EXISTS evidence_id_uq ON evidence (id, ingested_at);

-- ---------- Time series ----------
CREATE TABLE IF NOT EXISTS bars (
  symbol  text NOT NULL,
  ts      timestamptz NOT NULL,
  open double precision, high double precision, low double precision, close double precision, volume double precision,
  PRIMARY KEY (symbol, ts)
);
SELECT create_hypertable('bars', 'ts', if_not_exists => TRUE);
ALTER TABLE bars SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol');
SELECT add_compression_policy('bars', INTERVAL '30 days', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS predictions (
  symbol text NOT NULL, ts timestamptz NOT NULL, model_version text NOT NULL,
  regime text, mu double precision, sigma double precision, target_weight double precision,
  expert_weights jsonb,
  PRIMARY KEY (symbol, ts, model_version)
);
SELECT create_hypertable('predictions', 'ts', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  portfolio_id bigint NOT NULL, ts timestamptz NOT NULL,
  equity double precision, gross_exposure double precision, drawdown double precision,
  PRIMARY KEY (portfolio_id, ts)
);
SELECT create_hypertable('portfolio_snapshots', 'ts', if_not_exists => TRUE);

-- Pre-aggregated daily P&L (expensive query materialised once, refreshed continuously)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_pnl WITH (timescaledb.continuous) AS
  SELECT portfolio_id, time_bucket('1 day', ts) AS day, last(equity, ts) AS close_equity, max(drawdown) AS max_dd
  FROM portfolio_snapshots GROUP BY portfolio_id, day WITH NO DATA;
