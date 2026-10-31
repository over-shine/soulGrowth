-- SoulGrowth 结构化真相（docs/13-数据契约 §SQLite DDL）
-- 约定：只增不改；删除一律走 status/archived（不变量 I7）。
-- 事件日志的**真源**是 data/<id>/events.jsonl；这里 events 表只是可查询的镜像。

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- 一次醒着的片刻
CREATE TABLE IF NOT EXISTS episodes (
  id        TEXT PRIMARY KEY,
  ts        TEXT    NOT NULL,
  age_days  INTEGER NOT NULL,
  stage     TEXT    NOT NULL,
  kind      TEXT    NOT NULL,
  salience  REAL    NOT NULL,
  catch_up  INTEGER NOT NULL DEFAULT 0,
  json      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience DESC);

-- 全文检索：只索引摘要，正文在 json 里
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
  id UNINDEXED,
  summary,
  tokenize = 'unicode61'
);

-- 记忆条目：importance 与 confidence 必须分开（前者是重要性，后者是可信度）
CREATE TABLE IF NOT EXISTS memories (
  id           TEXT PRIMARY KEY,
  type         TEXT    NOT NULL,
  content      TEXT    NOT NULL,
  importance   REAL    NOT NULL DEFAULT 0.5,
  confidence   REAL    NOT NULL DEFAULT 0.5,
  created      TEXT    NOT NULL,
  last_access  TEXT,
  access_count INTEGER NOT NULL DEFAULT 0,
  decay        REAL    NOT NULL DEFAULT 0.03,
  status       TEXT    NOT NULL DEFAULT 'active',
  superseded_by TEXT,
  refs         TEXT    NOT NULL DEFAULT '[]',
  tags         TEXT    NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type, status);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);

-- 关系向量（对每个 entity 一行）
CREATE TABLE IF NOT EXISTS relationships (
  entity     TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,
  vec        TEXT NOT NULL,
  attachment TEXT,
  updated    TEXT NOT NULL
);

-- 特质：每周恰好一条（不变量 I6），无数据周也写 n_episodes=0 且 T 不变
CREATE TABLE IF NOT EXISTS traits (
  week        TEXT PRIMARY KEY,
  stage       TEXT NOT NULL,
  T           TEXT NOT NULL,
  B           TEXT NOT NULL,
  eta         REAL NOT NULL,
  xi          REAL NOT NULL,
  n_episodes  INTEGER NOT NULL DEFAULT 0,
  indicators  TEXT NOT NULL DEFAULT '{}',
  missing     TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS milestones (
  id           TEXT PRIMARY KEY,
  stage        TEXT NOT NULL,
  status       TEXT NOT NULL,
  value        REAL,
  threshold    REAL,
  evaluated_at TEXT NOT NULL,
  evidence     TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS stage_history (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  from_stage       TEXT,
  to_stage         TEXT NOT NULL,
  at               TEXT NOT NULL,
  age_days         INTEGER NOT NULL,
  milestones_passed TEXT NOT NULL DEFAULT '[]',
  unlocked         TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS events (
  seq     INTEGER PRIMARY KEY,
  ts      TEXT NOT NULL,
  type    TEXT NOT NULL,
  ref     TEXT,
  actor   TEXT,
  version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
