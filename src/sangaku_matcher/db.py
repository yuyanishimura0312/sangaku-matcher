"""SQLite access layer for sangaku-matcher."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    edinet_code         TEXT PRIMARY KEY,
    sec_code            TEXT,
    name                TEXT NOT NULL,
    industry            TEXT,
    revenue             REAL,
    operating_profit    REAL,
    market_cap          REAL,
    employees           INTEGER,
    rd_expense          REAL,
    rd_intensity        REAL,
    rd_text             TEXT,
    rd_text_vector      BLOB,
    midterm_plan_text   TEXT,
    estimated_needs     TEXT,
    needs_vector        BLOB,
    needs_generated_at  TEXT,
    open_inno_score     REAL DEFAULT 0.0,
    source_doc_id       TEXT,
    updated_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS industry_stats (
    industry        TEXT PRIMARY KEY,
    mean_rd_int     REAL NOT NULL,
    std_rd_int      REAL NOT NULL,
    company_count   INTEGER NOT NULL,
    computed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patents_company (
    edinet_code     TEXT NOT NULL,
    ipc_code        TEXT NOT NULL,
    count           INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (edinet_code, ipc_code),
    FOREIGN KEY (edinet_code) REFERENCES companies(edinet_code)
);

CREATE TABLE IF NOT EXISTS collaborations (
    edinet_code     TEXT NOT NULL,
    university_name TEXT NOT NULL,
    type            TEXT NOT NULL,
    count           INTEGER NOT NULL DEFAULT 1,
    last_year       INTEGER,
    PRIMARY KEY (edinet_code, university_name, type),
    FOREIGN KEY (edinet_code) REFERENCES companies(edinet_code)
);

CREATE TABLE IF NOT EXISTS seeds (
    seed_id         TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    doi             TEXT,
    patent_no       TEXT,
    semantic_vector BLOB,
    source_type     TEXT NOT NULL DEFAULT 'text',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    seed_id         TEXT NOT NULL,
    edinet_code     TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    total_score     REAL NOT NULL,
    tech_prox       REAL,
    abs_cap         REAL,
    need_fit        REAL,
    past_ties       REAL,
    trl_compat      REAL,
    open_inno_mat   REAL,
    humanities_fit  REAL,
    synergy         REAL,
    rationale       TEXT,
    recommended_mode TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (seed_id) REFERENCES seeds(seed_id),
    FOREIGN KEY (edinet_code) REFERENCES companies(edinet_code)
);

CREATE TABLE IF NOT EXISTS acquisition_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    action          TEXT NOT NULL,
    target_count    INTEGER,
    success_count   INTEGER,
    error_count     INTEGER,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    note            TEXT
);

CREATE INDEX IF NOT EXISTS idx_co_industry ON companies(industry);
CREATE INDEX IF NOT EXISTS idx_co_rd_expense ON companies(rd_expense);
CREATE INDEX IF NOT EXISTS idx_co_name ON companies(name);
CREATE INDEX IF NOT EXISTS idx_match_seed ON matches(seed_id);
CREATE INDEX IF NOT EXISTS idx_match_score ON matches(seed_id, total_score DESC);
CREATE INDEX IF NOT EXISTS idx_collab_uni ON collaborations(university_name);
CREATE INDEX IF NOT EXISTS idx_co_updated_at ON companies(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_match_created_at ON matches(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_seed_created_at ON seeds(created_at DESC);

CREATE TABLE IF NOT EXISTS multi_exit_matches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    seed_id         TEXT NOT NULL,
    exit_type       TEXT NOT NULL,
    edinet_code     TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    total_score     REAL NOT NULL,
    tech_prox       REAL,
    need_fit        REAL,
    abs_cap         REAL,
    past_ties       REAL,
    future_option   REAL,
    open_inno       REAL,
    humanities_fit  REAL,
    ambition_fit    REAL,
    theme_breadth   REAL,
    synergy         REAL,
    hypothesis      TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (seed_id) REFERENCES seeds(seed_id)
);
CREATE INDEX IF NOT EXISTS idx_mex_seed ON multi_exit_matches(seed_id);
CREATE INDEX IF NOT EXISTS idx_mex_exit ON multi_exit_matches(seed_id, exit_type);
"""


def init_db(db_path: Path) -> None:
    """Create tables and indexes if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a connection with Row factory, auto-commit on success."""
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    # Try WAL mode but fall back gracefully if filesystem is read-only
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass
    try:
        yield conn
        try:
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Read-only DB: skip commit
    finally:
        conn.close()
