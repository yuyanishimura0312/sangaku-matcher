# Data Model

## 1. ER Diagram

```
┌──────────────┐       ┌───────────────────┐
│  companies   │1    N │  patents_company   │
│──────────────│───────│───────────────────│
│ edinet_code  │PK     │ edinet_code (FK)  │
│ sec_code     │       │ ipc_code          │
│ name         │       │ count             │
│ industry     │       └───────────────────┘
│ ...          │
│ rd_text      │       ┌───────────────────┐
│ rd_text_vec  │  1  N │  collaborations    │
│ estimated_   │───────│───────────────────│
│  needs       │       │ edinet_code (FK)  │
│ needs_vec    │       │ university_name   │
│ open_inno_   │       │ type              │
│  score       │       │ count             │
└──────┬───────┘       └───────────────────┘
       │
       │ 1
       │
       │        N    ┌──────────────┐
       └─────────────│   matches    │
                     │──────────────│
  ┌──────────┐  N  1 │ match_id     │
  │  seeds   │───────│ seed_id (FK) │
  │──────────│       │ edinet_code  │
  │ seed_id  │PK     │ total_score  │
  │ title    │       │ tech_prox    │
  │ desc     │       │ abs_cap      │
  │ doi      │       │ need_fit     │
  │ vector   │       │ past_ties    │
  └──────────┘       │ rationale    │
                     └──────────────┘

  ┌─────────────────┐
  │ industry_stats   │
  │─────────────────│
  │ industry    PK  │
  │ mean_rd_int     │
  │ std_rd_int      │
  │ company_count   │
  └─────────────────┘
```

## 2. Full Schema (CREATE TABLE)

```sql
-- Core company data, enriched from EDINET + manual input
CREATE TABLE IF NOT EXISTS companies (
    edinet_code         TEXT PRIMARY KEY,
    sec_code            TEXT,
    name                TEXT NOT NULL,
    industry            TEXT,               -- 業種 (EDINET sector code)
    revenue             REAL,               -- 売上高 (百万円)
    operating_profit    REAL,               -- 営業利益 (百万円)
    market_cap          REAL,               -- 時価総額 (百万円)
    employees           INTEGER,            -- 従業員数
    rd_expense          REAL,               -- 研究開発費 (百万円)
    rd_intensity        REAL,               -- R&D集約度 = rd_expense / revenue
    rd_text             TEXT,               -- 有報 [研究開発活動] section full text
    rd_text_vector      BLOB,               -- numpy float32 array, shape (384,)
    midterm_plan_text   TEXT,               -- 中期計画テキスト (IR/統合報告書から)
    estimated_needs     TEXT,               -- LLM-generated JSON (Phase 2)
    needs_vector        BLOB,               -- embedding of estimated_needs (Phase 2)
    needs_generated_at  TEXT,               -- ISO 8601 timestamp
    open_inno_score     REAL DEFAULT 0.0,   -- 0.0 - 1.0 (Phase 3, manual for now)
    source_doc_id       TEXT,               -- EDINET doc_id of the source 有報
    updated_at          TEXT NOT NULL        -- ISO 8601
);

-- Industry-level statistics for normalizing R&D intensity
CREATE TABLE IF NOT EXISTS industry_stats (
    industry        TEXT PRIMARY KEY,
    mean_rd_int     REAL NOT NULL,
    std_rd_int      REAL NOT NULL,
    company_count   INTEGER NOT NULL,
    computed_at     TEXT NOT NULL
);

-- Patent IPC distribution per company (Phase 1: manual/lightweight)
CREATE TABLE IF NOT EXISTS patents_company (
    edinet_code     TEXT NOT NULL,
    ipc_code        TEXT NOT NULL,           -- 4-digit IPC (e.g., "H01L")
    count           INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (edinet_code, ipc_code),
    FOREIGN KEY (edinet_code) REFERENCES companies(edinet_code)
);

-- University-company collaboration records
CREATE TABLE IF NOT EXISTS collaborations (
    edinet_code     TEXT NOT NULL,
    university_name TEXT NOT NULL,
    type            TEXT NOT NULL,            -- 'patent' | 'paper'
    count           INTEGER NOT NULL DEFAULT 1,
    last_year       INTEGER,                  -- most recent collaboration year
    PRIMARY KEY (edinet_code, university_name, type),
    FOREIGN KEY (edinet_code) REFERENCES companies(edinet_code)
);

-- User-submitted technology seeds
CREATE TABLE IF NOT EXISTS seeds (
    seed_id         TEXT PRIMARY KEY,         -- UUID v4
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    doi             TEXT,
    patent_no       TEXT,
    semantic_vector BLOB,                     -- numpy float32 (384,)
    source_type     TEXT NOT NULL DEFAULT 'text',  -- 'text'|'doi'|'patent'|'mixed'
    created_at      TEXT NOT NULL
);

-- Match results (one row per seed-company pair in top N)
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
    rationale       TEXT,                     -- human-readable explanation
    recommended_mode TEXT,                    -- 'joint_research'|'license'|'contract'|'long_term'
    created_at      TEXT NOT NULL,
    FOREIGN KEY (seed_id) REFERENCES seeds(seed_id),
    FOREIGN KEY (edinet_code) REFERENCES companies(edinet_code)
);

-- Audit log for data acquisition
CREATE TABLE IF NOT EXISTS acquisition_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    action          TEXT NOT NULL,             -- 'load_companies'|'refresh_needs'|'import_patents'
    target_count    INTEGER,
    success_count   INTEGER,
    error_count     INTEGER,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    note            TEXT
);
```

## 3. Index Design

```sql
-- Company lookups
CREATE INDEX IF NOT EXISTS idx_co_industry ON companies(industry);
CREATE INDEX IF NOT EXISTS idx_co_rd_expense ON companies(rd_expense);
CREATE INDEX IF NOT EXISTS idx_co_name ON companies(name);

-- Match queries
CREATE INDEX IF NOT EXISTS idx_match_seed ON matches(seed_id);
CREATE INDEX IF NOT EXISTS idx_match_score ON matches(seed_id, total_score DESC);

-- Collaboration lookups
CREATE INDEX IF NOT EXISTS idx_collab_uni ON collaborations(university_name);
```

## 4. Data Acquisition & Update Flow

### 4.1 Initial Load (scripts/load_companies.py)

```
1. Connect to IR Collector DB (read-only)
2. SELECT distinct filers with doc_type_code='120'
3. For each filer:
   a. Get latest 有報 PDF path
   b. Extract text with pdfplumber
   c. Parse R&D section, financial metrics
   d. Compute rd_text_vector via EmbeddingModel
   e. INSERT INTO companies
4. Compute industry_stats (mean/std of rd_intensity per industry)
5. Log to acquisition_log
```

### 4.2 Manual Patent Import

Phase 1 uses a simple CSV import for known collaborations:

```csv
# data/seed_corpus/collaborations.csv
edinet_code,university_name,type,count,last_year
E02144,東京大学,patent,5,2024
E02144,東北大学,paper,12,2025
```

```bash
sangaku-matcher import-collabs --csv data/seed_corpus/collaborations.csv
```

### 4.3 Backcast Estimation (Phase 2)

```
1. SELECT companies WHERE estimated_needs IS NULL (or needs_generated_at < threshold)
2. For each: call Claude API with rd_text + midterm_plan_text
3. Parse JSON response → estimated_needs
4. Compute needs_vector via EmbeddingModel
5. UPDATE companies SET estimated_needs, needs_vector, needs_generated_at
6. Log to acquisition_log
```

## 5. Vector Storage

Embedding vectors (384-dim float32) are stored as BLOB in SQLite using numpy's `tobytes()` / `frombuffer()`:

```python
# Store
vec_bytes = vector.astype(np.float32).tobytes()
conn.execute("UPDATE companies SET rd_text_vector = ? WHERE edinet_code = ?", (vec_bytes, code))

# Load
row = conn.execute("SELECT rd_text_vector FROM companies WHERE edinet_code = ?", (code,)).fetchone()
vector = np.frombuffer(row[0], dtype=np.float32)
```

For bulk operations, load all vectors into a single numpy matrix (4000 x 384) for vectorized cosine similarity.
