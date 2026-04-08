# Architecture Design

## 1. System Overview

```
                        ┌─────────────────────────┐
                        │    Presentation Layer    │
                        │  ┌─────────┐ ┌────────┐ │
                        │  │ FastAPI │ │  CLI   │ │
                        │  │  (Web)  │ │(Click) │ │
                        │  └────┬────┘ └───┬────┘ │
                        └───────┼──────────┼──────┘
                                │          │
                        ┌───────▼──────────▼──────┐
                        │   Application Layer      │
                        │  ┌────────────────────┐  │
                        │  │     Matcher         │  │
                        │  │  (orchestrator)     │  │
                        │  └──┬───────────┬──────┘  │
                        │     │           │         │
                        │  ┌──▼──┐    ┌───▼──────┐  │
                        │  │Seed │    │ Reporter │  │
                        │  │Parser│   │(md/json) │  │
                        │  └─────┘    └──────────┘  │
                        └─────────┬─────────────────┘
                                  │
                        ┌─────────▼─────────────────┐
                        │     Scoring Engine         │
                        │  ┌────────┐ ┌────────┐    │
                        │  │TechProx│ │AbsCap  │    │
                        │  └────────┘ └────────┘    │
                        │  ┌────────┐ ┌────────┐    │
                        │  │NeedFit │ │PastTies│    │
                        │  │(Ph.2)  │ │        │    │
                        │  └────────┘ └────────┘    │
                        │  ┌────────┐ ┌────────┐    │
                        │  │TRLComp │ │OpenInno│    │
                        │  │(Ph.3)  │ │(Ph.3)  │    │
                        │  └────────┘ └────────┘    │
                        └─────────┬─────────────────┘
                                  │
                        ┌─────────▼─────────────────┐
                        │      Data Layer            │
                        │  ┌───────┐ ┌───────────┐  │
                        │  │SQLite │ │ Embedding │  │
                        │  │matcher│ │  Cache    │  │
                        │  │ .db   │ │ (.npy)    │  │
                        │  └───┬───┘ └───────────┘  │
                        └──────┼────────────────────┘
                               │
               ┌───────────────┼────────────────────┐
               │               │   Acquisition       │
               │  ┌────────────▼──┐ ┌─────────────┐  │
               │  │ IR Collector  │ │Semantic      │  │
               │  │ (pip dep)     │ │Scholar API   │  │
               │  │ → EDINET API  │ └─────────────┘  │
               │  └───────────────┘ ┌─────────────┐  │
               │                    │Claude API    │  │
               │                    │(Phase 2)     │  │
               │                    └─────────────┘  │
               └─────────────────────────────────────┘
```

## 2. Module Interface Definitions

### 2.1 Seed Parser (`seeds.py`)

```python
@dataclass
class Seed:
    seed_id: str           # UUID v4
    title: str
    description: str       # combined text (abstract + claims + user input)
    doi: str | None
    patent_no: str | None
    semantic_vector: np.ndarray  # shape (384,) from e5-small
    source_type: str       # 'text' | 'doi' | 'patent' | 'mixed'
    created_at: str        # ISO 8601

def parse_seed(text: str, doi: str | None, patent_no: str | None) -> Seed:
    """Parse user input into a unified Seed object.

    Fetches paper abstract from Semantic Scholar if doi is given.
    Generates semantic_vector using the shared EmbeddingModel.
    """
```

### 2.2 Scoring Engine (`scoring/`)

Each scorer implements a common protocol:

```python
class FeatureScorer(Protocol):
    name: str
    weight: float

    def score(self, seed: Seed, company: CompanyRow) -> FeatureResult:
        ...

@dataclass
class FeatureResult:
    value: float          # 0.0 - 1.0
    rationale: str        # human-readable explanation (1-2 sentences)
```

Scorers are registered in `config.py`:

```python
# Phase 1 config
SCORERS = [
    {"name": "tech_prox",  "weight": 0.35, "enabled": True},
    {"name": "abs_cap",    "weight": 0.35, "enabled": True},
    {"name": "past_ties",  "weight": 0.30, "enabled": True},
    {"name": "need_fit",   "weight": 0.00, "enabled": False},  # Phase 2
    {"name": "trl_compat", "weight": 0.00, "enabled": False},  # Phase 3
    {"name": "open_inno",  "weight": 0.00, "enabled": False},  # Phase 3
]
```

### 2.3 Matcher (`matcher.py`)

```python
@dataclass
class MatchResult:
    seed: Seed
    rankings: list[RankedCompany]  # sorted by total_score desc
    executed_at: str
    duration_sec: float

@dataclass
class RankedCompany:
    rank: int
    edinet_code: str
    company_name: str
    industry: str
    total_score: float
    feature_scores: dict[str, FeatureResult]
    recommended_mode: str  # 'joint_research' | 'license' | 'contract' | 'long_term'

def run_match(seed: Seed, top_n: int = 10) -> MatchResult:
    """Score all companies against the seed, return top N."""
```

### 2.4 Reporter (`reporter.py`)

```python
def to_markdown(result: MatchResult) -> str:
    """Render match result as Markdown report."""

def to_json(result: MatchResult) -> dict:
    """Render match result as JSON-serializable dict."""
```

### 2.5 Embedding Model (shared utility)

```python
class EmbeddingModel:
    """Singleton wrapper around sentence-transformers.

    Loaded once, shared across TechProx and NeedFit scorers.
    Uses intfloat/multilingual-e5-small (384 dims).
    """
    def encode(self, texts: list[str]) -> np.ndarray: ...
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float: ...
```

## 3. Data Flow

### 3.1 Matching Flow (single seed)

```
User Input (text/DOI/patent)
    │
    ▼
[Seed Parser]
    │ Seed object (with semantic_vector)
    ▼
[Matcher.run_match()]
    │
    ├─→ Load all companies from SQLite (in-memory DataFrame)
    │
    ├─→ For each enabled scorer:
    │     ├─→ TechProx: cosine_sim(seed.vector, company.rd_text_vector) → f(x)=4x(1-x)
    │     ├─→ AbsCap: normalized(rd_intensity, rd_expense_log, patent_density)
    │     └─→ PastTies: log1p(joint_patents + joint_papers)
    │
    ├─→ Weighted sum → total_score per company
    │
    ├─→ Sort desc, take top N
    │
    ├─→ Generate rationale text for each
    │
    ▼
[Reporter]
    │ Markdown + JSON output
    ▼
[Save to DB: matches table]
    │
    ▼
[Display in Web UI or CLI]
```

### 3.2 Data Acquisition Flow (batch)

```
[scripts/load_companies.py]
    │
    ├─→ Read IR Collector DB (ir-collector/data/ir.db)
    │     └─→ SELECT DISTINCT edinet_code, sec_code, filer_name FROM documents
    │          WHERE doc_type_code = '120'
    │
    ├─→ For each company, extract from latest 有報 PDF:
    │     ├─→ R&D section text (研究開発活動)
    │     ├─→ Financial metrics (売上高, 営業利益, 研究開発費)
    │     └─→ Employee info
    │
    ├─→ Generate rd_text_vector using EmbeddingModel
    │
    ├─→ INSERT/UPDATE into matcher.db companies table
    │
    └─→ Log to acquisition_log
```

## 4. Sequence Diagrams

### 4.1 UC1: Web UI Single Seed Match

```
User          FastAPI         Matcher        ScoringEngine    SQLite
  │               │               │               │             │
  │ POST /match   │               │               │             │
  │ {text, doi}   │               │               │             │
  │──────────────>│               │               │             │
  │               │ parse_seed()  │               │             │
  │               │──────────────>│               │             │
  │               │               │ load companies│             │
  │               │               │──────────────────────────-->│
  │               │               │<────────────── rows ────────│
  │               │               │               │             │
  │               │               │ score_all()   │             │
  │               │               │──────────────>│             │
  │               │               │<── results ───│             │
  │               │               │               │             │
  │               │               │ save matches  │             │
  │               │               │──────────────────────────-->│
  │               │<── MatchResult│               │             │
  │               │               │               │             │
  │  HTML response│               │               │             │
  │<──────────────│               │               │             │
```

### 4.2 UC2: CLI Batch Match

```
Operator      CLI            Matcher        Reporter       Filesystem
  │             │               │               │             │
  │ batch cmd   │               │               │             │
  │ --input-dir │               │               │             │
  │────────────>│               │               │             │
  │             │ glob(*.md)    │               │             │
  │             │──────────────────────────────────────────-->│
  │             │<───── file list ───────────────────────────│
  │             │               │               │             │
  │             │ for each seed:│               │             │
  │             │ run_match()   │               │             │
  │             │──────────────>│  (same as UC1)│             │
  │             │<── result ────│               │             │
  │             │               │               │             │
  │             │ to_markdown() │               │             │
  │             │──────────────────────────────>│             │
  │             │<── md string ─────────────────│             │
  │             │ write file    │               │             │
  │             │──────────────────────────────────────────-->│
  │             │               │               │             │
  │  summary    │               │               │             │
  │<────────────│               │               │             │
```

### 4.3 UC3: Backcast Need Estimation (Phase 2)

```
Operator      CLI            ClaudeClient   EmbeddingModel  SQLite
  │             │               │               │             │
  │ refresh-needs│              │               │             │
  │────────────>│               │               │             │
  │             │ load companies│               │             │
  │             │ where needs   │               │             │
  │             │ is NULL       │               │             │
  │             │──────────────────────────────────────────-->│
  │             │<───── company rows ────────────────────────│
  │             │               │               │             │
  │             │ for each co:  │               │             │
  │             │ estimate_needs│               │             │
  │             │──────────────>│               │             │
  │             │<── JSON ──────│               │             │
  │             │               │               │             │
  │             │ embed(needs)  │               │             │
  │             │──────────────────────────────>│             │
  │             │<── vector ───────────────────│             │
  │             │               │               │             │
  │             │ UPDATE company│               │             │
  │             │──────────────────────────────────────────-->│
  │             │               │               │             │
  │  done (N co)│               │               │             │
  │<────────────│               │               │             │
```

## 5. IR Collector Integration

### 5.1 Dependency Setup

IR Collector is installed as an editable local dependency:

```bash
# In sangaku-matcher pyproject.toml:
[project]
dependencies = [
    "ir-collector @ file:///Users/nishimura+/projects/apps/ir-collector",
    ...
]

# Or via pip:
pip install -e ~/projects/apps/ir-collector
```

For this to work, IR Collector needs a minimal `pyproject.toml`:

```toml
# ir-collector/pyproject.toml (to be created)
[project]
name = "ir-collector"
version = "0.1.0"
dependencies = ["requests>=2.31", "python-dotenv>=1.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

### 5.2 How sangaku-matcher Uses IR Collector

The integration is **read-only at the DB level**:

```python
# sangaku_matcher/acquisition/edinet_loader.py

from pathlib import Path
from ir_collector.db import connect  # reuse the connection helper

IR_COLLECTOR_DB = Path(os.environ.get(
    "IR_COLLECTOR_DB_PATH",
    Path.home() / "projects/apps/ir-collector/data/ir.db"
))

def load_filers() -> list[dict]:
    """Get unique filers (companies) from IR Collector's document index."""
    with connect(IR_COLLECTOR_DB) as conn:
        rows = conn.execute("""
            SELECT DISTINCT edinet_code, sec_code, filer_name
            FROM documents
            WHERE doc_type_code = '120'
              AND edinet_code != ''
            ORDER BY filer_name
        """).fetchall()
    return [dict(r) for r in rows]

def get_latest_pdf_path(edinet_code: str) -> Path | None:
    """Get the most recent 有報 PDF for a company."""
    with connect(IR_COLLECTOR_DB) as conn:
        row = conn.execute("""
            SELECT pdf_path FROM documents
            WHERE edinet_code = ? AND doc_type_code = '120' AND pdf_path IS NOT NULL
            ORDER BY submit_datetime DESC LIMIT 1
        """, (edinet_code,)).fetchone()
    if row and row["pdf_path"]:
        return Path(os.environ.get(
            "IR_COLLECTOR_STORAGE_PATH",
            Path.home() / "projects/apps/ir-collector"
        )) / row["pdf_path"]
    return None
```

### 5.3 PDF Text Extraction

For Phase 1, use `pdfplumber` to extract text from 有報 PDFs:

```python
# sangaku_matcher/acquisition/pdf_extractor.py

import pdfplumber

def extract_rd_section(pdf_path: Path) -> str | None:
    """Extract the 研究開発活動 section from a 有報 PDF.

    Strategy: scan pages for the header '研究開発活動', then
    collect text until the next major section header.
    """
```

This avoids XBRL parsing complexity in Phase 1.

## 6. Performance Considerations

### 6.1 In-Memory Company Data

With 4,000 companies, the full company table fits in memory (~50MB):
- 4,000 rows x ~10KB text per row ≈ 40MB
- Embedding vectors: 4,000 x 384 dims x 4 bytes = 6MB

Load once at startup, keep as pandas DataFrame or dict for fast scoring.

### 6.2 Embedding Computation

Pre-compute and store `rd_text_vector` for all companies during data load.
At match time, only the seed vector is computed (single encode call).
Cosine similarity: numpy vectorized operation, ~1ms for 4,000 comparisons.

### 6.3 Target: <5s per match (excluding model load)

- Model load (one-time): ~3s
- Seed parsing + embedding: ~0.5s
- Scoring 4,000 companies: ~0.1s (vectorized numpy)
- Report generation: ~0.1s
- Total after warmup: **<1s per match**

## 7. Configuration (`config.py`)

```python
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Paths
    matcher_db_path: Path = Path("data/matcher.db")
    ir_collector_db_path: Path = Path.home() / "projects/apps/ir-collector/data/ir.db"
    ir_collector_storage_path: Path = Path.home() / "projects/apps/ir-collector"

    # API Keys
    anthropic_api_key: str = ""
    edinet_api_key: str = ""
    semantic_scholar_api_key: str = ""

    # Model
    embedding_model: str = "intfloat/multilingual-e5-small"
    llm_model: str = "claude-sonnet-4-6-20250514"

    # Scoring weights (Phase 1)
    w_tech_prox: float = 0.35
    w_abs_cap: float = 0.35
    w_past_ties: float = 0.30
    w_need_fit: float = 0.00
    w_trl_compat: float = 0.00
    w_open_inno: float = 0.00

    # Matching
    default_top_n: int = 10

    # Cost guard
    llm_monthly_budget_usd: float = 20.0

    class Config:
        env_file = ".env"
        env_prefix = ""
```
