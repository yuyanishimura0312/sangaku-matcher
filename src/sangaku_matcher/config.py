"""Application settings loaded from .env and environment variables."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


def _default_db_path() -> Path:
    """Resolve DB path relative to project root, not CWD."""
    # In Docker, CWD is /app; locally it may vary
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "matcher.db",
        Path("data/matcher.db"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


class Settings(BaseSettings):
    # --- Paths ---
    matcher_db_path: Path = Path("data/matcher.db")
    ir_collector_db_path: Path = (
        Path.home() / "projects/apps/ir-collector/data/ir.db"
    )
    ir_collector_storage_path: Path = (
        Path.home() / "projects/apps/ir-collector"
    )

    # --- API Keys (optional for Phase 1) ---
    anthropic_api_key: str = ""
    edinet_api_key: str = ""
    semantic_scholar_api_key: str = ""

    # --- Embedding Model ---
    embedding_model: str = "intfloat/multilingual-e5-small"

    # --- Scoring Weights (Five-Layer Value Model: 6 features) ---
    w_tech_prox: float = 0.25       # Layer 1: Technical Value
    w_need_fit: float = 0.20        # Layer 1: Technical Value (need-pull)
    w_abs_cap: float = 0.20         # Layer 3: Knowledge Value
    w_open_inno: float = 0.15       # Layer 5: Ecosystem Value
    w_past_ties: float = 0.10       # Layer 2: Relational Value
    w_future_option: float = 0.10   # Layer 4: Future Value

    # --- Matching ---
    default_top_n: int = 10

    # --- Web ---
    host: str = "127.0.0.1"
    port: int = 8000

    model_config = {"env_file": ".env", "env_prefix": "", "extra": "ignore"}


# Singleton instance — import this throughout the app
settings = Settings()
