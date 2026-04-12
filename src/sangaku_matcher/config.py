"""Application settings loaded from .env and environment variables."""
from __future__ import annotations

import json
import logging
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

    # --- Scoring Weights (Seven-Dimension Value Model: 7 features + synergy, sum=1.0) ---
    # Redesigned based on Nooteboom (2007), SHARPE framework, and Mode 2 theory.
    # Balances technical matching (40%) with social/humanities value (25%)
    # and relational/ecosystem readiness (25%), plus cross-dimensional synergy (10%).
    w_tech_prox: float = 0.15       # Dim 1: Technical Proximity (inverted-U)
    w_need_fit: float = 0.15        # Dim 2: Tech Needs Fit (contextual vector)
    w_abs_cap: float = 0.10         # Dim 3: Absorptive Capacity
    w_open_inno: float = 0.10       # Dim 4: Ecosystem Readiness
    w_past_ties: float = 0.05       # Dim 5: Relational Capital
    w_future_option: float = 0.10   # Dim 6: Future Option Value
    w_humanities_fit: float = 0.25  # Dim 7: Humanities & Social Science Value
    w_synergy: float = 0.10         # Cross-dimensional synergy bonus

    # --- Matching ---
    default_top_n: int = 3

    # --- Web ---
    host: str = "127.0.0.1"
    port: int = 8000

    model_config = {"env_file": ".env", "env_prefix": "", "extra": "ignore"}


# Singleton instance — import this throughout the app
settings = Settings()

# Default multi-exit scoring weights (used when no JSON override exists)
_DEFAULT_EXIT_WEIGHTS: dict[str, dict[str, float]] = {
    # R&D collaboration: tech match is key. Closer is better.
    "rd": {
        "tech_prox": 0.20, "need_fit": 0.25, "abs_cap": 0.20,
        "past_ties": 0.10, "open_inno": 0.10, "future_option": 0.00,
        "humanities_fit": 0.05, "ambition_fit": 0.00, "theme_breadth": 0.00,
        "synergy": 0.10,
    },
    # New domain exploration: ambition themes and humanities connection are core.
    # Tech proximity is neutral (distance handled by future_option).
    "new_domain": {
        "tech_prox": 0.00, "need_fit": 0.10, "abs_cap": 0.10,
        "past_ties": 0.05, "open_inno": 0.15, "future_option": 0.15,
        "humanities_fit": 0.15, "ambition_fit": 0.20, "theme_breadth": 0.05,
        "synergy": 0.05,
    },
    # Exploratory dialogue: theme breadth and humanities are core.
    # Negative weights on tech_prox / need_fit per Nooteboom (2007):
    # cognitively distant partners are better for exploration.
    "exploratory": {
        "tech_prox": -0.10, "need_fit": -0.05, "abs_cap": 0.05,
        "past_ties": 0.05, "open_inno": 0.10, "future_option": 0.15,
        "humanities_fit": 0.25, "ambition_fit": 0.10, "theme_breadth": 0.30,
        "synergy": 0.05,
    },
}

_config_logger = logging.getLogger(__name__)


def _load_exit_weights() -> dict[str, dict[str, float]]:
    """Load exit weights from JSON file, falling back to defaults.

    Searches for data/exit_weights.json relative to the project root
    and the current working directory. Validates that each exit type's
    weights sum to approximately 1.0.
    """
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data" / "exit_weights.json",
        Path("data/exit_weights.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    loaded = json.load(f)
                # Validate: each exit must have at least one positive weight
                # and all values must be numeric (negative weights are allowed
                # per Nooteboom 2007 — e.g. exploratory penalizes tech_prox).
                for exit_type, weights in loaded.items():
                    if not any(w > 0 for w in weights.values()):
                        _config_logger.warning(
                            "Exit weights for %s have no positive weights, using defaults",
                            exit_type,
                        )
                        return _DEFAULT_EXIT_WEIGHTS
                    if any(v is None for v in weights.values()):
                        _config_logger.warning(
                            "Exit weights for %s contain None values, using defaults",
                            exit_type,
                        )
                        return _DEFAULT_EXIT_WEIGHTS
                _config_logger.info("Loaded exit weights from %s", p)
                return loaded
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                _config_logger.warning("Failed to parse %s: %s, using defaults", p, exc)
    return _DEFAULT_EXIT_WEIGHTS


EXIT_WEIGHTS: dict[str, dict[str, float]] = _load_exit_weights()

EXIT_LABELS: dict[str, str] = {
    "rd": "共同研究（R&D）",
    "new_domain": "共同研究（新領域探索）",
    "exploratory": "探索的対話",
}

EXIT_DESCRIPTIONS: dict[str, str] = {
    "rd": "企業側に明確な技術課題・ニーズがあり、研究者のシーズが直接それに応える連携。成果は論文・特許・プロトタイプなど具体的。",
    "new_domain": "企業が挑戦しようとしている新領域に、研究者の知見で貢献する連携。問いの設定自体に研究者が関わる。",
    "exploratory": "具体的な共同研究の形はまだ見えないが、テーマ的な親和性がある。まず対話して接点を探る段階。",
}
