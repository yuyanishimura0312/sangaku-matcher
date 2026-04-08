"""Matcher — orchestrate scoring across all companies for a seed."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from sangaku_matcher.config import settings
from sangaku_matcher.db import connect
from sangaku_matcher.scoring import FeatureResult
from sangaku_matcher.scoring.tech_prox import TechProxScorer
from sangaku_matcher.scoring.abs_cap import AbsCapScorer
from sangaku_matcher.scoring.past_ties import PastTiesScorer
from sangaku_matcher.seeds import Seed

logger = logging.getLogger(__name__)


@dataclass
class RankedCompany:
    rank: int
    edinet_code: str
    company_name: str
    industry: str
    total_score: float
    feature_scores: dict[str, FeatureResult] = field(default_factory=dict)
    recommended_mode: str = "joint_research"


@dataclass
class MatchResult:
    seed: Seed
    rankings: list[RankedCompany] = field(default_factory=list)
    executed_at: str = ""
    duration_sec: float = 0.0
    company_count: int = 0


def _load_companies(conn) -> list[dict]:
    """Load all companies from DB as list of dicts.

    Selects only the columns needed for scoring to reduce memory usage.
    """
    rows = conn.execute(
        """SELECT edinet_code, name, industry, rd_expense, rd_intensity,
                  rd_text_vector, needs_vector, open_inno_score
           FROM companies ORDER BY rd_expense DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def _load_industry_stats(conn) -> dict[str, tuple[float, float]]:
    """Load industry normalization stats."""
    rows = conn.execute("SELECT * FROM industry_stats").fetchall()
    return {r["industry"]: (r["mean_rd_int"], r["std_rd_int"]) for r in rows}


def _load_collaboration_data(conn) -> dict[str, dict[str, int]]:
    """Load collaboration data grouped by company.

    Uses defaultdict to avoid repeated .setdefault()/.get() overhead.
    """
    rows = conn.execute("SELECT * FROM collaborations").fetchall()
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        result[r["edinet_code"]][r["university_name"]] += r["count"]
    return result


def _infer_mode(total_score: float, tech_prox: float) -> str:
    """Suggest collaboration mode based on scores."""
    if tech_prox > 0.8:
        return "license"
    if total_score > 0.6:
        return "joint_research"
    if total_score > 0.3:
        return "contract"
    return "long_term"


def run_match(seed: Seed, top_n: int | None = None) -> MatchResult:
    """Score all companies against the seed, return top N.

    This is the main entry point for matching.
    """
    top_n = top_n or settings.default_top_n
    start = time.time()

    # Consolidate into a single DB connection for all data loading
    with connect(settings.matcher_db_path) as conn:
        companies = _load_companies(conn)
        industry_stats = _load_industry_stats(conn)
        collab_data = _load_collaboration_data(conn)

    # Initialize scorers — weights managed here, not inside scorers
    tech_prox = TechProxScorer()
    abs_cap = AbsCapScorer()
    abs_cap.set_industry_stats(industry_stats)
    past_ties = PastTiesScorer()
    past_ties.set_collaboration_data(collab_data)

    scorers = [
        (tech_prox, settings.w_tech_prox),
        (abs_cap, settings.w_abs_cap),
        (past_ties, settings.w_past_ties),
    ]

    scored: list[tuple[float, dict, dict[str, FeatureResult]]] = []

    for co in companies:
        features: dict[str, FeatureResult] = {}
        total = 0.0
        for scorer, weight in scorers:
            if weight <= 0:
                continue
            result = scorer.score(seed.semantic_vector, co)
            features[scorer.name] = result
            total += weight * result.value
        scored.append((total, co, features))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    rankings = []
    for rank_idx, (total, co, features) in enumerate(top, 1):
        tp_score = features.get("tech_prox", FeatureResult(0, "")).value
        rankings.append(RankedCompany(
            rank=rank_idx,
            edinet_code=co["edinet_code"],
            company_name=co["name"],
            industry=co.get("industry", ""),
            total_score=round(total, 4),
            feature_scores=features,
            recommended_mode=_infer_mode(total, tp_score),
        ))

    duration = time.time() - start
    result = MatchResult(
        seed=seed,
        rankings=rankings,
        executed_at=datetime.now().isoformat(timespec="seconds"),
        duration_sec=round(duration, 2),
        company_count=len(companies),
    )

    # Save to DB
    _save_match_result(result)
    return result


def _save_match_result(result: MatchResult) -> None:
    """Persist seed and match results to DB."""
    with connect(settings.matcher_db_path) as conn:
        # Save seed
        conn.execute(
            """INSERT OR REPLACE INTO seeds
               (seed_id, title, description, doi, patent_no, semantic_vector, source_type, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.seed.seed_id,
                result.seed.title,
                result.seed.description,
                result.seed.doi,
                result.seed.patent_no,
                result.seed.semantic_vector.tobytes(),
                result.seed.source_type,
                result.seed.created_at,
            ),
        )
        # Save matches
        for rc in result.rankings:
            fs = rc.feature_scores
            conn.execute(
                """INSERT INTO matches
                   (seed_id, edinet_code, rank, total_score,
                    tech_prox, abs_cap, need_fit, past_ties,
                    trl_compat, open_inno_mat, rationale, recommended_mode, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.seed.seed_id,
                    rc.edinet_code,
                    rc.rank,
                    rc.total_score,
                    fs.get("tech_prox", FeatureResult(0, "")).value,
                    fs.get("abs_cap", FeatureResult(0, "")).value,
                    fs.get("need_fit", FeatureResult(0, "")).value,
                    fs.get("past_ties", FeatureResult(0, "")).value,
                    fs.get("trl_compat", FeatureResult(0, "")).value,
                    fs.get("open_inno", FeatureResult(0, "")).value,
                    " | ".join(f.rationale for f in fs.values() if f.rationale),
                    rc.recommended_mode,
                    result.executed_at,
                ),
            )
