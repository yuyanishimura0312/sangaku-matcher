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
class CollaborationHypothesis:
    """A concrete collaboration hypothesis for a matched company."""
    title: str
    description: str
    collab_type: str  # "joint_research", "license", "contract", "long_term"
    rationale: str


@dataclass
class RankedCompany:
    rank: int
    edinet_code: str
    company_name: str
    industry: str
    total_score: float
    feature_scores: dict[str, FeatureResult] = field(default_factory=dict)
    recommended_mode: str = "joint_research"
    collaboration_hypotheses: list[CollaborationHypothesis] = field(default_factory=list)
    overall_comment: str = ""


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


MODE_LABELS = {
    "joint_research": "共同研究",
    "license": "ライセンス供与",
    "contract": "受託研究",
    "long_term": "中長期研究契約",
}


def _generate_hypotheses(
    seed: Seed,
    company_name: str,
    industry: str,
    total_score: float,
    features: dict[str, FeatureResult],
    mode: str,
) -> tuple[list[CollaborationHypothesis], str]:
    """Generate collaboration hypotheses and an overall comment for a match.

    Returns (hypotheses, overall_comment).
    """
    tp = features.get("tech_prox", FeatureResult(0, ""))
    ac = features.get("abs_cap", FeatureResult(0, ""))
    pt = features.get("past_ties", FeatureResult(0, ""))

    hypotheses: list[CollaborationHypothesis] = []

    # Hypothesis based on tech proximity
    if tp.value > 0.7:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との技術ライセンス",
            description=(
                f"技術近接性が高い（{tp.value:.2f}）ため、当該シーズの技術を"
                f"{industry}分野の{company_name}にライセンス供与し、"
                f"事業化を加速する形態が有効と考えられます。"
            ),
            collab_type="license",
            rationale="技術領域の重複度が高く、企業側の既存事業との親和性が見込まれる",
        ))
    elif tp.value > 0.4:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との共同研究",
            description=(
                f"適度な技術的距離（{tp.value:.2f}）があり、"
                f"異なる技術知見を組み合わせた共同研究により"
                f"新たなイノベーションが生まれる可能性があります。"
            ),
            collab_type="joint_research",
            rationale="中程度の技術近接性は、相補的な知識結合に最適な距離",
        ))
    else:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との探索的研究連携",
            description=(
                f"技術領域の距離が大きい（{tp.value:.2f}）ものの、"
                f"異分野融合による革新的成果を目指した"
                f"長期的な探索型研究連携の可能性があります。"
            ),
            collab_type="long_term",
            rationale="技術的距離は大きいが、異分野融合のポテンシャルがある",
        ))

    # Hypothesis based on absorptive capacity
    if ac.value > 0.5:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}のR&D基盤を活用した実用化加速",
            description=(
                f"{company_name}は高い吸収能力（{ac.value:.2f}）を持ち、"
                f"外部知識の取り込みと事業化に長けています。"
                f"共同研究の成果を迅速にプロトタイプ化・製品化できる体制が期待されます。"
            ),
            collab_type="joint_research",
            rationale="高い吸収能力は、大学の研究成果を事業価値に変換する能力を示す",
        ))

    # Hypothesis based on past ties
    if pt.value > 0.3:
        hypotheses.append(CollaborationHypothesis(
            title="既存の産学連携チャネルの活用",
            description=(
                f"{company_name}は大学との連携実績があり（{pt.value:.2f}）、"
                f"共同研究のマネジメント経験や知財管理の仕組みが整っていると推察されます。"
                f"既存チャネルを通じた速やかな連携開始が期待できます。"
            ),
            collab_type="joint_research",
            rationale="過去の産学連携実績は、新たな連携のスムーズな立ち上げを後押しする",
        ))

    # Overall comment
    if total_score > 0.6:
        level = "高い"
        outlook = "複数の観点から連携可能性が認められ、具体的な連携協議を推奨します"
    elif total_score > 0.3:
        level = "中程度の"
        outlook = "一部の観点で連携可能性が認められますが、詳細な検討が必要です"
    else:
        level = "限定的な"
        outlook = "現時点では直接的な連携ポイントは限られますが、中長期的な視点での検討余地があります"

    overall = (
        f"{company_name}（{industry}）は、入力シーズとの連携可能性が{level}と評価されました"
        f"（総合スコア: {total_score:.2f}）。"
        f"技術近接性 {tp.value:.2f}、吸収能力 {ac.value:.2f}、連携実績 {pt.value:.2f} の"
        f"各観点から総合的に判断し、{outlook}。"
        f"推奨連携形態は「{MODE_LABELS.get(mode, mode)}」です。"
    )

    return hypotheses, overall


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
        mode = _infer_mode(total, tp_score)
        rounded_total = round(total, 4)
        hypotheses, overall_comment = _generate_hypotheses(
            seed, co["name"], co.get("industry", ""),
            rounded_total, features, mode,
        )
        rankings.append(RankedCompany(
            rank=rank_idx,
            edinet_code=co["edinet_code"],
            company_name=co["name"],
            industry=co.get("industry", ""),
            total_score=rounded_total,
            feature_scores=features,
            recommended_mode=mode,
            collaboration_hypotheses=hypotheses,
            overall_comment=overall_comment,
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
