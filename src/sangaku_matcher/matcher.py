"""Matcher — orchestrate 6 scorers across all companies for a seed.

Implements the Five-Layer Value Model:
  Layer 1 (Technical):  tech_prox + need_fit
  Layer 2 (Relational): past_ties
  Layer 3 (Knowledge):  abs_cap
  Layer 4 (Future):     future_option
  Layer 5 (Ecosystem):  open_inno
"""
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
from sangaku_matcher.scoring.need_fit import NeedFitScorer
from sangaku_matcher.scoring.abs_cap import AbsCapScorer
from sangaku_matcher.scoring.past_ties import PastTiesScorer
from sangaku_matcher.scoring.future_option import FutureOptionScorer
from sangaku_matcher.scoring.open_inno import OpenInnoScorer
from sangaku_matcher.scoring.humanities_fit import HumanitiesFitScorer
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
    """Load all companies with columns needed for 6 scorers.

    Uses LENGTH(rd_text) instead of full rd_text to save memory.
    """
    rows = conn.execute(
        """SELECT edinet_code, name, industry, revenue, rd_expense, rd_intensity,
                  rd_text_vector, needs_vector, open_inno_score,
                  employees, market_cap,
                  LENGTH(rd_text) AS rd_text_len,
                  humanities_needs_vector, humanities_needs_text
           FROM companies ORDER BY rd_expense DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def _load_industry_stats(conn) -> dict[str, tuple[float, float]]:
    rows = conn.execute("SELECT * FROM industry_stats").fetchall()
    return {r["industry"]: (r["mean_rd_int"], r["std_rd_int"]) for r in rows}


def _load_collaboration_data(conn) -> dict[str, dict[str, dict]]:
    """Load collaboration data with count and last_year per university.

    Returns: {edinet_code: {uni_name: {"count": int, "last_year": int}}}
    """
    rows = conn.execute("SELECT * FROM collaborations").fetchall()
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        ec = r["edinet_code"]
        uni = r["university_name"]
        count = r["count"]
        try:
            last_year = r["last_year"] or 0
        except (IndexError, KeyError):
            last_year = 0
        if uni in result[ec]:
            result[ec][uni]["count"] += count
            if last_year > result[ec][uni]["last_year"]:
                result[ec][uni]["last_year"] = last_year
        else:
            result[ec][uni] = {"count": count, "last_year": last_year}
    return dict(result)


def _infer_mode(features: dict[str, FeatureResult]) -> str:
    """Infer collaboration mode from 6-dimensional score profile."""
    tp = features.get("tech_prox", FeatureResult(0, "")).value
    nf = features.get("need_fit", FeatureResult(0, "")).value
    ac = features.get("abs_cap", FeatureResult(0, "")).value
    pt = features.get("past_ties", FeatureResult(0, "")).value
    fo = features.get("future_option", FeatureResult(0, "")).value
    oi = features.get("open_inno", FeatureResult(0, "")).value

    hf = features.get("humanities_fit", FeatureResult(0, "")).value

    # High humanities fit + low tech → Social collaboration
    if hf > 0.5 and tp < 0.3:
        return "social_collaboration"
    # High tech overlap + high need fit → License
    if tp > 0.7 and nf > 0.5:
        return "license"
    # Strong need alignment + capability + trust → Joint Research
    if nf > 0.4 and ac > 0.4 and pt > 0.2:
        return "joint_research"
    # High abs_cap + OI maturity but modest tech/need fit → Contract
    if ac > 0.5 and oi > 0.4 and tp < 0.5:
        return "contract"
    # High future option value → Long-term exploratory
    if fo > 0.5 and tp < 0.4:
        return "long_term"
    # High OI + existing ties → Joint Research (ecosystem play)
    if oi > 0.5 and pt > 0.3:
        return "joint_research"
    # Fallback on aggregate
    total = tp + nf + ac + pt + fo + oi
    if total > 2.5:
        return "joint_research"
    if total > 1.5:
        return "contract"
    return "long_term"


MODE_LABELS = {
    "joint_research": "共同研究",
    "license": "ライセンス供与",
    "contract": "受託研究",
    "long_term": "中長期探索型連携",
    "social_collaboration": "社会課題連携",
}


def _generate_hypotheses(
    seed: Seed,
    company_name: str,
    industry: str,
    total_score: float,
    features: dict[str, FeatureResult],
    mode: str,
) -> tuple[list[CollaborationHypothesis], str]:
    """Generate collaboration hypotheses from 7-dimensional score profile."""
    tp = features.get("tech_prox", FeatureResult(0, ""))
    nf = features.get("need_fit", FeatureResult(0, ""))
    ac = features.get("abs_cap", FeatureResult(0, ""))
    pt = features.get("past_ties", FeatureResult(0, ""))
    fo = features.get("future_option", FeatureResult(0, ""))
    oi = features.get("open_inno", FeatureResult(0, ""))
    hf = features.get("humanities_fit", FeatureResult(0, ""))

    hypotheses: list[CollaborationHypothesis] = []

    # Need fit hypothesis (highest priority for backcast system)
    if nf.value > 0.5:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}の推定技術ニーズへの直接対応",
            description=(
                f"推定ニーズとの高い一致（{nf.value:.2f}）から、"
                f"{company_name}が{industry}分野で抱える技術課題に"
                f"当該シーズが直接対応できる可能性が高い。"
                f"ニーズ起点のアプローチにより、初期合意の確度が高まる。"
            ),
            collab_type="joint_research",
            rationale="バックキャスト推定によるニーズ一致は、企業側の受容性を高める",
        ))

    # Tech proximity hypothesis
    if tp.value > 0.7:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との技術ライセンス",
            description=(
                f"技術近接性が高い（{tp.value:.2f}）ため、当該シーズの技術を"
                f"{industry}分野の{company_name}にライセンス供与し、"
                f"事業化を加速する形態が有効と考えられる。"
            ),
            collab_type="license",
            rationale="技術領域の重複度が高く、既存事業との親和性が見込まれる",
        ))
    elif tp.value > 0.4:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との共同研究",
            description=(
                f"適度な技術的距離（{tp.value:.2f}）があり、"
                f"異なる技術知見を組み合わせた共同研究により"
                f"新たなイノベーションが生まれる可能性がある。"
            ),
            collab_type="joint_research",
            rationale="中程度の技術近接性は、相補的な知識結合に最適な距離",
        ))

    # Absorptive capacity hypothesis
    if ac.value > 0.5:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}の高い知識吸収能力を活用した実用化加速",
            description=(
                f"知識吸収能力（{ac.value:.2f}）が高く、認知的近接性も考慮すると"
                f"研究成果を効率的に取り込み、プロトタイプ化・事業化に"
                f"つなげる組織体制が整っていると推察される。"
            ),
            collab_type="joint_research",
            rationale="Cohen & Levinthal: R&D投資が高い企業は外部知識の吸収に長ける",
        ))

    # Past ties hypothesis
    if pt.value > 0.3:
        hypotheses.append(CollaborationHypothesis(
            title="既存の産学連携チャネルの活用",
            description=(
                f"{company_name}は大学との連携実績があり（{pt.value:.2f}）、"
                f"多様な大学パートナーとの関係性を持つ。"
                f"既存チャネルを通じた速やかな連携開始が期待できる。"
            ),
            collab_type="joint_research",
            rationale="過去の連携実績とネットワーク多様性は、新たな連携を後押しする",
        ))

    # Future option hypothesis
    if fo.value > 0.5:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との異分野融合による将来オプション創出",
            description=(
                f"技術距離は大きいものの、{company_name}のR&D活動の幅と"
                f"戦略的柔軟性（{fo.value:.2f}）から、異分野融合による"
                f"セレンディピティの可能性がある。リアルオプション的な"
                f"小規模PoC投資として位置づけが可能。"
            ),
            collab_type="long_term",
            rationale="McGrath: R&Dリアルオプション理論に基づく探索的投資",
        ))

    # Open innovation hypothesis
    if oi.value > 0.5:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}のオープンイノベーション体制を活用した連携",
            description=(
                f"OI成熟度が高く（{oi.value:.2f}）、エコシステム構築に"
                f"積極的な姿勢が見られる。CVC、アクセラレータ、産学連携窓口等の"
                f"既存の受け入れ体制を通じた迅速な連携開始が期待できる。"
            ),
            collab_type="joint_research",
            rationale="Chesbrough: OI成熟企業は外部シーズの受容性が構造的に高い",
        ))

    # Humanities collaboration hypothesis
    if hf.value > 0.4:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との人文社会科学的知見を活かした連携",
            description=(
                f"人文社会科学系ニーズとの高い親和性（{hf.value:.2f}）。"
                f"{hf.rationale} "
                f"技術開発とは異なる視点から、社会洞察・未来洞察・"
                f"組織文化・倫理等の領域での知的連携が期待できる。"
            ),
            collab_type="joint_research",
            rationale="SHARPE/Structural Holes: 人文社会科学と産業界の知識仲介による価値創造",
        ))
    elif hf.value > 0.25:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との社会課題視点での探索的連携",
            description=(
                f"人文系ニーズとの中程度の親和性（{hf.value:.2f}）。"
                f"直接的な技術連携に加え、社会的文脈の理解や"
                f"ステークホルダー対話の設計など、補完的な連携が考えられる。"
            ),
            collab_type="long_term",
            rationale="Mode 2知識生産: 社会的文脈埋め込み型の超学際的連携",
        ))

    # Cross-dimensional combination hypothesis
    if nf.value > 0.3 and oi.value > 0.3 and ac.value > 0.3:
        hypotheses.append(CollaborationHypothesis(
            title=f"ニーズ・OI・吸収能力の三条件が揃った有望連携",
            description=(
                f"推定ニーズとの一致（{nf.value:.2f}）、"
                f"OI体制（{oi.value:.2f}）、吸収能力（{ac.value:.2f}）の"
                f"三条件がそろっており、連携成功確率が特に高い候補。"
                f"優先的にアプローチすることを推奨する。"
            ),
            collab_type="joint_research",
            rationale="複数の価値層が同時に高いケースは連携成功率が有意に高い",
        ))

    # Fallback: ensure at least one hypothesis
    if not hypotheses:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との探索的研究連携",
            description=(
                f"現時点での直接的な連携ポイントは限られるが、"
                f"中長期的な視点での探索型連携の可能性がある。"
            ),
            collab_type="long_term",
            rationale="スコアは低いが、未知の接点が存在する可能性を排除しない",
        ))

    # Overall comment: 6-dimensional summary
    if total_score > 0.5:
        level = "高い"
        outlook = "複数の観点から連携可能性が認められ、具体的な連携協議を推奨"
    elif total_score > 0.25:
        level = "中程度の"
        outlook = "一部の観点で連携可能性があり、詳細な検討が必要"
    else:
        level = "限定的な"
        outlook = "直接的な連携ポイントは限られるが、中長期的な検討余地あり"

    overall = (
        f"{company_name}（{industry}）との連携可能性を五層価値モデルで評価しました"
        f"（総合: {total_score:.2f}）。"
        f"技術近接 {tp.value:.2f}、ニーズ適合 {nf.value:.2f}、"
        f"知識吸収 {ac.value:.2f}、関係資本 {pt.value:.2f}、"
        f"将来価値 {fo.value:.2f}、エコシステム {oi.value:.2f}。"
        f"連携可能性は{level}と評価され、{outlook}。"
        f"推奨連携形態: {MODE_LABELS.get(mode, mode)}。"
    )

    return hypotheses, overall


def run_match(seed: Seed, top_n: int | None = None) -> MatchResult:
    """Score all companies against the seed using 6 scorers, return top N."""
    top_n = top_n or settings.default_top_n
    start = time.time()

    with connect(settings.matcher_db_path) as conn:
        companies = _load_companies(conn)
        industry_stats = _load_industry_stats(conn)
        collab_data = _load_collaboration_data(conn)

    # Initialize all 6 scorers
    tech_prox = TechProxScorer()

    need_fit = NeedFitScorer()

    abs_cap = AbsCapScorer()
    abs_cap.set_industry_stats(industry_stats)

    past_ties = PastTiesScorer()
    past_ties.set_collaboration_data(collab_data)

    future_option = FutureOptionScorer()
    future_option.set_industry_stats(industry_stats)

    open_inno = OpenInnoScorer()
    open_inno.set_collaboration_data(collab_data)
    open_inno.set_industry_stats(industry_stats)

    humanities_fit = HumanitiesFitScorer()

    scorers = [
        (tech_prox, settings.w_tech_prox),
        (need_fit, settings.w_need_fit),
        (abs_cap, settings.w_abs_cap),
        (past_ties, settings.w_past_ties),
        (future_option, settings.w_future_option),
        (open_inno, settings.w_open_inno),
        (humanities_fit, settings.w_humanities_fit),
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

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    rankings = []
    for rank_idx, (total, co, features) in enumerate(top, 1):
        mode = _infer_mode(features)
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

    _save_match_result(result)
    return result


def _save_match_result(result: MatchResult) -> None:
    """Persist seed and match results to DB."""
    with connect(settings.matcher_db_path) as conn:
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
        for rc in result.rankings:
            fs = rc.feature_scores
            conn.execute(
                """INSERT INTO matches
                   (seed_id, edinet_code, rank, total_score,
                    tech_prox, abs_cap, need_fit, past_ties,
                    trl_compat, open_inno_mat, humanities_fit,
                    rationale, recommended_mode, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.seed.seed_id,
                    rc.edinet_code,
                    rc.rank,
                    rc.total_score,
                    fs.get("tech_prox", FeatureResult(0, "")).value,
                    fs.get("abs_cap", FeatureResult(0, "")).value,
                    fs.get("need_fit", FeatureResult(0, "")).value,
                    fs.get("past_ties", FeatureResult(0, "")).value,
                    fs.get("future_option", FeatureResult(0, "")).value,
                    fs.get("open_inno", FeatureResult(0, "")).value,
                    fs.get("humanities_fit", FeatureResult(0, "")).value,
                    " | ".join(f.rationale for f in fs.values() if f.rationale),
                    rc.recommended_mode,
                    result.executed_at,
                ),
            )
