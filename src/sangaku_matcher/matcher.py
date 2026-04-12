"""Matcher — orchestrate scorers across all companies for a seed.

Implements the Five-Layer Value Model (run_match) and
Multi-Exit Matching (run_multi_exit_match):
  Layer 1 (Technical):  tech_prox + need_fit
  Layer 2 (Relational): past_ties
  Layer 3 (Knowledge):  abs_cap
  Layer 4 (Future):     future_option
  Layer 5 (Ecosystem):  open_inno
  + ambition_fit, theme_breadth for multi-exit scoring
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from sangaku_matcher.config import (
    settings,
    EXIT_WEIGHTS,
    EXIT_LABELS,
    EXIT_DESCRIPTIONS,
)
from sangaku_matcher.db import connect
from sangaku_matcher.scoring import FeatureResult
from sangaku_matcher.scoring.tech_prox import TechProxScorer
from sangaku_matcher.scoring.need_fit import NeedFitScorer
from sangaku_matcher.scoring.abs_cap import AbsCapScorer
from sangaku_matcher.scoring.past_ties import PastTiesScorer
from sangaku_matcher.scoring.future_option import FutureOptionScorer
from sangaku_matcher.scoring.open_inno import OpenInnoScorer
from sangaku_matcher.scoring.theme_fit import ThemeFitScorer
from sangaku_matcher.scoring.ambition_fit import AmbitionFitScorer
from sangaku_matcher.scoring.theme_breadth import ThemeBreadthScorer
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
    """Load all companies with columns needed for 7 scorers.

    Uses LENGTH(rd_text) instead of full rd_text to save memory.
    """
    rows = conn.execute(
        """SELECT edinet_code, name, industry, revenue, rd_expense, rd_intensity,
                  rd_text_vector, needs_vector, open_inno_score,
                  employees, market_cap,
                  LENGTH(rd_text) AS rd_text_len,
                  humanities_needs_vector, humanities_needs_text,
                  tech_needs_vector, tech_needs_text
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
    """Infer collaboration mode from 7-dimensional score profile.

    Redesigned to give equal weight to humanities/social dimensions,
    reflecting Mode 2 knowledge production and transdisciplinary collaboration.
    """
    tp = features.get("tech_prox", FeatureResult(0, "")).value
    nf = features.get("need_fit", FeatureResult(0, "")).value
    ac = features.get("abs_cap", FeatureResult(0, "")).value
    pt = features.get("past_ties", FeatureResult(0, "")).value
    fo = features.get("future_option", FeatureResult(0, "")).value
    oi = features.get("open_inno", FeatureResult(0, "")).value
    hf = features.get("humanities_fit", FeatureResult(0, "")).value

    # Transdisciplinary: both tech AND humanities strong → integrated collaboration
    if nf > 0.4 and hf > 0.5:
        return "transdisciplinary"
    # Humanities-led: strong humanities, weak tech → social collaboration
    if hf > 0.5 and tp < 0.3:
        return "social_collaboration"
    # High tech overlap + high need fit → License
    if tp > 0.7 and nf > 0.5:
        return "license"
    # Strong need alignment + capability + trust → Joint Research
    if nf > 0.4 and ac > 0.4 and pt > 0.2:
        return "joint_research"
    # Humanities + OI → social innovation partnership
    if hf > 0.4 and oi > 0.4:
        return "social_collaboration"
    # High abs_cap + OI maturity but modest tech → Contract
    if ac > 0.5 and oi > 0.4 and tp < 0.5:
        return "contract"
    # High future option value → Long-term exploratory
    if fo > 0.5 and tp < 0.4:
        return "long_term"
    # High OI + existing ties → Joint Research (ecosystem play)
    if oi > 0.5 and pt > 0.3:
        return "joint_research"
    # Fallback on aggregate
    total = tp + nf + ac + pt + fo + oi + hf
    if total > 3.0:
        return "joint_research"
    if total > 2.0:
        return "contract"
    return "long_term"


MODE_LABELS = {
    "joint_research": "共同研究",
    "license": "ライセンス供与",
    "contract": "受託研究",
    "long_term": "中長期探索型連携",
    "social_collaboration": "社会課題連携",
    "transdisciplinary": "超学際連携",
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

    # Transdisciplinary hypothesis: tech AND humanities both strong
    if nf.value > 0.3 and hf.value > 0.4:
        hypotheses.append(CollaborationHypothesis(
            title=f"{company_name}との超学際的連携（技術×社会知の融合）",
            description=(
                f"技術ニーズ（{nf.value:.2f}）と人文社会科学的ニーズ（{hf.value:.2f}）"
                f"の双方が高く、技術開発と社会的価値創造を同時に追求する"
                f"超学際的（transdisciplinary）連携が期待できる。"
                f"技術的課題の解決と社会的インパクトの最大化を統合的に設計する"
                f"Mode 2型の知識生産が可能なケース。"
            ),
            collab_type="transdisciplinary",
            rationale="Gibbons (1994) Mode 2: 応用文脈での超学際的知識生産。技術と人文の交差が最大の革新を生む",
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
        f"{company_name}（{industry}）との連携可能性を七次元価値モデルで評価しました"
        f"（総合: {total_score:.2f}）。"
        f"技術近接 {tp.value:.2f}、技術ニーズ {nf.value:.2f}、"
        f"人文系 {hf.value:.2f}、知識吸収 {ac.value:.2f}、"
        f"関係資本 {pt.value:.2f}、将来価値 {fo.value:.2f}、"
        f"エコシステム {oi.value:.2f}。"
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

    humanities_fit = ThemeFitScorer()

    # Load theme data for GTA-based scoring
    with connect(settings.matcher_db_path) as theme_conn:
        humanities_fit.load_themes(theme_conn)
    humanities_fit.precompute_seed_themes(seed.semantic_vector)

    # Pre-compute similarity distributions for z-score normalization
    need_fit.precompute_distribution(seed.semantic_vector, companies)

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

        # Cross-dimensional synergy bonus (Mode 2 / transdisciplinary value)
        # Rewards companies where BOTH tech and humanities dimensions are strong,
        # capturing the "boundary-spanning" value that pure linear sums miss.
        if settings.w_synergy > 0:
            nf_val = features.get("need_fit", FeatureResult(0, "")).value
            hf_val = features.get("humanities_fit", FeatureResult(0, "")).value
            oi_val = features.get("open_inno", FeatureResult(0, "")).value
            # Geometric mean rewards balanced strength across dimensions
            synergy = (nf_val * hf_val) ** 0.5
            # OI readiness amplifies synergy (ecosystem enables cross-domain work)
            if oi_val > 0.3:
                synergy *= 1.0 + 0.2 * oi_val
            synergy = min(1.0, synergy)
            features["synergy"] = FeatureResult(
                value=round(synergy, 4),
                rationale=(
                    f"技術ニーズ（{nf_val:.2f}）×人文系ニーズ（{hf_val:.2f}）の"
                    f"領域横断的シナジー。"
                    + (f"OI体制（{oi_val:.2f}）による増幅効果あり。" if oi_val > 0.3 else "")
                ),
            )
            total += settings.w_synergy * synergy

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


# ---------------------------------------------------------------------------
# Multi-exit matching
# ---------------------------------------------------------------------------

@dataclass
class ExitRanking:
    """Ranking for a single exit type."""
    exit_type: str       # "rd", "new_domain", "exploratory"
    exit_label: str      # "共同研究（R&D）" etc.
    exit_description: str
    rankings: list[RankedCompany] = field(default_factory=list)


@dataclass
class MultiExitMatchResult:
    """Result of multi-exit matching across 3 collaboration types."""
    seed: Seed
    exits: list[ExitRanking] = field(default_factory=list)
    executed_at: str = ""
    duration_sec: float = 0.0
    company_count: int = 0


def _generate_exit_hypothesis(
    exit_type: str,
    company_name: str,
    features: dict[str, FeatureResult],
    theme_breadth_scorer: ThemeBreadthScorer,
) -> str:
    """Generate an exit-specific hypothesis text for a company."""
    nf = features.get("need_fit", FeatureResult(0, ""))
    tp = features.get("tech_prox", FeatureResult(0, ""))
    ac = features.get("abs_cap", FeatureResult(0, ""))
    pt = features.get("past_ties", FeatureResult(0, ""))
    hf = features.get("humanities_fit", FeatureResult(0, ""))
    oi = features.get("open_inno", FeatureResult(0, ""))
    af = features.get("ambition_fit", FeatureResult(0, ""))
    tb = features.get("theme_breadth", FeatureResult(0, ""))

    if exit_type == "rd":
        # Focus on need_fit, tech_prox, abs_cap
        parts = [
            f"{company_name}は技術ニーズとの一致度が高く(need_fit={nf.value:.2f})、"
            f"シーズが直接対応可能。"
        ]
        if tp.value > 0.3:
            parts.append(f"技術近接性(tech_prox={tp.value:.2f})も適度な距離にあり、知識移転が期待できる。")
        if ac.value > 0.3:
            parts.append(f"知識吸収能力(abs_cap={ac.value:.2f})が高く、共同研究契約に至りやすい候補。")
        if pt.value > 0.2:
            parts.append(f"過去の産学連携実績(past_ties={pt.value:.2f})もあり。")
        return " ".join(parts)

    elif exit_type == "new_domain":
        # Focus on ambition_fit, humanities_fit, open_inno
        parts = []
        if af.value > 0.2:
            af_rationale = af.rationale if af.rationale else ""
            parts.append(
                f"{company_name}は新領域への挑戦意欲が高い(ambition_fit={af.value:.2f})。"
                f"{af_rationale}"
            )
        else:
            parts.append(f"{company_name}の新領域適合度(ambition_fit={af.value:.2f})。")
        if hf.value > 0.2:
            parts.append(f"人文社会科学的知見との親和性(humanities_fit={hf.value:.2f})。")
        if oi.value > 0.3:
            parts.append(f"OI体制も整備(open_inno={oi.value:.2f})され、新領域探索の受け入れ態勢あり。")
        return " ".join(parts)

    else:  # exploratory
        # Focus on theme_breadth with matching theme details
        matching = theme_breadth_scorer.matching_themes
        n_themes = len(matching)
        if n_themes > 0:
            theme_names = [m["label"] for m in matching[:5]]
            parts = [
                f"{company_name}とは{n_themes}個のテーマで接点: {', '.join(theme_names)}。"
            ]
        else:
            parts = [f"{company_name}との直接的なテーマ接点は限定的。"]
        if hf.value > 0.2:
            parts.append(f"人文系親和性(humanities_fit={hf.value:.2f})。")
        if af.value > 0.2:
            parts.append(f"野心領域親和性(ambition_fit={af.value:.2f})。")
        parts.append("まず対話を通じて連携の接点を探索することを推奨。")
        return " ".join(parts)


def run_multi_exit_match(seed: Seed, top_n: int | None = None) -> MultiExitMatchResult:
    """Score all companies using 9 dimensions, then rank by 3 exit types."""
    top_n = top_n or settings.default_top_n
    start = time.time()

    with connect(settings.matcher_db_path) as conn:
        companies = _load_companies(conn)
        industry_stats = _load_industry_stats(conn)
        collab_data = _load_collaboration_data(conn)

    # Initialize all 9 scorers
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

    humanities_fit = ThemeFitScorer()

    ambition_fit = AmbitionFitScorer()

    theme_breadth = ThemeBreadthScorer()

    # Load theme data for GTA-based scoring
    with connect(settings.matcher_db_path) as theme_conn:
        humanities_fit.load_themes(theme_conn)
        ambition_fit.load_themes(theme_conn)
        theme_breadth.load_all_themes(theme_conn)

    humanities_fit.precompute_seed_themes(seed.semantic_vector)
    ambition_fit.precompute_seed_themes(seed.semantic_vector)
    theme_breadth.precompute_seed_themes(seed.semantic_vector)

    # Pre-compute similarity distributions for z-score normalization
    need_fit.precompute_distribution(seed.semantic_vector, companies)

    # All 9 scorers (no weights here; we apply exit-specific weights later)
    scorers = [
        tech_prox, need_fit, abs_cap, past_ties,
        future_option, open_inno, humanities_fit,
        ambition_fit, theme_breadth,
    ]

    # Score all companies once across all 9 dimensions + synergy
    scored_companies: list[tuple[dict, dict[str, FeatureResult]]] = []

    for co in companies:
        features: dict[str, FeatureResult] = {}
        for scorer in scorers:
            result = scorer.score(seed.semantic_vector, co)
            features[scorer.name] = result

        # Cross-dimensional synergy bonus (same as run_match)
        nf_val = features.get("need_fit", FeatureResult(0, "")).value
        hf_val = features.get("humanities_fit", FeatureResult(0, "")).value
        oi_val = features.get("open_inno", FeatureResult(0, "")).value
        synergy = (nf_val * hf_val) ** 0.5
        if oi_val > 0.3:
            synergy *= 1.0 + 0.2 * oi_val
        synergy = min(1.0, synergy)
        features["synergy"] = FeatureResult(
            value=round(synergy, 4),
            rationale=(
                f"技術ニーズ({nf_val:.2f})x人文系({hf_val:.2f})の"
                f"領域横断的シナジー。"
                + (f"OI体制({oi_val:.2f})による増幅効果あり。" if oi_val > 0.3 else "")
            ),
        )

        scored_companies.append((co, features))

    # Build rankings for each exit type
    exits: list[ExitRanking] = []

    for exit_type in ("rd", "new_domain", "exploratory"):
        weights = EXIT_WEIGHTS[exit_type]

        # Compute weighted total for each company
        exit_scored: list[tuple[float, dict, dict[str, FeatureResult]]] = []
        for co, features in scored_companies:
            total = 0.0
            for dim_name, w in weights.items():
                if w <= 0:
                    continue
                fr = features.get(dim_name, FeatureResult(0, ""))
                total += w * fr.value
            exit_scored.append((total, co, features))

        exit_scored.sort(key=lambda x: x[0], reverse=True)
        top = exit_scored[:top_n]

        rankings = []
        for rank_idx, (total, co, features) in enumerate(top, 1):
            # Re-score theme_breadth for this specific company to populate matching_themes
            theme_breadth.score(seed.semantic_vector, co)

            hypothesis = _generate_exit_hypothesis(
                exit_type, co["name"], features, theme_breadth,
            )

            rankings.append(RankedCompany(
                rank=rank_idx,
                edinet_code=co["edinet_code"],
                company_name=co["name"],
                industry=co.get("industry", ""),
                total_score=round(total, 4),
                feature_scores=features,
                recommended_mode=exit_type,
                overall_comment=hypothesis,
            ))

        exits.append(ExitRanking(
            exit_type=exit_type,
            exit_label=EXIT_LABELS[exit_type],
            exit_description=EXIT_DESCRIPTIONS[exit_type],
            rankings=rankings,
        ))

    duration = time.time() - start
    return MultiExitMatchResult(
        seed=seed,
        exits=exits,
        executed_at=datetime.now().isoformat(timespec="seconds"),
        duration_sec=round(duration, 2),
        company_count=len(companies),
    )


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
                    trl_compat, open_inno_mat, humanities_fit, synergy,
                    rationale, recommended_mode, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    fs.get("synergy", FeatureResult(0, "")).value,
                    " | ".join(f.rationale for f in fs.values() if f.rationale),
                    rc.recommended_mode,
                    result.executed_at,
                ),
            )
