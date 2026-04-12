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
    tech_prox.precompute_distribution(seed.semantic_vector, companies)

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
class ResearcherProfile:
    """Researcher type inferred from theme affinity pattern.

    Based on Stokes (1997) Pasteur's Quadrant and D'Este & Perkmann (2011).
    Classifies researchers into four types and recommends the best exit route.
    """
    researcher_type: str   # "applied", "basic", "pasteur", "interdisciplinary"
    type_label: str        # "応用志向（エジソン型）" etc.
    recommended_exit: str  # "rd", "new_domain", "exploratory"
    recommendation_text: str
    tech_affinity: float    # average affinity to tech themes
    humanities_affinity: float  # average affinity to humanities themes
    ambition_affinity: float   # average affinity to ambition themes


@dataclass
class MultiExitMatchResult:
    """Result of multi-exit matching across 3 collaboration types."""
    seed: Seed
    exits: list[ExitRanking] = field(default_factory=list)
    researcher_profile: ResearcherProfile | None = None
    executed_at: str = ""
    duration_sec: float = 0.0
    company_count: int = 0


def _generate_exit_hypothesis(
    exit_type: str,
    company_name: str,
    features: dict[str, FeatureResult],
    cached_matching_themes: list[dict],
) -> str:
    """Generate an exit-specific hypothesis text for a company.

    Args:
        exit_type: One of "rd", "new_domain", "exploratory".
        company_name: Display name of the target company.
        features: Scorer results keyed by scorer name.
        cached_matching_themes: Matching theme list captured during the initial
            score() call — avoids a redundant second score() invocation.

    Returns:
        Human-readable hypothesis string in Japanese.
    """
    nf = features.get("need_fit", FeatureResult(0, ""))
    tp = features.get("tech_prox", FeatureResult(0, ""))
    ac = features.get("abs_cap", FeatureResult(0, ""))
    pt = features.get("past_ties", FeatureResult(0, ""))
    hf = features.get("humanities_fit", FeatureResult(0, ""))
    oi = features.get("open_inno", FeatureResult(0, ""))
    af = features.get("ambition_fit", FeatureResult(0, ""))
    tb = features.get("theme_breadth", FeatureResult(0, ""))

    # Helper: extract top theme names from a scorer's rationale
    def _extract_theme_names(rationale: str, max_n: int = 3) -> list[str]:
        """Pull theme names from rationale like '高親和: テーマA(96%); テーマB(91%)'."""
        names = []
        if "高親和:" in rationale:
            after = rationale.split("高親和:")[1]
            for part in after.split(";"):
                part = part.strip()
                if "(" in part:
                    name = part.split("(")[0].strip()
                    if name and len(name) > 2:
                        names.append(name)
        return names[:max_n]

    # Helper: extract humanities_role from ambition_fit rationale
    def _extract_humanities_role(rationale: str) -> str:
        """Pull the humanities_role sentence from ambition_fit rationale."""
        # humanities_role follows the theme list, starts with a full sentence
        parts = rationale.split("。")
        for p in parts:
            p = p.strip()
            # humanities_role sentences typically reference academic disciplines
            if any(kw in p for kw in ["学の", "論の", "研究の", "学的", "倫理"]) and len(p) > 20:
                return p + "。"
        return ""

    if exit_type == "rd":
        # R&D: concrete tech alignment narrative
        parts = []
        parts.append(
            f"{company_name}は、有価証券報告書の分析から、"
            f"当該研究シーズと関連性の高い技術課題を抱えていると推定されます。"
        )
        if ac.value > 0.5 and pt.value > 0.2:
            parts.append(
                f"同社はR&D投資が活発で、大学との共同研究実績も確認されており、"
                f"産学連携の受け入れ体制が整った企業です。"
            )
        elif ac.value > 0.5:
            parts.append(
                f"同社はR&D投資が活発で、外部の研究知見を事業に取り込む"
                f"組織的な体制を持っています。"
            )
        elif pt.value > 0.2:
            parts.append(
                f"同社は大学との連携実績があり、"
                f"産学連携の進め方について一定の経験を有しています。"
            )
        parts.append(
            f"共同研究の具体的なテーマ設定に向けて、"
            f"まず技術担当者との面談を通じて課題の詳細を確認することを推奨します。"
        )
        return "".join(parts)

    elif exit_type == "new_domain":
        # New domain: ambition themes + how researcher can contribute
        theme_names = _extract_theme_names(af.rationale)
        hum_role = _extract_humanities_role(af.rationale)

        parts = []
        if theme_names:
            theme_str = "「" + "」「".join(theme_names[:2]) + "」"
            parts.append(
                f"{company_name}は、{theme_str}"
                f"といった新しい事業領域への挑戦を"
                f"有価証券報告書で示しています。"
            )
        else:
            parts.append(
                f"{company_name}は、新たな事業領域の開拓に"
                f"取り組む姿勢を有価証券報告書で示しています。"
            )
        if hum_role:
            parts.append(
                f"この新領域において、{hum_role}"
            )
        else:
            parts.append(
                f"こうした新領域への参入にあたっては、"
                f"研究者の専門的知見が問いの設定や方向性の検討に貢献できる可能性があります。"
            )
        if oi.value > 0.4:
            parts.append(
                f"同社はオープンイノベーション体制を整えており、"
                f"外部の研究者との新領域探索に対して組織的な受容性が高いと考えられます。"
            )
        return "".join(parts)

    else:  # exploratory
        # Exploratory: theme breadth grouped by axis
        n_themes = len(cached_matching_themes)
        if n_themes > 0:
            by_axis: dict[str, list[str]] = {}
            for m in cached_matching_themes[:10]:
                axis_label = {"humanities": "社会課題", "tech": "技術", "ambition": "新規事業"}.get(m.get("axis", ""), "")
                if axis_label:
                    by_axis.setdefault(axis_label, []).append(m["label"])

            parts = []
            parts.append(
                f"{company_name}とは、複数の分野にわたって対話の接点が見込まれます。"
            )
            topic_sentences = []
            for axis_label, names in by_axis.items():
                display_names = "「" + "」「".join(names[:2]) + "」"
                topic_sentences.append(f"{axis_label}の観点では{display_names}")
            if topic_sentences:
                parts.append("、".join(topic_sentences) + "などのテーマが挙げられます。")

            parts.append(
                f"まずは幅広い対話を通じて、"
                f"具体的な連携テーマの発見を目指すことを推奨します。"
            )
            return "".join(parts)
        else:
            return (
                f"{company_name}とは、直接的なテーマの重なりは限られますが、"
                f"異なる視点からの対話を通じて、新たな連携の接点が"
                f"見つかる可能性があります。"
            )


def _mmr_rerank(
    candidates: list[tuple[float, dict, dict, list]],
    top_n: int,
    lambda_param: float = 0.5,
) -> list[tuple[float, dict, dict, list]]:
    """Maximal Marginal Relevance reranking using R&D text vectors for diversity.

    Selects items that balance relevance (original score) with novelty
    (dissimilarity to already-selected items based on cosine distance of
    R&D text vectors).

    Args:
        candidates: List of (total_score, company_dict, features, themes).
        top_n: Number of items to select.
        lambda_param: Trade-off parameter. 0=pure diversity, 1=pure relevance.

    Returns:
        Reranked list of top_n candidates.
    """
    if len(candidates) <= top_n:
        return candidates

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    # Pre-extract and normalize R&D vectors for cosine similarity
    vectors: list[np.ndarray | None] = []
    for total, co, features, mt in candidates:
        vec_bytes = co.get("rd_text_vector")
        if vec_bytes and len(vec_bytes) > 0:
            vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)
        else:
            vectors.append(None)

    # First item: highest score
    remaining.sort(key=lambda i: candidates[i][0], reverse=True)
    selected.append(remaining.pop(0))

    while len(selected) < top_n and remaining:
        best_mmr = -float('inf')
        best_idx_in_remaining = 0

        for ri, ci in enumerate(remaining):
            relevance = candidates[ci][0]

            # Max cosine similarity to already-selected items
            max_sim = 0.0
            if vectors[ci] is not None:
                for si in selected:
                    if vectors[si] is not None:
                        sim = float(np.dot(vectors[ci], vectors[si]))
                        max_sim = max(max_sim, sim)

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx_in_remaining = ri

        selected.append(remaining.pop(best_idx_in_remaining))

    return [candidates[i] for i in selected]


def _infer_researcher_type(
    seed_vector: np.ndarray,
    humanities_themes: list[dict],
    tech_themes: list[dict],
    ambition_themes: list[dict],
) -> ResearcherProfile:
    """Infer researcher type from theme affinity pattern.

    Based on Stokes (1997) Pasteur's Quadrant and D'Este & Perkmann (2011).
    Computes average cosine similarity of the seed vector to each axis's
    theme centroids, then classifies into one of four types.
    """
    norm = np.linalg.norm(seed_vector)
    sv_norm = seed_vector / max(norm, 1e-10)

    def avg_affinity(themes: list[dict]) -> float:
        if not themes:
            return 0.0
        centroids = np.array([t["centroid"] for t in themes])
        sims = centroids @ sv_norm
        return float(np.mean(sims))

    tech_aff = avg_affinity(tech_themes)
    hum_aff = avg_affinity(humanities_themes)
    amb_aff = avg_affinity(ambition_themes)

    # Classification based on relative affinities:
    #   Tech-dominant  -> applied/Edison    -> R&D collaboration
    #   Humanities-dom -> basic/Bohr        -> exploratory dialogue
    #   Ambition-dom   -> Pasteur           -> new domain exploration
    #   Balanced       -> interdisciplinary -> new domain exploration
    if tech_aff > hum_aff + 0.02 and tech_aff > amb_aff:
        return ResearcherProfile(
            researcher_type="applied",
            type_label="応用志向（エジソン型）",
            recommended_exit="rd",
            recommendation_text=(
                "技術テーマへの親和性が高く、R&D共同研究が最も有望なルートです。"
                "企業の具体的な技術課題に直接応えられる連携が期待できます。"
            ),
            tech_affinity=tech_aff,
            humanities_affinity=hum_aff,
            ambition_affinity=amb_aff,
        )
    elif hum_aff > tech_aff + 0.02 and hum_aff > amb_aff:
        return ResearcherProfile(
            researcher_type="basic",
            type_label="基礎志向（ボーア型）",
            recommended_exit="exploratory",
            recommendation_text=(
                "人文社会科学テーマへの親和性が高く、探索的対話が最も有望なルートです。"
                "企業との対話から新たな研究課題や連携テーマが見つかる可能性があります。"
            ),
            tech_affinity=tech_aff,
            humanities_affinity=hum_aff,
            ambition_affinity=amb_aff,
        )
    elif amb_aff > tech_aff and amb_aff > hum_aff:
        return ResearcherProfile(
            researcher_type="pasteur",
            type_label="用途触発型（パスツール型）",
            recommended_exit="new_domain",
            recommendation_text=(
                "新領域・野心テーマへの親和性が高く、新領域探索型共同研究が最も有望なルートです。"
                "企業が挑戦する新領域に研究知見で貢献できます。"
            ),
            tech_affinity=tech_aff,
            humanities_affinity=hum_aff,
            ambition_affinity=amb_aff,
        )
    else:
        return ResearcherProfile(
            researcher_type="interdisciplinary",
            type_label="学際型",
            recommended_exit="new_domain",
            recommendation_text=(
                "技術・人文・野心の各テーマにバランスよく親和性があり、"
                "新領域探索型共同研究が有望です。"
                "分野横断的な視点を活かした連携が期待できます。"
            ),
            tech_affinity=tech_aff,
            humanities_affinity=hum_aff,
            ambition_affinity=amb_aff,
        )


def run_multi_exit_match(seed: Seed, top_n: int | None = None) -> MultiExitMatchResult:
    """Score all companies using 9 dimensions, then rank by 3 exit types.

    Runs a single pass over all companies scoring each on 9 dimensions
    (tech_prox, need_fit, abs_cap, past_ties, future_option, open_inno,
    humanities_fit, ambition_fit, theme_breadth) plus a synergy bonus.
    Results are then ranked separately for each of 3 exit types
    (rd, new_domain, exploratory) using exit-specific dimension weights.

    Args:
        seed: Researcher seed containing a semantic vector and metadata.
        top_n: Number of companies to include per exit type. Defaults to
            settings.default_top_n when None.

    Returns:
        MultiExitMatchResult with rankings grouped by exit type, timing
        metadata, and the total number of companies evaluated.
    """
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

    # Infer researcher type from theme affinities (Stokes 1997 / D'Este & Perkmann 2011)
    researcher_profile = _infer_researcher_type(
        seed.semantic_vector,
        humanities_themes=humanities_fit._themes or [],
        tech_themes=[
            t for t in (theme_breadth._all_themes or []) if t["axis"] == "tech"
        ],
        ambition_themes=ambition_fit._themes or [],
    )

    # Pre-compute similarity distributions for z-score normalization
    need_fit.precompute_distribution(seed.semantic_vector, companies)
    tech_prox.precompute_distribution(seed.semantic_vector, companies)

    # All 9 scorers (no weights here; we apply exit-specific weights later)
    scorers = [
        tech_prox, need_fit, abs_cap, past_ties,
        future_option, open_inno, humanities_fit,
        ambition_fit, theme_breadth,
    ]

    # Score all companies once across all 9 dimensions + synergy.
    # Also cache matching_themes from theme_breadth to avoid re-scoring later.
    scored_companies: list[tuple[dict, dict[str, FeatureResult], list[dict]]] = []

    for co in companies:
        features: dict[str, FeatureResult] = {}
        for scorer in scorers:
            result = scorer.score(seed.semantic_vector, co)
            features[scorer.name] = result

        # Cache matching_themes immediately after theme_breadth.score() runs,
        # so hypothesis generation can use them without a second score() call.
        cached_matching_themes = list(theme_breadth.matching_themes)

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

        scored_companies.append((co, features, cached_matching_themes))

    # --- Exclusive assignment (ACM Web Conference 2024 CCDF paper) ---
    # Step 1: Compute all 3 exit scores for every company
    exit_types = ("rd", "new_domain", "exploratory")
    # company_exit_data[edinet_code] = {exit_type: (score, co, features, matching_themes)}
    company_exit_data: dict[str, dict[str, tuple[float, dict, dict, list]]] = {}

    for co, features, matching_themes in scored_companies:
        ec = co["edinet_code"]
        company_exit_data[ec] = {}
        for exit_type in exit_types:
            weights = EXIT_WEIGHTS[exit_type]
            total = 0.0
            for dim_name, w in weights.items():
                # Allow negative weights (Nooteboom 2007: cognitive distance penalty)
                if w == 0:
                    continue
                fr = features.get(dim_name, FeatureResult(0, ""))
                total += w * fr.value
            company_exit_data[ec][exit_type] = (total, co, features, matching_themes)

    # Step 2: Exclusive assignment — each company goes to its highest-scoring exit.
    # Uses a two-pass approach to guarantee minimum candidates per exit:
    #   Pass 1: Assign each company to its best exit (greedy).
    #   Pass 2: If any exit has fewer than top_n companies, steal the
    #           highest-scoring unassigned-to-this-exit companies from
    #           over-populated exits.
    company_assignments: dict[str, str] = {}
    for ec, exit_scores in company_exit_data.items():
        best_exit = max(exit_scores, key=lambda et: exit_scores[et][0])
        company_assignments[ec] = best_exit

    # Pass 2: Ensure each exit has at least top_n candidates
    for exit_type in exit_types:
        assigned_count = sum(1 for v in company_assignments.values() if v == exit_type)
        if assigned_count < top_n:
            # Find companies not assigned to this exit, sorted by their score for this exit
            candidates = [
                (company_exit_data[ec][exit_type][0], ec)
                for ec in company_exit_data
                if company_assignments[ec] != exit_type
            ]
            candidates.sort(key=lambda x: x[0], reverse=True)
            needed = top_n - assigned_count
            for _, ec in candidates[:needed]:
                company_assignments[ec] = exit_type

    # Step 3: Build per-exit candidate lists (only assigned companies)
    exits: list[ExitRanking] = []

    # Order exits so the recommended one comes first
    recommended = researcher_profile.recommended_exit if researcher_profile else "rd"
    other_exits = [et for et in exit_types if et != recommended]
    ordered_exits = [recommended] + other_exits

    for exit_type in ordered_exits:
        # Filter to companies assigned to this exit
        exit_scored: list[tuple[float, dict, dict[str, FeatureResult], list[dict]]] = [
            company_exit_data[ec][exit_type]
            for ec, assigned in company_assignments.items()
            if assigned == exit_type
        ]
        exit_scored.sort(key=lambda x: x[0], reverse=True)

        # Apply exit-specific reranking for diversity
        if exit_type == "exploratory":
            pool = exit_scored[:top_n * 3]
            top = _mmr_rerank(pool, top_n, lambda_param=0.5)
        elif exit_type == "new_domain":
            pool = exit_scored[:top_n * 2]
            top = _mmr_rerank(pool, top_n, lambda_param=0.7)
        else:
            top = exit_scored[:top_n]

        rankings = []
        for rank_idx, (total, co, features, matching_themes) in enumerate(top, 1):
            hypothesis = _generate_exit_hypothesis(
                exit_type, co["name"], features, matching_themes,
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
    result = MultiExitMatchResult(
        seed=seed,
        exits=exits,
        researcher_profile=researcher_profile,
        executed_at=datetime.now().isoformat(timespec="seconds"),
        duration_sec=round(duration, 2),
        company_count=len(companies),
    )

    _save_multi_exit_result(result)
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


def _save_multi_exit_result(result: MultiExitMatchResult) -> None:
    """Persist multi-exit match results to DB.

    Saves the seed record and all exit-type rankings into the
    multi_exit_matches table. Deletes any previous results for the
    same seed_id to support re-runs.
    """
    with connect(settings.matcher_db_path) as conn:
        # Ensure multi_exit_matches table exists (safe for first run)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS multi_exit_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seed_id TEXT NOT NULL,
                exit_type TEXT NOT NULL,
                edinet_code TEXT NOT NULL,
                rank INTEGER NOT NULL,
                total_score REAL NOT NULL,
                tech_prox REAL, need_fit REAL, abs_cap REAL, past_ties REAL,
                future_option REAL, open_inno REAL, humanities_fit REAL,
                ambition_fit REAL, theme_breadth REAL, synergy REAL,
                hypothesis TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (seed_id) REFERENCES seeds(seed_id)
            );
            CREATE INDEX IF NOT EXISTS idx_mex_seed ON multi_exit_matches(seed_id);
            CREATE INDEX IF NOT EXISTS idx_mex_exit ON multi_exit_matches(seed_id, exit_type);
        """)

        # Upsert seed record
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

        # Delete existing multi-exit results for this seed (re-run support)
        conn.execute(
            "DELETE FROM multi_exit_matches WHERE seed_id = ?",
            (result.seed.seed_id,),
        )

        # Insert all exit rankings
        for exit_ranking in result.exits:
            for rc in exit_ranking.rankings:
                fs = rc.feature_scores
                conn.execute(
                    """INSERT INTO multi_exit_matches
                       (seed_id, exit_type, edinet_code, rank, total_score,
                        tech_prox, need_fit, abs_cap, past_ties,
                        future_option, open_inno, humanities_fit,
                        ambition_fit, theme_breadth, synergy,
                        hypothesis, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        result.seed.seed_id,
                        exit_ranking.exit_type,
                        rc.edinet_code,
                        rc.rank,
                        rc.total_score,
                        fs.get("tech_prox", FeatureResult(0, "")).value,
                        fs.get("need_fit", FeatureResult(0, "")).value,
                        fs.get("abs_cap", FeatureResult(0, "")).value,
                        fs.get("past_ties", FeatureResult(0, "")).value,
                        fs.get("future_option", FeatureResult(0, "")).value,
                        fs.get("open_inno", FeatureResult(0, "")).value,
                        fs.get("humanities_fit", FeatureResult(0, "")).value,
                        fs.get("ambition_fit", FeatureResult(0, "")).value,
                        fs.get("theme_breadth", FeatureResult(0, "")).value,
                        fs.get("synergy", FeatureResult(0, "")).value,
                        rc.overall_comment,
                        result.executed_at,
                    ),
                )
