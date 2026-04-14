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


def _load_company_texts(conn, edinet_codes: list[str]) -> dict[str, dict]:
    """Load full text fields for specific companies (for top-ranked only).

    Returns: {edinet_code: {rd_text, midterm_plan_text, ambitions_text, ...}}
    """
    if not edinet_codes:
        return {}
    placeholders = ",".join("?" * len(edinet_codes))
    rows = conn.execute(
        f"""SELECT edinet_code, rd_text, midterm_plan_text, estimated_needs,
                   tech_needs_text, ambitions_text, humanities_needs_text,
                   revenue, rd_expense, rd_intensity, employees, market_cap,
                   open_inno_score, industry
            FROM companies WHERE edinet_code IN ({placeholders})""",
        edinet_codes,
    ).fetchall()
    return {r["edinet_code"]: dict(r) for r in rows}


def _load_industry_peers(conn, industry: str, exclude_code: str) -> list[dict]:
    """Load summary stats of same-industry peers for comparison."""
    rows = conn.execute(
        """SELECT name, revenue, rd_intensity, open_inno_score, rd_expense
           FROM companies
           WHERE industry = ? AND edinet_code != ? AND rd_intensity > 0
           ORDER BY revenue DESC LIMIT 10""",
        (industry, exclude_code),
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
    brief: bool = False,
    company_text: dict | None = None,
    peers: list[dict] | None = None,
) -> str:
    """Generate an exit-specific hypothesis text for a company.

    Uses 5 internal specialist agents for top-ranked companies:
    1. Technical Analyst — technology alignment and TRL assessment
    2. Business Strategy Analyst — absorptive capacity, OI, ROI
    3. Inflection Point Analyst — market timing and structural change
    4. Collaboration Architect — contract design, IP, funding
    5. Securities Report Analyst — company-specific insights vs peers

    Args:
        exit_type: One of "rd", "new_domain", "exploratory".
        company_name: Display name of the target company.
        features: Scorer results keyed by scorer name.
        cached_matching_themes: Matching theme list captured during the initial
            score() call — avoids a redundant second score() invocation.
        brief: If True, generate a short summary for lower-ranked companies.
        company_text: Full text fields from the securities report (top 3 only).
        peers: Same-industry peer companies for comparison.

    Returns:
        Human-readable hypothesis string in Japanese (HTML for detailed mode).
    """
    company_text = company_text or {}
    peers = peers or []
    nf = features.get("need_fit", FeatureResult(0, ""))
    tp = features.get("tech_prox", FeatureResult(0, ""))
    ac = features.get("abs_cap", FeatureResult(0, ""))
    pt = features.get("past_ties", FeatureResult(0, ""))
    hf = features.get("humanities_fit", FeatureResult(0, ""))
    oi = features.get("open_inno", FeatureResult(0, ""))
    af = features.get("ambition_fit", FeatureResult(0, ""))
    tb = features.get("theme_breadth", FeatureResult(0, ""))
    fo = features.get("future_option", FeatureResult(0, ""))

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
        parts = rationale.split("。")
        for p in parts:
            p = p.strip()
            if any(kw in p for kw in ["学の", "論の", "研究の", "学的", "倫理"]) and len(p) > 20:
                return p + "。"
        return ""

    # Helper: extract matching tech theme names from need_fit / tech_prox rationale
    def _extract_tech_themes(rationale: str, max_n: int = 3) -> list[str]:
        names = []
        for marker in ["高親和:", "高類似:", "上位:"]:
            if marker in rationale:
                after = rationale.split(marker)[1]
                for part in after.split(";"):
                    part = part.strip()
                    if "(" in part:
                        name = part.split("(")[0].strip()
                        if name and len(name) > 2:
                            names.append(name)
                break
        return names[:max_n]

    # ── Brief mode: short summary for lower-ranked companies ──
    if brief:
        if exit_type == "rd":
            tech_themes = _extract_tech_themes(nf.rationale) or _extract_tech_themes(tp.rationale)
            if tech_themes:
                return f"技術テーマ「{'」「'.join(tech_themes[:2])}」で接点あり。ニーズ適合{nf.value*100:.0f}%、吸収力{ac.value*100:.0f}%。"
            return f"技術ニーズとの適合{nf.value*100:.0f}%。R&D体制（吸収力{ac.value*100:.0f}%）を有する企業。"
        elif exit_type == "new_domain":
            theme_names = _extract_theme_names(af.rationale)
            if theme_names:
                return f"新領域「{'」「'.join(theme_names[:2])}」で連携可能性。野心適合{af.value*100:.0f}%、OI度{oi.value*100:.0f}%。"
            return f"新事業領域への野心適合{af.value*100:.0f}%。オープンイノベーション体制{oi.value*100:.0f}%。"
        else:
            n_themes = len(cached_matching_themes)
            if n_themes > 0:
                axes = set(m.get("axis", "") for m in cached_matching_themes[:5])
                axis_labels = [{"humanities": "社会課題", "tech": "技術", "ambition": "新規事業"}.get(a, "") for a in axes]
                axis_labels = [a for a in axis_labels if a]
                return f"{'・'.join(axis_labels)}の{n_themes}テーマで接点。対話を通じた連携テーマの探索が見込まれる。"
            return f"テーマ幅{tb.value*100:.0f}%。異なる視点からの対話を通じた接点発見の余地あり。"

    # ── Detailed mode: structured HTML with 5 expert agent perspectives ──
    # Helper: wrap a section with heading
    def _section(heading: str, body: str) -> str:
        return (
            f'<div class="hyp-section">'
            f'<span class="hyp-heading">{heading}</span>'
            f'<span class="hyp-body">{body}</span>'
            f'</div>'
        )

    def _m(val: str) -> str:
        """Highlight a metric value."""
        return f'<span class="hyp-metric">{val}</span>'

    # ── Extract useful text snippets from securities report ──
    rd_text = company_text.get("rd_text", "") or ""
    midterm = company_text.get("midterm_plan_text", "") or ""
    ambitions_text = company_text.get("ambitions_text", "") or ""
    ct_revenue = company_text.get("revenue", 0) or 0
    ct_rd_exp = company_text.get("rd_expense", 0) or 0
    ct_rd_int = company_text.get("rd_intensity", 0) or 0
    ct_employees = company_text.get("employees", 0) or 0

    # ── Agent 1: Technical Analyst ──
    def _agent_tech() -> str:
        """Assess technology alignment with TRL-aware reasoning."""
        tech_themes = _extract_tech_themes(nf.rationale) or _extract_tech_themes(tp.rationale)
        theme_names_a = _extract_theme_names(af.rationale)
        sy = features.get("synergy", FeatureResult(0, ""))

        if exit_type == "rd":
            if tech_themes:
                theme_str = "「" + "」「".join(tech_themes) + "」"
                text = (
                    f"{company_name}の有価証券報告書の研究開発方針を分析した結果、"
                    f"{theme_str}の技術テーマで研究シーズとの高い親和性が確認されました"
                    f"（ニーズ適合度{_m(f'{nf.value*100:.0f}%')}、"
                    f"技術近接度{_m(f'{tp.value*100:.0f}%')}）。"
                )
            else:
                text = (
                    f"{company_name}は、当該研究シーズと関連性の高い技術課題を抱えています"
                    f"（ニーズ適合度{_m(f'{nf.value*100:.0f}%')}、"
                    f"技術近接度{_m(f'{tp.value*100:.0f}%')}）。"
                )
            # TRL-based staging advice
            if nf.value > 0.7:
                text += (
                    f"ニーズ適合度の高さは、企業側で既に課題が明確化されており、"
                    f"研究成果の技術実証（TRL 4-6）から社会実装（TRL 7-9）への"
                    f"橋渡しが比較的スムーズに進む可能性を示しています。"
                )
            elif nf.value > 0.4:
                text += (
                    f"基礎研究段階（TRL 1-3）の知見が、"
                    f"企業側の応用開発と接続できる領域です。"
                    f"共同研究を通じて技術の成熟度を高めていく段階にあります。"
                )
            # Synergy
            if sy.value > 0.3:
                text += (
                    f"技術ニーズと社会課題の交差領域でシナジー効果"
                    f"（{_m(f'{sy.value*100:.0f}%')}）も確認されており、"
                    f"複合的な研究価値が見込まれます。"
                )
            # Add R&D text insights
            if rd_text and len(rd_text) > 50:
                rd_keywords = []
                if "AI" in rd_text or "機械学習" in rd_text:
                    rd_keywords.append("AI・機械学習")
                if "素材" in rd_text or "材料" in rd_text:
                    rd_keywords.append("新素材・材料")
                if "バイオ" in rd_text or "医薬" in rd_text:
                    rd_keywords.append("バイオ・医薬")
                if "環境" in rd_text or "エネルギー" in rd_text:
                    rd_keywords.append("環境・エネルギー")
                if "半導体" in rd_text or "電子" in rd_text:
                    rd_keywords.append("半導体・電子")
                if "ロボット" in rd_text or "自動化" in rd_text:
                    rd_keywords.append("ロボティクス・自動化")
                if rd_keywords:
                    text += (
                        f"同社の研究開発活動は"
                        f"{'、'.join(rd_keywords[:3])}の領域を中心に展開されており、"
                        f"研究テーマとの技術的な補完関係が期待されます。"
                    )
        elif exit_type == "new_domain":
            if theme_names_a:
                theme_str = "「" + "」「".join(theme_names_a) + "」"
                text = (
                    f"{company_name}は{theme_str}等の"
                    f"新事業領域への挑戦を有価証券報告書で明示しています"
                    f"（野心適合度{_m(f'{af.value*100:.0f}%')}）。"
                    f"新領域での技術基盤はまだ発展途上（TRL 1-3相当）であり、"
                    f"研究者の知見が問いの設定段階から貢献できる余地が大きい領域です。"
                )
            else:
                text = (
                    f"{company_name}は新規事業領域の開拓に注力しています"
                    f"（野心適合度{_m(f'{af.value*100:.0f}%')}）。"
                    f"既存の技術資産とは異なる知識基盤が必要な段階であり、"
                    f"学術研究者との連携が技術的な探索の幅を広げます。"
                )
            if hf.value > 0.2:
                text += (
                    f"人文・社会科学との親和性（{_m(f'{hf.value*100:.0f}%')}）も高く、"
                    f"「なぜその事業が社会に必要か」という問いの構築に、"
                    f"研究者の視点が差別化要因となります。"
                )
        else:  # exploratory
            by_ax: dict[str, list[str]] = {}
            for m_t in cached_matching_themes[:10]:
                al = {"humanities": "社会課題", "tech": "技術", "ambition": "新規事業"}.get(m_t.get("axis", ""), "")
                if al:
                    by_ax.setdefault(al, []).append(m_t["label"])
            text = (
                f"{company_name}とは{_m(f'{len(by_ax)}つの軸')}にわたる"
                f"対話の接点が見込まれます（テーマ幅{_m(f'{tb.value*100:.0f}%')}）。"
            )
            for al, ns in by_ax.items():
                dn = "「" + "」「".join(ns[:2]) + "」"
                text += f"{al}では{dn}、"
            text += (
                f"などのテーマが接点となります。"
                f"技術的な距離がある分野間の対話は、既存の枠組みでは"
                f"生まれにくい新しい着想の源泉となります。"
            )
            if hf.value > 0.2:
                text += (
                    f"人文・社会科学の接点（{_m(f'{hf.value*100:.0f}%')}）も確認されており、"
                    f"技術者とは異なる視座での対話が可能です。"
                )
        return text

    # ── Agent 2: Business Strategy Analyst ──
    def _agent_biz() -> str:
        """Evaluate collaboration value from management perspective."""
        parts = []

        # Company scale context
        if ct_revenue > 0:
            rev_oku = ct_revenue / 1_000_000
            if ct_rd_exp > 0:
                rd_oku = ct_rd_exp / 1_000_000
                parts.append(
                    f"売上高{_m(f'{rev_oku:,.0f}百万円')}に対して"
                    f"研究開発費{_m(f'{rd_oku:,.0f}百万円')}"
                    f"（研究開発比率{_m(f'{ct_rd_int*100:.1f}%')}）を投じており、"
                )
            else:
                parts.append(f"売上高{_m(f'{rev_oku:,.0f}百万円')}の企業であり、")

        # Absorptive capacity — plain language
        if ac.value > 0.5 and pt.value > 0.2:
            parts.append(
                f"外部の研究知見を取り込み事業に活かす力（吸収力{_m(f'{ac.value*100:.0f}%')}）が高く、"
                f"大学との共同研究実績もあります。"
                f"研究成果が実際の製品・サービスに結びつく確度が高い企業です。"
            )
        elif ac.value > 0.5:
            parts.append(
                f"研究開発への投資が活発で（吸収力{_m(f'{ac.value*100:.0f}%')}）、"
                f"外部の研究成果を事業に取り込む体制があります。"
                f"大学との連携経験は限られるため、段階的なアプローチが効果的です。"
            )
        elif pt.value > 0.2:
            parts.append(
                f"大学との連携実績があり、共同研究の進め方に慣れた企業です。"
                f"契約や知財の社内手続きが整備されている可能性が高く、"
                f"スムーズな連携開始が期待できます。"
            )
        else:
            parts.append(
                f"研究開発投資は中程度（吸収力{_m(f'{ac.value*100:.0f}%')}）ですが、"
                f"テーマの適合度の高さが連携の動機づけとなります。"
                f"まず技術コンサルティングなど軽い形から関係構築するのが効果的です。"
            )

        # OI体制
        if oi.value > 0.4:
            parts.append(
                f"外部連携への積極性（OI度{_m(f'{oi.value*100:.0f}%')}）も高く、"
                f"社外との協業を推進する組織体制が整っています。"
            )
        elif oi.value > 0.2:
            parts.append(
                f"外部連携への取り組み（OI度{_m(f'{oi.value*100:.0f}%')}）も確認されています。"
            )

        # ESG / social value for non-rd exits
        if exit_type != "rd" and hf.value > 0.3:
            parts.append(
                f"ESG経営やサステナビリティが重要課題となる中、"
                f"社会科学的な視点を持つ研究者との連携は、"
                f"事業の社会的正当性を裏づける戦略的投資と位置づけられます。"
            )

        # Future option
        if fo.value > 0.3 and exit_type != "rd":
            parts.append(
                f"将来に向けた選択肢の価値（{_m(f'{fo.value*100:.0f}%')}）も高く、"
                f"中長期的な知的資産として連携の意義があります。"
            )

        # Strategic direction from midterm plan
        if midterm and len(midterm) > 200:
            strategic_dirs = []
            if "成長" in midterm[:3000] and "投資" in midterm[:3000]:
                strategic_dirs.append("成長投資の拡大")
            if "イノベーション" in midterm[:3000]:
                strategic_dirs.append("イノベーション推進")
            if "社会" in midterm[:2000] and "課題" in midterm[:2000]:
                strategic_dirs.append("社会課題への取り組み")
            if strategic_dirs:
                parts.append(
                    f"経営計画では{'・'.join(strategic_dirs)}が打ち出されており、"
                    f"産学連携を通じた外部知見の獲得は"
                    f"この経営方針とも整合します。"
                )

        return "".join(parts)

    # ── Agent 3: Inflection Point Analyst (平易な言葉で) ──
    def _agent_inflection() -> str:
        """Assess market timing using structural change concepts in plain language."""
        # Detect strategic keywords from midterm plan for context
        has_dx = midterm and ("DX" in midterm or "デジタル" in midterm)
        has_green = midterm and ("カーボン" in midterm or "脱炭素" in midterm or "環境" in midterm)
        has_global = midterm and ("グローバル" in midterm or "海外" in midterm[:2000])

        context_hints = []
        if has_dx:
            context_hints.append("デジタル変革の加速")
        if has_green:
            context_hints.append("脱炭素への転換")
        if has_global:
            context_hints.append("グローバル市場の構造変化")
        context_str = "・".join(context_hints) if context_hints else ""

        if exit_type == "rd":
            if nf.value > 0.7 and oi.value > 0.3:
                text = (
                    f"この技術領域では、従来のやり方に限界が見え始め、"
                    f"新しいアプローチへの切り替えが進みつつある局面です。"
                    f"企業側の技術ニーズが明確で、かつ外部連携への体制が整っていることから、"
                    f"研究成果を実用化に結びつけるのに最も効果的なタイミングと言えます。"
                    f"このような「転換期」に産学連携で参入することで、"
                    f"新技術の標準化や市場形成の初期段階から関与できます。"
                )
            elif nf.value > 0.5:
                text = (
                    f"既存技術の限界が業界内で認識され始めている段階です。"
                    f"まだ代替技術が確立されていないため、"
                    f"研究シーズが市場に受け入れられる「機会の窓」が開きつつあります。"
                    f"今のうちに企業との関係を構築しておくことで、"
                    f"本格的な技術転換が起きた際に先行者優位を確保できます。"
                )
            else:
                text = (
                    f"この領域の技術変化はまだ緩やかな段階にあります。"
                    f"今すぐの事業化は難しいかもしれませんが、"
                    f"将来の大きな変化に備えた「種まき」としての関係構築は有効です。"
                    f"産学連携は成果が出るまでに時間がかかるため、"
                    f"変化が本格化する前に始めることが戦略的に重要です。"
                )
            if context_str:
                text += (
                    f"同社の経営計画からは{context_str}といった"
                    f"大きな変化への対応が読み取れ、これらの流れが"
                    f"研究テーマとの連携機会をさらに後押しする可能性があります。"
                )
            return text
        elif exit_type == "new_domain":
            if af.value > 0.5 and fo.value > 0.3:
                text = (
                    f"この新領域では、規制の変化や社会的ニーズの高まりなど、"
                    f"市場が大きく動き始める兆候が見られます。"
                    f"企業が積極的に参入を検討しているこのタイミングは、"
                    f"研究者が「どんな問いを立てるべきか」という"
                    f"最も上流の段階から関与できる貴重な機会です。"
                    f"新市場のルール形成に研究知見が影響を与えられる可能性があります。"
                )
            else:
                text = (
                    f"新事業領域はまだ黎明期にあり、不確実性が高い段階です。"
                    f"しかし、不確実な時期だからこそ、"
                    f"学術研究者の「問いを立てる力」が最も活きる局面です。"
                    f"この段階から関わることで、研究テーマと事業方向性を"
                    f"同時に設計できるという、後からでは得られない戦略的優位があります。"
                )
            if context_str:
                text += (
                    f"同社は{context_str}を経営の方向性として掲げており、"
                    f"こうした構造的な変化が新領域での連携に追い風となります。"
                )
            return text
        else:  # exploratory
            if tb.value > 0.4 and hf.value > 0.2:
                text = (
                    f"複数の分野にわたるテーマの接点は、"
                    f"いくつもの変化が同時に進行している兆候です。"
                    f"技術・社会・市場の変化が重なり合う領域では、"
                    f"異なる分野間の対話から予想外の発見が生まれやすくなります。"
                    f"こうした「変化の交差点」こそ、探索的な産学対話の"
                    f"最も大きなリターンが期待できる場です。"
                )
            else:
                text = (
                    f"現時点では明確な変化の兆候は限定的ですが、"
                    f"産業構造の変化は徐々に進行するものです。"
                    f"早い段階で異分野との対話チャネルを持っておくことは、"
                    f"将来の変化をいち早く察知し、機会を掴むための"
                    f"「知的アンテナ」として機能します。"
                )
            if context_str:
                text += (
                    f"同社の事業環境では{context_str}が進行しており、"
                    f"これらの変化が新たな対話テーマを生み出す可能性があります。"
                )
            return text

    # ── Agent 4: Collaboration Architect ──
    def _agent_collab() -> str:
        """Design concrete collaboration approach with IP and funding guidance."""
        if exit_type == "rd":
            text = (
                f"推奨する連携プロセスは以下の通りです。"
                f"まず秘密保持契約（NDA）を締結し、3〜6ヶ月の技術ディスカッションで"
                f"共同研究のテーマと範囲を具体化します。"
                f"その後、共同研究契約に移行し、通常1〜3年の研究期間を設定します。"
                f"知的財産は、基盤技術は大学帰属・応用技術は企業への"
                f"実施許諾（ライセンス）が一般的ですが、"
                f"共同発明の取り扱いは契約前に明確化してください。"
            )
            if ac.value > 0.5:
                text += (
                    f"研究開発体制の充実した企業であるため、"
                    f"企業側からの研究者受入や設備共用など、"
                    f"実践的な共同研究体制の構築も検討できます。"
                )
            text += (
                f"資金面では、JSTのA-STEP（研究成果展開事業）や"
                f"産学共創プラットフォームが活用できます。"
                f"また、NEDOの技術開発プロジェクトへの共同提案も有効です。"
            )
            if nf.value > 0.7:
                text += (
                    f"ニーズ適合度の高さから、共著論文・共同特許・"
                    f"プロトタイプ開発など実用化直結型の成果が期待されます。"
                )
            text += (
                f"留意点として、研究者側の学術的関心と企業側の事業課題の間に"
                f"時間軸や優先度のズレが生じやすいため、"
                f"キックオフ時に双方の期待値と成果指標を明文化してください。"
                f"四半期ごとの進捗レビューを設定し、テーマの軌道修正を"
                f"柔軟に行える体制が連携成功の鍵です。"
            )
        elif exit_type == "new_domain":
            text = (
                f"新領域探索では、いきなり共同研究契約を結ぶより、"
                f"まずアドバイザリー契約（月1〜2回の助言）や"
                f"共同ワークショップで「問いの共同設計」から始めるのが効果的です。"
                f"3〜6ヶ月の探索フェーズで複数の仮説を検証し、"
                f"有望テーマに絞り込んでから本格的な共同研究へ移行する"
                f"段階的アプローチ（ステージゲート方式）が成功確率を高めます。"
                f"知的財産は、探索段階では共同所有とし、"
                f"事業化段階で企業への独占ライセンスに切り替えるのが一般的です。"
                f"JSTの「共創の場形成支援プログラム」や"
                f"経産省の関連事業（未来社会創造事業等）の活用も検討してください。"
            )
        else:  # exploratory
            text = (
                f"探索的対話段階では、契約関係の前に"
                f"カジュアルな接点づくりから始めることを推奨します。"
                f"具体的には、セミナーへの相互登壇、"
                f"ワークショップの共同開催、研究室訪問・工場見学の相互実施などです。"
                f"URA（リサーチ・アドミニストレーター）や"
                f"産学連携コーディネーターによるマッチングイベントも有効です。"
                f"この段階では知財の心配は不要ですが、"
                f"公開情報の範囲内で対話することを事前に合意してください。"
                f"相互理解が深まった段階で、NDAを締結し、"
                f"具体的な共同研究テーマの検討に移行します。"
                f"大学のオープンイノベーション機構を通じた"
                f"初期接点の設定も効果的です。"
            )
        return text

    # ── Agent 5: Securities Report Analyst (有報分析) ──
    def _agent_yuho() -> str:
        """Analyze company-specific characteristics from securities report texts,
        comparing with industry peers to highlight what's distinctive."""
        parts = []

        # Use midterm plan text to extract key strategy keywords
        if midterm and len(midterm) > 100:
            # Extract notable short phrases (first 500 chars of midterm plan)
            plan_excerpt = midterm[:500].replace("\n", " ").strip()
            parts.append(
                f"中期経営計画の分析から、同社は"
            )
            # Extract key strategic direction from text
            if "DX" in midterm or "デジタル" in midterm:
                parts.append(f"デジタル変革（DX）を重点戦略に掲げており、")
            if "カーボンニュートラル" in midterm or "脱炭素" in midterm:
                parts.append(f"脱炭素・カーボンニュートラルに注力しており、")
            if "グローバル" in midterm or "海外" in midterm[:2000]:
                parts.append(f"グローバル展開を推進しており、")
            if "M&A" in midterm or "買収" in midterm:
                parts.append(f"M&Aによる事業領域拡大を進めており、")
            if "人的資本" in midterm or "人材" in midterm[:1000]:
                parts.append(f"人的資本経営を重視しており、")
            parts.append(
                f"これらの経営方針と研究テーマの接点が連携の起点となります。"
            )

        # R&D text analysis
        if rd_text and len(rd_text) > 50:
            parts.append(
                f"研究開発の記述（{_m(f'{len(rd_text):,}文字')}）からは、"
            )
            if "AI" in rd_text or "機械学習" in rd_text or "人工知能" in rd_text:
                parts.append(f"AI・機械学習技術の研究開発、")
            if "素材" in rd_text or "材料" in rd_text:
                parts.append(f"新素材・材料技術の開発、")
            if "バイオ" in rd_text or "医薬" in rd_text or "ヘルスケア" in rd_text:
                parts.append(f"バイオ・医薬・ヘルスケア領域の研究、")
            if "環境" in rd_text or "エネルギー" in rd_text:
                parts.append(f"環境・エネルギー技術の開発、")
            parts.append(f"などの取り組みが確認されています。")

        # Peer comparison
        if peers:
            peer_rd_ints = [p["rd_intensity"] for p in peers if p.get("rd_intensity")]
            if peer_rd_ints and ct_rd_int > 0:
                avg_rd_int = sum(peer_rd_ints) / len(peer_rd_ints)
                if ct_rd_int > avg_rd_int * 1.3:
                    parts.append(
                        f"同業他社（{len(peers)}社）との比較では、"
                        f"研究開発比率が業界平均{_m(f'{avg_rd_int*100:.1f}%')}を"
                        f"大きく上回る{_m(f'{ct_rd_int*100:.1f}%')}であり、"
                        f"技術投資に積極的な企業として際立っています。"
                    )
                elif ct_rd_int > avg_rd_int * 0.8:
                    parts.append(
                        f"同業他社との比較では、研究開発比率"
                        f"（{_m(f'{ct_rd_int*100:.1f}%')} vs "
                        f"業界平均{_m(f'{avg_rd_int*100:.1f}%')}）は同水準であり、"
                        f"業界標準レベルの研究投資を行っています。"
                    )
                else:
                    parts.append(
                        f"研究開発比率は業界平均（{_m(f'{avg_rd_int*100:.1f}%')}）を"
                        f"下回る{_m(f'{ct_rd_int*100:.1f}%')}ですが、"
                        f"事業規模を活かした応用開発力に強みがある可能性があります。"
                    )

            # OI comparison
            peer_ois = [p["open_inno_score"] for p in peers if p.get("open_inno_score")]
            if peer_ois:
                avg_oi = sum(peer_ois) / len(peer_ois)
                co_oi = company_text.get("open_inno_score", 0) or 0
                if co_oi > avg_oi * 1.3 and co_oi > 0.3:
                    parts.append(
                        f"外部連携への積極性も業界内で突出しており、"
                        f"産学連携の受け皿として優位な位置にあります。"
                    )

        if not parts:
            # Fallback when no text data available
            parts.append(
                f"{company_name}の有価証券報告書の分析に基づく評価です。"
                f"企業の個別戦略の詳細は、面談時に直接確認することを推奨します。"
            )

        return "".join(parts)

    # ── Assemble structured HTML ──
    sections = [
        _section("技術分析", _agent_tech()),
        _section("経営戦略分析", _agent_biz()),
        _section("変化点分析", _agent_inflection()),
        _section("連携設計", _agent_collab()),
        _section("有報からの企業特徴", _agent_yuho()),
    ]
    return "\n".join(sections)


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

        # Load full text data for top 3 companies in this exit
        top3_codes = [co["edinet_code"] for _, co, _, _ in top[:3]]
        with connect(settings.matcher_db_path) as text_conn:
            company_texts = _load_company_texts(text_conn, top3_codes)
            # Load industry peer data for comparison
            industry_peers: dict[str, list[dict]] = {}
            for code in top3_codes:
                ct = company_texts.get(code, {})
                if ct.get("industry"):
                    industry_peers[code] = _load_industry_peers(
                        text_conn, ct["industry"], code)

        rankings = []
        for rank_idx, (total, co, features, matching_themes) in enumerate(top, 1):
            # Top 3 get detailed hypothesis; rest get brief summary
            hypothesis = _generate_exit_hypothesis(
                exit_type, co["name"], features, matching_themes,
                brief=(rank_idx > 3),
                company_text=company_texts.get(co["edinet_code"], {}),
                peers=industry_peers.get(co["edinet_code"], []),
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
