"""Reporter — render match results as Markdown and JSON."""
from __future__ import annotations

import json
from datetime import datetime

from sangaku_matcher.matcher import MatchResult


MODE_LABELS = {
    "joint_research": "共同研究",
    "license": "ライセンス",
    "contract": "受託研究",
    "long_term": "中長期探索型連携",
}

FEATURE_LABELS = {
    "tech_prox": ("技術近接", 0.25),
    "need_fit": ("ニーズ適合", 0.20),
    "abs_cap": ("知識吸収", 0.20),
    "past_ties": ("関係資本", 0.10),
    "future_option": ("将来価値", 0.10),
    "open_inno": ("エコシステム", 0.15),
}


def to_markdown(result: MatchResult) -> str:
    """Render match result as a Markdown report."""
    lines = [
        f"# マッチング結果: {result.seed.title}",
        "",
        f"実行日時: {result.executed_at}",
        f"処理時間: {result.duration_sec}秒",
        f"対象企業: {result.company_count}社",
        f"スコアリング: 五層価値モデル（技術0.25 + ニーズ0.20 + 知識0.20 + OI 0.15 + 関係0.10 + 将来0.10）",
        "",
        "---",
        "",
        "## 入力シーズ",
        "",
        f"**タイトル**: {result.seed.title}",
        "",
        result.seed.description[:500],
        "",
        "---",
        "",
        "## ランキング",
        "",
        "| # | 企業名 | 業種 | 総合 | 技術 | ニーズ | 知識 | 関係 | 将来 | OI |",
        "|---|--------|------|------|------|--------|------|------|------|-----|",
    ]

    for rc in result.rankings:
        fs = rc.feature_scores
        vals = []
        for k in ("tech_prox", "need_fit", "abs_cap", "past_ties", "future_option", "open_inno"):
            f = fs.get(k)
            vals.append(f"{f.value:.2f}" if f else "-")
        lines.append(
            f"| {rc.rank} | {rc.company_name} | {rc.industry} | "
            f"{rc.total_score:.2f} | {' | '.join(vals)} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    for rc in result.rankings:
        mode_ja = MODE_LABELS.get(rc.recommended_mode, rc.recommended_mode)
        lines.append(f"### {rc.rank}位: {rc.company_name} ({rc.edinet_code})")
        lines.append("")
        lines.append(f"**総合スコア**: {rc.total_score:.4f}")
        lines.append(f"**推奨連携モード**: {mode_ja}")
        lines.append("")

        lines.append("| 特徴量 | スコア | 重み |")
        lines.append("|--------|--------|------|")
        for fname, fr in rc.feature_scores.items():
            label, w = FEATURE_LABELS.get(fname, (fname, 0))
            lines.append(f"| {label} | {fr.value:.4f} | {w:.2f} |")

        lines.append("")
        lines.append("**各観点の評価**:")
        for fname, fr in rc.feature_scores.items():
            if fr.rationale:
                lines.append(f"- **{fname}**: {fr.rationale}")
        lines.append("")

        if rc.overall_comment:
            lines.append(f"**総合評価**: {rc.overall_comment}")
            lines.append("")

        if rc.collaboration_hypotheses:
            lines.append("**連携仮説**:")
            lines.append("")
            for i, hyp in enumerate(rc.collaboration_hypotheses, 1):
                type_ja = MODE_LABELS.get(hyp.collab_type, hyp.collab_type)
                lines.append(f"{i}. **{hyp.title}** [{type_ja}]")
                lines.append(f"   {hyp.description}")
                lines.append(f"   - 根拠: {hyp.rationale}")
                lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("")
    lines.append(
        "> 注: 本結果はAIによる仮説であり、最終的な連携判断は"
        "専門家の評価に基づいて行ってください。"
    )

    return "\n".join(lines)


def to_json(result: MatchResult) -> dict:
    """Render match result as JSON-serializable dict."""
    return {
        "seed": {
            "seed_id": result.seed.seed_id,
            "title": result.seed.title,
            "description": result.seed.description[:500],
            "doi": result.seed.doi,
            "patent_no": result.seed.patent_no,
            "source_type": result.seed.source_type,
        },
        "executed_at": result.executed_at,
        "duration_sec": result.duration_sec,
        "company_count": result.company_count,
        "matches": [
            {
                "rank": rc.rank,
                "edinet_code": rc.edinet_code,
                "company_name": rc.company_name,
                "industry": rc.industry,
                "total_score": rc.total_score,
                "feature_scores": {
                    k: {"value": v.value, "rationale": v.rationale}
                    for k, v in rc.feature_scores.items()
                },
                "recommended_mode": rc.recommended_mode,
                "overall_comment": rc.overall_comment,
                "collaboration_hypotheses": [
                    {
                        "title": h.title,
                        "description": h.description,
                        "collab_type": h.collab_type,
                        "rationale": h.rationale,
                    }
                    for h in rc.collaboration_hypotheses
                ],
            }
            for rc in result.rankings
        ],
    }
