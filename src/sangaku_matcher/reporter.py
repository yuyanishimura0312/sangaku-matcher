"""Reporter — render match results as Markdown and JSON."""
from __future__ import annotations

import json
from datetime import datetime

from sangaku_matcher.matcher import MatchResult


MODE_LABELS = {
    "joint_research": "共同研究",
    "license": "ライセンス",
    "contract": "受託開発",
    "long_term": "中長期研究契約",
}


def to_markdown(result: MatchResult) -> str:
    """Render match result as a Markdown report."""
    lines = [
        f"# マッチング結果: {result.seed.title}",
        "",
        f"実行日時: {result.executed_at}",
        f"処理時間: {result.duration_sec}秒",
        f"対象企業: {result.company_count}社",
        f"スコアリング: TechProx(0.35) + AbsCap(0.35) + PastTies(0.30)",
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
        "| # | 企業名 | 業種 | 総合 | 技術 | 吸収 | 実績 |",
        "|---|--------|------|------|------|------|------|",
    ]

    for rc in result.rankings:
        fs = rc.feature_scores
        tp = fs.get("tech_prox")
        ac = fs.get("abs_cap")
        pt = fs.get("past_ties")
        lines.append(
            f"| {rc.rank} | {rc.company_name} | {rc.industry} | "
            f"{rc.total_score:.2f} | {tp.value:.2f if tp else '-'} | "
            f"{ac.value:.2f if ac else '-'} | {pt.value:.2f if pt else '-'} |"
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
            w = {"tech_prox": 0.35, "abs_cap": 0.35, "past_ties": 0.30}.get(fname, 0)
            lines.append(f"| {fname} | {fr.value:.4f} | {w:.2f} |")

        lines.append("")
        lines.append("**根拠**:")
        for fname, fr in rc.feature_scores.items():
            if fr.rationale:
                lines.append(f"- {fr.rationale}")
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
            }
            for rc in result.rankings
        ],
    }
