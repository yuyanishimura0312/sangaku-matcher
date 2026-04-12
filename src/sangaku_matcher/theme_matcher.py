"""Theme-based hypothesis matching engine.

Uses 122 themes across 3 axes (humanities 33, tech 49, ambition 40) to generate
specific collaboration hypotheses between a researcher's profile and companies.

For each company, identifies overlapping themes where both the researcher's
expertise and the company's needs are strong, then generates explanatory hypotheses.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sangaku_matcher import embeddings
from sangaku_matcher.config import settings
from sangaku_matcher.db import connect


@dataclass
class ThemeMatch:
    """A single theme-level match between researcher and company."""
    axis: str  # "humanities", "tech", "ambition"
    theme_id: str
    theme_name: str
    major_name: str
    researcher_affinity: float  # how relevant this theme is to the researcher
    company_strength: float     # z-score normalized company proximity
    match_score: float          # researcher_affinity × company_strength
    humanities_role: str = ""   # how humanities can contribute (ambition themes)


@dataclass
class CompanyHypothesis:
    """Hypothesis for why a researcher should collaborate with a company."""
    edinet_code: str
    company_name: str
    industry: str
    total_score: float
    theme_matches: list[ThemeMatch] = field(default_factory=list)
    hypothesis_text: str = ""
    top_themes_summary: str = ""


@dataclass
class MatchResult:
    """Full matching result."""
    researcher_text: str
    researcher_profile: dict  # theme affinities
    companies: list[CompanyHypothesis] = field(default_factory=list)
    theme_count: int = 0


def _load_axis_data(conn, themes_table: str, proximity_table: str) -> tuple[list[dict], dict, dict]:
    """Load theme centroids, company proximities, and per-theme stats for one axis."""
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if themes_table not in tables or proximity_table not in tables:
        return [], {}, {}

    # Themes
    themes = []
    for r in conn.execute(f"SELECT * FROM {themes_table} ORDER BY theme_id"):
        d = dict(r)
        d["centroid"] = np.frombuffer(d["centroid"], dtype=np.float32)
        themes.append(d)

    # Company proximities
    prox = {}
    for r in conn.execute(f"SELECT edinet_code, theme_id, proximity FROM {proximity_table}"):
        ec = r["edinet_code"]
        if ec not in prox:
            prox[ec] = {}
        prox[ec][r["theme_id"]] = r["proximity"]

    # Per-theme stats for z-score normalization
    from collections import defaultdict
    theme_vals = defaultdict(list)
    for ec_prox in prox.values():
        for tid, p in ec_prox.items():
            theme_vals[tid].append(p)

    stats = {}
    for tid, vals in theme_vals.items():
        arr = np.array(vals)
        stats[tid] = (float(arr.mean()), max(float(arr.std()), 0.001))

    return themes, prox, stats


def run_theme_match(researcher_text: str, top_n: int = 10) -> MatchResult:
    """Run theme-based matching for a researcher's text."""
    # Embed researcher text
    vec = embeddings.encode_single(f"query: {researcher_text}")
    vec_norm = vec / np.linalg.norm(vec)

    with connect(settings.matcher_db_path) as conn:
        # Load all 3 axes
        hum_themes, hum_prox, hum_stats = _load_axis_data(
            conn, "taxonomy_themes", "company_taxonomy_proximity")
        tech_themes, tech_prox, tech_stats = _load_axis_data(
            conn, "tech_taxonomy_themes", "tech_taxonomy_proximity")
        amb_themes, amb_prox, amb_stats = _load_axis_data(
            conn, "ambition_taxonomy_themes", "ambition_taxonomy_proximity")

        # Load company info
        companies = {r["edinet_code"]: dict(r) for r in conn.execute(
            "SELECT edinet_code, name, industry FROM companies"
        )}

    # Compute researcher's affinity to each theme (cosine similarity)
    axes = [
        ("humanities", hum_themes, hum_prox, hum_stats),
        ("tech", tech_themes, tech_prox, tech_stats),
        ("ambition", amb_themes, amb_prox, amb_stats),
    ]

    researcher_profile = {}
    all_theme_data = []  # (axis, theme, affinity)

    for axis_name, themes, prox, stats in axes:
        if not themes:
            continue
        centroids = np.array([t["centroid"] for t in themes])
        sims = centroids @ vec_norm
        for i, theme in enumerate(themes):
            affinity = float(sims[i])
            researcher_profile[f"{axis_name}:{theme['theme_id']}"] = {
                "name": theme["name"],
                "major": theme["major_name"],
                "affinity": affinity,
            }
            all_theme_data.append((axis_name, theme, affinity, prox, stats))

    # Score each company
    company_scores = {}
    for axis_name, theme, affinity, prox, stats in all_theme_data:
        tid = theme["theme_id"]
        mean, std = stats.get(tid, (0.85, 0.01))

        for ec, ec_prox in prox.items():
            raw = ec_prox.get(tid, 0.0)
            z = (raw - mean) / std
            company_z = 1.0 / (1.0 + np.exp(-z))  # sigmoid

            # Match score = researcher_affinity × company_z
            match_score = affinity * company_z

            if ec not in company_scores:
                company_scores[ec] = {"total": 0.0, "matches": []}

            company_scores[ec]["total"] += match_score

            # Only track strong matches for hypothesis generation
            if affinity > 0.83 and company_z > 0.55:
                company_scores[ec]["matches"].append(ThemeMatch(
                    axis=axis_name,
                    theme_id=tid,
                    theme_name=theme["name"],
                    major_name=theme["major_name"],
                    researcher_affinity=round(affinity, 3),
                    company_strength=round(company_z, 3),
                    match_score=round(match_score, 3),
                    humanities_role=theme.get("humanities_role", ""),
                ))

    # Rank and select top N
    ranked = sorted(company_scores.items(), key=lambda x: x[1]["total"], reverse=True)

    result_companies = []
    for ec, data in ranked[:top_n]:
        co = companies.get(ec, {})
        # Sort theme matches by match_score
        matches = sorted(data["matches"], key=lambda m: m.match_score, reverse=True)[:8]

        # Generate hypothesis text
        hypothesis = _generate_theme_hypothesis(
            co.get("name", ""), co.get("industry", ""), matches)

        # Top themes summary
        if matches:
            axes_summary = {}
            for m in matches[:6]:
                if m.axis not in axes_summary:
                    axes_summary[m.axis] = []
                axes_summary[m.axis].append(m.theme_name)

            parts = []
            axis_labels = {"humanities": "社会的テーマ", "tech": "技術領域", "ambition": "野心領域"}
            for ax, names in axes_summary.items():
                parts.append(f"{axis_labels.get(ax, ax)}: {', '.join(names[:2])}")
            top_summary = " / ".join(parts)
        else:
            top_summary = ""

        result_companies.append(CompanyHypothesis(
            edinet_code=ec,
            company_name=co.get("name", ec),
            industry=co.get("industry", ""),
            total_score=round(data["total"], 2),
            theme_matches=matches,
            hypothesis_text=hypothesis,
            top_themes_summary=top_summary,
        ))

    # Top researcher themes
    top_researcher = sorted(
        researcher_profile.items(),
        key=lambda x: x[1]["affinity"],
        reverse=True
    )[:10]

    return MatchResult(
        researcher_text=researcher_text[:500],
        researcher_profile={k: v for k, v in top_researcher},
        companies=result_companies,
        theme_count=len(researcher_profile),
    )


def _generate_theme_hypothesis(company_name: str, industry: str,
                                matches: list[ThemeMatch]) -> str:
    """Generate a narrative hypothesis from theme matches."""
    if not matches:
        return f"{company_name}との連携可能性は限定的ですが、探索的な接点を検討する余地があります。"

    # Group by axis
    by_axis = {}
    for m in matches:
        if m.axis not in by_axis:
            by_axis[m.axis] = []
        by_axis[m.axis].append(m)

    parts = [f"{company_name}（{industry}）との連携仮説:"]

    if "tech" in by_axis:
        tech = by_axis["tech"][:2]
        tech_names = "・".join(t.theme_name for t in tech)
        parts.append(
            f"技術的には{tech_names}の領域で研究シーズとの親和性が高く、"
            f"共同研究や技術移転の可能性があります。"
        )

    if "ambition" in by_axis:
        amb = by_axis["ambition"][:2]
        amb_names = "・".join(a.theme_name for a in amb)
        parts.append(
            f"同社は{amb_names}に挑戦しようとしており、"
            f"この新領域への参入において研究知見が直接的に貢献できます。"
        )
        # Include humanities_role if available
        for a in amb[:1]:
            if a.humanities_role:
                parts.append(f"({a.humanities_role})")

    if "humanities" in by_axis:
        hum = by_axis["humanities"][:2]
        hum_names = "・".join(h.theme_name for h in hum)
        parts.append(
            f"社会的テーマとしては{hum_names}の課題を抱えており、"
            f"人文社会科学的知見による貢献が期待できます。"
        )

    # Multi-axis synergy
    if len(by_axis) >= 2:
        parts.append(
            "複数の軸で連携の接点が存在し、技術的課題と社会的課題を統合的に解決する"
            "超学際的な連携が実現できる可能性があります。"
        )

    return " ".join(parts)
