"""AmbitionFit scorer — business ambition taxonomy matching with 40 themes.

Uses a structured taxonomy of corporate ambition themes derived from
medium-term plans and vision statements. Measures how well a researcher's
expertise aligns with the directions companies want to explore.

Each theme includes a humanities_role field describing how humanities/social
science perspectives can contribute to that ambition area.
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class AmbitionFitScorer:
    """Score companies based on 40-theme ambition taxonomy proximity to seed."""

    name = "ambition_fit"

    def __init__(self):
        self._themes: list[dict] | None = None
        self._company_proximities: dict[str, dict[str, float]] | None = None
        self._theme_stats: dict[str, tuple[float, float]] | None = None
        self._seed_theme_sims: np.ndarray | None = None

    def load_themes(self, conn) -> None:
        """Load ambition taxonomy themes and company-theme proximities."""
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

        if "ambition_taxonomy_themes" not in tables or "ambition_taxonomy_proximity" not in tables:
            self._themes = None
            return

        # Load theme definitions with centroids and humanities_role
        rows = conn.execute(
            "SELECT theme_id, major_id, major_name, name, centroid, keywords, humanities_role "
            "FROM ambition_taxonomy_themes ORDER BY theme_id"
        ).fetchall()
        self._themes = []
        for r in rows:
            centroid = np.frombuffer(r["centroid"], dtype=np.float32)
            self._themes.append({
                "theme_id": r["theme_id"],
                "label": r["name"],
                "major": r["major_name"],
                "centroid": centroid,
                "humanities_role": (r["humanities_role"] if "humanities_role" in r.keys() else ""),
            })

        # Load per-company proximity values
        prox_rows = conn.execute(
            "SELECT edinet_code, theme_id, proximity FROM ambition_taxonomy_proximity"
        ).fetchall()
        self._company_proximities = {}
        for r in prox_rows:
            ec = r["edinet_code"]
            if ec not in self._company_proximities:
                self._company_proximities[ec] = {}
            self._company_proximities[ec][r["theme_id"]] = r["proximity"]

        self._compute_theme_stats()

    def _compute_theme_stats(self) -> None:
        """Pre-compute per-theme proximity stats for z-score normalization."""
        from collections import defaultdict
        theme_vals = defaultdict(list)
        for ec_prox in self._company_proximities.values():
            for tid, prox in ec_prox.items():
                theme_vals[tid].append(prox)

        self._theme_stats = {}
        for tid, vals in theme_vals.items():
            arr = np.array(vals)
            self._theme_stats[tid] = (float(arr.mean()), max(float(arr.std()), 0.001))

    def precompute_seed_themes(self, seed_vector: np.ndarray) -> None:
        """Compute seed similarity to each ambition theme centroid."""
        if not self._themes:
            return
        centroids = np.array([t["centroid"] for t in self._themes])
        sv_norm = seed_vector / np.linalg.norm(seed_vector)
        self._seed_theme_sims = centroids @ sv_norm

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        ec = company.get("edinet_code", "")

        if not self._themes or self._company_proximities is None:
            return FeatureResult(value=0.0, rationale="野心領域テーマデータなし。")

        if ec not in self._company_proximities:
            return FeatureResult(value=0.0, rationale="野心領域近接度データなし。")

        if self._seed_theme_sims is None:
            self.precompute_seed_themes(seed_vector)

        company_prox = self._company_proximities[ec]

        # Weighted score: sum of (seed-theme similarity x company-theme z-score)
        total_score = 0.0
        total_weight = 0.0
        top_themes = []

        for i, theme in enumerate(self._themes):
            tid = theme["theme_id"]
            seed_sim = self._seed_theme_sims[i]
            company_raw = company_prox.get(tid, 0.0)

            # Z-score normalize with sigmoid transform
            mean, std = self._theme_stats.get(tid, (0.85, 0.01))
            z = (company_raw - mean) / std
            company_score = 1.0 / (1.0 + np.exp(-z))

            weight = max(0.0, seed_sim)
            total_score += weight * company_score
            total_weight += weight

            if company_score > 0.6:
                top_themes.append((theme["label"], company_score, seed_sim, theme["humanities_role"]))

        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        final_score = max(0.0, min(1.0, final_score))

        # Build rationale with top matching themes and humanities_role
        top_themes.sort(key=lambda x: x[1] * x[2], reverse=True)
        if top_themes:
            theme_strs = [f"{t[0]}({t[1]:.0%})" for t in top_themes[:3]]
            rationale = f"{len(self._themes)}野心テーマ分析。高親和: {'; '.join(theme_strs)}。"
            # Include humanities_role from top theme if available
            top_role = top_themes[0][3]
            if top_role:
                rationale += f" {top_role}"
        else:
            rationale = f"{len(self._themes)}野心テーマ分析で顕著な親和性なし。"

        return FeatureResult(value=round(final_score, 4), rationale=rationale)
