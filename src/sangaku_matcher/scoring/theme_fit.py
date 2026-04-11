"""ThemeFit scorer — GTA-based bottom-up theme matching.

Instead of matching against a single composite humanities vector,
this scorer:
1. Finds which emergent themes the seed is closest to
2. Looks up each company's proximity to those themes
3. Uses z-score normalization within each theme for discrimination

This captures the specific domain of humanities collaboration need,
e.g., "organizational culture" vs "community design" vs "ethics".

Based on:
- Glaser & Strauss (1967): Grounded Theory — emergent categories from data
- Nooteboom (2007): Optimal cognitive distance in knowledge transfer
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class ThemeFitScorer:
    """Score companies based on theme-level proximity to seed."""

    name = "humanities_fit"  # Replaces the old HumanitiesFit scorer

    def __init__(self):
        self._themes: list[dict] | None = None
        self._company_proximities: dict[str, dict[int, float]] | None = None
        self._theme_stats: dict[int, tuple[float, float]] | None = None
        self._seed_theme_sims: np.ndarray | None = None

    def load_themes(self, conn) -> None:
        """Load pre-computed themes and company-theme proximities."""
        import sqlite3

        # Check if tables exist
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "humanities_themes" not in tables or "company_theme_proximity" not in tables:
            self._themes = None
            return

        # Load themes with centroids
        rows = conn.execute(
            "SELECT theme_id, label, narrative_count, company_count, centroid "
            "FROM humanities_themes ORDER BY theme_id"
        ).fetchall()
        self._themes = []
        for r in rows:
            centroid = np.frombuffer(r["centroid"], dtype=np.float32)
            self._themes.append({
                "theme_id": r["theme_id"],
                "label": r["label"],
                "count": r["narrative_count"],
                "companies": r["company_count"],
                "centroid": centroid,
            })

        # Load company-theme proximities
        prox_rows = conn.execute(
            "SELECT edinet_code, theme_id, proximity FROM company_theme_proximity"
        ).fetchall()
        self._company_proximities = {}
        for r in prox_rows:
            ec = r["edinet_code"]
            if ec not in self._company_proximities:
                self._company_proximities[ec] = {}
            self._company_proximities[ec][r["theme_id"]] = r["proximity"]

        # Pre-compute per-theme proximity stats for z-score normalization
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
        """Compute seed similarity to each theme centroid."""
        if not self._themes:
            return
        centroids = np.array([t["centroid"] for t in self._themes])
        self._seed_theme_sims = centroids @ (seed_vector / np.linalg.norm(seed_vector))

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        ec = company.get("edinet_code", "")

        # Fallback if theme data not loaded
        if not self._themes or self._company_proximities is None:
            return self._fallback_score(seed_vector, company)

        if ec not in self._company_proximities:
            return FeatureResult(value=0.0, rationale="テーマ近接度データなし。")

        if self._seed_theme_sims is None:
            self.precompute_seed_themes(seed_vector)

        company_prox = self._company_proximities[ec]

        # Weighted score: sum of (seed-theme similarity × company-theme z-score)
        # This rewards companies that are strong in the themes the seed cares about
        total_score = 0.0
        total_weight = 0.0
        top_themes = []

        for i, theme in enumerate(self._themes):
            tid = theme["theme_id"]
            seed_sim = self._seed_theme_sims[i]
            company_raw = company_prox.get(tid, 0.0)

            # Z-score normalize company proximity within this theme
            mean, std = self._theme_stats.get(tid, (0.9, 0.01))
            z = (company_raw - mean) / std
            # Sigmoid → [0, 1]
            company_score = 1.0 / (1.0 + np.exp(-z))

            # Seed's interest in this theme as weight
            weight = max(0.0, seed_sim)
            total_score += weight * company_score
            total_weight += weight

            if company_score > 0.6:
                top_themes.append((theme["label"], company_score, seed_sim))

        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        final_score = max(0.0, min(1.0, final_score))

        # Build rationale with top matching themes
        top_themes.sort(key=lambda x: x[1] * x[2], reverse=True)
        if top_themes:
            theme_strs = [f"{t[0]}（企業{t[1]:.2f}×関連度{t[2]:.2f}）"
                          for t in top_themes[:3]]
            rationale = (
                f"テーマ別近接度分析（{len(self._themes)}テーマ）。"
                f"高親和テーマ: {'; '.join(theme_strs)}。"
            )
        else:
            rationale = f"テーマ別分析（{len(self._themes)}テーマ）で顕著な親和性なし。"

        return FeatureResult(value=round(final_score, 4), rationale=rationale)

    def _fallback_score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        """Fallback to composite vector when theme data unavailable."""
        hum_vec_bytes = company.get("humanities_needs_vector")
        if hum_vec_bytes is None or len(hum_vec_bytes) == 0:
            return FeatureResult(value=0.0, rationale="人文系ニーズデータなし。")

        hum_vec = np.frombuffer(hum_vec_bytes, dtype=np.float32)
        raw_sim = cosine_similarity(seed_vector, hum_vec)
        score = max(0.0, min(1.0, raw_sim))
        return FeatureResult(
            value=round(score, 4),
            rationale=f"人文系ニーズとの類似度（フォールバック）: {raw_sim:.3f}",
        )
