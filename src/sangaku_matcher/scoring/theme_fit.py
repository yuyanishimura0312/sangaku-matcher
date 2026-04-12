"""ThemeFit scorer — GTA-based bottom-up theme matching with 33-theme taxonomy.

Uses a structured 2-level taxonomy (8 major / 33 sub-themes) derived from
bottom-up analysis of 19,495 individual narratives across 3,828 companies.

For each seed, computes similarity to all 33 theme centroids, then looks up
each company's proximity to those themes. Uses z-score normalization within
each theme for discrimination.

Based on:
- Glaser & Strauss (1967): Grounded Theory — emergent categories from data
- Nooteboom (2007): Optimal cognitive distance in knowledge transfer
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class ThemeFitScorer:
    """Score companies based on 33-theme taxonomy proximity to seed."""

    name = "humanities_fit"

    def __init__(self):
        self._themes: list[dict] | None = None
        self._company_proximities: dict[str, dict[str, float]] | None = None
        self._theme_stats: dict[str, tuple[float, float]] | None = None
        self._seed_theme_sims: np.ndarray | None = None

    def load_themes(self, conn) -> None:
        """Load taxonomy themes and company-theme proximities."""
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

        # Prefer 33-theme taxonomy, fall back to HDBSCAN clusters
        if "taxonomy_themes" in tables and "company_taxonomy_proximity" in tables:
            self._load_taxonomy(conn)
        elif "humanities_themes" in tables and "company_theme_proximity" in tables:
            self._load_hdbscan(conn)
        else:
            self._themes = None

    def _load_taxonomy(self, conn) -> None:
        """Load 33-theme structured taxonomy."""
        rows = conn.execute(
            "SELECT theme_id, major_id, major_name, name, centroid "
            "FROM taxonomy_themes ORDER BY theme_id"
        ).fetchall()
        self._themes = []
        for r in rows:
            centroid = np.frombuffer(r["centroid"], dtype=np.float32)
            self._themes.append({
                "theme_id": r["theme_id"],
                "label": r["name"],
                "major": r["major_name"],
                "centroid": centroid,
            })

        prox_rows = conn.execute(
            "SELECT edinet_code, theme_id, proximity FROM company_taxonomy_proximity"
        ).fetchall()
        self._company_proximities = {}
        for r in prox_rows:
            ec = r["edinet_code"]
            if ec not in self._company_proximities:
                self._company_proximities[ec] = {}
            self._company_proximities[ec][r["theme_id"]] = r["proximity"]

        self._compute_theme_stats()

    def _load_hdbscan(self, conn) -> None:
        """Fallback: load HDBSCAN cluster themes."""
        rows = conn.execute(
            "SELECT theme_id, label, centroid FROM humanities_themes ORDER BY theme_id"
        ).fetchall()
        self._themes = []
        for r in rows:
            centroid = np.frombuffer(r["centroid"], dtype=np.float32)
            self._themes.append({
                "theme_id": str(r["theme_id"]),
                "label": r["label"],
                "major": "",
                "centroid": centroid,
            })

        prox_rows = conn.execute(
            "SELECT edinet_code, theme_id, proximity FROM company_theme_proximity"
        ).fetchall()
        self._company_proximities = {}
        for r in prox_rows:
            ec = r["edinet_code"]
            if ec not in self._company_proximities:
                self._company_proximities[ec] = {}
            self._company_proximities[ec][str(r["theme_id"])] = r["proximity"]

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
        """Compute seed similarity to each theme centroid."""
        if not self._themes:
            return
        centroids = np.array([t["centroid"] for t in self._themes])
        sv_norm = seed_vector / np.linalg.norm(seed_vector)
        self._seed_theme_sims = centroids @ sv_norm

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        ec = company.get("edinet_code", "")

        if not self._themes or self._company_proximities is None:
            return self._fallback_score(seed_vector, company)

        if ec not in self._company_proximities:
            return FeatureResult(value=0.0, rationale="テーマ近接度データなし。")

        if self._seed_theme_sims is None:
            self.precompute_seed_themes(seed_vector)

        company_prox = self._company_proximities[ec]

        # Normalize seed affinities: relative strength among all themes.
        # Raw cosine sims cluster in a narrow range (e5-small); z-score
        # normalization highlights which themes are truly relevant to THIS seed.
        seed_sims = self._seed_theme_sims
        seed_mean = float(seed_sims.mean())
        seed_std = max(float(seed_sims.std()), 0.001)

        # Weighted score: sum of (normalized seed weight × company-theme z-score)
        total_score = 0.0
        total_weight = 0.0
        top_themes = []

        for i, theme in enumerate(self._themes):
            tid = theme["theme_id"]
            raw_seed_sim = float(seed_sims[i])
            company_raw = company_prox.get(tid, 0.0)

            # Z-score normalize company proximity
            mean, std = self._theme_stats.get(tid, (0.85, 0.01))
            z = (company_raw - mean) / std
            company_score = 1.0 / (1.0 + np.exp(-z))

            # Weight = z-scored seed affinity; themes below average get weight 0
            weight = max(0.0, (raw_seed_sim - seed_mean) / seed_std)
            total_score += weight * company_score
            total_weight += weight

            if company_score > 0.6:
                top_themes.append((theme["label"], company_score, raw_seed_sim))

        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0

        final_score = max(0.0, min(1.0, final_score))

        # Build rationale with top matching themes
        top_themes.sort(key=lambda x: x[1] * x[2], reverse=True)
        if top_themes:
            theme_strs = [f"{t[0]}({t[1]:.0%})" for t in top_themes[:3]]
            rationale = (
                f"{len(self._themes)}テーマ分析。"
                f"高親和: {'; '.join(theme_strs)}。"
            )
        else:
            rationale = f"{len(self._themes)}テーマ分析で顕著な親和性なし。"

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
