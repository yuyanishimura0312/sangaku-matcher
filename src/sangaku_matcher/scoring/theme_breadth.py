"""ThemeBreadth scorer — counts overlapping themes across all 3 axes.

Measures the breadth of thematic intersection between a researcher and a
company across 122 themes (humanities 33, tech 49, ambition 40).

A high breadth score indicates many potential conversation topics and
collaboration angles, making the pair suitable for exploratory dialogue
even when no single theme is dominant.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from sangaku_matcher.scoring import FeatureResult


class ThemeBreadthScorer:
    """Score companies based on breadth of thematic overlap across 3 axes."""

    name = "theme_breadth"

    def __init__(self):
        self._all_themes: list[dict] | None = None
        self._all_proximities: dict[str, dict[str, float]] | None = None
        self._all_stats: dict[str, tuple[float, float]] | None = None
        self._seed_sims: np.ndarray | None = None
        # Exposed for hypothesis generation — set per company during score()
        self.matching_themes: list[dict] = []

    def load_all_themes(self, conn) -> None:
        """Load theme data from all 3 axes (humanities, tech, ambition)."""
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

        axis_configs = [
            ("humanities", "taxonomy_themes", "company_taxonomy_proximity"),
            ("tech", "tech_taxonomy_themes", "tech_taxonomy_proximity"),
            ("ambition", "ambition_taxonomy_themes", "ambition_taxonomy_proximity"),
        ]

        self._all_themes = []
        self._all_proximities = {}
        theme_vals = defaultdict(list)

        for axis_name, themes_table, prox_table in axis_configs:
            if themes_table not in tables or prox_table not in tables:
                continue

            # Only ambition_taxonomy_themes has humanities_role column
            if axis_name == "ambition":
                cols = "theme_id, major_id, major_name, name, centroid, humanities_role"
            else:
                cols = "theme_id, major_id, major_name, name, centroid"

            rows = conn.execute(
                f"SELECT {cols} FROM {themes_table} ORDER BY theme_id"
            ).fetchall()
            for r in rows:
                centroid = np.frombuffer(r["centroid"], dtype=np.float32)
                composite_id = f"{axis_name}:{r['theme_id']}"
                hr = ""
                if axis_name == "ambition":
                    hr = r["humanities_role"] if "humanities_role" in r.keys() else ""
                self._all_themes.append({
                    "composite_id": composite_id,
                    "theme_id": r["theme_id"],
                    "axis": axis_name,
                    "label": r["name"],
                    "major": r["major_name"],
                    "centroid": centroid,
                    "humanities_role": hr,
                })

            # Load proximities
            prox_rows = conn.execute(
                f"SELECT edinet_code, theme_id, proximity FROM {prox_table}"
            ).fetchall()
            for r in prox_rows:
                ec = r["edinet_code"]
                composite_id = f"{axis_name}:{r['theme_id']}"
                if ec not in self._all_proximities:
                    self._all_proximities[ec] = {}
                self._all_proximities[ec][composite_id] = r["proximity"]
                theme_vals[composite_id].append(r["proximity"])

        # Compute per-theme stats for z-score normalization
        self._all_stats = {}
        for tid, vals in theme_vals.items():
            arr = np.array(vals)
            self._all_stats[tid] = (float(arr.mean()), max(float(arr.std()), 0.001))

    def precompute_seed_themes(self, seed_vector: np.ndarray) -> None:
        """Compute seed similarity to all theme centroids across 3 axes."""
        if not self._all_themes:
            return
        centroids = np.array([t["centroid"] for t in self._all_themes])
        sv_norm = seed_vector / np.linalg.norm(seed_vector)
        self._seed_sims = centroids @ sv_norm

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        ec = company.get("edinet_code", "")

        if not self._all_themes or self._all_proximities is None:
            return FeatureResult(value=0.0, rationale="テーマ広がりデータなし。")

        if ec not in self._all_proximities:
            return FeatureResult(value=0.0, rationale="テーマ近接度データなし。")

        if self._seed_sims is None:
            self.precompute_seed_themes(seed_vector)

        company_prox = self._all_proximities[ec]

        # Count themes where both researcher affinity and company z-score exceed thresholds
        breadth_count = 0
        matching = []
        axis_counts = {"humanities": 0, "tech": 0, "ambition": 0}

        for i, theme in enumerate(self._all_themes):
            cid = theme["composite_id"]
            seed_sim = float(self._seed_sims[i])
            company_raw = company_prox.get(cid, 0.0)

            # Z-score normalize with sigmoid
            mean, std = self._all_stats.get(cid, (0.85, 0.01))
            z = (company_raw - mean) / std
            company_z = 1.0 / (1.0 + np.exp(-z))

            # Threshold check: researcher_affinity > 0.83 AND company_z > 0.55
            if seed_sim > 0.83 and company_z > 0.55:
                breadth_count += 1
                axis_counts[theme["axis"]] += 1
                matching.append({
                    "axis": theme["axis"],
                    "label": theme["label"],
                    "major": theme["major"],
                    "researcher_affinity": round(seed_sim, 3),
                    "company_z": round(company_z, 3),
                })

        # Store matching themes for hypothesis generation
        self.matching_themes = matching

        # Score: breadth_count / 15, capped at 1.0
        final_score = min(1.0, breadth_count / 15.0)

        # Build rationale
        h_count = axis_counts["humanities"]
        t_count = axis_counts["tech"]
        a_count = axis_counts["ambition"]

        if breadth_count > 0:
            # Sort matching themes by combined score for display
            matching.sort(key=lambda m: m["researcher_affinity"] * m["company_z"], reverse=True)
            top_names = [m["label"] for m in matching[:4]]
            rationale = (
                f"{breadth_count}個のテーマで接点"
                f"(人文{h_count}, 技術{t_count}, 野心{a_count}): "
                f"{', '.join(top_names)}。"
            )
        else:
            rationale = "閾値を超える共通テーマなし。"

        return FeatureResult(value=round(final_score, 4), rationale=rationale)
