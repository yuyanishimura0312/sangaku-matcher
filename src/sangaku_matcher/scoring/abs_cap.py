"""AbsCap scorer — absorptive capacity based on R&D metrics.

Combines:
1. R&D intensity (R&D expense / revenue), normalized by industry
2. R&D expense absolute amount (log-transformed)

Based on Cohen & Levinthal (1990): firms with higher internal R&D
can better absorb external knowledge from universities.
"""
from __future__ import annotations

import math

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class AbsCapScorer:
    name = "abs_cap"

    def __init__(self):
        # Will be populated from industry_stats table
        self._industry_stats: dict[str, tuple[float, float]] = {}

    def set_industry_stats(self, stats: dict[str, tuple[float, float]]) -> None:
        """Set industry-level normalization data.

        Args:
            stats: {industry: (mean_rd_intensity, std_rd_intensity)}
        """
        self._industry_stats = stats

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        rd_intensity = company.get("rd_intensity") or 0.0
        rd_expense = company.get("rd_expense") or 0.0
        industry = company.get("industry") or ""

        if rd_expense <= 0:
            return FeatureResult(value=0.0, rationale="No R&D expense reported.")

        # 1) Industry-normalized R&D intensity (z-score, clamped to [0,1])
        mean, std = self._industry_stats.get(industry, (0.03, 0.03))
        if std > 0:
            z_score = (rd_intensity - mean) / std
        else:
            z_score = 0.0
        # Map z-score to [0, 1] using sigmoid-like transform
        intensity_score = 1.0 / (1.0 + math.exp(-z_score))

        # 2) R&D expense magnitude (log-scaled, normalized)
        # log10(1M JPY) = 6, log10(1T JPY) = 12 → range ~6-12
        log_rd = math.log10(max(rd_expense, 1.0))
        # Normalize: 6 → 0.0, 12 → 1.0
        magnitude_score = max(0.0, min(1.0, (log_rd - 6.0) / 6.0))

        # Cognitive proximity: can this company absorb THIS seed's technology?
        # Zahra & George (2002): potential vs. realized absorptive capacity
        cog_prox = 0.0
        rd_vec_bytes = company.get("rd_text_vector")
        if rd_vec_bytes and len(rd_vec_bytes) > 0:
            rd_vec = np.frombuffer(rd_vec_bytes, dtype=np.float32)
            cog_prox = cosine_similarity(seed_vector, rd_vec)
            cog_prox = max(0.0, min(1.0, cog_prox))

        # Combined: financial capacity (70%) + cognitive proximity (30%)
        combined = 0.4 * intensity_score + 0.3 * magnitude_score + 0.3 * cog_prox

        rationale = (
            f"R&D投資比率 {rd_intensity:.1%}"
            f"（業界平均{'以上' if z_score > 0 else '以下'}）、"
            f"R&D費 {rd_expense:,.0f}百万円、"
            f"認知的近接性 {cog_prox:.2f}。"
        )

        return FeatureResult(value=round(combined, 4), rationale=rationale)
