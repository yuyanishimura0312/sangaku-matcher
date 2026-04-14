"""NeedFit scorer — semantic match between seed and company's tech needs.

Uses cosine similarity between seed vector and the company's contextual
tech needs vector, with z-score normalization to handle the narrow distribution
of e5 embedding similarities (typical range: 0.79-0.87).

Based on Von Hippel (1988) need-pull innovation and Bercovitz & Feldman (2007)
demand articulation.
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


def _sigmoid(x: float) -> float:
    """Sigmoid mapping: z-score → [0, 1] with midpoint at z=0."""
    return 1.0 / (1.0 + np.exp(-x))


class NeedFitScorer:
    name = "need_fit"

    def __init__(self):
        self._sim_cache: list[float] = []
        self._mean: float | None = None
        self._std: float | None = None

    def precompute_distribution(self, seed_vector: np.ndarray, companies: list[dict]):
        """Pre-compute similarity distribution for z-score normalization."""
        sims = []
        for co in companies:
            vec_bytes = co.get("tech_needs_vector") or co.get("needs_vector")
            if vec_bytes and len(vec_bytes) > 0:
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                sims.append(cosine_similarity(seed_vector, vec))
        if sims:
            self._mean = float(np.mean(sims))
            self._std = max(float(np.std(sims)), 0.001)  # avoid div by zero

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        # Prefer enriched contextual vector, fall back to legacy
        needs_vec_bytes = company.get("tech_needs_vector") or company.get("needs_vector")
        if needs_vec_bytes is None or len(needs_vec_bytes) == 0:
            return FeatureResult(
                value=0.0,
                rationale="推定ニーズデータなし。",
            )

        needs_vec = np.frombuffer(needs_vec_bytes, dtype=np.float32)
        raw_sim = cosine_similarity(seed_vector, needs_vec)
        raw_sim = max(0.0, min(1.0, raw_sim))

        source = "文脈型" if company.get("tech_needs_vector") else "従来型"

        # Z-score normalization: convert to relative score
        z = 0.0
        if self._mean is not None and self._std is not None:
            z = (raw_sim - self._mean) / self._std
            # Sigmoid maps z-score to [0, 1], centered at mean
            score = float(_sigmoid(z))
        else:
            score = raw_sim

        score = max(0.0, min(1.0, score))

        if score > 0.7:
            note = (
                f"技術ニーズとの相対的に高い一致（raw={raw_sim:.3f}、偏差値{50+z*10:.0f}、{source}）。"
                f"企業の技術課題と当該シーズが直接対応する可能性が高い。"
            )
        elif score > 0.4:
            note = (
                f"技術ニーズとの中程度の一致（raw={raw_sim:.3f}、偏差値{50+z*10:.0f}、{source}）。"
                f"関連する技術課題が存在する可能性がある。"
            )
        else:
            note = (
                f"技術ニーズとの相対的な一致度は低い（raw={raw_sim:.3f}、偏差値{50+z*10:.0f}、{source}）。"
                f"直接的なニーズ対応は限定的。"
            )

        return FeatureResult(value=round(score, 4), rationale=note)
