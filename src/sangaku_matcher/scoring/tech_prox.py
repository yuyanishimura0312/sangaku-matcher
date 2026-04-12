"""TechProx scorer — semantic similarity with z-score normalization.

Uses cosine similarity between seed vector and company's R&D text vector,
then applies z-score normalization + sigmoid to produce a relative score
that discriminates well even when raw similarities cluster in a narrow range
(typical for e5-small Japanese embeddings: 0.73-0.81).

Replaces the earlier inverted-U transform (Petruzzelli 2011). The "moderate
distance is optimal" insight is now handled by EXIT_WEIGHTS assigning
negative tech_prox weight to the exploratory exit.
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class TechProxScorer:
    name = "tech_prox"

    def __init__(self):
        self._mean: float | None = None
        self._std: float | None = None

    def precompute_distribution(self, seed_vector: np.ndarray, companies: list[dict]):
        """Pre-compute similarity distribution for z-score normalization."""
        sims = []
        for co in companies:
            vec_bytes = co.get("rd_text_vector")
            if vec_bytes and len(vec_bytes) > 0:
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                sims.append(cosine_similarity(seed_vector, vec))
        if sims:
            self._mean = float(np.mean(sims))
            self._std = max(float(np.std(sims)), 0.001)

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        rd_vec_bytes = company.get("rd_text_vector")
        if rd_vec_bytes is None or len(rd_vec_bytes) == 0:
            return FeatureResult(value=0.0, rationale="R&D text data not available.")

        rd_vec = np.frombuffer(rd_vec_bytes, dtype=np.float32)
        raw_sim = cosine_similarity(seed_vector, rd_vec)
        raw_sim = max(0.0, min(1.0, raw_sim))

        # Z-score normalization: relative position among all companies for THIS seed
        z = 0.0
        if self._mean is not None and self._std is not None:
            z = (raw_sim - self._mean) / self._std
            score = 1.0 / (1.0 + np.exp(-z))  # sigmoid
        else:
            # Fallback: inverted-U (legacy behavior when distribution unavailable)
            score = 4.0 * raw_sim * (1.0 - raw_sim)

        score = max(0.0, min(1.0, score))

        z_str = f", deviation={50+z*10:.0f}" if self._mean else ""
        if score > 0.7:
            note = f"Relatively high tech relevance (raw={raw_sim:.3f}{z_str})."
        elif score > 0.3:
            note = f"Moderate tech relevance (raw={raw_sim:.3f}{z_str})."
        else:
            note = f"Relatively low tech relevance (raw={raw_sim:.3f}{z_str})."

        return FeatureResult(value=round(score, 4), rationale=note)
