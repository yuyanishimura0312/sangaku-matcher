"""TechProx scorer — semantic similarity with inverted-U transform.

Uses cosine similarity between seed vector and company's R&D text vector,
then applies f(x) = 4x(1-x) so that medium similarity (~0.5) scores highest.
This follows Petruzzelli (2011) finding that moderate technological proximity
yields the most valuable collaborations.
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class TechProxScorer:
    name = "tech_prox"

    def __init__(self, weight: float = 0.35):
        self.weight = weight

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        rd_vec_bytes = company.get("rd_text_vector")
        if rd_vec_bytes is None or len(rd_vec_bytes) == 0:
            return FeatureResult(value=0.0, rationale="R&D text data not available.")

        rd_vec = np.frombuffer(rd_vec_bytes, dtype=np.float32)
        raw_sim = cosine_similarity(seed_vector, rd_vec)
        # Clamp to [0, 1] — cosine sim can be negative for unrelated content
        raw_sim = max(0.0, min(1.0, raw_sim))
        # Inverted-U: peaks at 0.5 similarity (moderate distance is optimal)
        transformed = 4.0 * raw_sim * (1.0 - raw_sim)

        if raw_sim > 0.7:
            note = f"High similarity ({raw_sim:.2f}) — technology overlap may be too close for novel collaboration."
        elif raw_sim > 0.3:
            note = f"Moderate similarity ({raw_sim:.2f}) — optimal zone for complementary collaboration."
        else:
            note = f"Low similarity ({raw_sim:.2f}) — technology fields may be too distant."

        return FeatureResult(value=round(transformed, 4), rationale=note)
