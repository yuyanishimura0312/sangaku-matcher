"""NeedFit scorer — semantic match between seed and company's estimated needs.

Uses cosine similarity between seed vector and the company's backcast-estimated
needs vector. Linear mapping (not inverted-U) because high alignment between
a technology seed and a company's need is unambiguously positive.

Based on Von Hippel (1988) need-pull innovation and Bercovitz & Feldman (2007)
demand articulation.
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class NeedFitScorer:
    name = "need_fit"

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        needs_vec_bytes = company.get("needs_vector")
        if needs_vec_bytes is None or len(needs_vec_bytes) == 0:
            return FeatureResult(
                value=0.0,
                rationale="推定ニーズデータなし。",
            )

        needs_vec = np.frombuffer(needs_vec_bytes, dtype=np.float32)
        raw_sim = cosine_similarity(seed_vector, needs_vec)
        raw_sim = max(0.0, min(1.0, raw_sim))

        if raw_sim > 0.6:
            note = (
                f"推定ニーズとの高い一致（{raw_sim:.2f}）。"
                f"企業の潜在的な技術課題と当該シーズが直接対応する可能性が高い。"
            )
        elif raw_sim > 0.3:
            note = (
                f"推定ニーズとの中程度の一致（{raw_sim:.2f}）。"
                f"関連する技術課題が存在する可能性がある。"
            )
        else:
            note = (
                f"推定ニーズとの一致度は低い（{raw_sim:.2f}）。"
                f"直接的なニーズ対応は限定的。"
            )

        return FeatureResult(value=round(raw_sim, 4), rationale=note)
