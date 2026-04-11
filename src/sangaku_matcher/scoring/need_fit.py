"""NeedFit scorer — semantic match between seed and company's tech needs.

Uses cosine similarity between seed vector and the company's contextual
tech needs vector. Prefers the enriched tech_needs_vector (derived from
contextual narrative descriptions) over the legacy needs_vector.

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

        # Check which vector source was used
        source = "文脈型" if company.get("tech_needs_vector") else "従来型"

        if raw_sim > 0.6:
            note = (
                f"技術ニーズとの高い一致（{raw_sim:.2f}、{source}）。"
                f"企業の技術課題と当該シーズが直接対応する可能性が高い。"
            )
        elif raw_sim > 0.3:
            note = (
                f"技術ニーズとの中程度の一致（{raw_sim:.2f}、{source}）。"
                f"関連する技術課題が存在する可能性がある。"
            )
        else:
            note = (
                f"技術ニーズとの一致度は低い（{raw_sim:.2f}、{source}）。"
                f"直接的なニーズ対応は限定的。"
            )

        return FeatureResult(value=round(raw_sim, 4), rationale=note)
