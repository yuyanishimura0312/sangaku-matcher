"""HumanitiesFit scorer — semantic match for humanities/social science collaboration.

Uses cosine similarity between seed vector and company's humanities needs vector,
with z-score normalization to handle the narrow distribution of e5 embedding
similarities (typical range: 0.78-0.85).

Based on:
- SHARPE framework (AHRC/ESRC): Social sciences, Humanities, Arts for People & Economy
- Burt (1992/2004): Structural Holes and cross-domain knowledge brokering
- Meyer (2010): Knowledge Broker as translator between knowledge communities
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult

# Collaboration type labels for humanities matching
COLLAB_TYPES = {
    "social_insight": "社会洞察連携",
    "futures_intelligence": "未来洞察連携",
    "ethics_governance": "倫理・ガバナンス連携",
    "design_anthropology": "デザイン人類学連携",
    "organizational_culture": "組織・文化変革連携",
    "community_design": "コミュニティ・場の設計連携",
    "narrative_communication": "ナラティブ・コミュニケーション連携",
}


def _sigmoid(x: float) -> float:
    """Sigmoid mapping: z-score → [0, 1] with midpoint at z=0."""
    return 1.0 / (1.0 + np.exp(-x))


class HumanitiesFitScorer:
    name = "humanities_fit"

    def __init__(self):
        self._mean: float | None = None
        self._std: float | None = None

    def precompute_distribution(self, seed_vector: np.ndarray, companies: list[dict]):
        """Pre-compute similarity distribution for z-score normalization."""
        sims = []
        for co in companies:
            vec_bytes = co.get("humanities_needs_vector")
            if vec_bytes and len(vec_bytes) > 0:
                vec = np.frombuffer(vec_bytes, dtype=np.float32)
                sims.append(cosine_similarity(seed_vector, vec))
        if sims:
            self._mean = float(np.mean(sims))
            self._std = max(float(np.std(sims)), 0.001)

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        hum_vec_bytes = company.get("humanities_needs_vector")
        if hum_vec_bytes is None or len(hum_vec_bytes) == 0:
            return FeatureResult(
                value=0.0,
                rationale="人文社会科学系ニーズデータなし。",
            )

        hum_vec = np.frombuffer(hum_vec_bytes, dtype=np.float32)
        raw_sim = cosine_similarity(seed_vector, hum_vec)
        raw_sim = max(0.0, min(1.0, raw_sim))

        # Extract detected need types for context
        needs_text = company.get("humanities_needs_text", "") or ""
        detected_types = []
        for key, label in COLLAB_TYPES.items():
            if label in needs_text:
                detected_types.append(label)

        types_str = "、".join(detected_types[:3]) if detected_types else "不明"

        # Z-score normalization
        z = 0.0
        if self._mean is not None and self._std is not None:
            z = (raw_sim - self._mean) / self._std
            score = float(_sigmoid(z))
        else:
            score = raw_sim

        score = max(0.0, min(1.0, score))

        if score > 0.7:
            note = (
                f"人文社会科学系ニーズとの高い親和性（raw={raw_sim:.3f}、偏差値{50+z*10:.0f}）。"
                f"検出された連携類型: {types_str}。"
                f"社会洞察・未来洞察・倫理等の領域での知的連携が期待できる。"
            )
        elif score > 0.4:
            note = (
                f"人文社会科学系ニーズとの中程度の親和性（raw={raw_sim:.3f}、偏差値{50+z*10:.0f}）。"
                f"関連する連携類型: {types_str}。"
            )
        else:
            note = (
                f"人文社会科学系ニーズとの相対的な一致度は低い（raw={raw_sim:.3f}、偏差値{50+z*10:.0f}）。"
                f"技術的連携が主な接点となる可能性が高い。"
            )

        return FeatureResult(value=round(score, 4), rationale=note)
