"""HumanitiesFit scorer — semantic match for humanities/social science collaboration.

Uses cosine similarity between seed vector and company's humanities needs vector.
The humanities needs vector is derived from contextual narrative descriptions
(not keywords) extracted from 有報 text, capturing social insight, futures
intelligence, ethics/governance, design anthropology, organizational culture,
community design, and narrative/communication collaboration needs.

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


class HumanitiesFitScorer:
    name = "humanities_fit"

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

        if raw_sim > 0.5:
            note = (
                f"人文社会科学系ニーズとの高い親和性（{raw_sim:.2f}）。"
                f"検出された連携類型: {types_str}。"
                f"技術開発以外の知的連携（社会洞察・未来洞察・倫理等）が期待できる。"
            )
        elif raw_sim > 0.25:
            note = (
                f"人文社会科学系ニーズとの中程度の親和性（{raw_sim:.2f}）。"
                f"関連する連携類型: {types_str}。"
            )
        else:
            note = (
                f"人文社会科学系ニーズとの一致度は低い（{raw_sim:.2f}）。"
                f"技術的連携が主な接点となる可能性が高い。"
            )

        return FeatureResult(value=round(raw_sim, 4), rationale=note)
