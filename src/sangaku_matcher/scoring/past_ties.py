"""PastTies scorer — prior collaboration between university and company.

Counts joint patents and co-authored papers. Log-transformed to avoid
domination by a single prolific partnership.
"""
from __future__ import annotations

import math

import numpy as np

from sangaku_matcher.scoring import FeatureResult


class PastTiesScorer:
    name = "past_ties"

    def __init__(self):
        # {edinet_code: {uni_name: count}} — loaded from collaborations table
        self._collab_data: dict[str, dict[str, int]] = {}

    def set_collaboration_data(self, data: dict[str, dict[str, int]]) -> None:
        """Load collaboration data.

        Args:
            data: {edinet_code: {university_name: total_count}}
        """
        self._collab_data = data

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        edinet_code = company.get("edinet_code", "")
        collabs = self._collab_data.get(edinet_code, {})

        if not collabs:
            return FeatureResult(
                value=0.0,
                rationale="No recorded university collaborations."
            )

        total_count = sum(collabs.values())
        uni_names = sorted(collabs.keys(), key=lambda k: collabs[k], reverse=True)[:3]
        # log1p normalization: log(1 + count) / log(1 + 50) as rough ceiling
        # 50 collaborations → score ~1.0
        normalized = math.log1p(total_count) / math.log1p(50)
        normalized = min(1.0, normalized)

        top_unis = ", ".join(f"{u} ({collabs[u]})" for u in uni_names)
        rationale = f"{total_count} collaboration(s) with universities. Top: {top_unis}."

        return FeatureResult(value=round(normalized, 4), rationale=rationale)
