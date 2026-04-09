"""PastTies scorer — prior collaboration with network diversity and recency.

Combines volume (log-normalized), network diversity (Shannon entropy),
and recency of collaborations.

Based on Granovetter (1973) strength of weak ties, Burt (2004) structural
holes theory, Powell et al. (1996) network diversity in innovation.
"""
from __future__ import annotations

import math

import numpy as np

from sangaku_matcher.scoring import FeatureResult


class PastTiesScorer:
    name = "past_ties"

    def __init__(self):
        # {edinet_code: {uni_name: {"count": int, "last_year": int}}}
        self._collab_data: dict[str, dict] = {}

    def set_collaboration_data(self, data: dict[str, dict]) -> None:
        """Load collaboration data.

        Args:
            data: {edinet_code: {university_name: {"count": int, "last_year": int}}}
        """
        self._collab_data = data

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        edinet_code = company.get("edinet_code", "")
        collabs = self._collab_data.get(edinet_code, {})

        if not collabs:
            return FeatureResult(
                value=0.0,
                rationale="大学との連携実績なし。",
            )

        counts = [c["count"] if isinstance(c, dict) else c for c in collabs.values()]
        total_count = sum(counts)
        uni_count = len(collabs)

        # Volume: log-normalized total count
        volume = min(1.0, math.log1p(total_count) / math.log1p(50))

        # Diversity: Shannon entropy of university distribution
        if uni_count > 1:
            total = sum(counts)
            probs = [c / total for c in counts if c > 0]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(uni_count)
            diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        elif uni_count == 1:
            diversity = 0.2  # single partner = low diversity but non-zero
        else:
            diversity = 0.0

        # Recency: boost if collaboration is recent
        last_years = []
        for c in collabs.values():
            if isinstance(c, dict) and c.get("last_year"):
                last_years.append(c["last_year"])
        max_last_year = max(last_years) if last_years else 0
        current_year = 2026
        if max_last_year > 0:
            recency = max(0.0, 1.0 - (current_year - max_last_year) / 10)
        else:
            recency = 0.3  # unknown recency = moderate default

        score = 0.5 * volume + 0.3 * diversity + 0.2 * recency

        uni_names = sorted(
            collabs.keys(),
            key=lambda k: collabs[k]["count"] if isinstance(collabs[k], dict) else collabs[k],
            reverse=True,
        )[:3]
        top_unis = ", ".join(
            f"{u} ({collabs[u]['count'] if isinstance(collabs[u], dict) else collabs[u]})"
            for u in uni_names
        )
        rationale = (
            f"{total_count}件の連携実績（{uni_count}大学）。"
            f"多様性 {diversity:.2f}、"
            f"{'直近 ' + str(max_last_year) + '年' if max_last_year > 0 else '時期不明'}。"
            f"主要: {top_unis}。"
        )

        return FeatureResult(value=round(score, 4), rationale=rationale)
