"""FutureOption scorer — option value estimation for exploratory collaboration.

Estimates the potential for serendipitous breakthroughs and future value
creation. Higher when: (a) company has broad R&D portfolio, (b) technology
distance is in the "adjacent possible" zone, (c) company has strategic
flexibility to pursue exploratory partnerships.

Based on McGrath (1997) real options in R&D, March (1991) exploration vs.
exploitation, Kauffman (2000) adjacent possible.
"""
from __future__ import annotations

import math

import numpy as np

from sangaku_matcher.embeddings import cosine_similarity
from sangaku_matcher.scoring import FeatureResult


class FutureOptionScorer:
    name = "future_option"

    def __init__(self):
        self._industry_stats: dict[str, tuple[float, float]] = {}

    def set_industry_stats(self, stats: dict[str, tuple[float, float]]) -> None:
        self._industry_stats = stats

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        # (a) R&D breadth: longer R&D text suggests diverse portfolio
        rd_text_len = company.get("rd_text_len") or 0
        breadth = min(1.0, rd_text_len / 5000)

        # (b) Exploration distance: reward moderate-far tech distance
        # Peak at cos_sim ~0.2 (adjacent possible zone)
        exploration = 0.0
        rd_vec_bytes = company.get("rd_text_vector")
        if rd_vec_bytes and len(rd_vec_bytes) > 0:
            rd_vec = np.frombuffer(rd_vec_bytes, dtype=np.float32)
            cos_sim = cosine_similarity(seed_vector, rd_vec)
            cos_sim = max(0.0, min(1.0, cos_sim))
            # Parabola peaking at 0.2: f(x) = max(0, 1 - (x-0.2)^2 / 0.16)
            exploration = max(0.0, 1.0 - (cos_sim - 0.2) ** 2 / 0.16)

        # (c) Strategic flexibility: capacity to pursue exploratory R&D
        industry = company.get("industry") or ""
        rd_intensity = company.get("rd_intensity") or 0.0
        mean_rd, _ = self._industry_stats.get(industry, (0.03, 0.03))

        flexibility = 0.0
        if rd_intensity > mean_rd:
            flexibility += 0.5
        employees = company.get("employees") or 0
        if employees > 5000:
            flexibility += 0.3
        elif employees > 1000:
            flexibility += 0.15
        market_cap = company.get("market_cap") or 0
        if market_cap > 500000:  # > 500B JPY
            flexibility += 0.2
        flexibility = min(1.0, flexibility)

        score = 0.35 * breadth + 0.35 * exploration + 0.30 * flexibility

        if score > 0.5:
            note = (
                f"R&D幅 {breadth:.2f}、探索距離 {exploration:.2f}、"
                f"戦略的柔軟性 {flexibility:.2f}。"
                f"技術距離は大きいが探索余力があり、セレンディピティの可能性がある。"
            )
        else:
            note = (
                f"R&D幅 {breadth:.2f}、探索距離 {exploration:.2f}、"
                f"戦略的柔軟性 {flexibility:.2f}。"
                f"将来オプション価値は限定的。"
            )

        return FeatureResult(value=round(score, 4), rationale=note)
