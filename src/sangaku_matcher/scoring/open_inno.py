"""OpenInno scorer — ecosystem readiness and open innovation maturity.

Combines pre-computed open innovation score, breadth of university
partnerships, and industry R&D hub position.

Based on Chesbrough (2003) Open Innovation, Etzkowitz & Leydesdorff (1998)
Triple Helix, Adner (2006) innovation ecosystems.
"""
from __future__ import annotations

import numpy as np

from sangaku_matcher.scoring import FeatureResult


class OpenInnoScorer:
    name = "open_inno"

    def __init__(self):
        self._collab_data: dict[str, dict] = {}
        self._industry_stats: dict[str, tuple[float, float]] = {}

    def set_collaboration_data(self, data: dict[str, dict]) -> None:
        self._collab_data = data

    def set_industry_stats(self, stats: dict[str, tuple[float, float]]) -> None:
        self._industry_stats = stats

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        # (a) Open innovation maturity (pre-computed)
        oi_score = company.get("open_inno_score") or 0.0

        # (b) Collaboration breadth: distinct university partners
        edinet_code = company.get("edinet_code", "")
        collabs = self._collab_data.get(edinet_code, {})
        uni_partners = len(collabs)
        partner_breadth = min(1.0, uni_partners / 10)

        # (c) Industry R&D hub position
        industry = company.get("industry") or ""
        rd_expense = company.get("rd_expense") or 0
        revenue = company.get("revenue") or 0
        mean_rd, _ = self._industry_stats.get(industry, (0.03, 0.03))

        # Approximate industry average R&D expense from mean intensity * revenue
        expected_rd = mean_rd * revenue if mean_rd > 0 and revenue > 0 else 0
        if expected_rd > 0:
            hub_score = min(1.0, rd_expense / (expected_rd * 3))
        else:
            hub_score = 0.0

        score = 0.50 * oi_score + 0.30 * partner_breadth + 0.20 * hub_score

        rationale = (
            f"OI成熟度 {oi_score:.2f}、連携大学数 {uni_partners}"
            f"（幅 {partner_breadth:.2f}）、"
            f"エコシステム貢献度 {hub_score:.2f}。"
        )

        return FeatureResult(value=round(score, 4), rationale=rationale)
