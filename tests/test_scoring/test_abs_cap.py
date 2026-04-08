"""Tests for AbsCap scorer."""
import numpy as np
import pytest

from sangaku_matcher.scoring.abs_cap import AbsCapScorer


@pytest.fixture
def scorer():
    s = AbsCapScorer()
    s.set_industry_stats({
        "医薬品": (0.12, 0.05),
        "電気機器": (0.04, 0.02),
        "化学": (0.03, 0.015),
    })
    return s


def _vec():
    return np.zeros(384, dtype=np.float32)


class TestAbsCap:
    def test_high_rd_pharma(self, scorer):
        """High R&D intensity in pharma → above-average score."""
        co = {"rd_intensity": 0.20, "rd_expense": 500000, "industry": "医薬品"}
        result = scorer.score(_vec(), co)
        assert result.value > 0.3  # R&D 500B JPY + above-avg intensity

    def test_zero_rd(self, scorer):
        """Zero R&D expense → 0.0."""
        co = {"rd_intensity": 0.0, "rd_expense": 0, "industry": "化学"}
        result = scorer.score(_vec(), co)
        assert result.value == 0.0

    def test_industry_normalization(self, scorer):
        """Same absolute intensity maps differently across industries."""
        pharma = {"rd_intensity": 0.05, "rd_expense": 100000, "industry": "医薬品"}
        electronics = {"rd_intensity": 0.05, "rd_expense": 100000, "industry": "電気機器"}
        r_pharma = scorer.score(_vec(), pharma)
        r_elec = scorer.score(_vec(), electronics)
        # 5% is below pharma avg (12%) but above electronics avg (4%)
        assert r_elec.value > r_pharma.value

    def test_missing_industry_uses_fallback(self, scorer):
        """Unknown industry → uses default (0.03, 0.03)."""
        co = {"rd_intensity": 0.05, "rd_expense": 50000, "industry": "unknown"}
        result = scorer.score(_vec(), co)
        assert 0.0 < result.value <= 1.0
