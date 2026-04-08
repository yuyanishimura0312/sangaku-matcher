"""Tests for PastTies scorer."""
import math

import numpy as np
import pytest

from sangaku_matcher.scoring.past_ties import PastTiesScorer


@pytest.fixture
def scorer():
    s = PastTiesScorer()
    s.set_collaboration_data({
        "E00001": {"東京大学": 10, "京都大学": 5},
        "E00002": {"大阪大学": 1},
        "E00003": {},
    })
    return s


def _vec():
    return np.zeros(384, dtype=np.float32)


class TestPastTies:
    def test_company_with_collaborations(self, scorer):
        """Company with 15 total collaborations gets meaningful score."""
        co = {"edinet_code": "E00001"}
        result = scorer.score(_vec(), co)
        expected = math.log1p(15) / math.log1p(50)
        assert abs(result.value - round(expected, 4)) < 0.001
        assert result.value > 0.3

    def test_company_with_single_collab(self, scorer):
        """Company with 1 collaboration gets low but non-zero score."""
        co = {"edinet_code": "E00002"}
        result = scorer.score(_vec(), co)
        assert 0.0 < result.value < 0.3

    def test_company_with_empty_collabs(self, scorer):
        """Company with empty collab dict scores 0."""
        co = {"edinet_code": "E00003"}
        result = scorer.score(_vec(), co)
        assert result.value == 0.0

    def test_unknown_company(self, scorer):
        """Company not in collab data scores 0."""
        co = {"edinet_code": "UNKNOWN"}
        result = scorer.score(_vec(), co)
        assert result.value == 0.0

    def test_high_collab_capped_at_one(self, scorer):
        """50+ collaborations should cap at 1.0."""
        scorer.set_collaboration_data({"E99999": {"MIT": 60}})
        co = {"edinet_code": "E99999"}
        result = scorer.score(_vec(), co)
        assert result.value == 1.0

    def test_rationale_contains_top_unis(self, scorer):
        """Rationale mentions top collaborating universities."""
        co = {"edinet_code": "E00001"}
        result = scorer.score(_vec(), co)
        assert "東京大学" in result.rationale
