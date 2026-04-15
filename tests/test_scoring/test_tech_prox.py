"""Tests for TechProx scorer."""
import numpy as np
import pytest

from sangaku_matcher.scoring.tech_prox import TechProxScorer


@pytest.fixture
def scorer():
    return TechProxScorer()


def _make_company(vec: np.ndarray | None) -> dict:
    return {"rd_text_vector": vec.tobytes() if vec is not None else None}


def _make_companies(vecs: list[np.ndarray]) -> list[dict]:
    return [_make_company(v) for v in vecs]


class TestTechProx:
    def test_none_vector_returns_zero(self, scorer):
        """Missing company vector -> 0.0 score."""
        v = np.random.randn(384).astype(np.float32)
        result = scorer.score(v, _make_company(None))
        assert result.value == 0.0

    def test_fallback_inverted_u_without_precompute(self, scorer):
        """Without precompute_distribution, falls back to inverted-U."""
        v = np.random.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        result = scorer.score(v, _make_company(v))
        # Identical vectors (sim=1.0) -> f(1)=0 in inverted-U fallback
        assert result.value < 0.1

    def test_z_score_discrimination(self):
        """After precompute, z-score normalization spreads scores across [0, 1]."""
        seed = np.random.randn(384).astype(np.float32)
        seed /= np.linalg.norm(seed)

        # Create companies at varying distances from seed
        np.random.seed(42)
        companies = []
        for scale in [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]:
            noise = np.random.randn(384).astype(np.float32) * scale
            vec = seed + noise
            vec /= np.linalg.norm(vec)
            companies.append(_make_company(vec))

        scorer = TechProxScorer()
        scorer.precompute_distribution(seed, companies)

        scores = [scorer.score(seed, co).value for co in companies]

        # Scores should span a wide range (not clustered)
        assert max(scores) - min(scores) > 0.3, (
            f"Score range too narrow: {min(scores):.3f}-{max(scores):.3f}"
        )
        # Most similar company should score highest
        assert scores[0] == max(scores)

    def test_precompute_distribution_sets_stats(self):
        """precompute_distribution sets mean and std."""
        seed = np.random.randn(384).astype(np.float32)
        seed /= np.linalg.norm(seed)

        vecs = [np.random.randn(384).astype(np.float32) for _ in range(10)]
        for v in vecs:
            v /= np.linalg.norm(v)
        companies = _make_companies(vecs)

        scorer = TechProxScorer()
        assert scorer._mean is None
        assert scorer._std is None

        scorer.precompute_distribution(seed, companies)

        assert scorer._mean is not None
        assert scorer._std is not None
        assert scorer._std > 0.0

    def test_score_range(self):
        """All scores should be in [0, 1]."""
        seed = np.random.randn(384).astype(np.float32)
        seed /= np.linalg.norm(seed)

        np.random.seed(123)
        companies = []
        for _ in range(20):
            v = np.random.randn(384).astype(np.float32)
            v /= np.linalg.norm(v)
            companies.append(_make_company(v))

        scorer = TechProxScorer()
        scorer.precompute_distribution(seed, companies)

        for co in companies:
            result = scorer.score(seed, co)
            assert 0.0 <= result.value <= 1.0
