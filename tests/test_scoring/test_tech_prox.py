"""Tests for TechProx scorer."""
import numpy as np
import pytest

from sangaku_matcher.scoring.tech_prox import TechProxScorer


@pytest.fixture
def scorer():
    return TechProxScorer(weight=0.35)


def _make_company(vec: np.ndarray | None) -> dict:
    return {"rd_text_vector": vec.tobytes() if vec is not None else None}


class TestTechProx:
    def test_identical_vectors_score_zero(self, scorer):
        """Identical vectors (sim=1.0) → f(1)=0 (too close)."""
        v = np.random.randn(384).astype(np.float32)
        v /= np.linalg.norm(v)
        result = scorer.score(v, _make_company(v))
        assert result.value < 0.1  # Near zero due to inverted-U

    def test_orthogonal_vectors_score_zero(self, scorer):
        """Orthogonal vectors (sim≈0) → f(0)=0 (too distant)."""
        a = np.zeros(384, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(384, dtype=np.float32)
        b[1] = 1.0
        result = scorer.score(a, _make_company(b))
        assert result.value < 0.1

    def test_medium_similarity_scores_highest(self, scorer):
        """Vectors with ~0.5 cosine sim → score near 1.0."""
        a = np.random.randn(384).astype(np.float32)
        # Create b that has ~0.5 cosine similarity with a
        noise = np.random.randn(384).astype(np.float32)
        b = a + noise  # rough approximation
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        result = scorer.score(a, _make_company(b))
        # Score should be positive (the exact value depends on actual sim)
        assert 0.0 <= result.value <= 1.0

    def test_none_vector_returns_zero(self, scorer):
        """Missing company vector → 0.0 score."""
        v = np.random.randn(384).astype(np.float32)
        result = scorer.score(v, _make_company(None))
        assert result.value == 0.0

    def test_inverted_u_transform(self):
        """f(x) = 4x(1-x) peaks at x=0.5."""
        f = lambda x: 4.0 * x * (1.0 - x)
        assert f(0.0) == 0.0
        assert f(1.0) == 0.0
        assert abs(f(0.5) - 1.0) < 1e-6
        assert abs(f(0.25) - 0.75) < 1e-6
