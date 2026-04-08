"""Shared embedding model for semantic similarity computation."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from sangaku_matcher.config import settings

        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Model loaded (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Encode texts into embedding vectors.

    For multilingual-e5-small, prepend 'query: ' for queries and
    'passage: ' for passages to improve accuracy.

    Returns shape (len(texts), 384) float32 array.
    """
    model = _get_model()
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return vecs.astype(np.float32)


def encode_single(text: str) -> np.ndarray:
    """Encode a single text. Returns shape (384,) float32."""
    return encode([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a matrix of vectors.

    Args:
        query: shape (D,)
        matrix: shape (N, D)

    Returns: shape (N,) similarity scores
    """
    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    # Avoid division by zero
    safe_norms = np.where(norms * query_norm > 0, norms * query_norm, 1.0)
    return np.dot(matrix, query) / safe_norms
