"""Shared embedding model for semantic similarity computation.

Supports two backends:
- sentence-transformers (default, requires PyTorch)
- ONNX Runtime via optimum (lightweight, set USE_ONNX=1)
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_use_onnx = os.environ.get("USE_ONNX", "0") == "1"


def _get_model():
    """Lazy-load embedding model (singleton)."""
    global _model
    if _model is not None:
        return _model

    from sangaku_matcher.config import settings
    model_name = settings.embedding_model

    if _use_onnx:
        logger.info("Loading ONNX embedding model: %s", model_name)
        _model = _OnnxEmbedder(model_name)
    else:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Model loaded (dim=%d)", _model.get_sentence_embedding_dimension())

    return _model


class _OnnxEmbedder:
    """Lightweight embedder using ONNX Runtime + tokenizers."""

    def __init__(self, model_name: str):
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        import os

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # If path is a local directory with pre-exported ONNX, load directly
        if os.path.isdir(model_name):
            self._model = ORTModelForFeatureExtraction.from_pretrained(model_name)
        else:
            self._model = ORTModelForFeatureExtraction.from_pretrained(
                model_name, export=True
            )
        # Get embedding dimension from a test run
        test = self._tokenizer("test", return_tensors="np", padding=True, truncation=True)
        out = self._model(**{k: v for k, v in test.items()})
        self._dim = out.last_hidden_state.shape[-1]
        logger.info("ONNX model loaded (dim=%d)", self._dim)

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts: list[str], batch_size: int = 32,
               show_progress_bar: bool = False, convert_to_numpy: bool = True):
        """Encode texts, mimicking sentence-transformers API."""
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="np",
                padding=True, truncation=True, max_length=512,
            )
            outputs = self._model(**{k: v for k, v in inputs.items()})
            # Mean pooling over token embeddings
            token_embs = outputs.last_hidden_state  # (batch, seq, dim)
            mask = inputs["attention_mask"]  # (batch, seq)
            mask_expanded = np.expand_dims(mask, -1)  # (batch, seq, 1)
            summed = np.sum(token_embs * mask_expanded, axis=1)
            counts = np.clip(mask_expanded.sum(axis=1), 1, None)
            pooled = summed / counts
            all_vecs.append(pooled)
        return np.vstack(all_vecs).astype(np.float32)


def encode(texts: list[str]) -> np.ndarray:
    """Encode texts into embedding vectors.

    Returns shape (len(texts), 384) float32 array.
    """
    model = _get_model()
    if isinstance(model, _OnnxEmbedder):
        return model.encode(texts)
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
    """Cosine similarity between one query vector and a matrix of vectors."""
    norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    safe_norms = np.where(norms * query_norm > 0, norms * query_norm, 1.0)
    return np.dot(matrix, query) / safe_norms
