"""Scoring engine — pluggable feature scorers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class FeatureResult:
    """Result of a single feature scorer."""
    value: float       # 0.0 - 1.0
    rationale: str     # human-readable explanation


class FeatureScorer(Protocol):
    """Interface that all scorers implement."""
    name: str
    weight: float

    def score(self, seed_vector: np.ndarray, company: dict) -> FeatureResult:
        """Score a seed against a company.

        Args:
            seed_vector: seed's semantic embedding (384,)
            company: dict-like row from companies table
        """
        ...
