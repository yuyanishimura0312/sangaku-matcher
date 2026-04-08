"""Seed parser — convert user input into a unified Seed object."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from sangaku_matcher import embeddings


@dataclass
class Seed:
    seed_id: str
    title: str
    description: str
    doi: str | None = None
    patent_no: str | None = None
    semantic_vector: np.ndarray = field(default_factory=lambda: np.zeros(384, dtype=np.float32))
    source_type: str = "text"
    created_at: str = ""


def parse_seed(
    description: str,
    title: str = "",
    doi: str | None = None,
    patent_no: str | None = None,
) -> Seed:
    """Parse user input into a Seed with embedding vector.

    Args:
        description: free-text description (required, 200-2000 chars recommended)
        title: optional title
        doi: optional paper DOI (Semantic Scholar fetch is Phase 2)
        patent_no: optional JP patent number
    """
    if not description or len(description.strip()) < 10:
        raise ValueError("Seed description must be at least 10 characters.")

    # Truncate overly long text to keep embedding quality
    MAX_CHARS = 8000
    if len(description) > MAX_CHARS:
        cut = description[:MAX_CHARS].rfind("。")
        description = description[: cut + 1] if cut > 0 else description[:MAX_CHARS]

    source_type = "text"
    if doi and patent_no:
        source_type = "mixed"
    elif doi:
        source_type = "doi"
    elif patent_no:
        source_type = "patent"

    # Prepend 'query: ' for e5 model best practice
    embed_text = f"query: {title} {description}".strip()
    vector = embeddings.encode_single(embed_text)

    return Seed(
        seed_id=str(uuid.uuid4()),
        title=title or description[:60],
        description=description,
        doi=doi or None,
        patent_no=patent_no or None,
        semantic_vector=vector,
        source_type=source_type,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
