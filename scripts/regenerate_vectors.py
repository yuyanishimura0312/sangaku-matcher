"""Regenerate needs_vector for all companies with estimated_needs.

Re-embeds estimated_needs text using the embedding model.
Skips companies that already have up-to-date vectors (same text hash).

Usage:
    python scripts/regenerate_vectors.py [--batch-size 100]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "matcher.db"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    companies = conn.execute(
        "SELECT edinet_code, estimated_needs FROM companies WHERE estimated_needs IS NOT NULL"
    ).fetchall()
    logger.info("Regenerating vectors for %d companies", len(companies))

    now = datetime.now().isoformat(timespec="seconds")
    processed = 0

    for batch_start in range(0, len(companies), args.batch_size):
        batch = companies[batch_start:batch_start + args.batch_size]
        texts = [f"passage: {co['estimated_needs']}" for co in batch]

        vectors = embeddings.encode(texts)

        for i, co in enumerate(batch):
            conn.execute(
                "UPDATE companies SET needs_vector = ? WHERE edinet_code = ?",
                (vectors[i].tobytes(), co["edinet_code"]),
            )

        conn.commit()
        processed += len(batch)
        logger.info("Progress: %d / %d", processed, len(companies))

    conn.close()
    logger.info("Done. Regenerated vectors for %d companies.", processed)


if __name__ == "__main__":
    main()
