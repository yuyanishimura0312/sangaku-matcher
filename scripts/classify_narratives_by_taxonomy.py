"""Classify company narratives against the 33-theme taxonomy and build proximity matrix.

For each of the 33 sub-themes, creates a representative vector from the theme name + keywords,
then computes cosine similarity between each company's individual narratives and each theme.
Stores the max similarity per company-theme pair as the proximity score.

Usage:
    python scripts/classify_narratives_by_taxonomy.py
"""
from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"
TAXONOMY_FILE = Path(__file__).parent.parent / "data" / "theme_taxonomy.json"


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings
    from sangaku_matcher.embeddings import cosine_similarity

    # Load taxonomy
    with open(TAXONOMY_FILE) as f:
        taxonomy = json.load(f)

    # Flatten to sub-themes
    sub_themes = []
    for major in taxonomy["themes"]:
        for sub in major["sub_themes"]:
            sub_themes.append({
                "theme_id": sub["id"],
                "major_id": major["id"],
                "major_name": major["name"],
                "name": sub["name"],
                "keywords": sub["keywords"],
            })

    logger.info("Loaded %d sub-themes from taxonomy", len(sub_themes))

    # Create theme vectors by embedding theme name + keywords
    theme_texts = []
    for st in sub_themes:
        text = f"passage: {st['name']}。{st['major_name']}。{'、'.join(st['keywords'])}"
        theme_texts.append(text)

    logger.info("Embedding %d theme vectors...", len(theme_texts))
    theme_vectors = embeddings.encode(theme_texts)
    # Normalize
    norms = np.linalg.norm(theme_vectors, axis=1, keepdims=True)
    theme_vectors = theme_vectors / norms

    logger.info("Theme vectors shape: %s", theme_vectors.shape)

    # Load company narratives
    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT edinet_code, name, humanities_needs_json "
        "FROM companies WHERE humanities_needs_json IS NOT NULL"
    ).fetchall()

    logger.info("Processing %d companies...", len(rows))

    # Create tables
    conn.execute("DROP TABLE IF EXISTS taxonomy_themes")
    conn.execute("""CREATE TABLE taxonomy_themes (
        theme_id TEXT PRIMARY KEY,
        major_id TEXT NOT NULL,
        major_name TEXT NOT NULL,
        name TEXT NOT NULL,
        keywords TEXT NOT NULL,
        centroid BLOB NOT NULL
    )""")

    for i, st in enumerate(sub_themes):
        conn.execute(
            "INSERT INTO taxonomy_themes VALUES (?, ?, ?, ?, ?, ?)",
            (
                st["theme_id"],
                st["major_id"],
                st["major_name"],
                st["name"],
                json.dumps(st["keywords"], ensure_ascii=False),
                theme_vectors[i].astype(np.float32).tobytes(),
            ),
        )

    conn.execute("DROP TABLE IF EXISTS company_taxonomy_proximity")
    conn.execute("""CREATE TABLE company_taxonomy_proximity (
        edinet_code TEXT NOT NULL,
        theme_id TEXT NOT NULL,
        proximity REAL NOT NULL,
        narrative_count INTEGER DEFAULT 0,
        PRIMARY KEY (edinet_code, theme_id)
    )""")

    # Process each company
    batch_insert = []
    BATCH_SIZE = 256
    processed = 0

    for row in rows:
        ec = row["edinet_code"]
        try:
            data = json.loads(row["humanities_needs_json"])
        except json.JSONDecodeError:
            continue

        narratives = []
        for need in data.get("needs", []):
            narrative = need.get("narrative", "")
            if narrative and len(narrative) > 30:
                narratives.append(narrative)

        if not narratives:
            continue

        # Embed all narratives for this company
        narrative_texts = [f"passage: {n}" for n in narratives]
        narrative_vecs = embeddings.encode(narrative_texts)
        n_norms = np.linalg.norm(narrative_vecs, axis=1, keepdims=True)
        n_norms[n_norms == 0] = 1
        narrative_vecs = narrative_vecs / n_norms

        # Compute similarity: each narrative × each theme
        # sim_matrix shape: (n_narratives, n_themes)
        sim_matrix = narrative_vecs @ theme_vectors.T

        # For each theme: max similarity across all narratives + count above threshold
        for j, st in enumerate(sub_themes):
            theme_sims = sim_matrix[:, j]
            max_sim = float(theme_sims.max())
            # Count narratives with >0.85 similarity (strong match)
            count = int((theme_sims > 0.85).sum())
            batch_insert.append((ec, st["theme_id"], max_sim, count))

        processed += 1
        if processed % 500 == 0:
            logger.info("Processed %d/%d companies", processed, len(rows))

        # Batch insert
        if len(batch_insert) >= 10000:
            conn.executemany(
                "INSERT INTO company_taxonomy_proximity VALUES (?, ?, ?, ?)",
                batch_insert,
            )
            conn.commit()
            batch_insert.clear()

    # Final batch
    if batch_insert:
        conn.executemany(
            "INSERT INTO company_taxonomy_proximity VALUES (?, ?, ?, ?)",
            batch_insert,
        )
        conn.commit()

    logger.info("Processed %d companies total", processed)

    # Print summary stats
    stats = conn.execute("""
        SELECT t.theme_id, t.name,
               AVG(p.proximity) as avg_prox,
               MIN(p.proximity) as min_prox,
               MAX(p.proximity) as max_prox,
               AVG(p.narrative_count) as avg_count
        FROM taxonomy_themes t
        JOIN company_taxonomy_proximity p ON t.theme_id = p.theme_id
        GROUP BY t.theme_id
        ORDER BY t.theme_id
    """).fetchall()

    logger.info("\n=== Theme Proximity Summary ===")
    for s in stats:
        logger.info("  %s %s: avg=%.3f [%.3f-%.3f] avg_matches=%.1f",
                     s["theme_id"], s["name"], s["avg_prox"],
                     s["min_prox"], s["max_prox"], s["avg_count"])

    conn.close()
    logger.info("\nDone. Tables: taxonomy_themes, company_taxonomy_proximity")


if __name__ == "__main__":
    main()
