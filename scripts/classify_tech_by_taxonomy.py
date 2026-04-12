"""Classify tech need narratives against 49-theme tech taxonomy.
Also classify ambition narratives against 40-theme ambition taxonomy.
Builds company × theme proximity matrices for both.

Usage:
    python scripts/classify_tech_by_taxonomy.py
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
DATA_DIR = Path(__file__).parent.parent / "data"


def _build_theme_vectors(taxonomy_file: str, embeddings):
    """Load taxonomy and create theme centroid vectors."""
    with open(DATA_DIR / taxonomy_file) as f:
        taxonomy = json.load(f)

    sub_themes = []
    for major in taxonomy["themes"]:
        for sub in major["sub_themes"]:
            sub_themes.append({
                "theme_id": sub["id"],
                "major_id": major["id"],
                "major_name": major["name"],
                "name": sub["name"],
                "keywords": sub["keywords"],
                "humanities_role": sub.get("humanities_role", ""),
            })

    texts = [f"passage: {st['name']}。{st['major_name']}。{'、'.join(st['keywords'])}"
             for st in sub_themes]
    vectors = embeddings.encode(texts)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    return sub_themes, vectors


def _classify_narratives(conn, sub_themes, theme_vectors, json_col, table_name, embeddings):
    """Classify company narratives against theme vectors and store proximity matrix."""
    rows = conn.execute(
        f"SELECT edinet_code, name, {json_col} FROM companies WHERE {json_col} IS NOT NULL"
    ).fetchall()
    logger.info("Processing %d companies for %s...", len(rows), table_name)

    # Create theme table
    conn.execute(f"DROP TABLE IF EXISTS {table_name}_themes")
    conn.execute(f"""CREATE TABLE {table_name}_themes (
        theme_id TEXT PRIMARY KEY,
        major_id TEXT NOT NULL,
        major_name TEXT NOT NULL,
        name TEXT NOT NULL,
        keywords TEXT NOT NULL,
        humanities_role TEXT,
        centroid BLOB NOT NULL
    )""")
    for i, st in enumerate(sub_themes):
        conn.execute(
            f"INSERT INTO {table_name}_themes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (st["theme_id"], st["major_id"], st["major_name"], st["name"],
             json.dumps(st["keywords"], ensure_ascii=False),
             st.get("humanities_role", ""),
             theme_vectors[i].astype(np.float32).tobytes()),
        )

    # Create proximity table
    conn.execute(f"DROP TABLE IF EXISTS {table_name}_proximity")
    conn.execute(f"""CREATE TABLE {table_name}_proximity (
        edinet_code TEXT NOT NULL,
        theme_id TEXT NOT NULL,
        proximity REAL NOT NULL,
        narrative_count INTEGER DEFAULT 0,
        PRIMARY KEY (edinet_code, theme_id)
    )""")

    # Process companies
    batch_insert = []
    processed = 0
    # Determine the JSON key for needs/ambitions
    needs_key = "ambitions" if "ambition" in json_col else "needs"

    for row in rows:
        try:
            data = json.loads(row[json_col])
        except (json.JSONDecodeError, TypeError):
            continue

        narratives = []
        for item in data.get(needs_key, []):
            text = item.get("narrative", "") or item.get("description", "")
            if text and len(text) > 30:
                narratives.append(text)

        if not narratives:
            continue

        # Embed narratives
        texts = [f"passage: {n}" for n in narratives]
        vecs = embeddings.encode(texts)
        n_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        n_norms[n_norms == 0] = 1
        vecs = vecs / n_norms

        # Similarity matrix
        sim_matrix = vecs @ theme_vectors.T

        ec = row["edinet_code"]
        for j, st in enumerate(sub_themes):
            sims = sim_matrix[:, j]
            max_sim = float(sims.max())
            count = int((sims > 0.85).sum())
            batch_insert.append((ec, st["theme_id"], max_sim, count))

        processed += 1
        if processed % 500 == 0:
            logger.info("  %d/%d", processed, len(rows))

        if len(batch_insert) >= 10000:
            conn.executemany(
                f"INSERT INTO {table_name}_proximity VALUES (?, ?, ?, ?)",
                batch_insert,
            )
            conn.commit()
            batch_insert.clear()

    if batch_insert:
        conn.executemany(
            f"INSERT INTO {table_name}_proximity VALUES (?, ?, ?, ?)",
            batch_insert,
        )
        conn.commit()

    logger.info("  Done: %d companies processed for %s", processed, table_name)


def main():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings

    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row

    # Tech taxonomy (49 themes)
    logger.info("=== Tech Taxonomy (49 themes) ===")
    tech_themes, tech_vectors = _build_theme_vectors("tech_taxonomy.json", embeddings)
    _classify_narratives(conn, tech_themes, tech_vectors,
                         "tech_needs_json", "tech_taxonomy", embeddings)

    # Ambition taxonomy (40 themes)
    logger.info("=== Ambition Taxonomy (40 themes) ===")
    amb_themes, amb_vectors = _build_theme_vectors("ambition_taxonomy.json", embeddings)
    _classify_narratives(conn, amb_themes, amb_vectors,
                         "ambitions_json", "ambition_taxonomy", embeddings)

    # Summary
    for table in ["tech_taxonomy", "ambition_taxonomy"]:
        stats = conn.execute(f"""
            SELECT COUNT(DISTINCT edinet_code) as companies,
                   COUNT(*) as total_rows,
                   ROUND(AVG(proximity), 3) as avg_prox
            FROM {table}_proximity
        """).fetchone()
        logger.info("%s: %d companies, %d rows, avg proximity %.3f",
                    table, stats["companies"], stats["total_rows"], stats["avg_prox"])

    conn.close()
    logger.info("Done. Tables created: tech_taxonomy_themes/proximity, ambition_taxonomy_themes/proximity")


if __name__ == "__main__":
    main()
