"""Bottom-up theme extraction from humanities needs narratives using GTA-inspired approach.

Instead of imposing a top-down taxonomy (7 fixed categories), this script:
1. Extracts individual need narratives from all companies' humanities_needs_json
2. Embeds each narrative independently (not as composite text)
3. Clusters narratives using HDBSCAN to discover emergent themes
4. Labels each cluster with a representative theme description
5. Stores per-company theme proximity scores for fine-grained matching

This enables researchers to find companies with needs in THEIR specific area,
rather than matching against a generic composite vector.

Usage:
    python scripts/extract_themes_gta.py [--min-cluster-size 30] [--test]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cluster-size", type=int, default=30,
                        help="HDBSCAN min cluster size (smaller = more themes)")
    parser.add_argument("--test", action="store_true", help="Use subset for testing")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings

    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row

    # Step 1: Extract individual narratives from all companies
    logger.info("Step 1: Extracting individual need narratives...")
    rows = conn.execute(
        "SELECT edinet_code, name, humanities_needs_json "
        "FROM companies WHERE humanities_needs_json IS NOT NULL"
    ).fetchall()

    if args.test:
        rows = rows[:200]

    narratives = []  # (edinet_code, company_name, type_key, narrative_text)
    for r in rows:
        try:
            data = json.loads(r["humanities_needs_json"])
        except json.JSONDecodeError:
            continue
        for need in data.get("needs", []):
            narrative = need.get("narrative", "")
            if narrative and len(narrative) > 30:
                narratives.append((
                    r["edinet_code"],
                    r["name"],
                    need.get("type", "unknown"),
                    narrative,
                ))

    logger.info("Extracted %d individual narratives from %d companies",
                len(narratives), len(rows))

    # Step 2: Embed each narrative independently
    logger.info("Step 2: Embedding %d narratives...", len(narratives))
    texts = [f"passage: {n[3]}" for n in narratives]

    # Batch encode
    BATCH_SIZE = 256
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vecs = embeddings.encode(batch)
        all_vectors.append(vecs)
        if (i + BATCH_SIZE) % 1000 == 0:
            logger.info("  Embedded %d/%d", min(i + BATCH_SIZE, len(texts)), len(texts))

    vectors = np.vstack(all_vectors)
    logger.info("Embedding complete: shape=%s", vectors.shape)

    # Step 3: Cluster using HDBSCAN
    logger.info("Step 3: Clustering with HDBSCAN (min_cluster_size=%d)...",
                args.min_cluster_size)
    try:
        import hdbscan
    except ImportError:
        logger.error("hdbscan not installed. Run: pip install hdbscan")
        sys.exit(1)

    # Reduce dimensionality first with UMAP for better clustering
    try:
        import umap
        logger.info("  Reducing dimensionality with UMAP (384 -> 20)...")
        reducer = umap.UMAP(n_components=20, metric="cosine", random_state=42, n_jobs=1)
        reduced = reducer.fit_transform(vectors)
    except ImportError:
        logger.warning("umap-learn not installed, clustering on full 384-dim vectors")
        reduced = vectors

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info("Found %d clusters, %d noise points (%.1f%%)",
                n_clusters, n_noise, 100 * n_noise / len(labels))

    # Step 4: Analyze clusters — extract theme labels
    logger.info("Step 4: Analyzing clusters and generating theme labels...")

    themes = {}  # cluster_id -> {centroid, label, keywords, count, companies}
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_vectors = vectors[mask]
        cluster_narratives = [narratives[i] for i in range(len(narratives)) if mask[i]]

        # Centroid
        centroid = cluster_vectors.mean(axis=0)
        centroid /= np.linalg.norm(centroid)

        # Original type distribution
        type_counter = Counter(n[2] for n in cluster_narratives)
        top_types = type_counter.most_common(3)

        # Extract key phrases using TF-IDF-like approach
        all_text = " ".join(n[3] for n in cluster_narratives)

        # Company coverage
        companies = set(n[0] for n in cluster_narratives)

        themes[cluster_id] = {
            "centroid": centroid,
            "count": int(mask.sum()),
            "companies": len(companies),
            "top_types": top_types,
            "sample_narratives": [n[3][:200] for n in cluster_narratives[:3]],
        }

    # Step 5: Use LLM to generate theme labels from sample narratives
    logger.info("Step 5: Generating theme labels with LLM...")
    import os
    import subprocess
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-w"],
                capture_output=True, text=True,
            )
            api_key = result.stdout.strip()
        client = anthropic.Anthropic(api_key=api_key)

        for cid, theme in themes.items():
            samples = "\n---\n".join(theme["sample_narratives"])
            type_info = ", ".join(f"{t}({c}件)" for t, c in theme["top_types"])

            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": (
                    f"以下は企業の人文社会科学系連携ニーズの文章群（{theme['count']}件、{theme['companies']}社）です。\n"
                    f"元の類型分布: {type_info}\n\n"
                    f"サンプル:\n{samples}\n\n"
                    f"これらに共通するテーマを、10-20字の簡潔な日本語ラベルで1つ付けてください。"
                    f"ラベルのみ出力してください。"
                )}],
            )
            theme["label"] = msg.content[0].text.strip().strip("「」")
            logger.info("  Cluster %d (%d narratives, %d companies): %s",
                        cid, theme["count"], theme["companies"], theme["label"])

    except Exception as e:
        logger.warning("LLM labeling failed: %s. Using type-based labels.", e)
        for cid, theme in themes.items():
            top_type = theme["top_types"][0][0] if theme["top_types"] else f"Theme_{cid}"
            theme["label"] = top_type

    # Step 6: Compute per-company theme proximity matrix
    logger.info("Step 6: Computing company × theme proximity matrix...")

    # For each company, compute cosine similarity to each theme centroid
    # using the company's humanities_needs_vector
    company_rows = conn.execute(
        "SELECT edinet_code, humanities_needs_vector FROM companies "
        "WHERE humanities_needs_vector IS NOT NULL"
    ).fetchall()

    theme_ids = sorted(themes.keys())
    centroids = np.array([themes[tid]["centroid"] for tid in theme_ids])

    # Create themes table
    conn.execute("DROP TABLE IF EXISTS humanities_themes")
    conn.execute("""CREATE TABLE humanities_themes (
        theme_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        narrative_count INTEGER,
        company_count INTEGER,
        top_types TEXT,
        sample_narratives TEXT,
        centroid BLOB
    )""")

    for tid in theme_ids:
        t = themes[tid]
        conn.execute(
            "INSERT INTO humanities_themes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                tid,
                t["label"],
                t["count"],
                t["companies"],
                json.dumps(t["top_types"], ensure_ascii=False),
                json.dumps(t["sample_narratives"], ensure_ascii=False),
                t["centroid"].astype(np.float32).tobytes(),
            ),
        )

    # Create company-theme proximity table
    conn.execute("DROP TABLE IF EXISTS company_theme_proximity")
    conn.execute("""CREATE TABLE company_theme_proximity (
        edinet_code TEXT NOT NULL,
        theme_id INTEGER NOT NULL,
        proximity REAL NOT NULL,
        PRIMARY KEY (edinet_code, theme_id),
        FOREIGN KEY (edinet_code) REFERENCES companies(edinet_code),
        FOREIGN KEY (theme_id) REFERENCES humanities_themes(theme_id)
    )""")

    batch_insert = []
    for cr in company_rows:
        ec = cr["edinet_code"]
        cv = np.frombuffer(cr["humanities_needs_vector"], dtype=np.float32)
        cv_norm = cv / np.linalg.norm(cv)

        # Compute similarity to each theme centroid
        sims = centroids @ cv_norm
        for i, tid in enumerate(theme_ids):
            batch_insert.append((ec, tid, float(sims[i])))

        if len(batch_insert) >= 10000:
            conn.executemany(
                "INSERT INTO company_theme_proximity VALUES (?, ?, ?)",
                batch_insert,
            )
            batch_insert.clear()

    if batch_insert:
        conn.executemany(
            "INSERT INTO company_theme_proximity VALUES (?, ?, ?)",
            batch_insert,
        )

    conn.commit()

    # Step 7: Summary statistics
    logger.info("\n=== Theme Extraction Summary ===")
    logger.info("Total narratives: %d", len(narratives))
    logger.info("Clusters found: %d", n_clusters)
    logger.info("Noise narratives: %d (%.1f%%)", n_noise, 100 * n_noise / len(narratives))
    logger.info("")
    for tid in theme_ids:
        t = themes[tid]
        logger.info("Theme %d: %s (%d narratives, %d companies)",
                    tid, t["label"], t["count"], t["companies"])

    # Compute theme proximity distribution stats
    stats = conn.execute("""
        SELECT theme_id, AVG(proximity) as avg,
               MIN(proximity) as min, MAX(proximity) as max,
               COUNT(*) as n
        FROM company_theme_proximity
        GROUP BY theme_id
    """).fetchall()
    logger.info("\nTheme proximity distributions:")
    for s in stats:
        t = themes.get(s["theme_id"], {})
        logger.info("  %s: avg=%.3f, range=[%.3f, %.3f]",
                    t.get("label", f"T{s['theme_id']}"),
                    s["avg"], s["min"], s["max"])

    conn.close()
    logger.info("\nDone. Tables created: humanities_themes, company_theme_proximity")


if __name__ == "__main__":
    main()
