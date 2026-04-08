#!/usr/bin/env python3
"""Load companies from EDINET code list CSV into matcher DB.

Downloads the EDINET code list and inserts all listed domestic corporations
(上場 内国法人) into the companies table. Generates embeddings from
company name + industry for basic scoring capability.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import sys
import tempfile
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sangaku_matcher.config import settings
from sangaku_matcher.db import connect, init_db
from sangaku_matcher import embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EDINET_CODE_LIST_URL = (
    "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/codelist/Edinetcode.zip"
)


def download_edinet_codes() -> list[dict]:
    """Download and parse EDINET code list CSV."""
    logger.info("Downloading EDINET code list...")
    resp = requests.get(EDINET_CODE_LIST_URL, timeout=60)
    resp.raise_for_status()

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "edinetcode.zip"
        zip_path.write_bytes(resp.content)
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            csv_path = zf.extract(csv_name, tmpdir)

        with open(csv_path, "r", encoding="cp932") as f:
            lines = f.readlines()

    # Line 0 = metadata, Line 1 = headers, Line 2+ = data
    reader = csv.reader(lines[2:])
    companies = []
    for row in reader:
        if len(row) < 12:
            continue
        (edinet_code, submitter_type, listing, consolidated,
         capital, fiscal, name, name_en, name_yomi,
         address, industry, sec_code, *rest) = row

        # Only domestic listed corporations
        if submitter_type != "内国法人・組合":
            continue
        if listing != "上場":
            continue
        if not edinet_code or not name:
            continue

        # Parse capital (in million JPY)
        try:
            capital_val = float(capital) if capital else 0
        except ValueError:
            capital_val = 0

        companies.append({
            "edinet_code": edinet_code,
            "sec_code": sec_code or "",
            "name": name,
            "name_en": name_en or "",
            "industry": industry or "",
            "address": address or "",
            "capital": capital_val,
        })

    logger.info("Parsed %d listed companies from EDINET code list.", len(companies))
    return companies


def load_to_db(companies: list[dict], batch_size: int = 100) -> None:
    """Insert companies into matcher DB with embeddings."""
    init_db(settings.matcher_db_path)
    now = datetime.now().isoformat(timespec="seconds")

    total = len(companies)
    loaded = 0

    for i in range(0, total, batch_size):
        batch = companies[i : i + batch_size]

        # Generate embeddings from company name + industry
        texts = []
        for co in batch:
            text_parts = [co["name"]]
            if co["industry"]:
                text_parts.append(co["industry"])
            texts.append(f"passage: {' '.join(text_parts)}")

        vectors = embeddings.encode(texts)

        with connect(settings.matcher_db_path) as conn:
            for j, co in enumerate(batch):
                conn.execute(
                    """INSERT OR REPLACE INTO companies
                       (edinet_code, sec_code, name, industry, revenue, rd_expense,
                        rd_intensity, rd_text, rd_text_vector, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        co["edinet_code"],
                        co["sec_code"],
                        co["name"],
                        co["industry"],
                        co["capital"],  # Use capital as placeholder for revenue
                        0,  # rd_expense unknown
                        0,  # rd_intensity unknown
                        "",  # No R&D text yet
                        vectors[j].tobytes(),
                        now,
                    ),
                )

        loaded += len(batch)
        logger.info("Loaded %d / %d companies...", loaded, total)

    # Compute industry stats
    with connect(settings.matcher_db_path) as conn:
        industries: dict[str, list[float]] = {}
        for row in conn.execute("SELECT industry, rd_intensity FROM companies"):
            industries.setdefault(row["industry"], []).append(row["rd_intensity"] or 0)

        for ind, vals in industries.items():
            mean = sum(vals) / len(vals) if vals else 0
            variance = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.01
            std = math.sqrt(variance) if variance > 0 else 0.01
            conn.execute(
                """INSERT OR REPLACE INTO industry_stats
                   (industry, mean_rd_int, std_rd_int, company_count, computed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (ind, mean, std, len(vals), now),
            )

    logger.info("Done! Loaded %d companies total.", loaded)


def main():
    companies = download_edinet_codes()
    if not companies:
        logger.error("No companies found. Aborting.")
        sys.exit(1)
    load_to_db(companies)


if __name__ == "__main__":
    main()
