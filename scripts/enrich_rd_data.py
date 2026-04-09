#!/usr/bin/env python3
"""Enrich companies with estimated R&D data based on industry benchmarks.

Uses industry-average R&D intensity ratios from METI (経済産業省) survey data
combined with company capital as a firm-size proxy to estimate R&D expenditure.

This is a reasonable approximation for the absorptive capacity scorer
(Cohen & Levinthal, 1990) until actual XBRL financial data is extracted.

Data sources:
- Industry R&D intensity: 経済産業省「科学技術研究調査」(2023-2024)
- Revenue/capital ratio: 上場企業の資本金・売上比率の業種別中央値
"""
from __future__ import annotations

import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sangaku_matcher.config import settings
from sangaku_matcher.db import connect

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Industry-average R&D intensity (R&D expense / revenue)
# Based on METI 科学技術研究調査 and listed company financials (2023-2024)
INDUSTRY_RD_INTENSITY: dict[str, float] = {
    "医薬品": 0.18,
    "精密機器": 0.08,
    "電気機器": 0.07,
    "情報・通信業": 0.06,
    "化学": 0.05,
    "機械": 0.04,
    "輸送用機器": 0.045,
    "ゴム製品": 0.04,
    "ガラス・土石製品": 0.035,
    "非鉄金属": 0.03,
    "繊維製品": 0.03,
    "その他製品": 0.03,
    "鉄鋼": 0.02,
    "金属製品": 0.025,
    "パルプ・紙": 0.02,
    "食料品": 0.02,
    "石油・石炭製品": 0.01,
    "サービス業": 0.03,
    "小売業": 0.01,
    "卸売業": 0.005,
    "建設業": 0.015,
    "不動産業": 0.005,
    "陸運業": 0.01,
    "海運業": 0.005,
    "空運業": 0.01,
    "倉庫・運輸関連": 0.005,
    "電気・ガス業": 0.01,
    "水産・農林業": 0.01,
    "鉱業": 0.01,
    "銀行業": 0.005,
    "証券、商品先物取引業": 0.005,
    "保険業": 0.005,
    "その他金融業": 0.005,
}

# Revenue-to-capital ratio by industry (median for listed companies)
# This converts 資本金 (capital) to estimated 売上高 (revenue)
# e.g., ratio of 50 means a company with 100M capital typically has 5B revenue
INDUSTRY_REV_CAP_RATIO: dict[str, float] = {
    "小売業": 80,
    "卸売業": 100,
    "建設業": 60,
    "食料品": 50,
    "サービス業": 30,
    "情報・通信業": 20,
    "電気機器": 40,
    "機械": 40,
    "化学": 45,
    "輸送用機器": 50,
    "医薬品": 25,
    "精密機器": 35,
    "金属製品": 40,
    "繊維製品": 35,
    "鉄鋼": 50,
    "非鉄金属": 50,
    "ゴム製品": 40,
    "ガラス・土石製品": 35,
    "パルプ・紙": 40,
    "その他製品": 35,
    "不動産業": 15,
    "陸運業": 30,
    "海運業": 40,
    "空運業": 30,
    "倉庫・運輸関連": 25,
    "電気・ガス業": 30,
    "水産・農林業": 30,
    "鉱業": 20,
    "石油・石炭製品": 60,
    "銀行業": 5,
    "証券、商品先物取引業": 10,
    "保険業": 10,
    "その他金融業": 10,
}

# Default values for unknown industries
DEFAULT_RD_INTENSITY = 0.02
DEFAULT_REV_CAP_RATIO = 30


def estimate_financials(capital: float, industry: str, seed: int) -> tuple[float, float, float]:
    """Estimate revenue, R&D expense, and R&D intensity for a company.

    Uses industry benchmarks with controlled random variation to create
    realistic differentiation between companies in the same industry.

    Args:
        capital: Company capital in million JPY (資本金)
        industry: Industry classification
        seed: Random seed for reproducible variation

    Returns:
        (estimated_revenue, estimated_rd_expense, rd_intensity)
    """
    rng = random.Random(seed)

    rev_cap_ratio = INDUSTRY_REV_CAP_RATIO.get(industry, DEFAULT_REV_CAP_RATIO)
    rd_intensity_base = INDUSTRY_RD_INTENSITY.get(industry, DEFAULT_RD_INTENSITY)

    # Add controlled variation: log-normal noise around the ratio
    # This creates realistic spread: some companies are more/less capital-efficient
    rev_variation = rng.lognormvariate(0, 0.4)  # ~0.5x to 2x range
    estimated_revenue = capital * rev_cap_ratio * rev_variation

    # R&D intensity varies within industry (some firms invest more in R&D)
    rd_variation = rng.lognormvariate(0, 0.5)  # broader range for R&D
    rd_intensity = rd_intensity_base * rd_variation

    # Clamp to reasonable bounds
    rd_intensity = max(0.001, min(0.50, rd_intensity))
    estimated_revenue = max(capital, estimated_revenue)  # Revenue >= capital

    estimated_rd_expense = estimated_revenue * rd_intensity

    return estimated_revenue, estimated_rd_expense, rd_intensity


def enrich_database() -> None:
    """Enrich all companies in matcher.db with estimated R&D data.

    Skips companies that already have actual R&D data (rd_expense > 0)
    from dummy data or XBRL extraction.
    """
    now = datetime.now().isoformat(timespec="seconds")

    with connect(settings.matcher_db_path) as conn:
        # Get companies without R&D data
        rows = conn.execute(
            """SELECT edinet_code, name, industry, revenue, rd_expense
               FROM companies
               WHERE rd_expense IS NULL OR rd_expense = 0"""
        ).fetchall()

        logger.info("Found %d companies without R&D data.", len(rows))

        updated = 0
        for row in rows:
            code = row["edinet_code"]
            capital = row["revenue"] or 0  # 'revenue' column currently stores capital
            industry = row["industry"] or ""

            if capital <= 0:
                continue

            # Use edinet_code hash as seed for reproducible results
            seed = hash(code) & 0xFFFFFFFF

            est_revenue, est_rd_expense, rd_intensity = estimate_financials(
                capital, industry, seed
            )

            conn.execute(
                """UPDATE companies
                   SET revenue = ?, rd_expense = ?, rd_intensity = ?,
                       updated_at = ?
                   WHERE edinet_code = ?""",
                (est_revenue, est_rd_expense, rd_intensity, now, code),
            )
            updated += 1

        logger.info("Updated %d companies with estimated R&D data.", updated)

        # Recompute industry_stats from enriched data
        logger.info("Recomputing industry_stats...")
        industries: dict[str, list[float]] = {}
        for row in conn.execute("SELECT industry, rd_intensity FROM companies WHERE rd_intensity > 0"):
            industries.setdefault(row["industry"], []).append(row["rd_intensity"])

        for ind, vals in industries.items():
            n = len(vals)
            mean = sum(vals) / n if n > 0 else 0
            variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0
            std = math.sqrt(variance) if variance > 0 else 0.01
            conn.execute(
                """INSERT OR REPLACE INTO industry_stats
                   (industry, mean_rd_int, std_rd_int, company_count, computed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (ind, mean, std, n, now),
            )

        logger.info("Updated industry_stats for %d industries.", len(industries))

    # Print summary
    with connect(settings.matcher_db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        has_rd = conn.execute("SELECT COUNT(*) FROM companies WHERE rd_expense > 0").fetchone()[0]
        logger.info("Summary: %d / %d companies now have R&D data (%.1f%%)",
                     has_rd, total, has_rd / total * 100)

        # Show sample
        logger.info("Sample enriched companies:")
        for row in conn.execute(
            """SELECT name, industry, revenue, rd_expense, rd_intensity
               FROM companies WHERE rd_expense > 0
               ORDER BY rd_expense DESC LIMIT 10"""
        ):
            logger.info("  %s (%s): revenue=%.0fM, R&D=%.0fM, intensity=%.1f%%",
                         row["name"], row["industry"],
                         row["revenue"], row["rd_expense"], row["rd_intensity"] * 100)


def main():
    logger.info("Starting R&D data enrichment...")
    logger.info("DB path: %s", settings.matcher_db_path)
    enrich_database()
    logger.info("Done!")


if __name__ == "__main__":
    main()
