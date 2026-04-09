"""Enrich collaboration data using CiNii Research API (free, no key required).

Searches for co-authored papers between companies and universities.
CiNii Research API: https://support.nii.ac.jp/ja/cir/r_opensearch

Usage:
    python scripts/enrich_collaborations.py [--limit N] [--delay SEC]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "matcher.db"

# Major Japanese universities for targeted search
MAJOR_UNIVERSITIES = [
    "東京大学", "京都大学", "大阪大学", "東北大学", "名古屋大学",
    "北海道大学", "九州大学", "東京工業大学", "筑波大学", "広島大学",
    "神戸大学", "早稲田大学", "慶應義塾大学", "千葉大学", "金沢大学",
    "岡山大学", "新潟大学", "熊本大学", "長崎大学", "信州大学",
    "東京医科歯科大学", "横浜国立大学", "名古屋工業大学", "大阪公立大学",
    "東京農工大学", "電気通信大学", "奈良先端科学技術大学院大学",
    "豊橋技術科学大学", "北陸先端科学技術大学院大学",
    "産業技術総合研究所", "理化学研究所", "物質・材料研究機構",
    "国立がん研究センター", "国立環境研究所",
]


def _search_cinii(company_name: str, university_name: str) -> int:
    """Search CiNii Research for co-authored papers between company and university.

    Returns the number of results found.
    """
    # Clean company name: remove common suffixes for better search
    short_name = re.sub(r"(株式会社|有限会社|合同会社)", "", company_name).strip()
    if len(short_name) < 2:
        return 0

    query = f'"{short_name}" "{university_name}"'
    params = urllib.parse.urlencode({
        "q": query,
        "count": 1,  # only need the total count
        "format": "json",
    })
    url = f"https://cir.nii.ac.jp/opensearch/articles?{params}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "sangaku-matcher/1.0 (research tool)",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            # CiNii returns opensearch format
            total = data.get("opensearch:totalResults") or data.get("totalResults", 0)
            return int(total)
    except Exception as e:
        logger.debug("CiNii search failed for %s x %s: %s", short_name, university_name, e)
        return 0


def _get_companies(conn: sqlite3.Connection, limit: int | None = None) -> list[dict]:
    """Get companies sorted by R&D expense (higher R&D = more likely to collaborate)."""
    sql = "SELECT edinet_code, name, industry FROM companies ORDER BY rd_expense DESC"
    if limit:
        sql += f" LIMIT {limit}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Enrich collaboration data via CiNii")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of companies to process")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls (seconds)")
    parser.add_argument("--min-results", type=int, default=1, help="Minimum co-authored papers to record")
    args = parser.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    companies = _get_companies(conn, args.limit)
    logger.info("Processing %d companies against %d universities", len(companies), len(MAJOR_UNIVERSITIES))

    # Load existing collaboration data to avoid duplicates
    existing = set()
    for row in conn.execute("SELECT edinet_code, university_name FROM collaborations"):
        existing.add((row["edinet_code"], row["university_name"]))

    new_records = 0
    now = datetime.now().isoformat(timespec="seconds")

    for i, co in enumerate(companies):
        found_any = False
        for uni in MAJOR_UNIVERSITIES:
            key = (co["edinet_code"], uni)
            if key in existing:
                continue

            count = _search_cinii(co["name"], uni)
            time.sleep(args.delay)

            if count >= args.min_results:
                conn.execute(
                    """INSERT OR REPLACE INTO collaborations
                       (edinet_code, university_name, type, count, last_year)
                       VALUES (?, ?, ?, ?, ?)""",
                    (co["edinet_code"], uni, "co_authored_paper", count, 2025),
                )
                new_records += 1
                found_any = True
                logger.info(
                    "  [%d/%d] %s x %s: %d papers",
                    i + 1, len(companies), co["name"], uni, count,
                )

        if found_any:
            conn.commit()

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d companies processed, %d new records", i + 1, len(companies), new_records)

    conn.commit()
    conn.close()
    logger.info("Done. Added %d new collaboration records.", new_records)


if __name__ == "__main__":
    main()
