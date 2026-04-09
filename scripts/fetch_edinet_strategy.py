"""Fetch management strategy text from EDINET 有報 for all companies.

Downloads CSV (type=5) from EDINET API v2, extracts the
'経営方針、経営環境及び対処すべき課題等' section via XBRL tag.

Usage:
    python scripts/fetch_edinet_strategy.py [--limit N] [--start-date 2024-04-01] [--end-date 2025-03-31]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "matcher.db"
EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"

# XBRL tag for management strategy section
STRATEGY_TAG = "jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock"
# Fallback tags
FALLBACK_TAGS = [
    "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock",
    "jpcrp_cor:ResearchAndDevelopmentActivitiesTextBlock",
]


def _get_api_key() -> str:
    key = os.environ.get("EDINET_API_KEY")
    if not key:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "EDINET_API_KEY", "-w"],
                capture_output=True, text=True,
            )
            key = result.stdout.strip()
        except Exception:
            pass
    if not key:
        raise RuntimeError("EDINET_API_KEY not found")
    return key


def _api_get(path: str, params: dict, api_key: str) -> dict | bytes:
    """Make EDINET API request."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{EDINET_BASE}/{path}?{query}&Subscription-Key={api_key}"
    req = Request(url, headers={"User-Agent": "sangaku-matcher/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "json" in content_type:
                return json.loads(data)
            return data
    except HTTPError as e:
        if e.code == 429:
            logger.warning("Rate limited, waiting 10s...")
            time.sleep(10)
            return _api_get(path, params, api_key)
        raise


def _strip_html(text: str) -> str:
    """Remove HTML tags and clean up text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _find_yuho_doc_ids(api_key: str, start_date: str, end_date: str, target_codes: set[str]) -> dict[str, str]:
    """Scan EDINET documents to find 有報 doc_ids for target companies.

    Returns: {edinet_code: doc_id}
    """
    found: dict[str, str] = {}
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            data = _api_get("documents.json", {"date": date_str, "type": "2"}, api_key)
            results = data.get("results", [])
            for doc in results:
                if doc.get("docTypeCode") != "120":
                    continue
                ec = doc.get("edinetCode")
                if ec and ec in target_codes and ec not in found:
                    found[ec] = doc["docID"]
            if results:
                logger.debug("  %s: %d docs, %d 有報 found so far", date_str, len(results), len(found))
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", date_str, e)

        time.sleep(0.5)
        current += timedelta(days=1)

        # Progress every 30 days
        if (current - datetime.strptime(start_date, "%Y-%m-%d")).days % 30 == 0:
            logger.info("Scanning %s... found %d/%d 有報", date_str, len(found), len(target_codes))

    return found


def _extract_strategy_from_csv(zip_data: bytes) -> str | None:
    """Extract strategy text from EDINET CSV zip."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            # Look for the main CSV file (jpcrp*.csv pattern)
            csv_files = [n for n in zf.namelist() if n.endswith('.csv') and 'jpcrp' in n.lower()]
            if not csv_files:
                # Try any CSV
                csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_files:
                return None

            for csv_file in csv_files:
                raw = zf.read(csv_file)
                # Try UTF-16 (EDINET default) then UTF-8
                for encoding in ('utf-16', 'utf-8', 'shift_jis'):
                    try:
                        text = raw.decode(encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    continue

                # Parse TSV/CSV
                reader = csv.reader(io.StringIO(text), delimiter='\t')
                for row in reader:
                    if len(row) < 2:
                        continue
                    # Check if any column contains the target tag
                    for i, cell in enumerate(row):
                        if STRATEGY_TAG in cell:
                            # The value is typically in the next column or a specific column
                            for j in range(i + 1, min(i + 5, len(row))):
                                if row[j] and len(row[j]) > 50:
                                    return _strip_html(row[j])
                        # Also check fallback tags
                        for tag in FALLBACK_TAGS:
                            if tag in cell:
                                for j in range(i + 1, min(i + 5, len(row))):
                                    if row[j] and len(row[j]) > 50:
                                        return _strip_html(row[j])
    except Exception as e:
        logger.debug("CSV extraction failed: %s", e)

    return None


def _extract_strategy_from_xbrl(zip_data: bytes) -> str | None:
    """Extract strategy text from EDINET XBRL zip (fallback)."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            xbrl_files = [n for n in zf.namelist()
                         if n.endswith('.htm') or n.endswith('.xbrl')]
            for xf in xbrl_files:
                content = zf.read(xf).decode('utf-8', errors='ignore')
                # Search for the strategy tag in XBRL
                pattern = rf'<{STRATEGY_TAG}[^>]*>(.*?)</{STRATEGY_TAG}>'
                m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if m:
                    return _strip_html(m.group(1))
    except Exception as e:
        logger.debug("XBRL extraction failed: %s", e)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-date", default="2024-07-01", help="Scan start (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-06-30", help="Scan end (YYYY-MM-DD)")
    args = parser.parse_args()

    api_key = _get_api_key()
    logger.info("EDINET API key loaded")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Get target companies
    sql = "SELECT edinet_code, name FROM companies ORDER BY rd_expense DESC"
    if args.limit:
        sql += f" LIMIT {args.limit}"
    companies = conn.execute(sql).fetchall()
    target_codes = {r["edinet_code"] for r in companies}
    company_names = {r["edinet_code"]: r["name"] for r in companies}
    logger.info("Target: %d companies", len(target_codes))

    # Skip companies that already have midterm_plan_text
    existing = set()
    for row in conn.execute(
        "SELECT edinet_code FROM companies WHERE midterm_plan_text IS NOT NULL AND LENGTH(midterm_plan_text) > 100"
    ):
        existing.add(row["edinet_code"])
    target_codes -= existing
    logger.info("After excluding existing: %d companies to fetch", len(target_codes))

    if not target_codes:
        logger.info("All companies already have strategy text. Done.")
        conn.close()
        return

    # Phase 1: Find doc_ids
    logger.info("Phase 1: Scanning EDINET for 有報 doc_ids (%s to %s)...", args.start_date, args.end_date)
    doc_ids = _find_yuho_doc_ids(api_key, args.start_date, args.end_date, target_codes)
    logger.info("Found %d 有報 documents", len(doc_ids))

    # Phase 2: Download and extract strategy text
    logger.info("Phase 2: Downloading CSV and extracting strategy text...")
    success = 0
    failed = 0

    for i, (edinet_code, doc_id) in enumerate(doc_ids.items()):
        name = company_names.get(edinet_code, edinet_code)
        try:
            # Try CSV first (type=5)
            csv_data = _api_get(f"documents/{doc_id}", {"type": "5"}, api_key)
            time.sleep(1.0)

            strategy_text = None
            if isinstance(csv_data, bytes) and len(csv_data) > 100:
                strategy_text = _extract_strategy_from_csv(csv_data)

            # Fallback to XBRL (type=1) if CSV failed
            if not strategy_text:
                xbrl_data = _api_get(f"documents/{doc_id}", {"type": "1"}, api_key)
                time.sleep(1.0)
                if isinstance(xbrl_data, bytes) and len(xbrl_data) > 100:
                    strategy_text = _extract_strategy_from_xbrl(xbrl_data)

            if strategy_text and len(strategy_text) > 100:
                conn.execute(
                    "UPDATE companies SET midterm_plan_text = ? WHERE edinet_code = ?",
                    (strategy_text[:20000], edinet_code),
                )
                conn.commit()
                success += 1
                logger.info("[%d/%d] %s: %d chars extracted", i + 1, len(doc_ids), name, len(strategy_text))
            else:
                failed += 1
                logger.debug("[%d/%d] %s: no strategy text found", i + 1, len(doc_ids), name)

        except Exception as e:
            failed += 1
            logger.warning("[%d/%d] %s: error - %s", i + 1, len(doc_ids), name, e)
            time.sleep(2)

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d processed, %d success, %d failed", i + 1, len(doc_ids), success, failed)

    conn.close()
    logger.info("Done. Success: %d, Failed: %d, Total: %d", success, failed, len(doc_ids))


if __name__ == "__main__":
    main()
