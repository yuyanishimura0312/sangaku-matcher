"""Fetch R&D activities text from EDINET 有報 for all companies.

Downloads CSV (type=5) from EDINET API v2, extracts the
'研究開発活動' section via XBRL tag, and stores in both
matcher.db (companies.rd_text) and ir-collector sections table.

Supports parallel downloads with rate limiting.

Usage:
    python scripts/fetch_rd_section.py --skip-scan [--workers 3] [--limit N]
    python scripts/fetch_rd_section.py --skip-scan --batch 500  # process in batches of 500
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Semaphore
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/fetch_rd_section.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"
IR_COLLECTOR_DB = Path.home() / "projects/apps/ir-collector/data/ir.db"

EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"

# XBRL tag for R&D activities section
RD_TAG = "jpcrp_cor:ResearchAndDevelopmentActivitiesTextBlock"
RD_NAME = "研究開発活動"

# Rate limiter: max N requests per second across all threads
_rate_semaphore: Semaphore | None = None


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


def _api_get(path: str, params: dict, api_key: str, retries: int = 3) -> dict | bytes:
    """Make EDINET API request with retry and rate limiting."""
    if _rate_semaphore:
        _rate_semaphore.acquire()
        # Release after delay to enforce rate limit
        import threading
        threading.Timer(1.0, _rate_semaphore.release).start()

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{EDINET_BASE}/{path}?{query}&Subscription-Key={api_key}"
    req = Request(url, headers={"User-Agent": "sangaku-matcher/1.0"})

    for attempt in range(retries):
        try:
            with urlopen(req, timeout=120) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()
                if "json" in content_type:
                    return json.loads(data)
                return data
        except HTTPError as e:
            if e.code == 429:
                wait = 30 * (attempt + 1)
                logger.warning("Rate limited (429), waiting %ds...", wait)
                time.sleep(wait)
                continue
            raise
        except (URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                logger.warning("Network error (attempt %d/%d): %s. Retrying in %ds...",
                             attempt + 1, retries, e, wait)
                time.sleep(wait)
                continue
            raise

    raise RuntimeError(f"Failed after {retries} retries: {path}")


def _strip_html(text: str) -> str:
    """Remove HTML tags and clean up text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _get_cached_doc_ids(conn: sqlite3.Connection) -> dict[str, dict]:
    """Reuse doc_ids already stored in companies table."""
    rows = conn.execute(
        "SELECT edinet_code, name, source_doc_id, yuho_period_end FROM companies "
        "WHERE source_doc_id IS NOT NULL"
    ).fetchall()
    return {
        r["edinet_code"]: {
            "doc_id": r["source_doc_id"],
            "period_end": r["yuho_period_end"],
            "filer_name": r["name"],
        }
        for r in rows
    }


def _extract_rd_from_csv(zip_data: bytes) -> str | None:
    """Extract R&D activities text from EDINET CSV zip."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith('.csv') and 'jpcrp' in n.lower()]
            if not csv_files:
                csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_files:
                return None

            for csv_file in csv_files:
                raw = zf.read(csv_file)
                for encoding in ('utf-16', 'utf-8', 'shift_jis'):
                    try:
                        text = raw.decode(encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                else:
                    continue

                reader = csv.reader(io.StringIO(text), delimiter='\t')
                for row in reader:
                    if len(row) < 2:
                        continue
                    for i, cell in enumerate(row):
                        if RD_TAG in cell:
                            for j in range(i + 1, min(i + 5, len(row))):
                                if row[j] and len(row[j]) > 50:
                                    return _strip_html(row[j])
    except Exception as e:
        logger.debug("CSV extraction failed: %s", e)
    return None


def _extract_rd_from_xbrl(zip_data: bytes) -> str | None:
    """Extract R&D activities text from EDINET XBRL zip (fallback)."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            xbrl_files = [n for n in zf.namelist()
                         if n.endswith('.htm') or n.endswith('.xbrl')]
            for xf in xbrl_files:
                content = zf.read(xf).decode('utf-8', errors='ignore')
                pattern = rf'<{RD_TAG}[^>]*>(.*?)</{RD_TAG}>'
                m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if m:
                    return _strip_html(m.group(1))
    except Exception as e:
        logger.debug("XBRL extraction failed: %s", e)
    return None


def _fetch_one(edinet_code: str, doc_info: dict, api_key: str) -> dict:
    """Fetch R&D text for a single company. Thread-safe."""
    doc_id = doc_info["doc_id"]
    filer_name = doc_info.get("filer_name", edinet_code)
    period_end = doc_info.get("period_end")

    try:
        # Try CSV first (type=5)
        csv_data = _api_get(f"documents/{doc_id}", {"type": "5"}, api_key)
        rd_text = None
        if isinstance(csv_data, bytes) and len(csv_data) > 100:
            rd_text = _extract_rd_from_csv(csv_data)

        # Fallback to XBRL (type=1)
        if not rd_text:
            xbrl_data = _api_get(f"documents/{doc_id}", {"type": "1"}, api_key)
            if isinstance(xbrl_data, bytes) and len(xbrl_data) > 100:
                rd_text = _extract_rd_from_xbrl(xbrl_data)

        return {
            "edinet_code": edinet_code,
            "filer_name": filer_name,
            "doc_id": doc_id,
            "period_end": period_end,
            "rd_text": rd_text,
            "error": None,
        }
    except Exception as e:
        return {
            "edinet_code": edinet_code,
            "filer_name": filer_name,
            "doc_id": doc_id,
            "period_end": period_end,
            "rd_text": None,
            "error": str(e),
        }


def _save_to_dbs(result: dict, matcher_conn: sqlite3.Connection) -> str:
    """Save a successful result to both DBs. Returns status string."""
    rd_text = result["rd_text"]
    edinet_code = result["edinet_code"]

    if not rd_text or len(rd_text) <= 50:
        return "no_rd"

    # Save to matcher.db
    matcher_conn.execute(
        "UPDATE companies SET rd_text = ? WHERE edinet_code = ?",
        (rd_text, edinet_code),
    )
    matcher_conn.commit()

    # Save to ir-collector
    if IR_COLLECTOR_DB.exists():
        try:
            ir_conn = sqlite3.connect(str(IR_COLLECTOR_DB))
            ir_conn.execute(
                """
                INSERT INTO sections (
                    doc_id, edinet_code, filer_name, section_tag, section_name,
                    text_content, char_count, period_end, extracted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edinet_code, section_tag, period_end) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    filer_name = excluded.filer_name,
                    text_content = excluded.text_content,
                    char_count = excluded.char_count,
                    extracted_at = excluded.extracted_at
                """,
                (result["doc_id"], edinet_code, result["filer_name"],
                 RD_TAG, RD_NAME, rd_text, len(rd_text), result["period_end"],
                 datetime.now().isoformat(timespec="seconds")),
            )
            ir_conn.commit()
            ir_conn.close()
        except Exception as e:
            logger.warning("IR Collector save failed for %s: %s", edinet_code, e)

    return "success"


def main():
    global _rate_semaphore

    parser = argparse.ArgumentParser(description="Fetch R&D activities section from EDINET 有報")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of companies")
    parser.add_argument("--workers", type=int, default=3, help="Parallel download threads (default: 3)")
    parser.add_argument("--batch", type=int, default=None, help="Process in batches of N (for resumability)")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Skip doc_id scan, reuse cached source_doc_id from companies table")
    parser.add_argument("--start-date", default="2024-04-01")
    parser.add_argument("--end-date", default="2025-06-30")
    args = parser.parse_args()

    # Rate limiter: allow N concurrent API requests (each waits 1s after acquiring)
    _rate_semaphore = Semaphore(args.workers)

    api_key = _get_api_key()
    logger.info("EDINET API key loaded (workers=%d)", args.workers)

    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row

    # Get target companies
    sql = "SELECT edinet_code, name FROM companies WHERE rd_text IS NULL OR LENGTH(rd_text) < 100"
    if args.limit:
        sql += f" LIMIT {args.limit}"
    companies = conn.execute(sql).fetchall()
    target_codes = {r["edinet_code"] for r in companies}
    company_names = {r["edinet_code"]: r["name"] for r in companies}
    logger.info("Target: %d companies without R&D text", len(target_codes))

    if not target_codes:
        logger.info("All companies already have R&D text. Done.")
        conn.close()
        return

    # Get doc_ids
    if args.skip_scan:
        logger.info("Using cached doc_ids...")
        all_doc_ids = _get_cached_doc_ids(conn)
        doc_ids = {k: v for k, v in all_doc_ids.items() if k in target_codes}
    else:
        logger.info("Scanning EDINET for doc_ids (%s to %s)...", args.start_date, args.end_date)
        from scripts.fetch_edinet_strategy import _find_yuho_doc_ids
        doc_ids = _find_yuho_doc_ids(api_key, args.start_date, args.end_date, target_codes)

    logger.info("Found %d doc_ids for target companies", len(doc_ids))

    # Optional batching
    items = list(doc_ids.items())
    if args.batch:
        items = items[:args.batch]
        logger.info("Batch mode: processing first %d companies", len(items))

    # Phase 2: Parallel download and extraction
    logger.info("Downloading R&D text with %d workers...", args.workers)
    success = 0
    failed = 0
    no_rd = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_fetch_one, ec, info, api_key): (ec, info)
            for ec, info in items
        }

        for future in as_completed(futures):
            processed += 1
            result = future.result()

            if result["error"]:
                failed += 1
                logger.warning("[%d/%d] %s: error - %s",
                             processed, len(items), result["filer_name"], result["error"])
            else:
                status = _save_to_dbs(result, conn)
                if status == "success":
                    success += 1
                    logger.info("[%d/%d] %s: %d chars",
                              processed, len(items), result["filer_name"],
                              len(result["rd_text"]))
                else:
                    no_rd += 1

            if processed % 100 == 0:
                logger.info("=== Progress: %d/%d | success=%d no_rd=%d failed=%d ===",
                           processed, len(items), success, no_rd, failed)

    conn.close()
    logger.info("=== DONE. success=%d no_rd=%d failed=%d total=%d ===",
                success, no_rd, failed, len(items))


if __name__ == "__main__":
    main()
