"""Re-fetch 有報 text with parallel downloads for speed.

Uses concurrent.futures to download multiple documents simultaneously.
Respects EDINET rate limits with a semaphore-based throttle.

Usage:
    python scripts/refetch_parallel.py [--workers 5] [--start-date 2024-04-01] [--end-date 2026-04-01]
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
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock, Semaphore
from urllib.request import Request, urlopen
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"
IR_DB = Path.home() / "projects/apps/ir-collector/data/ir.db"
EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"

STRATEGY_TAG = "jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock"
STRATEGY_NAME = "経営方針、経営環境及び対処すべき課題等"
FALLBACK_TAGS = [
    "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock",
    "jpcrp_cor:ResearchAndDevelopmentActivitiesTextBlock",
]

# Rate limiter: max N requests per second
_rate_semaphore: Semaphore | None = None
_rate_lock = Lock()
_last_request_time = 0.0
MIN_INTERVAL = 0.3  # 300ms between requests


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


def _throttled_get(path: str, params: dict, api_key: str) -> dict | bytes:
    """Rate-limited EDINET API request."""
    global _last_request_time
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        _last_request_time = time.time()

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{EDINET_BASE}/{path}?{query}&Subscription-Key={api_key}"
    req = Request(url, headers={"User-Agent": "sangaku-matcher/1.0"})
    try:
        with urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "json" in content_type:
                return json.loads(data)
            return data
    except HTTPError as e:
        if e.code == 429:
            logger.warning("Rate limited, waiting 10s...")
            time.sleep(10)
            return _throttled_get(path, params, api_key)
        raise


def _strip_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_from_csv(zip_data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith('.csv') and 'jpcrp' in n.lower()]
            if not csv_files:
                csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
            for csv_file in csv_files:
                raw = zf.read(csv_file)
                for enc in ('utf-16', 'utf-8', 'shift_jis'):
                    try:
                        text = raw.decode(enc)
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
                        if STRATEGY_TAG in cell:
                            for j in range(i + 1, min(i + 5, len(row))):
                                if row[j] and len(row[j]) > 50:
                                    return _strip_html(row[j])
                        for tag in FALLBACK_TAGS:
                            if tag in cell:
                                for j in range(i + 1, min(i + 5, len(row))):
                                    if row[j] and len(row[j]) > 50:
                                        return _strip_html(row[j])
    except Exception:
        pass
    return None


def _extract_from_xbrl(zip_data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            for xf in [n for n in zf.namelist() if n.endswith('.htm') or n.endswith('.xbrl')]:
                content = zf.read(xf).decode('utf-8', errors='ignore')
                pattern = rf'<{STRATEGY_TAG}[^>]*>(.*?)</{STRATEGY_TAG}>'
                m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if m:
                    return _strip_html(m.group(1))
    except Exception:
        pass
    return None


def _fetch_one(edinet_code: str, doc_id: str, period_end: str | None,
               name: str, api_key: str) -> dict | None:
    """Fetch and extract strategy text for one company. Returns result dict or None."""
    try:
        csv_data = _throttled_get(f"documents/{doc_id}", {"type": "5"}, api_key)
        strategy_text = None
        if isinstance(csv_data, bytes) and len(csv_data) > 100:
            strategy_text = _extract_from_csv(csv_data)

        if not strategy_text:
            xbrl_data = _throttled_get(f"documents/{doc_id}", {"type": "1"}, api_key)
            if isinstance(xbrl_data, bytes) and len(xbrl_data) > 100:
                strategy_text = _extract_from_xbrl(xbrl_data)

        return {
            "edinet_code": edinet_code,
            "doc_id": doc_id,
            "period_end": period_end,
            "name": name,
            "text": strategy_text,
        }
    except Exception as e:
        logger.warning("%s: error - %s", name, e)
        return {
            "edinet_code": edinet_code,
            "doc_id": doc_id,
            "period_end": period_end,
            "name": name,
            "text": None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5, help="Parallel download workers")
    parser.add_argument("--start-date", default="2024-04-01")
    parser.add_argument("--end-date", default="2026-04-01")
    args = parser.parse_args()

    api_key = _get_api_key()

    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row

    targets = conn.execute(
        """SELECT edinet_code, name FROM companies
           WHERE midterm_plan_text IS NOT NULL AND LENGTH(midterm_plan_text) > 100
             AND (source_doc_id IS NULL
                  OR yuho_period_end IS NULL
                  OR LENGTH(midterm_plan_text) >= 19999)"""
    ).fetchall()
    target_codes = {r["edinet_code"] for r in targets}
    company_names = {r["edinet_code"]: r["name"] for r in targets}
    logger.info("Target: %d companies", len(target_codes))

    if not target_codes:
        logger.info("Nothing to refetch.")
        conn.close()
        return

    # Phase 1: Find doc_ids (sequential, day-by-day scan)
    logger.info("Phase 1: Scanning EDINET for doc_ids (%s to %s)...", args.start_date, args.end_date)
    found: dict[str, dict] = {}
    current = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            data = _throttled_get("documents.json", {"date": date_str, "type": "2"}, api_key)
            for doc in data.get("results", []):
                if doc.get("docTypeCode") != "120":
                    continue
                ec = doc.get("edinetCode")
                if ec and ec in target_codes and ec not in found:
                    found[ec] = {"doc_id": doc["docID"], "period_end": doc.get("periodEnd")}
        except Exception as e:
            logger.warning("Failed %s: %s", date_str, e)
        current += timedelta(days=1)
        if (current - datetime.strptime(args.start_date, "%Y-%m-%d")).days % 30 == 0:
            logger.info("Scanning %s... found %d/%d", date_str, len(found), len(target_codes))

    logger.info("Found %d doc_ids. Starting parallel download with %d workers...", len(found), args.workers)

    # Phase 2: Parallel download
    ir_conn = None
    if IR_DB.exists():
        ir_conn = sqlite3.connect(str(IR_DB))

    now = datetime.now().isoformat(timespec="seconds")
    success = 0
    failed = 0
    db_lock = Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _fetch_one, ec, info["doc_id"], info.get("period_end"),
                company_names.get(ec, ec), api_key
            ): ec
            for ec, info in found.items()
        }

        for future in as_completed(futures):
            result = future.result()
            if not result:
                failed += 1
                continue

            with db_lock:
                if result["text"] and len(result["text"]) > 100:
                    conn.execute(
                        """UPDATE companies
                           SET midterm_plan_text = ?, source_doc_id = ?, yuho_period_end = ?
                           WHERE edinet_code = ?""",
                        (result["text"], result["doc_id"], result["period_end"], result["edinet_code"]),
                    )

                    if ir_conn:
                        ir_conn.execute(
                            """INSERT INTO sections
                               (doc_id, edinet_code, filer_name, section_tag, section_name,
                                text_content, char_count, period_end, extracted_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT(edinet_code, section_tag, period_end) DO UPDATE SET
                                doc_id = excluded.doc_id,
                                text_content = excluded.text_content,
                                char_count = excluded.char_count,
                                extracted_at = excluded.extracted_at""",
                            (result["doc_id"], result["edinet_code"], result["name"],
                             STRATEGY_TAG, STRATEGY_NAME,
                             result["text"], len(result["text"]),
                             result["period_end"], now),
                        )

                    success += 1
                else:
                    # Save doc_id/period_end even without text
                    conn.execute(
                        """UPDATE companies SET source_doc_id = ?, yuho_period_end = ?
                           WHERE edinet_code = ?""",
                        (result["doc_id"], result["period_end"], result["edinet_code"]),
                    )
                    failed += 1

                # Batch commit every 50
                if (success + failed) % 50 == 0:
                    conn.commit()
                    if ir_conn:
                        ir_conn.commit()
                    logger.info("Progress: %d/%d, success=%d, failed=%d",
                               success + failed, len(found), success, failed)

    conn.commit()
    if ir_conn:
        ir_conn.commit()
        ir_conn.close()
    conn.close()
    logger.info("Done. Success: %d, Failed: %d, Total: %d", success, failed, len(found))


if __name__ == "__main__":
    main()
