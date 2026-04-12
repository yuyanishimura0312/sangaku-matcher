"""Fetch R&D activities text from EDINET 有報 for all companies.

Downloads CSV (type=5) from EDINET API v2, extracts the
'研究開発活動' section via XBRL tag, and stores in both
matcher.db (companies.rd_text) and ir-collector sections table.

Usage:
    python scripts/fetch_rd_section.py [--limit N] [--start-date 2024-04-01] [--end-date 2025-03-31]
    python scripts/fetch_rd_section.py --skip-scan  # reuse cached doc_ids from matcher.db
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

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"
IR_COLLECTOR_DB = Path.home() / "projects/apps/ir-collector/data/ir.db"

EDINET_BASE = "https://api.edinet-fsa.go.jp/api/v2"

# XBRL tag for R&D activities section
RD_TAG = "jpcrp_cor:ResearchAndDevelopmentActivitiesTextBlock"
RD_NAME = "研究開発活動"


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
    """Make EDINET API request with rate limit handling."""
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


def _find_yuho_doc_ids(api_key: str, start_date: str, end_date: str, target_codes: set[str]) -> dict[str, dict]:
    """Scan EDINET documents to find 有報 doc_ids for target companies."""
    found: dict[str, dict] = {}
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
                    found[ec] = {
                        "doc_id": doc["docID"],
                        "period_end": doc.get("periodEnd"),
                        "filer_name": doc.get("filerName"),
                    }
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", date_str, e)

        time.sleep(0.5)
        current += timedelta(days=1)

        if (current - datetime.strptime(start_date, "%Y-%m-%d")).days % 30 == 0:
            logger.info("Scanning %s... found %d/%d 有報", date_str, len(found), len(target_codes))

    return found


def _get_cached_doc_ids(conn: sqlite3.Connection) -> dict[str, dict]:
    """Reuse doc_ids already stored in companies table (from prior strategy fetch)."""
    rows = conn.execute(
        "SELECT edinet_code, name, source_doc_id, yuho_period_end FROM companies "
        "WHERE source_doc_id IS NOT NULL"
    ).fetchall()
    result = {}
    for r in rows:
        result[r["edinet_code"]] = {
            "doc_id": r["source_doc_id"],
            "period_end": r["yuho_period_end"],
            "filer_name": r["name"],
        }
    return result


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


def _save_to_ir_collector(edinet_code: str, filer_name: str, doc_id: str,
                          period_end: str | None, rd_text: str) -> None:
    """Save R&D section to ir-collector's sections table."""
    if not IR_COLLECTOR_DB.exists():
        logger.debug("IR Collector DB not found, skipping")
        return
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
            (doc_id, edinet_code, filer_name, RD_TAG, RD_NAME,
             rd_text, len(rd_text), period_end,
             datetime.now().isoformat(timespec="seconds")),
        )
        ir_conn.commit()
        ir_conn.close()
    except Exception as e:
        logger.warning("Failed to save to IR Collector: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Fetch R&D activities section from EDINET 有報")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of companies")
    parser.add_argument("--start-date", default="2024-04-01", help="Scan start (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-06-30", help="Scan end (YYYY-MM-DD)")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Skip doc_id scan, reuse cached source_doc_id from companies table")
    args = parser.parse_args()

    api_key = _get_api_key()
    logger.info("EDINET API key loaded")

    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row

    # Get target companies (those without rd_text or with very short rd_text)
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

    # Phase 1: Get doc_ids (scan or cached)
    if args.skip_scan:
        logger.info("Phase 1: Using cached doc_ids from companies table...")
        all_doc_ids = _get_cached_doc_ids(conn)
        # Filter to only target companies
        doc_ids = {k: v for k, v in all_doc_ids.items() if k in target_codes}
        logger.info("Found %d cached doc_ids for target companies", len(doc_ids))
    else:
        logger.info("Phase 1: Scanning EDINET for 有報 doc_ids (%s to %s)...", args.start_date, args.end_date)
        doc_ids = _find_yuho_doc_ids(api_key, args.start_date, args.end_date, target_codes)
        logger.info("Found %d 有報 documents", len(doc_ids))

    # Phase 2: Download and extract R&D text
    logger.info("Phase 2: Downloading and extracting R&D activities text...")
    success = 0
    failed = 0
    no_rd_section = 0

    for i, (edinet_code, doc_info) in enumerate(doc_ids.items()):
        doc_id = doc_info["doc_id"]
        period_end = doc_info.get("period_end")
        filer_name = doc_info.get("filer_name") or company_names.get(edinet_code, edinet_code)
        try:
            # Try CSV first (type=5)
            csv_data = _api_get(f"documents/{doc_id}", {"type": "5"}, api_key)
            time.sleep(1.0)

            rd_text = None
            if isinstance(csv_data, bytes) and len(csv_data) > 100:
                rd_text = _extract_rd_from_csv(csv_data)

            # Fallback to XBRL (type=1) if CSV failed
            if not rd_text:
                xbrl_data = _api_get(f"documents/{doc_id}", {"type": "1"}, api_key)
                time.sleep(1.0)
                if isinstance(xbrl_data, bytes) and len(xbrl_data) > 100:
                    rd_text = _extract_rd_from_xbrl(xbrl_data)

            if rd_text and len(rd_text) > 50:
                # Save to matcher.db
                conn.execute(
                    "UPDATE companies SET rd_text = ? WHERE edinet_code = ?",
                    (rd_text, edinet_code),
                )
                conn.commit()

                # Save to ir-collector
                _save_to_ir_collector(edinet_code, filer_name, doc_id, period_end, rd_text)

                success += 1
                logger.info("[%d/%d] %s: %d chars (doc=%s)",
                           i + 1, len(doc_ids), filer_name, len(rd_text), doc_id)
            else:
                # Not all companies have an R&D section (e.g., financial firms)
                no_rd_section += 1
                logger.debug("[%d/%d] %s: no R&D section found", i + 1, len(doc_ids), filer_name)

        except Exception as e:
            failed += 1
            logger.warning("[%d/%d] %s: error - %s", i + 1, len(doc_ids), filer_name, e)
            time.sleep(2)

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d processed, %d success, %d no-rd, %d failed",
                        i + 1, len(doc_ids), success, no_rd_section, failed)

    conn.close()
    logger.info("Done. Success: %d, No R&D section: %d, Failed: %d, Total: %d",
                success, no_rd_section, failed, len(doc_ids))


if __name__ == "__main__":
    main()
