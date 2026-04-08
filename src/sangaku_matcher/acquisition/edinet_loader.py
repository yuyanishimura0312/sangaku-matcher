"""Load company data from IR Collector's SQLite database."""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from sangaku_matcher.config import settings

logger = logging.getLogger(__name__)


def _ir_db_path() -> Path:
    return Path(os.environ.get("IR_COLLECTOR_DB_PATH", settings.ir_collector_db_path))


def ir_collector_available() -> bool:
    """Check if IR Collector DB exists and has data."""
    db = _ir_db_path()
    if not db.exists():
        return False
    try:
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT COUNT(*) FROM documents WHERE doc_type_code = '120'").fetchone()
        conn.close()
        return row[0] > 0
    except Exception:
        return False


def load_filers() -> list[dict]:
    """Get unique companies that have filed 有価証券報告書.

    Returns list of dicts with keys: edinet_code, sec_code, filer_name
    """
    db = _ir_db_path()
    if not db.exists():
        logger.warning("IR Collector DB not found at %s", db)
        return []

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT DISTINCT edinet_code, sec_code, filer_name
        FROM documents
        WHERE doc_type_code = '120'
          AND edinet_code IS NOT NULL
          AND edinet_code != ''
        ORDER BY filer_name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_pdf_path(edinet_code: str) -> Path | None:
    """Get the file path to the most recent 有報 PDF for a company."""
    db = _ir_db_path()
    if not db.exists():
        return None

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT pdf_path FROM documents
        WHERE edinet_code = ?
          AND doc_type_code = '120'
          AND pdf_path IS NOT NULL
        ORDER BY submit_datetime DESC
        LIMIT 1
    """, (edinet_code,)).fetchone()
    conn.close()

    if row and row["pdf_path"]:
        base = Path(os.environ.get("IR_COLLECTOR_STORAGE_PATH", settings.ir_collector_storage_path))
        return base / row["pdf_path"]
    return None
