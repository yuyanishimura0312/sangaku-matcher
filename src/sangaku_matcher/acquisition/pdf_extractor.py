"""Extract sections from 有価証券報告書 PDFs using pdfplumber.

Supports:
- R&D activity section (研究開発活動)
- Management strategy / new business sections (経営方針、対処すべき課題)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Section headers that typically bracket the R&D section in 有報
RD_START_PATTERNS = [
    r"研究開発活動",
    r"研\s*究\s*開\s*発\s*活\s*動",
]
RD_END_PATTERNS = [
    r"設備の状況",
    r"設\s*備\s*の\s*状\s*況",
    r"経理の状況",
    r"提出会社の状況",
]

# Patterns for management strategy / mid-term plan / new business areas
STRATEGY_START_PATTERNS = [
    r"経営方針、経営環境及び対処すべき課題等",
    r"経\s*営\s*方\s*針.*?対\s*処\s*す\s*べ\s*き\s*課\s*題",
    r"対処すべき課題",
    r"経営上の重要な契約等",
    r"中\s*期\s*経\s*営\s*計\s*画",
    r"経営方針",
]
STRATEGY_END_PATTERNS = [
    r"事業等のリスク",
    r"事\s*業\s*等\s*の\s*リ\s*ス\s*ク",
    r"経営上の重要な契約等",
    r"研究開発活動",
    r"コーポレート・ガバナンスの状況",
]


def _read_pdf_text(pdf_path: Path) -> str | None:
    """Read all text from a PDF file."""
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", pdf_path, e)
        return None

    return full_text if full_text else None


def _extract_section(
    full_text: str,
    start_patterns: list[str],
    end_patterns: list[str],
    min_length: int = 50,
    max_length: int = 30000,
    label: str = "section",
) -> str | None:
    """Generic section extractor using regex patterns."""
    start_pos = None
    for pat in start_patterns:
        m = re.search(pat, full_text)
        if m:
            start_pos = m.start()
            break

    if start_pos is None:
        return None

    # Skip header text (first 20 chars after match)
    remaining = full_text[start_pos + 20:]
    end_pos = min(len(remaining), max_length)
    for pat in end_patterns:
        m = re.search(pat, remaining)
        if m and m.start() > min_length:
            end_pos = min(end_pos, m.start())

    text = remaining[:end_pos].strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) < min_length:
        return None

    return text


def extract_rd_section(pdf_path: Path) -> str | None:
    """Extract the R&D activity section from a 有報 PDF.

    Strategy: scan page texts for the '研究開発活動' header, then collect
    text until the next major section header.

    Returns None if the PDF cannot be read or the section is not found.
    """
    full_text = _read_pdf_text(pdf_path)
    if not full_text:
        return None

    result = _extract_section(
        full_text, RD_START_PATTERNS, RD_END_PATTERNS,
        min_length=50, label="R&D",
    )
    if result is None:
        logger.debug("R&D section not found in %s", pdf_path)
    return result


def extract_strategy_section(pdf_path: Path) -> str | None:
    """Extract management strategy / mid-term plan / new business area text.

    Targets the '経営方針、経営環境及び対処すべき課題等' section in 有報,
    which typically contains mid-term plan references and new business area
    exploration descriptions.
    """
    full_text = _read_pdf_text(pdf_path)
    if not full_text:
        return None

    result = _extract_section(
        full_text, STRATEGY_START_PATTERNS, STRATEGY_END_PATTERNS,
        min_length=100, max_length=20000, label="strategy",
    )
    if result is None:
        logger.debug("Strategy section not found in %s", pdf_path)
    return result
