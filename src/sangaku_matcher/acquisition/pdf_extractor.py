"""Extract R&D section text from 有価証券報告書 PDFs using pdfplumber."""
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


def extract_rd_section(pdf_path: Path) -> str | None:
    """Extract the R&D activity section from a 有報 PDF.

    Strategy: scan page texts for the '研究開発活動' header, then collect
    text until the next major section header.

    Returns None if the PDF cannot be read or the section is not found.
    """
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

    if not full_text:
        return None

    # Find R&D section start
    start_pos = None
    for pat in RD_START_PATTERNS:
        m = re.search(pat, full_text)
        if m:
            start_pos = m.start()
            break

    if start_pos is None:
        logger.debug("R&D section not found in %s", pdf_path)
        return None

    # Find section end
    remaining = full_text[start_pos + 10:]  # skip header itself
    end_pos = len(remaining)
    for pat in RD_END_PATTERNS:
        m = re.search(pat, remaining)
        if m:
            end_pos = min(end_pos, m.start())

    rd_text = remaining[:end_pos].strip()
    # Remove excessive whitespace
    rd_text = re.sub(r"\n{3,}", "\n\n", rd_text)

    if len(rd_text) < 50:
        logger.debug("R&D section too short (%d chars) in %s", len(rd_text), pdf_path)
        return None

    return rd_text
