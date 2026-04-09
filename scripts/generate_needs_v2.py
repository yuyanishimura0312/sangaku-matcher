"""Generate v2 needs from midterm_plan_text using Claude API.

Reads midterm_plan_text (EDINET 経営方針セクション) and uses Claude to
extract structured technology needs per company. Falls back to v1
template-based approach for companies without midterm_plan_text.

Usage:
    python scripts/generate_needs_v2.py [--limit N] [--batch-size 10]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "matcher.db"


def _get_anthropic_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-w"],
                capture_output=True, text=True,
            )
            key = result.stdout.strip()
        except Exception:
            pass
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not found")
    return key


def _strip_html(text: str) -> str:
    """Remove HTML tags."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_needs_with_claude(client, company_name: str, industry: str, strategy_text: str) -> str | None:
    """Use Claude to extract technology needs from strategy text."""
    # Truncate to ~4000 chars to control cost
    clean_text = _strip_html(strategy_text)[:4000]

    prompt = f"""以下は{company_name}（{industry}）の有価証券報告書「経営方針、経営環境及び対処すべき課題等」からの抜粋です。

---
{clean_text}
---

この企業が産学連携で求めうる技術ニーズを推定してください。以下の形式で、200-400字程度の自然文で出力してください:

「{company_name}の推定技術ニーズ。[R&Dの方向性の要約]。[新規事業領域への言及]。求める技術領域: [具体的な技術キーワードを読点区切りで5-10個]」

経営方針テキストに言及されている具体的な技術分野、新規事業領域、DX施策、成長戦略に基づいて推定してください。テンプレート的な記述ではなく、この企業固有の内容を反映してください。"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.warning("Claude API error for %s: %s", company_name, e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    import anthropic

    api_key = _get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)
    logger.info("Claude API ready")

    # Import embedding module
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Get companies with midterm_plan_text
    sql = """SELECT edinet_code, name, industry, rd_expense, rd_intensity, revenue, midterm_plan_text
             FROM companies
             WHERE midterm_plan_text IS NOT NULL AND LENGTH(midterm_plan_text) > 100
             ORDER BY rd_expense DESC"""
    if args.limit:
        sql += f" LIMIT {args.limit}"
    companies = conn.execute(sql).fetchall()
    logger.info("Companies with strategy text: %d", len(companies))

    now = datetime.now().isoformat(timespec="seconds")
    processed = 0
    success = 0

    for batch_start in range(0, len(companies), args.batch_size):
        batch = companies[batch_start:batch_start + args.batch_size]
        needs_texts = []
        edinet_codes = []

        for co in batch:
            needs_text = _extract_needs_with_claude(
                client, co["name"], co["industry"] or "", co["midterm_plan_text"]
            )
            if needs_text and len(needs_text) > 50:
                needs_texts.append(f"passage: {needs_text}")
                edinet_codes.append((co["edinet_code"], needs_text))
                success += 1
            else:
                logger.debug("Skipping %s (no needs generated)", co["name"])

            time.sleep(0.3)  # Rate limit

        if needs_texts:
            vectors = embeddings.encode(needs_texts)
            for i, (code, text) in enumerate(edinet_codes):
                conn.execute(
                    """UPDATE companies
                       SET estimated_needs = ?, needs_vector = ?, needs_generated_at = ?
                       WHERE edinet_code = ?""",
                    (text, vectors[i].tobytes(), now, code),
                )
            conn.commit()

        processed += len(batch)
        logger.info("Progress: %d/%d processed, %d needs generated", processed, len(companies), success)

    conn.close()
    logger.info("Done. Generated v2 needs for %d companies.", success)


if __name__ == "__main__":
    main()
