"""Extract humanities/social science collaboration needs from 有報 text.

Generates contextual narrative descriptions (not keywords) for each
detected need type, preserving the full business context. These narratives
are then vectorized for semantic matching.

Usage:
    python scripts/extract_humanities_needs.py [--limit N] [--batch-size 20] [--test]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"
IR_DB = Path.home() / "projects/apps/ir-collector/data/ir.db"

# 7 types of humanities/social science collaboration needs
NEED_TYPES = {
    "social_insight": "社会洞察連携",
    "futures_intelligence": "未来洞察連携",
    "ethics_governance": "倫理・ガバナンス連携",
    "design_anthropology": "デザイン人類学連携",
    "organizational_culture": "組織・文化変革連携",
    "community_design": "コミュニティ・場の設計連携",
    "narrative_communication": "ナラティブ・コミュニケーション連携",
}

SYSTEM_PROMPT = """あなたは産学連携の専門アナリストです。企業の有価証券報告書のテキストを深く読み解き、その企業が人文社会科学系の研究者と連携することで価値を生み出せる領域を、文脈を保持した文章として記述してください。

重要な原則：
- キーワードの羅列ではなく、企業の具体的な経営課題・戦略の文脈の中でニーズを記述してください
- 「なぜその企業がその知見を必要としているのか」という背景を含めてください
- 定型的なCSR/ESG表現（「社会課題に取り組む」等）は、具体的な事業課題と結びついている場合のみ取り上げてください
- 免責定型文（「将来に関する事項は〜が判断した」）は無視してください
- 検出されない類型については無理に書かないでください"""

USER_PROMPT = """以下は{company_name}（{industry}）の有価証券報告書「経営方針、経営環境及び対処すべき課題等」からの抜粋です。

---
{ir_text}
---

この企業が人文社会科学系の研究者と連携することで価値を生み出せる領域を分析し、以下の7類型について検出結果を文章で記述してください。

1. 社会洞察連携 — 生活者の価値観・文化・消費行動の変化を理解するための連携
2. 未来洞察連携 — 長期的な社会変化・シナリオプランニングのための連携
3. 倫理・ガバナンス連携 — 企業倫理・AI倫理・人権・ステークホルダー対話のための連携
4. デザイン人類学連携 — 人間中心設計・フィールドリサーチ・文化的デザインのための連携
5. 組織・文化変革連携 — 組織変革・ダイバーシティ・人材文化の研究との連携
6. コミュニティ・場の設計連携 — 地域共創・コミュニティ形成・場づくりのための連携
7. ナラティブ・コミュニケーション連携 — 企業の存在意義・ブランドストーリーの言語化のための連携

以下のJSON形式で出力してください。各ニーズは「この企業は〜という経営課題に直面しており、〜の知見を持つ研究者との連携により〜が期待できる」という形式の、文脈を保持した100-200字程度の文章で記述してください。確信度が低い類型（定型的記述のみの場合）は含めないでください。

```json
{{
  "needs": [
    {{
      "type": "類型の英語キー（social_insight等）",
      "confidence": "high|medium",
      "narrative": "文脈を保持したニーズの記述文（100-200字）"
    }}
  ],
  "summary": "この企業の人文系連携ポテンシャルの全体像を200-300字で記述"
}}
```"""


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
    import re
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_needs(client, company_name: str, industry: str, ir_text: str) -> dict | None:
    """Extract humanities needs as contextual narratives using Claude API."""
    clean = _strip_html(ir_text)[:6000]  # Allow more context than tech needs

    prompt = USER_PROMPT.format(
        company_name=company_name,
        industry=industry or "不明",
        ir_text=clean,
    )

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()

        # Extract JSON from response, cleaning common LLM artifacts
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            raw_json = json_match.group()
            # Remove // comments that LLMs sometimes add
            raw_json = re.sub(r'//[^\n]*', '', raw_json)
            # Remove trailing commas before } or ]
            raw_json = re.sub(r',\s*([}\]])', r'\1', raw_json)
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                # Fallback: fix unescaped control characters in strings
                raw_json = re.sub(r'[\x00-\x1f]', ' ', raw_json)
                # Try again with cleaned text
                try:
                    return json.loads(raw_json)
                except json.JSONDecodeError:
                    pass

        # Last resort: retry with simpler prompt asking for strict JSON
        logger.debug("Retrying %s with strict JSON prompt...", company_name)
        try:
            retry_msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1500,
                system="Output only valid JSON. No markdown fences, no comments, no trailing commas.",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "The JSON was malformed. Please output only the corrected valid JSON object, starting with { and ending with }. No markdown code fences."},
                ],
            )
            retry_text = retry_msg.content[0].text.strip()
            retry_match = re.search(r'\{[\s\S]*\}', retry_text)
            if retry_match:
                cleaned = re.sub(r'//[^\n]*', '', retry_match.group())
                cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
                cleaned = re.sub(r'[\x00-\x1f]', ' ', cleaned)
                return json.loads(cleaned)
        except Exception:
            pass
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error for %s: %s", company_name, e)
    except Exception as e:
        logger.warning("Claude API error for %s: %s", company_name, e)
    return None


def _build_composite_text(company_name: str, result: dict) -> str:
    """Build a single composite narrative from extracted needs for vectorization.

    This preserves the full context in a single text that can be
    meaningfully compared via cosine similarity with research descriptions.
    """
    parts = [f"{company_name}の人文社会科学系連携ニーズ。"]

    for need in result.get("needs", []):
        type_key = need.get("type", "")
        type_name = NEED_TYPES.get(type_key, type_key)
        narrative = need.get("narrative", "")
        if narrative:
            parts.append(f"【{type_name}】{narrative}")

    summary = result.get("summary", "")
    if summary:
        parts.append(f"【総合評価】{summary}")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--test", action="store_true", help="Test mode: 10 companies only")
    args = parser.parse_args()

    if args.test:
        args.limit = 10

    import anthropic
    api_key = _get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings

    # Ensure humanities columns exist
    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row
    for col_sql in [
        "ALTER TABLE companies ADD COLUMN humanities_needs_json TEXT",
        "ALTER TABLE companies ADD COLUMN humanities_needs_text TEXT",
        "ALTER TABLE companies ADD COLUMN humanities_needs_vector BLOB",
        "ALTER TABLE companies ADD COLUMN humanities_needs_at TEXT",
    ]:
        try:
            conn.execute(col_sql)
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()

    # Get companies from ir-collector (has the full strategy text)
    ir_conn = sqlite3.connect(str(IR_DB))
    ir_conn.row_factory = sqlite3.Row

    # Join with matcher DB for industry info, skip already processed
    sql = """
        SELECT s.edinet_code, s.filer_name, s.text_content, c.industry
        FROM sections s
        LEFT JOIN (SELECT edinet_code, industry, humanities_needs_at
                   FROM companies) c USING(edinet_code)
        WHERE s.section_tag = 'jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock'
          AND LENGTH(s.text_content) > 500
    """

    # In non-test mode, skip already processed
    if not args.test:
        sql += " AND (c.humanities_needs_at IS NULL)"

    sql += " ORDER BY LENGTH(s.text_content) DESC"
    if args.limit:
        sql += f" LIMIT {args.limit}"

    # We need to attach matcher DB to ir_conn for the join
    ir_conn.execute(f"ATTACH DATABASE '{MATCHER_DB}' AS matcher")
    companies = ir_conn.execute(sql.replace("FROM sections", "FROM main.sections").replace(
        "FROM companies", "FROM matcher.companies"
    ).replace("LEFT JOIN (SELECT", "LEFT JOIN (SELECT")).fetchall()

    # Simpler approach: just query separately
    ir_conn.execute(f"DETACH DATABASE matcher")

    # Get sections from ir_conn
    section_sql = """
        SELECT edinet_code, filer_name, text_content
        FROM sections
        WHERE section_tag = 'jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock'
          AND LENGTH(text_content) > 500
        ORDER BY char_count DESC
    """
    if args.limit:
        section_sql += f" LIMIT {args.limit}"
    sections = ir_conn.execute(section_sql).fetchall()
    ir_conn.close()

    # Get industry map and skip already processed
    industry_map = {}
    processed = set()
    for row in conn.execute("SELECT edinet_code, industry, humanities_needs_at FROM companies"):
        industry_map[row["edinet_code"]] = row["industry"]
        if row["humanities_needs_at"] and not args.test:
            processed.add(row["edinet_code"])

    companies_to_process = [
        s for s in sections if s["edinet_code"] not in processed
    ]
    logger.info("Total sections: %d, already processed: %d, to process: %d",
                len(sections), len(processed), len(companies_to_process))

    now = datetime.now().isoformat(timespec="seconds")
    success = 0
    failed = 0

    for batch_start in range(0, len(companies_to_process), args.batch_size):
        batch = companies_to_process[batch_start:batch_start + args.batch_size]
        texts_to_embed = []
        records = []

        for co in batch:
            ec = co["edinet_code"]
            name = co["filer_name"]
            industry = industry_map.get(ec, "")

            result = _extract_needs(client, name, industry, co["text_content"])
            time.sleep(0.3)

            if result and result.get("needs"):
                composite = _build_composite_text(name, result)
                texts_to_embed.append(f"passage: {composite}")
                records.append((ec, json.dumps(result, ensure_ascii=False), composite))
                success += 1
            else:
                failed += 1
                logger.debug("No needs detected for %s", name)

        # Batch vectorize
        if texts_to_embed:
            vectors = embeddings.encode(texts_to_embed)
            for i, (ec, needs_json, needs_text) in enumerate(records):
                conn.execute(
                    """UPDATE companies
                       SET humanities_needs_json = ?,
                           humanities_needs_text = ?,
                           humanities_needs_vector = ?,
                           humanities_needs_at = ?
                       WHERE edinet_code = ?""",
                    (needs_json, needs_text, vectors[i].tobytes(), now, ec),
                )
            conn.commit()

        total = batch_start + len(batch)
        logger.info("Progress: %d/%d, success=%d, failed=%d",
                    total, len(companies_to_process), success, failed)

    conn.close()
    logger.info("Done. Success: %d, Failed: %d", success, failed)


if __name__ == "__main__":
    main()
