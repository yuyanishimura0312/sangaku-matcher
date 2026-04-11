"""Parallel extraction of contextual tech needs from 有報 text.

Replaces keyword-based estimated_needs with rich narrative descriptions
that capture the business context behind each technology need.
Uses ThreadPoolExecutor for concurrent Claude API calls with rate limiting.
Skips already-processed companies for safe restarts.

Usage:
    python scripts/extract_tech_needs_parallel.py [--workers 16] [--limit N] [--test]
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"
IR_DB = Path.home() / "projects/apps/ir-collector/data/ir.db"

# Technology needs taxonomy based on industry-academia collaboration literature
NEED_TYPES = {
    "core_rd": "コアR&D課題",
    "digital_transformation": "DX・データ活用",
    "materials_process": "素材・プロセス技術",
    "sustainability": "環境・サステナビリティ技術",
    "product_service": "製品・サービス革新",
    "production_efficiency": "生産・効率化技術",
    "safety_quality": "安全・品質技術",
    "frontier_exploration": "フロンティア探索",
}

SYSTEM_PROMPT = """あなたは産学連携の専門アナリストです。企業の有価証券報告書のテキストを深く読み解き、その企業が大学・研究機関の技術シーズと連携することで価値を生み出せる領域を、文脈を保持した文章として記述してください。

重要な原則：
- キーワードの羅列ではなく、企業の具体的な経営課題・戦略の文脈の中で技術ニーズを記述してください
- 「なぜその企業がその技術を必要としているのか」という事業上の背景を含めてください
- 具体的な技術名・製品名・数値目標がある場合は積極的に含めてください
- 免責定型文（「将来に関する事項は〜が判断した」）は無視してください
- 検出されない類型については無理に書かないでください
- 各ニーズは独立した文脈保持型の文章で、研究者が読んで連携の可能性を判断できる情報量を含めてください"""

USER_PROMPT = """以下は{company_name}（{industry}）の有価証券報告書「経営方針、経営環境及び対処すべき課題等」からの抜粋です。

---
{ir_text}
---

この企業が大学・研究機関の技術シーズと連携することで解決できる課題を分析し、以下の8類型について検出結果を文章で記述してください。

1. コアR&D課題 — 企業の中核事業に直結する研究開発テーマ（次世代製品、基盤技術等）
2. DX・データ活用 — AI、IoT、データ分析、デジタルツイン等のデジタル技術活用
3. 素材・プロセス技術 — 新素材開発、製造プロセス革新、触媒、バイオテクノロジー等
4. 環境・サステナビリティ技術 — カーボンニュートラル、省エネ、リサイクル、環境負荷低減
5. 製品・サービス革新 — 新製品開発、サービスモデル変革、ユーザー体験向上
6. 生産・効率化技術 — 自動化、ロボティクス、サプライチェーン最適化、品質管理
7. 安全・品質技術 — 安全性評価、信頼性工学、非破壊検査、規制対応技術
8. フロンティア探索 — 既存事業の延長線上にない新領域探索（量子、宇宙、合成生物学等）

JSON形式で出力してください。各ニーズは「この企業は〜という事業課題に取り組んでおり、〜分野の技術シーズとの連携により〜が期待できる」という形式の100-200字程度の文章で記述してください。確信度が低い類型は含めないでください。

{{"needs": [{{"type": "類型キー", "confidence": "high|medium", "narrative": "文脈保持した記述文"}}], "summary": "全体像200-300字"}}"""


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
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _parse_json_response(text: str) -> dict | None:
    """Robustly parse JSON from LLM response."""
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        return None
    raw = json_match.group()
    raw = re.sub(r'//[^\n]*', '', raw)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    raw = re.sub(r'[\x00-\x1f]', ' ', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# Rate limiter
_rate_lock = Lock()
_last_request = 0.0
MIN_INTERVAL = 0.15


def _throttle():
    global _last_request
    with _rate_lock:
        now = time.time()
        wait = MIN_INTERVAL - (now - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()


def _extract_one(client, edinet_code: str, name: str, industry: str, ir_text: str) -> dict:
    """Extract tech needs for one company. Thread-safe."""
    clean = _strip_html(ir_text)[:6000]
    prompt = USER_PROMPT.format(company_name=name, industry=industry or "不明", ir_text=clean)

    for attempt in range(2):
        _throttle()
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            result = _parse_json_response(msg.content[0].text)
            if result and result.get("needs"):
                return {"edinet_code": edinet_code, "name": name, "result": result}

            if attempt == 0:
                _throttle()
                retry = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2500,
                    system="Output only valid JSON. No markdown, no comments.",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": msg.content[0].text},
                        {"role": "user", "content": "JSONが不正でした。正しいJSON（{で始まり}で終わる）のみ出力してください。"},
                    ],
                )
                result = _parse_json_response(retry.content[0].text)
                if result and result.get("needs"):
                    return {"edinet_code": edinet_code, "name": name, "result": result}
        except Exception as e:
            logger.warning("%s: API error (attempt %d) - %s", name, attempt + 1, e)
            time.sleep(2)

    return {"edinet_code": edinet_code, "name": name, "result": None}


def _build_composite_text(name: str, result: dict) -> str:
    """Build a single narrative text for embedding."""
    parts = [f"{name}の技術連携ニーズ。"]
    for need in result.get("needs", []):
        type_name = NEED_TYPES.get(need.get("type", ""), need.get("type", ""))
        narrative = need.get("narrative", "")
        if narrative:
            parts.append(f"【{type_name}】{narrative}")
    summary = result.get("summary", "")
    if summary:
        parts.append(f"【総合評価】{summary}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Test with 10 companies")
    args = parser.parse_args()

    if args.test:
        args.limit = 10

    import anthropic
    api_key = _get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings

    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row

    # Ensure columns exist
    for col in ["tech_needs_json TEXT", "tech_needs_text TEXT",
                "tech_needs_vector BLOB", "tech_needs_at TEXT"]:
        try:
            conn.execute(f"ALTER TABLE companies ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.commit()

    # Get already processed
    processed = {r["edinet_code"] for r in conn.execute(
        "SELECT edinet_code FROM companies WHERE tech_needs_at IS NOT NULL"
    )}

    # Get industry map
    industry_map = {r["edinet_code"]: r["industry"] for r in conn.execute(
        "SELECT edinet_code, industry FROM companies"
    )}

    # Get sections from ir-collector
    ir_conn = sqlite3.connect(str(IR_DB))
    ir_conn.row_factory = sqlite3.Row
    sections = ir_conn.execute(
        "SELECT edinet_code, filer_name, text_content FROM sections "
        "WHERE section_tag = 'jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock' "
        "AND LENGTH(text_content) > 500"
    ).fetchall()
    ir_conn.close()

    # Filter unprocessed
    to_process = [s for s in sections if s["edinet_code"] not in processed]
    if args.limit:
        to_process = to_process[:args.limit]

    logger.info("Total: %d, processed: %d, remaining: %d, workers: %d",
                len(sections), len(processed), len(to_process), args.workers)

    if not to_process:
        logger.info("All companies already processed.")
        conn.close()
        return

    now = datetime.now().isoformat(timespec="seconds")
    success = 0
    failed = 0
    db_lock = Lock()
    COMMIT_EVERY = 20

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _extract_one, client,
                s["edinet_code"], s["filer_name"],
                industry_map.get(s["edinet_code"], ""),
                s["text_content"]
            ): s["edinet_code"]
            for s in to_process
        }

        batch_texts = []
        batch_records = []

        for future in as_completed(futures):
            result = future.result()
            ec = result["edinet_code"]
            name = result["name"]

            if result["result"]:
                composite = _build_composite_text(name, result["result"])
                needs_json = json.dumps(result["result"], ensure_ascii=False)

                with db_lock:
                    batch_texts.append(f"passage: {composite}")
                    batch_records.append((ec, needs_json, composite))
                    success += 1

                    if len(batch_records) >= COMMIT_EVERY:
                        vectors = embeddings.encode(batch_texts)
                        for i, (code, nj, nt) in enumerate(batch_records):
                            conn.execute(
                                """UPDATE companies
                                   SET tech_needs_json=?, tech_needs_text=?,
                                       tech_needs_vector=?, tech_needs_at=?
                                   WHERE edinet_code=?""",
                                (nj, nt, vectors[i].tobytes(), now, code),
                            )
                        conn.commit()
                        logger.info("Committed %d. Total: success=%d, failed=%d / %d",
                                    len(batch_records), success, failed, len(to_process))
                        batch_texts.clear()
                        batch_records.clear()
            else:
                with db_lock:
                    failed += 1

        # Commit remaining
        if batch_records:
            with db_lock:
                vectors = embeddings.encode(batch_texts)
                for i, (code, nj, nt) in enumerate(batch_records):
                    conn.execute(
                        """UPDATE companies
                           SET tech_needs_json=?, tech_needs_text=?,
                               tech_needs_vector=?, tech_needs_at=?
                           WHERE edinet_code=?""",
                        (nj, nt, vectors[i].tobytes(), now, code),
                    )
                conn.commit()
                logger.info("Final commit %d records", len(batch_records))

    conn.close()
    logger.info("Done. Success: %d, Failed: %d, Total: %d", success, failed, len(to_process))


if __name__ == "__main__":
    main()
