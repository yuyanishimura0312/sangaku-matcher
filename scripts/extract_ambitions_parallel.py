"""Extract business ambition domains from 有報 text using Claude API.

For each company, identifies new business areas, strategic challenges,
and ambitious domains the company is trying to enter or build.
Classifies each by maturity (concrete/directional/exploratory) using
the Three Horizons model.

Usage:
    python scripts/extract_ambitions_parallel.py [--workers 16] [--limit N] [--test]
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

SYSTEM_PROMPT = """あなたは産学連携の専門アナリストです。企業の有価証券報告書を読み解き、その企業が挑戦しようとしている「新規事業・野心的領域」を抽出してください。

重要な原則：
- 既存事業の維持・改善ではなく、「新しく挑戦しようとしていること」に焦点を当ててください
- 「やりたいが、まだ十分にできていないこと」「これから参入しようとしている領域」を重視してください
- 各野心領域について、なぜその企業がその領域に挑戦するのか（事業上の背景）を含めてください
- 野心度をThree Horizonsモデルで分類してください:
  - concrete (H1寄り): 具体的な計画・投資・組織が存在する
  - directional (H2): 方向性は示されているが具体策は構築中
  - exploratory (H3): 探索段階、非連続的な挑戦
- 免責定型文（「将来に関する事項は〜が判断した」）は無視してください
- 検出されない場合は無理に作らないでください"""

USER_PROMPT = """以下は{company_name}（{industry}）の有価証券報告書「経営方針、経営環境及び対処すべき課題等」からの抜粋です。

---
{ir_text}
---

この企業が挑戦しようとしている新規事業・野心的領域を分析し、JSON形式で出力してください。
各野心領域は100-200字の文脈保持型記述で、「この企業は〜という背景から〜という新領域に挑戦しようとしている」という形式です。
また、その野心を実現するために人文社会科学の知見がどう貢献できるかを50-100字で記述してください。

{{"ambitions": [{{"domain": "野心領域の名称（10-20字）", "maturity": "concrete|directional|exploratory", "description": "文脈保持型記述100-200字", "humanities_connection": "人文社会科学の貢献50-100字"}}], "overall": "この企業の野心の全体像100-200字"}}"""


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
            if result and result.get("ambitions"):
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
                        {"role": "user", "content": "JSONが不正でした。正しいJSONのみ出力してください。"},
                    ],
                )
                result = _parse_json_response(retry.content[0].text)
                if result and result.get("ambitions"):
                    return {"edinet_code": edinet_code, "name": name, "result": result}
        except Exception as e:
            logger.warning("%s: API error (attempt %d) - %s", name, attempt + 1, e)
            time.sleep(2)

    return {"edinet_code": edinet_code, "name": name, "result": None}


def _build_composite_text(name: str, result: dict) -> str:
    parts = [f"{name}の新規事業野心領域。"]
    for amb in result.get("ambitions", []):
        domain = amb.get("domain", "")
        desc = amb.get("description", "")
        maturity = {"concrete": "具体的計画", "directional": "方向性提示", "exploratory": "探索段階"}.get(
            amb.get("maturity", ""), amb.get("maturity", ""))
        hc = amb.get("humanities_connection", "")
        if desc:
            parts.append(f"【{domain}（{maturity}）】{desc}")
            if hc:
                parts.append(f"→人文知の接続: {hc}")
    overall = result.get("overall", "")
    if overall:
        parts.append(f"【全体像】{overall}")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test", action="store_true")
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
    for col in ["ambitions_json TEXT", "ambitions_text TEXT",
                "ambitions_vector BLOB", "ambitions_at TEXT"]:
        try:
            conn.execute(f"ALTER TABLE companies ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    conn.commit()

    processed = {r["edinet_code"] for r in conn.execute(
        "SELECT edinet_code FROM companies WHERE ambitions_at IS NOT NULL"
    )}

    industry_map = {r["edinet_code"]: r["industry"] for r in conn.execute(
        "SELECT edinet_code, industry FROM companies"
    )}

    ir_conn = sqlite3.connect(str(IR_DB))
    ir_conn.row_factory = sqlite3.Row
    sections = ir_conn.execute(
        "SELECT edinet_code, filer_name, text_content FROM sections "
        "WHERE section_tag = 'jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock' "
        "AND LENGTH(text_content) > 500"
    ).fetchall()
    ir_conn.close()

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
                ambitions_json = json.dumps(result["result"], ensure_ascii=False)

                with db_lock:
                    batch_texts.append(f"passage: {composite}")
                    batch_records.append((ec, ambitions_json, composite))
                    success += 1

                    if len(batch_records) >= COMMIT_EVERY:
                        vectors = embeddings.encode(batch_texts)
                        for i, (code, aj, at) in enumerate(batch_records):
                            conn.execute(
                                """UPDATE companies
                                   SET ambitions_json=?, ambitions_text=?,
                                       ambitions_vector=?, ambitions_at=?
                                   WHERE edinet_code=?""",
                                (aj, at, vectors[i].tobytes(), now, code),
                            )
                        conn.commit()
                        logger.info("Committed %d. Total: success=%d, failed=%d / %d",
                                    len(batch_records), success, failed, len(to_process))
                        batch_texts.clear()
                        batch_records.clear()
            else:
                with db_lock:
                    failed += 1

        if batch_records:
            with db_lock:
                vectors = embeddings.encode(batch_texts)
                for i, (code, aj, at) in enumerate(batch_records):
                    conn.execute(
                        """UPDATE companies
                           SET ambitions_json=?, ambitions_text=?,
                               ambitions_vector=?, ambitions_at=?
                           WHERE edinet_code=?""",
                        (aj, at, vectors[i].tobytes(), now, code),
                    )
                conn.commit()
                logger.info("Final commit %d records", len(batch_records))

    conn.close()
    logger.info("Done. Success: %d, Failed: %d, Total: %d", success, failed, len(to_process))


if __name__ == "__main__":
    main()
