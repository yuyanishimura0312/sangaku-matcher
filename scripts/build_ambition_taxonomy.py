"""Build ambition taxonomy from extracted business ambition domains.

Analyzes 16,000+ unique ambition domains, clusters them, and generates
a structured 2-level taxonomy using LLM.
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import anthropic

MATCHER_DB = Path(__file__).parent.parent / "data" / "matcher.db"

def main():
    conn = sqlite3.connect(str(MATCHER_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT ambitions_json FROM companies WHERE ambitions_json IS NOT NULL").fetchall()

    # Extract all domain names with maturity
    domains = []
    maturity_count = Counter()
    for r in rows:
        try:
            data = json.loads(r[0])
        except:
            continue
        for amb in data.get("ambitions", []):
            domain = amb.get("domain", "")
            maturity = amb.get("maturity", "")
            if domain:
                domains.append(domain)
                maturity_count[maturity] += 1

    # Get frequency distribution of key terms in domains
    term_counts = Counter()
    for d in domains:
        # Split by common delimiters
        for term in re.split(r'[・の（）へによる]', d):
            term = term.strip()
            if len(term) >= 2:
                term_counts[term] += 1

    # Build context for LLM
    top_terms = [f"{t}({c})" for t, c in term_counts.most_common(100)]
    sample_domains = list(set(domains))[:200]  # Deduplicated samples

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-w"],
            capture_output=True, text=True,
        )
        key = result.stdout.strip()

    client = anthropic.Anthropic(api_key=key)

    prompt = f"""日本企業約3,800社の有価証券報告書から抽出した「新規事業野心領域」16,676件を分析し、テーマ体系を構築してください。

頻出用語（上位100）:
{', '.join(top_terms[:100])}

野心領域名のサンプル（200件）:
{chr(10).join(sample_domains[:200])}

maturity分布: directional 51%, concrete 40%, exploratory 9%

要件:
1. 「企業が挑戦しようとしている事業領域」を軸とした2階層テーマ体系
2. 大テーマ8-10個、小テーマ各3-5個（合計30-40個）
3. 各小テーマは「研究者がこの領域で貢献できる」と判断できる具体性
4. 各小テーマにキーワード3個
5. 各小テーマに「人文社会科学の研究者がどう貢献できるか」の一文
6. JSON形式のみ出力

{{"themes": [{{"id": "A", "name": "大テーマ名", "sub_themes": [{{"id": "A1", "name": "小テーマ名", "keywords": ["k1","k2","k3"], "humanities_role": "人文知の貢献（一文）"}}]}}]}}"""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = msg.content[0].text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        raw = match.group()
        raw = re.sub(r'//[^\n]*', '', raw)
        raw = re.sub(r',\s*([}\]])', r'\1', raw)
        raw = re.sub(r'[\x00-\x1f]', ' ', raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to fix by finding the valid JSON prefix
            for end in range(len(raw), 100, -1):
                try:
                    data = json.loads(raw[:end])
                    break
                except json.JSONDecodeError:
                    continue
            else:
                print("Failed to parse JSON. Raw output:")
                print(text[:2000])
                return
        output_path = Path(__file__).parent.parent / "data" / "ambition_taxonomy.json"
        with open(output_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Display
        total_sub = 0
        for theme in data["themes"]:
            print(f"\n{theme['id']}. {theme['name']}")
            for sub in theme["sub_themes"]:
                kws = ", ".join(sub["keywords"])
                print(f"   {sub['id']}. {sub['name']}  [{kws}]")
                print(f"      -> {sub.get('humanities_role', '')}")
                total_sub += 1

        print(f"\nTotal: {len(data['themes'])} major, {total_sub} sub")
        print(f"Saved to {output_path}")
    else:
        print(text)

    conn.close()


if __name__ == "__main__":
    main()
