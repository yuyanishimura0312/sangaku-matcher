"""Generate a structured theme taxonomy from bottom-up keyword analysis."""
import json
import os
import re
import subprocess

import anthropic

key = os.environ.get("ANTHROPIC_API_KEY")
if not key:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", "ANTHROPIC_API_KEY", "-w"],
        capture_output=True, text=True,
    )
    key = result.stdout.strip()

client = anthropic.Anthropic(api_key=key)

prompt = """日本企業約3,800社の有価証券報告書から抽出した「人文社会科学系連携ニーズ」のボトムアップ分析で、以下の社会的テーマが頻出しています:

地域社会(1314), 組織文化(1205), 企業倫理(707), 組織変革(626), カーボンニュートラル(446), 消費社会(440), 脱炭素社会(361), 地域コミュニティ(342), 人口減少(318), AI倫理(227), 消費文化(187), 少子高齢化(169), 人口動態(164), 地域共創(128), 高齢化社会(127), コーポレートガバナンス(112), 気候変動(99), データ倫理(91), 医療倫理(86), デジタルトランスフォーメーション(84), 労働倫理(81), 生命倫理(59)

これらを「研究者が自分の専門分野とマッチングできるレベルの具体性」で2階層テーマ体系にしてください。

要件:
- 大テーマ7-8個、小テーマ各3-5個（合計25-35個）
- 小テーマは研究者の専門領域に対応する具体性
- 各小テーマにキーワード3個
- JSON形式のみ出力

{"themes": [{"id": "A", "name": "...", "sub_themes": [{"id": "A1", "name": "...", "keywords": ["k1","k2","k3"]}]}]}"""

msg = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4000,
    messages=[{"role": "user", "content": prompt}],
)

text = msg.content[0].text
match = re.search(r'\{[\s\S]*\}', text)
if match:
    data = json.loads(match.group())
    print(json.dumps(data, ensure_ascii=False, indent=2))
    # Save to file
    with open("data/theme_taxonomy.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to data/theme_taxonomy.json")
    print(f"Total themes: {len(data['themes'])} major, {sum(len(t['sub_themes']) for t in data['themes'])} sub")
else:
    print(text)
