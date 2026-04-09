"""Generate estimated needs text and vectors for all companies.

Uses industry-specific templates + company R&D characteristics to create
synthetic needs descriptions, then embeds them as needs_vector.

This is a heuristic approach that works without external API calls.
For higher quality, replace with Claude API-based needs generation
using mid-term plan text from EDINET.

Usage:
    python scripts/generate_needs.py [--batch-size 100]
"""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "matcher.db"

# Industry-specific technology needs templates
# Each maps to typical external technology needs for that sector
INDUSTRY_NEEDS = {
    "情報・通信業": (
        "AI・機械学習の高度化、サイバーセキュリティ強化、クラウドネイティブ技術、"
        "自然言語処理・大規模言語モデル活用、IoTプラットフォーム構築、"
        "ブロックチェーン・分散システム、量子コンピューティング基盤技術、"
        "プライバシー保護技術、エッジコンピューティング、デジタルツイン技術"
    ),
    "サービス業": (
        "顧客行動分析・パーソナライゼーション技術、業務自動化・RPA、"
        "需要予測・最適化アルゴリズム、ヘルスケアテクノロジー、"
        "人材マッチング・HR Tech、フードテック・飲食DX、"
        "観光・ホスピタリティ向けAI、教育テクノロジー"
    ),
    "電気機器": (
        "パワー半導体・SiC/GaN素子、次世代ディスプレイ技術、"
        "センサーフュージョン・LiDAR、ロボティクス・自動制御、"
        "エネルギーハーベスティング、全固体電池・蓄電技術、"
        "6G通信技術、光半導体・フォトニクス、MEMSデバイス、"
        "AI推論チップ・エッジAIハードウェア"
    ),
    "化学": (
        "機能性材料・高分子設計、触媒技術・反応プロセス最適化、"
        "バイオプラスチック・サステナブル素材、電池材料・電極材料、"
        "半導体プロセス材料、マテリアルズインフォマティクス、"
        "CO2固定・回収技術、ナノテクノロジー、バイオケミカル"
    ),
    "医薬品": (
        "抗体薬物複合体(ADC)技術、核酸医薬・mRNA技術、"
        "細胞治療・遺伝子治療、AI創薬・ドラッグリポジショニング、"
        "バイオマーカー・コンパニオン診断、再生医療・iPS細胞応用、"
        "マイクロバイオーム、デジタルセラピューティクス"
    ),
    "機械": (
        "スマートファクトリー・Industry 4.0、予知保全・デジタルツイン、"
        "産業用ロボット・協働ロボット、3Dプリンティング・AM技術、"
        "水素エネルギー関連機器、省エネルギー技術、"
        "自動搬送・物流自動化、精密加工・微細加工技術"
    ),
    "輸送用機器": (
        "電動化技術(BEV/PHEV/FCEV)、自動運転・ADAS、"
        "車載半導体・車載OS、軽量化材料・CFRP、"
        "コネクティッドカー・V2X通信、全固体電池、"
        "水素エンジン・燃料電池、MaaS・モビリティサービス"
    ),
    "食料品": (
        "フードテック・代替タンパク質、食品保存・鮮度管理技術、"
        "発酵・バイオテクノロジー、食品安全・トレーサビリティ、"
        "健康機能性食品・ニュートリゲノミクス、スマート農業連携、"
        "食品ロス削減技術、フレーバー・テクスチャー制御"
    ),
    "建設業": (
        "BIM/CIM・建設DX、ロボット施工・自動化施工、"
        "ZEB/ZEH・省エネ建築技術、インフラモニタリング・非破壊検査、"
        "コンクリート長寿命化、木造・CLT技術、"
        "災害対策・レジリエンス技術、スマートシティ基盤"
    ),
    "小売業": (
        "EC・オムニチャネル技術、需要予測・在庫最適化AI、"
        "店舗DX・無人店舗技術、サプライチェーン最適化、"
        "顧客データ分析・CRM高度化、決済テクノロジー、"
        "ラストワンマイル配送最適化"
    ),
    "卸売業": (
        "サプライチェーンマネジメント高度化、貿易・物流DX、"
        "需要予測・価格最適化、B2Bマーケットプレイス、"
        "倉庫自動化・ロボティクス、ブロックチェーン活用トレーサビリティ"
    ),
    "不動産業": (
        "PropTech・不動産テック、スマートビル管理、"
        "エネルギーマネジメント・ZEB、VR/AR内覧技術、"
        "不動産価値評価AI、コンバージョン・リノベーション技術"
    ),
    "精密機器": (
        "光学技術・イメージング、微細加工・ナノ精度制御、"
        "医療機器・診断技術、バイオセンサー、"
        "量子計測・超精密測定、ウェアラブルデバイス"
    ),
    "繊維製品": (
        "高機能繊維・スマートテキスタイル、サステナブル繊維、"
        "炭素繊維・複合材料、フィルター・膜技術、"
        "バイオ繊維・セルロースナノファイバー"
    ),
    "金属製品": (
        "軽量化・高強度金属材料、金属3Dプリンティング、"
        "表面処理・コーティング技術、金属リサイクル技術、"
        "耐熱合金・超合金"
    ),
    "ガラス・土石製品": (
        "高機能ガラス・光学材料、セラミックス・電子材料、"
        "建材の環境性能向上、リサイクル・循環型材料技術"
    ),
    "銀行業": (
        "フィンテック・デジタルバンキング、AI融資審査・信用スコアリング、"
        "ブロックチェーン・CBDC、サイバーセキュリティ、"
        "リスク管理・RegTech"
    ),
    "陸運業": (
        "自動運転・隊列走行、物流最適化AI、"
        "ドローン配送、EV・水素トラック、"
        "スマート倉庫・自動仕分け"
    ),
}

# Default template for industries not in the map
DEFAULT_NEEDS = (
    "DX推進・業務デジタル化、データ分析・AI活用、"
    "サステナビリティ・環境対応技術、新規事業開発支援、"
    "省エネルギー・カーボンニュートラル対応"
)


def _generate_needs_text(company: dict) -> str:
    """Generate estimated needs text from company attributes."""
    industry = company.get("industry") or ""
    name = company.get("name") or ""
    rd_expense = company.get("rd_expense") or 0
    rd_intensity = company.get("rd_intensity") or 0
    revenue = company.get("revenue") or 0

    # Base needs from industry template
    base_needs = INDUSTRY_NEEDS.get(industry, DEFAULT_NEEDS)

    # R&D intensity modifier
    if rd_intensity > 0.05:
        rd_context = "研究開発投資比率が高く、先端技術の外部導入に積極的。"
    elif rd_intensity > 0.02:
        rd_context = "一定のR&D投資を行い、技術課題の解決パートナーを求めている。"
    else:
        rd_context = "R&D投資は限定的だが、外部技術による事業革新を模索している。"

    # Scale modifier
    if revenue > 1000000:  # > 1T JPY
        scale = "大手企業として、大規模な共同研究や技術ライセンスの受入体制がある。"
    elif revenue > 100000:  # > 100B JPY
        scale = "中堅〜大手として、特定領域の技術パートナーシップを重視。"
    else:
        scale = "成長企業として、大学発技術のスピーディーな事業化に関心。"

    return f"{name}の推定技術ニーズ。{rd_context}{scale}求める技術領域: {base_needs}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    # Import embedding module (triggers model load)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sangaku_matcher import embeddings

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    companies = conn.execute(
        "SELECT edinet_code, name, industry, revenue, rd_expense, rd_intensity FROM companies"
    ).fetchall()
    logger.info("Generating needs for %d companies", len(companies))

    now = datetime.now().isoformat(timespec="seconds")
    processed = 0

    for batch_start in range(0, len(companies), args.batch_size):
        batch = companies[batch_start:batch_start + args.batch_size]
        needs_texts = []
        edinet_codes = []

        for co in batch:
            co_dict = dict(co)
            needs_text = _generate_needs_text(co_dict)
            needs_texts.append(f"passage: {needs_text}")
            edinet_codes.append(co["edinet_code"])

        # Batch encode
        vectors = embeddings.encode(needs_texts)

        for i, code in enumerate(edinet_codes):
            # Extract plain text (without "passage: " prefix)
            plain_text = needs_texts[i][len("passage: "):]
            conn.execute(
                """UPDATE companies
                   SET estimated_needs = ?, needs_vector = ?, needs_generated_at = ?
                   WHERE edinet_code = ?""",
                (plain_text, vectors[i].tobytes(), now, code),
            )

        conn.commit()
        processed += len(batch)
        logger.info("Progress: %d / %d companies", processed, len(companies))

    conn.close()
    logger.info("Done. Generated needs for %d companies.", processed)


if __name__ == "__main__":
    main()
