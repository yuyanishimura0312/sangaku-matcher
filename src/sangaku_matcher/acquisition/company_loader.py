"""Load and register companies into the matcher database.

Supports two modes:
1. IR Collector mode: reads from the IR Collector SQLite DB
2. Dummy mode: inserts synthetic companies for testing
"""
from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np

from sangaku_matcher.config import settings
from sangaku_matcher.db import connect, init_db
from sangaku_matcher import embeddings
from sangaku_matcher.acquisition.edinet_loader import load_filers, get_latest_pdf_path, ir_collector_available
from sangaku_matcher.acquisition.pdf_extractor import extract_rd_section

logger = logging.getLogger(__name__)

# Dummy companies for testing when IR Collector is not available
DUMMY_COMPANIES = [
    {"edinet_code": "E02144", "sec_code": "7203", "name": "トヨタ自動車株式会社", "industry": "輸送用機器", "revenue": 45095325, "rd_expense": 1220000, "employees": 375235, "rd_text": "当社グループは自動車事業を中心に、水素エネルギー、電動化技術、自動運転、コネクティッド技術の研究開発を推進。カーボンニュートラルに向けた全方位戦略として、BEV、PHEV、HEV、FCEVの各パワートレインの開発を加速している。"},
    {"edinet_code": "E01777", "sec_code": "6758", "name": "ソニーグループ株式会社", "industry": "電気機器", "revenue": 13020768, "rd_expense": 750000, "employees": 113000, "rd_text": "半導体イメージセンサー、ゲーム・ネットワークサービス、音楽・映画等のエンタテインメント領域に加え、AIロボティクス、メタバース関連技術の研究開発を実施。CMOSイメージセンサーでは世界シェアトップを維持し、車載向けセンサーの開発を強化。"},
    {"edinet_code": "E00919", "sec_code": "4502", "name": "武田薬品工業株式会社", "industry": "医薬品", "revenue": 4302706, "rd_expense": 541820, "employees": 49095, "rd_text": "消化器系疾患、希少疾患、血漿分画製剤、オンコロジー、ニューロサイエンスの5つの主要事業領域に注力。パイプラインには約40の新薬候補があり、細胞治療・遺伝子治療の基盤構築を推進。外部連携プラットフォームCOCKPI-Tを通じたオープンイノベーションを積極展開。"},
    {"edinet_code": "E00513", "sec_code": "4901", "name": "富士フイルムホールディングス株式会社", "industry": "化学", "revenue": 2961900, "rd_expense": 213400, "employees": 73878, "rd_text": "ヘルスケア、マテリアルズ、イメージング、ビジネスイノベーションの4事業領域で研究開発。バイオCDMO、再生医療（iPS細胞）、AI画像診断、高機能材料（半導体プロセス材料）に注力。Open Innovation Hubを通じた産学連携を推進。"},
    {"edinet_code": "E00855", "sec_code": "3402", "name": "東レ株式会社", "industry": "繊維製品", "revenue": 2489809, "rd_expense": 68200, "employees": 48842, "rd_text": "炭素繊維複合材料、水処理膜、リチウムイオン電池用セパレータ、有機EL材料の研究開発を推進。グリーンイノベーション戦略として、炭素繊維リサイクル技術、水素関連材料、バイオマス由来ポリマーの開発に注力。"},
    {"edinet_code": "E01225", "sec_code": "6501", "name": "株式会社日立製作所", "industry": "電気機器", "revenue": 10881150, "rd_expense": 336000, "employees": 268655, "rd_text": "社会イノベーション事業のもと、Lumada（IoTプラットフォーム）を核にデジタル・グリーン・イノベーション関連技術を開発。量子コンピューティング、生成AI、パワーグリッド、鉄道システムの研究開発を推進。"},
    {"edinet_code": "E02529", "sec_code": "6861", "name": "キーエンス", "industry": "電気機器", "revenue": 922430, "rd_expense": 30000, "employees": 12286, "rd_text": "FAセンサ、測定器、画像処理システム、レーザマーカ等の開発。顧客の製造現場の課題解決に直結する独自技術の研究開発を実施。"},
    {"edinet_code": "E00857", "sec_code": "4063", "name": "信越化学工業株式会社", "industry": "化学", "revenue": 2374433, "rd_expense": 69200, "employees": 27557, "rd_text": "塩化ビニル樹脂、半導体シリコン、シリコーン、合成石英、希土類磁石等の研究開発。半導体材料ではEUV用フォトマスク基板、ArF用フォトレジストの開発を推進。"},
    {"edinet_code": "E01110", "sec_code": "4568", "name": "第一三共株式会社", "industry": "医薬品", "revenue": 1616500, "rd_expense": 359600, "employees": 17435, "rd_text": "がん領域を最重点領域として、抗体薬物複合体（ADC）技術を核としたパイプライン構築を推進。エンハーツ（DXd ADC）の適応拡大、次世代ADCの開発、及び遺伝子治療、核酸医薬の探索研究を実施。"},
    {"edinet_code": "E01619", "sec_code": "6902", "name": "デンソー", "industry": "輸送用機器", "revenue": 7147814, "rd_expense": 500000, "employees": 162310, "rd_text": "電動化、自動運転、コネクティッド技術の研究開発。車載半導体、全固体電池材料、LiDAR、車載通信モジュールの開発を推進。カーボンニュートラルに向けたファクトリーオートメーション技術も開発。"},
    {"edinet_code": "E02160", "sec_code": "4543", "name": "テルモ株式会社", "industry": "精密機器", "revenue": 921696, "rd_expense": 50600, "employees": 30924, "rd_text": "カテーテル治療、輸血・細胞治療、医療DXの3領域で研究開発。細胞培養・加工技術、血液浄化システム、低侵襲治療デバイスの開発を推進。"},
    {"edinet_code": "E01600", "sec_code": "6762", "name": "TDK株式会社", "industry": "電気機器", "revenue": 2183000, "rd_expense": 130000, "employees": 101844, "rd_text": "電子部品・磁性材料の研究開発。全固体電池、次世代センサ、高周波部品、エナジーハーベスティング技術の開発を推進。"},
]


def load_companies_to_db(use_ir_collector: bool = False, dummy: bool = False) -> None:
    """Load companies into matcher DB.

    Args:
        use_ir_collector: if True, read from IR Collector DB
        dummy: if True, insert synthetic test data
    """
    init_db(settings.matcher_db_path)

    if dummy or (not use_ir_collector and not ir_collector_available()):
        if not dummy:
            logger.warning("IR Collector not available, falling back to dummy data.")
        _load_dummy()
        return

    if use_ir_collector:
        _load_from_ir_collector()


def _load_dummy() -> None:
    """Insert dummy companies with pre-computed embeddings."""
    logger.info("Loading %d dummy companies...", len(DUMMY_COMPANIES))

    # Compute embeddings for all R&D texts
    rd_texts = [f"passage: {c['rd_text']}" for c in DUMMY_COMPANIES]
    vectors = embeddings.encode(rd_texts)

    now = datetime.now().isoformat(timespec="seconds")
    with connect(settings.matcher_db_path) as conn:
        for i, co in enumerate(DUMMY_COMPANIES):
            revenue = co.get("revenue", 0) or 0
            rd_expense = co.get("rd_expense", 0) or 0
            rd_intensity = rd_expense / revenue if revenue > 0 else 0.0

            conn.execute(
                """INSERT OR REPLACE INTO companies
                   (edinet_code, sec_code, name, industry, revenue, rd_expense,
                    rd_intensity, employees, rd_text, rd_text_vector, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    co["edinet_code"],
                    co.get("sec_code", ""),
                    co["name"],
                    co.get("industry", ""),
                    revenue,
                    rd_expense,
                    rd_intensity,
                    co.get("employees", 0),
                    co["rd_text"],
                    vectors[i].tobytes(),
                    now,
                ),
            )

        # Compute industry stats (SQLite lacks STDEV, so compute in Python)
        industries: dict[str, list[float]] = {}
        for row in conn.execute("SELECT industry, rd_intensity FROM companies"):
            industries.setdefault(row["industry"], []).append(row["rd_intensity"] or 0)

        for ind, vals in industries.items():
            mean = sum(vals) / len(vals) if vals else 0
            variance = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.01
            std = math.sqrt(variance) if variance > 0 else 0.01
            conn.execute(
                """INSERT OR REPLACE INTO industry_stats
                   (industry, mean_rd_int, std_rd_int, company_count, computed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (ind, mean, std, len(vals), now),
            )

    logger.info("Loaded %d dummy companies with embeddings.", len(DUMMY_COMPANIES))


def _load_from_ir_collector() -> None:
    """Load companies from IR Collector DB and extract R&D text from PDFs."""
    filers = load_filers()
    logger.info("Found %d filers in IR Collector DB.", len(filers))

    if not filers:
        logger.warning("No filers found. Run IR Collector first.")
        return

    now = datetime.now().isoformat(timespec="seconds")
    loaded = 0

    # Process in batches for embedding efficiency
    batch_size = 50
    batch_companies = []
    batch_texts = []

    with connect(settings.matcher_db_path) as conn:
        for filer in filers:
            code = filer["edinet_code"]
            pdf_path = get_latest_pdf_path(code)
            rd_text = ""
            if pdf_path and pdf_path.exists():
                rd_text = extract_rd_section(pdf_path) or ""

            batch_companies.append({
                "edinet_code": code,
                "sec_code": filer.get("sec_code", ""),
                "name": filer["filer_name"],
                "rd_text": rd_text,
            })
            batch_texts.append(f"passage: {rd_text}" if rd_text else "passage: ")

            if len(batch_companies) >= batch_size:
                _flush_batch(conn, batch_companies, batch_texts, now)
                loaded += len(batch_companies)
                logger.info("Loaded %d / %d companies...", loaded, len(filers))
                batch_companies = []
                batch_texts = []

        if batch_companies:
            _flush_batch(conn, batch_companies, batch_texts, now)
            loaded += len(batch_companies)

    logger.info("Loaded %d companies from IR Collector.", loaded)


def _flush_batch(conn, companies: list[dict], texts: list[str], now: str) -> None:
    """Insert a batch of companies with embeddings."""
    vectors = embeddings.encode(texts)
    for i, co in enumerate(companies):
        conn.execute(
            """INSERT OR REPLACE INTO companies
               (edinet_code, sec_code, name, rd_text, rd_text_vector, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                co["edinet_code"],
                co["sec_code"],
                co["name"],
                co["rd_text"],
                vectors[i].tobytes(),
                now,
            ),
        )
