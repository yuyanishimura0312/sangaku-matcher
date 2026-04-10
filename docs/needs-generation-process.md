# ニーズデータ生成プロセス

本ドキュメントは、sangaku-matcherの推定ニーズデータがどのように生成されたかを記録し、将来のアップデート時の参照資料として機能する。

---

## 概要

sangaku-matcherの `need_fit` スコアラーは、企業の推定技術ニーズ（`estimated_needs`テキスト + `needs_vector`ベクトル）と入力シーズのベクトルのコサイン類似度を算出する。

現行版（v2）では、EDINET API v2から有報の「経営方針、経営環境及び対処すべき課題等」セクションを取得し、Claude APIで個社別のニーズテキストを生成している。有報が取得できなかった企業にはv1テンプレートがフォールバックとして適用される。

## 現行の生成方式（v2: EDINET + Claude API）

### 生成パイプライン

```
[EDINET API v2] → [有報テキスト取得] → [Claude Haiku 4.5] → [個社別ニーズテキスト] → [ベクトル化]
```

### Step 1: 有報テキスト取得

**スクリプト**: `scripts/fetch_edinet_strategy.py`

EDINET API v2の書類一覧API（`documents.json`）を日付スキャンし、各企業の有価証券報告書（docTypeCode: 120）を特定。CSV形式（type=5）またはXBRL形式（type=1）で以下のセクションを抽出する。

| 優先度 | XBRLタグ | セクション名 |
|---|---|---|
| 1 | `jpcrp_cor:BusinessPolicyBusinessEnvironmentIssuesToAddressEtcTextBlock` | 経営方針、経営環境及び対処すべき課題等 |
| 2 | `jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock` | 経営者による財政状態の分析 |
| 3 | `jpcrp_cor:ResearchAndDevelopmentActivitiesTextBlock` | 研究開発活動 |

- **スキャン期間**: 2024-04-01 〜 2026-04-01（約2年間）
- **取得結果**: 3,829社（98.9%）

### Step 2: Claude APIによる構造化抽出

**スクリプト**: `scripts/generate_needs_v2.py`

取得した経営方針テキスト（最大4,000字）をClaude Haiku 4.5に入力し、200-400字の個社別ニーズテキストを生成する。

```
{企業名}の推定技術ニーズ。[R&Dの方向性の要約]。[新規事業領域への言及]。
求める技術領域: [具体的な技術キーワード 5-10個]
```

経営方針テキストに言及されている具体的な技術分野、新規事業領域、DX施策、成長戦略に基づいて推定する。テンプレート的な記述ではなく、企業固有の内容を反映する。

### Step 3: ベクトル化

**スクリプト**: `scripts/regenerate_vectors.py`

生成テキストを `intfloat/multilingual-e5-small`（384次元）でエンコードし、`needs_vector` として保存する。

## フォールバック（v1: テンプレートベース）

**スクリプト**: `scripts/generate_needs.py`

有報テキストが取得できなかった企業（43社、1.1%）には、以下の3要素を組み合わせたテンプレートベースのニーズテキストを適用する。

1. **業種テンプレート**: 主要業種ごとの典型的技術ニーズキーワード
2. **R&D特性修飾**: R&D比率に応じた3段階の文脈
3. **規模修飾**: 売上高に応じた3段階の文脈

### 業種テンプレート一覧

| 業種 | 主要ニーズキーワード |
|---|---|
| 情報・通信業 | AI・機械学習、サイバーセキュリティ、IoT、量子コンピューティング |
| 電気機器 | パワー半導体、センサー、ロボティクス、全固体電池、6G |
| 化学 | 機能性材料、触媒、バイオプラスチック、マテリアルズインフォマティクス |
| 医薬品 | ADC、核酸医薬、AI創薬、再生医療、デジタルセラピューティクス |
| 機械 | スマートファクトリー、予知保全、3Dプリンティング、水素関連機器 |
| 輸送用機器 | 電動化、自動運転、車載半導体、全固体電池、MaaS |
| 食料品 | フードテック、代替タンパク質、発酵バイオ、ニュートリゲノミクス |
| 建設業 | BIM/CIM、ロボット施工、ZEB/ZEH、スマートシティ |
| 小売業 | EC、需要予測AI、無人店舗、サプライチェーン最適化 |
| 卸売業 | SCM高度化、B2Bマーケットプレイス、倉庫自動化 |
| サービス業 | 顧客分析、RPA、需要予測、ヘルスケアテック、HR Tech |

上記以外の業種にはデフォルトテンプレート（DX、AI、サステナビリティ等）が適用される。

## ニーズ領域カテゴリ（ダッシュボード用）

ニーズテキストから以下の11カテゴリにタグ付けされる（`app.py` の `NEEDS_DOMAINS`）:

1. AI・データ
2. 半導体・電子部品
3. エネルギー・環境
4. バイオ・ヘルスケア
5. モビリティ
6. ロボティクス・製造
7. 素材・化学
8. 通信・ネットワーク
9. DX・ソフトウェア
10. 建設・インフラ
11. 食・農業

## カバレッジ

| 項目 | 値 |
|---|---|
| 全社数 | 3,872社 |
| v2 個社別ニーズ | 3,829社（98.9%） |
| v1 テンプレート（フォールバック） | 43社（1.1%） |
| ベクトル化率 | 100%（384次元） |
| LLMモデル | Claude Haiku 4.5 |
| EDINETスキャン期間 | 2024-04-01 〜 2026-04-01 |
| 産学連携レコード | 約3,000件（CiNii Research） |

## 改善ロードマップ

### v3: マルチソース統合（将来構想）

- **特許出願パターン分析**: Google Patents BigQueryから企業のIPC分類推移を取得し、新規参入分野を特定
- **ニュース・プレスリリース解析**: 産学連携ニュース、プレスリリースからリアルタイムのニーズ変化を検出
- **ユーザーフィードバック学習**: マッチング結果へのフィードバックでニーズ精度を継続改善
- **研究開発活動セクション統合**: 有報の「研究開発活動」テキストも活用し、より詳細な技術領域を特定

## 再生成手順

```bash
cd ~/projects/apps/sangaku-matcher

# Step 1: EDINET有報テキスト取得（新規企業分のみ）
python3 scripts/fetch_edinet_strategy.py --start-date 2024-04-01 --end-date 2026-04-01

# Step 2: Claude APIでv2ニーズ生成（未処理企業分のみ）
python3 scripts/generate_needs_v2.py --batch-size 50

# Step 3: ベクトル再生成（全社）
python3 scripts/regenerate_vectors.py --batch-size 200

# Step 4: コミット・プッシュ
git add data/matcher.db
git commit -m "data: regenerate v2 needs"
git push
```

## 関連ファイル

| ファイル | 役割 |
|---|---|
| `scripts/fetch_edinet_strategy.py` | EDINET API v2から有報テキスト取得 |
| `scripts/generate_needs_v2.py` | Claude APIによる個社別ニーズ生成 |
| `scripts/generate_needs.py` | v1テンプレートベース生成（フォールバック） |
| `scripts/regenerate_vectors.py` | ニーズベクトル一括再生成 |
| `scripts/enrich_collaborations.py` | 産学連携実績データ収集（CiNii Research） |
| `src/sangaku_matcher/scoring/need_fit.py` | need_fitスコアラー |
| `src/sangaku_matcher/web/app.py` | ニーズダッシュボード（`NEEDS_DOMAINS`定義） |

## 更新履歴

- **2026-04-11 v2**: EDINET API v2から3,829社の有報経営方針テキストを取得し、Claude Haiku 4.5で個社別ニーズを生成。カバレッジ98.9%。
- **2026-04-10 v1**: テンプレートベースの初期生成。3,872社全社にニーズデータを付与。CiNii Researchから産学連携実績データを収集。
