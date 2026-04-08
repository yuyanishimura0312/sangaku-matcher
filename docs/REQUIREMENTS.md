# 産学マッチャー（sangaku-matcher）開発要件書

**バージョン**: v0.1
**作成日**: 2026-04-08
**位置付け**: SPEC.md を補完する開発実装の詳細要件

---

## 1. アーキテクチャ概要

```
┌────────────────────────────────────────────────────────────┐
│  CLI / Web UI (Phase 3)                                    │
└─────────────────────────┬──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│  Application Layer                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Seed Parser  │  │ Matcher      │  │ Reporter     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────┬──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│  Scoring Engine                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ TechProx │ │ AbsCap   │ │ NeedFit  │ │ PastTies │ ...   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────┬──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│  Data Layer                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ SQLite DB    │  │ Embeddings   │  │ Cache        │      │
│  │ (companies,  │  │ (sentence-   │  │ (LLM出力)    │      │
│  │  patents,    │  │  transformers│  │              │      │
│  │  matches)    │  │  vectors)    │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────┬──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│  Data Acquisition Layer (バッチ)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │IR Collector│ J-PlatPat │ │Semantic  │ │ Claude   │      │
│  │ (既存)   │ │ scraper  │ │ Scholar  │ │ API      │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└────────────────────────────────────────────────────────────┘
```

---

## 2. ディレクトリ構造

```
sangaku-matcher/
├── README.md
├── docs/
│   ├── SPEC.md                  # 仕様書
│   ├── REQUIREMENTS.md          # 本書
│   ├── ARCHITECTURE.md          # アーキテクチャ詳細（Phase 1で作成）
│   └── API.md                   # 内部API仕様（Phase 1で作成）
├── src/
│   └── sangaku_matcher/
│       ├── __init__.py
│       ├── cli.py               # Click CLI エントリポイント
│       ├── config.py            # 設定管理
│       ├── db.py                # SQLite アクセス層
│       ├── seeds.py             # シーズ入力パーサ
│       ├── reporter.py          # 出力レポート生成
│       ├── matcher.py           # マッチング統括
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── tech_prox.py
│       │   ├── abs_cap.py
│       │   ├── need_fit.py      # Phase 2
│       │   ├── past_ties.py
│       │   ├── trl_compat.py    # Phase 3
│       │   └── open_inno.py     # Phase 3
│       ├── acquisition/
│       │   ├── __init__.py
│       │   ├── edinet_loader.py # IR Collector連携
│       │   ├── jplatpat.py
│       │   ├── semantic_scholar.py
│       │   └── ir_reports.py    # 統合報告書
│       └── llm/
│           ├── __init__.py
│           └── claude_client.py # Claude API クライアント
├── data/
│   ├── matcher.db               # SQLite (gitignore)
│   ├── embeddings/              # ベクトルキャッシュ (gitignore)
│   └── seed_corpus/             # サンプルシーズ
├── scripts/
│   ├── init_db.py
│   ├── load_companies.py        # 対象4000社の初期登録（IR Collector連携）
│   ├── monthly_batch.sh         # 月次更新バッチ
│   └── nightly_match.sh         # 夜間バッチマッチング
├── tests/
│   ├── test_scoring/
│   ├── test_acquisition/
│   └── test_e2e/
├── .env.example
├── .gitignore
├── pyproject.toml               # 依存管理
└── requirements.txt             # 互換性のため
```

---

## 3. データソース別の取得仕様

### 3.1 EDINET API（IR Collector経由）

**取得対象**:
- 有価証券報告書（書類種別 120）の最新版
- 統合報告書（公開している企業のみ）

**取得項目**:
- [研究開発活動] セクション全文
- [従業員の状況] セクション
- 主要な経営指標（売上高、営業利益、純資産）

**実装方針**:
- IR Collectorのストレージ（PDF/XBRL）から該当セクションを抽出
- XBRLパーサーは `arelle` を使用予定（Phase 1ではPDFテキスト抽出で代替可）
- 月次バッチで全対象企業を更新

### 3.2 J-PlatPat（特許情報）

**取得対象**:
- 対象4,000社が出願人となっている特許（Phase 1では手動登録の小規模データのみ。Phase 3で本格連携）
- 大学（特に主要研究大学）との共同出願特許

**取得項目**:
- IPCコード（4桁分類）
- 出願人・発明者
- 出願日・公開日

**実装方針**:
- 公式API（特許情報利用支援サイト）の利用を第一選択
- 利用不可の場合は「特許情報標準データ」の一括ダウンロード（年1回更新）
- スクレイピングは規約上避ける
- 月次バッチで増分更新

### 3.3 Semantic Scholar API

**取得対象**:
- 入力シーズの論文（DOI指定時）
- 大学研究者と企業所属研究者の共著論文

**取得項目**:
- タイトル・抄録
- 著者所属
- 引用数
- フィールド（分野）

**実装方針**:
- 既存の `mcp__semantic-scholar__*` ツールまたは公式 API クライアント
- レート制限（100 req/sec、APIキー無しは1 req/sec）に注意

### 3.4 Claude API（バックキャスト推定）

**用途**: 企業の中期計画・統合報告書テキストから潜在ニーズを推定

**プロンプト設計の方針**（Phase 2で詳細化）:
```
あなたは産学連携アドバイザー。以下の企業の公開情報から、
今後3〜5年で必要とする可能性が高い技術領域を5つ推定してください。
各候補について、根拠となるテキスト箇所を引用し、確度を高/中/低で示してください。

【企業】{企業名}・{業種}
【中期計画】{中期計画テキスト}
【研究開発活動】{有報R&Dセクション}

出力フォーマット: JSON
```

**モデル選択**: Claude Sonnet 4.6（コストと精度のバランス）

**コスト管理**: 月次バッチでの推定に上限額を設定（例: $30/月）。超過時は警告。

---

## 4. 主要コンポーネントの詳細要件

### 4.1 Seed Parser (`seeds.py`)

**入力**: 自由記述テキスト / DOI / 特許番号 / これらの混在

**処理**:
1. 入力形式の自動判定
2. DOIの場合: Semantic Scholar APIで論文取得 → 抄録抽出
3. 特許番号の場合: J-PlatPatで特許取得 → クレーム・要約抽出
4. テキストの場合: そのまま使用
5. 統合された記述テキストから IPCコード推定（特許分類器、Phase 1では簡易的に）
6. Sentence-BERT エンベディング生成

**出力**: `Seed` データクラス
```python
@dataclass
class Seed:
    seed_id: str
    title: str
    description: str
    doi: Optional[str]
    patent_no: Optional[str]
    ipc_codes: list[str]
    semantic_vector: np.ndarray
    source_type: str  # 'text' | 'doi' | 'patent' | 'mixed'
```

### 4.2 Scoring Engine (`scoring/`)

各特徴量モジュールは以下のインターフェースを実装する。

```python
class FeatureScorer(Protocol):
    name: str
    weight: float

    def score(self, seed: Seed, company: Company) -> tuple[float, str]:
        """Returns (score in [0,1], rationale text)"""
```

これにより特徴量の追加・削除が容易になる。`config.py` で有効化する特徴量と重みを管理する。

### 4.3 Matcher (`matcher.py`)

**入力**: Seed、対象企業リスト

**処理**:
1. 全企業についてスコアリング実行（並列化、最大10並列）
2. スコア降順で並べ替え
3. 上位N社を抽出
4. 各社について推奨連携モードを判定
5. `Match` オブジェクトをDBに保存

**並列化**: `concurrent.futures.ThreadPoolExecutor`、Claude APIへのリクエストは別途レート制御

### 4.4 Reporter (`reporter.py`)

**出力フォーマット**:

**Markdownレポート**（人間向け）:
```markdown
# マッチング結果: {シーズタイトル}

実行日時: {datetime}
対象企業数: {N}社
処理時間: {秒}

## 入力シーズ
{description}

## トップ10候補

### 1. {企業名} (スコア: 0.78)

| 特徴量 | スコア | 重み |
|---|---|---|
| 技術的近接性 | 0.85 | 0.20 |
| 吸収能力 | 0.72 | 0.20 |
| ニーズ適合度 | 0.81 | 0.30 |
| 過去連携実績 | 0.40 | 0.10 |
| TRL適合 | 1.00 | 0.10 |
| OI成熟度 | 0.60 | 0.10 |

**根拠説明**:
{LLMまたはテンプレートで生成された3〜5文の説明}

**推奨連携モード**: 共同研究

---
（以下、同様に2位〜10位）
```

**JSON出力**（プログラム連携向け）:
```json
{
  "seed": {...},
  "executed_at": "...",
  "matches": [
    {
      "rank": 1,
      "edinet_code": "...",
      "company_name": "...",
      "total_score": 0.78,
      "feature_scores": {...},
      "rationale": "...",
      "recommended_mode": "共同研究"
    },
    ...
  ]
}
```

### 4.5 CLI (`cli.py`)

**コマンド**:
```bash
# 単一シーズマッチング
sangaku-matcher match --seed seed.md --top 10 --output result.md

# バッチマッチング
sangaku-matcher batch --input-dir seeds/ --output-dir results/

# 企業データ更新
sangaku-matcher update-companies --source edinet,jplatpat

# バックキャスト推定（月次）
sangaku-matcher refresh-needs --limit 500

# DB初期化
sangaku-matcher init-db
```

---

## 5. 技術的な依存関係

### 5.1 Python パッケージ

```toml
[project]
dependencies = [
    "click>=8.1",
    "python-dotenv>=1.0",
    "requests>=2.31",
    "anthropic>=0.40",        # Claude API SDK
    "sentence-transformers>=3.0",
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "pydantic>=2.5",
    "tenacity>=8.2",          # リトライ
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff", "mypy"]
xbrl = ["arelle-release"]    # Phase 1後半
web = ["fastapi", "uvicorn"] # Phase 3
```

### 5.2 外部システム依存

- **IR Collector** (`~/projects/apps/ir-collector/`): EDINETデータの取得・保管
  - 連携方法: SQLite DBファイルを直接読み取り、ストレージディレクトリ構造を共有
  - もしくは IR Collectorをパッケージ化して `pip install -e` で取り込む（推奨）

### 5.3 ハードウェア要件

- **開発環境**: MacBook Air M2、メモリ16GB以上推奨
- **ストレージ**: 10GB以上（EDINETデータ + 特許メタデータ + 埋め込みキャッシュ）
- **ネットワーク**: バッチ実行時に各種APIアクセスが集中する

---

## 6. 開発プロセス（7段階モデルに沿った段取り）

### Stage 0: 仕様策定 ← 現在地
- SPEC.md / REQUIREMENTS.md（本書）の作成
- 西村のレビュー・承認

### Stage 1: 調査・設計
- ARCHITECTURE.md の詳細化
- データモデルのER図作成
- 主要モジュールのインターフェース設計
- テストケースの定義（test-first）
- IR Collector との統合方法の決定

### Stage 2: 実装（Phase 1 MVP）
- リポジトリ初期化、GitHubプライベートリポジトリ作成
- データ取得層（EDINET連携）
- スコアリング3特徴量（TechProx, AbsCap, PastTies）
- CLI: `match` コマンド
- ユニットテスト＋スモークテスト
- セキュリティスキャン（bandit）

### Stage 3: QA
- functional / UI(CLI) / security の3面レビュー
- 5件のテストシーズで実マッチング → 西村レビュー

### Stage 4: 改善
- レビュー指摘の修正

### Stage 5: 承認・デプロイ
- v1.0 リリースタグ
- README整備
- Notion登録

### Stage 6: 観測・学習
- 実利用ログの蓄積
- フィードバックを Phase 2 仕様に反映

---

## 7. 運用要件

### 7.1 環境変数

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
EDINET_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...   # 任意（無くても動作）

# パス
MATCHER_DB_PATH=./data/matcher.db
IR_COLLECTOR_DB_PATH=~/projects/apps/ir-collector/data/ir.db
IR_COLLECTOR_STORAGE_PATH=~/projects/apps/ir-collector/storage

# モデル
EMBEDDING_MODEL=intfloat/multilingual-e5-small
LLM_MODEL=claude-sonnet-4-6

# コスト管理
LLM_MONTHLY_BUDGET_USD=30
```

### 7.2 cron 設定例

```cron
# 月次: 企業データ更新（毎月1日 03:00）
0 3 1 * * cd ~/projects/apps/sangaku-matcher && ./scripts/monthly_batch.sh

# 月次: バックキャスト推定再生成（毎月2日 03:00）
0 3 2 * * cd ~/projects/apps/sangaku-matcher && python -m sangaku_matcher.cli refresh-needs

# 夜間: バッチマッチング（毎日 23:00、シーズが置かれていれば実行）
0 23 * * * cd ~/projects/apps/sangaku-matcher && ./scripts/nightly_match.sh
```

### 7.3 ログ・監視

- 全バッチ処理は `logs/` 配下にタイムスタンプ付きログを残す
- エラー発生時は標準エラー出力に書き、終了コード1で抜ける
- 将来的にSlack通知連携（Phase 3）

---

## 8. テスト要件

### 8.1 ユニットテスト

- 各特徴量モジュール（scoring/）に対して、既知の入出力ペアでのテスト
- データパーサ（seeds.py）に対して、各入力形式のパーステスト
- DBアクセス層のCRUD動作確認

### 8.2 統合テスト

- ダミー企業3社・ダミーシーズ2件で end-to-end のスモークテスト
- 出力Markdownが期待構造を持つこと
- スコアが [0, 1] 範囲に収まること

### 8.3 受け入れテスト（西村実施）

- Phase 1完了時: 5件の実シーズでマッチング実行 → 「妥当な候補が含まれているか」を5段階評価
- Phase 2完了時: 同じ5件でPhase 1出力との差分を確認 → 「より妥当か」を評価

### 8.4 カバレッジ目標

- ユニットテスト: 主要モジュールで70%以上
- 完璧主義は求めない（個人プロジェクトであるため）

---

## 9. ドキュメント要件

各 Phase 完了時に以下を最新化する。

- `README.md`: インストール、使い方、サンプル出力
- `docs/ARCHITECTURE.md`: 全体構造、データフロー、シーケンス図（簡易）
- `docs/API.md`: 主要モジュールのpublic interface
- `docs/CHANGELOG.md`: 変更履歴
- 各モジュール: docstring必須、複雑なロジックには「why」コメント

---

## 10. 決定事項（Stage 0 確定、2026-04-08）

| # | 項目 | 決定 |
|---|---|---|
| 1 | 対象企業 | EDINET有報提出の全約4,000社（業種絞らない） |
| 2 | 特許情報 | Phase 1では簡略化（共同出願実績の手動登録のみ）。J-PlatPat本格連携はPhase 3以降 |
| 3 | 対象大学 | 絞り込みなし |
| 4 | 利用範囲 | 当面は西村本人のみ。クライアント共有なし、法務レビュー不要 |
| 5 | UI形態 | Phase 1からWeb UI（FastAPI）含む |
| 6 | LLM予算 | 月$30前後を上限。優先度別Tier方式（Tier1=Sonnet月次、Tier2=Sonnet四半期、Tier3=Haiku半年）で月$33程度に収める |
| 7 | IR Collector連携 | 内部依存として連結（pip install -e） |

これらの決定はSPEC.md §2.4 と本書 §11 に反映されている。

## 11. 4,000社対応の追加要件

対象が500社→4,000社に拡大したことで、以下の追加考慮が必要。

### 11.1 ストレージ
- 有報PDFのストレージ: 4,000社 × 平均5MB = 約20GB（年次1回更新前提）
- XBRLデータ: 約10GB
- メタデータDB: 約500MB
- 埋め込みベクトル: 4,000社 × 768次元 × 4byte = 約12MB（無視できる）
- **合計: 約30〜40GB**。Mac内蔵SSDで十分

### 11.2 性能
- 4,000社の全件スコアリング: 1シーズあたり目標5分以内
- 簡略な特徴量（TechProx, AbsCap, PastTies）なら全件 in-memory で計算可能
- バックキャスト推定（Phase 2）はキャッシュ前提なので実行時負荷に影響しない

### 11.3 Tier分類ロジック
- Tier 1（500社）: 研究開発費の絶対額上位
- Tier 2（1,500社）: R&D集約度（売上比）または絶対額の501-2000位
- Tier 3（残り約2,000社）: それ以外
- 分類は四半期ごとに見直し
- DB上に `companies.tier` カラムを持つ

### 11.4 初回バックフィル
- 4,000社の有報初回取得: IR Collector経由で1日程度
- 4,000社のSonnetによる初回バックキャスト推定: 約$80、約8時間
- いずれも一度きりのコストとして容認

### 11.5 Web UI 最小要件（Phase 1）
- FastAPI + Jinja2テンプレート（フロントエンドフレームワークなし）
- ページ:
  - `/` シーズ入力フォーム
  - `/result/{seed_id}` マッチング結果表示
  - `/companies` 対象企業一覧（検索可）
- 認証なし（西村のローカル環境のみで動作）
- ブートストラップCSS等で最低限の見た目

実装はStage 1（調査・設計）でアーキテクチャ詳細を詰めた後にStage 2に進む。
