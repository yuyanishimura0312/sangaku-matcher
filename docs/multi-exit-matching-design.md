# 多出口型マッチングアルゴリズム設計メモ

## ステータス: Stage 0 検討中（2026-04-12）

## 背景・課題

現在のアルゴリズムは7次元価値モデルの重み付き線形結合で単一スコアを算出し、事後的にif-elseで連携形態を推定する構造。以下の構造的問題がある:

1. **単一スコアへの圧縮**: 7次元の情報が1つの数値に潰れ、多面性が失われる
2. **事後的なモード推論**: ランキング後にモードを推定 — 連携形態が「おまけ」
3. **出口が研究者視点で設計されていない**: 研究者が実際に必要としているのは「何について話すべきか」「どういう関係を築けるか」
4. **重み付けが固定**: 人類学者とAI研究者で同じ重み

## 提案: 多出口型マッチングアーキテクチャ

単一ランキングではなく、**連携形態ごとに独立したスコアリングモデル**を持つ。

### 連携形態（出口）の定義案

| 出口 | 名称 | 意味 | 主に使う次元 |
|------|------|------|-------------|
| A | 共同研究パートナー | 特定テーマで共同研究を組む | need_fit + tech_prox + abs_cap + past_ties |
| B | 知見パートナー | 研究知見を企業の意思決定に活かす（委員・アドバイザー・研修講師） | humanities_fit + テーマ一致 + open_inno |
| C | フィールド提供者 | 企業現場が研究のフィールドになる（調査協力・データ提供） | テーマ一致 + 業界特性 + past_ties |
| D | 社会実装パートナー | 研究成果の社会実装を共に推進する | need_fit + abs_cap + open_inno + ambition一致 |
| E | 探索的対話相手 | まだ形のない新領域を一緒に探索する | future_option + ambition一致 + humanities_fit |
| F | 資金提供者 | 研究費の出し手として（受託研究・寄附講座） | abs_cap + past_ties + テーマ一致 |

### 出口別スコアリング（重み配合の例）

```python
EXIT_WEIGHTS = {
    "joint_research":    {"tech_prox": 0.25, "need_fit": 0.25, "abs_cap": 0.20, "past_ties": 0.15, "open_inno": 0.10, "humanities_fit": 0.05},
    "knowledge_partner": {"tech_prox": 0.05, "need_fit": 0.10, "abs_cap": 0.05, "past_ties": 0.10, "open_inno": 0.15, "humanities_fit": 0.40, "theme_match": 0.15},
    "field_provider":    {"tech_prox": 0.05, "need_fit": 0.10, "abs_cap": 0.05, "past_ties": 0.20, "industry_fit": 0.30, "humanities_fit": 0.15, "theme_match": 0.15},
    "social_implement":  {"tech_prox": 0.10, "need_fit": 0.20, "abs_cap": 0.20, "past_ties": 0.05, "open_inno": 0.15, "humanities_fit": 0.10, "ambition_fit": 0.20},
    "exploratory":       {"tech_prox": 0.05, "need_fit": 0.05, "abs_cap": 0.05, "past_ties": 0.05, "open_inno": 0.10, "humanities_fit": 0.20, "future_option": 0.25, "ambition_fit": 0.25},
    "funding":           {"tech_prox": 0.10, "need_fit": 0.15, "abs_cap": 0.30, "past_ties": 0.25, "open_inno": 0.10, "humanities_fit": 0.05, "theme_match": 0.05},
}
```

### 処理フロー

```
研究者入力（テーマ・専門・関心）
        │
        ▼
研究者プロファイル構築（122テーマへの親和度 + 研究者タイプ推定）
        │
        ▼
出口別スコアリング（6つの独立モデル）
        │
        ▼
企業ごとの「最適連携ポートフォリオ」構築
        │
        ▼
表示: 出口別トップ + 総合ポートフォリオ
```

### 新しいスコアリング次元の候補

- **industry_fit**: 研究テーマと企業の業界の関連性（フィールド提供で重要）
- **ambition_fit**: 野心領域テーマとの一致度（探索・社会実装で重要）
- **theme_match**: 122テーマの多軸一致度（テーマ横断的な親和性）

### 研究者タイプの自動推定

122テーマへの親和度パターンから推定し、出口の優先度を調整:
- 人文系テーマ > 技術テーマ → 知見パートナー・フィールド提供を優先表示
- 技術テーマ > 人文系 → 共同研究・社会実装を優先
- 野心テーマへの親和度が高い → 探索的対話を強調

### 「フィールド提供者」の重要性

人類学・社会科学の研究者にとって、企業は「連携先」であると同時に「研究対象・フィールド」。現在のモデルに完全に欠けている視点。業界分類とナラティブ内容から「この企業の現場が研究にとって興味深い」を推定するスコアが必要。

### 表示方法: ランキングからポートフォリオへ

- **出口別セクション**: 「知見パートナーとして推奨する企業」「共同研究候補」など
- **企業カード**: 各企業で「何が最も適切な連携形態か」をレーダーチャートで表示
- **ポートフォリオビュー**: 「あなたの研究にとっての連携戦略」

## 未決定事項

1. 出口の6分類は妥当か？追加・統合したいものはあるか？
2. 研究者タイプの自動推定は必要か、それとも研究者が自分で「知見パートナーを探している」と指定する方がよいか？
3. 表示の優先度として、出口別セクション表示と企業ごとのポートフォリオ表示、どちらが先か？

## 現在のコード参照

- `src/sangaku_matcher/matcher.py` — 現在の7次元マッチャー（run_match）
- `src/sangaku_matcher/theme_matcher.py` — 122テーマ仮説マッチャー（run_theme_match）
- `src/sangaku_matcher/config.py` — 重み設定（Settings）
- `src/sangaku_matcher/scoring/theme_fit.py` — GTA/テーマスコアラー
- `src/sangaku_matcher/scoring/need_fit.py` — ニーズ適合スコアラー
- `src/sangaku_matcher/scoring/humanities_fit.py` — 人文系適合スコアラー
- `data/theme_taxonomy.json` — 人文33テーマ
- `data/tech_taxonomy.json` — 技術50テーマ
- `data/ambition_taxonomy.json` — 野心40テーマ（人文知の接続付き）
