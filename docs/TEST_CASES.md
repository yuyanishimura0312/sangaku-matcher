# Test Cases

## 1. Test Seeds (5 cases for acceptance testing)

These seeds represent real-world scenarios. Expected outputs are approximate — the acceptance criterion is "plausible top-10 list," not exact rank order.

### Seed 1: Honeycomb Porous Film (材料科学)

**Input**:
```
Title: 自己組織化ハニカム構造多孔質フィルム
Description: 水滴の自己組織化現象を利用し、マイクロメートルスケールの
均一なハニカム構造を持つ多孔質高分子フィルムを簡便に製造する技術。
細胞培養基材、センサー、光学フィルター等への応用が期待される。
Source type: text
```

**Expected top candidates** (industries: 化学, 精密機器, 医療機器):
- 富士フイルム (E00513) — ハニカムフィルムの共同研究実績あり
- 日東電工, 東レ, 三菱ケミカル, JSR — 高分子フィルム領域
- テルモ, ニプロ — 細胞培養・医療機器用途
- 信越化学 — 素材

**Why this seed**: tests cross-industry matching (chemistry + medical devices) and the inverted-U proximity logic.

---

### Seed 2: Perovskite Solar Cell (エネルギー)

**Input**:
```
Title: 高効率ペロブスカイト太陽電池の大面積成膜技術
Description: 塗布法によりペロブスカイト太陽電池の大面積（30cm角以上）
均一成膜を実現する技術。変換効率22%以上、耐久性1000時間以上を達成。
低コスト・軽量・フレキシブルな次世代太陽電池として期待。
Source type: text
```

**Expected top candidates** (industries: 電機, ガラス・セラミックス, 化学):
- パナソニック, 京セラ — 太陽電池事業
- AGC, 日本板硝子 — ガラス基板
- 積水化学 — ペロブスカイト太陽電池開発公表済
- 三菱ケミカル, 東レ — 有機材料

**Why this seed**: tests energy/sustainability domain, expects companies with explicit solar/green-energy R&D mentions.

---

### Seed 3: NLP for Medical Records (IT × 医療)

**Input**:
```
Title: 日本語医療カルテの自然言語処理による構造化技術
Description: 電子カルテの非構造化テキスト（日本語）から、疾患名、
薬剤名、検査値を自動抽出し、構造化データに変換する
自然言語処理モデル。医療ビッグデータ解析の前処理に有用。
DOI: 10.xxxx/example (dummy — for testing DOI flow)
Source type: mixed
```

**Expected top candidates** (industries: 情報通信, 医療機器, 製薬):
- 富士通, NEC, NTTデータ — 医療IT
- エムスリー — 医療情報プラットフォーム
- 第一三共, 武田薬品, アステラス — RWD活用
- PHC (旧パナソニックヘルスケア) — 電子カルテ

**Why this seed**: tests IT-medical cross-domain, where TechProx alone is insufficient and NeedFit (Phase 2) would add value.

---

### Seed 4: Carbon Fiber Recycling (環境 × 素材)

**Input**:
```
Title: 炭素繊維強化プラスチック（CFRP）のケミカルリサイクル技術
Description: 超臨界水を用いてCFRPから炭素繊維を99%以上の回収率で
再生する技術。航空機・自動車の廃材から高品質リサイクル炭素繊維を得る。
Source type: text
```

**Expected top candidates** (industries: 繊維, 自動車, 化学):
- 東レ — 炭素繊維トップメーカー
- 帝人 — 炭素繊維事業
- 三菱ケミカル — 複合材料
- トヨタ, ホンダ — CFRP使用増加
- 三菱重工, IHI — 航空機構造材

**Why this seed**: tests sustainability/circular economy keywords matching against corporate midterm plans.

---

### Seed 5: Quantum Sensing (量子技術)

**Input**:
```
Title: ダイヤモンドNVセンタを用いた高感度磁気センサ
Description: ダイヤモンド中の窒素-空孔(NV)センタの量子特性を利用した
室温動作の高感度磁気センサ。医療診断（心磁図）、
非破壊検査、地質探査への応用が期待される。
Source type: text
```

**Expected top candidates** (industries: 電機, 精密, 医療):
- 浜松ホトニクス — 光学センサ
- キーサイト（日本法人は非上場, フィルタされるべき）
- 日立製作所 — 量子技術全般
- 横河電機, 島津製作所 — 計測機器
- キヤノン — イメージング

**Why this seed**: tests niche/frontier technology. Expect fewer high-score matches, validating that the system doesn't force-fit.

---

## 2. Unit Test Scope

### 2.1 Scoring Modules

| Module | Test Cases |
|---|---|
| `tech_prox.py` | (a) Identical vectors → raw sim=1.0, after f(x) transform → 0.0 (too close). (b) Orthogonal vectors → 0.0. (c) Medium similarity (0.5) → maximum score (1.0 from f(x)=4*0.5*0.5). (d) None vector → returns 0.0 with rationale. |
| `abs_cap.py` | (a) Top R&D intensity in pharma → high score. (b) Zero R&D → 0.0. (c) Industry normalization: pharma 10% vs auto 4% both map to similar z-scores. (d) Missing industry → falls back to global stats. |
| `past_ties.py` | (a) 10 joint patents → high score. (b) Zero collaborations → 0.0 (not penalized in final score due to low weight). (c) University name partial match ("東大" matches "東京大学"). |

### 2.2 Seed Parser

| Test Case | Input | Expected |
|---|---|---|
| Plain text | 200-char description | Seed with source_type='text', vector shape (384,) |
| DOI | "10.1234/test" | Fetches from Semantic Scholar mock, sets doi field |
| Patent | "特開2024-123456" | Sets patent_no field |
| Empty text | "" | Raises ValueError |
| Overlong text | 50,000 chars | Truncated to 8,000 chars before embedding |

### 2.3 Matcher

| Test Case | Expected |
|---|---|
| Normal match with 3 mock companies | Returns 3 ranked results, scores in [0,1], sorted desc |
| All companies score 0.0 | Returns top N with 0.0 scores, does not crash |
| Single company in DB | Returns 1 result |

### 2.4 Reporter

| Test Case | Expected |
|---|---|
| Markdown output | Contains "# Match Result", company table, rationale sections |
| JSON output | Valid JSON, matches the MatchResult schema |
| Empty result | Markdown says "No matches found" |

### 2.5 DB Operations

| Test Case | Expected |
|---|---|
| init_db on fresh path | Creates all tables without error |
| Upsert company twice | Second call updates, no duplicate |
| Store/load vector roundtrip | numpy array survives BLOB storage |

## 3. Integration / Smoke Tests

### 3.1 End-to-End Smoke Test

**Preconditions**: SQLite DB with at least 10 mock companies pre-loaded.

**Steps**:
1. Call `run_match(seed)` with Seed 1 (honeycomb film)
2. Assert `len(result.rankings) == 10`
3. Assert all scores in `[0.0, 1.0]`
4. Assert result can be serialized to both Markdown and JSON
5. Assert match rows are saved to DB

### 3.2 Web UI Smoke Test

1. Start FastAPI dev server
2. GET `/` returns 200 with input form
3. POST `/match` with text returns 200 with result page
4. GET `/companies` returns 200 with company list
5. GET `/result/{seed_id}` for a saved match returns 200

## 4. Acceptance Criteria (for Nishimura)

After Phase 1 completion, Nishimura will:

1. Run all 5 test seeds through the Web UI
2. For each seed, evaluate: "Are the top 10 candidates plausible?"
3. Rating scale:
   - **A**: 7+ of top 10 are relevant industries/companies
   - **B**: 4-6 relevant
   - **C**: <4 relevant (needs redesign)
4. Phase 1 passes if all 5 seeds score B or above, with at least 2 scoring A
5. Specific attention to: (a) cross-industry seeds (#1, #3) producing diverse results, (b) niche seed (#5) not over-fitting
