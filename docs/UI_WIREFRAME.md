# UI Wireframe & Route Design

## 1. URL Design

| Path | Method | Description |
|---|---|---|
| `/` | GET | Seed input form (home page) |
| `/match` | POST | Execute matching, redirect to result |
| `/result/{seed_id}` | GET | Display match result for a seed |
| `/results` | GET | List of past match results |
| `/companies` | GET | Company database browser |
| `/company/{edinet_code}` | GET | Single company detail |
| `/api/match` | POST | JSON API for programmatic access |
| `/api/result/{seed_id}` | GET | JSON API for result retrieval |

## 2. Screen Wireframes

### Screen 1: Seed Input (`/`)

```
┌─────────────────────────────────────────────────────────┐
│  sangaku-matcher                              [結果一覧] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  技術シーズを入力                                        │
│                                                         │
│  タイトル                                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  技術の説明 *                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │                                                   │  │
│  │                                                   │  │
│  │  (200〜2000字のテキスト)                            │  │
│  │                                                   │  │
│  │                                                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  論文DOI (任意)           特許番号 (任意)               │
│  ┌────────────────────┐  ┌────────────────────┐        │
│  │ 10.xxxx/xxxxx      │  │ 特開2024-123456    │        │
│  └────────────────────┘  └────────────────────┘        │
│                                                         │
│  表示件数: [10 ▼]                                       │
│                                                         │
│           [ マッチング実行 ]                              │
│                                                         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│  対象: 約4,000社 | DB更新: 2026-04-08                    │
└─────────────────────────────────────────────────────────┘
```

**Key interactions**:
- "技術の説明" is required (min 200 chars), others optional
- Submit triggers POST `/match` with form data
- Loading state: "マッチング実行中... (約4,000社を評価しています)" spinner
- On completion: redirect to `/result/{seed_id}`

---

### Screen 2: Match Result (`/result/{seed_id}`)

```
┌─────────────────────────────────────────────────────────┐
│  sangaku-matcher                     [新規入力] [結果一覧]│
├─────────────────────────────────────────────────────────┤
│                                                         │
│  マッチング結果                                          │
│  ───────────                                            │
│  シーズ: 自己組織化ハニカム構造多孔質フィルム              │
│  実行日時: 2026-04-08 15:30 | 処理時間: 1.2秒            │
│  対象企業: 4,012社 | スコアリング: TechProx + AbsCap +    │
│  PastTies                                               │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │ # │ 企業名        │業種    │総合  │技術 │吸収 │実績 ││
│  │───┼───────────────┼────────┼──────┼─────┼─────┼─────││
│  │ 1 │ 富士フイルム   │化学    │ 0.82 │0.91 │0.78 │0.72 ││
│  │ 2 │ 東レ          │繊維    │ 0.76 │0.85 │0.72 │0.65 ││
│  │ 3 │ 日東電工      │化学    │ 0.74 │0.88 │0.69 │0.58 ││
│  │ ...                                                 ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  --- 1位: 富士フイルム (E00513) ---                       │
│                                                         │
│  総合スコア: 0.82                                        │
│                                                         │
│  | 特徴量     | スコア | 重み  | 寄与   |                │
│  |-----------|--------|-------|--------|                 │
│  | 技術的近接性| 0.91   | 0.35  | 0.318  |               │
│  | 吸収能力   | 0.78   | 0.35  | 0.273  |               │
│  | 過去実績   | 0.72   | 0.30  | 0.216  |               │
│                                                         │
│  根拠:                                                   │
│  有報の研究開発活動セクションにおいて「高分子フィルム」     │
│  「バイオマテリアル」「ヘルスケア素材」のキーワードが確認   │
│  され、入力シーズの技術領域と高い類似度を示す。            │
│  R&D集約度7.2%は化学業種の上位に位置し、外部技術を         │
│  受容する能力が高い。東北大学との共同特許が5件存在する。    │
│                                                         │
│  推奨連携モード: 共同研究                                 │
│                                                         │
│  --- 2位: 東レ (E00855) ---                               │
│  (同様の詳細展開)                                        │
│                                                         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│  [ Markdownダウンロード ]  [ JSONダウンロード ]            │
│                                                         │
│  注: 本結果はAIによる仮説であり、最終的な連携判断は        │
│  専門家の評価に基づいて行ってください。                     │
└─────────────────────────────────────────────────────────┘
```

**Key interactions**:
- Top table shows ranked list, clickable to scroll to detail
- Each company detail shows score breakdown table + rationale text
- Download buttons for Markdown (.md) and JSON (.json) files
- Disclaimer footer on every result page

---

### Screen 3: Company Browser (`/companies`)

```
┌─────────────────────────────────────────────────────────┐
│  sangaku-matcher                     [新規入力] [結果一覧]│
├─────────────────────────────────────────────────────────┤
│                                                         │
│  企業データベース (4,012社)                               │
│                                                         │
│  検索: [________________] 業種: [全業種 ▼]  [検索]       │
│                                                         │
│  並び順: [R&D費 ▼]   [昇順/降順]                         │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐│
│  │ 企業名       │EDINET  │業種  │売上高  │R&D費 │R&D比 ││
│  │──────────────┼────────┼──────┼────────┼──────┼──────││
│  │ トヨタ自動車  │E02144  │輸送  │45兆   │1.2兆 │ 2.7% ││
│  │ ソニーG      │E01777  │電機  │13兆   │7,500 │ 5.8% ││
│  │ 武田薬品     │E00919  │医薬  │4.3兆  │5,400 │12.6% ││
│  │ ...                                                 ││
│  └─────────────────────────────────────────────────────┘│
│                                                         │
│  [< 前へ]  1 / 81  [次へ >]   (50件/ページ)              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key interactions**:
- Full-text search by company name
- Filter by industry dropdown
- Sort by any numeric column
- Click company name → `/company/{edinet_code}` detail page
- Pagination: 50 companies per page

---

### Screen 3b: Company Detail (`/company/{edinet_code}`)

```
┌─────────────────────────────────────────────────────────┐
│  sangaku-matcher                                 [戻る]  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  武田薬品工業株式会社                                     │
│  ───────────────                                        │
│  EDINET: E00919 | 証券: 4502 | 業種: 医薬品              │
│                                                         │
│  財務指標                                                │
│  売上高: 4,302,706百万円 | 営業利益: 321,046百万円         │
│  R&D費: 541,820百万円 | R&D比率: 12.6%                    │
│  従業員数: 49,095名                                      │
│                                                         │
│  吸収能力スコア: 0.92                                     │
│  OI成熟度スコア: 0.80 (CVC:有, 専担組織:有, 大学連携:有)   │
│                                                         │
│  研究開発活動 (有報より抜粋)                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 当社グループは、消化器系疾患、希少疾患、...         │  │
│  │ (最大500文字表示、全文は展開ボタン)                  │  │
│  └───────────────────────────────────────────────────┘  │
│  [全文を表示]                                            │
│                                                         │
│  推定潜在ニーズ (AI推定・仮説)          推定日: 未実施    │
│  ┌───────────────────────────────────────────────────┐  │
│  │ (Phase 2で表示)                                    │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  大学連携実績                                            │
│  - 東京大学: 共同特許 3件, 共著論文 8件                   │
│  - 京都大学: 共同特許 2件                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 3. FastAPI Route Definitions

```python
# sangaku_matcher/web/app.py

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="sangaku-matcher", version="0.1.0")
templates = Jinja2Templates(directory="src/sangaku_matcher/web/templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Seed input form."""

@app.post("/match", response_class=HTMLResponse)
async def match(
    request: Request,
    description: str = Form(...),
    title: str = Form(""),
    doi: str = Form(""),
    patent_no: str = Form(""),
    top_n: int = Form(10),
):
    """Run matching and redirect to result page."""

@app.get("/result/{seed_id}", response_class=HTMLResponse)
async def result(request: Request, seed_id: str):
    """Display match result."""

@app.get("/result/{seed_id}/download/{format}")
async def download_result(seed_id: str, format: str):
    """Download result as .md or .json."""

@app.get("/results", response_class=HTMLResponse)
async def results_list(request: Request, page: int = 1):
    """List past match results."""

@app.get("/companies", response_class=HTMLResponse)
async def companies(
    request: Request,
    q: str = "",
    industry: str = "",
    sort: str = "rd_expense",
    order: str = "desc",
    page: int = 1,
):
    """Browse company database."""

@app.get("/company/{edinet_code}", response_class=HTMLResponse)
async def company_detail(request: Request, edinet_code: str):
    """Single company detail page."""

# JSON API
@app.post("/api/match")
async def api_match(payload: SeedInput) -> MatchResultJSON:
    """Programmatic matching endpoint."""

@app.get("/api/result/{seed_id}")
async def api_result(seed_id: str) -> MatchResultJSON:
    """Get match result as JSON."""
```

## 4. Template Files

```
src/sangaku_matcher/web/templates/
├── base.html          # Layout with nav, footer, Bootstrap CSS CDN
├── home.html          # Seed input form
├── result.html        # Match result display
├── results_list.html  # Past results list
├── companies.html     # Company browser with pagination
└── company.html       # Single company detail
```

## 5. Styling

- Bootstrap 5 via CDN (no build step needed)
- Minimal custom CSS
- Favicon: esse-sense black-e icon (per project standard)
- Color scheme: neutral grays, blue accent for scores
- Responsive: works on desktop and tablet (mobile is low priority for Phase 1)

## 6. Static Assets

```
src/sangaku_matcher/web/static/
├── favicon.ico        # From ~/press-release-generator/favicon.ico
└── style.css          # Minimal overrides (~50 lines)
```
