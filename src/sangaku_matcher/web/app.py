"""FastAPI web application for sangaku-matcher."""
from __future__ import annotations

import json
import math
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sangaku_matcher.config import settings
from sangaku_matcher.db import connect

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="sangaku-matcher", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _company_count() -> int:
    try:
        with connect(settings.matcher_db_path) as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM companies").fetchone()
            return row["c"]
    except Exception:
        return 0


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {
        "company_count": _company_count(),
    })


@app.post("/match", response_class=HTMLResponse)
async def match(
    request: Request,
    description: str = Form(...),
    title: str = Form(""),
    doi: str = Form(""),
    patent_no: str = Form(""),
    top_n: int = Form(10),
):
    from sangaku_matcher.seeds import parse_seed
    from sangaku_matcher.matcher import run_multi_exit_match

    # Clamp top_n to a safe range to prevent excessive queries or zero-result runs
    top_n = max(1, min(top_n, 50))

    try:
        seed = parse_seed(
            description=description,
            title=title or description[:60],
            doi=doi or None,
            patent_no=patent_no or None,
        )
        result = run_multi_exit_match(seed, top_n=top_n)
    except ValueError as e:
        return templates.TemplateResponse(request, "home.html", {
            "company_count": _company_count(),
            "error": str(e),
        })
    except Exception:
        return templates.TemplateResponse(request, "home.html", {
            "company_count": _company_count(),
            "error": "マッチング処理中にエラーが発生しました。入力内容を確認してください。",
        })

    return templates.TemplateResponse(request, "multi_exit_result.html", {
        "result": result,
        "seed": seed,
    })


@app.post("/hypothesis", response_class=HTMLResponse)
async def hypothesis_match(
    request: Request,
    description: str = Form(...),
    top_n: int = Form(10),
):
    """Redirect legacy /hypothesis to multi-exit /match."""
    from sangaku_matcher.seeds import parse_seed
    from sangaku_matcher.matcher import run_multi_exit_match

    top_n = max(1, min(top_n, 50))
    try:
        seed = parse_seed(
            description=description,
            title=description[:60],
        )
        result = run_multi_exit_match(seed, top_n=top_n)
    except ValueError as e:
        return templates.TemplateResponse(request, "home.html", {
            "company_count": _company_count(),
            "error": str(e),
        })
    except Exception:
        return templates.TemplateResponse(request, "home.html", {
            "company_count": _company_count(),
            "error": "マッチング処理中にエラーが発生しました。",
        })

    return templates.TemplateResponse(request, "multi_exit_result.html", {
        "result": result,
        "seed": seed,
    })


@app.get("/hypothesis", response_class=HTMLResponse)
async def hypothesis_get(request: Request):
    """Redirect GET /hypothesis to home page."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/", status_code=302)


def _load_match_result(seed_id: str):
    """Load a saved match result from DB. Returns (seed, result) or (None, None)."""
    from sangaku_matcher.matcher import (
        MatchResult, RankedCompany, _generate_hypotheses, _infer_mode,
    )
    from sangaku_matcher.scoring import FeatureResult
    from sangaku_matcher.seeds import Seed
    import numpy as np

    with connect(settings.matcher_db_path) as conn:
        seed_row = conn.execute("SELECT * FROM seeds WHERE seed_id = ?", (seed_id,)).fetchone()
        if not seed_row:
            return None, None

        match_rows = conn.execute(
            "SELECT m.*, c.name, c.industry FROM matches m "
            "JOIN companies c ON m.edinet_code = c.edinet_code "
            "WHERE m.seed_id = ? ORDER BY m.rank",
            (seed_id,),
        ).fetchall()

    seed = Seed(
        seed_id=seed_row["seed_id"],
        title=seed_row["title"],
        description=seed_row["description"],
        doi=seed_row.get("doi"),
        patent_no=seed_row.get("patent_no"),
        semantic_vector=np.zeros(384),
        source_type=seed_row["source_type"],
        created_at=seed_row["created_at"],
    )

    rankings = []
    for r in match_rows:
        fs = {}
        # Map DB columns to scorer names (trl_compat stores future_option)
        col_to_scorer = {
            "tech_prox": "tech_prox",
            "need_fit": "need_fit",
            "abs_cap": "abs_cap",
            "past_ties": "past_ties",
            "trl_compat": "future_option",
            "open_inno_mat": "open_inno",
            "humanities_fit": "humanities_fit",
        }
        for col, scorer_name in col_to_scorer.items():
            val = r.get(col)
            if val is not None:
                fs[scorer_name] = FeatureResult(val, "")
        mode = r["recommended_mode"] or "joint_research"
        hypotheses, overall_comment = _generate_hypotheses(
            seed, r["name"], r["industry"] or "",
            r["total_score"], fs, mode,
        )
        rankings.append(RankedCompany(
            rank=r["rank"],
            edinet_code=r["edinet_code"],
            company_name=r["name"],
            industry=r["industry"] or "",
            total_score=r["total_score"],
            feature_scores=fs,
            recommended_mode=mode,
            collaboration_hypotheses=hypotheses,
            overall_comment=overall_comment,
        ))

    result = MatchResult(
        seed=seed,
        rankings=rankings,
        executed_at=match_rows[0]["created_at"] if match_rows else "",
        company_count=_company_count(),
    )
    return seed, result


def _load_multi_exit_result(seed_id: str):
    """Load a saved multi-exit match result from DB. Returns MultiExitMatchResult or None."""
    from sangaku_matcher.matcher import (
        MultiExitMatchResult, ExitRanking, RankedCompany,
    )
    from sangaku_matcher.scoring import FeatureResult
    from sangaku_matcher.seeds import Seed
    from sangaku_matcher.config import EXIT_LABELS, EXIT_DESCRIPTIONS
    import numpy as np

    with connect(settings.matcher_db_path) as conn:
        # Check if multi_exit_matches table exists
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='multi_exit_matches'"
        ).fetchone()
        if not table_check:
            return None

        seed_row = conn.execute("SELECT * FROM seeds WHERE seed_id = ?", (seed_id,)).fetchone()
        if not seed_row:
            return None

        mex_rows = conn.execute(
            """SELECT m.*, c.name, c.industry FROM multi_exit_matches m
               JOIN companies c ON m.edinet_code = c.edinet_code
               WHERE m.seed_id = ? ORDER BY m.exit_type, m.rank""",
            (seed_id,),
        ).fetchall()

        if not mex_rows:
            return None

    # Convert Row to dict for safe .get() access
    seed_dict = dict(seed_row)
    seed = Seed(
        seed_id=seed_dict["seed_id"],
        title=seed_dict["title"],
        description=seed_dict["description"],
        doi=seed_dict.get("doi"),
        patent_no=seed_dict.get("patent_no"),
        semantic_vector=np.zeros(384),
        source_type=seed_dict["source_type"],
        created_at=seed_dict["created_at"],
    )

    # Group rows by exit_type
    from collections import defaultdict
    grouped: dict[str, list] = defaultdict(list)
    for r in mex_rows:
        grouped[r["exit_type"]].append(r)

    exits: list[ExitRanking] = []
    # Maintain stable order: rd, new_domain, exploratory
    for exit_type in ("rd", "new_domain", "exploratory"):
        rows = grouped.get(exit_type, [])
        if not rows:
            continue

        rankings = []
        for r in rows:
            # Reconstruct feature_scores from DB columns
            score_cols = {
                "tech_prox": r["tech_prox"],
                "need_fit": r["need_fit"],
                "abs_cap": r["abs_cap"],
                "past_ties": r["past_ties"],
                "future_option": r["future_option"],
                "open_inno": r["open_inno"],
                "humanities_fit": r["humanities_fit"],
                "ambition_fit": r["ambition_fit"],
                "theme_breadth": r["theme_breadth"],
                "synergy": r["synergy"],
            }
            fs = {}
            for col, val in score_cols.items():
                if val is not None:
                    fs[col] = FeatureResult(val, "")

            rankings.append(RankedCompany(
                rank=r["rank"],
                edinet_code=r["edinet_code"],
                company_name=r["name"],
                industry=r["industry"] or "",
                total_score=r["total_score"],
                feature_scores=fs,
                recommended_mode=exit_type,
                overall_comment=r["hypothesis"] or "",
            ))

        exits.append(ExitRanking(
            exit_type=exit_type,
            exit_label=EXIT_LABELS.get(exit_type, exit_type),
            exit_description=EXIT_DESCRIPTIONS.get(exit_type, ""),
            rankings=rankings,
        ))

    return MultiExitMatchResult(
        seed=seed,
        exits=exits,
        executed_at=mex_rows[0]["created_at"] if mex_rows else "",
        company_count=_company_count(),
    )


@app.get("/result/{seed_id}", response_class=HTMLResponse)
async def result_page(request: Request, seed_id: str):
    # Try multi-exit first
    multi_result = _load_multi_exit_result(seed_id)
    if multi_result:
        return templates.TemplateResponse(request, "multi_exit_result.html", {
            "result": multi_result,
            "seed": multi_result.seed,
        })
    # Fall back to legacy single-ranking
    seed, result = _load_match_result(seed_id)
    if seed is None:
        return HTMLResponse("Seed not found", status_code=404)

    return templates.TemplateResponse(request, "result.html", {
        "result": result,
        "seed": seed,
    })


@app.get("/result/{seed_id}/download/{fmt}")
async def download_result(seed_id: str, fmt: str):
    """Download match result as Markdown or JSON."""
    from sangaku_matcher.reporter import to_markdown, to_json

    seed, result = _load_match_result(seed_id)
    if seed is None:
        return Response("Not found", status_code=404)

    if fmt == "json":
        return JSONResponse(to_json(result),
                           headers={"Content-Disposition": f"attachment; filename=match_{seed_id[:8]}.json"})
    md = to_markdown(result)
    return Response(md, media_type="text/markdown",
                   headers={"Content-Disposition": f"attachment; filename=match_{seed_id[:8]}.md"})


@app.get("/results", response_class=HTMLResponse)
async def results_list(request: Request, page: int = 1):
    per_page = 20
    offset = (page - 1) * per_page
    with connect(settings.matcher_db_path) as conn:
        # Check if multi_exit_matches table exists
        has_mex = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='multi_exit_matches'"
        ).fetchone() is not None

        # Count total seeds that have any match results (legacy or multi-exit)
        if has_mex:
            total = conn.execute(
                """SELECT COUNT(DISTINCT s.seed_id) as c FROM seeds s
                   WHERE EXISTS (SELECT 1 FROM matches WHERE seed_id = s.seed_id)
                      OR EXISTS (SELECT 1 FROM multi_exit_matches WHERE seed_id = s.seed_id)"""
            ).fetchone()["c"]

            rows = conn.execute(
                """SELECT s.seed_id, s.title, s.created_at,
                          COALESCE(
                              (SELECT COUNT(*) FROM multi_exit_matches WHERE seed_id = s.seed_id),
                              0
                          ) + COALESCE(
                              (SELECT COUNT(*) FROM matches WHERE seed_id = s.seed_id),
                              0
                          ) as match_count,
                          COALESCE(
                              (SELECT MAX(total_score) FROM multi_exit_matches WHERE seed_id = s.seed_id),
                              (SELECT MAX(total_score) FROM matches WHERE seed_id = s.seed_id)
                          ) as top_score,
                          CASE WHEN EXISTS (SELECT 1 FROM multi_exit_matches WHERE seed_id = s.seed_id)
                               THEN 1 ELSE 0 END as is_multi_exit
                   FROM seeds s
                   WHERE EXISTS (SELECT 1 FROM matches WHERE seed_id = s.seed_id)
                      OR EXISTS (SELECT 1 FROM multi_exit_matches WHERE seed_id = s.seed_id)
                   ORDER BY s.created_at DESC
                   LIMIT ? OFFSET ?""",
                (per_page, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(DISTINCT seed_id) as c FROM matches").fetchone()["c"]
            rows = conn.execute(
                """SELECT s.seed_id, s.title, s.created_at, COUNT(m.match_id) as match_count,
                          MAX(m.total_score) as top_score, 0 as is_multi_exit
                   FROM seeds s LEFT JOIN matches m ON s.seed_id = m.seed_id
                   GROUP BY s.seed_id ORDER BY s.created_at DESC
                   LIMIT ? OFFSET ?""",
                (per_page, offset),
            ).fetchall()

    total_pages = max(1, math.ceil(total / per_page))
    return templates.TemplateResponse(request, "results_list.html", {
        "results": rows, "page": page, "total_pages": total_pages,
    })


import re as _re
from collections import defaultdict as _defaultdict

# Needs domain categories for grouping
NEEDS_DOMAINS = {
    "AI・データ": ["AI", "機械学習", "データ分析", "自然言語処理", "大規模言語モデル", "深層学習", "推論チップ", "エッジAI", "AI創薬", "AI画像", "需要予測", "最適化アルゴリズム", "パーソナライゼーション"],
    "半導体・電子部品": ["半導体", "SiC", "GaN", "MEMS", "フォトニクス", "光半導体", "高周波", "パワー半導体", "センサ", "電子部品"],
    "エネルギー・環境": ["電池", "蓄電", "水素", "カーボンニュートラル", "CO2", "省エネ", "エネルギー", "再生可能", "太陽光", "グリーン"],
    "バイオ・ヘルスケア": ["バイオ", "医薬", "医療", "ヘルスケア", "iPS", "再生医療", "遺伝子", "抗体", "核酸", "ゲノム", "診断", "細胞"],
    "モビリティ": ["自動運転", "電動化", "EV", "ADAS", "車載", "MaaS", "モビリティ", "FCEV", "BEV"],
    "ロボティクス・製造": ["ロボティクス", "ロボット", "自動化", "3Dプリン", "AM技術", "スマートファクトリー", "FA", "加工", "製造DX"],
    "素材・化学": ["材料", "触媒", "ポリマー", "ナノテク", "コーティング", "繊維", "セラミック", "金属", "高分子", "複合材"],
    "通信・ネットワーク": ["5G", "6G", "通信", "IoT", "ネットワーク", "V2X", "コネクティッド", "ブロックチェーン"],
    "DX・ソフトウェア": ["DX", "クラウド", "SaaS", "RPA", "業務自動化", "デジタル", "サイバーセキュリティ", "フィンテック"],
    "建設・インフラ": ["BIM", "建設", "インフラ", "ZEB", "スマートシティ", "施工", "コンクリート", "防災", "レジリエンス"],
    "食・農業": ["フードテック", "食品", "発酵", "農業", "代替タンパク", "鮮度"],
}


def _extract_domains(needs_text: str) -> list[str]:
    """Extract matching needs domains from a needs text."""
    if not needs_text:
        return []
    domains = []
    for domain, keywords in NEEDS_DOMAINS.items():
        for kw in keywords:
            if kw in needs_text:
                domains.append(domain)
                break
    return domains


@app.get("/needs", response_class=HTMLResponse)
async def needs_page(
    request: Request,
    q: str = "",
    industry: str = "",
    domain: str = "",
    sort: str = "rd_expense",
    view: str = "dashboard",
    page: int = 1,
):
    per_page = 30 if view == "list" else 500
    offset = (page - 1) * per_page if view == "list" else 0
    allowed_sorts = {"name", "rd_expense", "rd_intensity"}
    if sort not in allowed_sorts:
        sort = "rd_expense"
    order_sql = "DESC" if sort != "name" else "ASC"

    conditions = ["estimated_needs IS NOT NULL AND LENGTH(estimated_needs) > 0"]
    params: list = []
    if q:
        conditions.append("(name LIKE ? OR estimated_needs LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    if industry:
        conditions.append("industry = ?")
        params.append(industry)
    if domain and domain in NEEDS_DOMAINS:
        # Filter by any keyword in the domain
        kw_conditions = [f"estimated_needs LIKE ?" for _ in NEEDS_DOMAINS[domain][:5]]
        conditions.append(f"({' OR '.join(kw_conditions)})")
        params.extend([f"%{kw}%" for kw in NEEDS_DOMAINS[domain][:5]])

    where = "WHERE " + " AND ".join(conditions)

    with connect(settings.matcher_db_path) as conn:
        total = conn.execute(f"SELECT COUNT(*) as c FROM companies {where}", params).fetchone()["c"]
        total_all = conn.execute("SELECT COUNT(*) as c FROM companies").fetchone()["c"]
        with_needs = conn.execute(
            "SELECT COUNT(*) as c FROM companies WHERE estimated_needs IS NOT NULL AND LENGTH(estimated_needs) > 0"
        ).fetchone()["c"]

        rows = conn.execute(
            f"SELECT edinet_code, name, industry, rd_expense, rd_intensity, estimated_needs "
            f"FROM companies {where} ORDER BY {sort} {order_sql} LIMIT ? OFFSET ?",
            params + [per_page, offset],
        ).fetchall()
        industries_rows = conn.execute(
            "SELECT DISTINCT industry FROM companies WHERE industry IS NOT NULL ORDER BY industry"
        ).fetchall()

        # Industry summary for grouped + dashboard views
        industry_summary = []
        if view in ("industry", "dashboard"):
            industry_summary = conn.execute(
                """SELECT industry, COUNT(*) as cnt, AVG(rd_expense) as avg_rd
                   FROM companies
                   WHERE estimated_needs IS NOT NULL AND LENGTH(estimated_needs) > 0
                     AND industry IS NOT NULL AND industry != ''
                   GROUP BY industry ORDER BY cnt DESC"""
            ).fetchall()

        # Collaboration stats for dashboard
        collab_stats = None
        collab_top = []
        if view == "dashboard":
            cs = conn.execute(
                "SELECT COUNT(*) as total_records, COUNT(DISTINCT edinet_code) as companies_with FROM collaborations"
            ).fetchone()
            collab_stats = {"total_records": cs["total_records"], "companies_with": cs["companies_with"]}
            collab_top_rows = conn.execute(
                """SELECT c.edinet_code, c.name, c.industry, c.estimated_needs,
                          COUNT(DISTINCT cl.university_name) as uni_count,
                          SUM(cl.count) as total_papers
                   FROM collaborations cl JOIN companies c ON cl.edinet_code = c.edinet_code
                   GROUP BY cl.edinet_code ORDER BY total_papers DESC LIMIT 10"""
            ).fetchall()
            collab_top = []
            for r in collab_top_rows:
                ct = dict(r)
                ct["domains"] = _extract_domains(ct.get("estimated_needs", ""))
                collab_top.append(ct)

    # Domain summary for domain + dashboard views
    domain_summary = []
    if view in ("domain", "dashboard"):
        domain_counts: dict[str, int] = {}
        with connect(settings.matcher_db_path) as conn:
            all_needs = conn.execute(
                "SELECT estimated_needs FROM companies WHERE estimated_needs IS NOT NULL"
            ).fetchall()
        for row in all_needs:
            for d in _extract_domains(row["estimated_needs"]):
                domain_counts[d] = domain_counts.get(d, 0) + 1
        domain_summary = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)

    # Attach domains to each company for display
    companies_with_domains = []
    for r in rows:
        co = dict(r)
        co["domains"] = _extract_domains(co.get("estimated_needs", ""))
        companies_with_domains.append(co)

    # Group companies by industry if view == "industry"
    grouped_by_industry = {}
    if view == "industry":
        for co in companies_with_domains:
            ind = co.get("industry") or "その他"
            grouped_by_industry.setdefault(ind, []).append(co)

    total_pages = max(1, math.ceil(total / per_page)) if view == "list" else 1
    return templates.TemplateResponse(request, "needs.html", {
        "companies": companies_with_domains,
        "industries": [r["industry"] for r in industries_rows],
        "q": q, "industry": industry, "domain": domain, "sort": sort, "view": view,
        "page": page, "total_pages": total_pages, "total": total, "total_all": total_all,
        "with_needs": with_needs,
        "industry_summary": industry_summary,
        "domain_summary": domain_summary,
        "grouped_by_industry": grouped_by_industry,
        "all_domains": list(NEEDS_DOMAINS.keys()),
        "all_domain_keywords": NEEDS_DOMAINS,
        "collab_stats": collab_stats,
        "collab_top": collab_top,
    })


@app.get("/needs/about", response_class=HTMLResponse)
async def needs_about(request: Request):
    with connect(settings.matcher_db_path) as conn:
        company_count = conn.execute("SELECT COUNT(*) as c FROM companies").fetchone()["c"]
        v2_count = conn.execute(
            "SELECT COUNT(*) as c FROM companies "
            "WHERE midterm_plan_text IS NOT NULL AND LENGTH(midterm_plan_text) > 100"
        ).fetchone()["c"]
        v1_count = company_count - v2_count
        collab_count = conn.execute(
            "SELECT COUNT(*) as c FROM collaborations"
        ).fetchone()["c"]
    v2_pct = round(v2_count / company_count * 100, 1) if company_count else 0
    return templates.TemplateResponse(request, "needs_about.html", {
        "company_count": company_count,
        "v2_count": v2_count,
        "v1_count": v1_count,
        "v2_pct": v2_pct,
        "collab_count": collab_count,
        "domain_keywords": NEEDS_DOMAINS,
    })


@app.get("/themes", response_class=HTMLResponse)
async def themes_page(request: Request):
    """Display humanities theme taxonomy and business ambition taxonomy."""
    with connect(settings.matcher_db_path) as conn:
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

        # Humanities themes (33-theme taxonomy)
        themes_data = []
        if "taxonomy_themes" in tables and "company_taxonomy_proximity" in tables:
            rows = conn.execute("""
                SELECT t.theme_id, t.major_id, t.major_name, t.name, t.keywords,
                       ROUND(AVG(p.proximity), 3) as avg_prox,
                       ROUND(MAX(p.proximity), 3) as max_prox,
                       SUM(CASE WHEN p.narrative_count > 0 THEN 1 ELSE 0 END) as match_count
                FROM taxonomy_themes t
                JOIN company_taxonomy_proximity p ON t.theme_id = p.theme_id
                GROUP BY t.theme_id
                ORDER BY t.major_id, t.theme_id
            """).fetchall()
            for r in rows:
                d = dict(r)
                try:
                    d["parsed_keywords"] = json.loads(d.get("keywords", "[]"))
                except (json.JSONDecodeError, TypeError):
                    d["parsed_keywords"] = []
                themes_data.append(d)

        # Tech taxonomy (from JSON file)
        tech_taxonomy_data = []
        tech_path = settings.matcher_db_path.parent / "tech_taxonomy.json"
        if tech_path.exists():
            with open(tech_path) as f:
                tech_taxonomy_data = json.load(f).get("themes", [])

        tech_needs_count = conn.execute(
            "SELECT COUNT(*) as c FROM companies WHERE tech_needs_at IS NOT NULL"
        ).fetchone()["c"]

        # Business ambition taxonomy (from JSON file)
        ambition_data = []
        ambition_path = settings.matcher_db_path.parent / "ambition_taxonomy.json"
        if ambition_path.exists():
            with open(ambition_path) as f:
                ambition_data = json.load(f).get("themes", [])

        # Ambition stats
        ambition_count = conn.execute(
            "SELECT COUNT(*) as c FROM companies WHERE ambitions_at IS NOT NULL"
        ).fetchone()["c"]
        ambition_avg = 0
        if ambition_count > 0:
            row = conn.execute(
                "SELECT ROUND(AVG(json_array_length(json_extract(ambitions_json, '$.ambitions'))), 1) as avg "
                "FROM companies WHERE ambitions_json IS NOT NULL"
            ).fetchone()
            ambition_avg = row["avg"] if row else 0

        # Maturity distribution
        maturity_dist = {"concrete": 0, "directional": 0, "exploratory": 0}
        mat_rows = conn.execute(
            "SELECT ambitions_json FROM companies WHERE ambitions_json IS NOT NULL"
        ).fetchall()
        for mr in mat_rows:
            try:
                data = json.loads(mr["ambitions_json"])
                for amb in data.get("ambitions", []):
                    m = amb.get("maturity", "")
                    if m in maturity_dist:
                        maturity_dist[m] += 1
            except (json.JSONDecodeError, TypeError):
                pass

        company_count = conn.execute("SELECT COUNT(*) as c FROM companies").fetchone()["c"]
    return templates.TemplateResponse(request, "themes.html", {
        "themes": themes_data,
        "tech_themes": tech_taxonomy_data,
        "tech_needs_count": tech_needs_count,
        "ambition_themes": ambition_data,
        "ambition_count": ambition_count,
        "ambition_avg": ambition_avg,
        "maturity_dist": maturity_dist,
        "company_count": company_count,
    })


@app.get("/companies", response_class=HTMLResponse)
async def companies_page(
    request: Request,
    q: str = "",
    industry: str = "",
    sort: str = "rd_expense",
    order: str = "desc",
    page: int = 1,
):
    per_page = 50
    offset = (page - 1) * per_page
    allowed_sorts = {"name", "industry", "revenue", "rd_expense", "rd_intensity"}
    if sort not in allowed_sorts:
        sort = "rd_expense"
    order_sql = "DESC" if order == "desc" else "ASC"

    conditions = []
    params: list = []
    if q:
        conditions.append("name LIKE ?")
        params.append(f"%{q}%")
    if industry:
        conditions.append("industry = ?")
        params.append(industry)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    with connect(settings.matcher_db_path) as conn:
        total = conn.execute(f"SELECT COUNT(*) as c FROM companies {where}", params).fetchone()["c"]
        rows = conn.execute(
            f"SELECT edinet_code, sec_code, name, industry, revenue, rd_expense, rd_intensity "
            f"FROM companies {where} ORDER BY {sort} {order_sql} LIMIT ? OFFSET ?",
            params + [per_page, offset],
        ).fetchall()
        industries = conn.execute(
            "SELECT DISTINCT industry FROM companies WHERE industry IS NOT NULL ORDER BY industry"
        ).fetchall()

    total_pages = max(1, math.ceil(total / per_page))
    return templates.TemplateResponse(request, "companies.html", {
        "companies": rows, "industries": [r["industry"] for r in industries],
        "q": q, "industry": industry, "sort": sort, "order": order,
        "page": page, "total_pages": total_pages, "total": total,
    })


@app.get("/company/{edinet_code}", response_class=HTMLResponse)
async def company_detail(request: Request, edinet_code: str):
    with connect(settings.matcher_db_path) as conn:
        co = conn.execute("SELECT * FROM companies WHERE edinet_code = ?", (edinet_code,)).fetchone()
        if not co:
            return HTMLResponse("Company not found", status_code=404)
        collabs = conn.execute(
            "SELECT * FROM collaborations WHERE edinet_code = ? ORDER BY count DESC",
            (edinet_code,),
        ).fetchall()
    return templates.TemplateResponse(request, "company.html", {
        "company": co, "collaborations": collabs,
    })


# JSON API
@app.post("/api/match")
async def api_match(payload: dict):
    from sangaku_matcher.seeds import parse_seed
    from sangaku_matcher.matcher import run_multi_exit_match

    # Validate required field
    description = payload.get("description", "")
    if not description or not description.strip():
        return JSONResponse(
            {"error": "description is required and must not be empty"},
            status_code=400,
        )

    # Clamp top_n to a safe range (same rule as HTML endpoint)
    # Guard against non-numeric values by falling back to default 10
    try:
        top_n = int(payload.get("top_n", 3))
    except (TypeError, ValueError):
        top_n = 3
    top_n = max(1, min(top_n, 50))

    try:
        seed = parse_seed(
            description=description,
            title=payload.get("title", ""),
            doi=payload.get("doi"),
            patent_no=payload.get("patent_no"),
        )
        result = run_multi_exit_match(seed, top_n=top_n)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception:
        return JSONResponse(
            {"error": "マッチング処理中にエラーが発生しました。入力内容を確認してください。"},
            status_code=500,
        )

    # Serialize multi-exit result to JSON
    return _multi_exit_to_json(result)


def _multi_exit_to_json(result) -> dict:
    """Serialize MultiExitMatchResult to JSON-compatible dict."""
    data = {
        "seed": {
            "seed_id": result.seed.seed_id,
            "title": result.seed.title,
            "description": result.seed.description[:500],
            "doi": result.seed.doi,
            "patent_no": result.seed.patent_no,
            "source_type": result.seed.source_type,
        },
        "executed_at": result.executed_at,
        "duration_sec": result.duration_sec,
        "company_count": result.company_count,
        "exits": [
            {
                "exit_type": ex.exit_type,
                "exit_label": ex.exit_label,
                "exit_description": ex.exit_description,
                "rankings": [
                    {
                        "rank": rc.rank,
                        "edinet_code": rc.edinet_code,
                        "company_name": rc.company_name,
                        "industry": rc.industry,
                        "total_score": rc.total_score,
                        "feature_scores": {
                            k: {"value": v.value, "rationale": v.rationale}
                            for k, v in rc.feature_scores.items()
                        },
                        "hypothesis": rc.overall_comment,
                    }
                    for rc in ex.rankings
                ],
            }
            for ex in result.exits
        ],
    }
    # Include researcher profile if available (Stokes 1997 classification)
    rp = getattr(result, "researcher_profile", None)
    if rp:
        data["researcher_profile"] = {
            "researcher_type": rp.researcher_type,
            "type_label": rp.type_label,
            "recommended_exit": rp.recommended_exit,
            "recommendation_text": rp.recommendation_text,
            "tech_affinity": round(rp.tech_affinity, 4),
            "humanities_affinity": round(rp.humanities_affinity, 4),
            "ambition_affinity": round(rp.ambition_affinity, 4),
        }
    return data
