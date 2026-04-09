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
    from sangaku_matcher.matcher import run_match

    try:
        seed = parse_seed(
            description=description,
            title=title or description[:60],
            doi=doi or None,
            patent_no=patent_no or None,
        )
        result = run_match(seed, top_n=top_n)
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

    return templates.TemplateResponse(request, "result.html", {
        "result": result,
        "seed": seed,
    })


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


@app.get("/result/{seed_id}", response_class=HTMLResponse)
async def result_page(request: Request, seed_id: str):
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
        total = conn.execute("SELECT COUNT(DISTINCT seed_id) as c FROM matches").fetchone()["c"]
        rows = conn.execute(
            """SELECT s.seed_id, s.title, s.created_at, COUNT(m.match_id) as match_count,
                      MAX(m.total_score) as top_score
               FROM seeds s LEFT JOIN matches m ON s.seed_id = m.seed_id
               GROUP BY s.seed_id ORDER BY s.created_at DESC
               LIMIT ? OFFSET ?""",
            (per_page, offset),
        ).fetchall()
    total_pages = max(1, math.ceil(total / per_page))
    return templates.TemplateResponse(request, "results_list.html", {
        "results": rows, "page": page, "total_pages": total_pages,
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
    from sangaku_matcher.matcher import run_match
    from sangaku_matcher.reporter import to_json

    seed = parse_seed(
        description=payload.get("description", ""),
        title=payload.get("title", ""),
        doi=payload.get("doi"),
        patent_no=payload.get("patent_no"),
    )
    result = run_match(seed, top_n=payload.get("top_n", 10))
    return to_json(result)
