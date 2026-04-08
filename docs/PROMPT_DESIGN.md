# Prompt Design for Backcast Need Estimation

## 1. Overview

The backcast estimation module (Phase 2) uses Claude API to read a company's
R&D activity section and midterm plan, then infer latent technology needs
that the company is likely to pursue in the next 3-5 years. This document
defines the prompt template, output schema, and token management strategy.

## 2. Prompt Template

```
You are an expert industry analyst specializing in Japanese corporate R&D strategy
and university-industry collaboration. Your task is to infer the latent technology
needs of a company based on its public disclosures.

## Company Profile
- Name: {company_name}
- Industry: {industry}
- Revenue: {revenue}百万円
- R&D Expense: {rd_expense}百万円 (R&D Intensity: {rd_intensity:.1%})

## R&D Activity Section (from 有価証券報告書)
{rd_text_truncated}

## Midterm Plan / Integrated Report Excerpts
{midterm_plan_text_truncated}

## Instructions
Based on the above information, identify 5 technology areas that this company
is likely to need from external sources (universities, research institutes)
in the next 3-5 years.

For each technology need:
1. State the technology area concisely (1 sentence)
2. Explain why this company needs it (2-3 sentences, citing specific text)
3. Assess confidence: "high" (explicitly mentioned in R&D plan),
   "medium" (implied by strategy), or "low" (inferred from industry trends)

Respond in JSON format only. Do not include markdown code fences.
```

## 3. Output JSON Schema

```json
{
  "company_name": "string",
  "edinet_code": "string",
  "generated_at": "ISO 8601 string",
  "technology_needs": [
    {
      "area": "string — concise technology area name",
      "rationale": "string — 2-3 sentences explaining why",
      "confidence": "high | medium | low",
      "source_quote": "string — exact quote from input text that supports this"
    }
  ]
}
```

Example output:

```json
{
  "company_name": "富士フイルム株式会社",
  "edinet_code": "E00513",
  "generated_at": "2026-04-10T03:00:00+09:00",
  "technology_needs": [
    {
      "area": "再生医療向けバイオマテリアル",
      "rationale": "統合報告書でヘルスケア領域を重点成長事業と位置付けており、iPS細胞関連の投資を拡大中。自社の高分子フィルム技術と組み合わせた細胞培養基材の研究開発需要が高い。",
      "confidence": "high",
      "source_quote": "ヘルスケア分野における再生医療領域への投資を加速"
    },
    {
      "area": "AI画像診断アルゴリズム",
      "rationale": "医療IT事業でAI診断支援の開発を進めているが、画像認識の精度向上には外部の深層学習研究との連携が必要と推察される。",
      "confidence": "medium",
      "source_quote": "AI技術を活用した医療画像解析ソリューションの開発"
    }
  ]
}
```

## 4. Token Management

### 4.1 Input Budget per Company

| Component | Max Tokens | Strategy |
|---|---|---|
| System prompt + instructions | ~400 | Fixed |
| Company profile | ~100 | Fixed |
| R&D section text | ~3,000 | Truncate at 3,000 tokens (≈6,000 chars JP) |
| Midterm plan text | ~1,500 | Truncate at 1,500 tokens (≈3,000 chars JP) |
| **Total input** | **~5,000** | |
| **Output** | **~1,000** | 5 needs × ~200 tokens each |

### 4.2 Truncation Strategy

```python
def truncate_text(text: str, max_chars: int = 6000) -> str:
    """Truncate Japanese text to fit token budget.

    Cuts at sentence boundaries (。) to avoid mid-sentence breaks.
    Adds '[... truncated]' marker if shortened.
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind("。")
    if cut == -1:
        cut = max_chars
    return text[:cut + 1] + "\n[... 以下省略]"
```

### 4.3 Cost Estimate

For the initial backfill of 4,000 companies:
- Input: 4,000 × 5,000 tokens = 20M tokens → $60 (Sonnet at $3/MTok)
- Output: 4,000 × 1,000 tokens = 4M tokens → $60 (Sonnet at $15/MTok)
- **Total: ~$80** (one-time)

### 4.4 Rate Limiting

- Sonnet 4.6 rate limit: 4,000 RPM (requests per minute)
- With 1 request/company, 4,000 companies can be processed in ~10 minutes
- Add 0.5s sleep between requests for safety → ~33 min
- Practical estimate with retries: **~1 hour**

## 5. Error Handling

### 5.1 Malformed JSON Response

```python
def parse_needs_response(raw: str, edinet_code: str) -> dict | None:
    """Parse Claude's response. Return None if invalid."""
    try:
        data = json.loads(raw)
        # Validate structure
        assert "technology_needs" in data
        assert isinstance(data["technology_needs"], list)
        assert len(data["technology_needs"]) <= 10
        return data
    except (json.JSONDecodeError, AssertionError) as e:
        logger.warning("Malformed response for %s: %s", edinet_code, e)
        return None
```

On parse failure: log the error, skip the company, retry up to 2 times with a shorter prompt.

### 5.2 Empty or Minimal R&D Text

Some companies have very short R&D sections (<100 chars). For these:
- If rd_text < 100 chars AND midterm_plan_text is empty: skip estimation, set estimated_needs = NULL
- This is expected for ~500-1,000 companies (financial holding companies, etc.)

### 5.3 Cost Guard

```python
def check_budget(estimated_cost: float) -> bool:
    """Check if estimated cost is within monthly budget."""
    current_month_spend = get_current_month_spend()  # from acquisition_log
    remaining = settings.llm_monthly_budget_usd - current_month_spend
    if estimated_cost > remaining:
        logger.error("Budget exceeded: need $%.2f, remaining $%.2f", estimated_cost, remaining)
        return False
    return True
```

## 6. Embedding of Estimated Needs

After Claude generates the 5 technology needs, combine them into a single text and embed:

```python
needs_text = " ".join(n["area"] + ": " + n["rationale"] for n in needs["technology_needs"])
needs_vector = embedding_model.encode([needs_text])[0]
```

This vector is stored as `needs_vector` in the companies table, used by the NeedFit scorer
to compute cosine similarity against seed vectors.

## 7. Phase 2 Implementation Checklist

- [ ] `sangaku_matcher/llm/claude_client.py` — API wrapper with retry, rate limit, cost tracking
- [ ] `sangaku_matcher/llm/prompt_templates.py` — Template rendering
- [ ] `sangaku_matcher/scoring/need_fit.py` — NeedFit scorer using cached needs_vector
- [ ] `scripts/refresh_needs.py` — Batch estimation command
- [ ] Unit tests: prompt rendering, JSON parsing, cost guard
- [ ] Integration test: 10 companies end-to-end
