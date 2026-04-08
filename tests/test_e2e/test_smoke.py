"""End-to-end smoke test with dummy data."""
import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def db_path():
    """Create a temp DB and load dummy companies."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = Path(tmpdir) / "test.db"
        os.environ["MATCHER_DB_PATH"] = str(db)

        # Re-import to pick up env var
        from sangaku_matcher.config import Settings
        s = Settings()
        s.matcher_db_path = db

        from sangaku_matcher.db import init_db
        init_db(db)

        # Patch settings
        import sangaku_matcher.config
        sangaku_matcher.config.settings = s

        # Load dummy data
        from sangaku_matcher.acquisition.company_loader import load_companies_to_db
        load_companies_to_db(dummy=True)

        yield db


@pytest.mark.slow
class TestSmoke:
    def test_match_returns_results(self, db_path):
        from sangaku_matcher.seeds import parse_seed
        from sangaku_matcher.matcher import run_match

        seed = parse_seed(
            description="自己組織化ハニカム構造を持つ多孔質高分子フィルム。"
                        "細胞培養基材、センサー、光学フィルター等への応用が期待される。",
            title="ハニカムフィルム",
        )
        result = run_match(seed, top_n=5)

        assert len(result.rankings) == 5
        assert all(0.0 <= r.total_score <= 1.0 for r in result.rankings)
        # Scores should be descending
        scores = [r.total_score for r in result.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_match_produces_markdown(self, db_path):
        from sangaku_matcher.seeds import parse_seed
        from sangaku_matcher.matcher import run_match
        from sangaku_matcher.reporter import to_markdown

        seed = parse_seed(description="ペロブスカイト太陽電池の大面積成膜技術", title="ペロブスカイト")
        result = run_match(seed, top_n=3)
        md = to_markdown(result)

        assert "# マッチング結果" in md
        assert "ペロブスカイト" in md

    def test_match_produces_json(self, db_path):
        from sangaku_matcher.seeds import parse_seed
        from sangaku_matcher.matcher import run_match
        from sangaku_matcher.reporter import to_json

        seed = parse_seed(description="炭素繊維強化プラスチックのケミカルリサイクル技術", title="CFRP再資源化")
        result = run_match(seed, top_n=3)
        data = to_json(result)

        assert "matches" in data
        assert len(data["matches"]) == 3
        # JSON should be serializable
        json.dumps(data, ensure_ascii=False)
