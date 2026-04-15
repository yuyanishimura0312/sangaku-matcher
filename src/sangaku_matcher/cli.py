"""CLI entry point for sangaku-matcher."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from sangaku_matcher.config import settings


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """sangaku-matcher: University-Industry matching system."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
def init_db() -> None:
    """Initialize the SQLite database."""
    from sangaku_matcher.db import init_db as _init

    _init(settings.matcher_db_path)
    click.echo(f"Database initialized at {settings.matcher_db_path}")


@main.command()
@click.option("--seed", "-s", type=click.Path(exists=True), help="Seed file (.md or .json)")
@click.option("--text", "-t", type=str, help="Seed description text")
@click.option("--title", default="", help="Seed title")
@click.option("--doi", default="", help="Paper DOI")
@click.option("--top", "-n", default=10, help="Number of top matches")
@click.option("--output", "-o", type=click.Path(), help="Output file (.md or .json)")
def match(seed: str | None, text: str | None, title: str, doi: str, top: int, output: str | None) -> None:
    """Run matching for a technology seed."""
    from sangaku_matcher.seeds import parse_seed
    from sangaku_matcher.matcher import run_match
    from sangaku_matcher.reporter import to_markdown, to_json

    if seed:
        content = Path(seed).read_text(encoding="utf-8")
        if seed.endswith(".json"):
            data = json.loads(content)
            text = data.get("description", "")
            title = data.get("title", title)
            doi = data.get("doi", doi)
        else:
            text = content
            if not title:
                title = Path(seed).stem

    if not text:
        click.echo("Error: provide --seed file or --text", err=True)
        sys.exit(1)

    click.echo("Parsing seed and generating embedding...")
    s = parse_seed(description=text, title=title, doi=doi or None)

    click.echo(f"Running match against {settings.matcher_db_path}...")
    result = run_match(s, top_n=top)

    click.echo(f"Done in {result.duration_sec}s. {result.company_count} companies scored.")

    if output:
        out_path = Path(output)
        if output.endswith(".json"):
            out_path.write_text(json.dumps(to_json(result), ensure_ascii=False, indent=2))
        else:
            out_path.write_text(to_markdown(result), encoding="utf-8")
        click.echo(f"Result saved to {out_path}")
    else:
        click.echo(to_markdown(result))


@main.command()
@click.option("--use-ir-collector", is_flag=True, help="Load from IR Collector DB")
@click.option("--dummy", is_flag=True, help="Load dummy data for testing")
def load_companies(use_ir_collector: bool, dummy: bool) -> None:
    """Load company data into the database."""
    from sangaku_matcher.acquisition.company_loader import load_companies_to_db

    load_companies_to_db(use_ir_collector=use_ir_collector, dummy=dummy)


@main.command()
@click.argument("csv_path", type=click.Path(exists=True))
def import_collabs(csv_path: str) -> None:
    """Import university-company collaboration data from CSV."""
    import csv
    from sangaku_matcher.db import connect

    count = 0
    batch_size = 1000
    with connect(settings.matcher_db_path) as conn:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            batch: list[tuple] = []
            for row in reader:
                batch.append((
                    row["edinet_code"],
                    row["university_name"],
                    row.get("type", "patent"),
                    int(row.get("count", 1)),
                    int(row["last_year"]) if row.get("last_year") else None,
                ))
                count += 1
                # Flush in chunks of 1000 for better performance
                if len(batch) >= batch_size:
                    conn.executemany(
                        """INSERT OR REPLACE INTO collaborations
                           (edinet_code, university_name, type, count, last_year)
                           VALUES (?, ?, ?, ?, ?)""",
                        batch,
                    )
                    batch.clear()
            # Flush remaining rows
            if batch:
                conn.executemany(
                    """INSERT OR REPLACE INTO collaborations
                       (edinet_code, university_name, type, count, last_year)
                       VALUES (?, ?, ?, ?, ?)""",
                    batch,
                )
    click.echo(f"Imported {count} collaboration records.")


@main.command()
def serve() -> None:
    """Start the web UI server."""
    import uvicorn

    # Only enable hot-reload for local development
    is_dev = settings.host in ("127.0.0.1", "localhost")
    click.echo(f"Starting web UI at http://{settings.host}:{settings.port}")
    uvicorn.run(
        "sangaku_matcher.web.app:app",
        host=settings.host,
        port=settings.port,
        reload=is_dev,
    )


if __name__ == "__main__":
    main()
