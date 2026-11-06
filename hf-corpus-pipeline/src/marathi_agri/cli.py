"""Command-line entry point: `marathi-agri <stage>`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from .config import PipelineConfig, load_config

app = typer.Typer(add_completion=False, help="Marathi agriculture corpus pipeline.")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="pipeline.yaml path")]
OnlyOpt = Annotated[list[str] | None, typer.Option("--only", help="Source key(s) to ingest")]


def _setup(config: Path | None) -> PipelineConfig:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return load_config(config)


@app.command()
def check(config: ConfigOpt = None) -> None:
    """Verify access, file layout and text column of every enabled source."""
    from .ingest import check_sources

    cfg = _setup(config)
    results = check_sources(cfg)
    typer.echo(json.dumps(results, indent=2, ensure_ascii=False))
    if not all(r["ok"] for r in results):
        raise typer.Exit(code=1)


@app.command()
def ingest(config: ConfigOpt = None, only: OnlyOpt = None) -> None:
    """Stage 1: read the text column of each source into data/raw/."""
    from .ingest import ingest as run

    run(_setup(config), only=only)


@app.command(name="doc-filter")
def doc_filter(config: ConfigOpt = None) -> None:
    """Stage 1b: screen documents of doc_filter sources; keep agri docs in data/raw/."""
    from .docfilter import run_doc_filter

    run_doc_filter(_setup(config))


@app.command(name="doc-trial")
def doc_trial(
    config: ConfigOpt = None,
    n: Annotated[int, typer.Option(help="Number of random documents")] = 200,
) -> None:
    """Screen a random document sample (not stored) -> output/doc_trial.csv + speed."""
    from .docfilter import run_doc_trial

    run_doc_trial(_setup(config), n)


@app.command(name="eval-prompt")
def eval_prompt(
    csv: Annotated[Path, typer.Option(help="CSV with columns text, gold [, <name>_label]")],
    config: ConfigOpt = None,
) -> None:
    """Score the current sentence prompt (and any *_label columns) against gold labels."""
    from .evaluate import run_eval

    for name, table in run_eval(_setup(config), csv).items():
        typer.echo(f"\n== {name} ==")
        typer.echo(str(table))


@app.command()
def clean(config: ConfigOpt = None) -> None:
    """Stage 2: normalize text."""
    from .prepare import run_clean

    run_clean(_setup(config))


@app.command()
def split(config: ConfigOpt = None) -> None:
    """Stage 3: split rows into sentences."""
    from .prepare import run_split

    run_split(_setup(config))


@app.command(name="filter")
def filter_(config: ConfigOpt = None) -> None:
    """Stage 4: drop too-short / too-long / non-Devanagari sentences, add size metadata."""
    from .prepare import run_filter

    run_filter(_setup(config))


@app.command()
def dedup(config: ConfigOpt = None) -> None:
    """Stage 5: remove duplicates across all sources."""
    from .dedup import run_dedup

    run_dedup(_setup(config))


@app.command()
def prepare(config: ConfigOpt = None) -> None:
    """Stages 2–5 in one go: clean, split, filter, dedup."""
    from .dedup import run_dedup
    from .prepare import run_clean, run_filter, run_split

    cfg = _setup(config)
    run_clean(cfg)
    run_split(cfg)
    run_filter(cfg)
    run_dedup(cfg)


@app.command()
def trial(
    config: ConfigOpt = None,
    n: Annotated[int, typer.Option(help="Number of random sentences")] = 2000,
) -> None:
    """Classify a random sample (results not stored) -> output/trial.csv + speed."""
    from .classify import run_trial

    run_trial(_setup(config), n)


@app.command()
def classify(
    config: ConfigOpt = None,
    limit: Annotated[int | None, typer.Option(help="Classify at most N new sentences")] = None,
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Only sentences from these source key(s)")
    ] = None,
) -> None:
    """Stage 6: classify unique sentences with no result under the current prompt (resumable)."""
    from .classify import run_classify

    stats = run_classify(_setup(config), limit=limit, only=only)
    if stats["errors"]:
        typer.echo(
            f"{stats['errors']} sentences failed; re-run 'classify' to retry them.", err=True
        )


@app.command()
def report(config: ConfigOpt = None) -> None:
    """Stage 7: write output/ corpus files and report.md."""
    from .report import write_outputs

    write_outputs(_setup(config))


@app.command(name="run-all")
def run_all(config: ConfigOpt = None) -> None:
    """ingest -> doc-filter -> prepare -> classify -> report (vLLM server must be running)."""
    from .classify import run_classify
    from .dedup import run_dedup
    from .docfilter import run_doc_filter
    from .ingest import ingest as run_ingest
    from .prepare import run_clean, run_filter, run_split
    from .report import write_outputs

    cfg = _setup(config)
    run_ingest(cfg)
    run_doc_filter(cfg)
    run_clean(cfg)
    run_split(cfg)
    run_filter(cfg)
    run_dedup(cfg)
    run_classify(cfg)
    write_outputs(cfg)


if __name__ == "__main__":
    app()
