"""Stages 2–4: clean -> split into sentences -> filter.

Row accounting is kept exact so the report can prove nothing was silently lost:
    raw rows      = cleaned rows
    cleaned rows  = rows with >=1 sentence + rows rejected as empty
    split rows    = filtered rows + rows rejected by a filter
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import polars as pl

from . import text as T
from .config import PipelineConfig
from .ingest import load_raw
from .paths import Paths, require, write_parquet_atomic

log = logging.getLogger(__name__)

A = TypeVar("A")
B = TypeVar("B")

REJECTED_SCHEMA = {
    "row_id": pl.Utf8,
    "source_key": pl.Utf8,
    "source_dataset": pl.Utf8,
    "text": pl.Utf8,
    "reason": pl.Utf8,
    "stage": pl.Utf8,
}


def _map_chunk(args: tuple[Callable[[A], B], list[A]]) -> list[B]:
    fn, items = args
    return [fn(x) for x in items]


def parallel_map(
    fn: Callable[[A], B], items: list[A], workers: int, chunk: int = 20_000
) -> list[B]:
    """Order-preserving map over processes. `fn` must be a module-level function."""
    if workers <= 1 or len(items) < chunk:
        return [fn(x) for x in items]
    chunks: Iterable[tuple[Callable[[A], B], list[A]]] = (
        (fn, items[i : i + chunk]) for i in range(0, len(items), chunk)
    )
    out: list[B] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_map_chunk, chunks):
            out.extend(part)
    return out


# ---------------------------------------------------------------------------
# Stage: clean
# ---------------------------------------------------------------------------


def run_clean(cfg: PipelineConfig) -> int:
    paths = Paths.from_config(cfg)
    raw = load_raw(cfg).collect()
    log.info("clean: %d raw rows", raw.height)
    cleaned = parallel_map(T.clean_text, raw["text_raw"].to_list(), cfg.workers)
    df = raw.with_columns(text_clean=pl.Series(cleaned, dtype=pl.Utf8)).drop("text_raw")
    write_parquet_atomic(df, paths.cleaned)
    return df.height


# ---------------------------------------------------------------------------
# Stage: split
# ---------------------------------------------------------------------------


def run_split(cfg: PipelineConfig) -> tuple[int, int]:
    """Explode each cleaned row into sentences. Returns (sentences, empty rows rejected)."""
    paths = Paths.from_config(cfg)
    df = pl.read_parquet(require(paths.cleaned, "clean"))
    pieces = parallel_map(T.split_sentences, df["text_clean"].to_list(), cfg.workers)
    df = df.with_columns(sentences=pl.Series(pieces, dtype=pl.List(pl.Utf8)))
    df = df.with_columns(n_sentences=pl.col("sentences").list.len())

    empty = df.filter(pl.col("n_sentences") == 0).select(
        "row_id",
        "source_key",
        "source_dataset",
        text=pl.col("text_clean"),
        reason=pl.lit("empty_after_clean"),
        stage=pl.lit("split"),
    )

    long_words = cfg.split.unsplit_long_words
    exploded = (
        df.filter(pl.col("n_sentences") > 0)
        .rename({"row_id": "parent_row_id"})
        .explode("sentences")
        .rename({"sentences": "text"})
        .with_columns(sentence_index=pl.int_range(pl.len(), dtype=pl.Int32).over("parent_row_id"))
        .with_columns(word_count=pl.col("text").str.split(" ").list.len().cast(pl.Int32))
        .with_columns(
            split_method=pl.when(pl.col("n_sentences") > 1)
            .then(pl.lit("punctuation"))
            .when(pl.col("word_count") > long_words)
            .then(pl.lit("unsplit_long"))
            .otherwise(pl.lit("none"))
        )
        .drop("text_clean", "n_sentences")
    )
    write_parquet_atomic(exploded, paths.split)
    _append_rejected(paths, empty, stage="split")
    log.info(
        "split: %d rows -> %d sentences (%d empty rows)", df.height, exploded.height, empty.height
    )
    return exploded.height, empty.height


# ---------------------------------------------------------------------------
# Stage: filter
# ---------------------------------------------------------------------------


def _measure(text: str) -> tuple[float, int, bool]:
    return T.devanagari_ratio(text), len(text), T.has_end_punct(text)


def run_filter(cfg: PipelineConfig) -> tuple[int, int]:
    """Drop too-short / too-long / non-Devanagari sentences; add size metadata."""
    paths = Paths.from_config(cfg)
    f, b = cfg.filter, cfg.length_buckets
    df = pl.read_parquet(require(paths.split, "split"))

    measures = parallel_map(_measure, df["text"].to_list(), cfg.workers)
    df = df.with_columns(
        devanagari_ratio=pl.Series([m[0] for m in measures], dtype=pl.Float32),
        char_count=pl.Series([m[1] for m in measures], dtype=pl.Int32),
        has_end_punct=pl.Series([m[2] for m in measures], dtype=pl.Boolean),
    )
    df = df.with_columns(
        reason=pl.when(pl.col("word_count") < f.min_words)
        .then(pl.lit("too_short"))
        .when(pl.col("word_count") > f.max_words)
        .then(pl.lit("too_long"))
        .when(pl.col("devanagari_ratio") < f.min_devanagari_ratio)
        .then(pl.lit("low_devanagari"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )

    rejected = df.filter(pl.col("reason").is_not_null()).select(
        row_id=pl.format("{}~{}", "parent_row_id", "sentence_index"),
        source_key="source_key",
        source_dataset="source_dataset",
        text="text",
        reason="reason",
        stage=pl.lit("filter"),
    )
    kept = (
        df.filter(pl.col("reason").is_null())
        .drop("reason")
        .with_columns(
            length_bucket=pl.when(pl.col("word_count") <= b.short_max)
            .then(pl.lit("short"))
            .when(pl.col("word_count") <= b.medium_max)
            .then(pl.lit("medium"))
            .when(pl.col("word_count") <= b.long_max)
            .then(pl.lit("long"))
            .otherwise(pl.lit("very_long"))
        )
    )
    write_parquet_atomic(kept, paths.filtered)
    _append_rejected(paths, rejected, stage="filter")
    log.info("filter: kept %d, rejected %d", kept.height, rejected.height)
    return kept.height, rejected.height


def _append_rejected(paths: Paths, new: pl.DataFrame, stage: str) -> None:
    """Replace this stage's rows in rejected.parquet (idempotent on re-run)."""
    new = new.select([pl.col(c).cast(t) for c, t in REJECTED_SCHEMA.items()])
    if paths.rejected.exists():
        old = pl.read_parquet(paths.rejected).filter(pl.col("stage") != stage)
        new = pl.concat([old, new], how="vertical")
    write_parquet_atomic(new, paths.rejected)
