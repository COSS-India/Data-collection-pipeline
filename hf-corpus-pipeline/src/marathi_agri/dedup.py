"""Stage 5: exact dedup across all sources on a normalized key.

The kept copy is the first occurrence in source priority order (sources.yaml order),
then file / row / sentence position. `all_sources` and `dup_count` record every hit.
"""

from __future__ import annotations

import logging

import polars as pl

from . import text as T
from .config import PipelineConfig
from .paths import Paths, require, write_parquet_atomic
from .prepare import parallel_map

log = logging.getLogger(__name__)


def _key_and_id(text: str) -> tuple[str, str]:
    key = T.dedup_key(text)
    return key, T.sentence_id(key)


def run_dedup(cfg: PipelineConfig) -> tuple[int, int]:
    """Returns (unique sentences, duplicates removed)."""
    paths = Paths.from_config(cfg)
    df = pl.read_parquet(require(paths.filtered, "filter"))
    priority = cfg.source_priority()

    ids = parallel_map(_key_and_id, df["text"].to_list(), cfg.workers)
    df = df.with_columns(
        sentence_id=pl.Series([i for _, i in ids], dtype=pl.Utf8),
        _priority=pl.col("source_key").replace_strict(priority, return_dtype=pl.Int32),
    ).sort(["_priority", "source_file", "source_row", "sentence_index"], maintain_order=True)

    first_cols = [c for c in df.columns if c not in ("sentence_id", "_priority")]
    unique = df.group_by("sentence_id", maintain_order=True).agg(
        *[pl.col(c).first() for c in first_cols],
        all_sources=pl.col("source_dataset").unique(maintain_order=True),
        dup_count=pl.len().cast(pl.Int32),
    )
    write_parquet_atomic(unique, paths.unique)
    removed = df.height - unique.height
    log.info("dedup: %d sentences -> %d unique (%d duplicates)", df.height, unique.height, removed)
    return unique.height, removed
