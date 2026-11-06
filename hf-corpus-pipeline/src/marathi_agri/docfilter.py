"""Stage 1b (doc-filter): screen whole documents of a general web corpus (approach C).

For every file of a `doc_filter: true` source:
1. read text + doc id, keep rows passing `row_filter` (e.g. Sangraha type == "web");
2. build a deterministic "spread read" excerpt: the opening of the document plus sentences
   at evenly spaced positions through the rest;
3. Gemma labels the excerpt AGRI / NON_AGRI and gives P(AGRI);
4. keep the document if it is labelled AGRI or P(AGRI) >= doc_keep_threshold.

Outputs per file (resumable: a file whose audit exists is skipped):
- data/docfilter/<source>/<file>.parquet  every document read, with label, P(AGRI), kept
- data/raw/<source>/<file>.parquet        kept documents only, in the standard raw schema,
                                           so clean / split / filter / dedup run unchanged.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, HfFileSystem

from . import text as T
from .classify import check_server, classify_parallel
from .config import PipelineConfig, SourceConfig
from .ingest import (
    RAW_SCHEMA,
    list_files,
    load_manifest,
    raw_file_path,
    read_doc_columns,
    resolve_revision,
    save_manifest,
)
from .paths import Paths, write_parquet_atomic
from .prepare import parallel_map
from .prompt import AGRI, DOC_PROMPT_VERSION, EXCERPT_GAP

log = logging.getLogger(__name__)

AUDIT_SCHEMA = {
    "row_id": pl.Utf8,
    "source_key": pl.Utf8,
    "source_file": pl.Utf8,
    "source_row": pl.Int64,
    "doc_id": pl.Utf8,
    "doc_words": pl.Int32,
    "doc_label": pl.Utf8,
    "doc_p_agri": pl.Float32,
    "kept": pl.Boolean,
    "model_id": pl.Utf8,
    "doc_prompt_version": pl.Utf8,
}


# ---------------------------------------------------------------------------
# Spread read
# ---------------------------------------------------------------------------


def _first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def spread_excerpt(text: str, max_words: int = 200, head_words: int = 100) -> str:
    """Deterministic excerpt of a cleaned document, at most `max_words` words.

    The opening sentences fill up to `head_words`; the remaining budget is spent on
    sentences picked at evenly spaced positions through the rest of the document, in
    document order. Gaps between non-adjacent parts are marked with EXCERPT_GAP.
    A document that fits in the budget is returned whole.
    """
    sentences = [s for s in T.split_sentences(text) if s.strip()]
    if not sentences:
        return ""
    lengths = [len(s.split()) for s in sentences]
    if sum(lengths) <= max_words:
        return " ".join(sentences)

    # Head: whole sentences up to head_words (the first one is truncated if it alone is longer).
    head: list[str] = []
    used = 0
    i = 0
    while i < len(sentences) and used + lengths[i] <= head_words:
        head.append(sentences[i])
        used += lengths[i]
        i += 1
    if not head:
        head = [_first_words(sentences[0], head_words)]
        used = len(head[0].split())
        i = 1

    rest = list(range(i, len(sentences)))
    budget = max_words - used
    if not rest or budget <= 0:
        return " ".join(head)

    # Evenly spaced picks through the rest; enough candidates to fill the budget.
    avg = max(1, round(sum(lengths[j] for j in rest) / len(rest)))
    k = min(len(rest), max(1, budget // avg))
    picks = sorted({rest[int((j + 0.5) * len(rest) / k)] for j in range(k)})

    parts = [" ".join(head)]
    prev = i - 1
    for j in picks:
        if budget <= 0:
            break
        piece = sentences[j] if lengths[j] <= budget else _first_words(sentences[j], budget)
        parts.append((" " if j == prev + 1 else EXCERPT_GAP) + piece)
        budget -= len(piece.split())
        prev = j
    return "".join(parts)


def _clean_and_excerpt(text: str | None, max_words: int, head_words: int) -> tuple[int, str]:
    cleaned = T.clean_text(text)
    return len(cleaned.split()), spread_excerpt(cleaned, max_words, head_words)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _audit_path(paths: Paths, source: SourceConfig, rel_path: str) -> Path:
    return paths.docfilter_dir / source.key / rel_path.replace("/", "__")


def screen_documents(docs: pl.DataFrame, cfg: PipelineConfig) -> pl.DataFrame:
    """Add doc_words, excerpt, doc_label, doc_p_agri, kept to a frame with `text_raw`.

    Raises RuntimeError if any request failed after retries (nothing is written then, so a
    re-run retries the whole file instead of silently dropping documents).
    """
    ccfg = cfg.classify
    fn = partial(
        _clean_and_excerpt, max_words=ccfg.doc_excerpt_words, head_words=ccfg.doc_head_words
    )
    measured = parallel_map(fn, docs["text_raw"].to_list(), cfg.workers)
    docs = docs.with_columns(
        doc_words=pl.Series([n for n, _ in measured], dtype=pl.Int32),
        excerpt=pl.Series([e for _, e in measured], dtype=pl.Utf8),
    )
    todo = docs.filter(pl.col("excerpt") != "")
    items = list(zip(todo["row_id"].to_list(), todo["excerpt"].to_list(), strict=True))
    results = classify_parallel(items, ccfg, kind="doc") if items else []
    failed = [r for r in results if r.error]
    if failed:
        raise RuntimeError(
            f"{len(failed)} documents failed after retries (first: {failed[0].error}); "
            "nothing written, re-run 'doc-filter' to retry"
        )
    labels = pl.DataFrame(
        {
            "row_id": [r.item_id for r in results],
            "doc_label": [r.label for r in results],
            "doc_p_agri": [r.confidence for r in results],
        },
        schema={"row_id": pl.Utf8, "doc_label": pl.Utf8, "doc_p_agri": pl.Float32},
    )
    return docs.join(labels, on="row_id", how="left").with_columns(
        kept=(
            (pl.col("doc_label") == AGRI) | (pl.col("doc_p_agri") >= ccfg.doc_keep_threshold)
        ).fill_null(False)
    )


def _filter_file(
    cfg: PipelineConfig,
    fs: HfFileSystem,
    source: SourceConfig,
    revision: str,
    rel: str,
) -> tuple[int, int]:
    """Screen one file; returns (documents read, documents kept)."""
    paths = Paths.from_config(cfg)
    audit_out = _audit_path(paths, source, rel)
    raw_out = raw_file_path(paths, source, rel)
    if audit_out.exists() and raw_out.exists():
        audit = pl.read_parquet(audit_out, columns=["kept"])
        log.info("  %s: done earlier (%d docs)", rel, audit.height)
        return audit.height, int(audit["kept"].sum())

    t0 = time.monotonic()
    docs = read_doc_columns(fs, f"datasets/{source.hf_id}@{revision}/{rel}", source)
    docs = docs.with_columns(
        row_id=pl.format("{}/{}#{}", pl.lit(source.key), pl.lit(rel), "source_row"),
        source_key=pl.lit(source.key),
        source_dataset=pl.lit(source.hf_id),
        source_file=pl.lit(rel),
    )
    log.info("  %s: %d documents after row filter %s", rel, docs.height, source.row_filter)
    screened = screen_documents(docs, cfg)

    raw = screened.filter("kept").select([pl.col(c).cast(t) for c, t in RAW_SCHEMA.items()])
    audit = screened.with_columns(
        model_id=pl.lit(cfg.classify.model),
        doc_prompt_version=pl.lit(DOC_PROMPT_VERSION),
    ).select([pl.col(c).cast(t) for c, t in AUDIT_SCHEMA.items()])
    write_parquet_atomic(raw, raw_out)
    write_parquet_atomic(audit, audit_out)  # written last: its presence marks the file done
    secs = time.monotonic() - t0
    log.info(
        "  %s: kept %d of %d documents (%.1f%%) in %.0fs (%.0f docs/s)",
        rel, raw.height, audit.height, 100 * raw.height / max(audit.height, 1),
        secs, audit.height / max(secs, 1e-9),
    )  # fmt: skip
    return audit.height, raw.height


def run_doc_filter(cfg: PipelineConfig) -> dict[str, tuple[int, int]]:
    """Screen every file of every enabled doc_filter source. Returns {key: (read, kept)}."""
    sources = [s for s in cfg.enabled_sources if s.doc_filter]
    if not sources:
        log.info("doc-filter: no enabled source has doc_filter: true")
        return {}
    check_server(cfg.classify)
    paths = Paths.from_config(cfg)
    api, fs = HfApi(), HfFileSystem()
    out: dict[str, tuple[int, int]] = {}
    for source in sources:
        revision = resolve_revision(api, source)
        files = list_files(fs, source, revision)
        log.info("%s: %d files at revision %s", source.hf_id, len(files), revision[:10])
        read = kept = 0
        for rel in files:
            r, k = _filter_file(cfg, fs, source, revision, rel)
            read, kept = read + r, kept + k
        out[source.key] = (read, kept)

        manifest = load_manifest(paths)
        manifest["sources"][source.key] = {
            "hf_id": source.hf_id,
            "url": source.url,
            "license": source.license,
            "revision": revision,
            "files": files,
            "rows": kept,
            "text_column": source.text_column,
            "row_filter": source.row_filter,
            "doc_filter": {
                "docs_read": read,
                "docs_kept": kept,
                "doc_prompt_version": DOC_PROMPT_VERSION,
                "excerpt_words": cfg.classify.doc_excerpt_words,
                "head_words": cfg.classify.doc_head_words,
                "keep_threshold": cfg.classify.doc_keep_threshold,
            },
            "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        save_manifest(paths, manifest)
        log.info("%s: kept %d of %d documents", source.key, kept, read)
    return out


def run_doc_trial(cfg: PipelineConfig, n: int, seed: int = 0) -> Path:
    """Screen n random documents of the first doc_filter file (nothing stored as results);
    write output/doc_trial.csv with excerpt, label and P(AGRI) for review."""
    source = next((s for s in cfg.enabled_sources if s.doc_filter), None)
    if source is None:
        raise ValueError("no enabled source has doc_filter: true")
    check_server(cfg.classify)
    api, fs = HfApi(), HfFileSystem()
    revision = resolve_revision(api, source)
    rel = list_files(fs, source, revision)[0]
    docs = read_doc_columns(fs, f"datasets/{source.hf_id}@{revision}/{rel}", source)
    rows = sorted(random.Random(seed).sample(range(docs.height), min(n, docs.height)))
    docs = docs[rows].with_columns(
        row_id=pl.format("{}/{}#{}", pl.lit(source.key), pl.lit(rel), "source_row")
    )
    t0 = time.monotonic()
    screened = screen_documents(docs, cfg)
    secs = time.monotonic() - t0
    paths = Paths.from_config(cfg)
    out = paths.output / "doc_trial.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    screened.select("row_id", "doc_label", "doc_p_agri", "kept", "doc_words", "excerpt").write_csv(
        out
    )
    p = screened["doc_p_agri"].drop_nulls()
    log.info(
        "doc trial: %d docs in %.1fs (%.0f docs/s); AGRI label %d, kept %d; "
        "P(AGRI) strictly between 0.01 and 0.99: %d -> %s",
        screened.height, secs, screened.height / max(secs, 1e-9),
        (screened["doc_label"] == AGRI).sum(), screened["kept"].sum(),
        ((p > 0.01) & (p < 0.99)).sum(), out,
    )  # fmt: skip
    return out
