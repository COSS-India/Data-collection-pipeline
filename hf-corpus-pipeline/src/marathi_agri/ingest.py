"""Stage 0 (check) and stage 1 (ingest): read only the text column from each HF parquet file.

Parquet column projection means only the text column's byte ranges are fetched, so the
audio in the speech datasets is never downloaded.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from .config import PipelineConfig, SourceConfig
from .paths import Paths, write_parquet_atomic

log = logging.getLogger(__name__)

RAW_SCHEMA = {
    "row_id": pl.Utf8,
    "source_key": pl.Utf8,
    "source_dataset": pl.Utf8,
    "source_file": pl.Utf8,
    "source_row": pl.Int64,
    "text_raw": pl.Utf8,
}


class SourceAccessError(RuntimeError):
    pass


def resolve_revision(api: HfApi, source: SourceConfig) -> str:
    """Pin the dataset to a commit hash so the run is reproducible."""
    if source.revision:
        return source.revision
    try:
        info = api.dataset_info(source.hf_id)
    except GatedRepoError as exc:
        raise SourceAccessError(
            f"{source.hf_id} is gated. Accept its terms at {source.url} with the account "
            "that owns HF_TOKEN, then retry."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise SourceAccessError(
            f"{source.hf_id} not found or not accessible. If it is gated, set HF_TOKEN."
        ) from exc
    if not info.sha:
        raise SourceAccessError(f"{source.hf_id}: could not resolve a commit hash")
    return info.sha


def list_files(fs: HfFileSystem, source: SourceConfig, revision: str) -> list[str]:
    """Return repo-relative parquet paths matching the source's globs, sorted."""
    prefix = f"datasets/{source.hf_id}@{revision}/"
    found: set[str] = set()
    for pattern in source.files:
        for path in fs.glob(prefix + pattern):
            found.add(path.removeprefix(prefix))
    if not found:
        raise SourceAccessError(f"{source.hf_id}: no files match {source.files}")
    return sorted(found)


def _open_parquet(fs: Any, full_path: str) -> pq.ParquetFile:
    try:
        # cache_type="none": every pyarrow read becomes one exact HTTP range request, so only
        # the footer and the text column chunks are downloaded (a read-ahead cache would pull
        # in megabytes of the neighbouring audio column per row group).
        return pq.ParquetFile(fs.open(full_path, "rb", block_size=1024 * 1024, cache_type="none"))
    except GatedRepoError as exc:
        repo = full_path.split("@", 1)[0].removeprefix("datasets/")
        raise SourceAccessError(
            f"{repo} is gated: no access with the current HF_TOKEN. Accept the dataset terms at "
            f"https://huggingface.co/datasets/{repo} with the account that owns HF_TOKEN."
        ) from exc


def text_field(schema: pa.Schema, source: SourceConfig) -> pa.Field:
    if source.text_column not in schema.names:
        raise SourceAccessError(
            f"{source.hf_id}: text column '{source.text_column}' not found. "
            f"Available columns: {schema.names}. Fix text_column in sources.yaml."
        )
    field = schema.field(source.text_column)
    if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
        raise SourceAccessError(
            f"{source.hf_id}: column '{source.text_column}' has type {field.type}, expected string"
        )
    return field


def read_text_column(fs: Any, full_path: str, source: SourceConfig, rel_path: str) -> pl.DataFrame:
    """Read one parquet file's text column, row group by row group."""
    pf = _open_parquet(fs, full_path)
    text_field(pf.schema_arrow, source)
    chunks: list[pa.Array] = []
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=[source.text_column])
        chunks.extend(c.cast(pa.large_string()) for c in table.column(0).chunks)
    texts = pa.chunked_array(chunks, type=pa.large_string())
    n = len(texts)
    rows = pl.Series("source_row", range(n), dtype=pl.Int64)
    df = pl.DataFrame(
        {
            "source_key": [source.key] * n,
            "source_dataset": [source.hf_id] * n,
            "source_file": [rel_path] * n,
            "source_row": rows,
            "text_raw": pl.Series("text_raw", texts, dtype=pl.Utf8),
        },
        schema={k: v for k, v in RAW_SCHEMA.items() if k != "row_id"},
    )
    return df.with_columns(
        row_id=pl.format("{}/{}#{}", "source_key", "source_file", "source_row")
    ).select(list(RAW_SCHEMA))


def read_doc_columns(fs: Any, full_path: str, source: SourceConfig) -> pl.DataFrame:
    """Read text + id column of one file, keeping rows that pass `row_filter`.

    Returns columns source_row (row index in the file, before filtering), doc_id, text_raw.
    Used by the document filter; text-only sources go through `read_text_column`.
    """
    pf = _open_parquet(fs, full_path)
    text_field(pf.schema_arrow, source)
    filters = source.row_filter or {}
    wanted = [source.text_column, *filters]
    if source.id_column:
        wanted.append(source.id_column)
    missing = [c for c in wanted if c not in pf.schema_arrow.names]
    if missing:
        raise SourceAccessError(
            f"{source.hf_id}: columns {missing} not found. Available: {pf.schema_arrow.names}"
        )
    columns = list(dict.fromkeys(wanted))
    tables = [pf.read_row_group(rg, columns=columns) for rg in range(pf.num_row_groups)]
    df = pl.from_arrow(pa.concat_tables(tables, promote_options="permissive"))
    assert isinstance(df, pl.DataFrame)
    df = df.with_row_index("source_row").with_columns(pl.col("source_row").cast(pl.Int64))
    for col, value in filters.items():
        df = df.filter(pl.col(col).cast(pl.Utf8) == value)
    doc_id = pl.col(source.id_column).cast(pl.Utf8) if source.id_column else pl.lit(None, pl.Utf8)
    return df.select(
        "source_row",
        doc_id.alias("doc_id"),
        pl.col(source.text_column).cast(pl.Utf8).alias("text_raw"),
    )


def raw_file_path(paths: Paths, source: SourceConfig, rel_path: str) -> Path:
    safe = rel_path.replace("/", "__")
    return paths.raw_dir / source.key / safe


def load_manifest(paths: Paths) -> dict[str, Any]:
    if paths.manifest.exists():
        return json.loads(paths.manifest.read_text(encoding="utf-8"))
    return {"sources": {}}


def save_manifest(paths: Paths, manifest: dict[str, Any]) -> None:
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.manifest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, paths.manifest)


def check_sources(cfg: PipelineConfig) -> list[dict[str, Any]]:
    """Verify access, file layout and text column for every enabled source (no data download)."""
    api, fs = HfApi(), HfFileSystem()
    results = []
    for source in cfg.enabled_sources:
        result: dict[str, Any] = {"key": source.key, "hf_id": source.hf_id, "ok": False}
        try:
            revision = resolve_revision(api, source)
            files = list_files(fs, source, revision)
            pf = _open_parquet(fs, f"datasets/{source.hf_id}@{revision}/{files[0]}")
            field = text_field(pf.schema_arrow, source)
            result.update(
                ok=True,
                revision=revision,
                n_files=len(files),
                columns=pf.schema_arrow.names,
                text_column=f"{field.name} ({field.type})",
            )
        except Exception as exc:  # report every source, don't stop at the first failure
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def ingest(cfg: PipelineConfig, only: list[str] | None = None) -> dict[str, int]:
    """Download the text column of every enabled source. Resumable per file."""
    paths = Paths.from_config(cfg)
    api, fs = HfApi(), HfFileSystem()
    manifest = load_manifest(paths)
    counts: dict[str, int] = {}

    for source in cfg.enabled_sources:
        if only and source.key not in only:
            continue
        if source.doc_filter:
            log.info("%s: document-filtered source, ingested by the 'doc-filter' stage", source.key)
            continue
        revision = resolve_revision(api, source)
        files = list_files(fs, source, revision)
        log.info("%s: %d files at revision %s", source.hf_id, len(files), revision[:10])

        def fetch(rel: str, source: SourceConfig = source, revision: str = revision) -> int:
            out = raw_file_path(paths, source, rel)
            if out.exists():
                return pl.scan_parquet(out).select(pl.len()).collect().item()
            df = read_text_column(fs, f"datasets/{source.hf_id}@{revision}/{rel}", source, rel)
            write_parquet_atomic(df, out)
            return df.height

        total = 0
        with ThreadPoolExecutor(max_workers=cfg.ingest_threads) as pool:
            for i, (rel, n) in enumerate(zip(files, pool.map(fetch, files), strict=True), 1):
                total += n
                if i % 10 == 0 or i == len(files):
                    log.info(
                        "  [%d/%d] files, %d rows so far (last: %s)", i, len(files), total, rel
                    )

        counts[source.key] = total
        manifest["sources"][source.key] = {
            "hf_id": source.hf_id,
            "url": source.url,
            "license": source.license,
            "revision": revision,
            "files": files,
            "rows": total,
            "text_column": source.text_column,
            "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        save_manifest(paths, manifest)
        log.info("%s: %d rows", source.key, total)
    return counts


def load_raw(cfg: PipelineConfig) -> pl.LazyFrame:
    paths = Paths.from_config(cfg)
    keys = [s.key for s in cfg.enabled_sources]
    files = [str(p) for k in keys for p in sorted((paths.raw_dir / k).glob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"No raw files under {paths.raw_dir}. Run 'ingest' first.")
    return pl.scan_parquet(files, schema=RAW_SCHEMA)
