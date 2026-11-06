"""End-to-end test of ingest-read -> clean -> split -> filter -> dedup on local fixtures."""

from pathlib import Path

import fsspec
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from marathi_agri.config import SourceConfig, load_config
from marathi_agri.dedup import run_dedup
from marathi_agri.ingest import SourceAccessError, read_text_column
from marathi_agri.paths import Paths, write_parquet_atomic
from marathi_agri.prepare import run_clean, run_filter, run_split


def _source(key: str, column: str = "text") -> SourceConfig:
    return SourceConfig(
        key=key, hf_id=f"org/{key}", files=["*.parquet"], text_column=column, license="CC0", url="u"
    )


def _write_parquet(path: Path, columns: dict) -> None:
    pq.write_table(pa.table(columns), path, row_group_size=2)


def test_read_text_column_projects_one_column(tmp_path):
    f = tmp_path / "a.parquet"
    _write_parquet(f, {"audio": [b"x" * 10] * 3, "text": ["एक", "दोन", None]})
    df = read_text_column(fsspec.filesystem("file"), str(f), _source("s"), "a.parquet")
    assert df.columns == [
        "row_id",
        "source_key",
        "source_dataset",
        "source_file",
        "source_row",
        "text_raw",
    ]
    assert df["text_raw"].to_list() == ["एक", "दोन", None]
    assert df["row_id"][1] == "s/a.parquet#1"


def test_read_text_column_missing_column(tmp_path):
    f = tmp_path / "a.parquet"
    _write_parquet(f, {"sentence": ["एक"]})
    with pytest.raises(SourceAccessError, match="Available columns"):
        read_text_column(fsspec.filesystem("file"), str(f), _source("s"), "a.parquet")


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "configs").mkdir()
    sources = {
        "sources": [
            {
                "key": "first",
                "hf_id": "org/first",
                "files": ["*"],
                "text_column": "text",
                "license": "A",
                "url": "u1",
            },
            {
                "key": "second",
                "hf_id": "org/second",
                "files": ["*"],
                "text_column": "text",
                "license": "B",
                "url": "u2",
            },
        ]
    }
    (tmp_path / "configs" / "sources.yaml").write_text(yaml.safe_dump(sources), encoding="utf-8")
    pipeline = {
        "data_dir": "data",
        "output_dir": "output",
        "sources_file": "configs/sources.yaml",
        "workers": 1,
        "filter": {"min_words": 3, "max_words": 50, "min_devanagari_ratio": 0.8},
        "split": {"unsplit_long_words": 10},
    }
    (tmp_path / "configs" / "pipeline.yaml").write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    cfg = load_config(tmp_path / "configs" / "pipeline.yaml")

    paths = Paths.from_config(cfg)
    rows = {
        "first": [
            "शेतकऱ्यांनी वेळेवर पेरणी करावी। पावसाचा अंदाज चांगला आहे।",
            "This is English text only here",
            "हो",
        ],
        "second": [
            "शेतकऱ्यांनी वेळेवर पेरणी करावी",  # duplicate of a sentence in 'first' (no danda)
            "एक दोन तीन चार पाच सहा सात आठ नऊ दहा अकरा बारा",  # unsplit long row
            "[noise]",
        ],
    }
    for key, texts in rows.items():
        df = pl.DataFrame(
            {
                "row_id": [f"{key}/f.parquet#{i}" for i in range(len(texts))],
                "source_key": key,
                "source_dataset": f"org/{key}",
                "source_file": "f.parquet",
                "source_row": list(range(len(texts))),
                "text_raw": texts,
            },
            schema_overrides={"source_row": pl.Int64},
        )
        write_parquet_atomic(df, paths.raw_dir / key / "f.parquet")
    return cfg


def test_prepare_and_dedup_end_to_end(cfg):
    paths = Paths.from_config(cfg)

    assert run_clean(cfg) == 6
    n_sentences, n_empty = run_split(cfg)
    assert (n_sentences, n_empty) == (6, 1)  # 2 + 1 + 1 + 1 + 1 sentences; "[noise]" row empty

    split = pl.read_parquet(paths.split)
    long_row = split.filter(
        pl.col("source_key") == "second", pl.col("parent_row_id").str.ends_with("#1")
    )
    assert long_row["split_method"].item() == "unsplit_long"
    first_row = split.filter(pl.col("parent_row_id") == "first/f.parquet#0").sort("sentence_index")
    assert first_row["sentence_index"].to_list() == [0, 1]
    assert set(first_row["split_method"]) == {"punctuation"}

    kept, rejected = run_filter(cfg)
    assert (kept, rejected) == (4, 2)  # English row -> low_devanagari, "हो" -> too_short
    reasons = pl.read_parquet(paths.rejected)["reason"].to_list()
    assert sorted(reasons) == ["empty_after_clean", "low_devanagari", "too_short"]

    filtered = pl.read_parquet(paths.filtered)
    assert set(filtered["length_bucket"]) == {"short", "medium"}

    n_unique, n_dups = run_dedup(cfg)
    assert (n_unique, n_dups) == (3, 1)
    unique = pl.read_parquet(paths.unique)
    dup = unique.filter(pl.col("dup_count") == 2)
    assert (
        dup["text"].item() == "शेतकऱ्यांनी वेळेवर पेरणी करावी।"
    )  # punctuated copy from higher-priority source
    assert dup["all_sources"].item().to_list() == ["org/first", "org/second"]


def test_report_joins_doc_metadata(cfg):
    from marathi_agri.report import build_corpus

    paths = Paths.from_config(cfg)
    run_clean(cfg)
    run_split(cfg)
    run_filter(cfg)
    run_dedup(cfg)
    audit = pl.DataFrame(
        {
            "row_id": ["second/f.parquet#1", "second/f.parquet#2"],
            "doc_id": ["doc-1", "doc-2"],
            "doc_p_agri": [0.9, 0.1],
            "kept": [True, False],
        },
        schema_overrides={"doc_p_agri": pl.Float32},
    )
    write_parquet_atomic(audit, paths.docfilter_dir / "second" / "f.parquet")
    corpus = build_corpus(cfg)
    long_row = corpus.filter(pl.col("parent_row_id") == "second/f.parquet#1")
    assert long_row["doc_id"].item() == "doc-1"
    assert corpus.filter(pl.col("doc_id").is_null()).height == corpus.height - 1
