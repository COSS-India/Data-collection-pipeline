"""Stage 7: assemble the final corpus files and the Markdown report."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from itertools import pairwise

import polars as pl

from .classify import load_results
from .config import PipelineConfig
from .paths import Paths, require, write_parquet_atomic
from .prompt import LABELS, PROMPT_VERSION

log = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "sentence_id",
    "text",
    "label",
    "confidence",
    "word_count",
    "char_count",
    "length_bucket",
    "has_end_punct",
    "split_method",
    "source_dataset",
    "source_file",
    "source_row",
    "parent_row_id",
    "sentence_index",
    "all_sources",
    "dup_count",
    "source_license",
    "doc_id",
    "doc_p_agri",
    "model_id",
    "prompt_version",
]
BUCKETS = ["short", "medium", "long", "very_long"]


def _md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = [f"{v:,}" if isinstance(v, int) else ("" if v is None else str(v)) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _label_pivot(corpus: pl.DataFrame, index: str) -> pl.DataFrame:
    """Counts of each label per `index` value, columns in a fixed order plus TOTAL."""
    cols = [*LABELS, "UNCLASSIFIED"]
    counts = (
        corpus.with_columns(pl.col("label").fill_null("UNCLASSIFIED"))
        .group_by(index, "label")
        .agg(pl.len().alias("n"))
        .pivot(on="label", index=index, values="n")
    )
    for c in cols:
        if c not in counts.columns:
            counts = counts.with_columns(pl.lit(0, dtype=pl.UInt32).alias(c))
    counts = counts.select(index, *cols).fill_null(0)
    if counts.get_column("UNCLASSIFIED").sum() == 0:
        counts = counts.drop("UNCLASSIFIED")
    return counts.with_columns(TOTAL=pl.sum_horizontal(pl.exclude(index)))


def load_doc_audit(paths: Paths) -> pl.DataFrame | None:
    """Every document screened by the doc filter (None if the stage never ran)."""
    files = sorted(paths.docfilter_dir.glob("*/*.parquet"))
    return pl.read_parquet(files) if files else None


def build_corpus(cfg: PipelineConfig) -> pl.DataFrame:
    paths = Paths.from_config(cfg)
    unique = pl.read_parquet(require(paths.unique, "dedup"))
    audit = load_doc_audit(paths)
    if audit is not None:
        doc_meta = audit.filter("kept").select(
            pl.col("row_id").alias("parent_row_id"), "doc_id", "doc_p_agri"
        )
        unique = unique.join(doc_meta, on="parent_row_id", how="left")
    else:
        unique = unique.with_columns(
            doc_id=pl.lit(None, pl.Utf8), doc_p_agri=pl.lit(None, pl.Float32)
        )
    results = load_results(paths.classified_dir, cfg.classify.model, any_version=True).drop("error")
    licenses = {s.key: s.license for s in cfg.sources}
    df = unique.join(results, on="sentence_id", how="left").with_columns(
        source_license=pl.col("source_key").replace_strict(licenses, return_dtype=pl.Utf8),
        model_id=pl.col("model_id").fill_null(cfg.classify.model),
        prompt_version=pl.col("prompt_version").fill_null(PROMPT_VERSION),
    )
    return df.select(OUTPUT_COLUMNS)


def write_outputs(cfg: PipelineConfig) -> dict[str, int]:
    paths = Paths.from_config(cfg)
    corpus = build_corpus(cfg)
    write_parquet_atomic(corpus, paths.sentences)

    counts: dict[str, int] = {}
    for label in LABELS:
        part = corpus.filter(pl.col("label") == label)
        counts[label] = part.height
        part.write_ndjson(paths.output / f"{label.lower()}.jsonl")
        n = min(cfg.report.samples_per_label, part.height)
        paths.samples_dir.mkdir(parents=True, exist_ok=True)
        part.sample(n=n, seed=cfg.report.seed).select(
            "label", "confidence", "length_bucket", "source_dataset", "text"
        ).write_csv(paths.samples_dir / f"{label.lower()}.csv")
    counts["UNCLASSIFIED"] = corpus.filter(pl.col("label").is_null()).height

    if paths.rejected.exists():
        pl.read_parquet(paths.rejected).write_parquet(paths.output / "rejected.parquet")

    paths.report.write_text(render_report(cfg, corpus), encoding="utf-8")
    log.info("report: %s", counts)
    return counts


def render_report(cfg: PipelineConfig, corpus: pl.DataFrame) -> str:
    paths = Paths.from_config(cfg)
    manifest = (
        json.loads(paths.manifest.read_text(encoding="utf-8"))
        if paths.manifest.exists()
        else {"sources": {}}
    )
    split = pl.read_parquet(paths.split, columns=["source_key"]) if paths.split.exists() else None
    filtered = (
        pl.read_parquet(paths.filtered, columns=["source_key"]) if paths.filtered.exists() else None
    )
    rejected = pl.read_parquet(paths.rejected) if paths.rejected.exists() else None

    agri = corpus.filter(pl.col("label") == "AGRI").height
    target = cfg.report.target_agri_sentences
    errors = (
        pl.read_parquet(sorted(paths.classified_dir.glob("shard-*.parquet")))
        .filter(pl.col("error").is_not_null() & (pl.col("prompt_version") == PROMPT_VERSION))
        .height
        if any(paths.classified_dir.glob("shard-*.parquet"))
        else 0
    )

    n_labelled = corpus.filter(pl.col("label").is_not_null()).height
    n_unlabelled = corpus.height - n_labelled
    generated = datetime.now(UTC).isoformat(timespec="seconds")

    out: list[str] = []
    out += [
        "# Marathi agriculture corpus: run report",
        "",
        f"Generated {generated} · model `{cfg.classify.model}` · current prompt"
        f" `{PROMPT_VERSION}` (each row records the prompt that labelled it in `prompt_version`)",
        "",
        "## Headline",
        "",
        f"- **AGRI sentences: {agri:,}** (target {target:,}: "
        + ("**met**" if agri >= target else f"**short by {target - agri:,}**")
        + ")",
        f"- Unique sentences classified: {n_labelled:,} of {corpus.height:,}",
        f"- Unclassified (no result yet): {n_unlabelled:,} · request errors: {errors:,}",
        "",
    ]

    # Funnel per source
    rows = []
    for key, meta in manifest.get("sources", {}).items():
        rows.append(
            {
                "source": meta["hf_id"],
                "raw rows": meta["rows"],
                "sentences after split": split.filter(pl.col("source_key") == key).height
                if split is not None
                else None,
                "after filter": filtered.filter(pl.col("source_key") == key).height
                if filtered is not None
                else None,
                "unique (kept copy)": corpus.filter(
                    pl.col("source_dataset") == meta["hf_id"]
                ).height,
                "already in a higher-priority source": corpus.filter(
                    pl.col("all_sources").list.contains(meta["hf_id"])
                    & (pl.col("source_dataset") != meta["hf_id"])
                ).height,
            }
        )
    if rows:
        out += ["## Funnel per source", "", _md_table(pl.DataFrame(rows)), ""]
        out += [
            "Duplicates are kept once, credited to the highest-priority source (order in "
            "`configs/sources.yaml`); `all_sources` on each row lists every source it appeared in.",
            "",
        ]

    # Document filter
    audit = load_doc_audit(paths)
    if audit is not None:
        per_file = (
            audit.group_by("source_key", "source_file")
            .agg(
                docs_read=pl.len(),
                labelled_agri=(pl.col("doc_label") == "AGRI").sum(),
                kept=pl.col("kept").sum(),
            )
            .with_columns(kept_pct=(100 * pl.col("kept") / pl.col("docs_read")).round(1))
            .sort("source_key", "source_file")
        )
        edges = [0.0, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0001]
        names = ["<0.01", "0.01–0.2", "0.2–0.5", "0.5–0.8", "0.8–0.99", "≥0.99"]
        hist = pl.DataFrame(
            {
                "P(AGRI)": names,
                "documents": [
                    audit.filter((pl.col("doc_p_agri") >= lo) & (pl.col("doc_p_agri") < hi)).height
                    for lo, hi in pairwise(edges)
                ],
            }
        )
        c = cfg.classify
        out += [
            "## Document filter (general web sources)",
            "",
            f"Gemma reads a spread excerpt of each document ({c.doc_excerpt_words} words: the"
            f" opening {c.doc_head_words} + evenly spaced sentences) and the document is kept if"
            f" labelled AGRI or P(AGRI) ≥ {c.doc_keep_threshold}. Only kept documents are split"
            " into sentences; agri sentences inside dropped documents are not recovered.",
            "",
            _md_table(per_file),
            "",
            _md_table(hist),
            "",
        ]

    # Rejections
    if rejected is not None and rejected.height:
        rej = rejected.group_by("reason").agg(pl.len().alias("rows")).sort("rows", descending=True)
        out += ["## Dropped rows by reason", "", _md_table(rej), ""]

    # Labels per source
    by_src = _label_pivot(corpus, "source_dataset").sort("source_dataset")
    out += ["## Labels per source (unique sentences)", "", _md_table(by_src), ""]

    # Labels per prompt version (sources added later are classified with the newer prompt)
    by_ver = _label_pivot(corpus.filter(pl.col("label").is_not_null()), "prompt_version").sort(
        "prompt_version"
    )
    out += ["## Labels per prompt version", "", _md_table(by_ver), ""]

    # Labels per length bucket
    by_len = (
        _label_pivot(corpus, "length_bucket")
        .with_columns(
            pl.col("length_bucket")
            .replace_strict({b: i for i, b in enumerate(BUCKETS)}, return_dtype=pl.Int8)
            .alias("_o")
        )
        .sort("_o")
        .drop("_o")
    )
    b = cfg.length_buckets
    out += [
        "## Labels per length bucket",
        "",
        f"short ≤{b.short_max} words · medium ≤{b.medium_max} · long ≤{b.long_max}"
        f" · very_long >{b.long_max}",
        "",
        _md_table(by_len),
        "",
    ]

    # Confidence
    conf = (
        corpus.filter(pl.col("label").is_not_null())
        .group_by("label")
        .agg(
            n=pl.len(),
            mean_conf=pl.col("confidence").cast(pl.Float64).mean().round(3),
            share_below_0_7=(pl.col("confidence") < 0.7).mean().round(3),
        )
        .sort("label")
    )
    out += ["## Classifier confidence", "", _md_table(conf), ""]

    # Sources
    src_rows = [
        {
            "dataset": s.hf_id,
            "url": s.url,
            "license": s.license,
            "revision": manifest.get("sources", {})
            .get(s.key, {})
            .get("revision", "not ingested")[:12],
            "used": "yes" if s.enabled and s.key in manifest.get("sources", {}) else "no",
        }
        for s in cfg.sources
    ]
    out += ["## Sources", "", _md_table(pl.DataFrame(src_rows)), ""]
    out += [
        "## Output files",
        "",
        "- `output/sentences.parquet`: every unique sentence with all metadata",
        "- `output/agri.jsonl`, `output/non_agri.jsonl`, `output/not_marathi.jsonl`",
        "- `output/rejected.parquet`: rows dropped during cleaning/filtering, with reason",
        "- `output/samples/*.csv`: random samples per label for spot-checking",
        "",
    ]
    return "\n".join(out)
