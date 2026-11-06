"""Typed configuration loaded from configs/pipeline.yaml and configs/sources.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"


class SourceConfig(BaseModel):
    key: str
    hf_id: str
    files: list[str]
    text_column: str
    license: str
    url: str
    enabled: bool = True
    revision: str | None = None
    notes: str = ""
    # Keep only rows whose column equals the value, e.g. {"type": "web"} (read with the text).
    row_filter: dict[str, str] | None = None
    # Column holding the upstream document id, stored as doc_id (doc_filter sources only).
    id_column: str | None = None
    # True: whole documents are screened by the doc-level classifier (`doc-filter` stage)
    # and only agri documents enter data/raw/. Plain `ingest` skips these sources.
    doc_filter: bool = False


class FilterConfig(BaseModel):
    min_words: int = 3
    max_words: int = 200
    min_devanagari_ratio: float = 0.8


class SplitConfig(BaseModel):
    unsplit_long_words: int = 40


class LengthBuckets(BaseModel):
    short_max: int = 8
    medium_max: int = 20
    long_max: int = 40

    @model_validator(mode="after")
    def _ordered(self) -> LengthBuckets:
        if not (0 < self.short_max < self.medium_max < self.long_max):
            raise ValueError("length_buckets must satisfy 0 < short_max < medium_max < long_max")
        return self


class ClassifyConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "google/gemma-4-26B-A4B-it"
    shard_size: int = 50_000
    concurrency: int = 256
    client_processes: int = 4
    max_tokens: int = 5
    timeout_s: float = 120.0
    max_retries: int = 5
    # Document filter (approach C): Gemma reads a "spread" excerpt of each document.
    doc_excerpt_words: int = 200  # total words shown to the model
    doc_head_words: int = 100  # of which taken from the start of the document
    doc_keep_threshold: float = 0.2  # keep a doc if label is AGRI or P(AGRI) >= this


class ReportConfig(BaseModel):
    target_agri_sentences: int = 200_000
    samples_per_label: int = 200
    seed: int = 42


class PipelineConfig(BaseModel):
    root: Path
    data_dir: Path
    output_dir: Path
    sources_file: Path
    workers: int = 8
    ingest_threads: int = 8
    filter: FilterConfig = Field(default_factory=FilterConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    length_buckets: LengthBuckets = Field(default_factory=LengthBuckets)
    classify: ClassifyConfig = Field(default_factory=ClassifyConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    sources: list[SourceConfig] = Field(default_factory=list)

    @property
    def enabled_sources(self) -> list[SourceConfig]:
        return [s for s in self.sources if s.enabled]

    def source_priority(self) -> dict[str, int]:
        """Dedup priority: position of each source key in sources.yaml (lower wins)."""
        return {s.key: i for i, s in enumerate(self.sources)}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path | None = None) -> PipelineConfig:
    config_path = Path(path or DEFAULT_CONFIG).resolve()
    root = config_path.parent.parent
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    for key in ("data_dir", "output_dir", "sources_file"):
        if key not in raw:
            raise ValueError(f"{config_path}: missing required key '{key}'")
        raw[key] = _resolve(root, raw[key])

    sources_raw = yaml.safe_load(Path(raw["sources_file"]).read_text(encoding="utf-8")) or {}
    raw["sources"] = sources_raw.get("sources", [])
    cfg = PipelineConfig(root=root, **raw)

    keys = [s.key for s in cfg.sources]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{cfg.sources_file}: source keys must be unique, got {keys}")
    return cfg
