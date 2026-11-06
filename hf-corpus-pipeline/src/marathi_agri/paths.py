"""Locations of every stage artefact, derived from the config."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .config import PipelineConfig


@dataclass(frozen=True)
class Paths:
    data: Path
    output: Path

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> Paths:
        return cls(data=cfg.data_dir, output=cfg.output_dir)

    # --- intermediate (data/) ---
    @property
    def raw_dir(self) -> Path:
        return self.data / "raw"

    @property
    def docfilter_dir(self) -> Path:
        return self.data / "docfilter"

    @property
    def manifest(self) -> Path:
        return self.data / "manifest.json"

    @property
    def cleaned(self) -> Path:
        return self.data / "cleaned.parquet"

    @property
    def split(self) -> Path:
        return self.data / "split.parquet"

    @property
    def filtered(self) -> Path:
        return self.data / "filtered.parquet"

    @property
    def rejected(self) -> Path:
        return self.data / "rejected.parquet"

    @property
    def unique(self) -> Path:
        return self.data / "unique.parquet"

    @property
    def stats(self) -> Path:
        return self.data / "stage_stats.json"

    @property
    def classified_dir(self) -> Path:
        return self.data / "classified"

    # --- final (output/) ---
    @property
    def sentences(self) -> Path:
        return self.output / "sentences.parquet"

    @property
    def report(self) -> Path:
        return self.output / "report.md"

    @property
    def samples_dir(self) -> Path:
        return self.output / "samples"


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write to a temp file and rename, so a crash never leaves a half-written stage output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the '{stage}' stage first.")
    return path
