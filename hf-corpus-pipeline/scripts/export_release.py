"""Build the corpus release (Draft v0.1) from output/sentences.parquet into release/.

Release = AGRI sentences only, with every metadata column, minus excluded sources:
- MILU is an evaluation benchmark: releasing its text would contaminate MILU scores.
- The Samanantar copy (shivam/marathi_samanantar_processed) is kept, but its rows carry a
  `license_note`, because that copy states no license on its dataset card.

Usage:  .venv/bin/python scripts/export_release.py [--root .] [--out release]
Writes: release/data/agri-00000.parquet, release/README.md (dataset card), release/report.md
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl

EXCLUDED_SOURCES = {"ai4bharat/MILU": "evaluation benchmark (contamination risk)"}
LICENSE_NOTES = {
    "shivam/marathi_samanantar_processed": (
        "unclear: third-party copy of Samanantar with no license on its dataset card"
    ),
}


def select_release_rows(sentences: pl.DataFrame) -> pl.DataFrame:
    """AGRI rows of non-excluded sources, plus a `license_note` column (null if none)."""
    notes = pl.col("source_dataset").replace_strict(
        LICENSE_NOTES, default=None, return_dtype=pl.Utf8
    )
    return (
        sentences.filter(
            (pl.col("label") == "AGRI") & ~pl.col("source_dataset").is_in(list(EXCLUDED_SOURCES))
        )
        .with_columns(license_note=notes)
        .sort("sentence_id")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/release")
    args = ap.parse_args()
    root: Path = args.root
    out: Path = args.out or root / "release"

    sentences = pl.read_parquet(root / "output" / "sentences.parquet")
    release = select_release_rows(sentences)

    (out / "data").mkdir(parents=True, exist_ok=True)
    target = out / "data" / "agri-00000.parquet"
    release.write_parquet(target, compression="zstd", statistics=True)
    shutil.copyfile(root / "docs" / "DATASET_CARD.md", out / "README.md")
    shutil.copyfile(root / "output" / "report.md", out / "report.md")

    agri = sentences.filter(pl.col("label") == "AGRI")
    excluded = agri.filter(pl.col("source_dataset").is_in(list(EXCLUDED_SOURCES))).height
    print(f"AGRI in corpus: {agri.height:,} | excluded ({', '.join(EXCLUDED_SOURCES)}): "
          f"{excluded:,} | released: {release.height:,}")  # fmt: skip
    print(release.group_by("source_dataset").len().sort("len", descending=True))
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
