"""Prompt evaluation against a hand-labelled CSV (`marathi-agri eval-prompt`).

The CSV needs columns `text` and `gold`. Every other column ending in `_label`
(e.g. `v1_label`, the label an earlier prompt gave) is scored too, so prompt versions can
be compared on the same sentences.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from .classify import check_server, classify_parallel
from .config import PipelineConfig
from .prompt import AGRI, LABELS, PROMPT_VERSION

log = logging.getLogger(__name__)


def score(gold: list[str], pred: list[str | None]) -> pl.DataFrame:
    """Per-label precision / recall / F1 plus support, and an 'accuracy' row."""
    rows = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred, strict=True))
        n_pred = sum(p == label for p in pred)
        n_gold = sum(g == label for g in gold)
        prec = tp / n_pred if n_pred else None
        rec = tp / n_gold if n_gold else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        rows.append(
            {"label": label, "precision": prec, "recall": rec, "f1": f1,
             "predicted": n_pred, "gold": n_gold}
        )  # fmt: skip
    acc = sum(g == p for g, p in zip(gold, pred, strict=True)) / max(len(gold), 1)
    rows.append(
        {"label": "accuracy", "precision": acc, "recall": None, "f1": None,
         "predicted": len(pred), "gold": len(gold)}
    )  # fmt: skip
    return pl.DataFrame(rows).with_columns(pl.col("precision", "recall", "f1").round(3))


def run_eval(cfg: PipelineConfig, csv_path: Path) -> dict[str, pl.DataFrame]:
    """Classify the CSV with the current prompt; return scores per labelling."""
    df = pl.read_csv(csv_path, infer_schema=False)
    for col in ("text", "gold"):
        if col not in df.columns:
            raise ValueError(f"{csv_path}: missing column '{col}'")
    bad = sorted(set(df["gold"].to_list()) - set(LABELS))
    if bad:
        raise ValueError(f"{csv_path}: unknown gold labels {bad}")

    check_server(cfg.classify)
    items = [(str(i), t) for i, t in enumerate(df["text"].to_list())]
    results = classify_parallel(items, cfg.classify)
    current = f"{PROMPT_VERSION}_label"
    df = df.with_columns(
        pl.Series(current, [r.label for r in results], dtype=pl.Utf8),
        pl.Series(f"{PROMPT_VERSION}_confidence", [r.confidence for r in results]),
    )
    out = csv_path.with_name(f"{csv_path.stem}.{PROMPT_VERSION}.csv")
    df.write_csv(out)

    gold = df["gold"].to_list()
    scores = {
        col.removesuffix("_label"): score(gold, df[col].to_list())
        for col in df.columns
        if col.endswith("_label")
    }
    for name, table in scores.items():
        agri = table.filter(pl.col("label") == AGRI).row(0, named=True)
        log.info(
            "%s: AGRI precision %s recall %s | accuracy %s",
            name, agri["precision"], agri["recall"],
            table.filter(pl.col("label") == "accuracy")["precision"].item(),
        )  # fmt: skip
    log.info("predictions -> %s", out)
    return scores
