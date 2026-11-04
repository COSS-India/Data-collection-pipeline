import csv
import io
import json
from pathlib import Path

from charset_normalizer import from_bytes

from dcp.extract import register
from dcp.models import TextBlock


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    best = from_bytes(raw).best()
    return str(best) if best else raw.decode("utf-8", errors="replace")


@register("text", ".txt", ".md", ".markdown", ".rst", ".log")
def extract_text(path: Path) -> list[TextBlock]:
    return [TextBlock(read_text(path))]


@register("csv", ".csv", ".tsv")
def extract_csv(path: Path) -> list[TextBlock]:
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    reader = csv.reader(io.StringIO(read_text(path)), delimiter=delim)
    blocks = []
    for i, row in enumerate(reader, 1):
        # Each text cell is its own block so cells never merge into one sentence.
        for j, cell in enumerate(row):
            if _is_texty(cell):
                blocks.append(TextBlock(cell, f"row={i}:col={j + 1}"))
    return blocks


@register("json", ".json", ".jsonl", ".ndjson")
def extract_json(path: Path) -> list[TextBlock]:
    raw = read_text(path)
    if path.suffix.lower() == ".json":
        records = [json.loads(raw)]
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    blocks: list[TextBlock] = []
    for i, rec in enumerate(records, 1):
        for key, val in _walk_strings(rec):
            if _is_texty(val):
                blocks.append(TextBlock(val, f"record={i}:{key}"))
    return blocks


def _walk_strings(obj, prefix=""):
    if isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{prefix}[{i}]")


def _is_texty(s: str) -> bool:
    """Skip ids, numbers, dates and other non-prose cells."""
    s = s.strip()
    return len(s) >= 8 and sum(c.isalpha() for c in s) / len(s) > 0.4 and " " in s
