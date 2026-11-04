"""File extractors, registered by extension."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from dcp.models import Document, TextBlock

Extractor = Callable[[Path], list[TextBlock]]
_REGISTRY: dict[str, tuple[str, Extractor]] = {}


def register(source_type: str, *exts: str):
    def deco(fn: Extractor) -> Extractor:
        for ext in exts:
            _REGISTRY[ext.lower()] = (source_type, fn)
        return fn

    return deco


def supported_extensions() -> list[str]:
    return sorted(_REGISTRY)


def extract(path: str | Path) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _REGISTRY:
        raise ValueError(f"Unsupported file type: {ext or path.name}")
    source_type, fn = _REGISTRY[ext]
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
    return Document(
        doc_id=digest,
        source_path=str(path),
        source_type=source_type,
        blocks=[b for b in fn(path) if b.text.strip()],
    )


# Import modules so their @register decorators run.
from dcp.extract import office, pdf, text, web  # noqa: E402,F401
