from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    """A contiguous chunk of text from a source, with where it came from."""

    text: str
    location: str = ""  # e.g. "page=3", "slide=2", "sheet=Sheet1:row=4"


@dataclass
class Document:
    doc_id: str
    source_path: str
    source_type: str
    blocks: list[TextBlock]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sentence:
    sent_id: str
    text: str
    doc_id: str
    source_path: str
    source_type: str
    location: str
    index: int  # position within the document
    lang: str = "und"
    script: str = "unknown"
    quality: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
