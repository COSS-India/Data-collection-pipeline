"""extract -> normalize -> segment -> language/quality -> dedupe -> classify."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Iterable

from dcp import classify, extract as extractors
from dcp.langid import detect_language
from dcp.models import Document, Sentence
from dcp.normalize import dedup_key, normalize
from dcp.quality import assess
from dcp.segment import split_sentences

log = logging.getLogger(__name__)


def iter_files(inputs: Iterable[str | Path]) -> list[Path]:
    exts = set(extractors.supported_extensions())
    files: list[Path] = []
    for p in map(Path, inputs):
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in exts)
        elif p.is_file():
            files.append(p)
    return files


def document_to_sentences(doc: Document) -> list[Sentence]:
    out: list[Sentence] = []
    for block in doc.blocks:
        for text in split_sentences(normalize(block.text)):
            lang, script = detect_language(text)
            idx = len(out)
            out.append(Sentence(
                sent_id=hashlib.sha1(f"{doc.doc_id}:{idx}:{text}".encode()).hexdigest()[:16],
                text=text,
                doc_id=doc.doc_id,
                source_path=doc.source_path,
                source_type=doc.source_type,
                location=block.location,
                index=idx,
                lang=lang,
                script=script,
                quality=assess(text, script),
            ))
    return out


def mark_duplicates(sentences: list[Sentence], seen: set[str] | None = None) -> None:
    seen = set() if seen is None else seen
    for s in sentences:
        key = dedup_key(s.text)
        s.quality["duplicate"] = key in seen
        seen.add(key)


def run_classifiers(sentences: list[Sentence], names: list[str] | None = None) -> None:
    for clf in classify.get_classifiers(names):
        for s, pred in zip(sentences, clf.predict(sentences)):
            s.labels[clf.task] = {**pred, "classifier": clf.name}


def run(
    inputs: Iterable[str | Path],
    classifiers: list[str] | None = None,
    on_document: Callable[[Document, list[Sentence]], None] | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> list[Sentence]:
    all_sentences: list[Sentence] = []
    seen: set[str] = set()
    for path in iter_files(inputs):
        try:
            doc = extractors.extract(path)
        except Exception as e:  # one bad file shouldn't kill the run
            log.exception("Failed to extract %s", path)
            if on_error:
                on_error(path, e)
            continue
        sentences = document_to_sentences(doc)
        mark_duplicates(sentences, seen)
        run_classifiers(sentences, classifiers)
        if on_document:
            on_document(doc, sentences)
        all_sentences.extend(sentences)
    return all_sentences
