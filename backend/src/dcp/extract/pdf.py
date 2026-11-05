"""PDF extraction from the embedded text layer with PyMuPDF. No OCR: scanned pages
(no text layer) yield nothing and are logged.

Many Marathi PDFs use legacy non-Unicode fonts (Shree-Lipi, Kruti Dev, ...) or broken
font maps. Their text extracts as gibberish; the quality stage flags it
(`broken_pdf_encoding`, `possibly_legacy_font`) so it stays out of the corpus.
"""

import logging
from pathlib import Path

from dcp.extract import register
from dcp.models import TextBlock

log = logging.getLogger(__name__)


@register("pdf", ".pdf")
def extract_pdf(path: Path) -> list[TextBlock]:
    import pymupdf

    blocks = []
    no_text = 0
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc, 1):
            # sort=True gives reading order for multi-column layouts.
            text = page.get_text("text", sort=True)
            if text.strip():
                blocks.append(TextBlock(text, f"page={i}"))
            else:
                no_text += 1
    if no_text:
        log.info("%s: skipped %d page(s) with no text layer", path.name, no_text)
    return blocks
