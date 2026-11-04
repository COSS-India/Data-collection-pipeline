"""PDF extraction with PyMuPDF, falling back to OCR for scanned pages.

Many Marathi PDFs use legacy non-Unicode fonts (Shree-Lipi, Kruti Dev, ...).
Text from those extracts as Latin gibberish; the quality stage flags it via
`garbled_ratio`, and OCR is the usual fix.
"""

import os
from pathlib import Path

from dcp.extract import register
from dcp.models import TextBlock

OCR_LANGS = os.environ.get("DCP_OCR_LANGS", "mar+hin+eng")
MIN_CHARS_PER_PAGE = 25


@register("pdf", ".pdf")
def extract_pdf(path: Path) -> list[TextBlock]:
    import pymupdf

    blocks = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc, 1):
            # sort=True gives reading order for multi-column layouts.
            text = page.get_text("text", sort=True)
            if len(text.strip()) < MIN_CHARS_PER_PAGE:
                text = _ocr_page(page) or text
                loc = f"page={i}:ocr"
            else:
                loc = f"page={i}"
            blocks.append(TextBlock(text, loc))
    return blocks


def _ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANGS)
    except pytesseract.TesseractError:
        return ""
