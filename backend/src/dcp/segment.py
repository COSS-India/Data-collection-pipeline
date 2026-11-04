"""Sentence segmentation for Marathi / Hindi / English mixed text.

Rule-based on purpose: handles the danda (।), double danda (॥), ?, !, and '.'
(which Marathi uses as a full stop more often than Hindi), while protecting
common abbreviations, initials, decimals and Devanagari abbreviation dots
(e.g. "डॉ.", "श्री.", "कि.मी.").
"""

import re

_TERMINATORS = "।॥?!.;"

# Abbreviations that end with '.' but don't end a sentence.
_ABBREVIATIONS = {
    # English
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "no", "fig", "approx", "dept", "govt", "kg", "km", "ha", "qtl", "rs", "inc", "ltd",
    # Marathi / Hindi
    "डॉ", "श्री", "श्रीमती", "सौ", "कु", "प्रा", "मा", "ता", "जि", "क्र", "कि", "मी",
    "ग्रॅ", "ली", "से", "रु", "स", "इ", "डॉं", "सं", "पृ", "हे", "एकर",
}

_SENT_END = re.compile(r"([।॥?!;]+|\.(?!\d))([\"'”’)\]]*)(\s+|$)")
_BULLET = re.compile(r"^\s*(?:[-*•▪●◦‣]|\(?[0-9०-९a-zA-Zअ-ह]{1,3}[.)])\s+")


def _is_abbreviation(before: str) -> bool:
    token = re.split(r"\s", before)[-1] if before else ""
    token = token.strip("(\"'“‘[")
    if not token:
        return False
    if token.lower() in _ABBREVIATIONS:
        return True
    # Single letters / initials: "A." "प." "U.S."
    bare = token.replace(".", "")
    return len(bare) == 1 or (len(bare) <= 3 and "." in token)


def split_sentences(text: str, min_chars: int = 3) -> list[str]:
    sentences: list[str] = []
    # Paragraph/line breaks are hard boundaries only when the line looks
    # complete; PDFs wrap mid-sentence, so join soft-wrapped lines first.
    for para in _paragraphs(text):
        start = 0
        for m in _SENT_END.finditer(para):
            if m.group(1) == "." and _is_abbreviation(para[start : m.start()]):
                continue
            chunk = para[start : m.end()].strip()
            start = m.end()
            if chunk:
                sentences.append(chunk)
        tail = para[start:].strip()
        if tail:
            sentences.append(tail)
    return [s for s in sentences if len(s) >= min_chars]


def _paragraphs(text: str) -> list[str]:
    paras = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        merged = [lines[0]]
        for ln in lines[1:]:
            prev = merged[-1]
            # Keep list items and lines that already ended a sentence separate.
            if _BULLET.match(ln) or prev[-1] in _TERMINATORS or prev.endswith(":"):
                merged.append(ln)
            else:
                merged[-1] = f"{prev} {ln}"
        paras.extend(_BULLET.sub("", m) for m in merged)
    return paras
