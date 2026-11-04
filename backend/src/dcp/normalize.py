"""Unicode and whitespace normalization, tuned for Devanagari text."""

import re
import unicodedata

# Zero-width / BOM chars that leak in from PDFs and Word. ZWJ/ZWNJ (U+200D/U+200C)
# are kept: they change conjunct rendering in Devanagari (e.g. Marathi eyelash ra).
_INVISIBLE = re.compile("[​⁠﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Hyphenated line breaks: "agri-\nculture" -> "agriculture"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t  -   　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")

# Common ASCII stand-ins for the danda.
_PIPE_DANDA = re.compile(r"(?<=[ऀ-ॿ])\s*\|(?!\|)")
_DOUBLE_PIPE = re.compile(r"(?<=[ऀ-ॿ])\s*\|\|")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _DOUBLE_PIPE.sub("॥", text)
    text = _PIPE_DANDA.sub("।", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def dedup_key(text: str) -> str:
    """Key for near-exact duplicate detection: casefold, drop punctuation and spaces."""
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LMN")
