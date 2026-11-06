"""Pure text functions: cleaning, sentence splitting, measurements, dedup keys.

Everything here is deterministic and side-effect free so it can be unit-tested and
run in worker processes.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from .config import LengthBuckets

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# Format/control characters that are kept: newline/tab (handled later) and
# ZWNJ/ZWJ, which are meaningful in Marathi (e.g. eyelash reph र्‍).
_KEEP_FORMAT = {"\n", "\t", "‌", "‍"}

_DEVANAGARI_RANGES = ((0x0900, 0x097F), (0xA8E0, 0xA8FF), (0x1CD0, 0x1CFF))

# ASR / annotation tags: [noise], <unk>, {laugh}, (laughter). Only removed when the
# content has no Devanagari, so Marathi text inside brackets is preserved.
_BRACKET_TAG_RE = re.compile(
    r"\[[^\]\nऀ-ॿ]{0,40}\]"
    r"|<[^>\nऀ-ॿ]{0,40}>"
    r"|\{[^}\nऀ-ॿ]{0,40}\}"
    r"|\(\s*[A-Za-z_][A-Za-z_ \-]{0,30}\)"
)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PIPE_DANDA_RE = re.compile(r"(?<=[ऀ-ॿ])\s*\|")
_REPEATED_DANDA_RE = re.compile(r"।{2,}")
_INLINE_SPACE_RE = re.compile(r"[ \t  -   　]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([।?!,.;:])")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")


def _strip_format_chars(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if ch in _KEEP_FORMAT or unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs")
    )


def clean_text(text: str | None) -> str:
    """Normalize one raw row. Returns "" when nothing usable remains."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_format_chars(text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _BRACKET_TAG_RE.sub(" ", text)
    text = text.replace("॥", "।")
    text = _PIPE_DANDA_RE.sub("।", text)
    text = _REPEATED_DANDA_RE.sub("।", text)

    lines = []
    for line in text.split("\n"):
        line = _INLINE_SPACE_RE.sub(" ", line)
        line = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", line).strip()
        if line:
            lines.append(line)
    return _MULTI_NEWLINE_RE.sub("\n", "\n".join(lines))


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Tokens that end with "." but do not end a sentence.
ABBREVIATIONS = frozenset(
    {
        # Marathi abbreviations with 2+ base letters (single-letter ones such as डॉ., ता.,
        # जि., रु., मा. are caught by the base-letter rule in _is_abbreviation)
        "श्री", "श्रीमती", "प्रा", "क्र", "स्व", "प्र", "उदा", "ग्रा",
        # Latin
        "Dr", "Mr", "Mrs", "Ms", "St", "Prof", "No", "vs", "etc", "Rs", "Sr", "Jr",
    }
)  # fmt: skip

_BOUNDARY_RE = re.compile(r"([।?!]+|\.)([\"'”’»)\]]*)(?=\s|$)")


def _base_letter_count(token: str) -> int:
    """Number of base letters (consonants / independent vowels / Latin letters), ignoring
    vowel signs and viramas. 'डॉ' -> 1, 'श्री' -> 2."""
    return sum(
        1 for ch in token if unicodedata.category(ch) == "Lo" or (ch.isascii() and ch.isalpha())
    )


def _is_abbreviation(token: str) -> bool:
    if not token:
        return True
    if token.endswith("."):  # "..." ellipsis: a real boundary
        return False
    if "." in token:  # dotted abbreviation such as इ.स or कि.मी
        return True
    if token.isdigit() or all(unicodedata.category(ch) == "Nd" for ch in token):
        return True  # list numbering "1." / "१."
    return token in ABBREVIATIONS or _base_letter_count(token) <= 1


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences on । ? ! and sentence-final '.'.

    Newlines are treated as hard boundaries. Text without punctuation is returned whole.
    """
    sentences: list[str] = []
    for line in text.split("\n"):
        start = 0
        for match in _BOUNDARY_RE.finditer(line):
            if match.group(1) == ".":
                token = line[start : match.start()].rsplit(maxsplit=1)
                if _is_abbreviation(token[-1] if token else ""):
                    continue
            end = match.end()
            piece = line[start:end].strip()
            if piece:
                sentences.append(piece)
            start = end
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

_END_PUNCT = ("।", "?", "!", ".")
_TRAILING_QUOTES = "\"'”’»)]"


def word_count(text: str) -> int:
    return len(text.split())


def devanagari_ratio(text: str) -> float:
    """Share of letters/marks that are Devanagari. Digits, punctuation, spaces ignored."""
    letters = 0
    deva = 0
    for ch in text:
        if unicodedata.category(ch)[0] in ("L", "M"):
            letters += 1
            cp = ord(ch)
            if any(lo <= cp <= hi for lo, hi in _DEVANAGARI_RANGES):
                deva += 1
    return deva / letters if letters else 0.0


def has_end_punct(text: str) -> bool:
    return text.rstrip(_TRAILING_QUOTES).endswith(_END_PUNCT)


def length_bucket(words: int, buckets: LengthBuckets) -> str:
    if words <= buckets.short_max:
        return "short"
    if words <= buckets.medium_max:
        return "medium"
    if words <= buckets.long_max:
        return "long"
    return "very_long"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def dedup_key(text: str) -> str:
    """Key under which two sentences count as duplicates: NFC, punctuation removed,
    case-folded, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return " ".join(text.casefold().split())


def sentence_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
