"""Per-sentence quality signals. Cheap heuristics; the evaluation UI tells you
which thresholds are worth tightening."""

import re
import unicodedata

_URL = re.compile(r"https?://|www\.|\S+@\S+\.\w+")
# Latin letters typical of legacy-font Devanagari (Kruti Dev / Shree-Lipi) output.
_LEGACY_FONT = re.compile(r"[a-zA-Z]{0,3}[\\\]\[{}~`^|<>@#]+[a-zA-Z]{0,3}")
# Broken PDF font maps turn Devanagari conjuncts into Latin-Extended / IPA /
# symbol glyphs inside otherwise-Devanagari words: "खरʟप", "महəďयात".
_BAD_GLYPH_IN_DEVA = re.compile(r"(?<=[\u0900-\u097F])[\u0100-\u036F\u2100-\u2BFF\uE000-\uF8FF]|[\u0100-\u036F\uE000-\uF8FF](?=[\u0900-\u097F])")


def assess(text: str, script: str, min_words: int = 3, max_chars: int = 1000) -> dict:
    letters = sum(unicodedata.category(c).startswith(("L", "M")) for c in text)
    digits = sum(c.isdigit() for c in text)
    n = max(len(text), 1)
    words = text.split()
    flags = []
    if len(words) < min_words:
        flags.append("too_short")
    if len(text) > max_chars:
        flags.append("too_long")
    if letters / n < 0.5:
        flags.append("low_alpha")
    if digits / n > 0.3:
        flags.append("numeric_heavy")
    if _URL.search(text):
        flags.append("has_url_or_email")
    if script == "mixed":
        flags.append("mixed_script")
    garbled = len("".join(_LEGACY_FONT.findall(text))) / n
    if garbled > 0.15:
        flags.append("possibly_legacy_font")
    bad_glyphs = len(_BAD_GLYPH_IN_DEVA.findall(text))
    if bad_glyphs:
        flags.append("broken_pdf_encoding")
    if text[-1:] not in "।॥?!.\"'”’)":
        flags.append("no_terminal_punct")
    hard = {"too_short", "low_alpha", "numeric_heavy", "possibly_legacy_font", "broken_pdf_encoding", "too_long"}
    return {
        "usable": not (hard & set(flags)),
        "flags": flags,
        "alpha_ratio": round(letters / n, 3),
        "garbled_ratio": round(garbled, 3),
        "bad_glyphs": bad_glyphs,
        "n_words": len(words),
    }
