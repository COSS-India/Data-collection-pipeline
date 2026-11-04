"""Lightweight script + language detection.

Script is detected from Unicode ranges. Marathi vs Hindi (both Devanagari) is
decided by marker words and the letter ळ, which is frequent in Marathi and
almost absent in standard Hindi. This is a heuristic baseline; swap in a model
(e.g. fastText lid.176 / IndicLID) when accuracy matters.
"""

import re
import unicodedata

_SCRIPT_RANGES = [
    ("Devanagari", 0x0900, 0x097F),
    ("Devanagari", 0xA8E0, 0xA8FF),
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Oriya", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Arabic", 0x0600, 0x06FF),
    ("Latin", 0x0041, 0x024F),
]

_SCRIPT_DEFAULT_LANG = {
    "Bengali": "bn", "Gurmukhi": "pa", "Gujarati": "gu", "Oriya": "or", "Tamil": "ta",
    "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Arabic": "ur", "Latin": "en",
}

_MARATHI_MARKERS = {
    "आहे", "आहेत", "आणि", "नाही", "होते", "होता", "असे", "म्हणून", "करून", "मध्ये",
    "त्यांच्या", "त्याचा", "त्याची", "यांनी", "आपल्या", "केली", "केले", "झाले", "झाली",
    "पाहिजे", "शेतकरी", "शेतकऱ्यांना", "पीक", "पिकांचे", "व", "हे", "या", "की", "तर",
}
_HINDI_MARKERS = {
    "है", "हैं", "और", "नहीं", "था", "थे", "थी", "में", "के", "का", "की", "को", "से",
    "लिए", "किया", "किसान", "किसानों", "फसल", "यह", "वह", "भी", "पर", "गया", "गई", "रहा",
}
# Marathi glues case markers onto the noun ("कापसाची", "शेतकऱ्यांना", "शहरात");
# Hindi writes them as separate words (का/की/के/में/को), counted via markers above.
_MARATHI_SUFFIX = re.compile(
    r"(?:च्या|ची|चा|चे|ांना|ाला|ाने|ामध्ये|ावर|ातील|ून|ीत|ात|णार|तात|ावी|ावे|लेल्या|ल्यास|ल्या)$"
)
_WORD = re.compile(r"[ऀ-ॿ]+")


def detect_script(text: str) -> tuple[str, dict[str, float]]:
    counts: dict[str, int] = {}
    total = 0
    for ch in text:
        if not unicodedata.category(ch).startswith(("L", "M")):
            continue
        total += 1
        cp = ord(ch)
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["Other"] = counts.get("Other", 0) + 1
    if not total:
        return "unknown", {}
    ratios = {k: v / total for k, v in counts.items()}
    top = max(ratios, key=ratios.get)
    return (top if ratios[top] >= 0.5 else "mixed"), ratios


def detect_language(text: str) -> tuple[str, str]:
    """Return (lang_code, script)."""
    script, ratios = detect_script(text)
    if script == "mixed":
        script_for_lang = max(ratios, key=ratios.get)
    else:
        script_for_lang = script
    if script_for_lang == "Devanagari":
        return _devanagari_lang(text), script
    return _SCRIPT_DEFAULT_LANG.get(script_for_lang, "und"), script


def _devanagari_lang(text: str) -> str:
    words = _WORD.findall(text)
    mr = (
        sum(w in _MARATHI_MARKERS for w in words)
        + 2 * text.count("ळ")
        + 0.5 * sum(len(w) > 3 and bool(_MARATHI_SUFFIX.search(w)) for w in words)
    )
    hi = sum(w in _HINDI_MARKERS for w in words)
    if mr == hi:
        return "mr" if mr else "und-Deva"  # Marathi is the primary corpus language
    return "mr" if mr > hi else "hi"
