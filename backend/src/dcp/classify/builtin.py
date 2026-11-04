"""Baseline classifiers. The keyword one is a placeholder to exercise the
evaluation loop until a real model (LLM or local) is chosen."""

from __future__ import annotations

import re

from dcp.classify import Classifier, register
from dcp.models import Sentence

# Stems of <=2 chars must match a whole word; longer stems match as prefixes
# ("शेतकऱ्यांना" -> "शेतकर"); Devanagari stems of 4+ chars also match inside
# compounds ("बोंडअळीचा" -> "अळी" is too short, but "प्रादुर्भाव" matches).
SUBDOMAIN_LEXICON: dict[str, list[str]] = {
    "crop_cultivation": [
        "पीक", "पिक", "पेरणी", "लागवड", "बियाण", "बीज", "कापूस", "सोयाबीन", "ऊस", "गहू", "गेहूं",
        "कापस", "धान", "भात", "तांदूळ", "ज्वारी", "बाजरी", "तूर", "हरभरा", "मका", "फसल", "बुवाई", "खरीप",
        "खरीफ", "रब्बी", "कापणी", "कटाई", "उत्पादन", "crop", "sowing", "seed", "harvest", "paddy",
        "wheat", "cotton", "soybean", "sugarcane", "yield",
    ],
    "soil_water": [
        "माती", "मृदा", "मिट्टी", "जमीन", "सिंचन", "सिंचाई", "ठिबक", "तुषार", "पाणी", "विहीर",
        "soil", "irrigation", "drip", "groundwater",
    ],
    "pest_disease": [
        "कीड", "कीट", "रोग", "अळी", "बोंडअळी", "प्रादुर्भाव", "करपा", "मावा", "तुडतुडे", "बुरशी", "फवारणी", "कीटकनाशक", "pest", "disease", "fungus",
        "spray", "pesticide", "insecticide", "blight",
    ],
    "livestock_dairy": [
        "पशु", "जनावर", "गाय", "म्हैस", "भैंस", "दूध", "दुग्ध", "शेळी", "बकरी", "कुक्कुट", "मुर्गी",
        "चारा", "livestock", "cattle", "dairy", "milk", "poultry", "goat", "fodder",
    ],
    "fisheries": ["मत्स्य", "मासे", "मछली", "fish", "aquaculture"],
    "weather_climate": [
        "पाऊस", "पावसा", "बारिश", "वर्षा", "मान्सून", "मानसून", "हवामान", "मौसम", "दुष्काळ", "सूखा",
        "गारपीट", "rain", "monsoon", "weather", "drought", "hailstorm",
    ],
    "market_prices": [
        "बाजारभाव", "बाजार", "मंडी", "भाव", "दर", "हमीभाव", "एमएसपी", "क्विंटल", "मार्केट", "market",
        "mandi", "msp", "price", "quintal", "apmc",
    ],
    "schemes_policy": [
        "योजना", "अनुदान", "सबसिडी", "कर्जमाफी", "विमा", "बीमा", "किसान सम्मान", "scheme",
        "subsidy", "insurance", "pm-kisan", "loan waiver",
    ],
    "machinery_inputs": [
        "खत", "उर्वरक", "युरिया", "यूरिया", "ट्रॅक्टर", "ट्रैक्टर", "fertilizer", "urea", "tractor",
        "dap", "manure", "compost", "सेंद्रिय", "जैविक", "organic",
    ],
    "horticulture": [
        "फळ", "फल", "भाजीपाला", "सब्जी", "द्राक्ष", "डाळिंब", "आंबा", "केळी", "कांदा", "प्याज",
        "टोमॅटो", "फूल", "horticulture", "vegetable", "fruit", "onion", "grape", "pomegranate",
        "mango", "banana",
    ],
    "other_agri": [
        "शेती", "शेतकर", "शेतकऱ्", "शेत", "कृषी", "कृषि", "किसान", "खेती", "agri", "farm", "farmer",
        "agronomy", "kvk",
    ],
}

_TOKEN = re.compile(r"[\wऀ-ॿ]+")


def _hits(text: str) -> dict[str, int]:
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    low = text.lower()
    out: dict[str, int] = {}
    for sub, stems in SUBDOMAIN_LEXICON.items():
        n = 0
        for stem in stems:
            if " " in stem:
                n += low.count(stem)
            else:
                n += sum(_match(t, stem) for t in tokens)
        if n:
            out[sub] = n
    return out


def _match(token: str, stem: str) -> bool:
    if len(stem) <= 2:
        return token == stem
    if token.startswith(stem):
        return True
    return len(stem) >= 4 and "\u0900" <= stem[0] <= "\u097f" and stem in token


class KeywordAgri(Classifier):
    name = "keyword_agri"
    task = "agri"

    def predict(self, sentences: list[Sentence]) -> list[dict]:
        out = []
        for s in sentences:
            hits = _hits(s.text)
            total = sum(hits.values())
            out.append({
                "label": "agri" if total else "non_agri",
                "score": round(min(1.0, 0.5 + 0.2 * total), 2) if total else 0.5,
                "matched": hits,
            })
        return out


class KeywordAgriSubdomain(Classifier):
    name = "keyword_agri_subdomain"
    task = "agri_subdomain"

    def predict(self, sentences: list[Sentence]) -> list[dict]:
        out = []
        for s in sentences:
            hits = _hits(s.text)
            specific = {k: v for k, v in hits.items() if k != "other_agri"}
            if specific:
                label = max(specific, key=specific.get)
            elif hits:
                label = "other_agri"
            else:
                label = "not_applicable"
            total = sum(hits.values()) or 1
            out.append({"label": label, "score": round(hits.get(label, total) / total, 2)})
        return out


class HeuristicLang(Classifier):
    """Mirrors the pipeline's language detection so it can be evaluated."""

    name = "heuristic_lang"
    task = "lang"

    def predict(self, sentences: list[Sentence]) -> list[dict]:
        return [
            {"label": s.lang if s.lang in ("mr", "hi", "en") else "other", "score": None}
            for s in sentences
        ]


class HeuristicQuality(Classifier):
    name = "heuristic_quality"
    task = "quality"

    def predict(self, sentences: list[Sentence]) -> list[dict]:
        return [
            {"label": "usable" if s.quality.get("usable") else "unusable", "score": None}
            for s in sentences
        ]


for _c in (KeywordAgri(), KeywordAgriSubdomain(), HeuristicLang(), HeuristicQuality()):
    register(_c)
