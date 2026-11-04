"""Pluggable classifiers.

A classifier fills one *task* (e.g. "agri") on a batch of sentences. To add a
new backend (LLM, local model, ...) subclass `Classifier`, then `register()` it.
The pipeline runs every enabled classifier; the evaluation UI compares each
task's predictions against human labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dcp.models import Sentence

# Task -> allowed labels. Shared with the frontend via /api/meta.
TASKS: dict[str, list[str]] = {
    "agri": ["agri", "non_agri"],
    "agri_subdomain": [
        "crop_cultivation", "soil_water", "pest_disease", "livestock_dairy",
        "fisheries", "weather_climate", "market_prices", "schemes_policy",
        "machinery_inputs", "horticulture", "other_agri", "not_applicable",
    ],
    "lang": ["mr", "hi", "en", "other"],
    "quality": ["usable", "unusable"],
}


class Classifier(ABC):
    name: str
    task: str

    @abstractmethod
    def predict(self, sentences: list[Sentence]) -> list[dict]:
        """Return one {"label": str, "score": float, ...} per sentence."""


_REGISTRY: dict[str, Classifier] = {}


def register(clf: Classifier) -> Classifier:
    _REGISTRY[clf.name] = clf
    return clf


def get_classifiers(names: list[str] | None = None) -> list[Classifier]:
    """Named classifiers, or by default the most recently registered one per task."""
    if names is None:
        return list({c.task: c for c in _REGISTRY.values()}.values())
    return [_REGISTRY[n] for n in names]


def available() -> list[dict]:
    return [{"name": c.name, "task": c.task} for c in _REGISTRY.values()]


from dcp.classify import builtin  # noqa: E402,F401
from dcp.classify import laya as _laya  # noqa: E402

if _laya.URL:
    for _task in ("agri", "agri_subdomain"):
        register(_laya.LayaClassifier(_task))
