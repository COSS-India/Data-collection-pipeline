"""Client for the laya-multilingual service (services/laya/server.py), usually on the GPU box.

Enabled when DCP_LAYA_URL is set (e.g. http://127.0.0.1:8160 through an SSH tunnel). Registered
after the built-ins, so it becomes the default for its tasks; pick others with --classifiers.
"""

from __future__ import annotations

import json
import os
import urllib.request

from dcp.classify import Classifier
from dcp.models import Sentence

URL = os.environ.get("DCP_LAYA_URL", "").rstrip("/")
CHUNK = int(os.environ.get("DCP_LAYA_CHUNK", "512"))  # sentences per HTTP request


class LayaClassifier(Classifier):
    def __init__(self, task: str):
        self.task = task
        self.name = f"laya_{task}"

    def predict(self, sentences: list[Sentence]) -> list[dict]:
        out: list[dict] = []
        for i in range(0, len(sentences), CHUNK):
            texts = [s.text for s in sentences[i : i + CHUNK]]
            body = json.dumps({"texts": texts, "tasks": [self.task]}).encode()
            req = urllib.request.Request(f"{URL}/predict", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                out.extend(r[self.task] for r in json.load(resp)["results"])
        return out
