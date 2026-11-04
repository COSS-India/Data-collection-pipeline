"""Batched laya-multilingual zero-shot classification service for the corpus pipeline.

Runs on the GPU box with its own venv (see README.md). Binds to localhost; reach it over an SSH tunnel.

POST /predict  {"texts": [...], "tasks": ["agri", ...]?, "questions": {...}?}
    -> {"results": [{task: {"label": str, "score": float, "probs": {label: p}}}], "ms": n}
GET  /health
"""
import os
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_ID = os.environ.get("LAYA_MODEL", "convaiinnovations/laya-multilingual")
BATCH_SIZE = int(os.environ.get("LAYA_BATCH_SIZE", "64"))
MAX_CHARS = 600  # sentences are short; this only guards against runaway segments

# laya is very sensitive to wording. Short label descriptions beat long keyword-laden ones by a wide
# margin (agri/non-agri on a 22-sentence mr/hi/en probe: 82% short vs 64% long; 91% with the threshold).
QUESTIONS = {
    "agri": {
        "type": "choice",
        "instructions": "What is the topic of this text?",
        "criteria": {"agri": "Agriculture and farming", "non_agri": "Other topic"},
    },
    "agri_subdomain": {
        "type": "choice",
        "instructions": "Which farming topic is this text about?",
        "criteria": {
            "crop_cultivation": "Growing crops",
            "soil_water": "Soil and irrigation",
            "pest_disease": "Pests and crop diseases",
            "livestock_dairy": "Livestock and dairy",
            "fisheries": "Fisheries",
            "weather_climate": "Weather and rainfall",
            "market_prices": "Crop market prices",
            "schemes_policy": "Government schemes for farmers",
            "machinery_inputs": "Fertilizers and farm machinery",
            "horticulture": "Fruits and vegetables",
            "other_agri": "Other farming topic",
            "not_applicable": "Not about farming",
        },
    },
}
# Binary tasks: predict the positive label only above this probability (zero-shot scores are skewed
# towards it). Tune against human labels in the evaluation UI.
THRESHOLDS = {"agri": ("agri", "non_agri", float(os.environ.get("LAYA_AGRI_THRESHOLD", "0.7")))}

app = FastAPI(title="laya corpus classifier")
STATE = {"ready": False, "error": None, "model": MODEL_ID, "batch_size": BATCH_SIZE,
         "thresholds": {k: v[2] for k, v in THRESHOLDS.items()}}
LOCK = threading.Lock()
AGENT = None


def _load():
    global AGENT
    try:
        import laya

        AGENT = laya.load(MODEL_ID, device="cuda" if torch.cuda.is_available() else "cpu")
        STATE["ready"] = True
    except Exception as e:  # surface load errors on /health instead of crashing
        STATE["error"] = repr(e)


threading.Thread(target=_load, daemon=True).start()


class PredictRequest(BaseModel):
    texts: list[str]
    tasks: list[str] | None = None
    questions: dict | None = None  # override/extend QUESTIONS, same schema


@app.get("/health")
def health():
    return {**STATE, "tasks": {k: list(v["criteria"]) for k, v in QUESTIONS.items()}}


@app.post("/predict")
@torch.no_grad()
def predict(req: PredictRequest):
    if not STATE["ready"]:
        raise HTTPException(503, STATE["error"] or "laya is still loading")
    questions = {**QUESTIONS, **(req.questions or {})}
    tasks = req.tasks or list(QUESTIONS)
    unknown = [t for t in tasks if t not in questions]
    if unknown:
        raise HTTPException(400, f"Unknown tasks: {unknown}")
    if not req.texts:
        return {"results": [], "ms": 0}
    asked = {t: questions[t] for t in tasks}
    started = time.perf_counter()
    with LOCK:
        out = AGENT.predict_batch(
            [t[:MAX_CHARS] for t in req.texts], asked, batch_size=BATCH_SIZE, sort_by_length=True,
        )
    results = []
    for r in out:
        row = {}
        for task in tasks:
            probs = r["answers"][task]["probabilities"]
            label = max(probs, key=probs.get)
            if task in THRESHOLDS and not req.questions:
                pos, neg, th = THRESHOLDS[task]
                label = pos if probs[pos] >= th else neg
            row[task] = {"label": label, "score": round(probs[label], 4),
                         "probs": {k: round(v, 4) for k, v in probs.items()}}
        results.append(row)
    return {"results": results, "ms": round((time.perf_counter() - started) * 1000, 1)}
