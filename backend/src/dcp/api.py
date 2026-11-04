"""FastAPI server: upload files, run the pipeline, browse and annotate sentences."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dcp import classify, pipeline
from dcp.extract import supported_extensions
from dcp.metrics import classification_report
from dcp.store import Store

DATA_DIR = Path(os.environ.get("DCP_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
store = Store(DATA_DIR / "dcp.sqlite3")

app = FastAPI(title="Data Collection Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("DCP_CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    return {
        "extensions": supported_extensions(),
        "tasks": classify.TASKS,
        "classifiers": classify.available(),
    }


# --- jobs ---
def _run_job(job_id: str, workdir: Path, classifiers: list[str] | None) -> None:
    errors: list[dict] = []
    counts = {"docs": 0, "sentences": 0}

    def on_document(doc, sentences):
        rel = str(Path(doc.source_path).relative_to(workdir))
        store.add_document(job_id, doc, sentences, rel)
        counts["docs"] += 1
        counts["sentences"] += len(sentences)
        store.update_job(job_id, n_docs=counts["docs"], n_sentences=counts["sentences"])

    def on_error(path, exc):
        errors.append({"file": str(path.relative_to(workdir)), "error": f"{type(exc).__name__}: {exc}"})

    store.update_job(job_id, status="running")
    try:
        pipeline.run([workdir], classifiers, on_document=on_document, on_error=on_error)
        store.update_job(job_id, status="done", errors=errors, finished_at=__import__("time").time())
    except Exception as e:
        errors.append({"file": None, "error": f"{type(e).__name__}: {e}"})
        store.update_job(job_id, status="failed", errors=errors)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@app.post("/api/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    classifiers: str = Form(""),
):
    exts = set(supported_extensions())
    workdir = Path(tempfile.mkdtemp(prefix="dcp-", dir=DATA_DIR))
    accepted, skipped = 0, []
    for f in files:
        # Keep folder structure from directory uploads, but never escape workdir.
        rel = Path(*[p for p in Path(f.filename or "upload").parts if p not in ("..", "/", "")])
        if rel.suffix.lower() not in exts:
            skipped.append(str(rel))
            continue
        dest = workdir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        accepted += 1
    if not accepted:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, f"No supported files. Supported: {', '.join(sorted(exts))}")
    job_name = name or (files[0].filename if len(files) == 1 else f"{accepted} files")
    job_id = store.create_job(job_name, accepted)
    clf = [c for c in classifiers.split(",") if c] or None
    threading.Thread(target=_run_job, args=(job_id, workdir, clf), daemon=True).start()
    return {"job_id": job_id, "accepted": accepted, "skipped": skipped}


@app.get("/api/jobs")
def list_jobs():
    return store.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {**job, "summary": store.summary(job_id), "documents": store.list_documents(job_id)}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    store.delete_job(job_id)
    return {"ok": True}


# --- sentences ---
@app.get("/api/jobs/{job_id}/sentences")
def sentences(
    job_id: str,
    doc_id: str | None = None,
    lang: str | None = None,
    usable: bool | None = None,
    duplicate: bool | None = None,
    q: str | None = None,
    task: str | None = None,
    predicted: str | None = None,
    gold: str | None = None,
    annotated: bool | None = None,
    disagree: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    total, items = store.query_sentences(
        job_id, doc_id=doc_id, lang=lang, usable=usable, duplicate=duplicate, q=q, task=task,
        predicted=predicted, gold=gold, annotated=annotated, disagree=disagree,
        limit=page_size, offset=(page - 1) * page_size,
    )
    return {"total": total, "page": page, "page_size": page_size, "items": items}


class Annotation(BaseModel):
    task: str
    label: str | None  # null clears the annotation
    note: str = ""
    annotator: str = ""


@app.put("/api/sentences/{sent_id}/annotations")
def annotate(sent_id: str, body: Annotation):
    if body.task not in classify.TASKS:
        raise HTTPException(400, f"Unknown task {body.task}")
    if body.label is not None and body.label not in classify.TASKS[body.task]:
        raise HTTPException(400, f"Label must be one of {classify.TASKS[body.task]}")
    store.annotate(sent_id, body.task, body.label, body.note, body.annotator)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/metrics")
def metrics(job_id: str):
    return {
        task: classification_report(store.predictions_vs_gold(job_id, task), labels)
        for task, labels in classify.TASKS.items()
    }


@app.get("/api/jobs/{job_id}/export")
def export(job_id: str):
    def gen():
        for row in store.export(job_id):
            yield json.dumps(row, ensure_ascii=False) + "\n"

    return StreamingResponse(
        gen(), media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.jsonl"'},
    )
