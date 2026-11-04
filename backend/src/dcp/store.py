"""SQLite storage for jobs, documents, sentences and human annotations."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from dcp.models import Document, Sentence

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, name TEXT, status TEXT, created_at REAL, finished_at REAL,
    n_files INTEGER DEFAULT 0, n_docs INTEGER DEFAULT 0, n_sentences INTEGER DEFAULT 0,
    errors TEXT DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS documents (
    job_id TEXT, doc_id TEXT, source_path TEXT, source_type TEXT, n_sentences INTEGER,
    PRIMARY KEY (job_id, doc_id)
);
CREATE TABLE IF NOT EXISTS sentences (
    sent_id TEXT PRIMARY KEY, job_id TEXT, doc_id TEXT, idx INTEGER, text TEXT,
    source_path TEXT, source_type TEXT, location TEXT, lang TEXT, script TEXT,
    usable INTEGER, duplicate INTEGER, quality TEXT, labels TEXT
);
CREATE INDEX IF NOT EXISTS ix_sent_job ON sentences(job_id, doc_id, idx);
CREATE TABLE IF NOT EXISTS annotations (
    sent_id TEXT, task TEXT, label TEXT, note TEXT, annotator TEXT, updated_at REAL,
    PRIMARY KEY (sent_id, task)
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.Lock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        with self._lock:
            c = sqlite3.connect(self.path)
            c.row_factory = sqlite3.Row
            try:
                yield c
                c.commit()
            finally:
                c.close()

    # --- jobs ---
    def create_job(self, name: str, n_files: int) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.conn() as c:
            c.execute(
                "INSERT INTO jobs (job_id, name, status, created_at, n_files) VALUES (?,?,?,?,?)",
                (job_id, name, "queued", time.time(), n_files),
            )
        return job_id

    def update_job(self, job_id: str, **fields) -> None:
        if "errors" in fields:
            fields["errors"] = json.dumps(fields["errors"], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE job_id=?", (*fields.values(), job_id))

    def list_jobs(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [_job(r) for r in rows]

    def get_job(self, job_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _job(r) if r else None

    def delete_job(self, job_id: str) -> None:
        with self.conn() as c:
            c.execute(
                "DELETE FROM annotations WHERE sent_id IN (SELECT sent_id FROM sentences WHERE job_id=?)",
                (job_id,),
            )
            for t in ("sentences", "documents", "jobs"):
                c.execute(f"DELETE FROM {t} WHERE job_id=?", (job_id,))

    # --- documents / sentences ---
    def add_document(self, job_id: str, doc: Document, sentences: list[Sentence], display_path: str) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?)",
                (job_id, doc.doc_id, display_path, doc.source_type, len(sentences)),
            )
            c.executemany(
                "INSERT OR REPLACE INTO sentences VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        f"{job_id}:{s.sent_id}", job_id, s.doc_id, s.index, s.text, display_path,
                        s.source_type, s.location, s.lang, s.script, int(s.quality.get("usable", False)),
                        int(s.quality.get("duplicate", False)),
                        json.dumps(s.quality, ensure_ascii=False), json.dumps(s.labels, ensure_ascii=False),
                    )
                    for s in sentences
                ],
            )

    def list_documents(self, job_id: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM documents WHERE job_id=? ORDER BY source_path", (job_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def query_sentences(
        self, job_id: str, *, doc_id=None, lang=None, usable=None, duplicate=None, q=None,
        task=None, predicted=None, gold=None, annotated=None, disagree=None, limit=50, offset=0,
    ) -> tuple[int, list[dict]]:
        where, args = ["s.job_id=?"], [job_id]
        join = ""
        if task:
            join = "LEFT JOIN annotations a ON a.sent_id=s.sent_id AND a.task=?"
            args.insert(0, task)
        if doc_id:
            where.append("s.doc_id=?"); args.append(doc_id)
        if lang:
            where.append("s.lang=?"); args.append(lang)
        if usable is not None:
            where.append("s.usable=?"); args.append(int(usable))
        if duplicate is not None:
            where.append("s.duplicate=?"); args.append(int(duplicate))
        if q:
            where.append("s.text LIKE ?"); args.append(f"%{q}%")
        if task and predicted:
            where.append("json_extract(s.labels, '$.' || ? || '.label')=?"); args += [task, predicted]
        if task and gold:
            where.append("a.label=?"); args.append(gold)
        if task and annotated is not None:
            where.append("a.label IS NOT NULL" if annotated else "a.label IS NULL")
        if task and disagree:
            where.append("a.label IS NOT NULL AND a.label != json_extract(s.labels, '$.' || ? || '.label')")
            args.append(task)
        sql_where = " AND ".join(where)
        with self.conn() as c:
            total = c.execute(f"SELECT COUNT(*) FROM sentences s {join} WHERE {sql_where}", args).fetchone()[0]
            rows = c.execute(
                f"SELECT s.* FROM sentences s {join} WHERE {sql_where} ORDER BY s.doc_id, s.idx LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
            ids = [r["sent_id"] for r in rows]
            anns: dict[str, dict] = {}
            if ids:
                marks = ",".join("?" * len(ids))
                for a in c.execute(f"SELECT * FROM annotations WHERE sent_id IN ({marks})", ids):
                    anns.setdefault(a["sent_id"], {})[a["task"]] = {
                        "label": a["label"], "note": a["note"], "annotator": a["annotator"],
                    }
        return total, [_sentence(r, anns.get(r["sent_id"], {})) for r in rows]

    # --- annotations ---
    def annotate(self, sent_id: str, task: str, label: str | None, note: str = "", annotator: str = "") -> None:
        with self.conn() as c:
            if label is None:
                c.execute("DELETE FROM annotations WHERE sent_id=? AND task=?", (sent_id, task))
            else:
                c.execute(
                    "INSERT OR REPLACE INTO annotations VALUES (?,?,?,?,?,?)",
                    (sent_id, task, label, note, annotator, time.time()),
                )

    def predictions_vs_gold(self, job_id: str, task: str) -> list[tuple[str | None, str]]:
        with self.conn() as c:
            rows = c.execute(
                """SELECT json_extract(s.labels, '$.' || ? || '.label') AS pred, a.label AS gold
                   FROM sentences s JOIN annotations a ON a.sent_id=s.sent_id AND a.task=?
                   WHERE s.job_id=?""",
                (task, task, job_id),
            ).fetchall()
        return [(r["pred"], r["gold"]) for r in rows]

    def summary(self, job_id: str) -> dict:
        with self.conn() as c:
            def group(col):
                return {
                    (r[0] if r[0] is not None else "none"): r[1]
                    for r in c.execute(f"SELECT {col}, COUNT(*) FROM sentences WHERE job_id=? GROUP BY 1", (job_id,))
                }
            out = {
                "lang": group("lang"),
                "usable": group("usable"),
                "duplicate": group("duplicate"),
                "source_type": group("source_type"),
                "predicted": {},
            }
            for task in ("agri", "agri_subdomain"):
                out["predicted"][task] = {
                    (r[0] or "none"): r[1]
                    for r in c.execute(
                        "SELECT json_extract(labels, '$.' || ? || '.label'), COUNT(*) FROM sentences WHERE job_id=? GROUP BY 1",
                        (task, job_id),
                    )
                }
            out["annotated"] = {
                r[0]: r[1]
                for r in c.execute(
                    "SELECT a.task, COUNT(*) FROM annotations a JOIN sentences s ON s.sent_id=a.sent_id WHERE s.job_id=? GROUP BY 1",
                    (job_id,),
                )
            }
        return out

    def export(self, job_id: str):
        with self.conn() as c:
            rows = c.execute("SELECT * FROM sentences WHERE job_id=? ORDER BY doc_id, idx", (job_id,)).fetchall()
            anns: dict[str, dict] = {}
            for a in c.execute(
                "SELECT a.* FROM annotations a JOIN sentences s ON s.sent_id=a.sent_id WHERE s.job_id=?", (job_id,)
            ):
                anns.setdefault(a["sent_id"], {})[a["task"]] = {"label": a["label"], "note": a["note"]}
        for r in rows:
            yield _sentence(r, anns.get(r["sent_id"], {}))


def _job(r) -> dict:
    d = dict(r)
    d["errors"] = json.loads(d["errors"] or "[]")
    return d


def _sentence(r, annotations: dict) -> dict:
    return {
        "sent_id": r["sent_id"], "doc_id": r["doc_id"], "index": r["idx"], "text": r["text"],
        "source_path": r["source_path"], "source_type": r["source_type"], "location": r["location"],
        "lang": r["lang"], "script": r["script"], "quality": json.loads(r["quality"]),
        "labels": json.loads(r["labels"]), "annotations": annotations,
    }
