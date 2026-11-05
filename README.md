# Data Collection Pipeline

Turns file-based text material with directly extractable text (no OCR) into clean, sentence-level data with metadata, plus a web UI to evaluate the output.

```
files ──► extract ──► normalize ──► sentence split ──► language + quality ──► dedupe ──► classify ──► JSONL / SQLite
                                                                                                   │
                                                                     Next.js UI: review, label, metrics
```

- **backend/**: Python package `dcp` (pipeline, CLI, FastAPI server)
- **frontend/**: Next.js evaluation app (upload runs, browse sentences, set human labels, see accuracy / F1 / confusion matrices)

Marathi is the primary language; Hindi, English and other scripts are supported.

## Quick start

```bash
# backend (Python ≥3.10, uv)
cd backend
uv sync
uv run dcp serve --port 8000          # API on http://127.0.0.1:8000

# frontend (Node ≥20), in another terminal
cd frontend
npm install
npm run dev                           # UI on http://localhost:3000, proxies /api to :8000
```

Set `DCP_API_URL` for the frontend if the backend isn't on `127.0.0.1:8000`. Backend data (SQLite plus temporary uploads) lives in `backend/data/`, or wherever `DCP_DATA_DIR` points.

CLI without the UI:

```bash
uv run dcp run path/to/files_or_dirs -o output/sentences.jsonl [--usable-only]
```

## Pipeline stages

| Stage | Module | Notes |
|---|---|---|
| Extract | `dcp/extract/` | txt/md, csv/tsv, json/jsonl, html, epub, pdf, docx, pptx, xlsx, odt/odp. Register new types with `@register("type", ".ext")`. |
| PDFs | `extract/pdf.py` | Text layer only, no OCR. Scanned pages are skipped (a scanned-only PDF shows 0 sentences in the UI). |
| Normalize | `normalize.py` | NFC, invisible chars, de-hyphenation, `|` → `।`. Keeps ZWJ/ZWNJ, which matter for Marathi conjuncts. |
| Segment | `segment.py` | Splits on `।` `॥` `?` `!` `.`; protects abbreviations (`डॉ.`, `श्री.`, `Dr.`), decimals and initials; rejoins PDF soft-wrapped lines; separates list items. |
| Language | `langid.py` | Script by Unicode range; Marathi vs Hindi by marker words, `ळ` frequency and Marathi's attached case suffixes. Heuristic baseline: swap in IndicLID or fastText when needed. |
| Quality | `quality.py` | Flags: too short/long, low alphabetic ratio, numeric-heavy, URLs, mixed script, **broken PDF font encoding** (very common with Marathi PDFs), legacy fonts (Kruti Dev/Shree-Lipi). Exact and near-exact duplicates are marked across a run. |
| Classify | `classify/` | Pluggable. Tasks and label sets live in `classify.TASKS`: `agri`, `agri_subdomain`, `lang`, `quality`. |

## Classifiers

The built-in `keyword_agri` / `keyword_agri_subdomain` classifiers are a **lexicon baseline**, there so the evaluation loop has something to measure until the real classifier is chosen. Adding one (LLM, local model, …):

```python
from dcp.classify import Classifier, register

class MyAgri(Classifier):
    name = "my_agri"
    task = "agri"                       # must be a key of TASKS
    def predict(self, sentences):
        return [{"label": "agri", "score": 0.9} for s in sentences]

register(MyAgri())
```

Each sentence keeps one prediction per task (with the classifier's name), so re-running with a different classifier and labelling the same data gives directly comparable metrics.

## Evaluation UI

- **Overview**: counts by language, quality, predicted label, source type; per-document sentence counts (0 = extraction problem).
- **Review**: filter by task, prediction, labelled/unlabelled, errors only, language, quality, document or text. Keyboard: `j`/`k` move, `1`–`9` set label, `a` accept prediction, `x` clear, `n`/`p` page.
- **Metrics**: accuracy, macro-F1, per-class P/R/F1 and a confusion matrix per task; click a cell to review those sentences.
- **Export**: JSONL with predictions and human labels.

## Tests

```bash
cd backend && uv run pytest
```
