"""Stage 6: label every unique sentence AGRI / NON_AGRI / NOT_MARATHI with Gemma 4 on vLLM.

Results are keyed by sentence_id (not row position), so:
- a crashed or interrupted run resumes where it stopped;
- adding sources later only classifies the new sentences;
- changing the prompt version or model re-classifies everything under the new version.

The same async client also serves the document filter (task kind "doc"): a different
prompt, label set and parser, but identical batching, retries and multi-process fan-out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import aiohttp
import polars as pl

from .config import ClassifyConfig, PipelineConfig
from .paths import Paths, require, write_parquet_atomic
from .prompt import (
    AGRI,
    DOC_LABELS,
    LABELS,
    NON_AGRI,
    PROMPT_HISTORY,
    PROMPT_VERSION,
    build_doc_messages,
    build_messages,
    format_doc_turn,
    format_user_turn,
)

log = logging.getLogger(__name__)

RESULT_SCHEMA = {
    "sentence_id": pl.Utf8,
    "label": pl.Utf8,
    "confidence": pl.Float32,
    "error": pl.Utf8,
    "model_id": pl.Utf8,
    "prompt_version": pl.Utf8,
}

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_JSON_HEADERS = {"Content-Type": "application/json"}


class ServerNotReady(RuntimeError):
    pass


TaskKind = Literal["sentence", "doc"]


@dataclass(frozen=True)
class Result:
    item_id: str
    label: str | None
    # sentence task: probability of the emitted label; doc task: P(AGRI)
    confidence: float | None
    error: str | None


def build_request(text: str, cfg: ClassifyConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "messages": build_messages(text),
        "temperature": 0,
        "max_tokens": cfg.max_tokens,
        "logprobs": True,
        "structured_outputs": {"choice": list(LABELS)},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_response(body: dict[str, Any]) -> tuple[str, float | None]:
    """Return (label, confidence). Confidence = model probability of the emitted label,
    i.e. exp(sum of the generated tokens' logprobs)."""
    choice = body["choices"][0]
    label = (choice["message"].get("content") or "").strip()
    if label not in LABELS:
        raise ValueError(f"unexpected label {label!r}")
    tokens = ((choice.get("logprobs") or {}).get("content")) or []
    logprobs = [t["logprob"] for t in tokens if t.get("logprob") is not None]
    if not logprobs:
        return label, None
    total = sum(logprobs)
    return label, (min(1.0, math.exp(total)) if math.isfinite(total) else 0.0)


def build_doc_request(excerpt: str, cfg: ClassifyConfig) -> dict[str, Any]:
    return {
        "model": cfg.model,
        "messages": build_doc_messages(excerpt),
        "temperature": 0,
        "max_tokens": cfg.max_tokens,
        "logprobs": True,
        "top_logprobs": 5,
        "structured_outputs": {"choice": list(DOC_LABELS)},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_doc_response(body: dict[str, Any]) -> tuple[str, float]:
    """Return (label, P(AGRI)).

    P(AGRI) is read from the first generated token's top logprobs: each candidate token is
    credited to the label it can start ("AG" -> AGRI, "NON" -> NON_AGRI); tokens that could
    start both or neither are ignored. The two masses are normalized against each other. If
    no candidate is usable, P(AGRI) is 1.0 for an AGRI answer and 0.0 otherwise.
    """
    choice = body["choices"][0]
    label = (choice["message"].get("content") or "").strip()
    if label not in DOC_LABELS:
        raise ValueError(f"unexpected label {label!r}")
    fallback = 1.0 if label == AGRI else 0.0
    tokens = ((choice.get("logprobs") or {}).get("content")) or []
    if not tokens:
        return label, fallback
    mass = {AGRI: 0.0, NON_AGRI: 0.0}
    for cand in tokens[0].get("top_logprobs") or []:
        tok, lp = cand.get("token") or "", cand.get("logprob")
        if not tok or lp is None or not math.isfinite(lp):
            continue
        starts = [lab for lab in DOC_LABELS if lab.startswith(tok)]
        if len(starts) == 1:
            mass[starts[0]] += math.exp(lp)
    total = mass[AGRI] + mass[NON_AGRI]
    return label, (mass[AGRI] / total if total > 0 else fallback)


# kind -> (request builder, user-turn formatter, response parser)
_TASKS = {
    "sentence": (build_request, format_user_turn, parse_response),
    "doc": (build_doc_request, format_doc_turn, parse_doc_response),
}


def check_server(cfg: ClassifyConfig) -> None:
    try:
        with urllib.request.urlopen(f"{cfg.base_url}/models", timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError) as exc:
        raise ServerNotReady(
            f"vLLM server not reachable at {cfg.base_url} ({exc}). Start it with scripts/serve.sh"
        ) from exc
    served = [m["id"] for m in data.get("data", [])]
    if cfg.model not in served:
        raise ServerNotReady(f"server serves {served}, config expects {cfg.model}")


class RequestEncoder:
    """Pre-serializes the request JSON around the text slot.

    The system prompt and few-shot turns are identical for every request; encoding them
    once (instead of json-dumping ~50 messages per item) keeps the client from becoming
    the bottleneck.
    """

    _MARKER = "\u0000SENTENCE\u0000"

    def __init__(self, cfg: ClassifyConfig, kind: TaskKind = "sentence") -> None:
        build, fmt, _ = _TASKS[kind]
        self._fmt = fmt
        full = json.dumps(build(self._MARKER, cfg), ensure_ascii=False)
        slot = json.dumps(fmt(self._MARKER), ensure_ascii=False)
        if full.count(slot) != 1:
            raise RuntimeError("could not locate the sentence slot in the request template")
        self._head, self._tail = full.split(slot)

    def encode(self, text: str) -> bytes:
        return (self._head + json.dumps(self._fmt(text), ensure_ascii=False) + self._tail).encode(
            "utf-8"
        )


async def _classify_one(
    session: aiohttp.ClientSession,
    url: str,
    body: bytes,
    sid: str,
    cfg: ClassifyConfig,
    parse: Any = parse_response,
) -> Result:
    last_error = "unknown"
    for attempt in range(cfg.max_retries + 1):
        try:
            async with session.post(url, data=body, headers=_JSON_HEADERS) as resp:
                if resp.status in _RETRY_STATUS:
                    last_error = f"HTTP {resp.status}"
                elif resp.status >= 400:
                    detail = (await resp.text())[:200]
                    return Result(sid, None, None, f"HTTP {resp.status}: {detail}")
                else:
                    label, conf = parse(await resp.json(content_type=None))
                    return Result(sid, label, conf, None)
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except (ValueError, KeyError, IndexError) as exc:
            return Result(sid, None, None, f"bad response: {exc}")
        await asyncio.sleep(min(30.0, 2.0**attempt) + random.random())
    return Result(sid, None, None, f"gave up after retries: {last_error}")


async def classify_batch(
    items: list[tuple[str, str]],
    cfg: ClassifyConfig,
    concurrency: int | None = None,
    progress_every: int = 10_000,
    kind: TaskKind = "sentence",
) -> list[Result]:
    """Classify (item_id, text) pairs concurrently; output order matches input order."""
    limit = concurrency or cfg.concurrency
    sem = asyncio.Semaphore(limit)
    encoder = RequestEncoder(cfg, kind)
    parse = _TASKS[kind][2]
    url = f"{cfg.base_url}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
    connector = aiohttp.TCPConnector(limit=limit)
    start = time.monotonic()
    done = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def run(sid: str, text: str) -> Result:
            nonlocal done
            async with sem:
                res = await _classify_one(session, url, encoder.encode(text), sid, cfg, parse)
            done += 1
            if done % progress_every == 0:
                rate = done / (time.monotonic() - start)
                log.info("  %d/%d classified (%.0f %ss/s)", done, len(items), rate, kind)
            return res

        return await asyncio.gather(*(run(sid, text) for sid, text in items))


def _classify_worker(
    items: list[tuple[str, str]], cfg_json: str, concurrency: int, kind: TaskKind = "sentence"
) -> list[Result]:
    """Entry point for one client process (own event loop)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = ClassifyConfig.model_validate_json(cfg_json)
    return asyncio.run(classify_batch(items, cfg, concurrency=concurrency, kind=kind))


def classify_parallel(
    items: list[tuple[str, str]], cfg: ClassifyConfig, kind: TaskKind = "sentence"
) -> list[Result]:
    """Split items over `client_processes` processes; results keep the input order."""
    n = max(1, min(cfg.client_processes, len(items)))
    if n == 1:
        return asyncio.run(classify_batch(items, cfg, kind=kind))
    per_proc = max(1, cfg.concurrency // n)
    size = math.ceil(len(items) / n)
    chunks = [items[i : i + size] for i in range(0, len(items), size)]
    cfg_json = cfg.model_dump_json()
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        parts = pool.map(
            _classify_worker,
            chunks,
            [cfg_json] * len(chunks),
            [per_proc] * len(chunks),
            [kind] * len(chunks),
        )
        return [r for part in parts for r in part]


def _results_frame(results: list[Result], cfg: ClassifyConfig) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sentence_id": [r.item_id for r in results],
            "label": [r.label for r in results],
            "confidence": [r.confidence for r in results],
            "error": [r.error for r in results],
            "model_id": [cfg.model] * len(results),
            "prompt_version": [PROMPT_VERSION] * len(results),
        },
        schema=RESULT_SCHEMA,
    )


def load_results(
    classified_dir: Path, model: str | None = None, any_version: bool = False
) -> pl.DataFrame:
    """Successful results, one per sentence (and only for `model`, if given).

    any_version=False: only the current PROMPT_VERSION (what `classify` treats as done).
    any_version=True: the newest prompt version available for each sentence (PROMPT_HISTORY
    order), so sentences classified under an earlier prompt keep that label.
    Within one version, later shards win.
    """
    files = sorted(classified_dir.glob("shard-*.parquet"))
    if not files:
        return pl.DataFrame(schema=RESULT_SCHEMA)
    df = pl.read_parquet(files).filter(pl.col("error").is_null())
    if not any_version:
        df = df.filter(pl.col("prompt_version") == PROMPT_VERSION)
    if model:
        df = df.filter(pl.col("model_id") == model)
    rank = {v: i for i, v in enumerate(PROMPT_HISTORY)}
    df = (
        df.with_row_index("_order")
        .with_columns(
            _rank=pl.col("prompt_version").replace_strict(rank, default=-1, return_dtype=pl.Int32)
        )
        .sort("_rank", "_order")
    )
    return df.unique("sentence_id", keep="last", maintain_order=True).drop("_order", "_rank")


def _next_shard_index(classified_dir: Path) -> int:
    existing = [int(p.stem.split("-")[1]) for p in classified_dir.glob("shard-*.parquet")]
    return max(existing, default=-1) + 1


def run_classify(
    cfg: PipelineConfig, limit: int | None = None, only: list[str] | None = None
) -> dict[str, int]:
    """Classify every unique sentence with no result under the current prompt yet.

    only: restrict to sentences whose kept copy comes from these source keys (e.g. a newly
    added source), leaving other sentences with the labels they already have.
    """
    paths = Paths.from_config(cfg)
    ccfg = cfg.classify
    check_server(ccfg)

    unique = pl.read_parquet(
        require(paths.unique, "dedup"), columns=["sentence_id", "text", "source_key"]
    )
    if only:
        unique = unique.filter(pl.col("source_key").is_in(only))
    unique = unique.drop("source_key")
    done_ids = load_results(paths.classified_dir, ccfg.model).select("sentence_id")
    todo = unique.join(done_ids, on="sentence_id", how="anti")
    if limit is not None:
        todo = todo.head(limit)
    log.info(
        "classify: %d unique, %d already done, %d to do",
        unique.height,
        done_ids.height,
        todo.height,
    )

    paths.classified_dir.mkdir(parents=True, exist_ok=True)
    shard = _next_shard_index(paths.classified_dir)
    errors = 0
    for offset in range(0, todo.height, ccfg.shard_size):
        part = todo.slice(offset, ccfg.shard_size)
        t0 = time.monotonic()
        results = classify_parallel(list(part.iter_rows()), ccfg)
        frame = _results_frame(results, ccfg)
        write_parquet_atomic(frame, paths.classified_dir / f"shard-{shard:05d}.parquet")
        n_err = frame["error"].is_not_null().sum()
        errors += n_err
        secs = max(time.monotonic() - t0, 1e-9)
        log.info(
            "shard %05d: %d sentences in %.0fs (%.0f/s), %d errors",
            shard, part.height, secs, part.height / secs, n_err,
        )  # fmt: skip
        shard += 1
    return {"todo": todo.height, "errors": errors}


def run_trial(cfg: PipelineConfig, n: int, seed: int = 0) -> Path:
    """Classify a random sample (not saved as results) and write output/trial.csv for review."""
    paths = Paths.from_config(cfg)
    check_server(cfg.classify)
    unique = pl.read_parquet(
        require(paths.unique, "dedup"), columns=["sentence_id", "text", "source_key"]
    )
    sample = unique.sample(n=min(n, unique.height), seed=seed)
    t0 = time.monotonic()
    results = classify_parallel(
        list(sample.select("sentence_id", "text").iter_rows()), cfg.classify
    )
    elapsed = time.monotonic() - t0
    frame = sample.join(_results_frame(results, cfg.classify), on="sentence_id", how="left")
    out = paths.output / "trial.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.select("source_key", "label", "confidence", "error", "text").write_csv(out)
    log.info(
        "trial: %d sentences in %.1fs = %.0f sentences/s -> %s",
        sample.height,
        elapsed,
        sample.height / elapsed,
        out,
    )
    log.info(
        "trial labels: %s", frame["label"].value_counts().sort("count", descending=True).rows()
    )
    return out
