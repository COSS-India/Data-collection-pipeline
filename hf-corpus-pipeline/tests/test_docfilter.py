import json
import math

import fsspec
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marathi_agri import docfilter
from marathi_agri.classify import RequestEncoder, Result, build_doc_request, parse_doc_response
from marathi_agri.config import ClassifyConfig, PipelineConfig, SourceConfig
from marathi_agri.docfilter import screen_documents, spread_excerpt
from marathi_agri.evaluate import score
from marathi_agri.ingest import SourceAccessError, read_doc_columns
from marathi_agri.prompt import (
    DOC_FEW_SHOT,
    DOC_LABELS,
    DOC_SYSTEM_PROMPT,
    EXCERPT_GAP,
    build_doc_messages,
)

GAP = EXCERPT_GAP.strip()


def _doc(n: int) -> str:
    return " ".join(f"वाक्य {i} मध्ये पाच शब्द आहेत." for i in range(n))


def _content_words(excerpt: str) -> int:
    return sum(1 for w in excerpt.split() if w != GAP)


# --- spread read -----------------------------------------------------------


def test_spread_short_doc_returned_whole():
    assert spread_excerpt(_doc(3), 200, 100) == _doc(3)
    assert spread_excerpt("", 200, 100) == ""


def test_spread_long_doc_head_budget_and_spread():
    doc = _doc(100)  # 600 words
    ex = spread_excerpt(doc, 200, 100)
    assert ex == spread_excerpt(doc, 200, 100)  # deterministic
    assert ex.startswith("वाक्य 0 मध्ये")  # opening kept
    assert _content_words(ex) <= 200
    assert GAP in ex
    # picks reach the last part of the document, not just the opening
    picked = [int(p.split()[1]) for p in ex.split(GAP)[1:]]
    assert max(picked) >= 80
    assert picked == sorted(picked)


def test_spread_truncates_overlong_first_sentence():
    doc = " ".join(["शब्द"] * 500)  # one sentence, no punctuation
    ex = spread_excerpt(doc, 200, 100)
    assert _content_words(ex) == 100


# --- reading documents ---------------------------------------------------


def _sangraha_like(tmp_path):
    f = tmp_path / "d.parquet"
    pq.write_table(
        pa.table(
            {
                "doc_id": ["a", "b", "c", "d"],
                "text": ["एक", "दोन", "तीन", "चार"],
                "type": ["web", "pdf", "web", "speech"],
            }
        ),
        f,
        row_group_size=2,
    )
    src = SourceConfig(
        key="s",
        hf_id="org/s",
        files=["*"],
        text_column="text",
        license="CC-BY-4.0",
        url="u",
        id_column="doc_id",
        row_filter={"type": "web"},
        doc_filter=True,
    )
    return f, src


def test_read_doc_columns_applies_row_filter_and_keeps_row_numbers(tmp_path):
    f, src = _sangraha_like(tmp_path)
    df = read_doc_columns(fsspec.filesystem("file"), str(f), src)
    assert df.columns == ["source_row", "doc_id", "text_raw"]
    assert df["source_row"].to_list() == [0, 2]
    assert df["doc_id"].to_list() == ["a", "c"]


def test_read_doc_columns_missing_filter_column(tmp_path):
    f, src = _sangraha_like(tmp_path)
    src = src.model_copy(update={"row_filter": {"kind": "web"}})
    with pytest.raises(SourceAccessError, match="not found"):
        read_doc_columns(fsspec.filesystem("file"), str(f), src)


# --- doc prompt / request / response ---------------------------------------


def test_doc_prompt_structure():
    assert {label for _, label in DOC_FEW_SHOT} == set(DOC_LABELS)
    msgs = build_doc_messages("उतारा")
    assert msgs[0] == {"role": "system", "content": DOC_SYSTEM_PROMPT}
    assert msgs[-1] == {"role": "user", "content": "<doc>उतारा</doc>"}
    assert len(msgs) == 1 + 2 * len(DOC_FEW_SHOT) + 1


def test_doc_request_constrains_output():
    req = build_doc_request("x", ClassifyConfig())
    assert req["structured_outputs"] == {"choice": list(DOC_LABELS)}
    assert req["temperature"] == 0
    assert req["logprobs"] is True and req["top_logprobs"] >= 2


@pytest.mark.parametrize("text", ["उतारा", 'quote " \\ x', "a\nb"])
def test_doc_request_encoder_matches_builder(text):
    cfg = ClassifyConfig()
    assert json.loads(RequestEncoder(cfg, "doc").encode(text)) == build_doc_request(text, cfg)


def _doc_body(content, top):
    return {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {
                    "content": [
                        {
                            "token": "x",
                            "logprob": 0.0,
                            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in top],
                        }
                    ]
                },
            }
        ]
    }


def test_parse_doc_response_reads_p_agri_from_first_token():
    label, p = parse_doc_response(_doc_body("NON_AGRI", [("NON", -0.1), ("AG", -2.4), ("", -1)]))
    assert label == "NON_AGRI"
    assert p == pytest.approx(math.exp(-2.4) / (math.exp(-2.4) + math.exp(-0.1)))


def test_parse_doc_response_ignores_ambiguous_tokens_and_falls_back():
    # "A" can only start AGRI; "N" only NON_AGRI; nothing else usable -> fallback
    _, p = parse_doc_response(_doc_body("AGRI", [("A", 0.0), ("N", math.log(3))]))
    assert p == pytest.approx(0.25)
    assert parse_doc_response({"choices": [{"message": {"content": "AGRI"}}]}) == ("AGRI", 1.0)
    with pytest.raises(ValueError):
        parse_doc_response(_doc_body("MAYBE", []))


# --- screening ---------------------------------------------------------------


def test_screen_documents_keep_rule(monkeypatch, tmp_path):
    cfg = PipelineConfig(
        root=tmp_path, data_dir=tmp_path, output_dir=tmp_path, sources_file=tmp_path, workers=1
    )
    answers = {"r0": ("AGRI", 0.9), "r1": ("NON_AGRI", 0.3), "r2": ("NON_AGRI", 0.05)}
    sent: list[str] = []

    def fake_classify(items, ccfg, kind):
        assert kind == "doc"
        sent.extend(i for i, _ in items)
        return [Result(i, *answers[i], None) for i, _ in items]

    monkeypatch.setattr(docfilter, "classify_parallel", fake_classify)
    docs = pl.DataFrame(
        {
            "row_id": ["r0", "r1", "r2", "r3"],
            "text_raw": ["शेतीची बातमी आहे.", "थोडी शेती.", "क्रिकेट सामना.", "   "],
        }
    )
    out = screen_documents(docs, cfg)
    assert sent == ["r0", "r1", "r2"]  # empty document never sent
    assert out["kept"].to_list() == [True, True, False, False]  # AGRI, P>=0.2, P<0.2, empty


def test_screen_documents_raises_on_failed_requests(monkeypatch, tmp_path):
    cfg = PipelineConfig(
        root=tmp_path, data_dir=tmp_path, output_dir=tmp_path, sources_file=tmp_path, workers=1
    )
    monkeypatch.setattr(
        docfilter,
        "classify_parallel",
        lambda items, ccfg, kind: [Result(i, None, None, "HTTP 500") for i, _ in items],
    )
    with pytest.raises(RuntimeError, match="failed after retries"):
        screen_documents(pl.DataFrame({"row_id": ["r0"], "text_raw": ["शेती आहे."]}), cfg)


# --- evaluation ----------------------------------------------------------------


def test_score_precision_recall():
    gold = ["AGRI", "AGRI", "NON_AGRI", "NOT_MARATHI"]
    pred = ["AGRI", "NON_AGRI", "AGRI", "NOT_MARATHI"]
    table = score(gold, pred)
    agri = table.filter(pl.col("label") == "AGRI").row(0, named=True)
    assert (agri["precision"], agri["recall"]) == (0.5, 0.5)
    assert table.filter(pl.col("label") == "accuracy")["precision"].item() == 0.5


# --- mixed prompt versions / classifying one source only ----------------------


def _shard(path, rows):
    from marathi_agri.classify import RESULT_SCHEMA

    pl.DataFrame(
        [
            {"sentence_id": s, "label": lab, "confidence": 0.9, "error": err,
             "model_id": "m", "prompt_version": v}
            for s, lab, err, v in rows
        ],
        schema=RESULT_SCHEMA,
    ).write_parquet(path)  # fmt: skip


def test_load_results_newest_version_wins(tmp_path):
    from marathi_agri.classify import load_results

    _shard(tmp_path / "shard-00000.parquet", [("a", "AGRI", None, "v1"), ("b", "AGRI", None, "v1")])
    _shard(
        tmp_path / "shard-00001.parquet",
        [("a", "NON_AGRI", None, "v2"), ("b", None, "HTTP 500", "v2"), ("c", "AGRI", None, "v2")],
    )
    both = load_results(tmp_path, any_version=True).sort("sentence_id")
    assert both.select("sentence_id", "label", "prompt_version").rows() == [
        ("a", "NON_AGRI", "v2"),  # v2 result replaces v1
        ("b", "AGRI", "v1"),  # v2 attempt failed -> keeps its v1 label
        ("c", "AGRI", "v2"),
    ]
    current = load_results(tmp_path).sort("sentence_id")
    assert current["sentence_id"].to_list() == ["a", "c"]


def test_run_classify_only_selected_sources(monkeypatch, tmp_path):
    from marathi_agri import classify
    from marathi_agri.paths import Paths, write_parquet_atomic

    cfg = PipelineConfig(
        root=tmp_path, data_dir=tmp_path, output_dir=tmp_path, sources_file=tmp_path
    )
    paths = Paths.from_config(cfg)
    write_parquet_atomic(
        pl.DataFrame(
            {"sentence_id": ["o1", "n1", "n2"], "text": ["x", "y", "z"],
             "source_key": ["old", "new", "new"]}
        ),
        paths.unique,
    )  # fmt: skip
    sent: list[str] = []
    monkeypatch.setattr(classify, "check_server", lambda c: None)
    monkeypatch.setattr(
        classify,
        "classify_parallel",
        lambda items, c: (
            sent.extend(i for i, _ in items) or [Result(i, "AGRI", 1.0, None) for i, _ in items]
        ),
    )
    assert classify.run_classify(cfg, only=["new"])["todo"] == 2
    assert sent == ["n1", "n2"]
    assert classify.run_classify(cfg, only=["new"])["todo"] == 0  # resumable
