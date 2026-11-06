import json
import math

import pytest

from marathi_agri.classify import RequestEncoder, build_request, parse_response
from marathi_agri.config import ClassifyConfig
from marathi_agri.prompt import FEW_SHOT, LABELS, SYSTEM_PROMPT, build_messages


def test_few_shot_labels_valid_and_all_labels_covered():
    labels = {label for _, label in FEW_SHOT}
    assert labels <= set(LABELS)
    assert labels == set(LABELS)


def test_messages_structure():
    msgs = build_messages("पीक आले")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(msgs) == 1 + 2 * len(FEW_SHOT) + 1
    assert msgs[-1] == {"role": "user", "content": "<text>पीक आले</text>"}
    # roles alternate user/assistant after the system turn
    roles = [m["role"] for m in msgs[1:]]
    assert roles == ["user", "assistant"] * len(FEW_SHOT) + ["user"]


def test_prefix_identical_across_sentences():
    a, b = build_messages("एक"), build_messages("दोन")
    assert a[:-1] == b[:-1]


def test_build_request_constrains_output():
    req = build_request("पीक", ClassifyConfig())
    assert req["temperature"] == 0
    assert req["structured_outputs"] == {"choice": list(LABELS)}
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert req["logprobs"] is True


def _body(content, logprobs):
    return {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {"content": [{"token": "x", "logprob": lp} for lp in logprobs]},
            }
        ]
    }


def test_parse_response_confidence():
    label, conf = parse_response(_body("NON_AGRI", [-0.1, -0.05, 0.0]))
    assert label == "NON_AGRI"
    assert conf == pytest.approx(math.exp(-0.15))


def test_parse_response_strips_whitespace_and_handles_missing_logprobs():
    label, conf = parse_response({"choices": [{"message": {"content": " AGRI\n"}}]})
    assert (label, conf) == ("AGRI", None)


def test_parse_response_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_response(_body("MAYBE", [-0.1]))


@pytest.mark.parametrize("text", ["पीक आले", 'quote " and \\ backslash', "नवी\nओळ", "<text>"])
def test_request_encoder_matches_build_request(text):
    cfg = ClassifyConfig()
    assert json.loads(RequestEncoder(cfg).encode(text)) == build_request(text, cfg)
