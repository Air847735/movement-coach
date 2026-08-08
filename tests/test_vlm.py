"""VLM client: reply parsing and transport failure reporting.

No GPU or model server is involved; the HTTP layer is stubbed so these run
anywhere.
"""

from __future__ import annotations

import pytest
import requests

from movement_coach.errors import VLMError
from movement_coach.vlm import OllamaVLM, _bullets, _json_string_array


class _Response:
    def __init__(self, status_code=200, payload=None, text="", json_error=False):
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def vlm():
    return OllamaVLM(seed=1, timeout=5)


def _stub_post(monkeypatch, response):
    monkeypatch.setattr(
        "movement_coach.vlm.requests.post", lambda *a, **k: response
    )


# -- reply parsing ---------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("- knees cave in\n- shallow depth", ("knees cave in", "shallow depth")),
        ("* one\n* two", ("one", "two")),
        ("1. first\n2) second", ("first", "second")),
        ("no bullets here", ("no bullets here",)),
        ("", ()),
    ],
)
def test_bullet_parsing(text, expected):
    assert _bullets(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ('["glutes", "abs"]', ("glutes", "abs")),
        ('Here you go: ["quads"] hope that helps', ("quads",)),
        ("- glutes\n- abs", ("glutes", "abs")),  # falls back to bullets
        ("[not, valid, json]", ("[not, valid, json]",)),
    ],
)
def test_json_array_parsing_tolerates_prose_and_bad_json(text, expected):
    assert _json_string_array(text) == expected


# -- stages ----------------------------------------------------------------


def test_describe_returns_first_non_empty_line(vlm, monkeypatch):
    _stub_post(monkeypatch, _Response(payload={"response": "\n\nA barbell back squat.\n"}))
    assert vlm.describe(["frame"]) == "A barbell back squat."


def test_assess_recognises_the_no_issues_sentinel(vlm, monkeypatch):
    _stub_post(monkeypatch, _Response(payload={"response": "NO ISSUES"}))
    assert vlm.assess(["frame"], "a squat") == ()


def test_assess_returns_bullets(vlm, monkeypatch):
    _stub_post(monkeypatch, _Response(payload={"response": "- knees cave\n- heels lift"}))
    assert vlm.assess(["frame"], "a squat") == ("knees cave", "heels lift")


def test_infer_causes_skips_the_call_when_there_are_no_problems(vlm, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("should not have called the model")

    monkeypatch.setattr("movement_coach.vlm.requests.post", explode)
    assert vlm.infer_causes("a squat", []) == ()


def test_map_to_muscles_skips_the_call_when_there_are_no_causes(vlm, monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("should not have called the model")

    monkeypatch.setattr("movement_coach.vlm.requests.post", explode)
    assert vlm.map_to_muscles([]) == ()


def test_analyse_uses_a_supplied_description_without_calling_describe(vlm, monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(json["prompt"])
        if "technically wrong" in json["prompt"]:
            return _Response(payload={"response": "- shallow depth"})
        return _Response(payload={"response": "- glutes"})

    monkeypatch.setattr("movement_coach.vlm.requests.post", fake_post)
    result = vlm.analyse(["frame"], "a user-corrected side kick")

    assert result.description == "a user-corrected side kick"
    assert not any("Name the movement" in prompt for prompt in calls)


# -- failures --------------------------------------------------------------


def test_connection_error_names_the_stage(vlm, monkeypatch):
    def boom(*_a, **_k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("movement_coach.vlm.requests.post", boom)
    with pytest.raises(VLMError) as exc:
        vlm.describe(["frame"])
    assert exc.value.stage == "describe"
    assert "could not reach" in str(exc.value)


def test_timeout_is_reported_as_a_timeout(vlm, monkeypatch):
    def boom(*_a, **_k):
        raise requests.Timeout()

    monkeypatch.setattr("movement_coach.vlm.requests.post", boom)
    with pytest.raises(VLMError, match="timed out"):
        vlm.describe(["frame"])


def test_http_error_surfaces_the_status(vlm, monkeypatch):
    _stub_post(monkeypatch, _Response(status_code=500, text="boom"))
    with pytest.raises(VLMError, match="HTTP 500"):
        vlm.describe(["frame"])


def test_empty_response_is_an_error_not_an_empty_result(vlm, monkeypatch):
    _stub_post(monkeypatch, _Response(payload={"response": "   "}))
    with pytest.raises(VLMError, match="empty response"):
        vlm.describe(["frame"])


def test_non_json_body(vlm, monkeypatch):
    _stub_post(monkeypatch, _Response(json_error=True))
    with pytest.raises(VLMError, match="non-JSON"):
        vlm.describe(["frame"])


def test_health_rejects_a_missing_model(vlm, monkeypatch):
    monkeypatch.setattr(
        "movement_coach.vlm.requests.get",
        lambda *a, **k: _Response(payload={"models": [{"name": "llama3:8b"}]}),
    )
    with pytest.raises(VLMError, match="is not installed"):
        vlm.health()


def test_health_passes_when_the_model_is_present(vlm, monkeypatch):
    monkeypatch.setattr(
        "movement_coach.vlm.requests.get",
        lambda *a, **k: _Response(payload={"models": [{"name": vlm.model}]}),
    )
    vlm.health()


def test_health_reports_an_unreachable_server(vlm, monkeypatch):
    def boom(*_a, **_k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("movement_coach.vlm.requests.get", boom)
    with pytest.raises(VLMError, match="not reachable"):
        vlm.health()


def test_seed_and_temperature_reach_the_server(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return _Response(payload={"response": "ok"})

    monkeypatch.setattr("movement_coach.vlm.requests.post", fake_post)
    OllamaVLM(seed=7, temperature=0.0).describe(["frame"])

    assert captured["options"] == {"temperature": 0.0, "seed": 7}
    assert captured["images"] == ["frame"]
    assert captured["stream"] is False
