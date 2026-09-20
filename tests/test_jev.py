from __future__ import annotations

import pytest

from dehydrator import JevReranker, ToolIndex
from tests.test_index import TOOLS


def _fake(probabilities, confidence=0.9):
    """Transport that records the request and returns a canned Jev answer."""
    calls: list[dict] = []

    def transport(body):
        calls.append(body)
        return {
            "model": "typesafe-ai/jev",
            "answers": {
                "tool": {
                    "type": "choice",
                    "choice": max(probabilities, key=probabilities.get),
                    "probabilities": probabilities,
                    "confidence": confidence,
                }
            },
            "usage": {"inputTokens": 100, "outputTokens": 5},
        }

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def test_requires_key_without_transport(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        JevReranker()


def test_request_shape():
    t = _fake({"get_weather": 0.9, "send_email": 0.1})
    r = JevReranker(transport=t)
    r.rerank("weather in Tokyo", [TOOLS[0], TOOLS[1]])
    body = t.calls[0]
    assert body["model"] == "typesafe-ai/jev"
    assert body["state"] == {"user_request": "weather in Tokyo"}
    q = body["questions"]["tool"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"get_weather", "send_email"}
    assert q["criteria"]["send_email"] == TOOLS[1]["description"]


def test_reorders_by_probability():
    t = _fake({"get_weather": 0.05, "send_email": 0.9, "list_files": 0.05})
    r = JevReranker(transport=t)
    names = r.rerank("email my boss", [TOOLS[0], TOOLS[1], TOOLS[2]])
    assert names == ["send_email", "get_weather", "list_files"]
    assert r.last_probabilities["send_email"] == 0.9
    assert r.last_confidence == 0.9


def test_min_probability_filters():
    t = _fake({"get_weather": 0.02, "send_email": 0.95, "list_files": 0.03})
    r = JevReranker(min_probability=0.1, transport=t)
    assert r.rerank("email", [TOOLS[0], TOOLS[1], TOOLS[2]]) == ["send_email"]


def test_single_candidate_skips_request():
    t = _fake({"get_weather": 1.0})
    r = JevReranker(transport=t)
    assert r.rerank("weather", [TOOLS[0]]) == ["get_weather"]
    assert t.calls == []


def test_falls_back_to_input_order_on_error():
    def boom(body):
        raise RuntimeError("503")

    r = JevReranker(transport=boom)
    names = r.rerank("anything", [TOOLS[2], TOOLS[0]])
    assert names == ["list_files", "get_weather"]
    assert r.last_probabilities == {}
    assert r.last_confidence is None


def test_tool_index_uses_reranker():
    # "send weather" matches both get_weather and send_email in BM25;
    # the reranker decides the final order.
    t = _fake({"get_weather": 0.1, "send_email": 0.9})
    index = ToolIndex(TOOLS, top_k=1, reranker=JevReranker(transport=t), candidates=10)
    assert index.search("send weather") == ["send_email"]
    sent = set(t.calls[0]["questions"]["tool"]["criteria"])
    # only BM25-positive tools were sent as candidates
    assert sent == {"get_weather", "send_email"}


def test_tool_index_without_reranker_unchanged():
    index = ToolIndex(TOOLS, top_k=2)
    assert index.search("send an email")[0] == "send_email"


def test_candidates_at_least_top_k():
    t = _fake({"get_weather": 0.5, "send_email": 0.5})
    index = ToolIndex(TOOLS, top_k=5, reranker=JevReranker(transport=t), candidates=1)
    assert index._candidates == 5


def test_last_error_set_on_fallback_and_cleared_on_success():
    state = {"fail": True}

    def flaky(body):
        if state["fail"]:
            raise RuntimeError("429")
        return _fake({"get_weather": 0.2, "send_email": 0.8})(body)

    r = JevReranker(transport=flaky)
    r.rerank("x", [TOOLS[0], TOOLS[1]])
    assert isinstance(r.last_error, RuntimeError)
    state["fail"] = False
    assert r.rerank("x", [TOOLS[0], TOOLS[1]]) == ["send_email", "get_weather"]
    assert r.last_error is None


def test_http_retries_on_429(monkeypatch):
    import io
    import urllib.error
    import urllib.request

    import dehydrator._jev as mod

    attempts = {"n": 0}

    def fake_urlopen(req, timeout):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise urllib.error.HTTPError(
                req.full_url, 429, "busy", {}, io.BytesIO(b"{}")
            )
        body = _fake({"get_weather": 0.1, "send_email": 0.9})({})
        return io.BytesIO(__import__("json").dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    r = JevReranker(api_key="k", retries=3)
    assert r.rerank("x", [TOOLS[0], TOOLS[1]]) == ["send_email", "get_weather"]
    assert attempts["n"] == 3
    assert r.last_error is None
