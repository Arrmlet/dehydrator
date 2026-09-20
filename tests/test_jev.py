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


# ---- JevIndex (pure Jev, no BM25) ------------------------------------------


def _scripted(answers_by_call):
    """Transport returning successive canned probability dicts."""
    calls: list[dict] = []
    it = iter(answers_by_call)

    def transport(body):
        calls.append(body)
        probs = next(it)
        return {
            "answers": {
                "tool": {
                    "type": "choice",
                    "choice": max(probs, key=probs.get),
                    "probabilities": probs,
                    "confidence": 0.8,
                }
            }
        }

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def test_jev_index_single_round():
    from dehydrator import JevIndex

    t = _scripted(
        [
            {
                "get_weather": 0.05,
                "send_email": 0.9,
                "list_files": 0.03,
                "create_calendar_event": 0.02,
            }
        ]
    )
    idx = JevIndex(TOOLS, top_k=2, reranker=JevReranker(transport=t))
    assert idx.search("email my boss") == ["send_email", "get_weather"]
    assert len(t.calls) == 1
    assert set(t.calls[0]["questions"]["tool"]["criteria"]) == {
        tt["name"] for tt in TOOLS
    }
    assert idx.last_probabilities["send_email"] == 0.9
    assert idx.last_confidence == 0.8


def test_jev_index_tournament_over_batch_size():
    from dehydrator import JevIndex

    # 4 tools, batch_size=2 -> two batch rounds, then a final over finalists.
    t = _scripted(
        [
            {"get_weather": 0.3, "send_email": 0.7},  # batch 1
            {"list_files": 0.9, "create_calendar_event": 0.1},  # batch 2
            {
                "send_email": 0.2,
                "get_weather": 0.05,
                "list_files": 0.7,
                "create_calendar_event": 0.05,
            },  # final
        ]
    )
    idx = JevIndex(
        TOOLS, top_k=1, reranker=JevReranker(transport=t), batch_size=2, finalists=2
    )
    assert idx.search("show me the files") == ["list_files"]
    assert len(t.calls) == 3
    final = set(t.calls[2]["questions"]["tool"]["criteria"])
    assert final == {"get_weather", "send_email", "list_files", "create_calendar_event"}


def test_jev_index_empty_query_and_validation():
    from dehydrator import JevIndex

    idx = JevIndex(TOOLS, reranker=JevReranker(transport=_scripted([])))
    assert idx.search("   ") == []
    with pytest.raises(ValueError, match="must not be empty"):
        JevIndex([], reranker=JevReranker(transport=_scripted([])))
    with pytest.raises(ValueError, match="batch_size"):
        JevIndex(TOOLS, reranker=JevReranker(transport=_scripted([])), batch_size=300)


def test_client_search_jev_builds_jev_index():
    from unittest.mock import MagicMock

    from dehydrator import DehydratedClient, JevIndex

    rr = JevReranker(transport=_scripted([]))
    client = DehydratedClient(MagicMock(), tools=TOOLS, search="jev", reranker=rr)
    assert isinstance(client._index, JevIndex)
    with pytest.raises(ValueError, match="search must be"):
        DehydratedClient(MagicMock(), tools=TOOLS, search="nope")  # type: ignore[arg-type]
