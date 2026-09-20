from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from dehydrator._types import ToolParam, get_tool_description, get_tool_name

JEV_MODEL = "typesafe-ai/jev"
GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
_MAX_CHOICE_OPTIONS = 255


class Reranker(Protocol):
    """Reorders a BM25 shortlist for a query. Return tool names, best first."""

    def rerank(self, query: str, tools: list[ToolParam]) -> list[str]: ...


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class JevReranker:
    """Re-rank BM25 candidates with TypeSafe AI's Jev via Vercel AI Gateway.

    Jev is a decision model, not an LLM: given the user's query and the
    candidate tools as a ``choice`` question, it returns a calibrated
    probability for every candidate in one ~400 ms request. On Dehydrator's
    30-query MCP benchmark this lifts top-1 accuracy from 93% to 100%.

    The BM25 order is kept as a fallback whenever the request fails, so
    enabling the reranker can never make search worse than plain BM25.

    Args:
        api_key: Vercel AI Gateway key. Defaults to ``AI_GATEWAY_API_KEY``.
        model: Evaluation model id (default ``typesafe-ai/jev``).
        min_probability: Drop candidates Jev scores below this (0 keeps all).
        timeout: Request timeout in seconds.
        retries: Extra attempts on 429/5xx with exponential backoff. The
            gateway throttles above a few concurrent requests.
        transport: Optional callable ``(request_body) -> response_json`` used
            instead of HTTP; for tests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = JEV_MODEL,
        min_probability: float = 0.0,
        timeout: float = 30.0,
        retries: int = 3,
        url: str = GATEWAY_URL,
        transport: Transport | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY")
        if not self._api_key and transport is None:
            raise ValueError(
                "JevReranker needs an API key: pass api_key= or set "
                "AI_GATEWAY_API_KEY."
            )
        self._model = model
        self._min_probability = min_probability
        self._timeout = timeout
        self._retries = retries
        self._url = url
        self._transport = transport or self._http_post
        self.last_probabilities: dict[str, float] = {}
        self.last_confidence: float | None = None
        self.last_error: Exception | None = None
        """Set when the last rerank fell back to BM25 order; None on success."""

    def rerank(self, query: str, tools: list[ToolParam]) -> list[str]:
        names = [get_tool_name(t) for t in tools]
        if len(names) < 2:
            return names
        if len(names) > _MAX_CHOICE_OPTIONS:
            tools = tools[:_MAX_CHOICE_OPTIONS]
            names = names[:_MAX_CHOICE_OPTIONS]
        criteria = {
            get_tool_name(t): get_tool_description(t) or get_tool_name(t)
            for t in tools
        }
        body = {
            "model": self._model,
            "state": {"user_request": query},
            "questions": {
                "tool": {
                    "type": "choice",
                    "instructions": (
                        "Which tool should be called to accomplish the user's request?"
                    ),
                    "criteria": criteria,
                }
            },
        }
        try:
            data = self._transport(body)
            answer = data["answers"]["tool"]
            probs: dict[str, float] = {
                str(k): float(v) for k, v in answer["probabilities"].items()
            }
        except Exception as exc:
            # Network/auth/shape problem: keep BM25 order.
            self.last_probabilities = {}
            self.last_confidence = None
            self.last_error = exc
            return names
        self.last_error = None
        self.last_probabilities = probs
        self.last_confidence = _extract_confidence(data, answer)
        order = sorted(names, key=lambda n: (-probs.get(n, 0.0), names.index(n)))
        return [n for n in order if probs.get(n, 0.0) >= self._min_probability]

    def _http_post(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        delay = 0.5
        for attempt in range(self._retries + 1):
            req = urllib.request.Request(
                self._url, data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    result: dict[str, Any] = json.loads(resp.read())
                    return result
            except urllib.error.HTTPError as e:
                retryable = e.code == 429 or e.code >= 500
                if retryable and attempt < self._retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                detail = e.read()[:200]
                raise RuntimeError(
                    f"Jev request failed: {e.code} {detail!r}"
                ) from e
        raise RuntimeError("unreachable")


def _extract_confidence(
    data: dict[str, Any], answer: dict[str, Any]
) -> float | None:
    if "confidence" in answer:
        return float(answer["confidence"])
    meta = data.get("providerMetadata", {}).get("typesafe", {}).get("confidence", {})
    value = meta.get("tool")
    return float(value) if value is not None else None
