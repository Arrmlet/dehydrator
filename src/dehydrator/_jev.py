from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Literal, Protocol

from dehydrator._types import (
    ToolParam,
    get_tool_description,
    get_tool_name,
    mcp_tool_to_dict,
)

Provider = Literal["typesafe", "gateway"]

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"
GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
GATEWAY_MODEL = "typesafe-ai/jev"
JEV_MODEL = GATEWAY_MODEL  # backwards compatibility
_MAX_CHOICE_OPTIONS = 255

_PROVIDERS: dict[str, tuple[str, str, str]] = {
    # provider: (env var, url, default model)
    "typesafe": ("TYPESAFE_API_KEY", TYPESAFE_URL, TYPESAFE_MODEL),
    "gateway": ("AI_GATEWAY_API_KEY", GATEWAY_URL, GATEWAY_MODEL),
}


def detect_provider() -> Provider | None:
    """TypeSafe's own API if TYPESAFE_API_KEY is set, else Vercel AI Gateway."""
    if os.environ.get("TYPESAFE_API_KEY"):
        return "typesafe"
    if os.environ.get("AI_GATEWAY_API_KEY"):
        return "gateway"
    return None


class Reranker(Protocol):
    """Reorders a BM25 shortlist for a query. Return tool names, best first."""

    def rerank(self, query: str, tools: list[ToolParam]) -> list[str]: ...


Transport = Callable[[dict[str, Any]], dict[str, Any]]


class JevReranker:
    """Re-rank BM25 candidates with TypeSafe AI's Jev.

    Jev is a decision model, not an LLM: given the user's query and the
    candidate tools as a ``choice`` question, it returns a calibrated
    probability for every candidate in one ~400 ms request. On Dehydrator's
    30-query MCP benchmark this lifts top-1 accuracy from 93% to 100%.

    Two providers speak the same request shape:

    * ``"typesafe"``: TypeSafe's own API, ``TYPESAFE_API_KEY``,
      ``https://api.typesafe.ai/v1/systemone``, model ``jev-latest``.
    * ``"gateway"``: Vercel AI Gateway, ``AI_GATEWAY_API_KEY``,
      ``https://ai-gateway.vercel.sh/v1/evaluate``, model ``typesafe-ai/jev``.

    With no ``provider`` given, TypeSafe is used when ``TYPESAFE_API_KEY``
    is set, otherwise the gateway.

    The BM25 order is kept as a fallback whenever the request fails, so
    enabling the reranker can never make search worse than plain BM25.

    Args:
        api_key: API key for the provider. Defaults to the provider's env var.
        provider: ``"typesafe"`` or ``"gateway"``; auto-detected from env.
        model: Model id; defaults to the provider's default.
        min_probability: Drop candidates Jev scores below this (0 keeps all).
        timeout: Request timeout in seconds.
        retries: Extra attempts on 429/529/5xx with exponential backoff.
        url: Override the endpoint (e.g. a proxy).
        transport: Optional callable ``(request_body) -> response_json`` used
            instead of HTTP; for tests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        min_probability: float = 0.0,
        timeout: float = 30.0,
        retries: int = 3,
        url: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        if provider is None:
            provider = detect_provider() or "typesafe"
        if provider not in _PROVIDERS:
            raise ValueError(
                f"provider must be 'typesafe' or 'gateway', got {provider!r}"
            )
        env_var, default_url, default_model = _PROVIDERS[provider]
        self.provider: Provider = provider
        self._api_key = api_key or os.environ.get(env_var)
        if not self._api_key and transport is None:
            raise ValueError(
                f"JevReranker needs an API key: pass api_key= or set {env_var}."
            )
        self._model = model or default_model
        self._min_probability = min_probability
        self._timeout = timeout
        self._retries = retries
        self._url = url or default_url
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
            get_tool_name(t): get_tool_description(t) or get_tool_name(t) for t in tools
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
                retryable = e.code in (408, 429) or e.code >= 500
                if retryable and attempt < self._retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                detail = e.read()[:200]
                raise RuntimeError(f"Jev request failed: {e.code} {detail!r}") from e
        raise RuntimeError("unreachable")


def _extract_confidence(data: dict[str, Any], answer: dict[str, Any]) -> float | None:
    if "confidence" in answer:
        return float(answer["confidence"])
    meta = data.get("providerMetadata", {}).get("typesafe", {}).get("confidence", {})
    value = meta.get("tool")
    return float(value) if value is not None else None


class JevIndex:
    """Tool search with Jev only, no BM25.

    Every search asks Jev one ``choice`` question over the tools. Jev accepts
    at most 255 options per question, so larger corpora run a tournament:
    tools are split into batches of ``batch_size``, each batch gets a Jev
    round, the top ``finalists`` of every batch meet in a final round.
    A 1,000-tool corpus is 5 batch rounds plus 1 final, about 2.5 s
    sequentially and roughly 3 cents per 1,000 searches.

    Drop-in for :class:`ToolIndex` in every client via ``search="jev"``.
    """

    def __init__(
        self,
        tools: list[ToolParam],
        *,
        top_k: int = 5,
        reranker: JevReranker | None = None,
        batch_size: int = 200,
        finalists: int = 10,
    ) -> None:
        if not tools:
            raise ValueError("tools must not be empty")
        if not 2 <= batch_size <= _MAX_CHOICE_OPTIONS:
            raise ValueError(f"batch_size must be 2..{_MAX_CHOICE_OPTIONS}")
        self._tools_by_name: dict[str, ToolParam] = {}
        for tool in tools:
            name = get_tool_name(tool)
            if name in self._tools_by_name:
                raise ValueError(f"Duplicate tool name: {name!r}")
            self._tools_by_name[name] = tool
        self._tools = list(tools)
        self._top_k = top_k
        self._reranker = reranker or JevReranker()
        self._batch_size = batch_size
        self._finalists = max(finalists, top_k)
        self.last_probabilities: dict[str, float] = {}
        self.last_confidence: float | None = None

    @classmethod
    def from_mcp(cls, tools: list[Any], **kwargs: Any) -> JevIndex:
        """Create a JevIndex from ``mcp.types.Tool`` objects."""
        return cls([mcp_tool_to_dict(t) for t in tools], **kwargs)

    @property
    def tool_names(self) -> list[str]:
        return [get_tool_name(t) for t in self._tools]

    @property
    def reranker(self) -> JevReranker:
        return self._reranker

    def search(self, query: str) -> list[str]:
        """Return up to *top_k* tool names, best first, as ranked by Jev."""
        if not query.strip():
            return []
        candidates = self._tools
        if len(candidates) > self._batch_size:
            finalists: list[ToolParam] = []
            for i in range(0, len(candidates), self._batch_size):
                batch = candidates[i : i + self._batch_size]
                names = self._reranker.rerank(query, batch)[: self._finalists]
                finalists.extend(self.get_tools(names))
            candidates = finalists
        ranked = self._reranker.rerank(query, candidates)
        self.last_probabilities = dict(self._reranker.last_probabilities)
        self.last_confidence = self._reranker.last_confidence
        return ranked[: self._top_k]

    def get_tools(self, names: list[str]) -> list[ToolParam]:
        return [self._tools_by_name[n] for n in names if n in self._tools_by_name]

    def get_tool(self, name: str) -> ToolParam | None:
        return self._tools_by_name.get(name)
