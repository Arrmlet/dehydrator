from __future__ import annotations

from typing import Any

from dehydrator._adapter import OpenAIAdapter
from dehydrator._index import ToolIndex
from dehydrator._interceptor import async_send, send
from dehydrator._search_tool import SEARCH_TOOL_NAME
from dehydrator._types import ToolParam, get_tool_name


class _ChatCompletions:
    """Namespace that mimics ``client.chat.completions`` for sync usage."""

    def __init__(self, parent: OpenAIDehydratedClient) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise NotImplementedError(
                "Streaming is not yet supported by OpenAIDehydratedClient. "
                "Pass stream=False or omit it."
            )
        kwargs.pop("tools", None)
        return send(
            client=self._parent._client,
            adapter=OpenAIAdapter(),
            index=self._parent._index,
            always_available=self._parent._always_available,
            discovered=self._parent._discovered,
            max_search_rounds=self._parent._max_search_rounds,
            **kwargs,
        )


class _AsyncChatCompletions:
    """Namespace that mimics ``client.chat.completions`` for async usage."""

    def __init__(self, parent: AsyncOpenAIDehydratedClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise NotImplementedError(
                "Streaming is not yet supported by AsyncOpenAIDehydratedClient. "
                "Pass stream=False or omit it."
            )
        kwargs.pop("tools", None)
        return await async_send(
            client=self._parent._client,
            adapter=OpenAIAdapter(),
            index=self._parent._index,
            always_available=self._parent._always_available,
            discovered=self._parent._discovered,
            max_search_rounds=self._parent._max_search_rounds,
            **kwargs,
        )


class _Chat:
    """Namespace that mimics ``client.chat``."""

    def __init__(self, completions: _ChatCompletions | _AsyncChatCompletions) -> None:
        self.completions = completions


class OpenAIDehydratedClient:
    """Wraps any OpenAI-compatible client with transparent BM25 tool search.

    Works with ``openai.OpenAI``, Groq, OpenRouter, Chutes, and any other
    client that implements ``client.chat.completions.create()``.

    Usage::

        from openai import OpenAI
        client = OpenAIDehydratedClient(
            OpenAI(),
            tools=all_my_tools,
            top_k=5,
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Send an email"}],
        )
    """

    def __init__(
        self,
        client: Any,
        tools: list[ToolParam],
        *,
        top_k: int = 5,
        always_available: list[str] | None = None,
        max_search_rounds: int = 3,
    ) -> None:
        self._validate_tool_names(tools)
        self._client = client
        all_tools, self._always_available = self._split_tools(
            tools, always_available or []
        )
        if not all_tools:
            raise ValueError("No searchable tools provided.")
        self._index = ToolIndex(all_tools, top_k=top_k)
        self._discovered: set[str] = set()
        self._max_search_rounds = max_search_rounds
        self.chat = _Chat(_ChatCompletions(self))

    @property
    def inner(self) -> Any:
        """The underlying OpenAI-compatible client."""
        return self._client

    def reset_discoveries(self) -> None:
        """Clear discovered tools. Call this when starting a new conversation."""
        self._discovered.clear()

    @staticmethod
    def _validate_tool_names(tools: list[ToolParam]) -> None:
        for tool in tools:
            if get_tool_name(tool) == SEARCH_TOOL_NAME:
                raise ValueError(
                    f"Tool name {SEARCH_TOOL_NAME!r} is reserved by Dehydrator. "
                    "Please rename your tool."
                )

    @staticmethod
    def _split_tools(
        tools: list[ToolParam], always_names: list[str]
    ) -> tuple[list[ToolParam], list[ToolParam]]:
        always_set = set(always_names)
        always: list[ToolParam] = []
        searchable: list[ToolParam] = []
        for tool in tools:
            if get_tool_name(tool) in always_set:
                always.append(tool)
            else:
                searchable.append(tool)
        return searchable, always


class AsyncOpenAIDehydratedClient:
    """Async version of :class:`OpenAIDehydratedClient`.

    Wraps any async OpenAI-compatible client.
    """

    def __init__(
        self,
        client: Any,
        tools: list[ToolParam],
        *,
        top_k: int = 5,
        always_available: list[str] | None = None,
        max_search_rounds: int = 3,
    ) -> None:
        OpenAIDehydratedClient._validate_tool_names(tools)
        self._client = client
        all_tools, self._always_available = OpenAIDehydratedClient._split_tools(
            tools, always_available or []
        )
        if not all_tools:
            raise ValueError("No searchable tools provided.")
        self._index = ToolIndex(all_tools, top_k=top_k)
        self._discovered: set[str] = set()
        self._max_search_rounds = max_search_rounds
        self.chat = _Chat(_AsyncChatCompletions(self))

    @property
    def inner(self) -> Any:
        """The underlying async OpenAI-compatible client."""
        return self._client

    def reset_discoveries(self) -> None:
        """Clear discovered tools. Call this when starting a new conversation."""
        self._discovered.clear()
