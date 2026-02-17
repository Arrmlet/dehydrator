from __future__ import annotations

from typing import Any

import anthropic

from dehydrator._index import ToolIndex
from dehydrator._interceptor import async_send, send
from dehydrator._search_tool import SEARCH_TOOL_NAME
from dehydrator._types import ToolParam


class _Messages:
    """Namespace that mimics ``client.messages`` for sync usage."""

    def __init__(self, parent: DehydratedClient) -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> anthropic.types.Message:
        if kwargs.get("stream"):
            raise NotImplementedError(
                "Streaming is not yet supported by DehydratedClient. "
                "Pass stream=False or omit it."
            )
        # Strip tools from kwargs — we manage them
        kwargs.pop("tools", None)
        return send(
            client=self._parent._client,
            index=self._parent._index,
            always_available=self._parent._always_available,
            discovered=self._parent._discovered,
            max_search_rounds=self._parent._max_search_rounds,
            **kwargs,
        )


class _AsyncMessages:
    """Namespace that mimics ``client.messages`` for async usage."""

    def __init__(self, parent: AsyncDehydratedClient) -> None:
        self._parent = parent

    async def create(self, **kwargs: Any) -> anthropic.types.Message:
        if kwargs.get("stream"):
            raise NotImplementedError(
                "Streaming is not yet supported by AsyncDehydratedClient. "
                "Pass stream=False or omit it."
            )
        kwargs.pop("tools", None)
        return await async_send(
            client=self._parent._client,
            index=self._parent._index,
            always_available=self._parent._always_available,
            discovered=self._parent._discovered,
            max_search_rounds=self._parent._max_search_rounds,
            **kwargs,
        )


class DehydratedClient:
    """Wraps an ``anthropic.Anthropic`` client with transparent BM25 tool search.

    Instead of sending all tools in every request, only a search tool and
    ``always_available`` tools are sent. When Claude calls the search tool,
    BM25 is run locally and matching tools are added to the next request.

    Usage::

        client = DehydratedClient(
            anthropic.Anthropic(),
            tools=all_my_tools,
            top_k=5,
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Send an email"}],
        )
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
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
        self.messages = _Messages(self)

    @property
    def inner(self) -> anthropic.Anthropic:
        """The underlying Anthropic client."""
        return self._client

    def reset_discoveries(self) -> None:
        """Clear discovered tools. Call this when starting a new conversation."""
        self._discovered.clear()

    @staticmethod
    def _validate_tool_names(tools: list[ToolParam]) -> None:
        for tool in tools:
            if tool["name"] == SEARCH_TOOL_NAME:
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
            if tool["name"] in always_set:
                always.append(tool)
            else:
                searchable.append(tool)
        return searchable, always


class AsyncDehydratedClient:
    """Async version of :class:`DehydratedClient`.

    Wraps an ``anthropic.AsyncAnthropic`` client.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        tools: list[ToolParam],
        *,
        top_k: int = 5,
        always_available: list[str] | None = None,
        max_search_rounds: int = 3,
    ) -> None:
        DehydratedClient._validate_tool_names(tools)
        self._client = client
        all_tools, self._always_available = DehydratedClient._split_tools(
            tools, always_available or []
        )
        if not all_tools:
            raise ValueError("No searchable tools provided.")
        self._index = ToolIndex(all_tools, top_k=top_k)
        self._discovered: set[str] = set()
        self._max_search_rounds = max_search_rounds
        self.messages = _AsyncMessages(self)

    @property
    def inner(self) -> anthropic.AsyncAnthropic:
        """The underlying async Anthropic client."""
        return self._client

    def reset_discoveries(self) -> None:
        """Clear discovered tools. Call this when starting a new conversation."""
        self._discovered.clear()
