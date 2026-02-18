from __future__ import annotations

from typing import Any

from dehydrator._adapter import ProviderAdapter
from dehydrator._index import ToolIndex
from dehydrator._types import ToolParam


def send(
    client: Any,
    adapter: ProviderAdapter,
    index: ToolIndex,
    always_available: list[ToolParam],
    discovered: set[str],
    max_search_rounds: int,
    **kwargs: Any,
) -> Any:
    """Synchronous send with search interception."""
    response: Any = None
    for _ in range(max_search_rounds):
        tools = adapter.build_tools(index, always_available, discovered)
        kwargs["tools"] = tools
        response = adapter.call_api(client, **kwargs)

        if not adapter.has_search_call(response):
            return response

        if adapter.has_non_search_tool_call(response):
            adapter.process_search_calls(response, index, discovered)
            return response

        search_results = adapter.process_search_calls(response, index, discovered)

        messages = list(kwargs.get("messages", []))
        kwargs["messages"] = adapter.append_search_round(
            messages, response, search_results
        )

    assert response is not None
    return response


async def async_send(
    client: Any,
    adapter: ProviderAdapter,
    index: ToolIndex,
    always_available: list[ToolParam],
    discovered: set[str],
    max_search_rounds: int,
    **kwargs: Any,
) -> Any:
    """Asynchronous send with search interception."""
    response: Any = None
    for _ in range(max_search_rounds):
        tools = adapter.build_tools(index, always_available, discovered)
        kwargs["tools"] = tools
        response = await adapter.acall_api(client, **kwargs)

        if not adapter.has_search_call(response):
            return response

        if adapter.has_non_search_tool_call(response):
            adapter.process_search_calls(response, index, discovered)
            return response

        search_results = adapter.process_search_calls(response, index, discovered)

        messages = list(kwargs.get("messages", []))
        kwargs["messages"] = adapter.append_search_round(
            messages, response, search_results
        )

    assert response is not None
    return response
