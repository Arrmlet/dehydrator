from __future__ import annotations

from typing import Any, cast

import anthropic

from dehydrator._index import ToolIndex
from dehydrator._search_tool import SEARCH_TOOL_DEFINITION, SEARCH_TOOL_NAME
from dehydrator._types import ToolParam

Message = anthropic.types.Message


def _build_tools(
    index: ToolIndex,
    always_available: list[ToolParam],
    discovered: set[str],
) -> list[ToolParam]:
    """Assemble the tool list sent to the API."""
    tools: list[ToolParam] = [SEARCH_TOOL_DEFINITION]
    tools.extend(always_available)
    tools.extend(index.get_tools(sorted(discovered)))
    return tools


def _format_search_result(matched_tools: list[ToolParam]) -> str:
    """Format matched tools into a human-readable result for Claude."""
    if not matched_tools:
        return "No matching tools found. Try a different search query."
    lines = ["Found the following tools:\n"]
    for tool in matched_tools:
        desc = tool.get("description", "No description")
        lines.append(f"- **{tool['name']}**: {desc}")
    lines.append("\nThese tools are now available for you to use.")
    return "\n".join(lines)


def _has_search_call(response: Message) -> bool:
    """Check if the response contains a tool_search call."""
    return any(
        block.type == "tool_use" and block.name == SEARCH_TOOL_NAME
        for block in response.content
    )


def _has_non_search_tool_call(response: Message) -> bool:
    """Check if response contains tool calls other than tool_search."""
    return any(
        block.type == "tool_use" and block.name != SEARCH_TOOL_NAME
        for block in response.content
    )


def _process_search_calls(
    response: Message,
    index: ToolIndex,
    discovered: set[str],
) -> list[dict[str, Any]]:
    """Process tool_search calls, returning tool_result blocks."""
    results: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "tool_use" and block.name == SEARCH_TOOL_NAME:
            query = str(block.input.get("query", ""))
            matched_names = index.search(query)
            discovered.update(matched_names)
            matched_tools = index.get_tools(matched_names)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _format_search_result(matched_tools),
                }
            )
    return results


def _response_content_to_params(
    response: Message,
) -> list[dict[str, Any]]:
    """Convert response content blocks to message param format."""
    blocks: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        elif block.type == "thinking":
            blocks.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                }
            )
        elif block.type == "redacted_thinking":
            blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": block.data,
                }
            )
    return blocks


def send(
    client: anthropic.Anthropic,
    index: ToolIndex,
    always_available: list[ToolParam],
    discovered: set[str],
    max_search_rounds: int,
    **kwargs: Any,
) -> Message:
    """Synchronous send with search interception."""
    response: Message | None = None
    for _ in range(max_search_rounds):
        tools = _build_tools(index, always_available, discovered)
        kwargs["tools"] = tools
        response = cast(Message, client.messages.create(**kwargs))

        if not _has_search_call(response):
            return response

        if _has_non_search_tool_call(response):
            _process_search_calls(response, index, discovered)
            return response

        search_results = _process_search_calls(response, index, discovered)

        messages = list(kwargs.get("messages", []))
        messages.append(
            {
                "role": "assistant",
                "content": _response_content_to_params(response),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": search_results,
            }
        )
        kwargs["messages"] = messages

    # Max rounds exceeded — return last response as-is
    assert response is not None
    return response


async def async_send(
    client: anthropic.AsyncAnthropic,
    index: ToolIndex,
    always_available: list[ToolParam],
    discovered: set[str],
    max_search_rounds: int,
    **kwargs: Any,
) -> Message:
    """Asynchronous send with search interception."""
    response: Message | None = None
    for _ in range(max_search_rounds):
        tools = _build_tools(index, always_available, discovered)
        kwargs["tools"] = tools
        response = cast(Message, await client.messages.create(**kwargs))

        if not _has_search_call(response):
            return response

        if _has_non_search_tool_call(response):
            _process_search_calls(response, index, discovered)
            return response

        search_results = _process_search_calls(response, index, discovered)

        messages = list(kwargs.get("messages", []))
        messages.append(
            {
                "role": "assistant",
                "content": _response_content_to_params(response),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": search_results,
            }
        )
        kwargs["messages"] = messages

    assert response is not None
    return response
