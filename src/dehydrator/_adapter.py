from __future__ import annotations

import json
from typing import Any, Protocol

from dehydrator._index import ToolIndex
from dehydrator._search_tool import SEARCH_TOOL_DEFINITION, SEARCH_TOOL_NAME
from dehydrator._types import (
    ToolParam,
    get_tool_description,
    get_tool_name,
    get_tool_schema,
)


def _format_search_result(matched_tools: list[ToolParam]) -> str:
    """Format matched tools into a human-readable result."""
    if not matched_tools:
        return "No matching tools found. Try a different search query."
    lines = ["Found the following tools:\n"]
    for tool in matched_tools:
        desc = get_tool_description(tool)
        lines.append(f"- **{get_tool_name(tool)}**: {desc}")
    lines.append("\nThese tools are now available for you to use.")
    return "\n".join(lines)


class ProviderAdapter(Protocol):
    """Protocol for provider-specific response handling."""

    def build_tools(
        self,
        index: ToolIndex,
        always_available: list[ToolParam],
        discovered: set[str],
    ) -> list[ToolParam]: ...

    def has_search_call(self, response: Any) -> bool: ...

    def has_non_search_tool_call(self, response: Any) -> bool: ...

    def process_search_calls(
        self,
        response: Any,
        index: ToolIndex,
        discovered: set[str],
    ) -> list[dict[str, Any]]: ...

    def append_search_round(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        search_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def call_api(self, client: Any, **kwargs: Any) -> Any: ...

    async def acall_api(self, client: Any, **kwargs: Any) -> Any: ...


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API."""

    def build_tools(
        self,
        index: ToolIndex,
        always_available: list[ToolParam],
        discovered: set[str],
    ) -> list[ToolParam]:
        tools: list[ToolParam] = [SEARCH_TOOL_DEFINITION]
        tools.extend(always_available)
        tools.extend(index.get_tools(sorted(discovered)))
        return tools

    def has_search_call(self, response: Any) -> bool:
        return any(
            block.type == "tool_use" and block.name == SEARCH_TOOL_NAME
            for block in response.content
        )

    def has_non_search_tool_call(self, response: Any) -> bool:
        return any(
            block.type == "tool_use" and block.name != SEARCH_TOOL_NAME
            for block in response.content
        )

    def process_search_calls(
        self,
        response: Any,
        index: ToolIndex,
        discovered: set[str],
    ) -> list[dict[str, Any]]:
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

    def append_search_round(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        search_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages = list(messages)
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
        return messages

    def call_api(self, client: Any, **kwargs: Any) -> Any:
        return client.messages.create(**kwargs)

    async def acall_api(self, client: Any, **kwargs: Any) -> Any:
        return await client.messages.create(**kwargs)


def _response_content_to_params(
    response: Any,
) -> list[dict[str, Any]]:
    """Convert Anthropic response content blocks to message param format."""
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


class OpenAIAdapter:
    """Adapter for OpenAI-compatible APIs."""

    def _to_openai_tool(self, tool: ToolParam) -> ToolParam:
        """Convert an Anthropic/MCP tool dict to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": get_tool_name(tool),
                "description": get_tool_description(tool),
                "parameters": get_tool_schema(tool),
            },
        }

    def build_tools(
        self,
        index: ToolIndex,
        always_available: list[ToolParam],
        discovered: set[str],
    ) -> list[ToolParam]:
        raw: list[ToolParam] = [SEARCH_TOOL_DEFINITION]
        raw.extend(always_available)
        raw.extend(index.get_tools(sorted(discovered)))
        return [self._to_openai_tool(t) for t in raw]

    def has_search_call(self, response: Any) -> bool:
        tool_calls = _get_openai_tool_calls(response)
        return any(tc.function.name == SEARCH_TOOL_NAME for tc in tool_calls)

    def has_non_search_tool_call(self, response: Any) -> bool:
        tool_calls = _get_openai_tool_calls(response)
        return any(tc.function.name != SEARCH_TOOL_NAME for tc in tool_calls)

    def process_search_calls(
        self,
        response: Any,
        index: ToolIndex,
        discovered: set[str],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for tc in _get_openai_tool_calls(response):
            if tc.function.name == SEARCH_TOOL_NAME:
                raw_args = tc.function.arguments
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                query = str(args.get("query", ""))
                matched_names = index.search(query)
                discovered.update(matched_names)
                matched_tools = index.get_tools(matched_names)
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": _format_search_result(matched_tools),
                    }
                )
        return results

    def append_search_round(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        search_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages = list(messages)
        message = response.choices[0].message
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)
        messages.extend(search_results)
        return messages

    def call_api(self, client: Any, **kwargs: Any) -> Any:
        return client.chat.completions.create(**kwargs)

    async def acall_api(self, client: Any, **kwargs: Any) -> Any:
        return await client.chat.completions.create(**kwargs)


def _get_openai_tool_calls(response: Any) -> list[Any]:
    """Extract tool_calls from an OpenAI-compatible response."""
    message = response.choices[0].message
    return message.tool_calls or []
