from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import anthropic.types

from dehydrator._adapter import AnthropicAdapter
from dehydrator._index import ToolIndex
from dehydrator._interceptor import send
from dehydrator._search_tool import SEARCH_TOOL_NAME


TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message to a recipient",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Email address"},
                "body": {"type": "string", "description": "Email body"},
            },
        },
    },
]


def _make_message(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
) -> anthropic.types.Message:
    """Build a Message from raw content block dicts."""
    blocks = []
    for block in content:
        if block["type"] == "text":
            blocks.append(anthropic.types.TextBlock(type="text", text=block["text"]))
        elif block["type"] == "tool_use":
            blocks.append(
                anthropic.types.ToolUseBlock(
                    type="tool_use",
                    id=block["id"],
                    name=block["name"],
                    input=block["input"],
                )
            )
    return anthropic.types.Message(
        id="msg_test",
        content=blocks,
        model="claude-sonnet-4-6",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        type="message",
        usage=anthropic.types.Usage(input_tokens=100, output_tokens=50),
    )


def test_no_search_call_passes_through():
    """When Claude doesn't call tool_search, response passes through."""
    mock_client = MagicMock()
    final_response = _make_message([{"type": "text", "text": "Hello!"}])
    mock_client.messages.create.return_value = final_response

    index = ToolIndex(TOOLS, top_k=5)
    result = send(
        client=mock_client,
        adapter=AnthropicAdapter(),
        index=index,
        always_available=[],
        discovered=set(),
        max_search_rounds=3,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert result is final_response
    assert mock_client.messages.create.call_count == 1


def test_search_then_use():
    """Search is intercepted, tools are discovered, Claude uses them."""
    mock_client = MagicMock()

    # First call: Claude searches for weather tools
    search_response = _make_message(
        [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": SEARCH_TOOL_NAME,
                "input": {"query": "weather"},
            },
        ],
        stop_reason="tool_use",
    )

    # Second call: Claude uses the discovered tool
    use_response = _make_message(
        [
            {
                "type": "tool_use",
                "id": "call_2",
                "name": "get_weather",
                "input": {"city": "London"},
            },
        ],
        stop_reason="tool_use",
    )

    mock_client.messages.create.side_effect = [search_response, use_response]

    index = ToolIndex(TOOLS, top_k=5)
    discovered: set[str] = set()

    result = send(
        client=mock_client,
        adapter=AnthropicAdapter(),
        index=index,
        always_available=[],
        discovered=discovered,
        max_search_rounds=3,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "What's the weather?"}],
    )

    # Should have called API twice
    assert mock_client.messages.create.call_count == 2
    # Discovered tools should include get_weather
    assert "get_weather" in discovered
    # Final response should be the tool use
    assert result.content[0].type == "tool_use"
    assert result.content[0].name == "get_weather"

    # Second call should have included get_weather in tools
    second_call_tools = mock_client.messages.create.call_args_list[1][1]["tools"]
    tool_names = [t["name"] for t in second_call_tools]
    assert "get_weather" in tool_names
    assert SEARCH_TOOL_NAME in tool_names


def test_mixed_calls():
    """If Claude calls tool_search AND another tool, search is processed
    but response is returned for user to handle the other tool."""
    mock_client = MagicMock()

    mixed_response = _make_message(
        [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": SEARCH_TOOL_NAME,
                "input": {"query": "email"},
            },
            {
                "type": "tool_use",
                "id": "call_2",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
        stop_reason="tool_use",
    )

    mock_client.messages.create.return_value = mixed_response

    index = ToolIndex(TOOLS, top_k=5)
    discovered: set[str] = set()

    result = send(
        client=mock_client,
        adapter=AnthropicAdapter(),
        index=index,
        always_available=[],
        discovered=discovered,
        max_search_rounds=3,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Send email and check weather"}],
    )

    # Only one API call — returned immediately
    assert mock_client.messages.create.call_count == 1
    # But search was still processed
    assert "send_email" in discovered
    # Response returned as-is
    assert result is mixed_response


def test_max_rounds_exceeded():
    """After max rounds, the last response is returned."""
    mock_client = MagicMock()

    search_response = _make_message(
        [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": SEARCH_TOOL_NAME,
                "input": {"query": "something"},
            },
        ],
        stop_reason="tool_use",
    )

    # Always returns a search call
    mock_client.messages.create.return_value = search_response

    index = ToolIndex(TOOLS, top_k=5)

    result = send(
        client=mock_client,
        adapter=AnthropicAdapter(),
        index=index,
        always_available=[],
        discovered=set(),
        max_search_rounds=2,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Help"}],
    )

    assert mock_client.messages.create.call_count == 2
    assert result is search_response


def test_always_available_tools_included():
    """Always-available tools are included in every request."""
    mock_client = MagicMock()
    final_response = _make_message([{"type": "text", "text": "OK"}])
    mock_client.messages.create.return_value = final_response

    always_tool = {
        "name": "help",
        "description": "Show help",
        "input_schema": {"type": "object", "properties": {}},
    }

    index = ToolIndex(TOOLS, top_k=5)
    send(
        client=mock_client,
        adapter=AnthropicAdapter(),
        index=index,
        always_available=[always_tool],
        discovered=set(),
        max_search_rounds=3,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )

    call_tools = mock_client.messages.create.call_args[1]["tools"]
    tool_names = [t["name"] for t in call_tools]
    assert "help" in tool_names
    assert SEARCH_TOOL_NAME in tool_names


def test_discovered_tools_persist():
    """Previously discovered tools are included in subsequent calls."""
    mock_client = MagicMock()
    final_response = _make_message([{"type": "text", "text": "Done"}])
    mock_client.messages.create.return_value = final_response

    index = ToolIndex(TOOLS, top_k=5)
    discovered = {"get_weather"}

    send(
        client=mock_client,
        adapter=AnthropicAdapter(),
        index=index,
        always_available=[],
        discovered=discovered,
        max_search_rounds=3,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hi"}],
    )

    call_tools = mock_client.messages.create.call_args[1]["tools"]
    tool_names = [t["name"] for t in call_tools]
    assert "get_weather" in tool_names


def test_no_results_search():
    """When BM25 finds nothing, a 'no results' message is returned to Claude."""
    mock_client = MagicMock()

    search_response = _make_message(
        [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": SEARCH_TOOL_NAME,
                "input": {"query": "xyznonexistent"},
            },
        ],
        stop_reason="tool_use",
    )
    final_response = _make_message([{"type": "text", "text": "Sorry, no tools."}])

    mock_client.messages.create.side_effect = [search_response, final_response]

    index = ToolIndex(TOOLS, top_k=5)

    result = send(
        client=mock_client,
        adapter=AnthropicAdapter(),
        index=index,
        always_available=[],
        discovered=set(),
        max_search_rounds=3,
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Do something weird"}],
    )

    assert result is final_response
    # Verify the tool_result content mentions "no matching tools"
    second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
    tool_result_msg = second_call_messages[-1]
    assert tool_result_msg["role"] == "user"
    result_content = tool_result_msg["content"][0]["content"]
    assert "No matching tools found" in result_content
