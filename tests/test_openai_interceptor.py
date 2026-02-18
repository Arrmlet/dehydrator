"""Tests for the interceptor with OpenAI adapter."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from dehydrator._adapter import OpenAIAdapter
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


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_openai_response(
    content: str | None = None,
    tool_calls: list[MagicMock] | None = None,
) -> MagicMock:
    response = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = message
    response.choices = [choice]
    return response


def test_no_search_call_passes_through():
    mock_client = MagicMock()
    final_response = _make_openai_response(content="Hello!")
    mock_client.chat.completions.create.return_value = final_response

    index = ToolIndex(TOOLS, top_k=5)
    result = send(
        client=mock_client,
        adapter=OpenAIAdapter(),
        index=index,
        always_available=[],
        discovered=set(),
        max_search_rounds=3,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert result is final_response
    assert mock_client.chat.completions.create.call_count == 1


def test_tools_in_openai_format():
    """Tools sent to API should be in OpenAI function format."""
    mock_client = MagicMock()
    final_response = _make_openai_response(content="OK")
    mock_client.chat.completions.create.return_value = final_response

    index = ToolIndex(TOOLS, top_k=5)
    send(
        client=mock_client,
        adapter=OpenAIAdapter(),
        index=index,
        always_available=[],
        discovered=set(),
        max_search_rounds=3,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )

    call_tools = mock_client.chat.completions.create.call_args[1]["tools"]
    # All tools should be in OpenAI function format
    for tool in call_tools:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

    # Search tool should be present
    names = [t["function"]["name"] for t in call_tools]
    assert SEARCH_TOOL_NAME in names


def test_search_then_use():
    mock_client = MagicMock()

    search_tc = _make_tool_call("call_1", SEARCH_TOOL_NAME, {"query": "weather"})
    search_response = _make_openai_response(tool_calls=[search_tc])

    use_tc = _make_tool_call("call_2", "get_weather", {"city": "London"})
    use_response = _make_openai_response(tool_calls=[use_tc])

    mock_client.chat.completions.create.side_effect = [search_response, use_response]

    index = ToolIndex(TOOLS, top_k=5)
    discovered: set[str] = set()

    result = send(
        client=mock_client,
        adapter=OpenAIAdapter(),
        index=index,
        always_available=[],
        discovered=discovered,
        max_search_rounds=3,
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather?"}],
    )

    assert mock_client.chat.completions.create.call_count == 2
    assert "get_weather" in discovered
    assert result is use_response

    # Second call should include discovered tools
    second_call_tools = mock_client.chat.completions.create.call_args_list[1][1][
        "tools"
    ]
    names = [t["function"]["name"] for t in second_call_tools]
    assert "get_weather" in names


def test_mixed_calls():
    mock_client = MagicMock()

    search_tc = _make_tool_call("call_1", SEARCH_TOOL_NAME, {"query": "email"})
    weather_tc = _make_tool_call("call_2", "get_weather", {"city": "Paris"})
    mixed_response = _make_openai_response(tool_calls=[search_tc, weather_tc])

    mock_client.chat.completions.create.return_value = mixed_response

    index = ToolIndex(TOOLS, top_k=5)
    discovered: set[str] = set()

    result = send(
        client=mock_client,
        adapter=OpenAIAdapter(),
        index=index,
        always_available=[],
        discovered=discovered,
        max_search_rounds=3,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Send email and check weather"}],
    )

    assert mock_client.chat.completions.create.call_count == 1
    assert "send_email" in discovered
    assert result is mixed_response


def test_max_rounds_exceeded():
    mock_client = MagicMock()

    search_tc = _make_tool_call("call_1", SEARCH_TOOL_NAME, {"query": "something"})
    search_response = _make_openai_response(tool_calls=[search_tc])
    mock_client.chat.completions.create.return_value = search_response

    index = ToolIndex(TOOLS, top_k=5)

    result = send(
        client=mock_client,
        adapter=OpenAIAdapter(),
        index=index,
        always_available=[],
        discovered=set(),
        max_search_rounds=2,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Help"}],
    )

    assert mock_client.chat.completions.create.call_count == 2
    assert result is search_response


def test_search_round_messages():
    """After a search round, messages include assistant tool_calls and tool results."""
    mock_client = MagicMock()

    search_tc = _make_tool_call("call_1", SEARCH_TOOL_NAME, {"query": "weather"})
    search_response = _make_openai_response(tool_calls=[search_tc])

    final_response = _make_openai_response(content="Here's the weather.")
    mock_client.chat.completions.create.side_effect = [search_response, final_response]

    index = ToolIndex(TOOLS, top_k=5)

    send(
        client=mock_client,
        adapter=OpenAIAdapter(),
        index=index,
        always_available=[],
        discovered=set(),
        max_search_rounds=3,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Weather?"}],
    )

    # Check the second call's messages
    second_call_messages = mock_client.chat.completions.create.call_args_list[1][1][
        "messages"
    ]
    # Should have: user, assistant (with tool_calls), tool result
    assert second_call_messages[-2]["role"] == "assistant"
    assert "tool_calls" in second_call_messages[-2]
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "call_1"
