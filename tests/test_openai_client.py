"""Tests for OpenAIDehydratedClient with mocked OpenAI-compatible responses."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dehydrator import OpenAIDehydratedClient, AsyncOpenAIDehydratedClient
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
    {
        "name": "help",
        "description": "Show help information",
        "input_schema": {"type": "object", "properties": {}},
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


def _make_response(
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


def test_create_client():
    mock_openai = MagicMock()
    client = OpenAIDehydratedClient(mock_openai, tools=TOOLS, always_available=["help"])
    assert client.inner is mock_openai


def test_reserved_tool_name_raises():
    mock_openai = MagicMock()
    tools_with_conflict = TOOLS + [
        {
            "name": SEARCH_TOOL_NAME,
            "description": "Conflict",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    with pytest.raises(ValueError, match="reserved"):
        OpenAIDehydratedClient(mock_openai, tools=tools_with_conflict)


def test_streaming_raises():
    mock_openai = MagicMock()
    client = OpenAIDehydratedClient(mock_openai, tools=TOOLS)
    with pytest.raises(NotImplementedError, match="Streaming"):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )


def test_end_to_end_search_and_use():
    mock_openai = MagicMock()

    search_tc = _make_tool_call("call_1", SEARCH_TOOL_NAME, {"query": "weather"})
    search_response = _make_response(tool_calls=[search_tc])

    use_tc = _make_tool_call("call_2", "get_weather", {"city": "Tokyo"})
    use_response = _make_response(tool_calls=[use_tc])

    mock_openai.chat.completions.create.side_effect = [search_response, use_response]

    client = OpenAIDehydratedClient(
        mock_openai,
        tools=TOOLS,
        top_k=5,
        always_available=["help"],
        max_search_rounds=3,
    )

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    )

    assert result.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert mock_openai.chat.completions.create.call_count == 2


def test_reset_discoveries():
    mock_openai = MagicMock()
    client = OpenAIDehydratedClient(mock_openai, tools=TOOLS)
    client._discovered.add("get_weather")
    assert "get_weather" in client._discovered
    client.reset_discoveries()
    assert len(client._discovered) == 0


def test_always_available_split():
    mock_openai = MagicMock()
    final_response = _make_response(content="Hi")
    mock_openai.chat.completions.create.return_value = final_response

    client = OpenAIDehydratedClient(
        mock_openai,
        tools=TOOLS,
        always_available=["help"],
    )

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )

    call_tools = mock_openai.chat.completions.create.call_args[1]["tools"]
    tool_names = [t["function"]["name"] for t in call_tools]
    assert "help" in tool_names
    assert SEARCH_TOOL_NAME in tool_names
    assert "get_weather" not in tool_names
    assert "send_email" not in tool_names


def test_user_tools_kwarg_stripped():
    """Even if user passes tools=, it's ignored (we manage tools)."""
    mock_openai = MagicMock()
    final_response = _make_response(content="OK")
    mock_openai.chat.completions.create.return_value = final_response

    client = OpenAIDehydratedClient(mock_openai, tools=TOOLS)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{"type": "function", "function": {"name": "should_be_ignored"}}],
    )

    call_tools = mock_openai.chat.completions.create.call_args[1]["tools"]
    tool_names = [t["function"]["name"] for t in call_tools]
    assert "should_be_ignored" not in tool_names


async def test_async_client():
    mock_openai = MagicMock()
    mock_chat = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat = mock_chat
    mock_chat.completions = mock_completions

    final_response = _make_response(content="Hello!")
    mock_completions.create.return_value = final_response

    client = AsyncOpenAIDehydratedClient(mock_openai, tools=TOOLS)
    result = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert result.choices[0].message.content == "Hello!"


async def test_async_streaming_raises():
    mock_openai = MagicMock()
    mock_chat = MagicMock()
    mock_completions = AsyncMock()
    mock_openai.chat = mock_chat
    mock_chat.completions = mock_completions

    client = AsyncOpenAIDehydratedClient(mock_openai, tools=TOOLS)
    with pytest.raises(NotImplementedError, match="Streaming"):
        await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )


def test_mcp_format_tools_accepted():
    """OpenAIDehydratedClient accepts tools with MCP inputSchema format."""
    mock_openai = MagicMock()
    final_response = _make_response(content="OK")
    mock_openai.chat.completions.create.return_value = final_response

    mcp_tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
            },
        },
        {
            "name": "send_email",
            "description": "Send email",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                },
            },
        },
    ]

    client = OpenAIDehydratedClient(mock_openai, tools=mcp_tools)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
    )

    call_tools = mock_openai.chat.completions.create.call_args[1]["tools"]
    # All tools should be in OpenAI format
    for tool in call_tools:
        assert tool["type"] == "function"
        assert "parameters" in tool["function"]
