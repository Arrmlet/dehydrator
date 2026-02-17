from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anthropic.types
import pytest

from dehydrator import AsyncDehydratedClient, DehydratedClient
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


def _make_message(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
) -> anthropic.types.Message:
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


def test_create_client():
    mock_anthropic = MagicMock()
    client = DehydratedClient(mock_anthropic, tools=TOOLS, always_available=["help"])
    assert client.inner is mock_anthropic


def test_reserved_tool_name_raises():
    mock_anthropic = MagicMock()
    tools_with_conflict = TOOLS + [
        {
            "name": SEARCH_TOOL_NAME,
            "description": "Conflict",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    with pytest.raises(ValueError, match="reserved"):
        DehydratedClient(mock_anthropic, tools=tools_with_conflict)


def test_streaming_raises():
    mock_anthropic = MagicMock()
    client = DehydratedClient(mock_anthropic, tools=TOOLS)
    with pytest.raises(NotImplementedError, match="Streaming"):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )


def test_end_to_end_search_and_use():
    mock_anthropic = MagicMock()

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
    use_response = _make_message(
        [
            {
                "type": "tool_use",
                "id": "call_2",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
            },
        ],
        stop_reason="tool_use",
    )

    mock_anthropic.messages.create.side_effect = [search_response, use_response]

    client = DehydratedClient(
        mock_anthropic,
        tools=TOOLS,
        top_k=5,
        always_available=["help"],
        max_search_rounds=3,
    )

    result = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    )

    assert result.content[0].name == "get_weather"
    assert mock_anthropic.messages.create.call_count == 2


def test_reset_discoveries():
    mock_anthropic = MagicMock()
    client = DehydratedClient(mock_anthropic, tools=TOOLS)
    client._discovered.add("get_weather")
    assert "get_weather" in client._discovered
    client.reset_discoveries()
    assert len(client._discovered) == 0


def test_always_available_split():
    mock_anthropic = MagicMock()
    final_response = _make_message([{"type": "text", "text": "Hi"}])
    mock_anthropic.messages.create.return_value = final_response

    client = DehydratedClient(
        mock_anthropic,
        tools=TOOLS,
        always_available=["help"],
    )

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )

    call_tools = mock_anthropic.messages.create.call_args[1]["tools"]
    tool_names = [t["name"] for t in call_tools]
    # help should be always available, not in the index search results
    assert "help" in tool_names
    assert SEARCH_TOOL_NAME in tool_names
    # Other tools should NOT be present (not yet discovered)
    assert "get_weather" not in tool_names
    assert "send_email" not in tool_names


def test_user_tools_kwarg_stripped():
    """Even if user passes tools=, it's ignored (we manage tools)."""
    mock_anthropic = MagicMock()
    final_response = _make_message([{"type": "text", "text": "OK"}])
    mock_anthropic.messages.create.return_value = final_response

    client = DehydratedClient(mock_anthropic, tools=TOOLS)
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hi"}],
        tools=[{"name": "should_be_ignored"}],  # type: ignore[list-item]
    )

    call_tools = mock_anthropic.messages.create.call_args[1]["tools"]
    tool_names = [t["name"] for t in call_tools]
    assert "should_be_ignored" not in tool_names


async def test_async_client():
    mock_anthropic = MagicMock()
    mock_messages = AsyncMock()
    mock_anthropic.messages = mock_messages

    final_response = _make_message([{"type": "text", "text": "Hello!"}])
    mock_messages.create.return_value = final_response

    client = AsyncDehydratedClient(mock_anthropic, tools=TOOLS)
    result = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hi"}],
    )

    assert result.content[0].text == "Hello!"


async def test_async_streaming_raises():
    mock_anthropic = MagicMock()
    mock_anthropic.messages = AsyncMock()

    client = AsyncDehydratedClient(mock_anthropic, tools=TOOLS)
    with pytest.raises(NotImplementedError, match="Streaming"):
        await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )
