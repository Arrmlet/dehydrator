"""Tests for MCP Tool object ingestion."""

from __future__ import annotations

from unittest.mock import MagicMock

from dehydrator._index import ToolIndex
from dehydrator._types import (
    get_tool_description,
    get_tool_name,
    get_tool_schema,
    mcp_tool_to_dict,
)


def _make_mcp_tool(name: str, description: str, schema: dict) -> MagicMock:
    """Create a mock mcp.types.Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema
    return tool


def test_get_tool_name_from_mcp():
    tool = _make_mcp_tool("my_tool", "A tool", {})
    assert get_tool_name(tool) == "my_tool"


def test_get_tool_description_from_mcp():
    tool = _make_mcp_tool("t", "A description", {})
    assert get_tool_description(tool) == "A description"


def test_get_tool_schema_from_mcp():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    tool = _make_mcp_tool("t", "d", schema)
    assert get_tool_schema(tool) == schema


def test_mcp_tool_to_dict():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    tool = _make_mcp_tool("get_weather", "Get weather", schema)
    d = mcp_tool_to_dict(tool)
    assert d["name"] == "get_weather"
    assert d["description"] == "Get weather"
    assert d["input_schema"] == schema


def test_get_helpers_with_anthropic_dict():
    tool = {
        "name": "send_email",
        "description": "Send email",
        "input_schema": {"type": "object", "properties": {}},
    }
    assert get_tool_name(tool) == "send_email"
    assert get_tool_description(tool) == "Send email"
    assert get_tool_schema(tool) == {"type": "object", "properties": {}}


def test_get_helpers_with_mcp_dict():
    tool = {
        "name": "send_email",
        "description": "Send email",
        "inputSchema": {"type": "object", "properties": {}},
    }
    assert get_tool_name(tool) == "send_email"
    assert get_tool_description(tool) == "Send email"
    assert get_tool_schema(tool) == {"type": "object", "properties": {}}


def test_tool_index_from_mcp_objects():
    tools = [
        _make_mcp_tool(
            "get_weather",
            "Get weather for a city",
            {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
            },
        ),
        _make_mcp_tool(
            "send_email",
            "Send an email message",
            {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient"},
                },
            },
        ),
    ]
    index = ToolIndex.from_mcp(tools, top_k=2)
    assert set(index.tool_names) == {"get_weather", "send_email"}
    results = index.search("weather")
    assert "get_weather" in results


def test_get_schema_prefers_camel_case():
    """When both keys exist, inputSchema wins."""
    tool = {
        "name": "t",
        "description": "",
        "inputSchema": {"from": "mcp"},
        "input_schema": {"from": "anthropic"},
    }
    assert get_tool_schema(tool) == {"from": "mcp"}


def test_get_schema_missing_returns_empty():
    tool = {"name": "t", "description": ""}
    assert get_tool_schema(tool) == {}
