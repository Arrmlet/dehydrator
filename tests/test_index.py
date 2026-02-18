import pytest

from dehydrator._index import ToolIndex


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
        "name": "list_files",
        "description": "List files in a directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"},
            },
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event with a date and time",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Event date"},
            },
        },
    },
]


def test_search_weather():
    index = ToolIndex(TOOLS, top_k=2)
    results = index.search("weather forecast")
    assert "get_weather" in results


def test_search_email():
    index = ToolIndex(TOOLS, top_k=2)
    results = index.search("send an email")
    assert "send_email" in results


def test_search_returns_top_k():
    index = ToolIndex(TOOLS, top_k=2)
    results = index.search("list")
    assert len(results) <= 2


def test_search_excludes_zero_score():
    index = ToolIndex(TOOLS, top_k=10)
    results = index.search("xyznonexistent")
    assert len(results) == 0


def test_get_tools():
    index = ToolIndex(TOOLS)
    tools = index.get_tools(["get_weather", "send_email"])
    names = [t["name"] for t in tools]
    assert names == ["get_weather", "send_email"]


def test_get_tools_skips_unknown():
    index = ToolIndex(TOOLS)
    tools = index.get_tools(["get_weather", "nonexistent"])
    assert len(tools) == 1
    assert tools[0]["name"] == "get_weather"


def test_get_tool():
    index = ToolIndex(TOOLS)
    tool = index.get_tool("send_email")
    assert tool is not None
    assert tool["name"] == "send_email"

    assert index.get_tool("nonexistent") is None


def test_tool_names():
    index = ToolIndex(TOOLS)
    assert set(index.tool_names) == {
        "get_weather",
        "send_email",
        "list_files",
        "create_calendar_event",
    }


def test_empty_tools_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        ToolIndex([])


def test_duplicate_tool_names_raises():
    dupes = [
        {"name": "a", "description": "first", "input_schema": {}},
        {"name": "a", "description": "second", "input_schema": {}},
    ]
    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolIndex(dupes)


def test_empty_query():
    index = ToolIndex(TOOLS)
    assert index.search("") == []


def test_mcp_format_tools():
    """ToolIndex works with MCP camelCase inputSchema tools."""
    mcp_tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
            },
        },
        {
            "name": "send_email",
            "description": "Send an email message to a recipient",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Email address"},
                },
            },
        },
    ]
    index = ToolIndex(mcp_tools, top_k=2)
    results = index.search("weather forecast")
    assert "get_weather" in results


def test_from_mcp():
    """ToolIndex.from_mcp converts MCP Tool objects to dicts."""
    from unittest.mock import MagicMock

    tool1 = MagicMock()
    tool1.name = "get_weather"
    tool1.description = "Get the current weather"
    tool1.inputSchema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
    }

    tool2 = MagicMock()
    tool2.name = "send_email"
    tool2.description = "Send an email message"
    tool2.inputSchema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Email address"},
        },
    }

    index = ToolIndex.from_mcp([tool1, tool2], top_k=2)
    assert set(index.tool_names) == {"get_weather", "send_email"}
    results = index.search("weather")
    assert "get_weather" in results
