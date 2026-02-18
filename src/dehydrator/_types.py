from __future__ import annotations

from typing import Any

ToolParam = dict[str, Any]
"""A tool definition dict (Anthropic or MCP format)."""


def get_tool_name(tool: Any) -> str:
    """Extract tool name from a dict or mcp.types.Tool object."""
    if isinstance(tool, dict):
        return str(tool["name"])
    return str(tool.name)


def get_tool_description(tool: Any) -> str:
    """Extract tool description from a dict or mcp.types.Tool object."""
    if isinstance(tool, dict):
        return str(tool.get("description", ""))
    return str(tool.description or "")


def get_tool_schema(tool: Any) -> dict[str, Any]:
    """Extract input schema from a dict or mcp.types.Tool object.

    Checks ``inputSchema`` (MCP camelCase) then ``input_schema``
    (Anthropic snake_case).
    """
    if isinstance(tool, dict):
        schema: Any = tool.get("inputSchema") or tool.get("input_schema")
        if isinstance(schema, dict):
            return schema
        return {}
    # mcp.types.Tool has .inputSchema
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    # Pydantic model — convert to dict
    return schema.model_dump()  # type: ignore[no-any-return]


def mcp_tool_to_dict(tool: Any) -> ToolParam:
    """Convert an mcp.types.Tool object to an Anthropic-format dict."""
    return {
        "name": get_tool_name(tool),
        "description": get_tool_description(tool),
        "input_schema": get_tool_schema(tool),
    }
