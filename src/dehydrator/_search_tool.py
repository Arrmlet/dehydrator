from __future__ import annotations

from dehydrator._types import ToolParam

SEARCH_TOOL_NAME = "tool_search"

_SEARCH_DESCRIPTION = (
    "Search for available tools by describing what you want to do. "
    "Use this before attempting to call a tool you haven't discovered yet. "
    "Returns the names and descriptions of matching tools which will then "
    "become available for you to use."
)

_SCHEMA: ToolParam = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "A natural language description of the action you want to "
                "perform. Be specific — e.g. 'send an email' or "
                "'get weather forecast'."
            ),
        },
    },
    "required": ["query"],
}

SEARCH_TOOL_DEFINITION: ToolParam = {
    "name": SEARCH_TOOL_NAME,
    "description": _SEARCH_DESCRIPTION,
    "input_schema": _SCHEMA,
}


def search_tool_for_openai() -> ToolParam:
    """Return the search tool definition in OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": SEARCH_TOOL_NAME,
            "description": _SEARCH_DESCRIPTION,
            "parameters": _SCHEMA,
        },
    }
