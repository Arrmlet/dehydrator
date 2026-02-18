from __future__ import annotations

import re
from typing import Any

from dehydrator._types import get_tool_description, get_tool_name, get_tool_schema


def tokenize_tool(tool: dict[str, Any]) -> list[str]:
    """Extract searchable tokens from a tool definition.

    Pulls text from the tool name, description, and input schema
    (property names, descriptions, enum values) and returns a flat
    list of lowercase tokens.  Accepts both Anthropic (``input_schema``)
    and MCP (``inputSchema``) key conventions.
    """
    tokens: list[str] = []

    name = get_tool_name(tool)
    tokens.extend(_split_identifier(name))

    description = get_tool_description(tool)
    tokens.extend(_tokenize_text(description))

    schema = get_tool_schema(tool)
    _walk_schema(schema, tokens)

    return tokens


def tokenize_query(query: str) -> list[str]:
    """Tokenize a free-text search query."""
    return _tokenize_text(query)


def _split_identifier(name: str) -> list[str]:
    """Split a snake_case or camelCase identifier into lowercase tokens."""
    # First split on underscores/hyphens
    parts = re.split(r"[_\-]+", name)
    tokens: list[str] = []
    for part in parts:
        # Then split camelCase: insert boundary before uppercase letters
        sub = re.sub(r"([a-z])([A-Z])", r"\1 \2", part)
        for word in sub.split():
            lower = word.lower()
            if lower:
                tokens.append(lower)
    return tokens


def _tokenize_text(text: str) -> list[str]:
    """Lowercase and split text on non-alphanumeric characters."""
    return [w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if w]


def _walk_schema(schema: dict[str, Any], tokens: list[str]) -> None:
    """Recursively extract tokens from a JSON Schema object."""
    properties: dict[str, Any] = schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        tokens.extend(_split_identifier(prop_name))
        if "description" in prop_schema:
            tokens.extend(_tokenize_text(prop_schema["description"]))
        if "enum" in prop_schema:
            for val in prop_schema["enum"]:
                if isinstance(val, str):
                    tokens.extend(_tokenize_text(val))
        # Recurse into nested objects
        if prop_schema.get("type") == "object":
            _walk_schema(prop_schema, tokens)
        # Recurse into array items
        items = prop_schema.get("items")
        if isinstance(items, dict) and items.get("type") == "object":
            _walk_schema(items, tokens)
