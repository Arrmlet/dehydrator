from __future__ import annotations

from typing import Any

from rank_bm25 import BM25L

from dehydrator._tokenizer import tokenize_query, tokenize_tool
from dehydrator._types import ToolParam, get_tool_name, mcp_tool_to_dict


class ToolIndex:
    """BM25 search index over tool definitions."""

    def __init__(self, tools: list[ToolParam], *, top_k: int = 5) -> None:
        if not tools:
            raise ValueError("tools must not be empty")
        self._tools_by_name: dict[str, ToolParam] = {}
        corpus: list[list[str]] = []
        names: list[str] = []
        for tool in tools:
            name = get_tool_name(tool)
            if name in self._tools_by_name:
                raise ValueError(f"Duplicate tool name: {name!r}")
            self._tools_by_name[name] = tool
            names.append(name)
            corpus.append(tokenize_tool(tool))
        self._names = names
        self._bm25 = BM25L(corpus)
        self._top_k = top_k

    @classmethod
    def from_mcp(cls, tools: list[Any], *, top_k: int = 5) -> ToolIndex:
        """Create a ToolIndex from a list of ``mcp.types.Tool`` objects."""
        return cls([mcp_tool_to_dict(t) for t in tools], top_k=top_k)

    @property
    def tool_names(self) -> list[str]:
        """All indexed tool names."""
        return list(self._names)

    def search(self, query: str) -> list[str]:
        """Return up to *top_k* tool names ranked by BM25 relevance.

        Only tools with a positive score are returned.
        """
        tokens = tokenize_query(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        scored = [
            (name, float(score))
            for name, score in zip(self._names, scores)
            if score > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored[: self._top_k]]

    def get_tools(self, names: list[str]) -> list[ToolParam]:
        """Return full tool definitions for the given names.

        Unknown names are silently skipped.
        """
        return [self._tools_by_name[n] for n in names if n in self._tools_by_name]

    def get_tool(self, name: str) -> ToolParam | None:
        """Return a single tool definition by name, or None."""
        return self._tools_by_name.get(name)
