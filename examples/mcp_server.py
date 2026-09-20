"""Dehydrator as an MCP server: one gateway in front of many MCP servers.

Exposes two tools to the client (e.g. Claude Code):
  tool_search(query)          BM25 shortlist -> Jev re-rank -> ranked tools + schemas
  call_tool(name, arguments)  proxy the call to whichever upstream server owns it

Upstream servers are spawned over stdio from DEHYDRATOR_SERVERS (JSON) or the
defaults below (filesystem + git on this repo). DEHYDRATOR_SEARCH=bm25 (default,
BM25 shortlist re-ranked by Jev when a key is set) or jev (Jev only, no BM25).

Register with Claude Code:
  claude mcp add dehydrator -e AI_GATEWAY_API_KEY=$AI_GATEWAY_API_KEY -- \
      uv run --directory ~/dehydrator python examples/mcp_server.py
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import Context, MCPServer

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root + "/src")

from dehydrator import JevIndex, JevReranker, ToolIndex  # noqa: E402
from dehydrator._types import mcp_tool_to_dict  # noqa: E402

DEFAULT_SERVERS = {
    "fs": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", _root]},
    "git": {"command": "uvx", "args": ["mcp-server-git"]},
}
TOP_K = int(os.environ.get("DEHYDRATOR_TOP_K", "5"))
SEARCH = os.environ.get("DEHYDRATOR_SEARCH", "bm25")  # bm25 | jev


@dataclass
class State:
    index: ToolIndex | JevIndex
    reranker: JevReranker | None
    sessions: dict[str, ClientSession]
    owner: dict[str, str]
    log: list[dict[str, Any]] = field(default_factory=list)


def _log(msg: str) -> None:
    print(f"[dehydrator] {msg}", file=sys.stderr, flush=True)


@asynccontextmanager
async def lifespan(_: MCPServer):
    spec = json.loads(os.environ.get("DEHYDRATOR_SERVERS", "null")) or DEFAULT_SERVERS
    async with AsyncExitStack() as stack:
        sessions: dict[str, ClientSession] = {}
        owner: dict[str, str] = {}
        tools: list[dict[str, Any]] = []
        for name, cfg in spec.items():
            params = StdioServerParameters(
                command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env")
            )
            read, write = await stack.enter_async_context(stdio_client(params))
            s = await stack.enter_async_context(ClientSession(read, write))
            await s.initialize()
            listed = (await s.list_tools()).tools
            sessions[name] = s
            for t in listed:
                owner[t.name] = name
                tools.append(mcp_tool_to_dict(t))
            _log(f"upstream {name}: {len(listed)} tools")
        use_jev = bool(os.environ.get("AI_GATEWAY_API_KEY")) and not os.environ.get("NO_JEV")
        reranker = JevReranker() if use_jev else None
        if SEARCH == "jev":
            if reranker is None:
                raise SystemExit("DEHYDRATOR_SEARCH=jev needs AI_GATEWAY_API_KEY")
            index: ToolIndex | JevIndex = JevIndex(tools, top_k=TOP_K, reranker=reranker)
        else:
            index = ToolIndex(tools, top_k=TOP_K, reranker=reranker)
        mode = SEARCH if reranker else "bm25 (no key)"
        _log(f"{len(tools)} tools indexed; search={mode}")
        yield State(index, reranker, sessions, owner)


server = MCPServer(
    "dehydrator",
    instructions=(
        "Gateway to many tools. Call tool_search with a short description of what "
        "you want to do, then call_tool with the chosen tool's name and arguments."
    ),
    lifespan=lifespan,
)


@server.tool()
async def tool_search(query: str, ctx: Context) -> str:  # type: ignore[type-arg]
    """Find tools by describing the action you want (e.g. 'show git diff of unstaged changes').
    Returns the best-matching tools with their input schemas, ranked by relevance."""
    st: State = ctx.request_context.lifespan_context
    return _search(st, query)


def _search(st: State, query: str) -> str:
    names = st.index.search(query)
    tools = st.index.get_tools(names)
    rr = st.reranker
    probs = rr.last_probabilities if rr else {}
    lines = []
    for t in tools:
        p = probs.get(t["name"])
        tag = f" (jev p={p:.2f})" if p is not None else ""
        schema = t.get("input_schema") or t.get("inputSchema") or {}
        props = schema.get("properties", {})
        req = set(schema.get("required", []))
        args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}" for k, v in props.items())
        lines.append(f"- {t['name']}{tag}: {t['description']}\n    args: {args or 'none'}")
    if rr:
        extra = f"jev confidence={rr.last_confidence}" if rr.last_error is None else f"jev fallback: {rr.last_error}"
        lines.append(f"\n[{extra}]")
    return "\n".join(lines) if tools else "No matching tools. Try different words."


@server.tool()
async def call_tool(  # type: ignore[type-arg]
    name: str, ctx: Context, arguments: dict[str, Any] | None = None
) -> str:
    """Call a tool found via tool_search. `arguments` is the JSON object for that tool."""
    st: State = ctx.request_context.lifespan_context
    srv = st.owner.get(name)
    if srv is None:
        return f"Unknown tool {name!r}. Use tool_search first."
    res = await st.sessions[srv].call_tool(name, arguments or {})
    out = "\n".join(getattr(c, "text", "") for c in res.content)
    return out if out else json.dumps(getattr(res, "structuredContent", None) or {})


if __name__ == "__main__":
    server.run("stdio")
