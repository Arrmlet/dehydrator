"""Chat with real MCP servers through Dehydrator, with Jev re-ranking tool search.

Spawns MCP servers over stdio, indexes their tools with ToolIndex.from_mcp,
and runs an LLM through Vercel AI Gateway (OpenAI-compatible). When the LLM
calls tool_search, BM25 shortlists and Jev re-ranks; when it calls a real
tool, the call is forwarded to the owning MCP server and the result fed back.

    uv run --with openai python examples/mcp_chat.py                # interactive
    uv run --with openai python examples/mcp_chat.py "list files in the repo root"
    LLM_MODEL=anthropic/claude-haiku-4.5 ... (needs paid gateway tier)

Env: AI_GATEWAY_API_KEY (required), LLM_MODEL (default google/gemini-2.5-flash),
     NO_JEV=1 to compare plain BM25.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root + "/src")

from dehydrator import AsyncOpenAIDehydratedClient, JevReranker  # noqa: E402
from dehydrator._types import mcp_tool_to_dict  # noqa: E402

REPO = str(Path(_root))
SERVERS: dict[str, StdioServerParameters] = {
    "fs": StdioServerParameters(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", REPO]
    ),
    "git": StdioServerParameters(command="uvx", args=["mcp-server-git"]),
}
KEY = os.environ["AI_GATEWAY_API_KEY"]
LLM = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")
SYSTEM = (
    f"You are a coding assistant working in the git repository at {REPO}. "
    "Discover tools with tool_search before using them. Use tools to answer; "
    "when a tool call asks for repo_path or a path, use that repository path."
)


async def main(one_shot: str | None) -> None:
    async with AsyncExitStack() as stack:
        sessions: dict[str, ClientSession] = {}
        owner: dict[str, str] = {}
        mcp_tools: list[Any] = []
        for name, params in SERVERS.items():
            read, write = await stack.enter_async_context(stdio_client(params))
            s = await stack.enter_async_context(ClientSession(read, write))
            await s.initialize()
            tools = (await s.list_tools()).tools
            sessions[name] = s
            for t in tools:
                owner[t.name] = name
            mcp_tools.extend(tools)
            print(f"[mcp] {name}: {len(tools)} tools")
        print(f"[mcp] {len(mcp_tools)} tools total; LLM sees only tool_search\n")

        reranker = None if os.environ.get("NO_JEV") else JevReranker()
        llm = AsyncOpenAI(base_url="https://ai-gateway.vercel.sh/v1", api_key=KEY)
        client = AsyncOpenAIDehydratedClient(
            llm,
            tools=[mcp_tool_to_dict(t) for t in mcp_tools],
            top_k=3,
            reranker=reranker,
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]

        async def turn(user: str) -> None:
            messages.append({"role": "user", "content": user})
            for _ in range(8):  # tool-call rounds
                t0 = time.perf_counter()
                resp = await client.chat.completions.create(
                    model=LLM, messages=messages, max_tokens=800
                )
                msg = resp.choices[0].message
                if reranker and reranker.last_probabilities:
                    top = max(
                        reranker.last_probabilities, key=reranker.last_probabilities.get
                    )
                    print(
                        f"  [jev] top={top} p={reranker.last_probabilities[top]:.2f} "
                        f"conf={reranker.last_confidence} discovered={sorted(client._discovered)}"
                    )
                    reranker.last_probabilities = {}
                if not msg.tool_calls:
                    print(
                        f"\nassistant ({time.perf_counter() - t0:.1f}s): {msg.content}\n"
                    )
                    messages.append({"role": "assistant", "content": msg.content or ""})
                    return
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    srv = owner.get(tc.function.name)
                    print(
                        f"  [tool] {tc.function.name}({json.dumps(args)[:100]}) -> {srv}"
                    )
                    if srv is None:
                        content = f"unknown tool {tc.function.name}"
                    else:
                        res = await sessions[srv].call_tool(tc.function.name, args)
                        content = "\n".join(
                            getattr(c, "text", "") for c in res.content
                        )[:4000]
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": content}
                    )
            print("  [stop] too many tool rounds")

        if one_shot:
            await turn(one_shot)
            return
        print("type a request (q to quit)")
        while True:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user in ("q", "quit", "exit"):
                break
            if user:
                await turn(user)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or None))
