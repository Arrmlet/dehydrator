# Dehydrator

Client-side BM25 tool search for LLM APIs. Use thousands of tools without bloating the context window.

Works with **Anthropic**, **OpenAI**, and any **OpenAI-compatible** provider (Groq, OpenRouter, Chutes, etc.). Accepts tools from **MCP servers** natively.

## The problem

LLM APIs require you to send all tool definitions in every request. With 100+ tools, this wastes tokens and degrades tool selection. Anthropic offers a server-side `tool_search_tool_bm25`, but it's not available on all platforms (e.g. Bedrock) and doesn't work with ZDR. Dehydrator gives you the same capability client-side, so it works everywhere — with any provider.

## How it works

Dehydrator wraps your LLM client and replaces the full tool list with a single `tool_search` tool. When the model needs a tool, it searches by description. Dehydrator intercepts the call, runs BM25 locally, and re-calls the API with only the matched tools injected.

```
User request
    │
    ▼
┌─────────────────────────────┐
│  API call #1                │
│  tools = [tool_search]      │
│                             │
│  Model responds:            │
│  tool_search("send email")  │
└─────────────┬───────────────┘
              │  intercepted by Dehydrator
              ▼
┌─────────────────────────────┐
│  BM25 search (local)        │
│  → matches: send_email,     │
│     send_slack_message       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  API call #2                │
│  tools = [tool_search,      │
│           send_email,        │
│           send_slack_message]│
│                             │
│  Model responds:            │
│  send_email({...})          │
└─────────────────────────────┘
              │
              ▼
        Returned to you
```

Only the tools the model actually needs are ever sent. Discovered tools persist across turns within a conversation.

## Installation

```bash
pip install dehydrator
```

## Quick start

### Anthropic

```python
import anthropic
from dehydrator import DehydratedClient

client = DehydratedClient(
    anthropic.Anthropic(),
    tools=tools,
    top_k=5,
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
)
```

The response is a standard `anthropic.types.Message`.

### OpenAI-compatible (OpenAI, Groq, OpenRouter, Chutes, etc.)

```python
from openai import OpenAI
from dehydrator import OpenAIDehydratedClient

client = OpenAIDehydratedClient(
    OpenAI(),
    tools=tools,
    top_k=5,
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
)
```

Works with any client that implements `client.chat.completions.create()`. No `openai` import required — fully duck-typed.

### MCP tools

Tools from MCP servers use `inputSchema` (camelCase) instead of `input_schema`. Dehydrator accepts both formats automatically:

```python
# MCP format tools work directly
tools = [
    {"name": "get_weather", "description": "...", "inputSchema": {...}},
]
client = DehydratedClient(anthropic.Anthropic(), tools=tools)

# Or use mcp.types.Tool objects with ToolIndex.from_mcp()
from dehydrator import ToolIndex

tools = await session.list_tools()  # returns list[mcp.types.Tool]
index = ToolIndex.from_mcp(tools, top_k=5)
```

## API

### `DehydratedClient(client, tools, *, top_k=5, always_available=None, max_search_rounds=3)`

Wraps an `anthropic.Anthropic` client.

| Parameter | Type | Description |
|---|---|---|
| `client` | `anthropic.Anthropic` | An Anthropic SDK client instance |
| `tools` | `list[dict]` | Tool definitions (Anthropic or MCP format) |
| `top_k` | `int` | Max tools returned per search (default: 5) |
| `always_available` | `list[str]` | Tool names to include in every request, bypassing search |
| `max_search_rounds` | `int` | Max search iterations per `create()` call (default: 3) |

#### Methods

- **`client.messages.create(**kwargs)`** — Same signature as the Anthropic SDK. The `tools` kwarg is ignored (Dehydrator manages tools). Returns `anthropic.types.Message`.
- **`client.reset_discoveries()`** — Clears discovered tools. Call this when starting a new conversation.
- **`client.inner`** — Access the underlying `anthropic.Anthropic` client.

### `AsyncDehydratedClient`

Same API as `DehydratedClient`, but wraps `anthropic.AsyncAnthropic` and `create()` is async.

### `OpenAIDehydratedClient(client, tools, *, top_k=5, always_available=None, max_search_rounds=3)`

Wraps any OpenAI-compatible client.

| Parameter | Type | Description |
|---|---|---|
| `client` | any | Any client with `client.chat.completions.create()` |
| `tools` | `list[dict]` | Tool definitions (Anthropic or MCP format — converted to OpenAI format automatically) |
| `top_k` | `int` | Max tools returned per search (default: 5) |
| `always_available` | `list[str]` | Tool names to include in every request, bypassing search |
| `max_search_rounds` | `int` | Max search iterations per `create()` call (default: 3) |

#### Methods

- **`client.chat.completions.create(**kwargs)`** — Same signature as the OpenAI SDK. The `tools` kwarg is ignored. Returns the provider's response object.
- **`client.reset_discoveries()`** — Clears discovered tools.
- **`client.inner`** — Access the underlying client.

### `AsyncOpenAIDehydratedClient`

Same API as `OpenAIDehydratedClient`, but `create()` is async.

### `ToolIndex`

The BM25 index is also available standalone if you want to use it directly.

```python
from dehydrator import ToolIndex

index = ToolIndex(tools, top_k=5)
matched_names = index.search("weather forecast")
matched_tools = index.get_tools(matched_names)

# From MCP Tool objects
index = ToolIndex.from_mcp(mcp_tools, top_k=5)
```

## Always-available tools

Some tools should always be in context (e.g. a `help` tool). Pass their names to `always_available`:

```python
client = DehydratedClient(
    anthropic.Anthropic(),
    tools=tools,
    always_available=["help", "get_current_user"],
)
```

These tools are sent in every request without requiring a search.

## Multi-turn conversations

Discovered tools persist across calls to `create()`. If the model found `send_email` in turn 1, it's still available in turn 2 without re-searching.

Call `client.reset_discoveries()` when starting a new conversation:

```python
# Turn 1: model discovers send_email
response = client.messages.create(...)

# Turn 2: send_email is still available
response = client.messages.create(...)

# New conversation
client.reset_discoveries()
```

## Benchmarks

Benchmarked against **139 real tool definitions** from 6 popular MCP servers (Chrome DevTools, GitHub, Playwright, Filesystem, Git, Notion).

### Token savings

Sending all tools in every request is expensive. Dehydrator replaces them with a single `tool_search` tool and only injects the tools the model actually needs:

| Tools | top_k=3 | top_k=5 | top_k=10 | Baseline |
|------:|--------:|--------:|---------:|---------:|
| 50 | 274 tokens (94%) | 349 tokens (93%) | 678 tokens (86%) | 4,864 |
| 100 | 274 tokens (97%) | 349 tokens (96%) | 678 tokens (92%) | 8,954 |
| 200 | 274 tokens (98%) | 349 tokens (98%) | 678 tokens (96%) | 18,159 |

With 200 tools and `top_k=5`, you go from **18,159 → 349 tokens** per request — a **98% reduction**.

### Search quality

BM25 finds the right tools reliably across all 6 MCP servers:

| Metric | k=3 | k=5 | k=10 |
|--------|----:|----:|-----:|
| Precision@k | 51.1% | 32.7% | 17.3% |
| Recall@k | 88.6% | 95.3% | 98.3% |
| **MRR** | | **95.8%** | |

30/30 test queries found at least one correct tool in the top 10. The right tool is ranked #1 or #2 in almost every case.

### Run the benchmarks

```bash
uv run python benchmarks/search_quality.py       # local, no API key
uv run python benchmarks/token_savings_openai.py  # local, uses tiktoken
```

## Limitations

- **No streaming** — `stream=True` raises `NotImplementedError`. Planned for a future release.
- **Reserved tool name** — You cannot have a tool named `tool_search`. Dehydrator will raise `ValueError` if you do.

## Development

```bash
git clone https://github.com/Arrmlet/dehydrator.git
cd dehydrator
uv sync

uv run pytest           # tests
uv run ruff check src/  # lint
uv run mypy src/        # type check
```

## License

MIT
