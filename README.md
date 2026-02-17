# Dehydrator

Client-side BM25 tool search for the Anthropic SDK. Use thousands of tools without bloating the context window.

## The problem

The Anthropic API requires you to send all tool definitions in every request. With 100+ tools, this wastes tokens and degrades tool selection. Anthropic offers a server-side `tool_search_tool_bm25`, but it's closed-source, not ZDR-eligible, and unavailable on some platforms (e.g. Bedrock).

## How it works

Dehydrator wraps your Anthropic client and replaces the full tool list with a single `tool_search` tool. When Claude needs a tool, it searches by description. Dehydrator intercepts the call, runs BM25 locally, and re-calls the API with only the matched tools injected.

```
User request
    │
    ▼
┌─────────────────────────────┐
│  API call #1                │
│  tools = [tool_search]      │
│                             │
│  Claude responds:           │
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
│  Claude responds:           │
│  send_email({...})          │
└─────────────────────────────┘
              │
              ▼
        Returned to you
```

Only the tools Claude actually needs are ever sent. Discovered tools persist across turns within a conversation.

## Installation

```bash
pip install dehydrator
```

## Quick start

```python
import anthropic
from dehydrator import DehydratedClient

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email message to a recipient",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "body"],
        },
    },
    # ... hundreds more tools
]

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

The response is a standard `anthropic.types.Message` — handle tool calls exactly as you normally would.

## API

### `DehydratedClient(client, tools, *, top_k=5, always_available=None, max_search_rounds=3)`

| Parameter | Type | Description |
|---|---|---|
| `client` | `anthropic.Anthropic` | An Anthropic SDK client instance |
| `tools` | `list[dict]` | All tool definitions |
| `top_k` | `int` | Max tools returned per search (default: 5) |
| `always_available` | `list[str]` | Tool names to include in every request, bypassing search |
| `max_search_rounds` | `int` | Max search iterations per `create()` call (default: 3) |

#### Methods

- **`client.messages.create(**kwargs)`** — Same signature as the Anthropic SDK. The `tools` kwarg is ignored (Dehydrator manages tools). Returns `anthropic.types.Message`.
- **`client.reset_discoveries()`** — Clears discovered tools. Call this when starting a new conversation.
- **`client.inner`** — Access the underlying `anthropic.Anthropic` client.

### `AsyncDehydratedClient`

Same API as `DehydratedClient`, but wraps `anthropic.AsyncAnthropic` and `create()` is async.

```python
import anthropic
from dehydrator import AsyncDehydratedClient

client = AsyncDehydratedClient(
    anthropic.AsyncAnthropic(),
    tools=tools,
)

response = await client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Send an email to alice@example.com"}],
)
```

### `ToolIndex`

The BM25 index is also available standalone if you want to use it directly.

```python
from dehydrator import ToolIndex

index = ToolIndex(tools, top_k=5)
matched_names = index.search("weather forecast")
matched_tools = index.get_tools(matched_names)
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

Discovered tools persist across calls to `create()`. If Claude found `send_email` in turn 1, it's still available in turn 2 without re-searching.

Call `client.reset_discoveries()` when starting a new conversation:

```python
# Turn 1: Claude discovers send_email
response = client.messages.create(...)

# Turn 2: send_email is still available
response = client.messages.create(...)

# New conversation
client.reset_discoveries()
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
