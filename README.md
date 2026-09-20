# Dehydrator

Client-side tool search for LLM APIs. Use thousands of tools without bloating the context window.

Works with **Anthropic**, **OpenAI**, and any **OpenAI-compatible** provider (Groq, OpenRouter, Chutes, etc.). Accepts tools from **MCP servers** natively. Search runs on **BM25** (local, free), on **[Jev](https://vercel.com/ai-gateway/models/jev)** (a decision model that reads meaning, not keywords), or on both.

## The problem

LLM APIs require you to send all tool definitions in every request. With 100+ tools, this wastes tokens and degrades tool selection. Anthropic offers a server-side `tool_search_tool_bm25`, but it's not available on all platforms (e.g. Bedrock) and doesn't work with ZDR. Dehydrator gives you the same capability client-side, so it works everywhere — with any provider.

## How it works

Dehydrator wraps your LLM client and replaces the full tool list with a single `tool_search` tool. When the model needs a tool, it searches by description. Dehydrator intercepts the call, runs the search locally, and re-calls the API with only the matched tools injected.

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
│  search (BM25 and/or Jev)   │
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

No extra dependency is needed for Jev; it uses the standard library.

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

Tools from MCP servers use `inputSchema` (camelCase) or `input_schema`. Dehydrator accepts both, as dicts or as `mcp.types.Tool` objects:

```python
tools = (await session.list_tools()).tools   # list[mcp.types.Tool]

from dehydrator import ToolIndex
index = ToolIndex.from_mcp(tools, top_k=5)
```

## Choosing a search mode

| Mode | How to enable | Cost | Best for |
|---|---|---|---|
| **BM25** (default) | nothing | free, offline | Tool descriptions and user queries share vocabulary. |
| **BM25 + Jev** | `reranker=JevReranker()` | ~$0.00002 / search | Best overall: BM25 shortlists 10, Jev picks. Keeps BM25 recall, fixes its ranking mistakes. |
| **Jev only** | `search="jev"` | ~$0.00016 / search at 139 tools | Queries that share no words with tool descriptions: paraphrases, other languages. |

```python
# BM25 + Jev
client = DehydratedClient(anthropic.Anthropic(), tools=tools, reranker=JevReranker())

# Jev only
client = DehydratedClient(anthropic.Anthropic(), tools=tools, search="jev")
```

Both work identically on `OpenAIDehydratedClient` and the async clients.

Why not always Jev only? BM25 is a hard gate: when the query shares no tokens with any description, BM25 returns nothing and Jev is never asked. `"Покажи останні 3 коміти"` returns no tools in BM25 or hybrid mode and routes to `git_log` at probability 1.00 in Jev-only mode. On the other hand Jev concentrates probability on the winner and leaves the tail unordered, so if you inject several tools per search (`top_k` > 1) the hybrid's recall is higher. Pick by your queries; numbers are in [Benchmarks](#benchmarks).

## Jev

[Jev](https://vercel.com/ai-gateway/models/jev) by TypeSafe AI is a decision model, not an LLM. It does not generate text. Given a *state* (here: the user's query) and a typed *question* (here: "which of these tools should be called?", with every tool as an option), it returns a calibrated probability for each option plus a confidence score, in about 400 ms. Dehydrator calls it through Vercel AI Gateway as `typesafe-ai/jev`. Input costs $0.042 per million tokens; output is free.

### Setup

1. Create a Vercel AI Gateway key at [vercel.com/ai-gateway](https://vercel.com/ai-gateway). Free credits are granted after a card is verified.
2. `export AI_GATEWAY_API_KEY=vck_...`
3. Add `reranker=JevReranker()` or `search="jev"` to your client.

### `JevReranker`

Re-orders a list of candidate tools for a query. Used by the hybrid mode and by `JevIndex`; also usable on its own.

```python
from dehydrator import JevReranker

rr = JevReranker(
    api_key=None,          # default: AI_GATEWAY_API_KEY
    min_probability=0.0,   # drop candidates Jev scores below this
    timeout=30.0,
    retries=3,             # on 429/5xx, exponential backoff
)
ranked = rr.rerank("send an email", tools)   # list[str], best first

rr.last_probabilities   # {"send_email": 0.93, "send_slack_message": 0.07}
rr.last_confidence      # 0.86  (1 = concentrated, 0 = spread out)
rr.last_error           # None, or the exception if the last call fell back
```

If a request fails for any reason, `rerank` returns the candidates in the order it received them and sets `last_error`. In hybrid mode that means BM25 order, so enabling Jev can never make results worse than plain BM25.

`min_probability` shrinks the injected tool list: with `min_probability=0.05`, tools Jev considers irrelevant are dropped even if `top_k` has room.

### `JevIndex`

Jev-only search. Same interface as `ToolIndex`.

```python
from dehydrator import JevIndex

index = JevIndex(tools, top_k=5, batch_size=200, finalists=10)
index.search("is my working copy dirty")     # ["git_status", ...]
index.last_probabilities, index.last_confidence
```

Jev accepts at most 255 options per question. Above `batch_size` tools, `JevIndex` runs a tournament: each batch gets a Jev round and the top `finalists` from every batch meet in a final round. With 1,000 tools that is six requests per search. Under 200 tools it is one request.

### Reading the output

Use the probabilities and confidence, not just the winner:

- `git_status 1.00, confidence 1.0`: route and move on.
- `move_file 0.88, create_directory 0.12, confidence 0.86`: two steps may be needed.
- `search_files 0.92, read_multiple_files 0.07`: the query was vague ("look at the tests"); consider asking the user.

A useful pattern is to escalate to a human or a larger model when `last_confidence` is below a threshold you choose per action.

### Dehydrator as an MCP gateway

`examples/mcp_server.py` runs Dehydrator as an MCP server in front of any number of other MCP servers. Clients such as Claude Code see two tools, `tool_search` and `call_tool`, instead of hundreds:

```bash
export AI_GATEWAY_API_KEY=vck_...
export DEHYDRATOR_SERVERS='{"fs":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/my/project"]},
                            "git": {"command": "uvx", "args": ["mcp-server-git"]}}'
export DEHYDRATOR_SEARCH=jev       # or bm25 (hybrid when a key is set)

claude mcp add dehydrator \
  -e AI_GATEWAY_API_KEY=$AI_GATEWAY_API_KEY \
  -e DEHYDRATOR_SERVERS="$DEHYDRATOR_SERVERS" \
  -e DEHYDRATOR_SEARCH=$DEHYDRATOR_SEARCH \
  -- uv run --directory /path/to/dehydrator python examples/mcp_server.py
```

`tool_search` results include each tool's argument schema and its Jev probability. `call_tool` forwards to whichever upstream server owns the tool.

## API

### `DehydratedClient(client, tools, *, top_k=5, always_available=None, max_search_rounds=3, reranker=None, search="bm25")`

Wraps an `anthropic.Anthropic` client.

| Parameter | Type | Description |
|---|---|---|
| `client` | `anthropic.Anthropic` | An Anthropic SDK client instance |
| `tools` | `list[dict]` | Tool definitions (Anthropic or MCP format) |
| `top_k` | `int` | Max tools returned per search (default: 5) |
| `always_available` | `list[str]` | Tool names to include in every request, bypassing search |
| `max_search_rounds` | `int` | Max search iterations per `create()` call (default: 3) |
| `reranker` | `Reranker \| None` | Re-orders the BM25 shortlist, e.g. `JevReranker()` |
| `search` | `"bm25" \| "jev"` | Search backend. `"jev"` uses `JevIndex` and no BM25 |

#### Methods

- **`client.messages.create(**kwargs)`** — Same signature as the Anthropic SDK. The `tools` kwarg is ignored (Dehydrator manages tools). Returns `anthropic.types.Message`.
- **`client.reset_discoveries()`** — Clears discovered tools. Call this when starting a new conversation.
- **`client.inner`** — Access the underlying `anthropic.Anthropic` client.

### `AsyncDehydratedClient`

Same API as `DehydratedClient`, but wraps `anthropic.AsyncAnthropic` and `create()` is async.

### `OpenAIDehydratedClient(client, tools, *, top_k=5, always_available=None, max_search_rounds=3, reranker=None, search="bm25")`

Wraps any OpenAI-compatible client. Same parameters as `DehydratedClient`; `client` is any object with `client.chat.completions.create()`, and tools are converted to OpenAI function format automatically.

#### Methods

- **`client.chat.completions.create(**kwargs)`** — Same signature as the OpenAI SDK. The `tools` kwarg is ignored. Returns the provider's response object.
- **`client.reset_discoveries()`** — Clears discovered tools.
- **`client.inner`** — Access the underlying client.

### `AsyncOpenAIDehydratedClient`

Same API as `OpenAIDehydratedClient`, but `create()` is async.

### `ToolIndex(tools, *, top_k=5, reranker=None, candidates=10)`

The BM25 index, standalone. With a `reranker`, BM25 retrieves `candidates` tools, the reranker orders them, and `top_k` are returned.

```python
from dehydrator import ToolIndex

index = ToolIndex(tools, top_k=5)
matched_names = index.search("weather forecast")
matched_tools = index.get_tools(matched_names)

index = ToolIndex.from_mcp(mcp_tools, top_k=5, reranker=JevReranker())
```

### `JevIndex`, `JevReranker`, `Reranker`, `SearchIndex`

See [Jev](#jev). `Reranker` and `SearchIndex` are protocols: implement `rerank(query, tools) -> list[str]` to plug in your own reranker, or `search / get_tools / get_tool / tool_names` to supply your own index to the adapters.

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

Benchmarked against **139 real tool definitions** from 6 popular MCP servers (Chrome DevTools, GitHub, Playwright, Filesystem, Git, Notion) and 30 ground-truth queries.

### Token savings

Sending all tools in every request is expensive. Dehydrator replaces them with a single `tool_search` tool and only injects the tools the model actually needs:

| Tools | top_k=3 | top_k=5 | top_k=10 | Baseline |
|------:|--------:|--------:|---------:|---------:|
| 50 | 274 tokens (94%) | 349 tokens (93%) | 678 tokens (86%) | 4,864 |
| 100 | 274 tokens (97%) | 349 tokens (96%) | 678 tokens (92%) | 8,954 |
| 200 | 274 tokens (98%) | 349 tokens (98%) | 678 tokens (96%) | 18,159 |

With 200 tools and `top_k=5`, you go from **18,159 → 349 tokens** per request — a **98% reduction**.

### Search quality

| Metric | BM25 | BM25 + Jev | Jev only |
|--------|-----:|-----------:|---------:|
| Precision@1 | 93.3% | **100.0%** | **100.0%** |
| Recall@1 | 59.7% | 66.4% | 66.4% |
| Recall@3 | 88.6% | 91.9% | 91.9% |
| Recall@5 | 95.3% | 96.1% | 91.9% |
| Recall@10 | 98.3% | 98.3% | 91.9% |
| **MRR** | 95.8% | **100.0%** | **100.0%** |
| input tokens / search | 0 | 553 | 3,741 |
| cost / search | $0 | $0.00002 | $0.00016 |
| median latency | <1 ms | 386 ms | 412 ms |

All three modes find a correct tool in the top 10 for 30/30 queries. BM25 alone misses top-1 on two lexical traps (`get_workflow_run` over `run_workflow`, `create_pull_request_review` over `create_pull_request`); both Jev modes fix them. Jev-only trades recall in the tail for independence from vocabulary.

Jev-only above 200 tools uses the tournament, which is several sequential requests per search. The gateway currently throttles around 3 concurrent requests, so expect a few seconds per search on very large corpora.

### Run the benchmarks

```bash
uv run python benchmarks/search_quality.py       # BM25, local, no API key
uv run python benchmarks/token_savings_openai.py  # local, uses tiktoken
uv run python benchmarks/search_quality_jev.py    # all three modes, needs AI_GATEWAY_API_KEY
```

## Examples

| File | What it shows |
|---|---|
| `examples/mcp_server.py` | Dehydrator as an MCP gateway in front of other MCP servers |
| `examples/mcp_chat.py` | Interactive chat: real MCP servers, LLM via Vercel AI Gateway, Jev search |
| `examples/e2e_gateway.py` | Five prompts with and without Jev against the benchmark corpus |

All examples need only `AI_GATEWAY_API_KEY`. Run with `uv run --with openai python examples/<file>`.

## Limitations

- **No streaming** — `stream=True` raises `NotImplementedError`. Planned for a future release.
- **Reserved tool name** — You cannot have a tool named `tool_search`. Dehydrator will raise `ValueError` if you do.
- **Jev option cap** — one Jev question holds at most 255 tools; `JevIndex` handles more via the tournament.

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
