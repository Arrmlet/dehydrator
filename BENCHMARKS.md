# Benchmarks

[Jev](https://typesafe.ai) (TypeSafe AI) vs BM25 for routing MCP tool calls. 139 real tool definitions from six MCP servers (Chrome DevTools, GitHub, Playwright, Filesystem, Git, Notion), 30 hand-labelled queries. Run on 2026-09-20 with `dehydrator` 0.3.0 and `typesafe-ai/jev` via Vercel AI Gateway.

| | BM25 | BM25 → Jev | Jev only |
|---|---:|---:|---:|
| **Precision@1** | 93.3% | **100%** | **100%** |
| Cost per search | $0 | $0.00002 | $0.00016 |
| Median latency | <1 ms | 386 ms | 412 ms |

## Search quality

| Metric | BM25 | BM25 → Jev | Jev only |
|---|---:|---:|---:|
| Precision@1 | 93.3% | **100.0%** | **100.0%** |
| Recall@1 | 59.7% | **66.4%** | **66.4%** |
| Recall@3 | 88.6% | **91.9%** | **91.9%** |
| Recall@5 | 95.3% | **96.1%** | 91.9% |
| Recall@10 | **98.3%** | **98.3%** | 91.9% |
| MRR | 95.8% | **100.0%** | **100.0%** |
| Queries with a hit in top 10 | 30 / 30 | 30 / 30 | 30 / 30 |
| Input tokens per search | 0 | 553 | 3,741 |
| Cost per search | $0 | $0.000023 | $0.000157 |
| Median latency | <1 ms | 386 ms | 412 ms |
| Mean Jev confidence | – | 0.86 | 0.85 |

**BM25 → Jev**: BM25 retrieves 10 candidates, Jev orders them, top-k returned. **Jev only**: one `choice` question with all 139 tools as options, no BM25.

Jev puts nearly all probability on its top pick, so positions 2 to 10 are effectively unordered. That is why Jev-only trails on Recall@5 and @10 while matching on top-1. The hybrid keeps BM25's tail. If you inject several tools per search, use the hybrid; if your queries share few words with your tool descriptions, use Jev only.

## Where BM25 goes wrong

Both BM25 misses in the benchmark set were lexical traps:

| Query | BM25 | Jev |
|---|---|---|
| "run a GitHub Actions workflow" | `get_workflow_run` | `run_workflow` p=1.00 |
| "create a pull request" | `create_pull_request_review` | `create_pull_request` p=1.00 |

A harder failure, outside the benchmark set, on 26 filesystem + git tools:

| Query | BM25 | BM25 → Jev | Jev only |
|---|---|---|---|
| "Покажи останні 3 коміти" (show the last 3 commits) | no matching tools | no matching tools | `git_log` p=1.00 |

BM25 is a hard gate on shared tokens. When the query and the descriptions share no words, the shortlist is empty and the hybrid inherits the failure because Jev is never asked. Jev-only has no such gate.

## What a Jev answer looks like

Query `"what did I change but not stage yet"` over 26 filesystem + git tools, Jev only. Three near-synonyms compete; the distribution separates them.

```
git_diff_unstaged   ██████████████████░░  0.90
git_status          ██░░░░░░░░░░░░░░░░░░  0.10
git_diff_staged     ░░░░░░░░░░░░░░░░░░░░  0.00
git_diff            ░░░░░░░░░░░░░░░░░░░░  0.00
read_text_file      ░░░░░░░░░░░░░░░░░░░░  0.00

confidence 0.89 · 1 request · ~400 ms
```

## Is the confidence real?

Separate check: 40 support tickets with known labels routed into 4 teams, one `choice` question each.

| Top probability | n | Mean p | Accuracy |
|---|---:|---:|---:|
| ≥ 0.90 | 38 | 1.00 | **100%** |
| 0.70 to 0.90 | 0 | – | – |
| < 0.70 | 2 | 0.57 | 50% |

Accuracy 97.5%, Brier score 0.010. The one miss ("TypeError in the console on login page", technical vs account) came back at p=0.51 with confidence 0.34. A threshold at 0.7 would have sent it to a human.

## Cost and latency

| Setup | Requests | Input tokens | Wall time |
|---|---:|---:|---:|
| 12 questions about one agent trace, one request | 1 | 742 | **378 ms** |
| Same 12 questions, one request each | 12 | 4,779 | 24 s incl. 429 backoff |

Price is $0.042 per million input tokens, output free. The gateway returns 429 above roughly 3 concurrent requests, so batch questions into one request and retry with backoff. One Jev question holds at most 255 options; `JevIndex` runs a tournament above that.

## Token savings

Independent of the ranker. Sending all tools in every request vs. one `tool_search` tool plus the injected hits:

| Tools | top_k=3 | top_k=5 | top_k=10 | Baseline |
|------:|--------:|--------:|---------:|---------:|
| 50 | 274 tokens (94%) | 349 tokens (93%) | 678 tokens (86%) | 4,864 |
| 100 | 274 tokens (97%) | 349 tokens (96%) | 678 tokens (92%) | 8,954 |
| 200 | 274 tokens (98%) | 349 tokens (98%) | 678 tokens (96%) | 18,159 |

## Method

Jev is called as `typesafe-ai/jev` through Vercel AI Gateway. State is `{"user_request": query}`. The question is one `choice` whose options are the candidate tool names with their descriptions as criteria. Tools are ranked by the returned probability. Probabilities are rounded to two decimals by the gateway.

Corpus and ground truth are in `benchmarks/_tools.py`. Each query has one or more correct tools.

## Reproduce

```bash
git clone https://github.com/Arrmlet/dehydrator && cd dehydrator && uv sync
uv run python benchmarks/search_quality.py                                  # BM25, offline
uv run python benchmarks/token_savings_openai.py                            # token savings, offline
AI_GATEWAY_API_KEY=vck_... uv run python benchmarks/search_quality_jev.py   # all three modes, ~60 requests
```
