# Changelog

## 0.3.1

### Added
- **TypeSafe AI as a provider.** `JevReranker` and `JevIndex` call `api.typesafe.ai/v1/systemone` with model `jev-latest` when `TYPESAFE_API_KEY` is set, and fall back to Vercel AI Gateway with `AI_GATEWAY_API_KEY`. `JevReranker(provider="typesafe" | "gateway")` forces one; `detect_provider()` exposes the choice. The MCP gateway accepts either key.
- `BENCHMARKS.md` with results on both providers.

### Changed
- Retries now also cover 408 and TypeSafe's 529 (overloaded).

## 0.3.0

### Added
- **Jev search.** `JevReranker` re-orders the BM25 shortlist with TypeSafe AI's Jev via Vercel AI Gateway (`reranker=JevReranker()`). `JevIndex` replaces BM25 entirely (`search="jev"`), with a tournament for corpora above Jev's 255-option limit. On the 139-tool benchmark both modes reach 100% Precision@1 and MRR, up from 93.3% and 95.8% with BM25 alone. Standard library only, no new dependency.
- `Reranker` and `SearchIndex` protocols so custom rankers and indexes can be plugged into the clients.
- `ToolIndex(reranker=, candidates=)`.
- `dehydrator-mcp` console script: Dehydrator as an MCP gateway in front of other MCP servers, for Claude Code, Codex, Cursor, Claude Desktop, or any MCP client. Configured with `DEHYDRATOR_SERVERS`, `DEHYDRATOR_SEARCH`, `DEHYDRATOR_TOP_K`.
- `examples/mcp_chat.py`, `examples/e2e_gateway.py`: live end-to-end examples.
- `benchmarks/search_quality_jev.py`: compares BM25, hybrid, and Jev-only.

### Fixed
- `ToolIndex.from_mcp` and `get_tool_schema` now read `Tool.input_schema` as well as `Tool.inputSchema`. Current `mcp` SDK releases expose the snake_case name, so tools from real servers previously indexed with empty schemas.

## 0.2.0
- Multi-provider support (OpenAI-compatible clients) and MCP-native tool ingestion.

## 0.1.0
- Initial release: BM25 tool search for the Anthropic SDK.
