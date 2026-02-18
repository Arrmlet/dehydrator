"""Token savings benchmark using tiktoken (local, no API calls).

Counts tokens via tiktoken (GPT-4o tokenizer) to compare:
- **Baseline**: all N tools
- **Dehydrated**: search tool + top_k tools

Grid: tool counts [50, 100, 200] × top_k [3, 5, 10].

Usage:
    uv run python benchmarks/token_savings_openai.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root + "/src")
sys.path.insert(0, _root)

import tiktoken

from dehydrator._search_tool import SEARCH_TOOL_DEFINITION

from benchmarks._tools import generate_tools


def _count_tokens(tools: list[dict], enc: tiktoken.Encoding) -> int:
    """Count tokens for a list of tool definitions."""
    return len(enc.encode(json.dumps(tools)))


def main() -> None:
    enc = tiktoken.encoding_for_model("gpt-4o")

    tool_counts = [50, 100, 200]
    top_ks = [3, 5, 10]

    print("Token savings benchmark (tiktoken / GPT-4o tokenizer)")
    print()
    col_w = 14
    header = f"  {'Tools':>6}"
    for k in top_ks:
        header += f" | {'top_k=' + str(k):^{col_w * 2 + 3}}"
    print(header)

    sub = f"  {'':>6}"
    for _ in top_ks:
        sub += f" | {'tokens':>{col_w}} {'savings':>{col_w}}"
    print(sub)
    print("  " + "-" * (len(sub) - 2))

    for n in tool_counts:
        tools = generate_tools(n)
        baseline_tokens = _count_tokens(tools, enc)

        row = f"  {n:>6}"
        for k in top_ks:
            dehydrated_tools = [SEARCH_TOOL_DEFINITION] + tools[:k]
            dehydrated_tokens = _count_tokens(dehydrated_tools, enc)
            saving_pct = (1 - dehydrated_tokens / baseline_tokens) * 100
            row += f" | {dehydrated_tokens:>{col_w},} {saving_pct:>{col_w - 1}.1f}%"

        row += f"   (baseline: {baseline_tokens:,})"
        print(row)

    print()
    print("Savings = 1 - (dehydrated_tokens / baseline_tokens)")


if __name__ == "__main__":
    main()
