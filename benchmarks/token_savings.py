"""Token savings benchmark — requires ANTHROPIC_API_KEY.

Uses ``client.messages.count_tokens()`` to compare:
- **Baseline**: all N tools sent to the model
- **Dehydrated**: ``SEARCH_TOOL_DEFINITION`` + top_k tools

Grid: tool counts [50, 100, 200] × top_k [3, 5, 10].

Usage:
    ANTHROPIC_API_KEY=sk-... uv run python benchmarks/token_savings.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so both ``dehydrator`` and
# ``benchmarks`` are importable when running as a script.
_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root + "/src")
sys.path.insert(0, _root)

from dehydrator._search_tool import SEARCH_TOOL_DEFINITION

from benchmarks._tools import generate_tools


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping token savings benchmark.")
        return

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    model = "claude-sonnet-4-20250514"
    tool_counts = [50, 100, 200]
    top_ks = [3, 5, 10]

    # Minimal message for counting — content doesn't affect tool token count much.
    messages = [{"role": "user", "content": "Hello"}]

    # Header
    print(f"Model: {model}")
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
        tool_defs: list[dict] = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

        # Baseline: send all tools
        baseline = client.messages.count_tokens(
            model=model,
            messages=messages,
            tools=tool_defs,  # type: ignore[arg-type]
        )
        baseline_tokens = baseline.input_tokens

        row = f"  {n:>6}"
        for k in top_ks:
            # Dehydrated: search tool + top_k tools
            dehydrated_tools: list[dict] = [SEARCH_TOOL_DEFINITION] + tool_defs[:k]
            dehydrated = client.messages.count_tokens(
                model=model,
                messages=messages,
                tools=dehydrated_tools,  # type: ignore[arg-type]
            )
            dehydrated_tokens = dehydrated.input_tokens
            saving_pct = (1 - dehydrated_tokens / baseline_tokens) * 100
            row += f" | {dehydrated_tokens:>{col_w},} {saving_pct:>{col_w - 1}.1f}%"

        row += f"   (baseline: {baseline_tokens:,})"
        print(row)

    print()
    print("Savings = 1 - (dehydrated_tokens / baseline_tokens)")


if __name__ == "__main__":
    main()
