"""End-to-end: LLM -> tool_search -> BM25 -> Jev re-rank -> LLM picks the tool.

Uses only AI_GATEWAY_API_KEY: the LLM goes through Vercel AI
Gateway's OpenAI-compatible endpoint, and Jev through /v1/evaluate.
Gemini 2.5 Flash is used because it is available on the free tier; set
LLM_MODEL to override.

    uv run --with openai python examples/e2e_gateway.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from openai import OpenAI

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root + "/src")
sys.path.insert(0, _root)

from benchmarks._tools import BASE_TOOLS  # noqa: E402
from dehydrator import JevReranker, OpenAIDehydratedClient  # noqa: E402

KEY = os.environ["AI_GATEWAY_API_KEY"]
LLM = os.environ.get("LLM_MODEL", "google/gemini-2.5-flash")

PROMPTS = [
    (
        "Run the CI workflow named deploy.yml on the main branch of acme/web.",
        "run_workflow",
    ),
    ("Take a screenshot of the current page.", "take_screenshot"),
    ("Show me the git log of the last 5 commits in /repo.", "git_log"),
    ("Merge PR #42 in acme/web with a squash.", "merge_pull_request"),
    ("Find all Notion pages that mention 'roadmap'.", "notion_search"),
]


def run(reranker: JevReranker | None) -> None:
    label = "BM25 + Jev" if reranker else "BM25 only"
    print(f"\n=== {label} ===")
    llm = OpenAI(base_url="https://ai-gateway.vercel.sh/v1", api_key=KEY)
    client = OpenAIDehydratedClient(llm, tools=BASE_TOOLS, top_k=3, reranker=reranker)
    for prompt, expected in PROMPTS:
        client.reset_discoveries()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=LLM,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        dt = time.perf_counter() - t0
        calls = resp.choices[0].message.tool_calls or []
        called = [tc.function.name for tc in calls]
        discovered = sorted(client._discovered)
        ok = expected in called
        jev = ""
        if reranker and reranker.last_probabilities:
            top = max(reranker.last_probabilities, key=reranker.last_probabilities.get)
            jev = f"  jev_top={top} p={reranker.last_probabilities[top]:.2f}"
            if reranker.last_error:
                jev += f"  FALLBACK({reranker.last_error})"
        print(
            f"[{'OK ' if ok else 'BAD'}] {prompt[:52]:52s} discovered={discovered} called={called}{jev}  {dt:.1f}s"
        )


if __name__ == "__main__":
    run(None)
    run(JevReranker())
