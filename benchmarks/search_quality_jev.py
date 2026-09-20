"""BM25 vs Jev vs BM25+Jev hybrid on the same 30 ground-truth queries.

Needs AI_GATEWAY_API_KEY (Vercel AI Gateway). ~60 Jev requests, well under a cent.

Usage:
    uv run python benchmarks/search_quality_jev.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root + "/src")
sys.path.insert(0, _root)

from benchmarks._tools import BASE_TOOLS, GROUND_TRUTH  # noqa: E402
from benchmarks.search_quality import (  # noqa: E402
    _precision_at_k,
    _recall_at_k,
    _reciprocal_rank,
)
from dehydrator import JevReranker, ToolIndex  # noqa: E402

KS = [1, 3, 5, 10]


def main() -> None:
    bm25 = ToolIndex(BASE_TOOLS, top_k=10)
    full = JevReranker()
    hybrid_rr = JevReranker()
    hybrid = ToolIndex(BASE_TOOLS, top_k=10, reranker=hybrid_rr, candidates=10)
    fallbacks = 0

    results: dict[str, list[tuple[list[str], set[str]]]] = {
        "bm25": [],
        "jev (all tools)": [],
        "bm25 -> jev": [],
    }
    for query, expected in GROUND_TRUTH:
        rel = set(expected)
        results["bm25"].append((bm25.search(query), rel))
        results["jev (all tools)"].append((full.rerank(query, BASE_TOOLS), rel))
        if full.last_error is not None:
            fallbacks += 1
            print(f"  fallback (all tools) {query!r}: {full.last_error}")
        results["bm25 -> jev"].append((hybrid.search(query), rel))
        if hybrid_rr.last_error is not None:
            fallbacks += 1
            print(f"  fallback (hybrid)    {query!r}: {hybrid_rr.last_error}")

    print(f"Corpus : {len(BASE_TOOLS)} tools   Queries: {len(GROUND_TRUTH)}\n")
    head = f"  {'Metric':<14}" + "".join(f"{n:>18}" for n in results)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for k in KS:
        row = f"  {'Precision@'+str(k):<14}"
        for runs in results.values():
            m = statistics.mean(_precision_at_k(r, rel, k) for r, rel in runs)
            row += f"{m:>17.1%} "
        print(row)
    for k in KS:
        row = f"  {'Recall@'+str(k):<14}"
        for runs in results.values():
            m = statistics.mean(_recall_at_k(r, rel, k) for r, rel in runs)
            row += f"{m:>17.1%} "
        print(row)
    row = f"  {'MRR':<14}"
    for runs in results.values():
        row += f"{statistics.mean(_reciprocal_rank(r, rel) for r, rel in runs):>17.1%} "
    print(row)
    print(f"\n  Jev requests that fell back to BM25 order: {fallbacks}")


if __name__ == "__main__":
    main()
