"""BM25 search quality benchmark — no API key needed.

Runs ground-truth queries through ``ToolIndex.search()`` and reports:
- Precision@k and Recall@k  (k = 3, 5, 10)
- MRR (Mean Reciprocal Rank)
- Per-query OK / MISS detail

Usage:
    uv run python benchmarks/search_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so both ``dehydrator`` and
# ``benchmarks`` are importable when running as a script.
_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root + "/src")
sys.path.insert(0, _root)

from dehydrator import ToolIndex

from benchmarks._tools import BASE_TOOLS, GROUND_TRUTH


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & relevant) / len(top)


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not relevant:
        return 0.0
    return len(set(top) & relevant) / len(relevant)


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, name in enumerate(retrieved):
        if name in relevant:
            return 1.0 / (i + 1)
    return 0.0


def main() -> None:
    ks = [3, 5, 10]
    max_k = max(ks)
    index = ToolIndex(BASE_TOOLS, top_k=max_k)

    precisions: dict[int, list[float]] = {k: [] for k in ks}
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    rrs: list[float] = []

    print(f"Corpus : {len(BASE_TOOLS)} tools")
    print(f"Queries: {len(GROUND_TRUTH)}")
    print()

    for query, expected in GROUND_TRUTH:
        relevant = set(expected)
        results = index.search(query)

        rr = _reciprocal_rank(results, relevant)
        rrs.append(rr)

        for k in ks:
            precisions[k].append(_precision_at_k(results, relevant, k))
            recalls[k].append(_recall_at_k(results, relevant, k))

        hit = bool(set(results[:max_k]) & relevant)
        status = "OK  " if hit else "MISS"
        # Show only first 5 results for readability
        shown = results[:5]
        matched = [r for r in shown if r in relevant]
        missed = [e for e in expected if e not in results[:max_k]]
        detail = f"got={shown}"
        if matched:
            detail += f"  matched={matched}"
        if missed:
            detail += f"  missed={missed}"
        print(f"  [{status}] {query!r}")
        print(f"         {detail}")

    print()
    print("=" * 68)
    print(f"  {'Metric':<24} {'k=3':>8} {'k=5':>8} {'k=10':>8}")
    print("-" * 68)

    for label, data in [("Precision@k", precisions), ("Recall@k", recalls)]:
        vals = {k: sum(v) / len(v) for k, v in data.items()}
        print(f"  {label:<24} {vals[3]:>7.1%} {vals[5]:>7.1%} {vals[10]:>7.1%}")

    mrr = sum(rrs) / len(rrs)
    print(f"  {'MRR':<24} {mrr:>7.1%}")
    print("=" * 68)

    # Summary
    hits = sum(1 for rr in rrs if rr > 0)
    print(f"\n  {hits}/{len(GROUND_TRUTH)} queries found at least one relevant tool in top-{max_k}")


if __name__ == "__main__":
    main()
