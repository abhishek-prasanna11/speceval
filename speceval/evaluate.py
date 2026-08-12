"""The evaluation loop: run a retriever over a query set and report the metrics.

Retrieval latency is timed here and *only* retrieval. From Phase 3 there will be a
separate end-to-end timing that includes generation; keeping the two apart is what makes
the four strategies comparable at all, since an LLM call is three orders of magnitude
slower than any of them and would swamp every difference.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from .metrics import mean, ndcg_at_k, recall_at_k
from .retrievers import Query, Retriever

K = 10


@dataclass
class Scores:
    recall: float
    ndcg: float
    n_queries: int


@dataclass
class Result:
    retriever: str
    overall: Scores
    per_category: dict[str, Scores] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    # Nearest-rank percentile: adequate here, and it never interpolates between two
    # real measurements into a value that was never observed.
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def evaluate(retriever: Retriever, queries: list[Query], k: int = K) -> Result:
    recalls: list[float] = []
    ndcgs: list[float] = []
    latencies_ms: list[float] = []
    by_category: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for query in queries:
        start = time.perf_counter()
        ranked = retriever.search(query, top_k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        relevant = set(query.relevant)
        recall = recall_at_k(ranked, relevant, k)
        ndcg = ndcg_at_k(ranked, relevant, k)

        recalls.append(recall)
        ndcgs.append(ndcg)
        by_category[query.category].append((recall, ndcg))

    latencies_ms.sort()
    return Result(
        retriever=retriever.name,
        overall=Scores(mean(recalls), mean(ndcgs), len(queries)),
        per_category={
            category: Scores(
                mean([r for r, _ in pairs]), mean([n for _, n in pairs]), len(pairs)
            )
            for category, pairs in sorted(by_category.items())
        },
        latency_p50_ms=_percentile(latencies_ms, 0.50),
        latency_p95_ms=_percentile(latencies_ms, 0.95),
    )


def format_table(results: list[Result], k: int = K) -> str:
    lines = [
        f"{'Retriever':<12} {'Recall@' + str(k):>10} {'nDCG@' + str(k):>10} "
        f"{'p50 ms':>9} {'p95 ms':>9}",
        "-" * 54,
    ]
    for result in results:
        lines.append(
            f"{result.retriever:<12} {result.overall.recall:>10.3f} "
            f"{result.overall.ndcg:>10.3f} {result.latency_p50_ms:>9.2f} "
            f"{result.latency_p95_ms:>9.2f}"
        )
    return "\n".join(lines)


def format_per_category(results: list[Result], k: int = K) -> str:
    categories = sorted({c for result in results for c in result.per_category})
    header = f"{'Retriever':<12}" + "".join(f"{c:>22}" for c in categories)
    lines = [header, "-" * len(header)]
    for result in results:
        row = f"{result.retriever:<12}"
        for category in categories:
            scores = result.per_category.get(category)
            cell = (
                f"R {scores.recall:.2f} / N {scores.ndcg:.2f}" if scores else "-"
            )
            row += f"{cell:>22}"
        lines.append(row)
    lines.append("")
    lines.append(f"(R = Recall@{k}, N = nDCG@{k}; n per category shown by run_phase1.py)")
    return "\n".join(lines)
