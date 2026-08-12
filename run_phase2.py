#!/usr/bin/env python3
"""Phase 2 driver: rungs 1-3 of the strategy ladder measured side by side.

    ollama serve &            # required: dense retrieval needs a live embedding model
    python3 run_phase2.py

The first run embeds all chunks (~10 minutes on an M-series Mac) and caches the vectors
under .cache/; later runs load them instantly. Pass --limit N to smoke-test the pipeline on
the first N PEPs -- results from a limited run are NOT comparable to the full corpus.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from speceval.chunking import chunk_corpus
from speceval.corpus import NON_AUTHORITATIVE, load_corpus
from speceval.embed import EmbeddingError, embed_cached
from speceval.evaluate import K, evaluate, format_per_category, format_table
from speceval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    OracleRetriever,
    RandomRetriever,
    load_queries,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="smoke-test on the first N PEPs only"
    )
    args = parser.parse_args()

    peps = load_corpus(REPO_ROOT / "peps" / "peps")
    if args.limit:
        peps = peps[: args.limit]
        print(f"!! --limit {args.limit}: results are NOT comparable to the full corpus\n")

    chunks = chunk_corpus(peps)
    queries = load_queries(REPO_ROOT / "eval" / "queries_gold.json")
    non_authoritative = {
        pep.number for pep in peps if pep.status in NON_AUTHORITATIVE
    }

    print(f"{len(peps)} PEPs, {len(chunks)} chunks, {len(queries)} queries")
    try:
        vectors = embed_cached(
            [chunk.indexed_text for chunk in chunks], cache_dir=REPO_ROOT / ".cache"
        )
    except EmbeddingError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1
    print(f"vectors {vectors.shape}\n")

    lexical = BM25Retriever(chunks)
    dense = DenseRetriever(chunks=chunks, vectors=vectors)
    hybrid = HybridRetriever(lexical=lexical, dense=dense, chunks=chunks)

    retrievers = [
        OracleRetriever(),
        RandomRetriever(pep_numbers=[pep.number for pep in peps]),
        lexical,
        dense,
        hybrid,
    ]
    results = [
        evaluate(retriever, queries, k=K, non_authoritative=non_authoritative)
        for retriever in retrievers
    ]

    print("=" * 63)
    print(f"THE LADDER  (k={K}, {len(queries)} gold queries)")
    print("=" * 63)
    print(format_table(results, k=K))
    print()
    print("trap@1 = fraction of queries whose rank-1 PEP is non-authoritative")
    print("         (Rejected / Withdrawn / Superseded / Deferred). Lower is better.")
    print("         A retrieval-side proxy for the citation metrics Phase 3 measures.")
    print()
    print("Oracle 1.000/1.000 and Random ~0.000 are the harness validating itself.")

    print()
    print("=" * 63)
    print("PER CATEGORY")
    print("=" * 63)
    print(format_per_category(results, k=K))
    counts = Counter(query.category for query in queries)
    print("n per category: " + ", ".join(f"{c}={n}" for c, n in sorted(counts.items())))

    print()
    print("=" * 63)
    print("TRAP vs ORDINARY")
    print("=" * 63)
    traps = [query for query in queries if query.trap]
    ordinary = [query for query in queries if not query.trap]
    print(f"{'Retriever':<12}" + f"{'trap (n=' + str(len(traps)) + ')':>26}"
          f"{'ordinary (n=' + str(len(ordinary)) + ')':>26}")
    print("-" * 64)
    for retriever in (lexical, dense, hybrid):
        row = f"{retriever.name:<12}"
        for subset in (traps, ordinary):
            scores = evaluate(
                retriever, subset, k=K, non_authoritative=non_authoritative
            )
            row += (
                f"{'R ' + format(scores.overall.recall, '.2f') + ' N ' + format(scores.overall.ndcg, '.2f') + ' T ' + format(scores.trap_at_1 or 0.0, '.2f'):>26}"
            )
        print(row)
    print()
    print("R = Recall@10, N = nDCG@10, T = trap@1. The ordinary subset is the control:")
    print("a reranker that only improves the trap column is buying its gain somewhere.")

    print()
    print("=" * 63)
    print("PER QUERY  (rank-1 PEP by strategy)")
    print("=" * 63)
    by_number = {pep.number: pep for pep in peps}

    def cell(pep_number: int | None) -> str:
        if pep_number is None:
            return "-"
        pep = by_number.get(pep_number)
        status = (pep.status[:4] if pep else "?")
        return f"{pep_number}[{status}]"

    header = f"{'qid':<5}{'cat':<13}{'BM25':>13}{'Dense':>13}{'Hybrid':>13}  want"
    print(header)
    print("-" * len(header))
    for query in queries:
        row = f"{query.qid:<5}{query.category:<13}"
        for retriever in (lexical, dense, hybrid):
            ranked = retriever.search(query, top_k=K)
            marker = "*" if ranked and ranked[0] in query.relevant else " "
            row += f"{cell(ranked[0] if ranked else None) + marker:>13}"
        row += f"  {sorted(query.relevant)}"
        print(row)
    print()
    print("* = rank-1 result is a labelled-relevant PEP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
