#!/usr/bin/env python3
"""Phase 1 driver: ingest the PEP corpus, index it, and measure the BM25 baseline.

    python3 run_phase1.py

Reports corpus statistics (including the size of the authority trap surface), then runs
the seed queries through three retrievers: the validated Oracle and Random bounds, and
the real BM25 baseline that Phase 2 will be compared against.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from speceval.chunking import chunk_corpus
from speceval.corpus import NON_AUTHORITATIVE, load_corpus
from speceval.evaluate import K, evaluate, format_per_category, format_table
from speceval.retrievers import (
    BM25Retriever,
    OracleRetriever,
    RandomRetriever,
    load_queries,
)

REPO_ROOT = Path(__file__).resolve().parent


def main() -> int:
    try:
        peps = load_corpus(REPO_ROOT / "peps" / "peps")
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    chunks = chunk_corpus(peps)
    queries = load_queries(REPO_ROOT / "eval" / "queries_seed.json")

    statuses = Counter(pep.status for pep in peps)
    non_authoritative = sum(count for s, count in statuses.items() if s in NON_AUTHORITATIVE)
    versioned = sum(1 for pep in peps if pep.python_version)
    superseded_edges = sum(1 for pep in peps if pep.superseded_by)

    print("=" * 54)
    print("CORPUS")
    print("=" * 54)
    print(f"PEPs parsed            {len(peps)}")
    print(f"Chunks                 {len(chunks)}")
    print(f"Mean chunks per PEP    {len(chunks) / len(peps):.1f}")
    print(f"With Python-Version    {versioned} ({versioned / len(peps):.0%})")
    print(f"Superseded-By edges    {superseded_edges}")
    print(
        f"Non-authoritative      {non_authoritative} "
        f"({non_authoritative / len(peps):.0%}) <- the trap surface"
    )
    print()
    for status, count in statuses.most_common():
        marker = "  (trap)" if status in NON_AUTHORITATIVE else ""
        print(f"  {status:<12} {count:>4}{marker}")

    print()
    print("=" * 54)
    print(f"RETRIEVAL  (k={K}, {len(queries)} seed queries)")
    print("=" * 54)

    results = [
        evaluate(OracleRetriever(), queries, k=K),
        evaluate(
            RandomRetriever(pep_numbers=[pep.number for pep in peps]), queries, k=K
        ),
        evaluate(BM25Retriever(chunks), queries, k=K),
    ]
    print(format_table(results, k=K))
    print()
    print("Oracle must read 1.000/1.000 and Random must read near zero. Those two rows")
    print("are the harness validating itself; BM25 is the only real measurement here.")

    print()
    print("=" * 54)
    print("PER CATEGORY")
    print("=" * 54)
    print(format_per_category(results, k=K))
    counts = Counter(query.category for query in queries)
    print("n per category: " + ", ".join(f"{c}={n}" for c, n in sorted(counts.items())))

    print()
    print("=" * 54)
    print("PER QUERY  (BM25)")
    print("=" * 54)
    bm25 = BM25Retriever(chunks)
    by_number = {pep.number: pep for pep in peps}
    for query in queries:
        ranked = bm25.search(query, top_k=K)
        hits = set(ranked) & set(query.relevant)
        verdict = "HIT " if hits else "MISS"
        print(f"{verdict} {query.qid} [{query.category}] {query.text}")
        print(f"     want {sorted(query.relevant)}")
        top = ", ".join(
            f"{n}{'*' if n in query.relevant else ''}"
            f"[{by_number[n].status[:4] if n in by_number else '?'}]"
            for n in ranked[:5]
        )
        print(f"     got  {top}")
    print()
    print("* = labelled relevant. Bracketed text is the PEP's Status: a non-authoritative")
    print("  status appearing above a relevant PEP is exactly what Phase 4 must fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
