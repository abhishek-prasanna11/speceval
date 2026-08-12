#!/usr/bin/env python3
"""Phase 3 driver: generation over each rung's retrieved context, scored on citations.

    ollama serve &
    .venv/bin/python run_phase3.py

Generates one answer per (retriever, query) -- 3 x 51 -- and scores each automatically
against the PEP headers. Answers are cached under .cache/answers/ keyed on model, options
and prompt, so re-running is free and Phase 4 only pays for the rung it adds.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from speceval.answer_metrics import AnswerScores, aggregate, score_answer
from speceval.chunking import chunk_corpus
from speceval.corpus import NON_AUTHORITATIVE, load_corpus
from speceval.embed import EmbeddingError, embed_cached
from speceval.generate import (
    CachedGenerator,
    GenerationError,
    build_prompt,
    extract_all_cited_numbers,
    extract_citations,
)
from speceval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    load_queries,
)

REPO_ROOT = Path(__file__).resolve().parent
TOP_K = 5  # chunks fed to the generator, not PEPs -- see note below


def format_row(name: str, scores: AnswerScores) -> str:
    version = (
        "-" if scores.version_correct_rate is None else f"{scores.version_correct_rate:.3f}"
    )
    return (
        f"{name:<12}{scores.superseded_citation_rate:>12.3f}"
        f"{scores.authoritative_citation_rate:>14.3f}{version:>12}"
        f"{scores.hallucinated_citation_rate:>13.3f}{scores.latency_p95_ms:>11.0f}"
    )


def main() -> int:
    peps = load_corpus(REPO_ROOT / "peps" / "peps")
    by_number = {pep.number: pep for pep in peps}
    chunks = chunk_corpus(peps)
    queries = load_queries(REPO_ROOT / "eval" / "queries_gold.json")

    try:
        vectors = embed_cached(
            [chunk.indexed_text for chunk in chunks],
            cache_dir=REPO_ROOT / ".cache",
            progress=False,
        )
    except EmbeddingError as error:
        print(f"{error}", file=sys.stderr)
        return 1

    lexical = BM25Retriever(chunks)
    dense = DenseRetriever(chunks=chunks, vectors=vectors)
    hybrid = HybridRetriever(lexical=lexical, dense=dense, chunks=chunks)
    generator = CachedGenerator(cache_dir=REPO_ROOT / ".cache" / "answers")

    print(f"{len(peps)} PEPs, {len(chunks)} chunks, {len(queries)} queries, top_k={TOP_K} chunks")
    print(f"{sum(1 for q in queries if q.asked_version)} version-scoped queries\n")

    all_records = {}
    for retriever in (lexical, dense, hybrid):
        records = []
        for query in queries:
            chunk_indices = retriever.search_chunks(query, depth=TOP_K)
            prompt = build_prompt(query.text, [chunks[i] for i in chunk_indices])
            start = time.perf_counter()
            try:
                answer = generator.generate(prompt)
            except GenerationError as error:
                print(f"{error}", file=sys.stderr)
                return 1
            latency_ms = (time.perf_counter() - start) * 1000.0

            records.append(
                score_answer(
                    qid=query.qid,
                    category=query.category,
                    trap=query.trap,
                    answer=answer,
                    citations=extract_citations(answer, known=set(by_number)),
                    all_cited=extract_all_cited_numbers(answer),
                    relevant=set(query.relevant),
                    asked_version=query.asked_version,
                    peps=by_number,
                    latency_ms=latency_ms,
                )
            )
        all_records[retriever.name] = records
        print(f"  {retriever.name:<8} done ({generator.hits} cached, {generator.misses} generated)")

    header = (
        f"{'Retriever':<12}{'superseded':>12}{'authoritative':>14}"
        f"{'version':>12}{'hallucin.':>13}{'p95 ms':>11}"
    )
    print()
    print("=" * len(header))
    print("ANSWER-LEVEL METRICS  (all queries)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, records in all_records.items():
        print(format_row(name, aggregate(records)))
    print()
    print("superseded    = answers citing a Rejected/Withdrawn/Superseded/Deferred PEP (lower better)")
    print("authoritative = answers citing a gold-labelled PEP (higher better)")
    print("version       = version-scoped answers surfacing the real release (higher better)")
    print("hallucin.     = answers citing a PEP number absent from the corpus (lower better)")

    print()
    print("=" * len(header))
    print("TRAP vs ORDINARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, records in all_records.items():
        for label, subset in (
            ("trap", [r for r in records if r.trap]),
            ("ordinary", [r for r in records if not r.trap]),
        ):
            print(format_row(f"{name[:6]}/{label[:4]}", aggregate(subset)))

    print()
    print("=" * len(header))
    print("PER CATEGORY  (superseded-citation rate)")
    print("=" * len(header))
    categories = sorted({q.category for q in queries})
    print(f"{'Retriever':<12}" + "".join(f"{c:>16}" for c in categories))
    print("-" * (12 + 16 * len(categories)))
    for name, records in all_records.items():
        scores = aggregate(records)
        row = f"{name:<12}"
        for category in categories:
            sub = scores.per_category.get(category)
            row += f"{sub.superseded_citation_rate:>16.3f}" if sub else f"{'-':>16}"
        print(row)

    print()
    print("=" * len(header))
    print("ANSWERS CITING A DEAD PEP  (Dense rung)")
    print("=" * len(header))
    for record in all_records["Dense"]:
        if not record.cited_superseded:
            continue
        detail = ", ".join(
            f"PEP {n} [{by_number[n].status}]" for n in record.cited_superseded
        )
        print(f"{record.qid} [{record.category}{'/trap' if record.trap else ''}] -> {detail}")
        print(f"     {record.answer.strip().splitlines()[0][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
