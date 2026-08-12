#!/usr/bin/env python3
"""Phase 4 driver: sweep the reranker's strength and produce the tradeoff curve.

    ollama serve &
    .venv/bin/python run_phase4.py

Two sweeps:

* **Retrieval level** -- fine grid, free (no generation). Fine at the low end because the
  knob's transition is concentrated there; see the note in speceval/rerank.py.
* **Answer level** -- four strengths, since each costs 51 generations.

The baseline is `strength = 0` **of this same pipeline**, not Phase 3's hybrid row. The
reranker draws a deeper candidate pool than rung 3 did, and RRF over a deeper pool can promote
different chunks, so only the in-pipeline zero point is a clean control. Phase 3's number is
printed alongside for reference, with the difference attributable to pool depth.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from speceval.answer_metrics import aggregate, score_answer
from speceval.chunking import chunk_corpus
from speceval.corpus import NON_AUTHORITATIVE, load_corpus
from speceval.embed import EmbeddingError, embed_cached
from speceval.evaluate import K, evaluate
from speceval.generate import (
    CachedGenerator,
    GenerationError,
    build_prompt,
    extract_all_cited_numbers,
    extract_citations,
)
from speceval.rerank import AuthorityReranker
from speceval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    load_queries,
)

REPO_ROOT = Path(__file__).resolve().parent
TOP_K_CHUNKS = 5

RETRIEVAL_GRID = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
ANSWER_GRID = [0.0, 0.05, 0.25, 1.0]


def main() -> int:
    peps = load_corpus(REPO_ROOT / "peps" / "peps")
    by_number = {pep.number: pep for pep in peps}
    chunks = chunk_corpus(peps)
    queries = load_queries(REPO_ROOT / "eval" / "queries_gold.json")
    non_authoritative = {p.number for p in peps if p.status in NON_AUTHORITATIVE}

    try:
        vectors = embed_cached(
            [c.indexed_text for c in chunks], cache_dir=REPO_ROOT / ".cache", progress=False
        )
    except EmbeddingError as error:
        print(error, file=sys.stderr)
        return 1

    lexical = BM25Retriever(chunks)
    dense = DenseRetriever(chunks=chunks, vectors=vectors)
    hybrid = HybridRetriever(lexical=lexical, dense=dense, chunks=chunks)
    generator = CachedGenerator(cache_dir=REPO_ROOT / ".cache" / "answers")

    def reranker(strength: float, version_penalty: bool = False) -> AuthorityReranker:
        return AuthorityReranker(
            base=hybrid,
            peps=by_number,
            strength=strength,
            version_penalty=version_penalty,
            chunks=chunks,
            name=f"lambda={strength:g}" + ("+ver" if version_penalty else ""),
        )

    print(f"{len(peps)} PEPs, {len(chunks)} chunks, {len(queries)} queries\n")

    # ---------------------------------------------------------------- retrieval sweep
    print("=" * 66)
    print("RETRIEVAL-LEVEL SWEEP  (k=10, free -- no generation)")
    print("=" * 66)
    print(f"{'lambda':>8}{'Recall@10':>12}{'nDCG@10':>10}{'trap@1':>9}{'trap@1 (trap set)':>20}")
    print("-" * 66)
    traps = [q for q in queries if q.trap]
    for strength in RETRIEVAL_GRID:
        r = reranker(strength)
        overall = evaluate(r, queries, k=K, non_authoritative=non_authoritative)
        trap_only = evaluate(r, traps, k=K, non_authoritative=non_authoritative)
        print(
            f"{strength:>8g}{overall.overall.recall:>12.3f}{overall.overall.ndcg:>10.3f}"
            f"{overall.trap_at_1 or 0.0:>9.3f}{trap_only.trap_at_1 or 0.0:>20.3f}"
        )

    # ------------------------------------------------------------------ answer sweep
    def run_answers(retr, label: str):
        records = []
        for query in queries:
            indices = retr.search_chunks(query, depth=TOP_K_CHUNKS)
            prompt = build_prompt(query.text, [chunks[i] for i in indices])
            start = time.perf_counter()
            answer = generator.generate(prompt)
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
        print(f"  {label:<16} ({generator.hits} cached, {generator.misses} generated)")
        return records

    print()
    print("generating answers...")
    try:
        by_strength = {s: run_answers(reranker(s), f"lambda={s:g}") for s in ANSWER_GRID}
        ablation = run_answers(reranker(1.0, version_penalty=True), "lambda=1 +version")
    except GenerationError as error:
        print(error, file=sys.stderr)
        return 1

    header = f"{'lambda':>8}{'superseded':>12}{'authorit.':>11}{'version':>9}{'halluc.':>9}"
    print()
    print("=" * 66)
    print("ANSWER-LEVEL SWEEP  (51 queries)")
    print("=" * 66)
    print(header + f"{'trap: superseded':>19}")
    print("-" * 66)
    for strength, records in by_strength.items():
        overall = aggregate(records)
        trap_scores = aggregate([r for r in records if r.trap])
        version = (
            "-" if overall.version_correct_rate is None
            else f"{overall.version_correct_rate:.3f}"
        )
        print(
            f"{strength:>8g}{overall.superseded_citation_rate:>12.3f}"
            f"{overall.authoritative_citation_rate:>11.3f}{version:>9}"
            f"{overall.hallucinated_citation_rate:>9.3f}"
            f"{trap_scores.superseded_citation_rate:>19.3f}"
        )

    print()
    print("Phase 3 reference (rung 3, shallower pool): superseded 0.235, authoritative 0.765")
    print("The lambda=0 row above is the clean in-pipeline control; any gap from the Phase 3")
    print("row is pool depth, not reranking.")

    # -------------------------------------------------------------------- ablation
    print()
    print("=" * 66)
    print("ABLATION -- the naive version rule")
    print("=" * 66)
    print(f"{'variant':>20}{'superseded':>12}{'authorit.':>11}{'version':>9}")
    print("-" * 66)
    for label, records in (
        ("lambda=1", by_strength[1.0]),
        ("lambda=1 +version", ablation),
    ):
        scores = aggregate(records)
        version = (
            "-" if scores.version_correct_rate is None
            else f"{scores.version_correct_rate:.3f}"
        )
        print(
            f"{label:>20}{scores.superseded_citation_rate:>12.3f}"
            f"{scores.authoritative_citation_rate:>11.3f}{version:>9}"
        )
    print()
    print("+version penalises any PEP whose Python-Version postdates the asked version.")
    print("Prediction under test: this HURTS, because answering 'no, it arrived in 3.8'")
    print("requires retrieving the very PEP the rule demotes.")

    # ------------------------------------------------------------------ what changed
    print()
    print("=" * 66)
    print("QUERIES FIXED AND BROKEN  (lambda=0 -> lambda=1)")
    print("=" * 66)
    base_records = {r.qid: r for r in by_strength[0.0]}
    full_records = {r.qid: r for r in by_strength[1.0]}
    fixed, broken = [], []
    for qid, before in base_records.items():
        after = full_records[qid]
        if before.cited_superseded and not after.cited_superseded:
            fixed.append((qid, before, after))
        elif not before.cited_superseded and after.cited_superseded:
            broken.append((qid, before, after))

    print(f"FIXED ({len(fixed)}) -- cited a dead PEP at lambda=0, clean at lambda=1")
    for qid, before, after in fixed:
        dead = ", ".join(f"PEP {n}" for n in before.cited_superseded)
        print(f"  {qid}{'/trap' if before.trap else '':<6} dropped {dead}")
    print(f"\nBROKEN ({len(broken)}) -- clean at lambda=0, cited a dead PEP at lambda=1")
    for qid, _before, after in broken:
        dead = ", ".join(f"PEP {n}" for n in after.cited_superseded)
        print(f"  {qid}{'/trap' if after.trap else '':<6} introduced {dead}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
