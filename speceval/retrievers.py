"""Retrievers, plus the two synthetic ones that validate the harness.

Every retriever answers the same question -- "given this query, which PEPs should be
consulted, best first?" -- so the evaluation loop is identical for all of them. Later
phases add DenseRetriever, HybridRetriever and the authority-aware reranker behind this
same interface.

``OracleRetriever`` and ``RandomRetriever`` exist to test the *measuring instrument*.
A metric implementation that cannot score the oracle at 1.0 and the random retriever at
roughly chance is broken -- and nothing else in the pipeline would reveal it, because a
broken metric still produces plausible-looking numbers.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .bm25 import BM25
from .chunking import Chunk


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    category: str
    relevant: frozenset[int]
    python_version: str | None = None
    note: str = ""


def load_queries(path: Path | str) -> list[Query]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Query(
            qid=record["qid"],
            text=record["text"],
            category=record["category"],
            relevant=frozenset(record["relevant"]),
            python_version=record.get("python_version"),
            note=record.get("note", ""),
        )
        for record in records
    ]


class Retriever(Protocol):
    name: str

    def search(self, query: Query, top_k: int) -> list[int]:
        """Return ranked, distinct PEP numbers, best first."""
        ...


@dataclass
class BM25Retriever:
    """Rung 1: lexical retrieval over chunks, collapsed to a PEP ranking.

    A PEP inherits the rank of its best-scoring chunk. Collapsing after ranking (rather
    than concatenating each PEP into one document) keeps long PEPs from being penalised
    by length normalisation.
    """

    chunks: list[Chunk]
    name: str = "BM25"
    # Chunks are retrieved deeper than the PEP cutoff because several chunks of the same
    # PEP routinely occupy the top positions; without this, top_k PEPs is unreachable.
    chunk_depth_multiplier: int = 10
    index: BM25 = field(init=False)

    def __post_init__(self) -> None:
        self.index = BM25([chunk.indexed_text for chunk in self.chunks])

    def search(self, query: Query, top_k: int) -> list[int]:
        hits = self.index.search(query.text, top_k=top_k * self.chunk_depth_multiplier)
        ranked: list[int] = []
        seen: set[int] = set()
        for chunk_index, _score in hits:
            pep_number = self.chunks[chunk_index].pep_number
            if pep_number not in seen:
                seen.add(pep_number)
                ranked.append(pep_number)
        return ranked[:top_k]


@dataclass
class OracleRetriever:
    """Returns exactly the ground truth. Must score 1.0 on every metric."""

    name: str = "Oracle"

    def search(self, query: Query, top_k: int) -> list[int]:
        return sorted(query.relevant)[:top_k]


@dataclass
class RandomRetriever:
    """Returns random PEPs. Must score near chance -- seeded, so it is reproducible."""

    pep_numbers: list[int]
    seed: int = 12345
    name: str = "Random"

    def search(self, query: Query, top_k: int) -> list[int]:
        # Seeded per query, so the result is stable across runs and across retrievers.
        rng = random.Random(f"{self.seed}:{query.qid}")
        return rng.sample(self.pep_numbers, min(top_k, len(self.pep_numbers)))
