"""The retrieval strategy ladder, plus the two synthetic retrievers that validate it.

    rung 1  BM25Retriever      lexical
    rung 2  DenseRetriever     embedding cosine
    rung 3  HybridRetriever    reciprocal rank fusion of 1 and 2
    rung 4  (Phase 4)          hybrid + version/authority reranking

Every retriever answers the same question -- "given this query, which PEPs should be
consulted, best first?" -- so the evaluation loop is identical for all of them.

Retrieval happens over *chunks*; the public `search` collapses chunks to a ranked list of
distinct PEPs. `search_chunks` is exposed separately because fusion has to happen at chunk
level: fusing two already-collapsed PEP lists would throw away the evidence about *how
many* chunks of a PEP each system liked.

`OracleRetriever` and `RandomRetriever` exist to test the *measuring instrument*. A metric
that cannot score the oracle at 1.0 and the random retriever at roughly chance is broken --
and nothing else in the pipeline would reveal it, because a broken metric still produces
plausible-looking numbers.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .bm25 import BM25
from .chunking import Chunk
from .embed import QUERY_PREFIX, OllamaEmbedder, normalise

# Conventional constant from the original reciprocal-rank-fusion paper. It damps the
# influence of top ranks: without it, rank 1 would dominate rank 2 by 2x.
RRF_K = 60


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    category: str
    relevant: frozenset[int]
    asked_version: str | None = None
    # True when the corpus contains a superseded/rejected predecessor that a naive
    # retriever is likely to surface instead of the answer. Reported as its own subset:
    # authority reranking must be measurable on the queries it cannot help, or its benefit
    # is being scored only where it was designed to win.
    trap: bool = False
    note: str = ""


def load_queries(path: Path | str) -> list[Query]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Query(
            qid=record["qid"],
            text=record["text"],
            category=record["category"],
            relevant=frozenset(record["relevant"]),
            asked_version=record.get("asked_version"),
            trap=bool(record.get("trap", False)),
            note=record.get("note", ""),
        )
        for record in records
    ]


class Retriever(Protocol):
    name: str

    def search(self, query: Query, top_k: int) -> list[int]:
        """Return ranked, distinct PEP numbers, best first."""
        ...


def collapse_to_peps(chunks: list[Chunk], chunk_indices: list[int], top_k: int) -> list[int]:
    """Map a ranked chunk list to ranked distinct PEPs.

    A PEP inherits the rank of its best-scoring chunk. Collapsing *after* ranking (rather
    than concatenating each PEP into one document) keeps BM25's length normalisation from
    punishing long PEPs for being thorough.
    """
    ranked: list[int] = []
    seen: set[int] = set()
    for chunk_index in chunk_indices:
        pep_number = chunks[chunk_index].pep_number
        if pep_number not in seen:
            seen.add(pep_number)
            ranked.append(pep_number)
            if len(ranked) == top_k:
                break
    return ranked


# Chunks are retrieved deeper than the PEP cutoff because several chunks of the same PEP
# routinely occupy the top positions; without this, top_k distinct PEPs is unreachable.
CHUNK_DEPTH_MULTIPLIER = 10


@dataclass
class BM25Retriever:
    """Rung 1 -- lexical retrieval."""

    chunks: list[Chunk]
    name: str = "BM25"
    chunk_depth_multiplier: int = CHUNK_DEPTH_MULTIPLIER
    index: BM25 = field(init=False)

    def __post_init__(self) -> None:
        self.index = BM25([chunk.indexed_text for chunk in self.chunks])

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        return [index for index, _score in self.index.search(query.text, top_k=depth)]

    def search(self, query: Query, top_k: int) -> list[int]:
        depth = top_k * self.chunk_depth_multiplier
        return collapse_to_peps(self.chunks, self.search_chunks(query, depth), top_k)


@dataclass
class DenseRetriever:
    """Rung 2 -- embedding cosine similarity, brute force.

    Brute force is the correct choice at this scale: one (19763, 768) by (768,) matrix
    product is a few milliseconds, and an ANN index would add approximation error to a
    study whose entire subject is measurement accuracy.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    embedder: OllamaEmbedder | None = None
    name: str = "Dense"
    chunk_depth_multiplier: int = CHUNK_DEPTH_MULTIPLIER
    _matrix: np.ndarray = field(init=False)
    _query_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.vectors.shape[0] != len(self.chunks):
            raise ValueError(
                f"{self.vectors.shape[0]} vectors for {len(self.chunks)} chunks"
            )
        self._matrix = normalise(self.vectors.astype(np.float32))
        self.embedder = self.embedder or OllamaEmbedder()

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a query, caching so repeated evaluation runs stay comparable.

        The cache is deliberately kept out of the timed region's first call: the reported
        latency includes the Ollama round trip, because that is what a real query costs.
        """
        if text not in self._query_cache:
            assert self.embedder is not None
            vector = self.embedder.embed([text], prefix=QUERY_PREFIX)
            self._query_cache[text] = normalise(vector)[0]
        return self._query_cache[text]

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        # errstate: numpy 1.26.4 built against Apple Accelerate raises a spurious
        # "divide by zero encountered in matmul" for matrices above ~64 rows -- the FP
        # status flags are set by the blocked BLAS path, not by the arithmetic. Verified
        # harmless at the time: results matched a float64 reference to 1e-7 with identical
        # ordering. Not reproducible on numpy 2.5.2 (same Accelerate BLAS), so it was an
        # upstream integration bug, but the guard stays for anyone running 1.26.x.
        # Suppressed narrowly here rather than globally, so a genuine numerical fault
        # elsewhere still surfaces; TestPlatformNumerics pins the arithmetic either way.
        with np.errstate(all="ignore"):
            similarities = self._matrix @ self._embed_query(query.text)
        depth = min(depth, similarities.shape[0])
        # argpartition finds the top `depth` without sorting all 19,763 scores, then only
        # those are sorted.
        candidates = np.argpartition(-similarities, depth - 1)[:depth]
        return candidates[np.argsort(-similarities[candidates])].tolist()

    def search(self, query: Query, top_k: int) -> list[int]:
        depth = top_k * self.chunk_depth_multiplier
        return collapse_to_peps(self.chunks, self.search_chunks(query, depth), top_k)


@dataclass
class HybridRetriever:
    """Rung 3 -- reciprocal rank fusion of BM25 and dense rankings.

    RRF rather than a weighted sum of scores, because BM25 scores are unbounded and
    corpus-dependent while cosine similarities live in [-1, 1]. Adding them requires
    calibrating one to the other, and any calibration constant would become an unexamined
    parameter sitting underneath every result. RRF uses only ranks, so it needs none::

        score(d) = SUM over systems s of  1 / (RRF_K + rank_s(d))
    """

    lexical: BM25Retriever
    dense: DenseRetriever
    chunks: list[Chunk] = field(default_factory=list)
    name: str = "Hybrid"
    chunk_depth_multiplier: int = CHUNK_DEPTH_MULTIPLIER
    rrf_k: int = RRF_K

    def __post_init__(self) -> None:
        self.chunks = self.chunks or self.lexical.chunks

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        fused: dict[int, float] = defaultdict(float)
        for retriever in (self.lexical, self.dense):
            for rank, chunk_index in enumerate(retriever.search_chunks(query, depth)):
                fused[chunk_index] += 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [chunk_index for chunk_index, _score in ranked[:depth]]

    def search(self, query: Query, top_k: int) -> list[int]:
        depth = top_k * self.chunk_depth_multiplier
        return collapse_to_peps(self.chunks, self.search_chunks(query, depth), top_k)


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
