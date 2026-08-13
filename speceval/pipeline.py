"""Assembles the full pipeline for ad-hoc questions.

**This module is not part of the study.** Nothing here is measured, and no driver imports it.
It exists so the finished system can be demonstrated on an arbitrary question rather than only
on the 51 gold queries.

It is deliberately a thin composition of the measured modules — it adds no retrieval logic, no
scoring, and no configuration that the evaluated pipeline does not already use. If it did, a
demo would show something other than what the numbers describe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, chunk_corpus
from .corpus import NON_AUTHORITATIVE, Pep, load_corpus
from .embed import embed_cached
from .generate import CachedGenerator, build_prompt, extract_citations
from .rerank import AuthorityReranker
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Query

# Chunks fed to the generator. Matches run_phase3/run_phase4 so a demo shows the same
# behaviour the measurements describe.
TOP_K_CHUNKS = 5

# Full strength. The sweep found the useful range is roughly 0.25-0.5, but 1.0 is the
# configuration the headline result quotes.
DEFAULT_STRENGTH = 1.0


@dataclass(frozen=True)
class Citation:
    number: int
    title: str
    status: str

    @property
    def is_authoritative(self) -> bool:
        return self.status not in NON_AUTHORITATIVE


@dataclass(frozen=True)
class Answer:
    """One answer, plus everything needed to show how it was reached."""

    question: str
    strength: float
    text: str
    retrieved: tuple[Citation, ...]
    cited: tuple[Citation, ...]
    elapsed_s: float

    @property
    def cited_non_authoritative(self) -> tuple[Citation, ...]:
        return tuple(c for c in self.cited if not c.is_authoritative)


@dataclass
class Pipeline:
    """Loads the corpus and models once, then answers questions."""

    corpus_root: Path | str = "peps/peps"
    cache_dir: Path | str = ".cache"
    peps: dict[int, Pep] = field(init=False)
    chunks: list[Chunk] = field(init=False)
    _hybrid: HybridRetriever = field(init=False)
    _generator: CachedGenerator = field(init=False)

    def __post_init__(self) -> None:
        peps = load_corpus(self.corpus_root)
        self.peps = {pep.number: pep for pep in peps}
        self.chunks = chunk_corpus(peps)
        vectors = embed_cached(
            [chunk.indexed_text for chunk in self.chunks],
            cache_dir=self.cache_dir,
            progress=False,
        )
        self._hybrid = HybridRetriever(
            lexical=BM25Retriever(self.chunks),
            dense=DenseRetriever(chunks=self.chunks, vectors=vectors),
            chunks=self.chunks,
        )
        self._generator = CachedGenerator(cache_dir=Path(self.cache_dir) / "answers")

    def _citation(self, number: int) -> Citation:
        pep = self.peps.get(number)
        if pep is None:
            return Citation(number=number, title="(not in corpus)", status="Unknown")
        return Citation(number=number, title=pep.title, status=pep.status)

    def ask(self, question: str, strength: float = DEFAULT_STRENGTH) -> Answer:
        """Answer one question. `strength` 0 disables authority reranking entirely."""
        if not question.strip():
            raise ValueError("question must not be empty")

        # relevant is empty because an ad-hoc question has no ground truth. Retrieval never
        # reads it; only the metrics do, and no metric runs here.
        query = Query(
            qid="adhoc", text=question, category="identifier", relevant=frozenset()
        )
        retriever = AuthorityReranker(
            base=self._hybrid, peps=self.peps, strength=strength, chunks=self.chunks
        )

        started = time.perf_counter()
        indices = retriever.search_chunks(query, depth=TOP_K_CHUNKS)
        prompt = build_prompt(question, [self.chunks[i] for i in indices])
        text = self._generator.generate(prompt)
        elapsed = time.perf_counter() - started

        # Distinct PEPs in the order their best chunk appeared.
        seen: list[int] = []
        for index in indices:
            number = self.chunks[index].pep_number
            if number not in seen:
                seen.append(number)

        return Answer(
            question=question,
            strength=strength,
            text=text.strip(),
            retrieved=tuple(self._citation(n) for n in seen),
            cited=tuple(
                self._citation(n)
                for n in extract_citations(text, known=set(self.peps))
            ),
            elapsed_s=elapsed,
        )

    def compare(self, question: str, strength: float = DEFAULT_STRENGTH) -> tuple[Answer, Answer]:
        """The same question without and with authority reranking.

        This is the demonstration that matters: the left side is what a conventional RAG
        pipeline returns, the right side is what the study's intervention returns.
        """
        return self.ask(question, strength=0.0), self.ask(question, strength=strength)
