"""Tests for dense retrieval, rank fusion, and the embedding cache.

None of these require a running Ollama server: the embedder is stubbed and the vectors are
hand-written, so the retrieval and fusion logic is tested independently of the model. A
test that needed a live model would be testing the model, not this code.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from speceval.chunking import Chunk
from speceval.embed import cache_key, embed_cached, normalise
from speceval.retrievers import (
    DenseRetriever,
    HybridRetriever,
    Query,
    collapse_to_peps,
)


def make_chunks(pep_numbers: list[int]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"pep-{number:04d}#{i}",
            pep_number=number,
            pep_title=f"PEP {number}",
            section="Abstract",
            text=f"body of {number}",
        )
        for i, number in enumerate(pep_numbers)
    ]


@dataclass
class StubEmbedder:
    """Returns a fixed vector for any query. Records what it was asked to embed."""

    vector: np.ndarray
    model: str = "stub"
    calls: list[str] = field(default_factory=list)

    def embed(self, texts: list[str], prefix: str = "", progress: bool = False) -> np.ndarray:
        self.calls.extend(prefix + text for text in texts)
        return np.tile(self.vector, (len(texts), 1)).astype(np.float32)


@dataclass
class StubRetriever:
    """A retriever with a hardcoded chunk ranking, for testing fusion."""

    ranking: list[int]
    chunks: list[Chunk] = field(default_factory=list)
    name: str = "Stub"

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        return self.ranking[:depth]


QUERY = Query(qid="t1", text="anything", category="identifier", relevant=frozenset({1}))


class TestNormalise(unittest.TestCase):
    def test_rows_become_unit_length(self) -> None:
        matrix = normalise(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0], atol=1e-6)

    def test_zero_vector_does_not_produce_nan(self) -> None:
        # A zero row would divide by zero and poison every similarity computed with it.
        matrix = normalise(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
        self.assertFalse(np.isnan(matrix).any())

    def test_dot_product_of_normalised_rows_is_cosine(self) -> None:
        matrix = normalise(np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32))
        self.assertAlmostEqual(float(matrix[0] @ matrix[1]), 1 / np.sqrt(2), places=6)


class TestDenseRetriever(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = make_chunks([10, 20, 30])
        # Chunk 1 (PEP 20) points exactly where the stub query vector points.
        self.vectors = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
        )
        self.embedder = StubEmbedder(vector=np.array([0.0, 1.0], dtype=np.float32))

    def _retriever(self) -> DenseRetriever:
        return DenseRetriever(
            chunks=self.chunks, vectors=self.vectors, embedder=self.embedder
        )

    def test_ranks_by_cosine_similarity(self) -> None:
        ranked = self._retriever().search(QUERY, top_k=3)
        self.assertEqual(ranked, [20, 30, 10])

    def test_applies_the_query_prefix(self) -> None:
        # Asymmetric model: a query embedded without its prefix silently retrieves worse.
        self._retriever().search(QUERY, top_k=1)
        self.assertTrue(self.embedder.calls[0].startswith("search_query: "))

    def test_query_embedding_is_cached(self) -> None:
        retriever = self._retriever()
        retriever.search(QUERY, top_k=1)
        retriever.search(QUERY, top_k=1)
        self.assertEqual(len(self.embedder.calls), 1)

    def test_vector_count_mismatch_raises(self) -> None:
        # Silently mismatched vectors would misattribute every score to the wrong chunk.
        with self.assertRaises(ValueError):
            DenseRetriever(
                chunks=self.chunks,
                vectors=np.zeros((2, 2), dtype=np.float32),
                embedder=self.embedder,
            )

    def test_top_k_larger_than_corpus_is_safe(self) -> None:
        ranked = self._retriever().search(QUERY, top_k=50)
        self.assertEqual(len(ranked), 3)


class TestPlatformNumerics(unittest.TestCase):
    """Guards the suppressed FP warning in DenseRetriever.search_chunks.

    numpy 1.26.4 on Apple Accelerate emitted a spurious "divide by zero encountered in
    matmul" for matrices above ~64 rows (fixed by numpy 2.5.2 on the same BLAS). The
    retriever suppresses it for anyone still on 1.26.x, so this test exists to make sure
    the suppression can only ever hide a cosmetic flag and not a real numerical fault: if
    the fast path stops matching a float64 reference, this fails regardless of numpy
    version or BLAS backend.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(12345)
        self.matrix = normalise(rng.normal(size=(200, 64)).astype(np.float32))
        self.query = self.matrix[7].copy()

    def test_matmul_matches_float64_reference(self) -> None:
        with np.errstate(all="ignore"):
            fast = self.matrix @ self.query
        reference = (self.matrix.astype(np.float64) @ self.query.astype(np.float64))
        np.testing.assert_allclose(fast, reference, atol=1e-5)

    def test_ordering_is_unaffected(self) -> None:
        with np.errstate(all="ignore"):
            fast = self.matrix @ self.query
        reference = self.matrix.astype(np.float64) @ self.query.astype(np.float64)
        np.testing.assert_array_equal(np.argsort(-fast)[:20], np.argsort(-reference)[:20])

    def test_self_similarity_is_one(self) -> None:
        with np.errstate(all="ignore"):
            similarities = self.matrix @ self.query
        self.assertAlmostEqual(float(similarities[7]), 1.0, places=5)

    def test_no_nan_or_inf_produced(self) -> None:
        with np.errstate(all="ignore"):
            similarities = self.matrix @ self.query
        self.assertFalse(np.isnan(similarities).any())
        self.assertFalse(np.isinf(similarities).any())


class TestCollapseToPeps(unittest.TestCase):
    def test_deduplicates_keeping_best_rank(self) -> None:
        chunks = make_chunks([10, 10, 20, 10, 30])
        self.assertEqual(collapse_to_peps(chunks, [1, 0, 2, 4], top_k=3), [10, 20, 30])

    def test_respects_top_k(self) -> None:
        chunks = make_chunks([10, 20, 30])
        self.assertEqual(collapse_to_peps(chunks, [0, 1, 2], top_k=2), [10, 20])


class TestReciprocalRankFusion(unittest.TestCase):
    def test_agreement_beats_single_system_preference(self) -> None:
        # Chunk 0 is ranked first by both; chunk 2 is ranked first by only one.
        chunks = make_chunks([10, 20, 30])
        hybrid = HybridRetriever(
            lexical=StubRetriever(ranking=[0, 1, 2], chunks=chunks),
            dense=StubRetriever(ranking=[0, 2, 1], chunks=chunks),
            chunks=chunks,
        )
        self.assertEqual(hybrid.search(QUERY, top_k=1), [10])

    def test_scores_match_the_formula(self) -> None:
        chunks = make_chunks([10, 20])
        hybrid = HybridRetriever(
            lexical=StubRetriever(ranking=[0, 1], chunks=chunks),
            dense=StubRetriever(ranking=[1, 0], chunks=chunks),
            chunks=chunks,
            rrf_k=60,
        )
        # Both chunks appear at ranks 1 and 2, so both score 1/61 + 1/62 and the tie is
        # broken by chunk index.
        self.assertEqual(hybrid.search_chunks(QUERY, depth=2), [0, 1])

    def test_a_chunk_only_one_system_found_is_still_included(self) -> None:
        # Fusion must union the candidate sets, not intersect them.
        chunks = make_chunks([10, 20, 30])
        hybrid = HybridRetriever(
            lexical=StubRetriever(ranking=[0], chunks=chunks),
            dense=StubRetriever(ranking=[2], chunks=chunks),
            chunks=chunks,
        )
        self.assertEqual(set(hybrid.search(QUERY, top_k=2)), {10, 30})

    def test_smaller_rrf_k_sharpens_top_rank_influence(self) -> None:
        chunks = make_chunks([10, 20])
        stubs = dict(
            lexical=StubRetriever(ranking=[0, 1], chunks=chunks),
            dense=StubRetriever(ranking=[1, 0], chunks=chunks),
            chunks=chunks,
        )
        sharp = HybridRetriever(**stubs, rrf_k=1)
        flat = HybridRetriever(**stubs, rrf_k=1000)
        # With k=1 the rank-1 slot is worth much more relative to rank 2 than with k=1000.
        self.assertGreater(1 / (1 + 1) - 1 / (1 + 2), 1 / (1000 + 1) - 1 / (1000 + 2))
        self.assertEqual(len(sharp.search(QUERY, top_k=2)), 2)
        self.assertEqual(len(flat.search(QUERY, top_k=2)), 2)


class TestEmbeddingCache(unittest.TestCase):
    def test_key_changes_with_text_model_and_prefix(self) -> None:
        base = cache_key(["a", "b"], "m", "p")
        self.assertNotEqual(base, cache_key(["a", "c"], "m", "p"))
        self.assertNotEqual(base, cache_key(["a", "b"], "other", "p"))
        self.assertNotEqual(base, cache_key(["a", "b"], "m", "other"))

    def test_key_is_stable(self) -> None:
        self.assertEqual(cache_key(["a"], "m", "p"), cache_key(["a"], "m", "p"))

    def test_second_call_hits_the_cache(self) -> None:
        embedder = StubEmbedder(vector=np.array([1.0, 0.0], dtype=np.float32))
        with TemporaryDirectory() as tmp:
            first = embed_cached(["x", "y"], embedder, cache_dir=tmp, progress=False)
            self.assertEqual(len(embedder.calls), 2)
            second = embed_cached(["x", "y"], embedder, cache_dir=tmp, progress=False)
            self.assertEqual(len(embedder.calls), 2, "cache was not used")
            np.testing.assert_array_equal(first, second)

    def test_changed_text_rebuilds(self) -> None:
        embedder = StubEmbedder(vector=np.array([1.0, 0.0], dtype=np.float32))
        with TemporaryDirectory() as tmp:
            embed_cached(["x"], embedder, cache_dir=tmp, progress=False)
            embed_cached(["z"], embedder, cache_dir=tmp, progress=False)
            self.assertEqual(len(embedder.calls), 2)

    def test_writes_a_metadata_sidecar(self) -> None:
        embedder = StubEmbedder(vector=np.array([1.0, 0.0], dtype=np.float32))
        with TemporaryDirectory() as tmp:
            embed_cached(["x"], embedder, cache_dir=tmp, progress=False)
            self.assertTrue(list(Path(tmp).glob("embeddings-*.json")))


if __name__ == "__main__":
    unittest.main()
