"""Tests for the authority reranker.

The invariant that matters most: at strength 0 the reranker must reproduce rung 3 exactly.
Without it, every point on the tradeoff curve is measured against a baseline that silently
differs from the one Phases 2 and 3 reported.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from pathlib import Path

from speceval.chunking import Chunk
from speceval.corpus import Pep
from speceval.rerank import (
    STATUS_WEIGHT,
    AuthorityReranker,
    authority_weight,
    parse_release,
)
from speceval.retrievers import Query


def make_pep(
    number: int,
    status: str,
    version: str | None = None,
    superseded_by: int | None = None,
) -> Pep:
    return Pep(
        number=number,
        title=f"PEP {number}",
        status=status,
        pep_type="Standards Track",
        python_version=version,
        superseded_by=superseded_by,
        replaces=(),
        body="",
        path=Path("/dev/null"),
    )


PEPS = {
    634: make_pep(634, "Final", "3.10"),
    622: make_pep(622, "Superseded", "3.10", superseded_by=634),
    333: make_pep(333, "Final", None, superseded_by=3333),  # Final AND superseded
    3333: make_pep(3333, "Final"),
    601: make_pep(601, "Rejected", "3.8"),
    572: make_pep(572, "Final", "3.8"),
    748: make_pep(748, "Draft", "3.14"),
}

CHUNKS = [
    Chunk(f"pep-{n:04d}#0", n, f"PEP {n}", "Abstract", f"body {n}")
    for n in (634, 622, 333, 3333, 601, 572, 748)
]
INDEX_OF = {chunk.pep_number: i for i, chunk in enumerate(CHUNKS)}

QUERY = Query(qid="t", text="q", category="identifier", relevant=frozenset({634}))
VERSIONED = Query(
    qid="v", text="can I use it in Python 3.7", category="availability",
    relevant=frozenset({572}), asked_version="3.7",
)


@dataclass
class StubHybrid:
    """Stands in for HybridRetriever with a fixed pool ordering."""

    pool: list[int]
    chunks: list[Chunk] = field(default_factory=lambda: CHUNKS)
    chunk_depth_multiplier: int = 1
    name: str = "StubHybrid"

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        return self.pool[:depth]


class TestParseRelease(unittest.TestCase):
    def test_tuple_comparison_beats_string_comparison(self) -> None:
        # "3.9" > "3.10" as strings; the whole point of parsing to tuples.
        self.assertGreater(parse_release("3.10"), parse_release("3.9"))

    def test_unparseable_is_none(self) -> None:
        self.assertIsNone(parse_release("3.x"))
        self.assertIsNone(parse_release(None))


class TestAuthorityWeight(unittest.TestCase):
    def test_live_statuses_outrank_dead_ones(self) -> None:
        live = authority_weight(PEPS[634], known=set(PEPS))
        dead = authority_weight(PEPS[622], known=set(PEPS))
        self.assertGreater(live, dead)

    def test_status_weights_are_graded_not_boolean(self) -> None:
        self.assertGreater(STATUS_WEIGHT["Final"], STATUS_WEIGHT["Draft"])
        self.assertGreater(STATUS_WEIGHT["Draft"], STATUS_WEIGHT["Deferred"])
        self.assertGreater(STATUS_WEIGHT["Deferred"], STATUS_WEIGHT["Rejected"])

    def test_superseded_by_penalises_a_final_pep(self) -> None:
        # The both-Final case: PEP 333 is Final and superseded by 3333. Status alone cannot
        # separate them; the supersession edge must.
        self.assertLess(
            authority_weight(PEPS[333], known=set(PEPS)),
            authority_weight(PEPS[3333], known=set(PEPS)),
        )

    def test_superseded_by_pointing_outside_the_corpus_is_ignored(self) -> None:
        orphan = make_pep(1, "Final", superseded_by=99999)
        self.assertAlmostEqual(
            authority_weight(orphan, known=set(PEPS)), STATUS_WEIGHT["Final"]
        )

    def test_unknown_status_gets_a_middling_weight(self) -> None:
        weight = authority_weight(make_pep(401, "April Fool!"), known=set(PEPS))
        self.assertGreater(weight, STATUS_WEIGHT["Rejected"])
        self.assertLess(weight, STATUS_WEIGHT["Final"])

    def test_missing_pep_does_not_crash(self) -> None:
        self.assertGreater(authority_weight(None), 0.0)

    def test_version_penalty_is_off_by_default(self) -> None:
        # PEP 572 is 3.8 and the query asks about 3.7. By default it must not be penalised:
        # answering "no, 3.8" requires retrieving exactly this document.
        self.assertAlmostEqual(
            authority_weight(PEPS[572], VERSIONED, known=set(PEPS)),
            STATUS_WEIGHT["Final"],
        )

    def test_version_penalty_when_enabled_demotes_later_peps(self) -> None:
        penalised = authority_weight(
            PEPS[572], VERSIONED, known=set(PEPS), version_penalty=True
        )
        self.assertLess(penalised, STATUS_WEIGHT["Final"])

    def test_version_penalty_ignores_unscoped_queries(self) -> None:
        self.assertAlmostEqual(
            authority_weight(PEPS[572], QUERY, known=set(PEPS), version_penalty=True),
            STATUS_WEIGHT["Final"],
        )


class TestReranker(unittest.TestCase):
    def _reranker(self, pool: list[int], **kwargs) -> AuthorityReranker:
        return AuthorityReranker(
            base=StubHybrid(pool=pool),  # type: ignore[arg-type]
            peps=PEPS,
            chunks=CHUNKS,
            pool_multiplier=1,
            **kwargs,
        )

    def test_strength_zero_is_exactly_rung_three(self) -> None:
        # The invariant the whole curve rests on.
        pool = [INDEX_OF[622], INDEX_OF[634], INDEX_OF[601]]
        reranked = self._reranker(pool, strength=0.0).search_chunks(QUERY, depth=3)
        self.assertEqual(reranked, pool)

    def test_full_strength_demotes_a_superseded_pep(self) -> None:
        pool = [INDEX_OF[622], INDEX_OF[634]]
        reranked = self._reranker(pool, strength=1.0).search_chunks(QUERY, depth=2)
        self.assertEqual(reranked[0], INDEX_OF[634])

    def test_full_strength_separates_the_both_final_pair(self) -> None:
        pool = [INDEX_OF[333], INDEX_OF[3333]]
        reranked = self._reranker(pool, strength=1.0).search_chunks(QUERY, depth=2)
        self.assertEqual(reranked[0], INDEX_OF[3333])

    def test_adjacent_rank_flips_need_only_a_tiny_strength(self) -> None:
        # Characterises the knob's real sensitivity. RRF base scores at adjacent ranks are
        # 1/61 and 1/62 -- 1.6% apart -- so an authority gap flips an adjacent pair at
        # strength ~0.0165. Below that the original order survives. This is why the sweep
        # grid has to be fine at the low end; a linear 0.25 grid would miss the transition
        # entirely and make the knob look like a switch.
        pool = [INDEX_OF[622], INDEX_OF[634]]
        below = self._reranker(pool, strength=0.001).search_chunks(QUERY, depth=2)
        above = self._reranker(pool, strength=0.05).search_chunks(QUERY, depth=2)
        self.assertEqual(below[0], INDEX_OF[622], "below threshold: order should hold")
        self.assertEqual(above[0], INDEX_OF[634], "above threshold: order should flip")

    def test_reranking_is_a_permutation_of_the_pool(self) -> None:
        pool = [INDEX_OF[622], INDEX_OF[634], INDEX_OF[601], INDEX_OF[572]]
        reranked = self._reranker(pool, strength=1.0).search_chunks(QUERY, depth=4)
        self.assertEqual(sorted(reranked), sorted(pool))

    def test_depth_truncates_after_reranking_not_before(self) -> None:
        # 634 sits last in the pool; at full strength it must still surface into the top 1.
        # pool_multiplier=3 so that depth=1 still draws a 3-chunk pool -- with a multiplier
        # of 1 the reranker would only ever see one candidate and could not reorder at all.
        reranker = AuthorityReranker(
            base=StubHybrid(pool=[INDEX_OF[622], INDEX_OF[601], INDEX_OF[634]]),  # type: ignore[arg-type]
            peps=PEPS,
            chunks=CHUNKS,
            pool_multiplier=3,
            strength=1.0,
        )
        self.assertEqual(reranker.search_chunks(QUERY, depth=1), [INDEX_OF[634]])

    def test_search_collapses_to_peps(self) -> None:
        pool = [INDEX_OF[622], INDEX_OF[634]]
        reranker = self._reranker(pool, strength=1.0)
        self.assertEqual(reranker.search(QUERY, top_k=2), [634, 622])

    def test_invalid_strength_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._reranker([0], strength=1.5)
        with self.assertRaises(ValueError):
            self._reranker([0], strength=-0.1)


if __name__ == "__main__":
    unittest.main()
