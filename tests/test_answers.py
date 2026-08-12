"""Tests for citation extraction and the answer-level metrics.

No live model: generation is stubbed and the PEP metadata is hand-built, so the scoring
logic is tested independently of what any model happens to say.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from speceval.answer_metrics import (
    aggregate,
    mentions_version,
    release_version,
    score_answer,
)
from speceval.chunking import Chunk
from speceval.corpus import Pep
from speceval.generate import (
    CachedGenerator,
    build_context,
    build_prompt,
    extract_all_cited_numbers,
    extract_citations,
)


def make_pep(number: int, status: str, version: str | None = None) -> Pep:
    return Pep(
        number=number,
        title=f"PEP {number}",
        status=status,
        pep_type="Standards Track",
        python_version=version,
        superseded_by=None,
        replaces=(),
        body="",
        path=Path("/dev/null"),
    )


PEPS = {
    634: make_pep(634, "Final", "3.10"),
    622: make_pep(622, "Superseded", "3.10"),
    601: make_pep(601, "Rejected", "3.8"),
    572: make_pep(572, "Final", "3.8"),
    566: make_pep(566, "Final", "3.x"),
}


class TestCitationExtraction(unittest.TestCase):
    def test_common_forms(self) -> None:
        answer = "See PEP 634, PEP-622 and pep572."
        self.assertEqual(extract_all_cited_numbers(answer), [634, 622, 572])

    def test_deduplicates_preserving_order(self) -> None:
        self.assertEqual(extract_all_cited_numbers("PEP 634 and PEP 634 again"), [634])

    def test_ignores_bare_numbers_and_versions(self) -> None:
        # "3.10" and a naked "634" must not become citations.
        self.assertEqual(extract_all_cited_numbers("Python 3.10 changed 634 things"), [])

    def test_known_filter_drops_invented_numbers(self) -> None:
        answer = "See PEP 634 and PEP 9999."
        self.assertEqual(extract_citations(answer, known=set(PEPS)), [634])
        # ...but the raw extractor still sees it, which is how hallucination is measured.
        self.assertIn(9999, extract_all_cited_numbers(answer))

    def test_no_citations(self) -> None:
        self.assertEqual(extract_citations("I do not know.", known=set(PEPS)), [])


class TestPromptBuilding(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            Chunk("pep-0634#0", 634, "Structural Pattern Matching: Specification", "Abstract", "body one"),
            Chunk("pep-0622#0", 622, "Structural Pattern Matching", "Abstract", "body two"),
        ]

    def test_context_never_leaks_status(self) -> None:
        # The whole comparison depends on the generator being unable to route around a bad
        # retrieval. If status reached the prompt, that would be a different experiment.
        context = build_context(self.chunks)
        for status in ("Superseded", "Rejected", "Withdrawn", "Status"):
            self.assertNotIn(status, context)

    def test_context_labels_each_chunk_with_its_pep(self) -> None:
        context = build_context(self.chunks)
        self.assertIn("[PEP 634]", context)
        self.assertIn("[PEP 622]", context)

    def test_context_respects_the_character_budget(self) -> None:
        context = build_context(self.chunks, max_chars=60)
        self.assertIn("[PEP 634]", context)
        self.assertNotIn("[PEP 622]", context)

    def test_oversized_first_chunk_is_truncated_not_dropped(self) -> None:
        # An empty context would send the model to parametric memory and the numbers would
        # read as a retrieval failure rather than a prompt bug.
        context = build_context(self.chunks, max_chars=10)
        self.assertTrue(context)
        self.assertLessEqual(len(context), 10)

    def test_prompt_contains_question_and_citation_instruction(self) -> None:
        prompt = build_prompt("does Python have match", self.chunks)
        self.assertIn("does Python have match", prompt)
        self.assertIn('"PEP 634"', prompt)


class TestVersionHelpers(unittest.TestCase):
    def test_release_version_parses_headers(self) -> None:
        self.assertEqual(release_version("3.10"), "3.10")
        self.assertEqual(release_version("3.11, 3.12"), "3.11")
        self.assertEqual(release_version("2.2"), "2.2")

    def test_unparseable_headers_yield_none(self) -> None:
        # "3.x" must not be coerced into a release, or the metric invents ground truth.
        self.assertIsNone(release_version("3.x"))
        self.assertIsNone(release_version(None))
        self.assertIsNone(release_version(""))

    def test_mentions_version_is_exact(self) -> None:
        self.assertTrue(mentions_version("added in 3.10", "3.10"))
        self.assertFalse(mentions_version("added in 3.1", "3.10"))
        self.assertFalse(mentions_version("added in 3.100", "3.10"))


class TestScoring(unittest.TestCase):
    def _score(self, answer: str, relevant=frozenset({634}), asked=None):
        return score_answer(
            qid="q",
            category="availability",
            trap=True,
            answer=answer,
            citations=extract_citations(answer, known=set(PEPS)),
            all_cited=extract_all_cited_numbers(answer),
            relevant=set(relevant),
            asked_version=asked,
            peps=PEPS,
        )

    def test_detects_superseded_citation(self) -> None:
        record = self._score("Use PEP 622 for pattern matching.")
        self.assertEqual(record.cited_superseded, [622])
        self.assertFalse(record.cited_authoritative)

    def test_rejected_counts_as_superseded(self) -> None:
        self.assertEqual(self._score("See PEP 601.").cited_superseded, [601])

    def test_authoritative_citation(self) -> None:
        record = self._score("Use PEP 634.")
        self.assertEqual(record.cited_superseded, [])
        self.assertTrue(record.cited_authoritative)

    def test_both_can_be_true(self) -> None:
        # Citing the right PEP does not excuse also citing a dead one.
        record = self._score("PEP 622 was superseded by PEP 634.")
        self.assertTrue(record.cited_authoritative)
        self.assertEqual(record.cited_superseded, [622])

    def test_hallucinated_citation(self) -> None:
        self.assertEqual(self._score("See PEP 9999.").hallucinated, [9999])

    def test_version_not_scored_when_query_is_unscoped(self) -> None:
        self.assertIsNone(self._score("PEP 634 covers it.").version_correct)

    def test_version_correct_when_real_release_surfaced(self) -> None:
        record = self._score("No -- it arrived in 3.10.", asked="3.9")
        self.assertTrue(record.version_correct)

    def test_version_wrong_when_release_absent(self) -> None:
        record = self._score("Yes, you can use it in 3.9.", asked="3.9")
        self.assertFalse(record.version_correct)

    def test_version_unscorable_when_pep_has_no_release(self) -> None:
        # PEP 566's header is "3.x", so there is no release to check against.
        record = self._score("See PEP 566.", relevant=frozenset({566}), asked="3.7")
        self.assertIsNone(record.version_correct)


class TestAggregate(unittest.TestCase):
    def _records(self):
        good = score_answer("a", "availability", False, "PEP 634 in 3.10",
                            [634], [634], {634}, "3.9", PEPS)
        bad = score_answer("b", "rationale", True, "PEP 622 says so",
                           [622], [622], {634}, None, PEPS)
        return [good, bad]

    def test_rates(self) -> None:
        scores = aggregate(self._records())
        self.assertEqual(scores.n, 2)
        self.assertAlmostEqual(scores.superseded_citation_rate, 0.5)
        self.assertAlmostEqual(scores.authoritative_citation_rate, 0.5)
        self.assertAlmostEqual(scores.hallucinated_citation_rate, 0.0)

    def test_version_rate_uses_only_scoped_queries(self) -> None:
        scores = aggregate(self._records())
        self.assertEqual(scores.n_version_scoped, 1)
        self.assertAlmostEqual(scores.version_correct_rate, 1.0)

    def test_empty_is_safe(self) -> None:
        self.assertEqual(aggregate([]).n, 0)

    def test_per_category_partitions(self) -> None:
        scores = aggregate(self._records())
        self.assertEqual(
            sum(s.n for s in scores.per_category.values()), scores.n
        )


class TestGenerationCache(unittest.TestCase):
    class StubGenerator:
        model = "stub"
        calls = 0

        def generate(self, prompt: str) -> str:
            type(self).calls += 1
            return f"answer to {prompt}"

        def cache_key(self, prompt: str) -> str:
            return str(abs(hash(prompt)) % 10**8)

    def test_second_identical_prompt_is_cached(self) -> None:
        with TemporaryDirectory() as tmp:
            stub = self.StubGenerator()
            type(stub).calls = 0
            cached = CachedGenerator(generator=stub, cache_dir=tmp)  # type: ignore[arg-type]
            first = cached.generate("p")
            second = cached.generate("p")
            self.assertEqual(first, second)
            self.assertEqual(type(stub).calls, 1)
            self.assertEqual((cached.hits, cached.misses), (1, 1))


if __name__ == "__main__":
    unittest.main()
