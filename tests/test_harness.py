"""Validation of the measuring instrument.

This file exists because a broken metric is undetectable from the outside: the pipeline
still runs, the numbers still look plausible, and the ranking between strategies may even
survive. So the metrics are pinned to hand-computed values, and the whole harness is run
against two synthetic retrievers whose scores are known in advance.

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from speceval.evaluate import evaluate
from speceval.metrics import ndcg_at_k, recall_at_k
from speceval.retrievers import OracleRetriever, Query, RandomRetriever, load_queries

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_QUERIES = REPO_ROOT / "eval" / "queries_seed.json"


class TestRecall(unittest.TestCase):
    """Hand-computed values. A hardwired implementation fails these."""

    def test_partial_overlap(self) -> None:
        # One of two relevant items retrieved.
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"a", "d"}, 3), 0.5)

    def test_no_overlap_is_zero(self) -> None:
        # The case that catches a recall() hardwired to 1.0.
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"d"}, 3), 0.0)

    def test_perfect_overlap_is_one(self) -> None:
        self.assertAlmostEqual(recall_at_k(["a", "b"], {"a", "b"}, 2), 1.0)

    def test_k_truncates(self) -> None:
        # 'c' is relevant but sits outside the top 2.
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"c"}, 2), 0.0)

    def test_is_set_overlap_not_positional(self) -> None:
        # Reordering the top k must not change recall -- that is nDCG's job. Getting
        # this wrong scores a provably exact retriever at well under 1.0.
        forward = recall_at_k(["a", "b", "c"], {"a", "c"}, 3)
        reversed_ = recall_at_k(["c", "b", "a"], {"a", "c"}, 3)
        self.assertAlmostEqual(forward, reversed_)
        self.assertAlmostEqual(forward, 1.0)


class TestNdcg(unittest.TestCase):
    def test_hit_at_rank_one(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a"}, 3), 1.0)

    def test_hit_at_rank_two(self) -> None:
        # 1/log2(3) / 1.0
        self.assertAlmostEqual(ndcg_at_k(["b", "a", "c"], {"a"}, 3), 1 / math.log2(3))

    def test_two_hits_at_ranks_one_and_three(self) -> None:
        # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1/log2(2) + 1/log2(3)
        expected = 1.5 / (1.0 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a", "c"}, 3), expected)

    def test_no_hits_is_zero(self) -> None:
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"z"}, 2), 0.0)

    def test_rewards_ordering_unlike_recall(self) -> None:
        good = ndcg_at_k(["a", "z", "z"], {"a"}, 3)
        bad = ndcg_at_k(["z", "z", "a"], {"a"}, 3)
        self.assertGreater(good, bad)


class TestGuards(unittest.TestCase):
    def test_empty_relevant_set_raises(self) -> None:
        # Scoring an unlabelled query as 0.0 or 1.0 would hide the bug inside a mean.
        with self.assertRaises(ValueError):
            recall_at_k(["a"], set(), 10)
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], set(), 10)

    def test_bad_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(["a"], {"a"}, 0)


class TestHarnessAgainstSyntheticRetrievers(unittest.TestCase):
    """End-to-end: the harness must score known retrievers at their known values."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.queries = load_queries(SEED_QUERIES)

    def test_seed_set_is_well_formed(self) -> None:
        self.assertGreater(len(self.queries), 0)
        qids = [q.qid for q in self.queries]
        self.assertEqual(len(qids), len(set(qids)), "duplicate qid in seed set")
        for query in self.queries:
            self.assertTrue(query.relevant, f"{query.qid} has no labels")
            self.assertIn(query.category, {"availability", "rationale", "identifier"})

    def test_oracle_scores_perfectly(self) -> None:
        result = evaluate(OracleRetriever(), self.queries, k=10)
        self.assertAlmostEqual(result.overall.recall, 1.0)
        self.assertAlmostEqual(result.overall.ndcg, 1.0)

    def test_random_scores_near_chance(self) -> None:
        # ~10 draws from 734 PEPs: expected recall is on the order of 0.01.
        retriever = RandomRetriever(pep_numbers=list(range(1, 735)))
        result = evaluate(retriever, self.queries, k=10)
        self.assertLess(result.overall.recall, 0.10)
        self.assertLess(result.overall.ndcg, 0.10)

    def test_oracle_beats_random(self) -> None:
        # The single assertion the whole harness exists to be able to make.
        oracle = evaluate(OracleRetriever(), self.queries, k=10)
        random_ = evaluate(
            RandomRetriever(pep_numbers=list(range(1, 735))), self.queries, k=10
        )
        self.assertGreater(oracle.overall.recall, random_.overall.recall)

    def test_per_category_covers_every_query(self) -> None:
        result = evaluate(OracleRetriever(), self.queries, k=10)
        counted = sum(scores.n_queries for scores in result.per_category.values())
        self.assertEqual(counted, len(self.queries))


if __name__ == "__main__":
    unittest.main()
