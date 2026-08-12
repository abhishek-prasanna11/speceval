"""Tests for header parsing, chunking and BM25.

Header parsing is load-bearing: the authority signal, the version signal and the
supersession edges all come from it, so a silent parse failure would quietly disable the
entire experiment.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from speceval.bm25 import BM25, tokenize
from speceval.chunking import chunk_pep, split_sections
from speceval.corpus import NON_AUTHORITATIVE, load_pep

# Mirrors the real shape of pep-0634.rst: a wrapped Author field, an empty header, and
# an odd run of spaces after a colon -- all three occur in the corpus.
SAMPLE = """PEP: 634
Title: Structural Pattern Matching: Specification
Author: Brandt Bucher <brandt@python.org>,
        Guido van Rossum <guido@python.org>
BDFL-Delegate:
Status:            Final
Type: Standards Track
Created: 12-Sep-2020
Python-Version: 3.10
Replaces: 622

Abstract
========

This PEP provides the technical specification.

Syntax and Semantics
====================

First paragraph here.

Second paragraph here.
"""


class TestHeaderParsing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        path = Path(self.tmp.name) / "pep-0634.rst"
        path.write_text(SAMPLE, encoding="utf-8")
        self.pep = load_pep(path)
        assert self.pep is not None

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scalar_fields(self) -> None:
        self.assertEqual(self.pep.number, 634)
        self.assertEqual(self.pep.pep_type, "Standards Track")
        self.assertEqual(self.pep.python_version, "3.10")

    def test_title_keeps_internal_colon(self) -> None:
        # Splitting on every colon would truncate this to "Structural Pattern Matching".
        self.assertEqual(self.pep.title, "Structural Pattern Matching: Specification")

    def test_extra_whitespace_after_colon_is_stripped(self) -> None:
        self.assertEqual(self.pep.status, "Final")
        self.assertTrue(self.pep.is_authoritative)

    def test_wrapped_author_is_consumed_as_a_continuation(self) -> None:
        # The indented line must be folded into the Author header, not leak into the
        # body and not be misread as a header of its own.
        self.assertNotIn("Guido van Rossum", self.pep.body)
        self.assertNotIn("brandt@python.org", self.pep.body)

    def test_supersession_edges(self) -> None:
        self.assertEqual(self.pep.replaces, (622,))
        self.assertIsNone(self.pep.superseded_by)

    def test_body_excludes_headers(self) -> None:
        self.assertTrue(self.pep.body.lstrip().startswith("Abstract"))
        self.assertNotIn("BDFL-Delegate", self.pep.body)

    def test_non_authoritative_statuses(self) -> None:
        for status in ("Rejected", "Withdrawn", "Superseded", "Deferred"):
            self.assertIn(status, NON_AUTHORITATIVE)
        for status in ("Final", "Active", "Accepted"):
            self.assertNotIn(status, NON_AUTHORITATIVE)


class TestChunking(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        path = Path(self.tmp.name) / "pep-0634.rst"
        path.write_text(SAMPLE, encoding="utf-8")
        self.pep = load_pep(path)
        assert self.pep is not None

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sections_detected_by_adornment(self) -> None:
        titles = [title for title, _ in split_sections(self.pep.body) if title]
        self.assertEqual(titles, ["Abstract", "Syntax and Semantics"])

    def test_adornment_lines_are_not_content(self) -> None:
        for _, text in split_sections(self.pep.body):
            self.assertNotIn("====", text)

    def test_chunks_carry_provenance(self) -> None:
        chunks = chunk_pep(self.pep)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.pep_number, 634)
            self.assertTrue(chunk.chunk_id.startswith("pep-0634#"))

    def test_indexed_text_includes_title_and_section(self) -> None:
        chunk = chunk_pep(self.pep)[0]
        self.assertIn("Structural Pattern Matching", chunk.indexed_text)
        self.assertIn(chunk.section, chunk.indexed_text)

    def test_long_section_splits(self) -> None:
        chunks = chunk_pep(self.pep, max_chars=20)
        self.assertGreater(len(chunks), 2)


class TestBm25(unittest.TestCase):
    def test_tokenizer_keeps_dunder_identifiers(self) -> None:
        # The reason for a custom tokenizer: `__future__` must survive intact.
        self.assertIn("__future__", tokenize("from __future__ import annotations"))
        self.assertEqual(tokenize("Match/Case!"), ["match", "case"])

    def test_exact_term_wins(self) -> None:
        index = BM25(
            [
                "the walrus operator assigns within an expression",
                "dataclasses generate init methods",
                "coroutines use async and await",
            ]
        )
        hits = index.search("walrus operator", top_k=3)
        self.assertEqual(hits[0][0], 0)

    def test_unknown_term_returns_nothing(self) -> None:
        index = BM25(["alpha beta", "gamma delta"])
        self.assertEqual(index.search("zzzzz", top_k=5), [])

    def test_rare_term_outranks_common_term(self) -> None:
        # idf must make "walrus" worth more than "python", which is in every document.
        index = BM25(
            ["python python python", "python walrus", "python python", "python"]
        )
        hits = index.search("python walrus", top_k=4)
        self.assertEqual(hits[0][0], 1)

    def test_scores_are_descending(self) -> None:
        index = BM25(["a b c", "a b", "a"])
        scores = [score for _, score in index.search("a b c", top_k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
