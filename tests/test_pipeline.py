"""Tests for the demo pipeline's presentation logic.

The pipeline itself is a thin composition of already-tested modules and needs a corpus and two
model servers to run, so it is not unit-tested here. What *is* tested is the logic this layer
adds on its own: authority classification, the non-authoritative filter, and the JSON
serialisation that reaches a browser -- including escaping, since answer text and PEP titles are
interpolated into a page.
"""

from __future__ import annotations

import unittest

from speceval.pipeline import Answer, Citation

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "serve", pathlib.Path(__file__).resolve().parent.parent / "serve.py"
)
serve = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(serve)


LIVE = Citation(number=649, title="Deferred Evaluation Of Annotations", status="Final")
DEAD = Citation(number=563, title="Postponed Evaluation of Annotations", status="Superseded")
REJECTED = Citation(number=601, title="Forbid return in finally", status="Rejected")
UNKNOWN = Citation(number=9999, title="(not in corpus)", status="Unknown")


def make_answer(cited: tuple[Citation, ...], text: str = "hello") -> Answer:
    return Answer(
        question="q", strength=1.0, text=text,
        retrieved=cited, cited=cited, elapsed_s=1.5,
    )


class TestCitation(unittest.TestCase):
    def test_live_statuses_are_authoritative(self) -> None:
        self.assertTrue(LIVE.is_authoritative)

    def test_dead_statuses_are_not(self) -> None:
        self.assertFalse(DEAD.is_authoritative)
        self.assertFalse(REJECTED.is_authoritative)

    def test_unknown_status_is_treated_as_authoritative(self) -> None:
        # Matches corpus.py: an unrecognised status is simply not in NON_AUTHORITATIVE.
        # Flagging it red would be a claim the corpus does not support.
        self.assertTrue(UNKNOWN.is_authoritative)


class TestAnswer(unittest.TestCase):
    def test_filters_non_authoritative_citations(self) -> None:
        answer = make_answer((LIVE, DEAD, REJECTED))
        self.assertEqual(
            [c.number for c in answer.cited_non_authoritative], [563, 601]
        )

    def test_clean_answer_has_none(self) -> None:
        self.assertEqual(make_answer((LIVE,)).cited_non_authoritative, ())


class TestSerialisation(unittest.TestCase):
    def test_shape(self) -> None:
        payload = serve.as_dict(make_answer((LIVE,)))
        self.assertEqual(payload["strength"], 1.0)
        self.assertEqual(payload["retrieved"][0]["number"], 649)
        self.assertTrue(payload["cited"][0]["authoritative"])

    def test_answer_text_is_escaped(self) -> None:
        # Answer text is interpolated into the page; a model emitting angle brackets must not
        # be able to inject markup.
        payload = serve.as_dict(make_answer((LIVE,), text="<script>alert(1)</script>"))
        self.assertNotIn("<script>", payload["text"])
        self.assertIn("&lt;script&gt;", payload["text"])

    def test_titles_are_escaped(self) -> None:
        nasty = Citation(number=1, title="<img src=x onerror=1>", status="Final")
        payload = serve.as_dict(make_answer((nasty,)))
        self.assertNotIn("<img", payload["retrieved"][0]["title"])

    def test_is_json_serialisable(self) -> None:
        import json
        json.dumps(serve.as_dict(make_answer((LIVE, DEAD))))


if __name__ == "__main__":
    unittest.main()
