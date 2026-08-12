"""Answer-level metrics, computed automatically from PEP headers.

No LLM judge and no human grading on this path. That single constraint is what keeps the
project small, and it is possible only because the corpus carries machine-readable authority
(`Status`) and version (`Python-Version`) metadata.

What is measured:

- **superseded-citation rate** -- did the answer cite a Rejected / Withdrawn / Superseded /
  Deferred PEP? The headline failure mode.
- **authoritative-citation rate** -- did it cite a PEP the gold set labels as an answer?
- **version-correct rate** -- on version-scoped queries only, did the answer surface the
  release in which the feature actually landed? Deliberately narrow: it checks that the right
  version *appears*, not that the surrounding claim is true. Full version correctness needs
  the deferred judge, and the docs say so rather than implying otherwise.
- **hallucinated-citation rate** -- did it cite a PEP number absent from the corpus? A
  grounding check, not an authority one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .corpus import NON_AUTHORITATIVE, Pep

_RELEASE_RE = re.compile(r"\b(\d+\.\d+)\b")


def release_version(python_version: str | None) -> str | None:
    """First concrete `major.minor` in a Python-Version header.

    Headers are not uniform: "3.10", "3.11, 3.12" and "3.x" all occur. "3.x" yields None,
    which excludes that PEP from version scoring rather than inventing a release for it.
    """
    if not python_version:
        return None
    match = _RELEASE_RE.search(python_version)
    return match.group(1) if match else None


def mentions_version(answer: str, version: str) -> bool:
    return version in set(_RELEASE_RE.findall(answer))


@dataclass
class AnswerRecord:
    """One generated answer plus everything derivable about it."""

    qid: str
    category: str
    trap: bool
    answer: str
    citations: list[int]
    hallucinated: list[int]
    cited_superseded: list[int]
    cited_authoritative: bool
    version_correct: bool | None  # None when the query is not version-scoped
    latency_ms: float = 0.0


@dataclass
class AnswerScores:
    n: int
    superseded_citation_rate: float
    authoritative_citation_rate: float
    hallucinated_citation_rate: float
    version_correct_rate: float | None
    n_version_scoped: int
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    per_category: dict[str, "AnswerScores"] = field(default_factory=dict)


def score_answer(
    qid: str,
    category: str,
    trap: bool,
    answer: str,
    citations: list[int],
    all_cited: list[int],
    relevant: set[int],
    asked_version: str | None,
    peps: dict[int, Pep],
    latency_ms: float = 0.0,
) -> AnswerRecord:
    cited_superseded = [
        number
        for number in citations
        if number in peps and peps[number].status in NON_AUTHORITATIVE
    ]
    hallucinated = [number for number in all_cited if number not in peps]

    version_correct: bool | None = None
    if asked_version:
        # The release in which the feature actually landed, taken from the labelled PEPs.
        expected = next(
            (
                version
                for number in sorted(relevant)
                if number in peps and (version := release_version(peps[number].python_version))
            ),
            None,
        )
        if expected is not None:
            version_correct = mentions_version(answer, expected)

    return AnswerRecord(
        qid=qid,
        category=category,
        trap=trap,
        answer=answer,
        citations=citations,
        hallucinated=hallucinated,
        cited_superseded=cited_superseded,
        cited_authoritative=bool(set(citations) & relevant),
        version_correct=version_correct,
        latency_ms=latency_ms,
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def aggregate(records: Sequence[AnswerRecord], by_category: bool = True) -> AnswerScores:
    if not records:
        return AnswerScores(0, 0.0, 0.0, 0.0, None, 0)

    n = len(records)
    versioned = [r for r in records if r.version_correct is not None]
    latencies = sorted(r.latency_ms for r in records)

    scores = AnswerScores(
        n=n,
        superseded_citation_rate=sum(1 for r in records if r.cited_superseded) / n,
        authoritative_citation_rate=sum(1 for r in records if r.cited_authoritative) / n,
        hallucinated_citation_rate=sum(1 for r in records if r.hallucinated) / n,
        version_correct_rate=(
            sum(1 for r in versioned if r.version_correct) / len(versioned)
            if versioned
            else None
        ),
        n_version_scoped=len(versioned),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
    )

    if by_category:
        categories = sorted({r.category for r in records})
        scores.per_category = {
            category: aggregate(
                [r for r in records if r.category == category], by_category=False
            )
            for category in categories
        }
    return scores
