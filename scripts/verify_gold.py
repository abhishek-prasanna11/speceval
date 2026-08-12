#!/usr/bin/env python3
"""Verify every label in the gold set against the corpus itself.

Label quality is the ceiling on this entire study: a brilliant harness over labels nobody
checked is worth nothing, and label rot is invisible once the corpus updates. So every
claim the gold set makes is re-derived from the PEP headers here rather than trusted.

    .venv/bin/python scripts/verify_gold.py

Exits non-zero on any inconsistency, so it can gate a commit.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from speceval.answer_metrics import release_version as _release_version  # noqa: E402
from speceval.corpus import NON_AUTHORITATIVE, load_corpus  # noqa: E402

CATEGORIES = {"availability", "identifier", "rationale"}


def main() -> int:
    peps = {pep.number: pep for pep in load_corpus(REPO_ROOT / "peps" / "peps")}
    records = json.loads((REPO_ROOT / "eval" / "queries_gold.json").read_text())
    problems: list[str] = []

    seen_qids: set[str] = set()
    seen_texts: set[str] = set()

    for record in records:
        qid = record.get("qid", "<missing>")

        if qid in seen_qids:
            problems.append(f"{qid}: duplicate qid")
        seen_qids.add(qid)

        text = record.get("text", "").strip().lower()
        if not text:
            problems.append(f"{qid}: empty query text")
        if text in seen_texts:
            problems.append(f"{qid}: duplicate query text")
        seen_texts.add(text)

        if record.get("category") not in CATEGORIES:
            problems.append(f"{qid}: category {record.get('category')!r} not in {CATEGORIES}")

        relevant = record.get("relevant") or []
        if not relevant:
            problems.append(f"{qid}: no labels")

        for number in relevant:
            pep = peps.get(number)
            if pep is None:
                problems.append(f"{qid}: PEP {number} does not exist in the corpus")
                continue
            # A labelled-relevant PEP that is itself non-authoritative is allowed only when
            # the note says so deliberately (q30's no-authoritative-answer case, and
            # rationale queries where a Rejected PEP genuinely holds the reasoning).
            if pep.status in NON_AUTHORITATIVE and not record.get("trap"):
                problems.append(
                    f"{qid}: labels PEP {number} which is {pep.status}, but trap is not set"
                )

        # A query marked trap must have a real trap available: some non-authoritative PEP
        # that points at, or is pointed at by, one of the labelled answers.
        if record.get("trap"):
            labelled = set(relevant)
            has_trap = any(
                pep.status in NON_AUTHORITATIVE
                and (
                    (pep.superseded_by in labelled)
                    or bool(labelled & set(pep.replaces))
                    or pep.number in labelled
                )
                for pep in peps.values()
            )
            # q27 / q28 are the both-Final pairs: no non-authoritative PEP is involved, so
            # the trap is recency rather than status. Those declare it in the note.
            declares_hard_case = "both" in record.get("note", "").lower()
            if not has_trap and not declares_hard_case:
                problems.append(
                    f"{qid}: marked trap but no superseded/rejected PEP links to {sorted(labelled)}"
                )

        # asked_version means exactly one thing: a Python version named in the query text.
        # It must not be confused with the version the *answer* involves -- conflating the
        # two silently corrupts the version metric, since a trap query deliberately asks
        # about a version predating the feature.
        version = record.get("asked_version")
        if version:
            if version not in record.get("text", ""):
                problems.append(
                    f"{qid}: asked_version {version!r} does not appear in the query text"
                )
            # The version metric needs a concrete availability version to check against.
            if not any(
                _release_version(peps[n].python_version)
                for n in relevant
                if n in peps
            ):
                problems.append(
                    f"{qid}: version-scoped but no labelled PEP has a parseable "
                    f"Python-Version, so version correctness cannot be scored"
                )
        elif re.search(r"\b\d+\.\d+\b", record.get("text", "")):
            problems.append(
                f"{qid}: query text names a version but asked_version is unset"
            )

    categories = Counter(record.get("category") for record in records)
    traps = sum(1 for record in records if record.get("trap"))

    print(f"queries        {len(records)}")
    print(f"categories     " + ", ".join(f"{c}={n}" for c, n in sorted(categories.items())))
    print(f"trap cases     {traps} ({traps / len(records):.0%})")
    print(f"ordinary       {len(records) - traps}")
    print(f"PEPs referenced {len({n for r in records for n in r['relevant']})}")

    if problems:
        print(f"\nFAILED -- {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("\nOK -- every label verified against the corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
