#!/usr/bin/env python3
"""Ask the finished system a question.

    ollama serve &
    .venv/bin/python ask.py "how should I specify the build backend for a package"
    .venv/bin/python ask.py --compare "how do I postpone the evaluation of annotations"

`--compare` runs the same question twice, without and with authority reranking, which is the
clearest way to see what the study measured.

Not part of the study: nothing here is measured, and no driver imports it.
"""

from __future__ import annotations

import argparse
import sys

from speceval.pipeline import DEFAULT_STRENGTH, Answer, Pipeline

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def show(answer: Answer, colour: bool) -> None:
    def paint(text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if colour else text

    label = "no reranking" if answer.strength == 0 else f"reranking lambda={answer.strength:g}"
    print(paint(f"--- {label} ---", BOLD))

    print("retrieved:")
    for rank, citation in enumerate(answer.retrieved, 1):
        mark = "ok " if citation.is_authoritative else "DEAD"
        code = GREEN if citation.is_authoritative else RED
        print(
            f"  {rank}. {paint(mark, code)} PEP {citation.number:<5} "
            f"[{citation.status:<10}] {citation.title[:56]}"
        )

    print(f"\n{answer.text}\n")

    if answer.cited:
        parts = [
            paint(f"PEP {c.number} [{c.status}]", GREEN if c.is_authoritative else RED)
            for c in answer.cited
        ]
        print("cited: " + ", ".join(parts))
    else:
        print("cited: (none)")

    dead = answer.cited_non_authoritative
    if dead:
        print(
            paint(
                f"WARNING: cites {len(dead)} non-authoritative "
                f"{'PEP' if len(dead) == 1 else 'PEPs'}",
                RED,
            )
        )
    print(paint(f"({answer.elapsed_s:.1f}s)", DIM))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question", nargs="+", help="the question to ask")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run without and with authority reranking, side by side",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=DEFAULT_STRENGTH,
        help=f"reranking strength 0-1 (default {DEFAULT_STRENGTH:g}; 0 disables)",
    )
    parser.add_argument("--no-colour", action="store_true", help="disable ANSI colour")
    args = parser.parse_args()

    question = " ".join(args.question)
    colour = not args.no_colour and sys.stdout.isatty()

    print("loading corpus and vectors...", file=sys.stderr)
    pipeline = Pipeline()
    print(
        f"{len(pipeline.peps)} PEPs, {len(pipeline.chunks)} chunks ready\n", file=sys.stderr
    )
    print(f"Q: {question}\n")

    if args.compare:
        before, after = pipeline.compare(question, strength=args.strength)
        show(before, colour)
        print()
        show(after, colour)
    else:
        show(pipeline.ask(question, strength=args.strength), colour)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
