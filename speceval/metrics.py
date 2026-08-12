"""Retrieval metrics.

Relevance is labelled at **PEP level**, not chunk level: a query's ground truth is the
set of PEP numbers that legitimately answer it. Retrievers return ranked chunks, which
are collapsed to ranked distinct PEPs before scoring. This is a deliberate choice --
labelling every relevant *chunk* by hand would cost several times more and would not
change which retrieval strategy wins.

Both functions raise on an empty relevant set. A query with no labels is a bug in the
golden set, and silently scoring it 1.0 (or 0.0) would hide that bug in an average.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence


def _validate(relevant: set[Hashable], k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant:
        raise ValueError("empty relevant set -- a query with no labels is a bug")


def recall_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Fraction of the relevant items that appear in the top k.

    Set overlap, never rank-by-rank: position within the top k does not matter here,
    only membership. (Ordering is what nDCG measures.)
    """
    _validate(relevant, k)
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance, log2 discount.

    A hit at rank 1 contributes 1/log2(2) = 1.0, at rank 2 it contributes
    1/log2(3) = 0.6309, and so on. The ideal ranking puts every relevant item first,
    which is what the denominator represents.
    """
    _validate(relevant, k)

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(ranked[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
