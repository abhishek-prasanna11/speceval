"""Rung 4 -- version- and authority-aware reranking.

Reranks the hybrid candidate pool by what the PEP headers say about a document's standing,
with a **tunable strength** so the output is a tradeoff curve rather than two points:

    final = base_rank_score * (1 - strength + strength * authority_weight)

At `strength = 0` this is exactly rung 3 -- an invariant worth having, and one the tests
pin. At `strength = 1` ranking is fully authority-weighted.

**The knob is very non-linear, and the sweep grid has to account for it.** RRF base scores at
adjacent ranks are 1/61 and 1/62 -- 1.6% apart -- while authority weights span 0.02 to 1.0.
So flipping an *adjacent* pair takes only `strength ~= 0.0165`, while promoting a document
from 20 ranks deeper takes `~0.25`. Almost all of the interesting transition sits below 0.3,
and a uniform 0.25-step grid would miss it and make the knob look like a switch.

## Two signals, not one

**Status**, graded rather than boolean. The 9-value enum carries real gradations: an `Active`
process PEP is as authoritative as a `Final` one, a `Draft` is weaker than both without being
dead, and `Rejected` is worse than `Deferred` (which may yet revive).

**The supersession edge, independently of status.** This is the signal that makes the corpus
worth studying: PEP 333 (WSGI 1.0) is `Final` *and* carries `Superseded-By: 3333`, and PEP 409
is `Final` *and* superseded by 415. Status alone cannot separate either pair -- both members
read `Final`. So a PEP whose `Superseded-By` points at a live document is down-weighted no
matter how healthy its own status looks. This also handles the multi-hop packaging chain
(241 -> 314 -> 345 -> 566), where every intermediate hop is itself superseded.

## Why version is deliberately NOT a ranking signal

The obvious rule -- penalise a PEP whose `Python-Version` is later than the version the query
asks about -- is wrong, and measurably so.

Asked *"can I use the walrus operator in Python 3.7?"*, the correct answer is "no, 3.8", and
producing it **requires retrieving PEP 572, whose version is 3.8**. Penalising it for being
later than 3.7 removes precisely the evidence needed to answer correctly. Version tells you
what the answer must *say*, not which document to trust.

The rule is implemented anyway, behind `version_penalty`, so the claim can be measured instead
of asserted. See `notes/phase4.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .answer_metrics import release_version
from .chunking import Chunk
from .corpus import Pep
from .retrievers import RRF_K, HybridRetriever, Query, collapse_to_peps

# Graded authority by status. Values are judgements, not measurements, and they are stated
# here in one place so they can be argued with rather than buried in a conditional.
STATUS_WEIGHT: dict[str, float] = {
    "Final": 1.00,
    "Active": 1.00,
    "Accepted": 0.90,
    "Provisional": 0.75,
    "Draft": 0.55,
    "Deferred": 0.25,  # dormant, not refused
    "Superseded": 0.15,
    "Rejected": 0.10,
    "Withdrawn": 0.10,
}
DEFAULT_STATUS_WEIGHT = 0.50  # unknown status, e.g. "April Fool!"

# Applied when a PEP's Superseded-By points at a document that exists, regardless of the
# PEP's own status. This is what separates the both-Final pairs.
SUPERSEDED_BY_FACTOR = 0.15

# Applied by the (deliberately wrong) version rule when a PEP postdates the asked version.
VERSION_PENALTY_FACTOR = 0.30

# The reranker only reorders what it is given, so it needs a pool deeper than the cutoff.
RERANK_POOL_MULTIPLIER = 10


def parse_release(version: str | None) -> tuple[int, ...] | None:
    release = release_version(version)
    if release is None:
        return None
    try:
        return tuple(int(part) for part in release.split("."))
    except ValueError:
        return None


def authority_weight(
    pep: Pep | None,
    query: Query | None = None,
    known: set[int] | None = None,
    version_penalty: bool = False,
) -> float:
    """Weight in [0, 1]: how much this PEP should be trusted as a current source."""
    if pep is None:
        return DEFAULT_STATUS_WEIGHT

    weight = STATUS_WEIGHT.get(pep.status, DEFAULT_STATUS_WEIGHT)

    # Superseded-By is checked independently of status: a Final PEP that points at a
    # successor is still the wrong thing to answer from.
    if pep.superseded_by is not None and (known is None or pep.superseded_by in known):
        weight *= SUPERSEDED_BY_FACTOR

    if version_penalty and query is not None and query.asked_version:
        asked = parse_release(query.asked_version)
        pep_release = parse_release(pep.python_version)
        if asked is not None and pep_release is not None and pep_release > asked:
            weight *= VERSION_PENALTY_FACTOR

    return weight


@dataclass
class AuthorityReranker:
    """Rung 4 -- hybrid, reranked by status and supersession edges."""

    base: HybridRetriever
    peps: dict[int, Pep]
    strength: float = 1.0
    version_penalty: bool = False
    chunks: list[Chunk] = field(default_factory=list)
    pool_multiplier: int = RERANK_POOL_MULTIPLIER
    name: str = "Hybrid+Auth"

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        self.chunks = self.chunks or self.base.chunks

    def _weight(self, chunk_index: int, query: Query) -> float:
        pep = self.peps.get(self.chunks[chunk_index].pep_number)
        return authority_weight(
            pep, query, known=set(self.peps), version_penalty=self.version_penalty
        )

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        # Retrieve a deeper pool, rerank it, then cut to depth. Reranking only the top
        # `depth` would leave almost nothing to reorder.
        pool = self.base.search_chunks(query, depth * self.pool_multiplier)
        scored: list[tuple[int, float]] = []
        for rank, chunk_index in enumerate(pool):
            base_score = 1.0 / (RRF_K + rank + 1)
            weight = self._weight(chunk_index, query)
            blended = 1.0 - self.strength + self.strength * weight
            scored.append((chunk_index, base_score * blended))
        # Ties broken by original pool order, so strength=0 reproduces rung 3 exactly.
        order = sorted(
            range(len(scored)), key=lambda i: (-scored[i][1], i)
        )
        return [scored[i][0] for i in order][:depth]

    def search(self, query: Query, top_k: int) -> list[int]:
        depth = top_k * self.base.chunk_depth_multiplier
        return collapse_to_peps(self.chunks, self.search_chunks(query, depth), top_k)
