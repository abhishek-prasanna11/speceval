# Chapter 20 — The Reranker

`speceval/rerank.py` (156 lines)

## Learning Objectives

- Explain the blending formula and why `strength = 0` must reproduce rung 3 exactly.
- Explain the two authority signals and why the supersession edge is checked independently of status.
- Derive the strength thresholds from RRF arithmetic, before seeing any measurement.
- Explain the rerank pool and why reranking the output would achieve nothing.
- Explain why the version rule is implemented behind a flag, defaulted off.

## Motivation

This is rung 4 — the intervention. Everything before it measured a problem; this module attempts to fix
it.

It is also the module with the most opportunity to fool yourself. An intervention evaluated against the
wrong baseline, with a parameter swept on the wrong grid, will produce a confident number that means
nothing. Most of this chapter is about the decisions that prevent that.

## The formula

```
final = base_rank_score * (1 - strength + strength * authority_weight)
```

`authority_weight` lies in `[0, 1]`; `strength` (written λ throughout) lies in `[0, 1]`.

Read the bracket at the two extremes:

- **λ = 0** → `1 - 0 + 0 = 1`. Every score is multiplied by 1. The ranking is untouched: **this is
  exactly rung 3**.
- **λ = 1** → `1 - 1 + weight = weight`. The ranking is fully authority-weighted.
- **λ = 0.25** → `0.75 + 0.25 × weight`. A document with weight 0.0225 keeps 75.6% of its base score; one
  with weight 1.0 keeps all of it.

The λ = 0 identity is not a nicety. It is the property that makes the sweep interpretable — the curve's
zero point *is* the baseline, computed by the same code path, rather than a separate system that might
differ for unrelated reasons. It is pinned:

```python
    def test_strength_zero_is_exactly_rung_three(self) -> None:
        # The invariant the whole curve rests on.
        pool = [INDEX_OF[622], INDEX_OF[634], INDEX_OF[601]]
        reranked = self._reranker(pool, strength=0.0).search_chunks(QUERY, depth=3)
        self.assertEqual(reranked, pool)
```

Achieving it requires care in the sort, because at λ = 0 every document's multiplier is identical and
the ordering must fall back to the pool's:

```python
        # Ties broken by original pool order, so strength=0 reproduces rung 3 exactly.
        order = sorted(
            range(len(scored)), key=lambda i: (-scored[i][1], i)
        )
```

Sorting indices rather than values, with the index as tiebreak, preserves input order among equals.

## Deep Explanation: the two authority signals

```python
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
    ...
```

### Signal 1: graded status

```python
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
```

Graded rather than binary. `Active` (a process PEP still in force) is as authoritative as `Final`. A
`Draft` is weaker than both without being dead. `Deferred` — dormant, might revive — outranks `Rejected`
— considered and refused.

The module is explicit that these are judgements:

```python
# Graded authority by status. Values are judgements, not measurements, and they are stated
# here in one place so they can be argued with rather than buried in a conditional.
```

This is the study's largest untested assumption, and chapter 23 says so. No sensitivity analysis was run
on these constants, so an unknown share of the measured effect may be attributable to the specific
numbers rather than to the idea. Stating that in the code, at the definition, is the minimum honest
treatment.

`DEFAULT_STATUS_WEIGHT = 0.50` handles unknown statuses — including the corpus's real `April Fool!` —
by placing them mid-scale rather than crashing or trusting them fully. Chapter 14's decision to keep
`status` as a string is what makes this graceful.

### Signal 2: the supersession edge, checked independently

This is the signal that justified the corpus choice (chapter 10). PEP 333 is `Final` **and** carries
`Superseded-By: 3333`. Status alone cannot separate the pair — both read `Final`.

So a PEP whose successor exists is penalised regardless of its own status:

```python
SUPERSEDED_BY_FACTOR = 0.15
```

The `known` check matters:

```python
    if pep.superseded_by is not None and (known is None or pep.superseded_by in known):
```

If `Superseded-By` points at a PEP not in the corpus — a parse error, a truncated corpus, a reference to
something unpublished — the penalty is skipped. Penalising a document for having a successor that does
not exist would be acting on data you cannot verify. Pinned:

```python
def test_superseded_by_pointing_outside_the_corpus_is_ignored(self) -> None:
    orphan = make_pep(1, "Final", superseded_by=99999)
    self.assertAlmostEqual(
        authority_weight(orphan, known=set(PEPS)), STATUS_WEIGHT["Final"]
    )
```

Multiplying rather than replacing means the two signals compose. A `Superseded` PEP that *also* points
at a successor gets `0.15 × 0.15 = 0.0225` — the lowest weight in the system, which is correct: it is
both marked dead and has a named replacement.

## Deep Explanation: the knob is violently non-linear

Here is the analysis that determined the sweep grid, worked out from arithmetic *before* any
measurement.

Chapter 6 established that RRF base scores at adjacent ranks differ by only 1.6%:

```
   rank 1 -> 1/61 = 0.016393
   rank 2 -> 1/62 = 0.016129        1.6% lower
```

Meanwhile authority weights span 0.0225 to 1.0 — a factor of 44.

So consider a superseded document at rank 1 and a live one at rank 2. The superseded one loses its
position when:

```
   base₁ · (1 - λ + λ·0.0225)  <  base₂ · (1 - λ + λ·1.0)

   (1 - 0.9775λ)  <  61/62 = 0.9839

   λ  >  0.0165
```

**An adjacent pair flips at λ ≈ 0.0165.** Almost nothing.

Now the same document twenty ranks deeper. `base₂₁/base₁ = 61/81 = 0.753`:

```
   1 - 0.9775λ  <  0.753     →     λ > 0.253
```

**Promoting from twenty ranks deeper needs λ ≈ 0.25.**

So the knob's entire useful range sits below about 0.3, and it is *not* linear: tiny values reorder
neighbours, moderate values reach into the pool. Recorded in the module:

```python
"""**The knob is very non-linear, and the sweep grid has to account for it.** RRF base scores at
adjacent ranks are 1/61 and 1/62 -- 1.6% apart -- while authority weights span 0.02 to 1.0.
So flipping an *adjacent* pair takes only `strength ~= 0.0165`, while promoting a document
from 20 ranks deeper takes `~0.25`. Almost all of the interesting transition sits below 0.3,
and a uniform 0.25-step grid would miss it and make the knob look like a switch.
"""
```

That prediction was made before the sweep and then confirmed by it. The measured retrieval curve:

```
  lambda   Recall@10   nDCG@10   trap@1   trap@1 (trap set)
------------------------------------------------------------
       0       0.951     0.771    0.294               0.550
   0.005       0.951     0.771    0.294               0.550
    0.01       0.951     0.771    0.294               0.550     <- nothing yet
    0.02       0.951     0.803    0.137               0.350     <- transition begins
    0.05       0.971     0.813    0.137               0.350
     0.1       0.971     0.831    0.059               0.150
    0.15       0.971     0.841    0.039               0.100
    0.25       0.980     0.856    0.020               0.050
     0.5       0.980     0.852    0.000               0.000
```

Nothing changes at 0.005 or 0.01, both below the 0.0165 threshold. Everything begins at 0.02, just
above it.

**A uniform grid of 0, 0.25, 0.5, 0.75, 1.0 would have shown four nearly identical rows** and concluded
the parameter was a switch. The fine low-end grid is why the shape is visible at all — and why chapter
23's second finding exists.

This is also why the two `test_adjacent_rank_flips_need_only_a_tiny_strength` assertions bracket the
threshold at 0.001 and 0.05 rather than using a "reasonable" middle value. An earlier version of that
test assumed 0.05 would be gentle; it is not, and the test failed. The failure was the discovery.

## Deep Explanation: the rerank pool

```python
# The reranker only reorders what it is given, so it needs a pool deeper than the cutoff.
RERANK_POOL_MULTIPLIER = 10

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
        order = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
        return [scored[i][0] for i in order][:depth]
```

**Retrieve deep, rerank, then cut.** Asked for 5 chunks, it fetches 50, reranks all 50, returns the best
5. If it reranked only the top 5, a live document sitting at rank 6 could never be promoted — and that
is exactly the case the intervention exists to fix.

Note the base score is *recomputed* from the pool position (`1.0 / (RRF_K + rank + 1)`) rather than
carried through from the hybrid retriever. The reranker only receives an ordering, not scores, so it
reconstructs the RRF curve from rank. This keeps the interface narrow — `search_chunks` returns
`list[int]` — at the cost of assuming the base ranker used RRF. That assumption is documented by the
type: the field is `base: HybridRetriever`, not `base: Retriever`.

Pinned:

```python
def test_depth_truncates_after_reranking_not_before(self) -> None:
    # 634 sits last in the pool; at full strength it must still surface into the top 1.
    reranker = AuthorityReranker(..., pool_multiplier=3, strength=1.0)
    self.assertEqual(reranker.search_chunks(QUERY, depth=1), [INDEX_OF[634]])
```

## Deep Explanation: the version rule, implemented to be disproved

The obvious version refinement: penalise a PEP whose `Python-Version` postdates the version the query
asks about. If you ask about 3.7, why trust a document about 3.10?

The module argues this is wrong, at length, in its docstring:

```python
"""## Why version is deliberately NOT a ranking signal

The obvious rule -- penalise a PEP whose `Python-Version` is later than the version the query
asks about -- is wrong, and measurably so.

Asked *"can I use the walrus operator in Python 3.7?"*, the correct answer is "no, 3.8", and
producing it **requires retrieving PEP 572, whose version is 3.8**. Penalising it for being
later than 3.7 removes precisely the evidence needed to answer correctly. Version tells you
what the answer must *say*, not which document to trust.

The rule is implemented anyway, behind `version_penalty`, so the claim can be measured instead
of asserted.
"""
```

That last sentence is the methodological point. A rejected design that is merely argued against remains
an opinion. Implemented behind a flag and measured, it becomes a result:

| variant | superseded ↓ | authoritative ↑ | version-correct ↑ |
|---|---|---|---|
| λ=1 | 0.039 | **0.863** | **0.714** |
| λ=1 + version penalty | 0.039 | 0.804 | 0.571 |

It changes authority not at all — the same 0.039 — and makes both other metrics **worse**. The version
metric it was supposed to improve drops from 0.714 to 0.571.

Prediction made in advance, then confirmed. Chapter 23 notes this was the *only* one of three
predictions in the project that held.

The implementation, and the reason `parse_release` returns tuples:

```python
def parse_release(version: str | None) -> tuple[int, ...] | None:
    release = release_version(version)
    if release is None:
        return None
    try:
        return tuple(int(part) for part in release.split("."))
    except ValueError:
        return None
```

```python
def test_tuple_comparison_beats_string_comparison(self) -> None:
    # "3.9" > "3.10" as strings; the whole point of parsing to tuples.
    self.assertGreater(parse_release("3.10"), parse_release("3.9"))
```

String comparison would rank 3.9 above 3.10 — a classic version-comparison bug, and one that would have
made the ablation's result meaningless rather than merely negative.

## The constructor guard

```python
    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        self.chunks = self.chunks or self.base.chunks
```

A strength outside `[0, 1]` breaks the blend's meaning — above 1, the multiplier goes negative for
low-weight documents and the sort inverts. A sweep loop with an off-by-one would otherwise produce a
silently nonsensical row in the results table.

## Measured results

The answer-level sweep, live:

```
  lambda  superseded  authorit.  version  halluc.   trap: superseded
--------------------------------------------------------------------
       0       0.157      0.765    0.714    0.000              0.300
    0.05       0.235      0.725    0.857    0.000              0.450
    0.25       0.078      0.824    0.714    0.000              0.150
       1       0.039      0.863    0.714    0.000              0.050
```

In counts: superseded citations fall from **8 of 51 to 2 of 51**; on the trap subset from **6 of 20 to
1 of 20**. Per query, λ=0 → λ=1: **6 fixed, 0 broken.**

```
FIXED (6) -- cited a dead PEP at lambda=0, clean at lambda=1
  q06/trap  dropped PEP 3103        q26/trap  dropped PEP 722
  q16/trap  dropped PEP 563         q31       dropped PEP 346
  q20/trap  dropped PEP 554         q47/trap  dropped PEP 431

BROKEN (0)
```

Chapter 23 interprets these — including the λ=0.05 row, which is worse than doing nothing.

## Systems Perspective

The reranker costs essentially nothing: one dictionary lookup and three arithmetic operations per pooled
chunk, over 50 chunks. Microseconds against the hybrid retrieval it wraps.

That cheapness is the practical headline of the whole project. The expensive components — an embedding
model, a matrix multiply over 58 MB — cannot see authority at all. The free component can. Chapter 11's
diagram made this point structurally; here it is as a cost measurement.

## Common Mistakes

**Comparing the intervention to a different system rather than to λ=0 of itself.** Chapter 21 shows how
badly this goes wrong here.

**A uniform sweep grid.** With RRF base scores 1.6% apart, a 0.25-step grid misses the entire
transition.

**Reranking only the top-k.** Nothing can be promoted from below the cutoff, which is the whole point.

**Filtering instead of weighting.** Loses the rejected proposals that legitimately answer "why is this
not in the language" — gold query `q06` depends on one.

**Replacing the status weight instead of multiplying.** The two signals stop composing.

**String comparison on version numbers.** `"3.9" > "3.10"`.

**Arguing against a design instead of measuring it.** The version rule cost about ten lines and one
extra sweep point, and it turned an opinion into a result.

## Interview Insight

> **"How does your reranker work?"**

It blends an authority weight into the fusion score with a tunable strength: `final = base × (1 − λ +
λ·weight)`. At λ=0 it reproduces the unreranked baseline exactly — that is a tested invariant, and it is
what makes the sweep's zero point a legitimate control rather than a separate system.

The weight combines two signals. Status is graded rather than boolean — `Active` is as authoritative as
`Final`, `Deferred` is dormant rather than refused, so it outranks `Rejected`. And the supersession edge
is checked *independently of status*, because this corpus contains pairs where both documents are
`Final` and only the edge separates them. A document that is both superseded and has a named successor
gets the two penalties multiplied.

> **"How did you choose the sweep grid?"**

From arithmetic, before measuring. RRF base scores at adjacent ranks differ by 1.6%, while authority
weights span a factor of 44 — so flipping an adjacent pair takes λ ≈ 0.0165, and promoting a document
from twenty ranks deeper takes λ ≈ 0.25. The entire transition lives below 0.3.

A uniform 0.25-step grid would have shown four near-identical rows and made a graded parameter look like
a switch. The measured curve confirmed it: nothing moves at 0.01, everything starts at 0.02.

> **"Was there anything you expected to work that didn't?"**

The version rule. The obvious refinement is to distrust documents newer than the version being asked
about — and it is wrong, because answering *"no, that arrived in 3.8"* requires retrieving the very
document the rule demotes. I implemented it behind a flag specifically so I could measure that rather
than assert it: it left the authority metric completely unchanged and made both other metrics worse,
including the version metric it was meant to improve.

I would rather ship a disproved idea with the evidence attached than an unexamined opinion in a
docstring.

## Debugging Tip

When a reranker does not behave as expected, print the pool with weights before and after:

```
rank  chunk   PEP  status        weight   base      final
   1    412   622  Superseded    0.0225   0.016393  0.000369
   2    118   634  Final         1.0000   0.016129  0.016129   <- promoted
```

Two things become visible at once: whether the weight is what you think (a status typo shows up
immediately as 0.50, the unknown-status default), and whether λ is large enough to overcome the base
gap. Both failure modes look identical in the output ranking and completely different here.

## Summary

- `final = base × (1 − λ + λ·weight)`, with λ=0 reproducing rung 3 exactly — a tested invariant that
  makes the sweep's zero point a legitimate control.
- Two signals: graded status (nine values, judgement-based, the study's largest untested assumption) and
  the supersession edge checked independently of status, skipped when the successor is outside the
  corpus. They multiply, so both apply.
- The knob is violently non-linear: adjacent pairs flip at λ≈0.0165, deeper promotion needs λ≈0.25.
  Derived from RRF arithmetic in advance and confirmed by the measured curve.
- Reranking operates on a pool ten times the cutoff, then truncates — otherwise nothing can be promoted
  from below.
- The version rule is implemented behind a flag *so it could be disproved*: it left authority unchanged
  and made both other metrics worse.
- Measured: superseded citations 8 of 51 → 2 of 51, six queries fixed and none broken.

## Key Takeaways

1. Make your intervention's zero point identical to your baseline, and test it.
2. Derive your sweep grid from the arithmetic before running it.
3. Rerank a pool, not your output.
4. Implement the design you intend to reject, so rejecting it is a result rather than an opinion.

## Why the Next Chapter Exists

Chapter 21 reads the gold set and the four drivers — the code that turns all of this into tables — and
covers the verification script that stops label rot, plus the baseline mistake that would have
misattributed a pool-depth effect to reranking on 50 of 51 queries.
