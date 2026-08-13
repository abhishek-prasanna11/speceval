# Chapter 6 — Combining Rankers: Reciprocal Rank Fusion

## Learning Objectives

- Explain why two retrievers' *scores* cannot simply be added.
- Explain why any calibration constant you introduce becomes a hidden parameter under every result.
- Derive **reciprocal rank fusion** (RRF) from the constraint "use only ranks".
- Explain what the constant `k = 60` controls, and compute the effect of changing it.
- Read `speceval`'s `HybridRetriever` and explain why fusion happens at chunk level.
- Understand the fusion pathology this project measured, and why hybrid did not win.

## Motivation

Chapters 3 and 4 produced two retrievers that fail differently:

| | strong on | weak on |
|---|---|---|
| **BM25** | exact identifiers (`__future__`, `:=`, version strings) | paraphrase (`postpone` vs `deferred`) |
| **Dense** | paraphrase, questions, rationale | exact tokens the model has smoothed over |

Complementary weaknesses are the classic argument for combining. The received wisdom in retrieval is
that hybrid beats either component, and it is repeated often enough to be treated as settled.

This chapter builds the combination properly. Chapter 23 reports that in this project **it did not
win**, and the reason is specific and instructive.

## First Principles

### Attempt 1: add the scores

```python
combined = bm25_score + cosine_score      # wrong
```

Look at what these numbers actually are.

**BM25 scores are unbounded and corpus-dependent.** A score is a sum over query terms of
`idf × saturation`. More query terms means a larger sum. Rarer terms mean larger `idf`. There is no
maximum. On this corpus a BM25 score might be 3, or 18, depending on the query.

**Cosine similarities live in [−1, 1].** In practice, for related text, between about 0.3 and 1.0.

Adding them means BM25 silently dominates by an order of magnitude, and *how much* it dominates
varies per query. The "hybrid" retriever would be BM25 with cosmetic noise.

### Attempt 2: normalise, then add

The obvious repair — scale both to `[0, 1]` and add with a weight:

```python
combined = α · normalise(bm25) + (1 - α) · normalise(cosine)
```

Better, and this is a legitimate technique. But it introduces two problems, one practical and one
methodological.

**Practical: how do you normalise?** Divide by the maximum score *in this result set*? Then the
scaling changes per query, and a query where BM25 found one strong match is scaled differently from
one where it found ten mediocre ones. Min-max normalisation makes the top result always 1.0 and the
bottom always 0.0, destroying the information that one query's matches were all weak.

**Methodological: `α` is now a parameter under every result you report.** You have to pick it. If you
pick it by trying values against your gold set, you have fitted a parameter to your evaluation data,
and your reported numbers are optimistic in a way you cannot quantify. If you pick 0.5 arbitrarily,
every conclusion carries an unexamined constant.

For a study whose entire output is comparative claims, that second problem is the serious one.
`speceval` says so explicitly:

```python
    """Rung 3 -- reciprocal rank fusion of BM25 and dense rankings.

    RRF rather than a weighted sum of scores, because BM25 scores are unbounded and
    corpus-dependent while cosine similarities live in [-1, 1]. Adding them requires
    calibrating one to the other, and any calibration constant would become an unexamined
    parameter sitting underneath every result. RRF uses only ranks, so it needs none::
    """
```

### Attempt 3: throw the scores away

Here is the insight. The scores are incomparable, but the **ranks** are not.

"BM25 put this chunk third" and "dense put this chunk seventh" are statements in the same units —
positions in a list — regardless of what produced them. Ranks are comparable by construction.

So use only ranks. This is **reciprocal rank fusion**:

```
                        1
   score(d) =   Σ   -----------
              systems  k + rank_s(d)
```

For each retriever, take the reciprocal of the document's rank in that retriever's list, and sum
across retrievers. A document ranked well by both accumulates two large contributions. One ranked
well by only one gets a single contribution.

*(RRF is standard from the literature — Cormack, Clarke and Buettcher, 2009. The constant `k = 60`
is their conventional value. This is one of the two external facts in this book not derived from the
project's code.)*

### What the constant `k` does

Without `k` (that is, `k = 0`), the score would be `1/rank`:

```
   rank 1 -> 1.000
   rank 2 -> 0.500        rank 1 is worth TWICE rank 2
   rank 3 -> 0.333
```

Rank 1 dominating rank 2 by a factor of two is too aggressive. A document ranked 1 by one system and
50 by another would beat a document ranked 2 by both, which is usually not what you want — agreement
between systems is evidence.

Adding `k = 60` flattens the curve dramatically:

```
   rank 1  -> 1/61  = 0.016393
   rank 2  -> 1/62  = 0.016129        only 1.6% less than rank 1
   rank 10 -> 1/70  = 0.014286
   rank 50 -> 1/110 = 0.009091
```

Now the top ranks are nearly equal, and what matters is *appearing near the top in both lists* rather
than being first in one.

```
  contribution
     |
 1/61|*  *  *  *  *  *  *          k=60: nearly flat at the top
     |                  *  *  *  *
     |                             *  *  *
     |
     |*                            k=0: rank 1 crushes everything
     | *
     |   *
     |      *   *   *   *   *   *
     +--------------------------------> rank
     1   5   10  15  20  ...    50
```

`speceval` names this:

```python
# Conventional constant from the original reciprocal-rank-fusion paper. It damps the
# influence of top ranks: without it, rank 1 would dominate rank 2 by 2x.
RRF_K = 60
```

**This flatness has a consequence that becomes central in chapter 20.** Because adjacent ranks differ
by only 1.6%, *any* other signal blended into the score can flip an adjacent pair almost immediately.
That is why the reranker's strength parameter turns out to be extremely non-linear, and why the
project's sweep grid had to be fine near zero. Remember the 1.6%.

## Mental Model

Two experienced colleagues each rank ten job candidates. Their internal scoring is not comparable —
one is generous, one is harsh, one uses a 1–10 scale and the other thinks in tiers. Averaging their
raw numbers is meaningless.

But you can ask each for an ordered list, then reward candidates who appear near the top of **both**.
That is RRF: you discard the opinions' magnitudes and keep only their orderings, on the grounds that
agreement between independent judges is the signal you actually trust.

## Deep Explanation

### The implementation

From `speceval/retrievers.py`:

```python
@dataclass
class HybridRetriever:
    lexical: BM25Retriever
    dense: DenseRetriever
    chunks: list[Chunk] = field(default_factory=list)
    name: str = "Hybrid"
    chunk_depth_multiplier: int = CHUNK_DEPTH_MULTIPLIER
    rrf_k: int = RRF_K

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        fused: dict[int, float] = defaultdict(float)
        for retriever in (self.lexical, self.dense):
            for rank, chunk_index in enumerate(retriever.search_chunks(query, depth)):
                fused[chunk_index] += 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
        return [chunk_index for chunk_index, _score in ranked[:depth]]
```

Four things to notice.

**`defaultdict(float)` unions the candidate sets.** A chunk found by only one retriever still enters
`fused` and still gets a score. Fusion is a union, not an intersection — if it were an intersection,
you would lose every document only one system could find, which is precisely the complementary
strength you were trying to exploit. Pinned:

```python
def test_a_chunk_only_one_system_found_is_still_included(self) -> None:
    # Fusion must union the candidate sets, not intersect them.
    ...
    self.assertEqual(set(hybrid.search(QUERY, top_k=2)), {10, 30})
```

**`rank + 1` converts zero-based enumerate to one-based rank.** Rank 0 would give `1/60` rather than
`1/61`; a small difference, but the formula is defined on one-based ranks.

**Ties break on chunk index**, same determinism discipline as chapter 3.

**Agreement wins.** A chunk ranked first by both scores `1/61 + 1/61`; one ranked first by one and
third by the other scores `1/61 + 1/63`. Tested:

```python
def test_agreement_beats_single_system_preference(self) -> None:
    # Chunk 0 is ranked first by both; chunk 2 is ranked first by only one.
    ...
    self.assertEqual(hybrid.search(QUERY, top_k=1), [10])
```

### Why fusion happens at chunk level, not document level

This is the subtlest decision in the module, and it is easy to get wrong.

Both retrievers ultimately return *documents*. So why not fuse the document lists?

Because collapsing to documents first destroys information. Suppose BM25 finds four chunks of PEP 634
in its top ten, and dense finds one. Both, collapsed, say "PEP 634 at rank 1". But those are
different strengths of evidence — four independent passages versus one.

So the module exposes both levels:

```python
"""Retrieval happens over *chunks*; the public `search` collapses chunks to a ranked list of
distinct PEPs. `search_chunks` is exposed separately because fusion has to happen at chunk
level: fusing two already-collapsed PEP lists would throw away the evidence about *how
many* chunks of a PEP each system liked.
"""
```

`search_chunks` is the primitive; `search` is the collapse. Fusion consumes the primitive. Chapter 17
covers this interface in full, and chapter 20's reranker uses the same primitive for the same reason.

### The depth multiplier

```python
# Chunks are retrieved deeper than the PEP cutoff because several chunks of the same PEP
# routinely occupy the top positions; without this, top_k distinct PEPs is unreachable.
CHUNK_DEPTH_MULTIPLIER = 10
```

With 26.9 chunks per PEP on average, a query about PEP 634 will often have five of its chunks in the
top ten. Collapsing those to distinct documents yields far fewer than ten. So to return 10 documents,
retrieve 100 chunks. Without this the retriever would routinely return three or four documents when
asked for ten, and Recall@10 would be capped by an implementation artefact rather than by retrieval
quality.

### The fusion pathology

Fusion is not guaranteed to return anything either component liked best. This project measured a
concrete instance, recorded in `notes/phase2.md`:

> On q34 ("can I use builtin list and dict directly in type hints") hybrid returned PEP 637
> (`Rejected`) at rank 1 — a PEP that *neither* BM25 nor dense ranked first.

The mechanism: a document ranked, say, third by both systems scores `1/63 + 1/63 = 0.0317`, while a
document ranked first by one and thirtieth by the other scores `1/61 + 1/91 = 0.0274`. Consistent
mediocrity beats inconsistent excellence.

Usually that is the behaviour you want — it is exactly the "agreement is evidence" property. But it
means **fusion can surface a document that no component would have chosen**, and if that document is
bad, you have introduced a failure that neither input had.

### Why hybrid did not win here

Measured on the 51-query gold set:

| Strategy | Recall@10 | nDCG@10 | rank-1 correct |
|---|---|---|---|
| BM25 | 0.863 | 0.671 | 47.1% |
| Dense | **0.971** | **0.801** | **60.8%** |
| Hybrid | 0.931 | 0.765 | 56.9% |

Hybrid sits *between* its components rather than above them. The likely mechanism is a direct
consequence of RRF's design:

**RRF weights both systems equally, and has no way not to.** That is the whole point — it avoids a
calibration constant. But here BM25 is substantially the weaker retriever (0.671 nDCG versus 0.801),
so fusing them at equal weight drags the strong ranker toward the weak one. A weighted variant could
express "trust dense more" — at the cost of reintroducing exactly the parameter RRF was chosen to
avoid.

This is a genuine tradeoff rather than a mistake, and it is worth being clear about: the choice to use
RRF bought methodological cleanliness and cost some retrieval quality. Both halves of that sentence
are true.

**One important caveat, and chapter 23 develops it.** Those numbers are *retrieval* metrics at
`K = 10` over documents. At the answer level, measured over the top 5 chunks the generator actually
consumes, **hybrid was the best rung.** The conclusion above is correct about what it measures and
wrong as a guide to the finished system.

## Systems Perspective

RRF costs almost nothing: run both retrievers, walk two lists of length `depth`, sum into a dict,
sort. Measured p50 for hybrid is ≈ 28 ms against dense's ≈ 37 ms and BM25's ≈ 28 ms — hybrid is not
the sum of its parts because the dominant cost in dense retrieval is the query-embedding round trip,
which hybrid pays exactly once and shares.

The parallelisation opportunity is obvious and untaken here: the two retrievers are independent and
could run concurrently. At these latencies it would not matter, and added concurrency in a
measurement harness is added nondeterminism for no benefit.

## Common Mistakes

**Adding raw scores from different retrievers.** Unbounded plus bounded means one dominates
arbitrarily.

**Min-max normalising per query.** Forces the top result to 1.0 and the bottom to 0.0 regardless of
whether the matches were strong, destroying exactly the information you wanted.

**Intersecting instead of unioning.** Discards the documents only one retriever could find, which was
the reason to combine them.

**Fusing collapsed document lists.** Throws away how much evidence each system had per document.

**Assuming hybrid always wins.** It is received wisdom, not a theorem. This project is a
counterexample at the retrieval level, and the reason (equal weighting of unequal rankers) is
predictable in advance.

**Fitting the fusion weight on your evaluation set.** Then your reported numbers include a parameter
tuned on the data you are reporting.

## Interview Insight

> **"How do you combine lexical and semantic retrieval?"**

Name the problem first: the scores are not comparable — BM25 is unbounded and corpus-dependent,
cosine sits in `[-1, 1]` — so you cannot add them without a calibration constant, and that constant
becomes an untested parameter under every result.

Reciprocal rank fusion avoids it by discarding scores and using only ranks:
`score(d) = Σ 1/(k + rank_s(d))`, conventionally `k = 60`, which flattens the top of the curve so
agreement between systems matters more than being first in one.

Then the sentence that shows judgement rather than recall: *hybrid is not guaranteed to win. In this
project it landed between its components at the retrieval level, because RRF weights an unequal pair
equally and BM25 was the weaker ranker — and interestingly it was the best rung at the answer level,
because the retrieval metric was measuring a different depth and unit than the generator consumed.*

> **"What does the constant 60 do?"**

It damps the influence of the top ranks. With `k = 0` you get `1/rank`, where rank 1 is worth twice
rank 2, so one system's top pick beats agreement between both. At `k = 60`, ranks 1 and 2 differ by
1.6%, so consensus dominates. It also means any other signal blended into an RRF score can flip
adjacent ranks very easily — which matters if you later add a reranking term.

## Debugging Tip

When a hybrid retriever behaves oddly, print the three lists side by side — BM25's ranking, dense's
ranking, and the fused ranking — for the offending query. The pathology cases are immediately visible
as a document sitting mid-list in both inputs and top of the output.

`speceval` builds this into `run_phase2.py`, which prints the rank-1 result for all three strategies
per query in one table, precisely so this class of surprise is visible without writing throwaway code.

## Summary

- Scores from different retrievers are not comparable; adding them lets the unbounded one dominate.
- Normalising and weighting works but introduces a calibration constant that then sits unexamined
  beneath every result.
- RRF discards scores and uses only ranks: `Σ 1/(k + rank)`, conventionally `k = 60`.
- `k` flattens the top of the curve so that agreement between systems outweighs being first in one.
  At `k = 60` adjacent ranks differ by 1.6% — remember this for chapter 20.
- Fusion must union candidate sets and must operate on chunks, not collapsed document lists.
- Fusion can promote a document neither component ranked first; usually a feature, occasionally a
  new failure.
- Hybrid did **not** beat dense at the retrieval level here, plausibly because RRF weights an unequal
  pair equally — but it *was* best at the answer level, which chapter 23 explains.

## Key Takeaways

1. Fuse ranks, not scores, unless you are prepared to defend a calibration constant.
2. `k = 60` makes consensus matter more than any single system's top pick.
3. Fuse at the finest granularity you have; collapse afterwards.
4. "Hybrid always wins" is folklore. Measure it.

## Why the Next Chapter Exists

Part I is complete: three retrieval strategies and the metrics to compare them. Part II now covers
the machinery this project uses to *build* those things reliably.

Chapter 7 examines the Python that measurement code specifically needs — immutable value objects,
structural typing so a fourth retriever can be added without touching the evaluation loop, and the
small set of language features that make an experiment reproducible rather than merely runnable.
