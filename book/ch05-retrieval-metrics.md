# Chapter 5 — Knowing Whether It Worked: Retrieval Metrics

## Learning Objectives

- Explain why "it looked better" is not an evaluation and what the minimum viable alternative is.
- Define **Recall@K** precisely and explain why it is set overlap and never positional.
- Derive **nDCG@K** from the question "does rank position matter?" and compute it by hand.
- Explain what each metric is blind to, and why this project reports both.
- Explain why the choice of `K` can invert a conclusion.
- Read `speceval/metrics.py` and justify its two guard clauses.

## Motivation

We have two retrievers built on incompatible principles. Which is better?

The way this question is usually answered is: run a few queries you have in mind, eyeball the
results, form an impression. This fails for reasons worth stating plainly, because it is genuinely
the default practice.

**You will pick queries you already know the answers to.** Those are the queries you built the
system thinking about, so they are the ones it handles.

**Impressions do not aggregate.** "A was better on two, B on one, and the fourth was ambiguous" does
not resolve into a decision, and it certainly does not survive adding a fifth query.

**You cannot detect a regression.** Six weeks later, after twenty changes, there is nothing to
compare against.

**You cannot tell a real improvement from noise.** If A beats B on one query out of four, that is
consistent with A being better and equally consistent with luck.

The alternative is a **metric**: a single number computed the same way every time, over a fixed set
of queries with known answers. Two things are needed — the known answers, and the number.

## First Principles

### Relevance judgements

Chapter 1 established that relevance is a human judgement about intent, not a property a document
has. So it must be recorded by hand, in advance:

```json
{"qid": "q02", "text": "what does the walrus operator := do",
 "category": "identifier", "relevant": [572], "trap": false,
 "note": "PEP 572, Final, 3.8."}
```

A set of these is a **gold set**. This project's has 51 entries — chapter 12 is about how it was
built and chapter 21 about how it is verified.

One decision inside it shapes both metrics that follow. Relevance here is recorded **per document**
(per PEP), not per chunk:

```python
"""Relevance is labelled at **PEP level**, not chunk level: a query's ground truth is the
set of PEP numbers that legitimately answer it. Retrievers return ranked chunks, which
are collapsed to ranked distinct PEPs before scoring. This is a deliberate choice --
labelling every relevant *chunk* by hand would cost several times more and would not
change which retrieval strategy wins.
"""
```

Labelling 19,763 chunks by hand is not feasible; labelling 51 queries against 734 documents is. The
cost is a loss of resolution — you cannot ask "did it find the *right paragraph*?", only "did it
find the right document?". Chapter 23 records a case where that loss of resolution mattered.

### Metric 1: Recall@K

The simplest useful question: *of the documents that were relevant, how many did we return in the
top K?*

```
                    | {top K results} ∩ {relevant} |
   Recall@K  =  ---------------------------------------
                           | {relevant} |
```

Worked example, `K = 3`:

```
returned:  [ 634, 622, 3103 ]
relevant:  { 634, 636 }

intersection = {634}      -> 1
|relevant|   = 2
Recall@3     = 0.5
```

We found one of the two relevant documents. Half credit.

### The trap inside Recall: it is set overlap, never positional

There is a natural-seeming but wrong way to implement this: walk the returned list and the relevant
list together, comparing position by position.

That implementation is broken, and broken in a way that produces plausible numbers. Consider a
retriever that is *provably perfect* — it returns exactly the relevant set — but in a different
order than your gold set happened to list them:

```
returned:  [ 636, 634 ]
relevant:  [ 634, 636 ]

positional comparison:  position 0: 636 != 634  ->  miss
                        position 1: 634 != 636  ->  miss
                        score: 0.0     <-- catastrophically wrong

set overlap:            {636, 634} ∩ {634, 636} = both  ->  1.0   <-- correct
```

A perfect retriever scored zero. And nothing crashes; you simply conclude your retriever is broken
when your metric is.

`speceval` states this and pins it with a test:

```python
def recall_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Fraction of the relevant items that appear in the top k.

    Set overlap, never rank-by-rank: position within the top k does not matter here,
    only membership. (Ordering is what nDCG measures.)
    """
    _validate(relevant, k)
    return len(set(ranked[:k]) & relevant) / len(relevant)
```

```python
def test_is_set_overlap_not_positional(self) -> None:
    # Reordering the top k must not change recall -- that is nDCG's job. Getting
    # this wrong scores a provably exact retriever at well under 1.0.
    forward = recall_at_k(["a", "b", "c"], {"a", "c"}, 3)
    reversed_ = recall_at_k(["c", "b", "a"], {"a", "c"}, 3)
    self.assertAlmostEqual(forward, reversed_)
    self.assertAlmostEqual(forward, 1.0)
```

This is not a hypothetical error. The author of this project hit exactly this bug in a previous
piece of work on approximate nearest-neighbour search, where a positional comparison scored a
provably exact algorithm at around 50%. The lesson transferred into this codebase as a test.

### What Recall@K is blind to

Recall does not care *where* in the top K a relevant document sits. These score identically at
`K = 10`:

```
A:  [ 634, junk, junk, junk, junk, junk, junk, junk, junk, junk ]   Recall@10 = 1.0
B:  [ junk, junk, junk, junk, junk, junk, junk, junk, junk, 634 ]   Recall@10 = 1.0
```

For a human reading results, A is excellent and B is nearly useless. For a language model handed
the top 5, A works and **B fails entirely** — 634 never arrives.

So we need a second metric that is sensitive to position.

### Metric 2: nDCG@K, derived

Build it in three steps.

**Step 1 — cumulative gain.** Give each result a gain: 1 if relevant, 0 if not. Sum over the top K.
This is just a count, and is still position-blind.

**Step 2 — discount by position.** A result at rank 1 should be worth more than one at rank 5. We
need a decreasing weight. The convention is `1 / log₂(rank + 1)`, with rank starting at 1:

```
   rank 1:  1/log₂(2) = 1/1.000 = 1.000
   rank 2:  1/log₂(3) = 1/1.585 = 0.631
   rank 3:  1/log₂(4) = 1/2.000 = 0.500
   rank 4:  1/log₂(5) = 1/2.322 = 0.431
   rank 5:  1/log₂(6) = 1/2.585 = 0.387
   rank 10: 1/log₂(11)= 1/3.459 = 0.289
```

Why a logarithm rather than `1/rank`? Because `1/rank` falls off brutally — rank 2 would be worth
half of rank 1, rank 10 a tenth. Empirically, user attention declines more gently than that. The log
gives a curve where the top few positions matter a lot and the difference between rank 8 and rank 9
is small, which matches how people read result lists.

Summing the discounted gains gives **DCG** — discounted cumulative gain:

```
              K      relevant(rank i)
   DCG@K  =  Σ    ---------------------
             i=1      log₂(i + 1)
```

**Step 3 — normalise.** DCG is not comparable across queries: a query with four relevant documents
can reach a higher DCG than a query with one, simply because there is more to find. So divide by the
best score *possible* for that query — the **ideal DCG**, computed by imagining all relevant
documents packed into the top positions:

```
   nDCG@K  =  DCG@K / IDCG@K
```

Now every query scores between 0 and 1, and 1.0 means "perfectly ordered".

### Computing nDCG by hand

Three relevant documents exist; `K = 3`; the retriever returns two of them at ranks 1 and 3.

```
returned:  [ 634*, junk, 636* ]        * = relevant
relevant:  { 634, 636 }

DCG  = 1/log₂(2) + 0 + 1/log₂(4)
     = 1.000 + 0 + 0.500
     = 1.500

IDCG = both relevant packed at ranks 1 and 2
     = 1/log₂(2) + 1/log₂(3)
     = 1.000 + 0.631
     = 1.631

nDCG@3 = 1.500 / 1.631 = 0.920
```

`speceval` encodes this exact arithmetic as a test, which is how you know the implementation matches
the definition rather than merely running:

```python
def test_two_hits_at_ranks_one_and_three(self) -> None:
    # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1/log2(2) + 1/log2(3)
    expected = 1.5 / (1.0 + 1 / math.log2(3))
    self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a", "c"}, 3), expected)
```

The implementation:

```python
def ndcg_at_k(ranked: Sequence[Hashable], relevant: set[Hashable], k: int) -> float:
    """Normalised discounted cumulative gain, binary relevance, log2 discount.
    ...
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
```

Note `rank + 2` rather than `rank + 1`: `enumerate` is zero-based, so rank 0 must produce
`log₂(2)`. And `min(len(relevant), k)` in the ideal: if there are five relevant documents but `K=3`,
the best achievable is three hits, not five — without that `min`, a perfect retriever would score
below 1.0.

## Mental Model

**Recall@K** is *"did you bring me the right book?"*

**nDCG@K** is *"did you bring me the right book, and was it on top of the pile?"*

You need both, because they fail differently. A retriever can find everything and order it terribly
(high recall, low nDCG) — which is exactly what this project's BM25 baseline does, and chapter 23
shows why that pattern was informative rather than merely bad.

## Deep Explanation

### The two guard clauses, and why they exist

```python
def _validate(relevant: set[Hashable], k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant:
        raise ValueError("empty relevant set -- a query with no labels is a bug")
```

The second is the interesting one. What should `recall_at_k(results, set(), 10)` return?

Mathematically it is `0/0`. Practically there are three options:

- Return `1.0` — "we found all zero of them". Vacuously true, and it silently *inflates* your
  average.
- Return `0.0` — silently *deflates* your average.
- Raise.

The project raises, and the docstring says why:

```python
"""Both functions raise on an empty relevant set. A query with no labels is a bug in the
golden set, and silently scoring it 1.0 (or 0.0) would hide that bug in an average.
"""
```

This is a general principle for measurement code and worth stating as a rule: **when the input is
malformed, fail loudly rather than returning a number.** A returned number gets averaged with 50
others and disappears. An exception gets fixed. The whole value of an evaluation harness is that its
numbers can be trusted, and a harness that quietly absorbs bad input has given that up.

### Why this project reports recall but not precision

Chapter 1 introduced precision alongside recall. `speceval` reports Recall@10 and nDCG@10 and does
not report precision. That is a deliberate choice, not an omission, and the reason is in the gold set
structure.

Most queries here have **one** relevant document. With `K = 10`, the best possible precision is
`1/10 = 0.1`, and it would be 0.1 for every well-performing system on every such query. A metric
whose maximum is 0.1 and which cannot distinguish good systems from each other is not measuring
anything useful. nDCG already captures the thing precision would be gesturing at — whether the good
result is near the top.

### The cutoff K can invert your conclusion

`K` is not a formatting detail. Consider two retrievers and `K = 5` versus `K = 10`:

```
              rank:  1     2     3     4     5     6     7     8     9    10
   system A:        junk  634*  junk  junk  junk  junk  junk  junk  junk  junk
   system B:        junk  junk  junk  junk  junk  junk  634*  junk  junk  junk

   Recall@5:   A = 1.0    B = 0.0        -> A is vastly better
   Recall@10:  A = 1.0    B = 1.0        -> indistinguishable
```

Same systems, same query, two different conclusions from the choice of one integer.

This is not a contrived example. It is the mechanism behind this project's most methodologically
useful finding. `speceval` measures retrieval at `K = 10` over **documents**, but its generator
consumes the top **5 chunks**. Two mismatches at once — unit and depth. In Phase 2, hybrid retrieval
looked no better than dense on Recall@10 and nDCG@10 and was the rung to discard. In Phase 3, at the
answer level, hybrid was the *best* rung. Same corpus, same queries, opposite conclusion, entirely
because the metric was scoring something different from what the system actually consumed.

The lesson, which chapter 23 states as the project's most transferable result: **a retrieval metric
is a proxy for downstream quality, and it is only as faithful as the match between what it scores and
what the consumer consumes.**

## Systems Perspective

Both metrics are `O(K)` with a set intersection, so cost is nothing. The expensive part of an
evaluation is retrieval and generation, not scoring.

That has a practical implication worth using: because scoring is free, you can afford to compute
metrics on **every subset you can think of** — per category, per query type, trap versus ordinary —
rather than only in aggregate. Chapter 12 argues that disaggregation is where findings actually live,
and there is no performance reason not to do it.

## Common Mistakes

**Positional recall.** Scores a provably exact retriever near zero. Test with a shuffled oracle.

**Forgetting `min(len(relevant), k)` in IDCG.** A perfect retriever then scores below 1.0 whenever
there are more relevant documents than `K`.

**Off-by-one in the log discount.** `enumerate` is zero-based; the discount needs `rank + 2`.

**Returning a number for malformed input.** It vanishes into an average.

**Reporting a metric without its `K`.** "Recall was 0.89" is not a claim.

**Averaging over queries with wildly different numbers of relevant documents without thinking about
it.** A query with four relevant documents contributes the same weight as one with a single relevant
document, which may or may not be what you want.

## Interview Insight

> **"How do you evaluate a retrieval system?"**

A fixed set of queries with human relevance judgements, and at least two metrics: one for whether
the right documents were found at all (Recall@K), one for whether they were ordered well (nDCG@K).
State `K` every time.

Then the part that distinguishes a real answer: **and you validate the metric implementation before
trusting it.** Run a synthetic retriever that returns exactly the ground truth — it must score 1.0.
Run one that returns random documents — it must score near chance. A metric bug produces
plausible-looking numbers and nothing else in the pipeline will reveal it. Chapter 13 is entirely
about this.

> **"What does nDCG measure that recall does not?"**

Position. Recall asks whether the relevant document is in the top K at all; nDCG asks whether it is
near the top, discounting by `1/log₂(rank+1)` and normalising against the best possible ordering for
that query. A system can score 1.0 on Recall@10 while putting every relevant document at rank 10 —
useless in practice, and nDCG catches it.

> **"How do you choose K?"**

Match it to what actually consumes the results. If a language model gets the top 5 chunks, measuring
Recall@10 over documents is measuring something your system does not do — and that mismatch can
reverse which strategy looks better. That is a mistake this project made and documented.

## Debugging Tip

Before trusting any evaluation, run three sanity checks that take minutes:

1. **The oracle.** A retriever returning exactly the gold answers must score 1.000 on everything.
   Anything less is a metric bug.
2. **Random.** A retriever returning random documents must score near chance. Anything much above
   means you are accidentally leaking the answer.
3. **Shuffle the oracle.** Reorder its output. Recall must not change; nDCG may. If recall moves,
   you have implemented it positionally.

## Summary

- "It looked better" is not an evaluation: it selects easy queries, does not aggregate, cannot detect
  regressions, and cannot separate signal from luck.
- A gold set of `(query, relevant documents)` pairs is the prerequisite. This project labels at
  document level, trading resolution for feasibility.
- **Recall@K** is set overlap over the top K. Position-blind by definition. Implementing it
  positionally scores a perfect retriever near zero.
- **nDCG@K** discounts by `1/log₂(rank+1)` and normalises by the ideal ordering, so it is
  position-sensitive and comparable across queries.
- Metrics must raise on malformed input, because a returned number disappears into an average.
- Precision is omitted here because most queries have one relevant document, capping it at 0.1.
- `K` is a substantive choice. A mismatch between the metric's `K`/unit and the consumer's can invert
  your conclusion — and did, in this project.

## Key Takeaways

1. Recall is membership; nDCG is ordering. Report both, always with `K`.
2. Recall is set overlap. Never rank-by-rank.
3. Fail loudly on malformed input in measurement code.
4. Measure at the depth and unit your system actually consumes, or your metric is a proxy for the
   wrong thing.

## Why the Next Chapter Exists

Part I is complete: we can score documents lexically, score them geometrically, and measure which
scoring works better.

The natural next question is whether we have to choose. BM25 and dense retrieval fail on *different*
queries — BM25 on paraphrase, dense on exact identifiers — which suggests combining them should beat
either. Chapter 6 shows how to combine two rankings when their scores are not comparable, introduces
reciprocal rank fusion, and sets up a result that contradicts the conventional wisdom about hybrid
retrieval.
