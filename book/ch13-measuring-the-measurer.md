# Chapter 13 — Measuring the Measurer

## Learning Objectives

- Explain why a metric bug is undetectable from anywhere else in a system.
- Explain the **oracle** and **random** retrievers and what each proves.
- Explain why hand-computed metric values belong in the test suite.
- Read two real validation results from this project and explain what each one confirmed.
- Explain why a metric's *floor* may not be zero, and why that is a property of the corpus.

## Motivation

Every chapter so far has built something that produces a number. This chapter asks the question that
makes those numbers worth anything: **how do you know the measuring instrument is not broken?**

The situation is genuinely unlike ordinary testing. If your retriever is broken, results look wrong and
you investigate. If your *metric* is broken:

- the pipeline runs to completion,
- every number is in a plausible range,
- the ranking between systems may even be preserved,
- and nothing, anywhere, reports a problem.

You then write conclusions. They are wrong, confidently, and in a way no downstream check can catch.

This is not hypothetical for the author of this codebase. On a previous project — an approximate
nearest-neighbour search engine — a recall function implemented positionally rather than as set
overlap scored a *provably exact* algorithm at around 50%. The algorithm was correct. The metric was
wrong. Days were spent looking in the wrong place. The lesson transferred into this project as
infrastructure.

## First Principles

### The core idea

You cannot test a metric against "the right answer" for real data, because if you knew the right answer
you would not need the metric.

But you *can* construct inputs whose correct score is known **by construction**. Feed the harness a
system that is definitionally perfect: it must score perfectly. Feed it a system that is definitionally
useless: it must score near zero. If either fails, the metric is wrong — regardless of what it says
about your real systems.

This is the same idea as a **positive and negative control** in an experiment. You do not only run the
treatment; you run something you know should work and something you know should not, to establish that
the apparatus can tell them apart.

### Control 1: the oracle

A retriever that returns exactly the ground truth.

```python
@dataclass
class OracleRetriever:
    """Returns exactly the ground truth. Must score 1.0 on every metric."""

    name: str = "Oracle"

    def search(self, query: Query, top_k: int) -> list[int]:
        return sorted(query.relevant)[:top_k]
```

Nine lines. It reads the answer out of the query it was asked. It is not a retriever in any useful
sense; it is a probe.

**What it proves:** if the oracle does not score 1.000, the metric is broken. There is no other
explanation available — the input was perfect.

This is the control that catches the positional-recall bug. A positional implementation compares
element by element, so an oracle returning `[636, 634]` against labels listed `[634, 636]` scores 0.0
despite being exactly right (chapter 5 works the arithmetic). The oracle test makes that failure loud
and immediate.

### Control 2: random

A retriever that returns arbitrary documents.

```python
@dataclass
class RandomRetriever:
    """Returns random PEPs. Must score near chance -- seeded, so it is reproducible."""

    pep_numbers: list[int]
    seed: int = 12345
    name: str = "Random"

    def search(self, query: Query, top_k: int) -> list[int]:
        # Seeded per query, so the result is stable across runs and across retrievers.
        rng = random.Random(f"{self.seed}:{query.qid}")
        return rng.sample(self.pep_numbers, min(top_k, len(self.pep_numbers)))
```

**What it proves:** that you are not accidentally leaking the answer. If random scores well, something
in the harness is helping it — a filter that pre-selects candidates, an evaluation that credits partial
matches too generously, a bug where the gold labels leak into the candidate set.

Note the seeding detail: `random.Random(f"{self.seed}:{query.qid}")` constructs a fresh generator per
query, seeded from both the base seed and the query id. A single generator seeded once would produce
different draws depending on how many queries ran before this one, so adding a query would change
earlier queries' results. Per-query seeding makes each query's draw independent of the others and stable
across runs.

### These are `Retriever`s, which is why this is cheap

Both satisfy the `Retriever` protocol from chapter 7, so they run through the **real** evaluation loop —
the same `evaluate()` that scores BM25 and dense. That is essential. Validating a *reimplementation* of
the metric would prove nothing about the one actually used.

```python
    def test_oracle_scores_perfectly(self) -> None:
        result = evaluate(OracleRetriever(), self.queries, k=10)
        self.assertAlmostEqual(result.overall.recall, 1.0)
        self.assertAlmostEqual(result.overall.ndcg, 1.0)

    def test_random_scores_near_chance(self) -> None:
        # ~10 draws from 734 PEPs: expected recall is on the order of 0.01.
        retriever = RandomRetriever(pep_numbers=list(range(1, 735)))
        result = evaluate(retriever, self.queries, k=10)
        self.assertLess(result.overall.recall, 0.10)
        self.assertLess(result.overall.ndcg, 0.10)

    def test_oracle_beats_random(self) -> None:
        # The single assertion the whole harness exists to be able to make.
        oracle = evaluate(OracleRetriever(), self.queries, k=10)
        random_ = evaluate(
            RandomRetriever(pep_numbers=list(range(1, 735))), self.queries, k=10
        )
        self.assertGreater(oracle.overall.recall, random_.overall.recall)
```

That third test's comment is the point of the whole chapter. Every conclusion in this study has the form
"system A scored higher than system B." `test_oracle_beats_random` asserts that the harness is capable of
detecting *any* difference at all. If it cannot separate a perfect system from a random one, it certainly
cannot be trusted to separate two similar ones.

## Deep Explanation: pinning the arithmetic

Controls catch structural bugs. They do not catch a wrong constant — an nDCG implementation using
`log₂(rank+1)` instead of `log₂(rank+2)` would still give the oracle 1.0, because the same error appears
in numerator and denominator and cancels.

So the metrics are additionally pinned to values computed **by hand**:

```python
    def test_hit_at_rank_two(self) -> None:
        # 1/log2(3) / 1.0
        self.assertAlmostEqual(ndcg_at_k(["b", "a", "c"], {"a"}, 3), 1 / math.log2(3))

    def test_two_hits_at_ranks_one_and_three(self) -> None:
        # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1/log2(2) + 1/log2(3)
        expected = 1.5 / (1.0 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c"], {"a", "c"}, 3), expected)
```

The expected value is written as the *formula*, not as `0.9197`. That means the test documents the
definition rather than merely locking in whatever the code produced. A magic number in a test tells you
the code has not changed; a formula tells you the code matches the definition.

The full set of pins, from `tests/test_harness.py`:

| Test | Pins |
|---|---|
| `test_partial_overlap` | Recall = 0.5 on one of two found |
| `test_no_overlap_is_zero` | **Catches a recall hardwired to 1.0** |
| `test_perfect_overlap_is_one` | Recall = 1.0 |
| `test_k_truncates` | A relevant document outside top-k does not count |
| `test_is_set_overlap_not_positional` | Reordering does not change recall |
| `test_hit_at_rank_one` | nDCG = 1.0 |
| `test_hit_at_rank_two` | nDCG = 1/log₂(3) |
| `test_two_hits_at_ranks_one_and_three` | Full DCG/IDCG arithmetic |
| `test_no_hits_is_zero` | nDCG = 0.0 |
| `test_rewards_ordering_unlike_recall` | nDCG is position-sensitive, recall is not |

`test_no_overlap_is_zero` is worth singling out. The degenerate implementation of any metric is
`return 1.0`. It passes every test that feeds it a correct system. The only thing that catches it is a
case whose correct answer is **zero** — and a validation suite without one is not a validation suite.

## Deep Explanation: two real results from running the controls

Both of these came out of the controls unprompted, and both turned out to be informative rather than
merely reassuring.

### Result 1: the random retriever recovered the corpus base rate

Chapter 11 introduced `trap@1` — the fraction of queries whose rank-1 document is non-authoritative.
Chapter 10 measured that 36% of the corpus is non-authoritative.

Measured, live:

```
Retriever     Recall@10    nDCG@10   trap@1    p50 ms    p95 ms
---------------------------------------------------------------
Oracle            1.000      1.000    0.020      0.00      0.00
Random            0.000      0.000    0.333      0.02      0.03
BM25              0.863      0.671    0.294     27.88     42.55
```

**Random's `trap@1` is 0.333, and the corpus base rate is 0.36.**

That agreement is not something anyone designed. A retriever choosing uniformly at random should hit a
non-authoritative document at exactly the corpus proportion, and it does. This is strong evidence that
`trap@1` measures what it claims and not something incidental — a bug in how status is looked up, or in
how "rank 1" is determined, would be very unlikely to reproduce the base rate by accident.

It also gives the metric a meaningful reference point. BM25's 0.294 is *better than random* but not by
much — a system with no authority awareness performs, on this axis, only slightly better than chance.
That framing is far more informative than 0.294 in isolation.

### Result 2: the oracle's floor is not zero

Look at the oracle's `trap@1`: **0.020**, not 0.000. A perfect retriever registers a trap.

This is not a bug. It is query `q30`, deliberately constructed (chapter 10, case 4):

```json
{"qid": "q30", "text": "is there a unified TLS API in Python",
 "relevant": [748, 543],
 "note": "Deliberate no-authoritative-answer case: 543 is Withdrawn and its successor
          748 is only a Draft. A correct system should decline to present either as settled."}
```

The ground truth itself contains a `Withdrawn` document, because no authoritative answer exists. So even
the oracle, returning exactly the correct labels, leads with PEP 543 — `Withdrawn`. One query in 51 is
0.0196, which rounds to 0.020.

Two things follow.

**The metric's floor is 1/51, not 0.** Any claim of the form "reranking eliminated the problem" must be
read against a floor of 0.020, not against zero. Chapter 23's retrieval sweep reaches exactly 0.000,
which is *better than the oracle* — and that is not a contradiction, because the reranker is free to
demote PEP 543 below PEP 748, whereas the oracle returns labels in sorted numerical order.

**Running the oracle surfaced a property of the corpus.** Nobody set out to establish that the metric
had a non-zero floor. It appeared because a control was run, and the resulting question — *why is this
not zero?* — led to a precise, documented statement about the corpus. That is what controls are for
beyond catching bugs.

## Mental Model

A **scale you check with a known weight** before weighing anything you care about.

Put a certified 1 kg mass on it. If it reads 1 kg, the scale is trustworthy for today. If it reads
0.98 kg you have learned something crucial *before* recording any real measurements.

The oracle is your 1 kg mass. Random is the empty pan — it should read zero, and if it reads 0.3 kg
something is resting on the scale that you cannot see.

## Systems Perspective

The controls cost nothing. The oracle does a dictionary lookup; random does a sample. The whole
validation suite runs as part of the 103-test suite in **0.102 seconds**, measured.

That cheapness has a design consequence: because validation is free, it can be *permanent* rather than a
one-off check during development. These are not scripts someone ran once; they are tests that run on
every change. If a future refactor breaks the metric, the oracle test fails immediately, before any
number is reported.

The reason it can be permanent is chapter 7's `Protocol` decision. If the controls had to inherit from a
retriever base class, they would carry index construction and model dependencies, would be slow, and
would probably have been written as throwaway scripts instead.

## Common Mistakes

**Validating a reimplementation of the metric.** The controls must go through the same `evaluate()` the
real results use.

**Only testing the positive control.** `return 1.0` passes every oracle test. You need a case whose
correct answer is zero.

**Magic numbers as expected values.** `assertAlmostEqual(x, 0.9197)` locks in behaviour; writing the
formula documents the definition.

**Unseeded random controls.** Then the negative control's score changes between runs, and you cannot set
a threshold.

**Seeding once instead of per query.** Adding a query changes earlier queries' draws.

**Assuming the floor is zero.** Check what the oracle actually scores on every metric. If it is not the
theoretical optimum, find out why before interpreting anything.

**Running the controls once during development.** Make them tests. Metrics get refactored.

## Interview Insight

> **"How do you know your evaluation is correct?"**

This is the question that separates people who have run an evaluation from people who have read about
one. The answer is controls.

Run a retriever that returns exactly the ground truth — it must score 1.000, and if it does not, the
metric is broken with no other possible explanation. Run a seeded random retriever — it must score near
chance, and if it scores well, something is leaking the answer. Run both through the *real* evaluation
loop, not a copy. And pin the metric arithmetic to hand-computed values written as formulas, including at
least one case whose correct answer is zero, because the degenerate implementation of any metric is
`return 1.0` and it passes everything else.

> **"Did that ever catch anything?"**

Two things, and neither was what I was looking for.

The random retriever's trap rate came out at 0.333 against a corpus base rate of 0.36 — it recovered the
base rate, which is strong evidence the metric measures what it claims, and it gave BM25's 0.294 a
reference point: barely better than chance.

And the oracle scored 0.020 rather than 0.000, which sent me looking for why. The answer was a query I
had deliberately included where no authoritative answer exists — the old document is withdrawn, its
replacement is only a draft. So the metric's floor is 1/51, not zero, which changes how the headline
result should be read. I would not have known that without running the control.

> **"What was the worst measurement bug you have hit?"**

A recall function implemented positionally instead of as set overlap, on an earlier project. It scored a
provably exact algorithm at around 50%. The algorithm was correct; the metric was comparing element by
element, so any tie broken in a different order looked like an error. That is exactly the class of bug an
oracle control catches in one second, and it is why the test exists in this codebase:
`test_is_set_overlap_not_positional`.

## Debugging Tip

Add the controls to the *output*, not just the test suite. Every driver in this project prints them as
the first two rows of its results table:

```
Retriever     Recall@10    nDCG@10   trap@1
-------------------------------------------
Oracle            1.000      1.000    0.020
Random            0.000      0.000    0.333
BM25              0.863      0.671    0.294
```

With this, every single run re-validates the harness in front of you. If the Oracle row ever reads
anything but 1.000/1.000, you know before reading the rows you care about. The drivers even say so:

```
Oracle must read 1.000/1.000 and Random must read near zero. Those two rows
are the harness validating itself; BM25 is the only real measurement here.
```

## Summary

- A metric bug produces plausible numbers and is undetectable from anywhere else in the system.
- The defence is controls: an **oracle** returning exactly the ground truth (must score 1.000) and a
  seeded **random** retriever (must score near chance).
- Both must run through the *real* evaluation loop, which the `Protocol` design makes trivial and fast.
- Controls catch structural bugs; hand-computed values pin the arithmetic. At least one test must have a
  correct answer of zero, or `return 1.0` passes.
- Two real findings from the controls: random recovered the 36% corpus base rate as 0.333, validating
  the trap metric and giving BM25's 0.294 a reference point; and the oracle's 0.020 floor revealed that
  one gold query deliberately has no authoritative answer.
- Validation is free (0.102 s for 103 tests), so it is permanent, and the controls are printed in every
  driver's output.

## Key Takeaways

1. Validate the instrument before believing the measurement — with a positive *and* a negative control.
2. Include a case whose correct answer is zero, or the degenerate `return 1.0` passes.
3. Write expected values as formulas, not magic numbers.
4. Print your controls in every run, so the harness re-validates itself in front of you.

## Why the Next Chapter Exists

Part III is complete. We have the corpus, the thesis, a gold set with a control group, and a harness we
have reason to trust.

Part IV now reads the implementation module by module. Chapter 14 begins at the bottom — the corpus
layer, where 734 files become 19,763 chunks, and where a parsing error would silently poison everything
above it.
