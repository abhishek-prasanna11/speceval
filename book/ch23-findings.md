# Chapter 23 — The Findings, and How They Were Reached

## Learning Objectives

- State each finding with its measured support and its confidence.
- Distinguish the results that survive the small-n objection from those that do not.
- Explain why the central hypothesis was wrong, and where it *would* have held.
- Account for every prediction the project made, including the two that failed.
- Enumerate the limitations, including one found while writing this book.

## How to read this chapter

Every number here was produced by running the code, on 51 gold queries, while writing this book. Each
finding is graded:

> **Robust** — large relative to n, or expressed in a form that does not depend on rates.
> **Suggestive** — directionally consistent with a mechanism, but a few queries wide.
> **Anecdote** — too few observations to support a claim.

That grading is the point. A study that presents all its numbers with equal confidence has not
understood its own data.

## Finding 1 — Better retrieval did not buy better authority

**Robust.**

| | Recall@10 | nDCG@10 | rank-1 correct | trap@1 |
|---|---|---|---|---|
| BM25 | 0.863 | 0.671 | 47.1% | 0.294 |
| Dense | **0.971** | **0.801** | **60.8%** | 0.255 |

Dense is decisively the better retriever — 13.7 points of rank-1 accuracy.

Split by subset:

```
                  trap (n=20)              ordinary (n=31)
BM25         R 0.82  N 0.62  T 0.50    R 0.89  N 0.71  T 0.16
Dense        R 0.93  N 0.67  T 0.55    R 1.00  N 0.89  T 0.06
```

On ordinary queries dense cuts the authority error rate from 0.16 to 0.06. On trap queries it is
**0.55 versus 0.50** — 11 of 20 against 10 of 20.

The robust claim is not "dense is worse on traps": that is one query. It is that **a large, unambiguous
improvement in retrieval quality produced no improvement whatever where authority matters**, and that
over half of trap queries lead with a dead document regardless of which retriever runs.

**Why this is mechanism and not coincidence.** The two retrievers fail on *different* queries — nine
shared, four dense-only (q16, q22, q24, q47), six BM25-only (q02, q04, q15, q20, q28, q48) — and the two
sets have different shapes. Every dense-only failure is a near-exact semantic match to a superseded
title. BM25's failures are ordinary retrieval misses that happened to land on a non-authoritative
document; for q02, *"what does the walrus operator do"*, its top hit is about pattern matching. Same
metric, opposite causes. An aggregate cannot show that; the disagreement set can.

## Finding 2 — Hybrid did not beat dense at the retrieval level

**Suggestive.**

Recall 0.931 vs 0.971, nDCG 0.765 vs 0.801, rank-1 56.9% vs 60.8%. Hybrid sits *between* its components
rather than above them, contradicting the received wisdom that fusion beats either input.

Plausible mechanism: RRF weights both systems equally and has no way not to (chapter 6). Fusing a strong
ranker with a weak one at equal weight drags the strong one down. A weighted variant would fix it at the
cost of the calibration constant RRF was chosen to avoid.

Every gap here is one or two queries. The direction is consistent across three metrics, which is why
this is graded suggestive rather than anecdote — but it is not robust.

**And this finding is superseded by Finding 3**, which is the interesting part.

## Finding 3 — The retrieval metrics did not predict answer quality

**Robust, and the most transferable result in the study.**

Finding 2 concluded hybrid was the rung to discard. At the answer level:

| | superseded ↓ | authoritative ↑ | version ↑ |
|---|---|---|---|
| BM25 | 0.275 | 0.686 | 0.429 |
| Dense | 0.235 | 0.686 | 0.429 |
| **Hybrid** | 0.235 | **0.765** | **0.571** |

Hybrid is the **best** rung. Same corpus, same queries, opposite conclusion.

The cause is not mysterious, and that is what makes it useful. **Retrieval was scored on the top-10
distinct documents; generation consumes the top-5 chunks.** Two mismatches at once — unit and depth.
Fusion reorders chunks, and its chunk-level ordering packs better evidence into a five-chunk window even
where its document-level ranking at depth 10 is marginally worse. Recall@10 over documents cannot see
that.

The general form:

> **A retrieval metric is a proxy for downstream quality, and it is only as faithful as the match
> between what it scores and what the consumer consumes.**

Anyone tuning retrieval on nDCG@10 and shipping a generator over top-5 chunks is optimising the wrong
quantity. This is graded robust because it does not rest on a margin — it is a *reversal*, and the
mechanism is structural rather than statistical.

It also produced a design rule for the rest of the project: rung 4 is judged on citations, not nDCG,
because the retrieval metrics have demonstrably ranked the rungs wrongly once.

## Finding 4 — Roughly one answer in four cited a dead specification

**Robust.**

23.5%–27.5% of all answers, rising to **40%–50% on trap queries** and falling to 10%–13% on ordinary
ones. A ~4× gap between subsets — exactly what the trap/ordinary split was built to expose.

Representative, measured verbatim:

> **q23** — *"how are Python package version numbers compared"* → *"According to PEP 386, Python package
> version numbers are compared using the standard schema specified in that…"*

PEP 386 is `Superseded` by 440. No hedge, no qualification.

## Finding 5 — No hallucinated citations at all

**Robust.**

**0 of 153 answers** cited a PEP number absent from the corpus. Every citation resolved to a real
document.

This matters because it *localises* the problem. Grounding is not the failure mode here; authority is.
The model does not invent sources — it faithfully cites dead ones, which is a considerably harder failure
to notice.

(Chapter 22 found a related gap this metric does not cover: an answer citing a real, live PEP that was
never in its context. Neither the status check nor the existence check sees that. Listed under
limitations.)

## Finding 6 — There is no recall/authority tradeoff

**Robust. This is the headline, and it contradicts the hypothesis the project was built to test.**

The premise, stated in the README before any measurement: filtering hard on authority must eventually
starve the retriever of legitimate context. The drafted résumé bullet said *"at a measured cost of X%
recall."*

The measured sweep:

```
  lambda   Recall@10   nDCG@10   trap@1   trap@1 (trap set)
------------------------------------------------------------
       0       0.951     0.771    0.294               0.550
    0.01       0.951     0.771    0.294               0.550
    0.02       0.951     0.803    0.137               0.350
     0.1       0.971     0.831    0.059               0.150
    0.25       0.980     0.856    0.020               0.050
     0.5       0.980     0.852    0.000               0.000
       1       0.961     0.840    0.000               0.000
```

Between λ=0 and λ=0.5: **trap@1 goes 0.294 → 0.000 while Recall@10 rises 0.951 → 0.980 and nDCG rises
0.771 → 0.856.** Everything improves together. Recall turns back down only at extreme strength (0.980 →
0.961, one query), and even λ=1 beats the baseline on both retrieval metrics.

At the answer level, same pattern:

| λ | superseded ↓ | authoritative ↑ | trap subset ↓ |
|---|---|---|---|
| 0 | 0.157 | 0.765 | 0.300 |
| 0.25 | 0.078 | 0.824 | 0.150 |
| **1** | **0.039** | **0.863** | **0.050** |

In counts: superseded citations **8 of 51 → 2 of 51**. Per query: **6 fixed, 0 broken.**

### Why the premise was wrong

The mental model was that authority reranking *discards* documents. It does not — it reorders a fixed
candidate pool.

And supersession pairs are topically near-identical. Whatever retrieved PEP 563 also retrieved PEP 649,
because they are about the same thing in the same words. **The intervention swaps a dead document for
its live twin rather than trading coverage for correctness.**

Chapter 22 shows this concretely: at λ=1 the pool still contained the same candidates; four of five
returned chunks became PEP 649 because PEP 563's weight collapsed by a factor of 44.

### Where the premise *would* have held

This mechanism bounds the claim, which is more valuable than the claim itself. The result should **not**
generalise to a corpus where:

- superseded documents have no live successor (nothing to promote in their place), or
- the successor is worded differently enough to fall outside the candidate pool, or
- authority is genuinely correlated with irrelevance rather than orthogonal to it.

This corpus is none of those. A corpus of retracted papers with no replacements would behave differently,
and the study cannot speak for it.

### The honest reading of "0.000"

trap@1 reaching 0.000 is *better than the oracle*, whose floor is 0.020 (chapter 13). Not a
contradiction: the oracle returns gold labels in sorted numerical order, so on q30 it leads with the
withdrawn PEP 543; the reranker is free to put the draft PEP 748 first. It is a reminder that a metric's
floor depends on the system being measured.

## Finding 7 — Partial reranking is worse than none

**Suggestive, with a mechanism.**

| | λ=0 | λ=0.05 | λ=0.25 |
|---|---|---|---|
| superseded-citation | 0.157 | **0.235** | 0.078 |
| authoritative-citation | 0.765 | **0.725** | 0.824 |

Non-monotonic. λ=0.05 is worse than doing nothing on both metrics, reversing cleanly by λ=0.25.

The mechanism follows from chapter 20's threshold arithmetic. At λ=0.05 the knob is past the ~0.0165
where adjacent pairs flip, but well short of the ~0.25 needed to promote a live document from deeper in
the pool. **Enough force to disturb the ordering, not enough to repair it.**

λ=0.05 is 12 of 51 answers against λ=0's 8 — four answers. Suggestive, not robust. But the direction is
consistent across two metrics and it has a derived explanation, and the practical warning stands: a
half-tuned authority reranker can be worse than none.

**It is only visible because the sweep grid was fine at the low end.** A uniform 0.25-step grid would
have reported clean monotonic improvement and missed a regime where the intervention actively hurts.

## Finding 8 — The naive version rule is harmful

**Robust for its size, and the one prediction that held.**

Penalise any PEP whose `Python-Version` postdates the version asked about:

| variant | superseded | authoritative | version-correct |
|---|---|---|---|
| λ=1 | 0.039 | **0.863** | **0.714** |
| λ=1 + version penalty | 0.039 | 0.804 | 0.571 |

Authority unchanged; both other metrics worse — including the version metric it was meant to improve.

The reason was argued in advance and is simple: answering *"no, that arrived in 3.8"* requires
retrieving the very document the rule demotes. **Version metadata tells you what the answer must say,
not which document to trust.**

## The prediction scorecard

Three substantive predictions. Two were wrong.

| Prediction | Outcome |
|---|---|
| Dense beats BM25 on rationale queries, **loses** on identifier queries | **Half wrong.** Direction held (widest gap on rationale, +0.17 nDCG) but dense won identifier too (+0.09) |
| Authority reranking will **cost recall** | **Wrong.** Recall improved across the useful range |
| The naive version rule will **hurt** | **Right.** Confirmed by ablation |

Both failures are recorded in the notes as failures. A prediction that only counts when it succeeds is
not a prediction, and the 1-in-3 hit rate is the argument for having built the measurement apparatus
before forming opinions.

The identifier-query miss has two untested candidate explanations: `indexed_text` prepends the PEP title
and section, giving the embedding a clean signal for identifier-style questions; and `nomic-embed-text`
saw code and technical prose in training, so `__future__`-style tokens may be less alien than assumed.
Chapter 22 supplies indirect support for the first — title prepending demonstrably dominates dense
retrieval's behaviour on q16.

## Limitations

Ordered by how much they should change your reading.

**n = 51.** The headline is 8 answers → 2. Real, but three fewer fixes would halve it. Several smaller
comparisons are one or two queries and are reported as not findings. *The 6-fixed/0-broken form is the
most robust statement of the main result because it does not depend on rates.*

**The status weights are hand-chosen judgements.** `Draft = 0.55`, `Deferred = 0.25`,
`SUPERSEDED_BY_FACTOR = 0.15`. No sensitivity analysis was run. An unknown share of Finding 6 may be
attributable to those constants rather than to the idea. **This is the study's largest untested
assumption**, and the single highest-value thing to address next.

**One embedding model, one generation model, one prompt, one temperature.** Whether the
superseded-title attraction is a property of embeddings generally or of `nomic-embed-text` is untested.

**The answer metrics measure citation hygiene, not answer truth.** Three identified cases where a
correct answer scores as a failure or a failure escapes scoring:

- **q47** — right conclusion (*"No, zoneinfo is not available in 3.8"*) drawn from a `Superseded` source.
  Scores as a failure.
- **q30** — a correct hedge (*"There is no definitive answer"*) citing a `Withdrawn` document. Scores as
  a failure.
- **q16 at λ=1** — cites PEP 634, which is real and `Final` but was **never in the context**. Scores as
  a success. Nothing checks groundedness. *Found while writing chapter 22.*

All three argue for the validated LLM-as-judge the project deliberately did not build.

**Categories are unbalanced** — 21 availability, 15 identifier, 15 rationale — because the six
version-scoped queries added later are all availability questions.

**The version metric rests on 7 queries.** Its apparent peak at λ=0.05 (0.857) is one answer. Anecdote.

**Deterministic, but not robustness-tested.** Temperature 0 with a fixed seed gives byte-identical runs,
so there is no variance to report — which is not the same as having tested sensitivity to prompt
wording or seed.

**One discrepancy found during the writing of this book.** The README reported retrieval numbers measured
when the gold set held 45 queries, before six version-scoped queries were added; those numbers were no
longer reproducible from the repository. They were re-measured on the current 51-query set and corrected,
and `notes/phase2.md` keeps its original figures with a re-measured addendum. Every conclusion survived;
the subset inversion in Finding 1 sharpened slightly from a tie to 11-versus-10.

## What the study establishes

1. In a corpus where documents supersede one another, retrieval optimised for relevance systematically
   returns superseded documents — measured here at roughly one answer in four, and about one in two on
   queries where a predecessor exists.
2. Improving retrieval quality does not fix this and may sharpen it, because semantic strength locks onto
   near-duplicate dead documents.
3. Reranking on metadata the corpus already carries fixes most of it — 6 queries fixed, 0 broken — and on
   *this* corpus costs nothing, because the replacement document is already in the candidate pool.
4. The obvious version refinement is harmful.
5. Retrieval metrics can rank strategies in the opposite order from answer-level metrics when the metric's
   unit and depth do not match what the generator consumes.

## Key Takeaways

1. Grade your findings. Robust, suggestive, anecdote — and say which is which next to the number.
2. State the mechanism, because the mechanism is what bounds the claim.
3. Record failed predictions. Two of three here were wrong, and that is the argument for measuring.
4. Report the form of a result that survives your weakest assumption — "6 fixed, 0 broken" over a rate.

## Why the Next Chapter Exists

The findings are stated and bounded. Chapter 24 converts all of it into interview-ready form: every
design decision as decision → reason → cost, the real bugs with what each one taught, a rapid-fire sheet,
and a question bank organised by topic.
