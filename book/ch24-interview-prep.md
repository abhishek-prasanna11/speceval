# Chapter 24 — Interview Preparation

## How to use this chapter

Three sections, in order of what actually gets asked.

1. **Design decisions** as decision → reason → cost. Interviewers probe the *cost*; a decision with no
   stated cost reads as unexamined.
2. **The bugs**, because "tell me about a bug you found" is the highest-signal question in a technical
   interview and a real one beats a rehearsed one.
3. **The question bank**, organised by topic.

There is also a rapid-fire sheet at the end for the numbers you should be able to produce without
thinking.

One framing point before the details. This project is not "I built a RAG system." It is **"I
investigated the tradeoff between retrieval recall and version/authority correctness in
technical-document retrieval."** The first invites "so did everyone"; the second names a tension and
invites a conversation. Lead with the tension.

---

## Part 1 — Design decisions

### Corpus: Python PEPs

**Decision.** 734 Python Enhancement Proposals rather than RFCs, Kubernetes KEPs, W3C specs or CVE
advisories.

**Reason.** The study needs documents that record their own standing. PEPs carry a nine-value `Status`
enum, a `Python-Version` field making correctness version-conditional, and explicit
`Superseded-By`/`Replaces` edges — so hard test cases can be *generated* from metadata rather than
invented.

**Cost.** Results are corpus-specific. The headline depends on supersession pairs being topically
near-identical, which is true here and would not be for, say, retracted papers with no replacements.

**If pushed:** *"I first picked KEPs, on the wrong premise that PEP authority was a flat status flag.
That was wrong — PEPs also have version and supersession edges — and correcting it flipped the choice.
PEPs then won on prose quality: KEP template boilerplate repeats near-verbatim across hundreds of
documents and would have confounded a 51-query study."*

### BM25 implemented, embedding model imported

**Decision.** Hand-rolled BM25 (40 lines); pretrained `nomic-embed-text` via a local server.

**Reason.** Own the components whose behaviour you will have to explain. One of the findings depends on
explaining why BM25 ranked the live spec above the superseded one while the embedding did the opposite —
which required printing per-term idf contributions.

**Cost.** Reimplementation risk, mitigated by five tests including one that fails if idf is removed.

### RRF rather than weighted score fusion

**Decision.** Fuse ranks, not scores. `Σ 1/(60 + rank)`.

**Reason.** BM25 scores are unbounded and corpus-dependent; cosine sits in `[-1, 1]`. Adding them needs
a calibration constant, which would sit unexamined under every reported result.

**Cost.** RRF cannot express "trust dense more", and dense was the stronger ranker — which is the likely
reason hybrid landed between its components at the retrieval level.

### Brute-force search, no ANN index

**Decision.** Exact cosine over 19,763 × 768.

**Reason.** 58 MB, a few milliseconds. And an approximate index adds its *own* error, so a missed
document could no longer be attributed to the ranking rather than the index.

**Cost.** Does not scale past ~1M vectors. Irrelevant here, disqualifying at production scale.

### Document-level relevance labels

**Decision.** Ground truth is a set of PEP numbers, not chunk IDs.

**Reason.** Chunk-level labelling costs several times more and would not change which strategy wins.

**Cost.** Loss of resolution — and it was paid. Retrieval was scored on documents at depth 10 while the
generator consumed chunks at depth 5, and the two disagreed about the best strategy.

### Answer metrics from metadata, no LLM judge

**Decision.** Citation checks are dictionary lookups against PEP headers.

**Reason.** Kept the project small and removed a component that would itself need validating.

**Cost.** Measures citation *hygiene*, not answer truth. Three identified failure cases (below).

### The prompt hides `Status`

**Decision.** Context shows PEP number, title, section, text — never status or version.

**Reason.** If the generator could see "Superseded" it could route around bad retrieval, and the measured
citation rate would reflect the model's caution rather than the retriever's ranking. Every rung would
improve by an unknown amount and the comparison would measure nothing.

**Cost.** Cannot answer "would telling the model be as good as reranking?" — a genuinely interesting
separate experiment, listed as future work. A test enforces the omission.

### 39% trap queries, 61% control

**Decision.** Deliberately mixed gold set.

**Reason.** An intervention evaluated only on queries it was designed to win has no measurable cost.

**Cost.** Fewer trap queries means the headline rests on 20 rather than 51.

### Tunable strength rather than on/off

**Decision.** `final = base × (1 − λ + λ·weight)`, swept.

**Reason.** A boolean gives two points and supports only "it helped". A knob gives a curve.

**Cost.** More runs. It paid for itself twice — it produced the no-tradeoff result *and* the
partial-reranking regression, neither visible from two points.

### Temperature 0, fixed seed

**Decision.** Greedy decoding, seed 7, verified byte-identical across three runs.

**Reason.** Run-to-run variance is zero, so differences between rungs are attributable to retrieval.

**Cost.** Not robustness-tested against prompt or seed variation. Determinism is not stability.

---

## Part 2 — The bugs

Five real ones. Each has the same useful shape: *what it was, why nothing caught it, what it taught.*

### 1. `build_context` returned an empty string

**What.** If the first chunk exceeded the character budget, the loop broke with nothing collected and
returned `""`.

**Why it was dangerous.** No crash. The model answered from parametric memory, fluently, and the numbers
looked like *a retrieval failure* — sending you to debug the retriever.

**Found by.** A test checking the budget was respected, using `max_chars=60`. It cannot trigger at the
production budget.

**Taught.** A test of an ordinary property finds bugs in extraordinary cases. And: when a bug's failure
mode is "wrong numbers that look like a different component's fault", it is worth a guard even if
unreachable today.

### 2. Spurious `divide by zero` from `matmul`

**What.** NumPy 1.26.4 on Apple Accelerate raised FP warnings from an operation containing no division,
for matrices above ~64 rows.

**How it was resolved.** Checked the data (no NaN, norms all 1.0), then **computed the same result in
float64 and compared** — max deviation 9.7e-08, identical top-20 ordering, self-similarity exactly 1.0.
Then found the boundary at 64 rows, which identified a blocked BLAS kernel setting flags on padding
lanes. Suppressed narrowly with `np.errstate` around one statement, with the reasoning in the code and
four version-independent tests pinning the arithmetic. Absent on NumPy 2.5.2, confirming an upstream
bug.

**Taught.** The procedure: read the warning literally, check the data, *compute it a different way*,
find the boundary, fix narrowly with a test that would catch the real fault. Step three is the only one
that converts suspicion into conclusion.

### 3. `python_version` meant two different things

**What.** On q01 it held the version *asked about* (3.9, where the answer is 3.10); elsewhere the version
of the *answer*.

**Why it was dangerous.** Any metric reading it would compare incomparable things, silently.

**Fixed by.** Renaming to `asked_version` with one meaning, enforced in *both* directions by the
verifier — the field must appear in the query text, and a query naming a version must set the field.
Only one query then qualified, so six version-scoped queries were added to make the metric testable.

**Taught.** A field whose meaning drifts is worse than a missing field. Enforce meaning in a check, not
a comment.

### 4. The wrong baseline (caught before it mattered)

**What.** The obvious baseline for the sweep was the earlier hybrid number. But the reranker draws a
10× deeper pool, and RRF over a deeper pool promotes different chunks.

**How it was sized.** The answer cache keys prompts by content, so identical prompts are hits. At λ=0:
**1 cache hit in 51**. Fifty of 51 queries saw different context from pool depth alone — and that effect
accounted for roughly half the apparent gain.

**Taught.** A baseline must differ from the treatment in exactly one respect. And instrumentation built
for one purpose (reporting cost) often answers a more important question.

### 5. Stale numbers in the project's own documentation

**What.** The README reported retrieval figures measured at n=45, before six queries were added. They
were no longer reproducible from the repository.

**Found by.** The verification pass while writing this book — re-running the drivers and noticing the
composition line said 51.

**Taught.** Numbers in documentation decay silently when the thing they measure changes. Re-run before
you publish; every conclusion survived, but the figures had to be corrected.

---

## Part 3 — Rapid-fire sheet

Numbers you should produce without hesitation.

| | |
|---|---|
| Corpus | 734 PEPs → 19,763 chunks (26.9 each) |
| Non-authoritative | 262 PEPs, **36%** — the trap surface |
| Gold set | 51 queries; 21/15/15 by category; **20 traps (39%)**; 48 distinct PEPs |
| Tests | **103**, run in 0.10 s |
| Best retriever | Dense: Recall@10 **0.971**, nDCG **0.801**, rank-1 **60.8%** |
| BM25 | 0.863 / 0.671 / 47.1% |
| Baseline citation error | **~1 in 4** answers cite a dead spec; ~1 in 2 on traps |
| After reranking | **8 of 51 → 2 of 51**; **6 fixed, 0 broken** |
| trap@1 sweep | 0.294 → **0.000**, while Recall@10 *rose* 0.951 → 0.980 |
| Hallucinations | **0 of 153** |
| Random control | trap@1 0.333 ≈ corpus base rate 0.36 |
| Oracle floor | trap@1 0.020, not 0 (q30 has no authoritative answer) |
| Retrieval vs generation | ~40 ms vs ~12 s — **~300×** |

Three sentences to have ready:

> **The problem.** Retrieval optimises relevance; users need authority. In a corpus where documents
> supersede one another those come apart, because the superseded document and its replacement are
> near-identical in topic and vocabulary — and the dead one often has the cleaner title.

> **The finding.** Better retrieval did not fix it. Dense beat BM25 by 13.7 points of rank-1 accuracy
> and gained nothing on the queries where supersession exists.

> **The result.** Reranking on metadata the corpus already carries cut dead-spec citations from 8 of 51
> answers to 2, fixing six queries and breaking none — and cost nothing, because the replacement was
> already in the candidate pool.

---

## Part 4 — Question bank

### Retrieval fundamentals

**Explain BM25.** Three ideas: idf weights rare terms (which is also why you don't need a stopword
list); `f/(f + k₁·…)` saturates repeated occurrences with `k₁` setting the knee; length normalisation
discounts by length *relative to the corpus mean*, with `b` setting how hard and `b=0` disabling it.

**What is an embedding, and why cosine?** A learned map from text to a fixed-length vector where
semantic similarity is geometric proximity, resting on the distributional hypothesis. Cosine because
magnitude in embedding space tracks incidental properties like passage length; normalise once at build
time and a dot product *is* cosine.

**When would you pick BM25 over embeddings?** Exact tokens carrying meaning — identifiers, error codes,
version strings — plus explainability, no GPU, a constantly-changing corpus, or no in-domain training
data. And it is cheap enough to run alongside rather than instead of.

**How do you combine them?** Ranks, not scores — the scales are incomparable and any calibration
constant becomes an untested parameter under every result. RRF: `Σ 1/(60 + rank)`. The 60 flattens the
top so agreement matters more than being first in one list.

**Does hybrid always win?** No. Here it landed *between* its components at the retrieval level, because
RRF weights an unequal pair equally. It was the best rung at the answer level — which is a separate and
more interesting story.

### Evaluation

**How do you evaluate retrieval?** Fixed queries with human relevance judgements; Recall@K for whether
it was found, nDCG@K for whether it was ranked well; always state K. Then validate the metric itself.

**How do you know your metric is right?** Controls. An oracle returning ground truth must score 1.000 —
if not, the metric is broken with no other explanation. A seeded random retriever must score near
chance — if not, something leaks the answer. Both through the *real* evaluation loop. Plus hand-computed
values, including at least one case whose answer is zero, because the degenerate implementation of any
metric is `return 1.0`.

**Did the controls catch anything?** Two things, neither expected. Random's trap rate came out 0.333
against a corpus base rate of 0.36 — it recovered the base rate, validating the metric and reframing
BM25's 0.294 as barely better than chance. And the oracle scored 0.020 rather than 0, which led to a
gold query where no authoritative answer exists, so the metric's floor is 1/51.

**Worst measurement bug you have hit?** Recall implemented positionally instead of as set overlap, on an
earlier project. It scored a provably exact algorithm at ~50% — the algorithm was fine, the metric was
comparing element by element. That is why `test_is_set_overlap_not_positional` exists in this codebase.

**How do you build a gold set?** Document-level labels (chunk-level costs several times more and does
not change which strategy wins); hard cases *generated* from the corpus's supersession edges rather than
invented; and only 39% traps, because an intervention evaluated where it cannot lose has no measurable
cost.

**Weakest part of your evaluation?** n=51. The headline is 8 answers → 2, which is real, but several
smaller comparisons are one or two queries and I report those as not findings. The version metric rests
on 7 queries — anecdote.

### The findings

**What did you actually find?** Retrieval optimises relevance, users need authority, and in an evolving
corpus those come apart. About one answer in four cited a superseded specification.

**Would a better model fix it?** No — measured. Dense beat BM25 by 13.7 points of rank-1 accuracy and
gained *nothing* on the trap subset. Its unique failures were all near-exact semantic matches to
superseded titles. Authority is in a header field and a graph edge; no amount of reading the prose
recovers it.

**How do you know that is the mechanism and not coincidence?** The disagreement set. Nine trap failures
were shared, four dense-only, six BM25-only — and the two sets have different shapes. Dense's are
semantic locks onto dead titles; BM25's are ordinary retrieval misses that happened to land on a
non-authoritative document. Same metric, opposite causes. Only the disagreement shows that.

**What was the cost of your fix?** Nothing, and that surprised me — I designed the study to measure a
tradeoff that turned out not to exist. Between λ=0 and 0.5, trap@1 went to zero while Recall@10 *rose*.
The premise assumed reranking discards documents; it reorders a fixed pool, and supersession pairs are
near-identical, so it swaps a dead document for its live twin.

**Where would your result not hold?** Corpora where superseded documents have no live successor, or
where the successor is worded differently enough to fall outside the pool. The mechanism bounds the
claim.

**Anything you expected to work that didn't?** The version rule — distrust documents newer than the
version asked about. It is wrong, because answering *"no, that arrived in 3.8"* requires the very
document the rule demotes. I implemented it behind a flag to measure rather than assert: authority
unchanged, both other metrics worse.

**Anything counterintuitive in the sweep?** Partial reranking is worse than none. λ=0.05 regressed on
both answer metrics before reversing by 0.25 — past the threshold where adjacent ranks flip, short of
what is needed to promote a live document from deeper in the pool. Only visible because the grid was
fine at the low end.

### Systems and engineering

**Why time retrieval and generation separately?** They differ by ~300×. A combined figure shows three
identical numbers dominated by the model's variance and hides the only difference retrieval has.

**How do you make LLM output reproducible?** Temperature 0 plus a fixed seed, then *verify* by hashing
repeated calls — three byte-identical runs here. And know it is a property of the configuration: enable
sampling and you need a variance estimate before any difference means anything.

**How do you cache an expensive precomputation?** Key on everything that determines the output — for
embeddings: model, prefix, and every input text. Write a metadata sidecar so the artefact is
interpretable later. And know what the cache does to your measurements: end-to-end latency on a warm
cache is a disk read.

**Why does your test suite run in 0.1 seconds if the system needs two neural models?** `typing.Protocol`.
Six retrievers satisfy one structural contract with no inheritance, so the oracle is nine lines and the
hybrid stub is fourteen. Nothing in the suite touches a model server.

**A library emits a warning you do not understand. What now?** Read it literally, check the data, then
compute the same result a different way and compare — that is the step that decides whether your numbers
are affected. Find the boundary, because a threshold identifies the mechanism. Fix narrowly, record the
reasoning in the code, and add a test that would catch the real fault despite the suppression.

### Judgement

**What would you do next, with one more week?** A sensitivity analysis on the status weights. They are
hand-chosen judgements with no validation, so an unknown share of the headline may be attributable to
the constants rather than the idea. That is the largest untested assumption and it is cheap to attack.

**And with a month?** Expand the gold set to ~150 queries and build a validated LLM judge — validated
meaning measured against my own labels with a κ before I trust it. I have three concrete cases where the
automatic metric is wrong: a correct conclusion from a dead source, a correct hedge, and a citation to a
document that was never in the context.

**What would you do differently?** Match the retrieval metric's unit and depth to what the generator
consumes, from the start. Scoring documents at depth 10 while feeding chunks at depth 5 is what made two
phases disagree about the best strategy. It was informative, but it was an accident.

---

## A closing note on how to talk about this

The strongest thing here is not the headline number. It is that **two of three predictions were wrong
and both are written down as wrong.**

The temptation in an interview is to present a clean narrative: I hypothesised X, measured X, found X.
That narrative is weaker, not stronger, because anyone experienced knows real measurement does not work
that way — and because the wrong predictions are where the mechanism was actually discovered. The
tradeoff hypothesis failing is what forced the explanation of *why* it failed, and that explanation is
what bounds the result and makes it worth anything.

Lead with the tension, give the number, volunteer the caveat before it is asked for, and have the
mechanism ready. The caveat volunteered reads as rigour; the same caveat extracted reads as
overclaiming.
