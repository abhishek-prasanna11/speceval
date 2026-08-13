# Chapter 3 — Lexical Ranking: Counting Words Properly

## Learning Objectives

- Show why raw term frequency is a bad relevance score, with concrete counterexamples.
- Derive **inverse document frequency** from the question "which words are informative?"
- Explain **term-frequency saturation** and **length normalisation**, and why each is needed.
- Read the BM25 formula and say what every symbol and both constants do.
- Read `speceval/bm25.py` line by line and explain each decision.

## Motivation

We have an inverted index of counts. The obvious score is to add up how often the query's words
appear in each document. This is wrong in two independent ways, and repairing both — carefully,
one at a time — produces BM25, the formula that was the state of the art for two decades and is
still the baseline every new retrieval method must beat.

BM25 is worth deriving rather than importing. It is only two ideas, both of which you would arrive
at yourself given the counterexamples, and understanding it is what lets you reason about *why* it
wins on some queries in this project and loses on others.

## First Principles

### Attempt 1: sum of term frequencies

```
score(d, q) = Σ  tf(t, d)
             t∈q
```

For each query term, add how many times it occurs in the document.

**Counterexample A — the common word.** Query: *"the walrus operator"*. In a corpus of Python
documents, `the` appears hundreds of times in every document and `walrus` appears in maybe three.
A long document containing `the` 400 times and `walrus` zero times scores 400. The document that
actually defines the walrus operator, containing `walrus` 5 times and `the` 100 times, scores 105.

The wrong document wins by a factor of four. The problem: **we are treating all words as equally
informative when they are not.**

**Counterexample B — the long document.** Two documents both genuinely about the walrus operator.
One is 200 words and mentions it 5 times. The other is 20,000 words and mentions it 20 times. The
long one scores four times higher, but the short one is far more *concentrated* on the topic.

The problem: **length is being rewarded for its own sake.**

Fix these two, and you have BM25.

### Fix 1: weight words by how rare they are

Intuition: a word appearing in every document tells you nothing about which document to pick. A
word appearing in three documents out of 734 is enormously informative — it nearly answers the
question by itself.

Define **document frequency** `df(t)` = the number of documents containing term `t`. We want a
weight that is large when `df` is small and small when `df` is large. The simplest such thing is
`N / df(t)`, and taking a logarithm keeps it from exploding:

```
idf(t) = log( N / df(t) )
```

This is **inverse document frequency**. Concretely in this corpus:

| term | roughly how many chunks contain it | idf | contribution |
|---|---|---|---|
| `python` | almost all 19,763 | near 0 | negligible |
| `annotations` | a few hundred | moderate | meaningful |
| `walrus` | a handful | large | dominant |

The word `the` now contributes essentially nothing, automatically, with no stopword list to
maintain. This is why modern lexical retrieval does not need to delete common words — it weights
them into irrelevance.

The version actually used adds smoothing to avoid a division by zero and to keep the value
positive:

```
idf(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
```

This is exactly what `speceval` computes:

```python
self.idf = {
    term: math.log((self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0)
    for term, posting in self.postings.items()
}
```

Note `len(posting)` *is* `df(t)` — the postings list for a term has one entry per document
containing it, so its length is the document frequency. The inverted index already stores what we
need.

### Fix 2a: saturate term frequency

Fixing document length naively (divide by length) turns out to interact badly with another problem,
so deal with the second problem first.

Is a document mentioning `walrus` 20 times *twenty times* more relevant than one mentioning it
once? Clearly not. The first mention is highly informative — it establishes the document is about
this. The second adds a little. The twentieth adds almost nothing.

We want a function that rises steeply at first and then flattens. The standard choice:

```
                f
   saturate(f) = -------
                f + k₁
```

where `k₁` is a constant controlling how fast the flattening happens. With `k₁ = 1.5`:

```
  f = 1   ->  1/(1+1.5)   = 0.40
  f = 2   ->  2/(2+1.5)   = 0.57      (+0.17 for the 2nd occurrence)
  f = 5   ->  5/(5+1.5)   = 0.77      (+0.20 for occurrences 3-5)
  f = 20  ->  20/(20+1.5) = 0.93      (+0.16 for occurrences 6-20)
  f = ∞   ->  1.0                     (hard ceiling)
```

Occurrence one buys 0.40. Occurrences six through twenty, all fifteen of them, buy 0.16 between
them. That is the shape we wanted.

```
  saturate(f)
    1.0 |                       ___________________
        |              ______/
        |         ___/
    0.5 |      _/
        |    /
        |  /
    0.0 +------------------------------------------  f
        0    2    5        10             20
```

**`k₁` controls the knee.** Small `k₁` saturates almost immediately (nearly binary: does the word
appear at all?). Large `k₁` stays nearly linear (raw counts). `1.5` is the conventional middle,
and it is what `speceval` uses.

### Fix 2b: normalise for length

Now handle length. The insight that makes this work: we do not want to penalise *long* documents,
we want to penalise documents that are long **relative to the corpus average**. A document of
average length should be unaffected.

So compare each document's length to the mean, and fold that ratio into the saturation constant:

```
                        f
   tf_component = ---------------------------
                  f + k₁ · (1 - b + b · |d|/avgdl)
```

Read the bracket carefully — it is the whole idea:

- If `|d| = avgdl` (average length), the bracket is `1 - b + b = 1`, and this reduces exactly to
  the previous formula. Average documents are untouched.
- If `|d| > avgdl`, the bracket exceeds 1, the denominator grows, and the score shrinks.
- If `|d| < avgdl`, the bracket is below 1 and the score grows.

**`b` controls how aggressively.** At `b = 0` the bracket is always 1 and length is ignored
entirely. At `b = 1` normalisation is full. `0.75` is the conventional compromise and is what
`speceval` uses.

### Putting it together: BM25

```
                  ┌                                              ┐
                  │             f(t,d) · (k₁ + 1)                │
score(d,q) =  Σ   │ idf(t) · ─────────────────────────────────── │
             t∈q  │          f(t,d) + k₁·(1 - b + b·|d|/avgdl)   │
                  └                                              ┘
```

The `(k₁ + 1)` in the numerator is a scaling convenience — it makes a single occurrence in an
average-length document score close to `idf(t)`. It does not change the ranking, only the
magnitude.

Three ideas, that is all:

1. **`idf(t)`** — rare words matter more.
2. **`f/(f + k₁·…)`** — repeated occurrences saturate.
3. **`b·|d|/avgdl`** — long documents are discounted relative to average.

BM25 stands for "Best Match 25", the twenty-fifth in a series of experiments by Robertson and
Spärck Jones. The name records that it was found empirically, not derived from theory. *(This
formulation is standard from the information-retrieval literature; it is the one external fact in
this chapter not derived from the project's own code.)*

## Mental Model

Think of scoring a job application against a role description.

**idf** — a candidate mentioning "Kubernetes" tells you more than one mentioning "teamwork",
because everyone says teamwork. Rare claims are informative.

**Saturation** — saying "Python" once establishes it. Saying it forty times does not make them
forty times better at Python; it makes them repetitive.

**Length normalisation** — a twelve-page CV will mention more keywords than a one-page CV by sheer
volume. You are judging density of relevance, not word count.

## Deep Explanation: the implementation

`speceval/bm25.py` is 93 lines including documentation. Here is the whole mechanism.

### Building the index

```python
class BM25:
    """An inverted index over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)

        # term -> list of (doc_index, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: list[int] = []

        for doc_index, document in enumerate(documents):
            tokens = tokenize(document)
            self.doc_lengths.append(len(tokens))
            frequencies: dict[str, int] = defaultdict(int)
            for token in tokens:
                frequencies[token] += 1
            for term, frequency in frequencies.items():
                self.postings[term].append((doc_index, frequency))

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        )
```

Everything the formula needs is precomputed here: postings lists, per-document lengths, the corpus
average, and the idf table. Note `if self.n_docs else 0.0` — an empty corpus would otherwise divide
by zero on construction.

### Scoring a query

```python
    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)

        for term in tokenize(query):
            posting = self.postings.get(term)
            if posting is None:
                continue
            idf = self.idf[term]
            for doc_index, frequency in posting:
                norm = 1.0 - self.b + self.b * (
                    self.doc_lengths[doc_index] / self.avg_doc_length
                )
                scores[doc_index] += idf * (
                    frequency * (self.k1 + 1.0) / (frequency + self.k1 * norm)
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]
```

Four details worth naming.

**The loop order is term-major, not document-major.** The outer loop is over query terms; the inner
loop walks that term's postings list. This is exactly the access pattern the inverted index was
built for. A document containing no query term is never touched — it never enters `scores` at all.

**`scores` is a `defaultdict(float)`**, so accumulating into an unseen document needs no
`if key in dict` check. The score for a document is built up incrementally, one query term at a
time.

**A query term absent from the corpus is skipped**, not treated as a zero-score match. Consequence:
a query of entirely unknown words returns an empty list rather than an arbitrary ranking. Pinned by
a test:

```python
def test_unknown_term_returns_nothing(self) -> None:
    index = BM25(["alpha beta", "gamma delta"])
    self.assertEqual(index.search("zzzzz", top_k=5), [])
```

**Ties break deterministically.** The sort key is `(-score, doc_index)`, so equal scores are ordered
by document index rather than by whatever order the dictionary happened to produce. In a study, a
nondeterministic tiebreak would make results unreproducible across runs for no reason. This one
line is why the retrieval numbers in this project are byte-stable.

### Why implemented rather than imported

There is a perfectly good `rank_bm25` package. The module docstring gives the reasoning:

```python
"""BM25 -- the lexical baseline (rung 1 of the strategy ladder).

Implemented directly rather than pulled in as a dependency: it is about forty lines,
and the lexical baseline is the one component whose behaviour needs to be inspectable
when the results look surprising.
"""
```

This is a scoped argument, not a blanket preference for writing everything yourself. The same
project happily uses a pretrained embedding model in chapter 18, because reimplementing that would
teach nothing relevant to the study. The rule being applied is: **own the components whose
behaviour you will need to explain.** BM25's behaviour is central to the findings, so it is owned.

### Verified behaviour

Four properties are pinned by tests. The most instructive:

```python
def test_rare_term_outranks_common_term(self) -> None:
    # idf must make "walrus" worth more than "python", which is in every document.
    index = BM25(
        ["python python python", "python walrus", "python python", "python"]
    )
    hits = index.search("python walrus", top_k=4)
    self.assertEqual(hits[0][0], 1)
```

Document 0 contains `python` three times and nothing else. Document 1 contains `python` once and
`walrus` once. The query asks for both. Document 1 wins, because `walrus` appears in one of four
documents (high idf) while `python` appears in all four (idf near zero). This test would fail if
idf were removed, and it directly encodes counterexample A from the start of this chapter.

## Systems Perspective

BM25 is memory-bound. The arithmetic per posting is a handful of floating-point operations; the
expensive part is chasing pointers through postings lists that do not fit in cache.

Two practical consequences.

**Query cost is proportional to postings length, not corpus size.** A query of rare terms is fast
even on a huge corpus. A query containing `python` in this corpus touches nearly every chunk.

**Real search engines compress postings lists** (delta encoding, variable-byte integers) because
the bottleneck is bytes moved from memory, not instructions executed. `speceval` does not, because
19,763 chunks fit comfortably in RAM and the study is not about index performance.

Measured on this corpus: BM25 retrieval is **p50 ≈ 28 ms, p95 ≈ 43 ms** over 51 queries, including
collapsing chunks to documents. That figure moved between runs on an idle machine (an earlier run
measured p50 ≈ 12 ms), so treat tens-of-milliseconds retrieval latency here as measurement noise
rather than a stable property.

## Common Mistakes

**Omitting idf.** The single most common way to build a bad lexical retriever. Without it, common
words dominate and results look random.

**Using raw term frequency with no saturation.** Rewards keyword stuffing and produces one runaway
document per query.

**Normalising length by dividing the score by document length.** Too aggressive; it over-penalises
long documents that are genuinely relevant. The `b` parameter exists to make normalisation partial.

**Tuning `k₁` and `b` before having an evaluation.** Without a gold set and a metric you cannot
tell an improvement from a coincidence. Chapter 5 comes before any tuning for this reason. This
project never tunes them at all — they stay at the conventional 1.5 and 0.75, which is a defensible
choice precisely because it was not fitted to the gold set.

**Nondeterministic tiebreaks.** Two runs, two different rankings, an afternoon lost.

## Interview Insight

> **"Explain BM25."**

Do not recite the formula. Give the three ideas and why each exists:

1. **idf** — rare terms are informative, common terms are not, so weight by inverse document
   frequency. This is also why you do not need a stopword list.
2. **Term-frequency saturation** — the twentieth occurrence of a word is not worth the same as the
   first, so `f/(f + k₁)` flattens out. `k₁` sets where the knee is.
3. **Length normalisation** — long documents accumulate matches by volume, so discount by length
   *relative to the corpus average*. `b` sets how strongly, and `b=0` disables it.

Then the sentence that shows you have used it rather than read about it: *BM25 was the state of the
art for twenty years and remains the baseline a new method has to beat — in this project it beat
the embedding model on three of fifty-one queries where exact identifier matching mattered, and
those were queries where the embedding matched a superseded document with a near-identical title.*

> **"When would you choose BM25 over an embedding model?"**

When exact tokens carry the meaning — identifiers, error codes, part numbers, names, version
strings. Also when you need explainability (you can point at the matching terms), when you have no
GPU, when the corpus changes constantly and re-embedding is expensive, or when there is no training
data in your domain. And it is cheap enough to run alongside a dense retriever rather than instead
of one, which is what chapter 6 is about.

## Performance Insight

The `(k₁ + 1)` numerator term does not affect ranking — it is a constant factor across all
documents for a given term. If you only need the ordering and not calibrated scores, you can drop
it. `speceval` keeps it for conformance with the standard formulation, at a cost of one
multiplication per posting.

## Debugging Tip

When a BM25 result is surprising, print the per-term contributions rather than the total:

```
query: "how do I postpone the evaluation of annotations"
  how        idf 0.31  contributes 0.12
  do         idf 0.28  contributes 0.10
  postpone   idf --    NOT IN CORPUS        <-- the whole problem
  the        idf 0.02  contributes 0.01
  evaluation idf 2.10  contributes 1.85
  annotations idf 2.40 contributes 2.02
```

That immediately shows the query's most distinctive word contributing nothing, which is chapter 2's
vocabulary mismatch appearing as a concrete diagnostic rather than a theory.

## Summary

- Raw term-frequency sums fail on common words and on long documents.
- idf fixes the first: weight terms by rarity. It also removes the need for stopword lists.
- Saturation fixes half of the second: `f/(f + k₁)` flattens repeated occurrences, `k₁` sets the
  knee.
- Length normalisation fixes the rest: discount by length relative to the corpus mean, `b` sets how
  strongly.
- BM25 is those three ideas in one expression, with `k₁ = 1.5` and `b = 0.75` conventional.
- `speceval` implements it in forty lines, term-major over postings lists, with deterministic
  tiebreaks, owning it because its behaviour is central to the study's findings.

## Key Takeaways

1. `idf` is the most important part. Without it nothing works.
2. `k₁` controls saturation; `b` controls length normalisation; both have sensible conventional
   values you should not tune without an evaluation.
3. Never let ties break nondeterministically in a measurement system.
4. Own the components whose behaviour you will have to explain; import the rest.

## Why the Next Chapter Exists

BM25 is strong, cheap, explainable, and permanently limited: it cannot match "postpone" to
"postponed", or "postpone" to "deferred". Chapter 2 named that as vocabulary mismatch; this chapter
has now built a system that fully exhibits it.

Chapter 4 introduces a representation where similarity of *meaning* becomes distance in space, so
that "postpone" and "deferred" can score highly against each other without sharing a single
character — and sets up the failure mode that this project's central finding is about.
