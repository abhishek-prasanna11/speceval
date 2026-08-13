# Chapter 15 — The Lexical Retriever

`speceval/bm25.py` (93 lines)

## Learning Objectives

- Read the index-construction loop and account for every data structure it builds.
- Explain why the search loop is term-major and what that has to do with the inverted index.
- Explain the deterministic tiebreak and why a measurement system requires one.
- Explain what the five BM25 tests each protect against.
- State the module's memory and time characteristics.

## Motivation

Chapter 3 derived BM25 from its counterexamples and showed the formula. This chapter reads the module
as *code*: what it allocates, in what order it iterates, and which lines exist to prevent a specific
failure. The formula is settled; the implementation choices are what would differ between a correct
implementation and a good one.

## The module in one view

```
   documents: list[str]                    (19,763 chunk texts)
        |
        | __init__: one pass
        v
   +--------------------------------------------------+
   |  postings:    term -> [(doc_index, freq), ...]   |
   |  doc_lengths: [int, ...]        per document      |
   |  avg_doc_length: float          corpus mean       |
   |  idf:         term -> float     precomputed       |
   +--------------------------------------------------+
        |
        | search(query, top_k): term-major accumulation
        v
   [(doc_index, score), ...]   sorted, truncated
```

Four structures, all built once. Everything the scoring formula needs is precomputed, so `search` does
arithmetic and nothing else.

## Deep Explanation: construction

```python
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
        self.idf = {
            term: math.log((self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0)
            for term, posting in self.postings.items()
        }
```

### The two-level count

Notice the inner loop counts into a *local* `frequencies` dict before appending to `postings`. The
direct alternative — appending to `postings[term]` once per token occurrence — would produce duplicate
entries for the same document, and the search loop would then add a term's contribution several times.

Counting locally first guarantees the invariant that matters: **each `(doc_index, frequency)` pair
appears exactly once per term.** That invariant is what makes `len(posting)` a valid document frequency
in the next line.

### `len(posting)` is `df(t)`

```python
            term: math.log((self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0)
```

The postings list holds one entry per document containing the term, so its length *is* the document
frequency. No separate counter is needed — the index already encodes it. This is a small thing, but it
is the kind of structural economy that makes an implementation readable: there is one source of truth
for how many documents contain a term.

### The empty-corpus guard

```python
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        )
```

`BM25([])` would otherwise divide by zero at construction. It cannot happen through the normal path —
`load_corpus` raises on an empty corpus (chapter 14) — but a unit test constructing an empty index
should get an empty index, not a `ZeroDivisionError`. Defensive code at a boundary, costing one
conditional.

## Deep Explanation: search

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

### Term-major, not document-major

The outer loop is over query terms; the inner walks that term's postings list. This is the access
pattern the inverted index exists to serve.

The consequence is worth stating precisely: **a document containing no query term is never touched.** It
never enters `scores`, never has a norm computed, never costs anything. For a query of rare terms over
19,763 chunks, the loop may touch a few dozen documents.

The document-major alternative — for each document, check each query term — is `O(n_docs × n_terms)`
regardless of how rare the terms are, and it wastes the index entirely.

### Unknown terms are skipped, not zeroed

```python
            posting = self.postings.get(term)
            if posting is None:
                continue
```

`.get` rather than `[]`, and `continue` rather than adding zero. The difference shows up in the
degenerate case: a query whose every term is absent from the corpus returns `[]` — an honest "I found
nothing" — rather than an arbitrary ranking of documents that all scored zero.

```python
def test_unknown_term_returns_nothing(self) -> None:
    index = BM25(["alpha beta", "gamma delta"])
    self.assertEqual(index.search("zzzzz", top_k=5), [])
```

That distinction propagates upward. An empty chunk list means an empty document list means a query the
system honestly could not serve, rather than five confidently-ranked irrelevant results.

### The deterministic tiebreak

```python
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
```

The key is a tuple: score descending, then document index ascending. The second element is the entire
point.

Ties are common. Two chunks of the same PEP containing the same query term the same number of times, at
the same length, score identically to the last bit. Sorting on score alone leaves their relative order
to whatever `sorted` does with equal keys — which is stable with respect to *dictionary insertion
order*, which depends on which documents happened to be encountered first, which depends on the
iteration order of the postings lists.

In a measurement system that is unacceptable. Two runs would produce different rankings, hence
different Recall@10, hence a difference between two systems that is an artefact of dictionary ordering.
Adding `pair[0]` to the sort key removes the possibility entirely.

This is the same discipline as chapter 6's fusion tiebreak and chapter 14's sorted globs: **anything
that could vary between runs for no reason must be pinned.**

## The tests, and what each protects

Five tests in `tests/test_corpus.py::TestBm25`:

| Test | Protects against |
|---|---|
| `test_tokenizer_keeps_dunder_identifiers` | A "cleanup" simplifying the regex to `[a-z0-9]+`, destroying `__future__` |
| `test_exact_term_wins` | Basic sanity: the document containing the query term ranks first |
| `test_unknown_term_returns_nothing` | Silently returning an arbitrary ranking for an unanswerable query |
| `test_rare_term_outranks_common_term` | **Removal or breakage of idf** |
| `test_scores_are_descending` | A sort-order regression |

The fourth is the load-bearing one, and it directly encodes chapter 3's first counterexample:

```python
def test_rare_term_outranks_common_term(self) -> None:
    # idf must make "walrus" worth more than "python", which is in every document.
    index = BM25(
        ["python python python", "python walrus", "python python", "python"]
    )
    hits = index.search("python walrus", top_k=4)
    self.assertEqual(hits[0][0], 1)
```

Document 0 has three occurrences of `python`; document 1 has one `python` and one `walrus`. Raw
term-frequency scoring picks document 0. BM25 picks document 1, because `walrus` appears in one of four
documents while `python` appears in all four and therefore carries almost no weight.

Delete the `idf` multiplication and this test fails. Nothing else in the suite would.

## Why this is implemented rather than imported

```python
"""Implemented directly rather than pulled in as a dependency: it is about forty lines,
and the lexical baseline is the one component whose behaviour needs to be inspectable
when the results look surprising.
"""
```

The argument is scoped, and worth reading as a general rule rather than as a preference for writing
everything yourself. The same project imports a pretrained embedding model without hesitation, because
reimplementing that would teach nothing about the question being asked.

The rule being applied: **own the components whose behaviour you will have to explain.**

BM25's behaviour is central to the findings. Chapter 11's mechanism argument depends on knowing exactly
why BM25 ranked PEP 649 above PEP 563 when dense retrieval did the opposite — and answering that
required printing per-term idf contributions. With a black-box dependency, that investigation would
have meant reading someone else's source anyway.

## Systems Perspective

**Build.** One pass over 19,763 documents, tokenising each. Dominated by regex matching and dictionary
increments; a couple of seconds. Rebuilt on every driver invocation rather than cached, for the same
staleness-avoidance reason as chapter 14's parse.

**Memory.** The postings lists hold one `(int, int)` tuple per distinct term per document. With a few
hundred distinct terms per chunk across 19,763 chunks, that is millions of small tuples — tens to low
hundreds of megabytes in Python's boxed representation. Real search engines store postings as packed
integer arrays with delta encoding for exactly this reason; at this scale it does not matter.

**Query.** Proportional to the total length of the postings lists for the query's terms. A query
containing `python` touches nearly every chunk; a query of rare identifiers touches a handful.

**Measured**, live, over 51 queries including the collapse to documents: **p50 ≈ 28 ms, p95 ≈ 43 ms**.

A caution on that figure: an earlier run on the same machine measured p50 ≈ 12 ms. Retrieval latency at
this scale is dominated by machine state, not by the algorithm, and differences of tens of milliseconds
between runs should not be read as differences between strategies. The one latency gap that *is*
consistently meaningful is dense retrieval's extra ~10 ms, and that is a network round trip (chapter 4),
not arithmetic.

## Common Mistakes

**Appending to the postings list per token occurrence** rather than per document, producing duplicate
entries and multiply-counted contributions.

**Maintaining a separate document-frequency counter** when `len(posting)` already is one.

**Document-major scoring**, which discards the entire benefit of the inverted index.

**Treating an unknown term as a zero-scoring match**, turning "no results" into "arbitrary results".

**Sorting on score alone**, leaving ties to dictionary iteration order.

**Caching the index to disk.** It builds in seconds and a stale index is a debugging afternoon.

## Interview Insight

> **"Why did you implement BM25 yourself?"**

Because it is forty lines and it is the component whose behaviour I most needed to be able to inspect.
One of the study's findings depends on explaining why BM25 ranked the live specification above the
superseded one on a specific query while the embedding model did the opposite — and answering that meant
printing per-term idf contributions. With a dependency I would have ended up reading its source anyway.

The rule I applied is narrow: own the components whose behaviour you will have to explain, import the
rest. The same project uses a pretrained embedding model without a second thought, because
reimplementing that would teach nothing relevant.

> **"How do you keep retrieval results reproducible?"**

Every ordering decision is pinned. BM25 sorts on `(-score, doc_index)` so ties break deterministically
rather than by dictionary iteration order; fusion does the same; the corpus loader sorts its glob so
chunk indices are stable across runs; and the random control is seeded per query rather than once, so
adding a query does not change earlier queries' draws.

That matters because cached embeddings are indexed positionally — if chunk order shifted between runs,
every cached vector would silently correspond to the wrong chunk.

## Debugging Tip

Chapter 3 gave the diagnostic; here it is as a habit. When a BM25 result is surprising, print
**per-term contributions**, not the total:

```
query: "how do I postpone the evaluation of annotations"
  postpone     NOT IN CORPUS        <-- the whole explanation
  evaluation   idf 2.10  ->  1.85
  annotations  idf 2.40  ->  2.02
```

The absent term is invisible in a total score and obvious in a breakdown. This is how chapter 11's
mechanism was established rather than guessed.

## Summary

- Construction builds four structures in one pass: postings, per-document lengths, the corpus mean, and
  a precomputed idf table.
- Counting into a local dict first guarantees one postings entry per document per term, which is what
  makes `len(posting)` a valid document frequency.
- Search is term-major over postings, so documents containing no query term cost nothing.
- Unknown terms are skipped rather than zeroed, so an unanswerable query returns nothing rather than an
  arbitrary ranking.
- The `(-score, doc_index)` tiebreak removes run-to-run variation that would otherwise come from
  dictionary ordering.
- Five tests; the idf test is the one that would catch the most damaging regression.
- Implemented rather than imported because its behaviour is load-bearing for the study's findings.
- Measured p50 ≈ 28 ms, p95 ≈ 43 ms over 51 queries — but retrieval latency here varies with machine
  state and should not be over-read.

## Key Takeaways

1. Term-major iteration is the reason the inverted index exists; document-major throws it away.
2. Deterministic tiebreaks are mandatory in measurement code.
3. "No results" and "arbitrary results" must not be the same output.
4. Own what you must explain; import what you need not.

## Why the Next Chapter Exists

We can rank chunks. Chapter 16 reads the layer that decides whether a ranking was any good —
`metrics.py` and `evaluate.py` — including the timing decision that chapter 5 flagged as essential and
the disaggregation machinery that chapter 12 argued findings live in.
