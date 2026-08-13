# Chapter 17 — The Retriever Ladder

`speceval/retrievers.py` (240 lines)

## Learning Objectives

- Explain the two-level interface (`search_chunks` / `search`) and why both exist.
- Explain `collapse_to_peps` and the ranking rule it implements.
- Explain the chunk depth multiplier and what it prevents.
- Explain why the query-embedding cache lives inside `DenseRetriever`.
- Explain how the `Query` record grew, and what each field is used by.

## Motivation

This module is the spine. It holds the `Retriever` protocol, the `Query` record, four of the six
retrievers, and the collapse logic that every one of them shares. Chapters 3, 4 and 6 covered the
*algorithms*; this chapter covers the interface they live behind — and one interface decision here is
what makes chapter 20's reranker possible at all.

## The two-level interface

Every real retriever exposes two methods:

```python
    def search_chunks(self, query: Query, depth: int) -> list[int]:
        """ranked chunk indices"""

    def search(self, query: Query, top_k: int) -> list[int]:
        """ranked, distinct PEP numbers"""
```

Both return `list[int]`, and they mean entirely different things — chunk indices into the corpus array
versus PEP numbers. The type system cannot distinguish them, which is precisely why the method names
and docstrings must.

Why both exist:

```python
"""Retrieval happens over *chunks*; the public `search` collapses chunks to a ranked list of
distinct PEPs. `search_chunks` is exposed separately because fusion has to happen at chunk
level: fusing two already-collapsed PEP lists would throw away the evidence about *how
many* chunks of a PEP each system liked.
"""
```

`search` is what the evaluation loop calls. `search_chunks` is what *other retrievers* call.

```
                 evaluate()                      run_phase3/4
                     |                                |
                     v                                v
                 search()                       search_chunks()
                     |                                |
                     v                                v
              +-------------+                  +-------------+
              | ranked PEPs |                  | ranked chunks|
              +-------------+                  +-------------+

   HybridRetriever  ---calls--->  lexical.search_chunks(), dense.search_chunks()
   AuthorityReranker ---calls--->  hybrid.search_chunks()
```

Three consumers need chunks rather than documents: fusion (chapter 6), reranking (chapter 20), and the
generator (chapter 19), which is fed chunk *text*. Only the evaluation loop wants documents.

Chapter 23 records the consequence of that split being invisible in the reported numbers: retrieval was
scored on documents at depth 10 while generation consumed chunks at depth 5, and the two disagreed
about which strategy was best.

## Deep Explanation: the collapse

```python
def collapse_to_peps(chunks: list[Chunk], chunk_indices: list[int], top_k: int) -> list[int]:
    """Map a ranked chunk list to ranked distinct PEPs.

    A PEP inherits the rank of its best-scoring chunk. Collapsing *after* ranking (rather
    than concatenating each PEP into one document) keeps BM25's length normalisation from
    punishing long PEPs for being thorough.
    """
    ranked: list[int] = []
    seen: set[int] = set()
    for chunk_index in chunk_indices:
        pep_number = chunks[chunk_index].pep_number
        if pep_number not in seen:
            seen.add(pep_number)
            ranked.append(pep_number)
            if len(ranked) == top_k:
                break
    return ranked
```

The rule is **first occurrence wins**: a PEP takes the rank of its best chunk, and its remaining chunks
are discarded.

Consider the alternative designs and why they were rejected.

**Concatenate each PEP into one document before indexing.** Then a 27-chunk PEP is one very long
document, and BM25's length normalisation (chapter 3) discounts it for being long. A thorough
specification would be systematically penalised against a short one. The docstring names this.

**Score a PEP by the sum of its chunk scores.** Then a PEP with many mediocre chunks beats one with a
single excellent chunk — length rewarded rather than penalised, the opposite error.

**Score by the mean of its chunk scores.** A PEP with one perfect chunk and twenty-six irrelevant ones
scores badly, even though that one chunk answers the question.

First-occurrence is the max, and max is right here because relevance is *existential*: the question is
whether this document contains the answer somewhere, not whether it is uniformly about the topic.

The early `break` matters for cost. The caller passes a chunk list ten times longer than `top_k`
(below); without the break, the loop would walk all of it after already having enough documents.

Pinned:

```python
def test_deduplicates_keeping_best_rank(self) -> None:
    chunks = make_chunks([10, 10, 20, 10, 30])
    self.assertEqual(collapse_to_peps(chunks, [1, 0, 2, 4], top_k=3), [10, 20, 30])
```

Chunk 1 and chunk 0 both belong to PEP 10; the ranking `[1, 0, 2, 4]` puts chunk 1 first, so PEP 10
enters at rank 1 and chunk 0 is skipped.

## Deep Explanation: the depth multiplier

```python
# Chunks are retrieved deeper than the PEP cutoff because several chunks of the same PEP
# routinely occupy the top positions; without this, top_k distinct PEPs is unreachable.
CHUNK_DEPTH_MULTIPLIER = 10
```

```python
    def search(self, query: Query, top_k: int) -> list[int]:
        depth = top_k * self.chunk_depth_multiplier
        return collapse_to_peps(self.chunks, self.search_chunks(query, depth), top_k)
```

Arithmetic: the corpus averages 26.9 chunks per PEP. A query strongly matching one PEP will often have
five or six of its chunks in the top ten. Collapse those and you have two or three distinct documents,
not ten.

Retrieving ten chunks per requested document makes ten distinct documents reachable in almost all
cases. Without it, Recall@10 would be capped by an implementation artefact rather than by retrieval
quality — and the cap would be *invisible*, because the metric would faithfully report a low number
that had nothing to do with the ranking.

This is a good example of a constant that looks arbitrary and is not. It is set by the corpus's
chunks-per-document ratio, and if that ratio changed — a different chunk size — the multiplier would
need revisiting.

## Deep Explanation: `DenseRetriever`'s query cache

```python
    _query_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def _embed_query(self, text: str) -> np.ndarray:
        """Embed a query, caching so repeated evaluation runs stay comparable.

        The cache is deliberately kept out of the timed region's first call: the reported
        latency includes the Ollama round trip, because that is what a real query costs.
        """
        if text not in self._query_cache:
            assert self.embedder is not None
            vector = self.embedder.embed([text], prefix=QUERY_PREFIX)
            self._query_cache[text] = normalise(vector)[0]
        return self._query_cache[text]
```

An in-memory cache, per retriever instance, keyed on query text.

Why it exists: chapter 20's sweep constructs a new reranker for each of eleven strengths, all wrapping
the *same* `DenseRetriever` instance. Without the cache, each of 51 queries would be embedded eleven
times — 561 round trips instead of 51.

Why it is per-instance rather than global: two `DenseRetriever` instances might hold different vectors
or a different embedder. A module-level cache keyed only on text would serve one instance's embedding
to another. Instance scope makes that impossible.

The docstring's second paragraph is the subtle part. The **first** call for a query pays the round trip
and is inside the timed region, so reported latency reflects what a real query costs. Subsequent calls
within the same run are free — which means the p50 across 51 distinct queries is a genuine
cold-per-query figure, while a re-run of the *same* query within one process would not be. Worth
knowing before quoting a latency number.

## Deep Explanation: the `Query` record

```python
@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    category: str
    relevant: frozenset[int]
    asked_version: str | None = None
    # True when the corpus contains a superseded/rejected predecessor that a naive
    # retriever is likely to surface instead of the answer. Reported as its own subset:
    # authority reranking must be measurable on the queries it cannot help, or its benefit
    # is being scored only where it was designed to win.
    trap: bool = False
    note: str = ""
```

Who consumes what:

| Field | Consumed by |
|---|---|
| `qid` | Result tables, per-query reports, the random control's seed |
| `text` | Every retriever; the prompt builder |
| `category` | `evaluate()`'s disaggregation |
| `relevant` | `recall_at_k`, `ndcg_at_k`, the oracle, the answer metrics |
| `asked_version` | The version metric; the reranker's optional version rule |
| `trap` | Subset reporting in the drivers |
| `note` | Nothing. It is for humans auditing labels |

The record grew as the study did — `trap` arrived with the subset analysis, `asked_version` arrived
(and was renamed) with the version metric. Both have defaults, so the gold set could be extended a
field at a time without rewriting every entry.

`frozenset` rather than `set` for `relevant`: chapter 7 covered why. `frozen=True` prevents rebinding
the attribute; only `frozenset` prevents mutating its contents.

## The six implementations

| Class | Rung | Lines | Notes |
|---|---|---|---|
| `BM25Retriever` | 1 | ~15 | Builds its index in `__post_init__` |
| `DenseRetriever` | 2 | ~35 | Validates vector/chunk agreement; caches query embeddings |
| `HybridRetriever` | 3 | ~25 | Fuses at chunk level |
| `AuthorityReranker` | 4 | in `rerank.py` | Imports from here; nothing here imports it |
| `OracleRetriever` | — | 9 | Positive control |
| `RandomRetriever` | — | 12 | Negative control, seeded per query |

The dependency direction is worth noting: `rerank.py` imports `HybridRetriever`, `Query`,
`collapse_to_peps` and `RRF_K` from this module, and this module imports nothing from `rerank.py`. Rung
4 is a strict extension. You can delete `rerank.py` and rungs 1–3 still run.

That is a consequence of the `Protocol` design (chapter 7). With an abstract base class, adding a rung
would mean touching the hierarchy that the other rungs depend on.

## Systems Perspective

Construction costs differ sharply across the ladder:

| Retriever | Construction | Per query |
|---|---|---|
| `BM25Retriever` | seconds (index build) | ~28 ms p50 |
| `DenseRetriever` | ~0.1 s (normalise 58 MB) | ~37 ms p50 |
| `HybridRetriever` | free (wraps two) | ~28 ms p50 |
| `AuthorityReranker` | free (wraps one) | hybrid + microseconds |

Two things stand out.

`DenseRetriever` construction normalises a 19,763 × 768 matrix — one pass over 58 MB, fast, but it
allocates a *second* 58 MB array. Peak memory during construction is ~116 MB. Fine here; worth knowing
if the corpus were ten times larger.

`HybridRetriever` is not slower than its slowest component in the way you might expect, because the
dominant cost in dense retrieval is the query-embedding round trip and hybrid pays that exactly once,
sharing it through the cached `DenseRetriever` instance.

## Common Mistakes

**Returning chunk indices where PEP numbers are expected.** Both are `list[int]`; nothing catches it at
runtime. The two-method naming convention is the defence.

**Collapsing before fusing.** Discards how many chunks each system liked, which is the evidence fusion
uses.

**Retrieving `top_k` chunks to produce `top_k` documents.** Silently caps recall at a fraction of what
the retriever could achieve.

**Summing or averaging chunk scores per document.** Sum rewards length; mean punishes a document with
one excellent chunk. Max — first occurrence in the ranking — matches what relevance actually means.

**A module-level query-embedding cache.** Two retrievers with different vectors would share it.

**Omitting the early `break` in the collapse.** Walks a list ten times longer than needed.

## Interview Insight

> **"How do you go from chunk-level retrieval to document-level results?"**

Rank chunks, then collapse to distinct documents keeping the *best* rank each document achieved — first
occurrence wins. Max rather than sum or mean, because relevance is existential: the question is whether
the document contains the answer somewhere, not whether it is uniformly on-topic. Sum would reward long
documents and mean would punish a document with one perfect section.

And retrieve deeper than you need — I retrieve ten chunks per requested document, because with ~27
chunks per document several chunks of the same document routinely occupy the top positions. Without
that, asking for ten documents returns two or three, and your recall metric faithfully reports a low
number caused by your own plumbing.

> **"Why expose two retrieval methods?"**

Because fusion and reranking have to operate on chunks. If a hybrid retriever fused two already-collapsed
document lists, it would lose the information about how many chunks of each document each system liked —
which is exactly the evidence rank fusion uses. So `search_chunks` is the primitive and `search` is the
collapse on top of it.

There is a cost to that split I would name too: my retrieval metrics ran on the collapsed document lists
at depth 10, while the generator consumed the top 5 chunks. Two mismatches, unit and depth — and the two
levels ended up disagreeing about which strategy was best.

## Debugging Tip

When a retriever returns fewer documents than requested, the cause is almost always the depth
multiplier interacting with chunk concentration. Print both levels:

```python
chunk_indices = retriever.search_chunks(query, depth=50)
print(len(chunk_indices), "chunks ->",
      len({chunks[i].pep_number for i in chunk_indices}), "distinct PEPs")
```

If 50 chunks collapse to 4 documents, the query is hitting one document very hard and the multiplier
needs to be larger for that corpus.

## Summary

- Two methods: `search_chunks` (the primitive, for fusion, reranking and the generator) and `search`
  (the collapse, for evaluation). Both return `list[int]` meaning different things, so naming carries
  the distinction.
- `collapse_to_peps` implements first-occurrence-wins — max, not sum or mean — because relevance is
  existential.
- The ×10 depth multiplier exists because ~27 chunks per document makes `top_k` distinct documents
  otherwise unreachable, and the resulting recall cap would be invisible.
- `DenseRetriever` caches query embeddings per instance; the first call still pays the round trip so
  reported latency is honest.
- `Query` grew field by field with defaults; `note` is read only by humans.
- `rerank.py` imports from here and not the reverse — rung 4 is a strict extension, which the `Protocol`
  design makes possible.

## Key Takeaways

1. Expose the primitive as well as the convenience; composition happens at the primitive.
2. Collapse with max, and retrieve deeper than your cutoff.
3. Same type, different meaning — let names carry what the type system cannot.
4. Cache at instance scope when the cache depends on instance state.

## Why the Next Chapter Exists

Chapter 18 reads `embed.py` in full — the batching loop, the fingerprinted cache, the retry policy, and
the normalisation guard — the module that turns 19,763 chunks into the 58 MB array this chapter's dense
retriever multiplies against.
