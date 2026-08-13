# Chapter 22 — One Query, End to End

## Learning Objectives

- Follow a single query through every function in the system, in order.
- See the central failure mode occur, at chunk granularity, in real output.
- See the pool-depth confound from chapter 21 occur live.
- Read the actual generated answers before and after the intervention.
- Account for every piece of shared state the query touches.

## The query

`q16` from the gold set — the query that produced the project's central mechanism:

```json
{"qid": "q16", "text": "how do I postpone the evaluation of annotations",
 "category": "availability", "relevant": [649], "trap": true,
 "note": "PEP 563 'Postponed Evaluation of Annotations' matches the wording exactly but is
          Superseded by 649 (Final, 3.14)."}
```

Two documents compete:

```
   PEP 563   Postponed Evaluation of Annotations                    Superseded (by 649)
   PEP 649   Deferred Evaluation Of Annotations Using Descriptors   Final
```

Everything below is real output, produced by running the system while writing this chapter.

## Stage 0 — startup

```
load_corpus("peps/peps")          734 Pep records
chunk_corpus(peps)                19,763 Chunk records
load_queries("eval/...json")      51 Query records
embed_cached([...])               cache hit: (19763, 768) float32, 58 MB
```

Three retrievers are constructed. `BM25Retriever.__post_init__` builds the inverted index;
`DenseRetriever.__post_init__` checks `19763 == 19763` and normalises the matrix;
`HybridRetriever` wraps both.

**Shared state now live:**

| State | Owner | Mutability |
|---|---|---|
| 734 `Pep` records | driver | frozen |
| 19,763 `Chunk` records | driver | frozen |
| Inverted index (postings, lengths, idf) | `BM25Retriever` | built once, then read-only |
| Normalised 58 MB matrix | `DenseRetriever` | built once, then read-only |
| Query-embedding cache | `DenseRetriever` | **mutable**, grows during the run |
| Answer cache | `CachedGenerator` | **mutable**, on disk |

Only the last two change during a run, and both are caches whose contents are functions of their keys.

## Stage 1 — BM25

`lex.search_chunks(q, depth=5)`:

```
  1. chunk 11548  PEP 649   [Final     ]  'Deferred Evaluation Of Annotations…' / 'Annotations On Local Var…'
  2. chunk 11547  PEP 649   [Final     ]  'Deferred Evaluation Of Annotations…' / 'Interactive REPL Shell'
  3. chunk  8903  PEP 563   [Superseded]  'Postponed Evaluation of Annotations' / 'Implementation'
  4. chunk  8898  PEP 563   [Superseded]  'Postponed Evaluation of Annotations' / 'Resolution'
  5. chunk 11512  PEP 649   [Final     ]  'Deferred Evaluation Of Annotations…' / 'Static typing users'
```

**BM25 gets this right.** Three of five chunks are from the live PEP, and it leads.

Why? Chapter 2's vocabulary mismatch, operating in the project's favour by accident. The query's most
distinctive token is `postpone`. PEP 563's title contains `postponed` — a *different token* under exact
matching, so it earns no credit for the near-match. The remaining query tokens (`evaluation`,
`annotations`) appear in both. With its strongest signal neutralised, BM25 falls back to body-text
frequency, and PEP 649 — the longer, more recent, more detailed document — wins on volume of matching
terms.

BM25 is right here for a reason unrelated to authority. It cannot see status; it simply failed to
notice the near-synonym that would have misled it.

## Stage 2 — dense retrieval

`den.search_chunks(q, depth=5)`:

```
  1. chunk  8900  PEP 563   [Superseded]  'Rationale and Goals'
  2. chunk  8916  PEP 563   [Superseded]  'Introducing a new dictio…'
  3. chunk  8931  PEP 563   [Superseded]  'Acknowledgements'
  4. chunk  8914  PEP 563   [Superseded]  'Keeping the ability to u…'
  5. chunk  8923  PEP 563   [Superseded]  'python/typing#400'
```

**Five out of five chunks are from the superseded document.** Total capture.

This is more severe than the document-level view in chapter 11 suggested. There, dense retrieval
"ranked PEP 563 first". At chunk granularity it does not merely rank it first — it fills the entire
window, including a chunk titled *Acknowledgements*, which contains no technical content at all.

The mechanism is chapter 4's and chapter 11's, sharpened. Every chunk's `indexed_text` begins with the
PEP title (chapter 2), so all 27 chunks of PEP 563 carry *Postponed Evaluation of Annotations* into
their embedding. The query is a near-paraphrase of that title. So every chunk of the superseded document
scores highly, and a document with 27 near-duplicate title-boosted chunks crowds out a competitor.

The title-prepending decision, which chapter 2 justified as improving recall, is here actively
amplifying the failure. That is not a reason to remove it — it helps far more often than it hurts — but
it is an honest interaction worth naming, and it is a candidate explanation for chapter 23's failed
prediction.

## Stage 3 — hybrid fusion

`hyb.search_chunks(q, depth=5)`:

```
  1. chunk  8900  PEP 563   [Superseded]
  2. chunk 11548  PEP 649   [Final     ]
  3. chunk  8916  PEP 563   [Superseded]
  4. chunk 11547  PEP 649   [Final     ]
  5. chunk  8903  PEP 563   [Superseded]
```

RRF interleaves. Each system's top picks alternate, exactly as chapter 6's rank-only formulation
predicts: dense's rank-1 and BM25's rank-1 both score `1/61`, and the tiebreak falls to chunk index.

The result is a mixed context — three superseded chunks, two live. Better than dense alone, worse than
BM25 alone. Fusing an unequal pair at equal weight lands between them, which is chapter 6's measured
conclusion appearing in a single query.

## Stage 4 — the pool-depth effect, live

Now the reranker at **λ = 0**, which chapter 20 established is arithmetically identical to no reranking.

It draws a pool of `5 × 10 = 50` chunks, multiplies every score by 1.0, and returns the top 5. Its
context contains:

```
PEPs in context: [563]
```

**Only PEP 563.** Not the mixed 563/649 that hybrid at depth 5 produced.

Nothing was reranked — λ=0 leaves every score untouched. The difference is entirely that
`hybrid.search_chunks(50)` fuses over a deeper candidate pool than `hybrid.search_chunks(5)`, and RRF
over a deeper pool produces a different top-5 (chapter 6: consistent mid-list agreement can outrank
inconsistent top placement).

This is chapter 21's confound, visible in one query. It is why the sweep's baseline must be λ=0 of the
same pipeline rather than the earlier hybrid row — and here the effect runs *against* the intervention,
making the baseline worse. Using the shallower hybrid number as the baseline would have understated the
starting point on this query while overstating it on others.

## Stage 5 — generation at λ=0

The prompt is 3,797 characters: five chunks, each headed `[PEP 563] Postponed Evaluation of
Annotations -- <section>`, with **no status field** (chapter 19's control).

The measured answer:

> To postpone the evaluation of annotations, you need to use a string form that is obtained from the
> AST during compilation. This means that the annotation will be preserved in its original formatting,
> but it may not be syntactically valid as a Python expression. **PEP 563** states that annotations can
> only use names present in the module scope, and local variable annotations are not evaluated at all
> since they are not accessible outside of the function's closure.

`extract_citations` → `[563]`. Status lookup → `Superseded`.

**This is the failure the project exists to measure.** The answer is fluent, technically accurate about
what PEP 563 says, correctly cited to its source, and describes a mechanism that has been superseded. A
reader has no signal that anything is wrong. The model did nothing incorrect — it was handed five chunks
of a dead specification and summarised them faithfully.

Note also what the model could not do: it had no access to PEP 649, and no way to know it existed. The
generator cannot repair the retriever (chapter 1).

## Stage 6 — the authority weights

At λ=1 the reranker computes a weight for every chunk's PEP:

```
  PEP 563 [Superseded]  superseded_by=649  ->  weight 0.0225
  PEP 649 [Final]       superseded_by=None ->  weight 1.0000
```

PEP 563's weight is the product of both signals (chapter 20):

```
   STATUS_WEIGHT["Superseded"]  ×  SUPERSEDED_BY_FACTOR
        0.15                    ×        0.15            =  0.0225
```

Marked dead *and* carrying a pointer to its replacement. A factor of **44** between the two documents.

Applied to the blend at λ=1, `final = base × weight`, so PEP 563's chunks retain 2.25% of their base
score while PEP 649's retain 100%. No base-rank difference within a 50-chunk RRF pool can survive that
(chapter 20's threshold arithmetic: the whole pool spans a factor of 1.8).

## Stage 7 — reranked retrieval

`AuthorityReranker(strength=1.0).search_chunks(q, depth=5)`:

```
  1. chunk 11530  PEP 649  [Final]  weight=1.0000
  2. chunk 10113  PEP 604  [Final]  weight=1.0000   'Allow writing union types as X | Y'
  3. chunk 11522  PEP 649  [Final]  weight=1.0000
  4. chunk 11497  PEP 649  [Final]  weight=1.0000
  5. chunk 11529  PEP 649  [Final]  weight=1.0000
```

PEP 563 is gone entirely. Four of five chunks are the live specification.

The intruder at rank 2 — PEP 604, about union type syntax — is worth noting rather than hiding. It was
present in the 50-chunk pool (annotations and typing are adjacent topics), scored middlingly, and rose
when every superseded competitor was demoted by a factor of 44. The reranker does not know PEP 604 is
off-topic; it knows only that it is *authoritative*.

That is a real cost of the intervention: demoting dead documents promotes whatever else was in the pool,
relevant or not. Here four of five slots are still correct, so the effect is minor. It is the mechanism
by which a reranker could, in principle, hurt — and chapter 23 reports that across 51 queries it never
did, with zero regressions.

## Stage 8 — generation at λ=1

Prompt: 3,720 characters, chunks from PEP 649 and PEP 604, still no status.

The measured answer:

> To postpone the evaluation of annotations, you can use the `__annotate__` descriptor, which delays the
> evaluation until the first time `__annotations__` is referenced. This allows for lazy computation of
> annotations on demand. PEP 634 does not exist in these excerpts, but **PEP 649** provides information
> about how to achieve this.

`extract_citations` → `[634, 649]`. Both `Final`. **No superseded citation.** Query q16 moves from the
`FIXED` list in chapter 21's per-query diff.

Two honest observations about this answer.

**The technical content is now correct** — `__annotate__` and lazy evaluation are what PEP 649
specifies, and the answer describes the live mechanism rather than the dead one.

**The prose is worse.** The final sentence — *"PEP 634 does not exist in these excerpts"* — is confused;
PEP 634 (structural pattern matching) is unrelated and was not in the context. The model appears to have
produced a spurious reference and then partially retracted it.

That matters for how the metrics should be read. `extract_citations` counts 634 as a citation. It
happens to be `Final`, so the superseded-citation metric is unaffected, and it is a real PEP, so the
hallucination metric is unaffected. But it *is* a citation to a document that was never in the context —
an ungrounded citation that both metrics are blind to, because one checks status and the other checks
existence, and neither checks provenance.

This is a third limitation of the automatic metrics, alongside chapter 19's two. It is not currently
measured. Measuring it would be cheap — the set of PEPs in the context is known at generation time — and
it belongs in the future-work list.

## The full path

```
   "how do I postpone the evaluation of annotations"
        |
        v
   [ BM25Retriever.search_chunks ]  tokenize -> postings -> BM25 score -> sort
        |                            3 of 5 chunks from PEP 649       (right, by luck)
        |
   [ DenseRetriever.search_chunks ]  embed query (cached) -> matmul -> argpartition
        |                            5 of 5 chunks from PEP 563       (captured)
        |
        v
   [ HybridRetriever.search_chunks ]  RRF over both rankings
        |                             interleaved 563/649
        v
   [ AuthorityReranker.search_chunks ]  pool 50 -> weight -> blend -> sort -> cut 5
        |    lambda=0  -> all PEP 563   (pool-depth effect, not reranking)
        |    lambda=1  -> 4x PEP 649    (weights 1.0 vs 0.0225)
        v
   [ build_prompt ]  title + section + text per chunk, NO status
        |
        v
   [ CachedGenerator.generate ]  temperature 0, seed 7, <=220 tokens
        |
        v
   [ extract_citations ]  regex -> filter to known PEPs
        |    lambda=0 -> [563] Superseded   <- the failure
        |    lambda=1 -> [634, 649] both Final
        v
   [ score_answer ]  status lookup -> AnswerRecord
```

## Edge and adversarial conditions

**Concurrency.** There is none. Every driver is single-threaded, deliberately (chapter 18): parallel
requests to one local model server add nondeterminism for throughput the study does not need. The two
mutable caches are therefore never contended.

**A missing corpus** raises with the fetch command named (chapter 14). A corpus that parses to zero PEPs
raises rather than producing an evaluation of 0.000 everywhere.

**A stale embedding cache** cannot be silently used: the key covers model, prefix and every chunk text,
and `DenseRetriever` additionally refuses a vector/chunk count mismatch.

**A down model server** produces a bounded three-attempt retry with linear backoff and an error naming
the likely cause, rather than a hang.

**A query with no lexical matches** returns an empty chunk list from BM25 (chapter 15), which flows
through as an empty document list rather than an arbitrary ranking.

**An unparseable status** — the corpus's real `April Fool!` — gets `DEFAULT_STATUS_WEIGHT = 0.50` rather
than crashing the reranker or being trusted at 1.0.

**A query whose gold labels are all non-authoritative** — q30 — is why the oracle's `trap@1` floor is
0.020 rather than zero (chapter 13).

## Resource ownership

Nothing in the system owns a resource requiring release. No file handles are held open (`read_text` and
`np.load` close), no sockets persist (`urllib.request.urlopen` is used as a context manager), no
subprocesses are spawned. The model server's lifecycle is the operator's.

Peak memory is dominated by the embedding matrix: 58 MB resident, briefly 116 MB during construction
because `normalise` allocates a second array (chapter 17).

## Summary

- One query touches nine stages: corpus load, chunking, three retrievers, the reranker, prompt
  construction, generation, and scoring.
- BM25 got q16 right for a reason unrelated to authority — its strongest signal, `postpone`, matched
  neither document, so it fell back to body-text volume.
- Dense retrieval returned **five of five chunks from the superseded document**, including an
  *Acknowledgements* section. Title-prepending amplifies this: all 27 chunks of PEP 563 carry the
  matching title into their embedding.
- The λ=0 reranker produced a *different* context than hybrid at depth 5 — the pool-depth confound from
  chapter 21, visible in one query, with reranking doing nothing.
- The λ=0 answer is fluent, accurate about a dead specification, and correctly cited to it. That is the
  failure mode.
- Authority weights differ by a factor of 44 (0.0225 vs 1.0), which no base-rank difference in a
  50-chunk RRF pool can overcome.
- The λ=1 answer describes the live mechanism, cites only `Final` documents — and contains a spurious
  reference to a PEP that was never in its context, revealing a third metric limitation: nothing checks
  whether a citation was *grounded* in the provided chunks.

## Key Takeaways

1. Trace at chunk granularity, not document granularity — the failure was more complete than the
   document view showed.
2. A decision that helps on average (title-prepending) can amplify a specific failure. Name the
   interaction rather than hiding it.
3. Reading real generated output finds limitations that aggregates cannot: this trace produced a third
   uncounted failure mode in one query.
4. The generator is downstream of everything and can repair nothing.

## Why the Next Chapter Exists

We have followed one query and seen the mechanism and the fix at full resolution. Chapter 23 steps back
to all 51: what was found, how confident each finding is, which predictions were wrong, and where the
central result stops being true.
