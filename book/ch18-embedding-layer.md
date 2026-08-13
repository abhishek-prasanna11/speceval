# Chapter 18 — The Embedding Layer

`speceval/embed.py` (190 lines)

## Learning Objectives

- Trace the full lifecycle of `embed_cached`, from call to cached array.
- Explain the metadata sidecar and what question it answers.
- Explain the shape-mismatch branch that should be unreachable.
- Explain how the module is tested without a running model server.
- Explain what this layer deliberately does *not* do.

## Motivation

Chapter 4 covered cosine similarity, normalisation and the asymmetric prefixes. Chapter 8 covered
batching, retries and cache-key design. This chapter reads the module as a unit — chiefly its cache
lifecycle, which is where the pieces meet and where the failure modes live.

Following the book's convention, mechanics already explained are referenced rather than repeated.

## The module's shape

Five public names, and the layering matters:

```
   OllamaEmbedder        HTTP client. Batching, retries, prefixes.       (ch 8)
   normalise             L2-normalise rows so dot product = cosine.      (ch 4)
   cache_key             Fingerprint model + prefix + every text.        (ch 8)
   embed_cached          The lifecycle. This chapter.
   EmbeddingError        One exception type for the whole layer.
```

`embed_cached` is the only function the rest of the system calls. Everything else is machinery it
composes. That is why `DenseRetriever` (chapter 17) takes a `vectors` array rather than an embedder —
the retriever is handed a finished matrix and knows nothing about how it was produced.

## Deep Explanation: the cache lifecycle

```python
def embed_cached(
    texts: list[str],
    embedder: OllamaEmbedder | None = None,
    prefix: str = DOC_PREFIX,
    cache_dir: Path | str = ".cache",
    progress: bool = True,
) -> np.ndarray:
    """Embed texts, reusing a cached array when the fingerprint matches."""
    embedder = embedder or OllamaEmbedder()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = cache_key(texts, embedder.model, prefix)
    vectors_path = cache_dir / f"embeddings-{key}.npy"
    meta_path = cache_dir / f"embeddings-{key}.json"

    if vectors_path.exists():
        vectors = np.load(vectors_path)
        if vectors.shape[0] == len(texts):
            if progress:
                print(f"  cache hit: {vectors_path} {vectors.shape}", file=sys.stderr)
            return vectors
        # Shape disagrees with the key, which should be impossible -- rebuild rather than
        # trust it.
        print(
            f"  cache at {vectors_path} has {vectors.shape[0]} rows, expected "
            f"{len(texts)} -- rebuilding",
            file=sys.stderr,
        )

    if progress:
        print(
            f"  embedding {len(texts)} texts with {embedder.model} "
            f"(one-time, then cached)",
            file=sys.stderr,
        )
    vectors = embedder.embed(texts, prefix=prefix, progress=progress)
    np.save(vectors_path, vectors)
    meta_path.write_text(
        json.dumps({...}, indent=2),
        encoding="utf-8",
    )
    return vectors
```

### The three paths

```
   embed_cached(texts)
        |
        +-- key = sha256(model, prefix, every text)
        |
        +-- file exists?
             |
             +-- yes, shape matches  -->  load, return          [common: instant]
             |
             +-- yes, shape differs  -->  warn, rebuild         [should be impossible]
             |
             +-- no                  -->  embed (~17 min), save, return
```

### Why the "impossible" branch exists

The cache key is a hash of every text, and `len(texts)` is one of the inputs. So a key hit *implies* a
matching row count — the shape check should never fail.

It is there because the consequence of being wrong is silent. If a `.npy` file were truncated, or
written by an older version of the code, or corrupted mid-write by an interrupted run, you would get a
matrix whose rows do not correspond to the current chunks. `DenseRetriever`'s constructor check
(chapter 17) would catch a *count* mismatch — but only if the counts differ. A file that happens to have
the right number of rows from a different corpus would pass both checks.

The reasoning is the same as chapter 14's empty-corpus guard: **seventeen minutes of recomputation is a
cheap premium against a silent misattribution of every score in the study.**

The message names both numbers, so the operator can tell immediately whether this is a truncation or a
different corpus.

### `mkdir(parents=True, exist_ok=True)`

Idempotent directory creation. `exist_ok=True` makes a second run a no-op rather than an error;
`parents=True` handles `.cache/answers/` where the parent may not exist. One line that removes a class
of first-run failure.

### `file=sys.stderr` throughout

All progress and diagnostic output goes to stderr, never stdout. This is deliberate: a driver's stdout
is its results table, and a caller redirecting it to a file should get the table, not seventeen minutes
of progress ticks interleaved with it.

### The metadata sidecar

```python
    meta_path.write_text(
        json.dumps(
            {
                "model": embedder.model,
                "prefix": prefix,
                "n_texts": len(texts),
                "dim": int(vectors.shape[1]),
                "key": key,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
```

The `.npy` file is opaque — a binary array with no provenance. Six months later a `.cache/` directory
containing `embeddings-dcc86aeb86a8db92.npy` answers no questions at all.

The sidecar answers them: which model, which prefix, how many texts, what dimensionality. It is never
read by code. Its entire purpose is that a human inspecting the cache can tell what is in it.

This is the same instinct as the gold set's `note` field (chapter 12): **artefacts that outlive the run
that produced them need enough metadata to be interpreted later.**

`int(vectors.shape[1])` rather than `vectors.shape[1]` — NumPy shape elements are `np.int64`, which
`json.dumps` refuses to serialise. A small compatibility detail that would otherwise raise at the very
end of a seventeen-minute run.

## Deep Explanation: testing without a server

The whole module is tested with no Ollama process running. The mechanism is a stub embedder that
satisfies the same shape as the real one:

```python
@dataclass
class StubEmbedder:
    """Returns a fixed vector for any query. Records what it was asked to embed."""

    vector: np.ndarray
    model: str = "stub"
    calls: list[str] = field(default_factory=list)

    def embed(self, texts: list[str], prefix: str = "", progress: bool = False) -> np.ndarray:
        self.calls.extend(prefix + text for text in texts)
        return np.tile(self.vector, (len(texts), 1)).astype(np.float32)
```

Two properties make it useful beyond merely avoiding the network.

**It records its calls.** `self.calls` is what lets the prefix test assert on what was actually sent:

```python
def test_applies_the_query_prefix(self) -> None:
    self._retriever().search(QUERY, top_k=1)
    self.assertTrue(self.embedder.calls[0].startswith("search_query: "))
```

A stub that only returned values could not verify that. Recording inputs turns the stub into an
observation point.

**It returns a *fixed* vector**, so similarity outcomes are fully determined by the hand-written
document vectors in the test. The chapter-4 ranking test works because the query vector is known
exactly:

```python
        # Chunk 1 (PEP 20) points exactly where the stub query vector points.
        self.vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        self.embedder = StubEmbedder(vector=np.array([0.0, 1.0], dtype=np.float32))
```

Two-dimensional vectors, hand-chosen, with an obvious correct ordering. Testing cosine ranking with
real 768-dimensional embeddings would prove nothing you could verify by inspection.

The cache tests use a temporary directory and count calls:

```python
def test_second_call_hits_the_cache(self) -> None:
    embedder = StubEmbedder(vector=np.array([1.0, 0.0], dtype=np.float32))
    with TemporaryDirectory() as tmp:
        first = embed_cached(["x", "y"], embedder, cache_dir=tmp, progress=False)
        self.assertEqual(len(embedder.calls), 2)
        second = embed_cached(["x", "y"], embedder, cache_dir=tmp, progress=False)
        self.assertEqual(len(embedder.calls), 2, "cache was not used")
        np.testing.assert_array_equal(first, second)
```

Asserting the call count is what actually tests the cache. Asserting only that the arrays match would
pass even if the cache were bypassed entirely.

## What this layer deliberately does not do

**No incremental or partial caching.** If the seventeen-minute run is interrupted, the work is lost. A
resume mechanism was considered and rejected: it adds state (a partial file, a row count, a resume
offset) and a class of bug where a resumed run mixes vectors from two model versions. The mitigation
chosen instead was the `--limit` smoke-test flag in the drivers, so the pipeline can be validated on 30
PEPs before committing to the full run.

**No dimensionality reduction, no quantisation.** 58 MB is small. Both would add approximation error to
a study about measurement accuracy (chapter 4's argument against ANN indexes applies identically).

**No concurrency.** Batches are sent sequentially. Parallel requests to a single local model server
contend for the same weights and add nondeterminism for a throughput gain that does not matter at
seventeen minutes, once.

Each of these is the same judgement: **the simplest thing that is correct, because complexity here buys
speed the study does not need and costs confidence it does.**

## Systems Perspective

Measured on this machine:

| | Value |
|---|---|
| Corpus | 19,763 chunks |
| Dimensionality | 768 |
| Sustained throughput | ~19 chunks/s |
| Full run | ~17 minutes |
| Cached array | 58 MB on disk, 58 MB resident |
| Cache hit | instant |

A short benchmark of 64 chunks measured 32 ms per chunk, implying ~10.6 minutes. The real sustained
rate was slower. Worth carrying forward as a general caution: **a short benchmark overestimates
sustained throughput**, because it misses thermal and memory-pressure effects that appear over minutes.

## Common Mistakes

**Writing progress to stdout.** Corrupts a redirected results table.

**A cache with no provenance file.** An opaque binary answers no questions later.

**Trusting a cache-key hit without checking the shape.** The key should guarantee it; the cost of being
wrong is every score misattributed.

**Serialising NumPy scalars to JSON without casting.** Raises at the end of a long run.

**Stubs that do not record their inputs.** You cannot assert on what was sent.

**Testing cosine ranking with real high-dimensional embeddings.** You cannot verify the expected order
by inspection, so the test asserts whatever the model happened to do.

**Building a resume mechanism before it is needed.** More state, more failure modes; a smoke-test flag
solves the same problem.

## Interview Insight

> **"How do you handle an expensive precomputation step?"**

Cache it on a fingerprint of everything that determines the output — for embeddings that is the model
name, the asymmetric prefix, and the exact text of every input. Write a metadata sidecar alongside the
binary so the cache is interpretable later. Report progress with an ETA to stderr, because a silent
seventeen-minute operation is indistinguishable from a hang and the natural response to a hang is to
kill it.

And validate on a subset first. I added a `--limit` flag so the whole pipeline could be exercised on 30
documents before committing to the full run — which is a cheaper answer than building resume logic, and
it does not introduce state that can be wrong.

> **"How do you test code that depends on an external model?"**

Stub the client at the smallest interface that satisfies the consumer, and make the stub *record* its
inputs so you can assert on what was sent — that is how I verify the asymmetric query prefix is applied,
which is a silent-degradation bug otherwise. Then use hand-written low-dimensional vectors so the
expected ranking is verifiable by inspection.

The result is that the entire suite runs in a tenth of a second with no model server, so it runs on
every change rather than occasionally.

## Debugging Tip

To find out what a cache directory contains, read the sidecars rather than the arrays:

```bash
cat .cache/embeddings-*.json
```

```json
{ "model": "nomic-embed-text", "prefix": "search_document: ",
  "n_texts": 19763, "dim": 768, "key": "dcc86aeb86a8db92" }
```

If `n_texts` does not match your current chunk count, the chunker changed and a rebuild is pending. If
`prefix` is wrong, dense retrieval is silently degraded and the fix is a re-embed, not a retriever
change.

## Summary

- `embed_cached` is the only entry point the system uses; everything else is machinery it composes.
- Three paths: cache hit, impossible-but-guarded shape mismatch, and cold build.
- The shape check should be unreachable and exists because the failure it prevents is silent.
- All diagnostics go to stderr so a redirected stdout is a clean results table.
- The metadata sidecar makes an opaque binary interpretable months later; it is never read by code.
- Stub embedders record their inputs, which is what makes the prefix assertion possible, and
  hand-written 2-D vectors make ranking expectations verifiable by inspection.
- The layer deliberately omits resume, quantisation and concurrency — simplicity chosen over speed the
  study does not need.
- ~17 minutes sustained for 19,763 chunks; a 64-chunk benchmark predicted ~10.6, so short benchmarks
  overestimate.

## Key Takeaways

1. Fingerprint every input that affects the output, and store provenance next to the artefact.
2. Diagnostics to stderr, results to stdout.
3. Make stubs observation points, not just value sources.
4. A smoke-test flag is usually cheaper and safer than resume logic.

## Why the Next Chapter Exists

Chapter 19 reads the generation layer: how retrieved chunks become a prompt, what the prompt
deliberately withholds and why, how citations are extracted, and the two places where the automatic
answer metrics mark a correct answer wrong — cases that argue precisely for the one component this
project chose not to build.
