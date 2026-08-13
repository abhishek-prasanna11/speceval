# Chapter 4 — Meaning as Geometry: Embeddings

## Learning Objectives

- Explain what an embedding is and what problem it solves that BM25 cannot.
- Explain why **cosine similarity** rather than Euclidean distance, and what normalisation buys.
- Explain the **distributional hypothesis** — the assumption embeddings rest on.
- Explain why embedding models are **asymmetric**, and what breaks if you ignore that.
- Read `speceval`'s dense retriever and say why each line is there.
- State the failure mode dense retrieval *introduces*, which is this project's central subject.

## Motivation

Chapter 3 left us with a retriever that cannot connect *"postpone"* to *"postponed"*, let alone to
*"deferred"*. The obstacle is structural: BM25 compares strings for identity, and identity has no
notion of similarity. `postpone` and `postponed` are exactly as different, to BM25, as `postpone`
and `banana`.

Patching this with string tricks — stemming, synonym lists — gets you a little way and then stops.
Stemming would unify `postpone`/`postponed`; nothing string-based will ever unify
`postpone`/`deferred`, because they share no substring at all yet mean nearly the same thing here.

What we actually want is a representation in which **similar meanings are near each other**, with
"near" being something we can compute. That is an embedding.

## First Principles

### Words as vectors

Suppose you could place every word at a point in space such that words with related meanings land
close together. Two dimensions, for illustration:

```
                    formal ^
                           |
              deferred  *  |  * postponed
                           |    * postpone
        delayed *          |
                           |
   ------------------------+------------------------> temporal
                           |
                           |     * banana
                           |
```

Now similarity is *distance*. `postpone` and `deferred` are close, so they are similar, and no
string comparison was involved. The vocabulary mismatch problem dissolves — not patched, dissolved,
because the representation itself never depended on spelling.

The real thing uses more dimensions. The model in this project uses **768**, and each dimension is
not interpretable — there is no "formality axis". The dimensions are whatever the model found
useful, and only the *geometry* has meaning.

### Where do the coordinates come from?

The **distributional hypothesis**: *words that appear in similar contexts have similar meanings.*
Put crudely, you know what a word means by the company it keeps.

This is not a fact about language, it is a *usable approximation* that happens to work remarkably
well. Train a model to predict a word from its neighbours (or vice versa) across billions of
sentences, and the internal representation it develops places `postpone` and `deferred` near each
other — because in the text it read, those words were surrounded by the same other words.

It also explains a characteristic failure. Words that appear in similar contexts but mean *opposite*
things — `hot`/`cold`, `increase`/`decrease`, `accepted`/`rejected` — often end up close together,
because their contexts are nearly identical. Hold onto that. It matters enormously in chapter 11:
a **superseded** proposal and its **live** replacement discuss the same topic in the same
vocabulary, so they sit almost on top of each other in embedding space, and the embedding has no way
to prefer the live one.

### From words to documents

We need to place *chunks* in space, not words. Modern practice hands the whole passage to a
transformer model trained for the purpose, which reads it in context and emits one vector. That
process is called **embedding** the text, and the model is an **embedding model**.

`speceval` uses `nomic-embed-text`, run locally, producing 768 dimensions. Verified from a live call:

```
n= 3 dim= 768 keys= ['model', 'embeddings', 'total_duration', ...]
```

### Comparing two vectors

Given a query vector `q` and a document vector `d`, how similar are they?

**Euclidean distance** — the straight-line gap — is the obvious first thought, and it is the wrong
one. It is sensitive to vector *magnitude*, and magnitude in embedding space tends to track
incidental things like passage length rather than meaning. A long chunk and a short chunk about the
identical topic can have very different magnitudes while pointing in the same direction.

**Direction** is what carries meaning. So measure the *angle*:

```
                    a · b            Σ aᵢbᵢ
   cos(a, b) = --------------- = --------------------
                 ‖a‖ · ‖b‖       √(Σaᵢ²) · √(Σbᵢ²)
```

This is **cosine similarity**. It is 1.0 when the vectors point the same way, 0 when perpendicular,
−1 when opposite. Length is divided out entirely.

```
        b
        ^
        |    a
        |   /
        |  /  θ           cos θ = 1.0  -> identical direction
        | /               cos θ = 0.0  -> unrelated
        |/                cos θ = -1.0 -> opposite
        +---------->
```

### The normalisation trick

Computing that fraction for every document on every query means recomputing `‖d‖` — a square root
over 768 terms — 19,763 times per query. Wasteful, because `‖d‖` never changes.

So normalise every document vector *once*, at build time, to unit length. Then `‖d‖ = 1`, and if the
query is normalised too:

```
   cos(q, d) = q · d          (a plain dot product)
```

Every query becomes one matrix-vector product. `speceval` does exactly this, in `speceval/embed.py`:

```python
def normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so that a dot product is a cosine similarity.

    Normalising once at build time turns every later query into a single matrix-vector
    product, instead of recomputing magnitudes on each search.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against a zero vector, which would otherwise produce NaN and silently poison
    # every similarity score computed against it.
    norms[norms == 0] = 1.0
    return matrix / norms
```

The guard is worth pausing on. A zero vector would divide by zero, giving `NaN`. And `NaN`
propagates: every comparison involving it is false, so `NaN` scores sort unpredictably and corrupt
results *without raising anything*. Three lines to prevent a silent, hard-to-diagnose failure. It is
pinned by a test:

```python
def test_zero_vector_does_not_produce_nan(self) -> None:
    # A zero row would divide by zero and poison every similarity computed with it.
    matrix = normalise(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    self.assertFalse(np.isnan(matrix).any())
```

## Mental Model

BM25 is a **crossword solver**: it works on letters and exact matches. Ask it for a five-letter word
meaning "delay" and it needs the letters.

An embedding model is someone who has **read enormously** and has intuitions about what things mean.
Ask it for something meaning "delay" and it offers *postpone, defer, hold off, table* without
needing a single shared letter.

And the analogy predicts the weakness too. The well-read person is excellent at gist and unreliable
on exact tokens — they might not recall whether the flag was `--no-cache` or `--nocache`, because
those *mean* the same thing even though only one works. That is precisely where BM25 keeps winning.

## Deep Explanation

### Asymmetric models: the prefix that is easy to get wrong

Here is a detail that silently costs a large fraction of dense retrieval quality when missed.

A query and a document are not the same kind of text. *"how do I postpone annotations"* is a short
question. The passage that answers it is long, declarative, and does not look like a question at
all. If both are embedded by the same function, the query lands near other *questions* rather than
near its *answers*.

Models trained for retrieval fix this by being **asymmetric** — they are told which role each piece
of text plays, conventionally via a prefix. For `nomic-embed-text`:

```python
# Documented prefixes for the nomic-embed-text family.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
```

*(This convention comes from the model's own documentation — one of the two external facts in this
book not derived from the project's source.)*

The module docstring in `speceval/embed.py` explains the stakes:

```python
"""`nomic-embed-text` is an *asymmetric* model: it expects documents and queries to be
prefixed differently, so a query embedding lands near documents that answer it rather than
near documents that merely resemble it. Getting these prefixes wrong silently degrades
dense retrieval, which would then be blamed on dense retrieval rather than on the prefix.
"""
```

The last clause is the important one. Omit the prefixes and dense retrieval scores worse. You would
then write "dense retrieval underperformed on this corpus" — a *conclusion about the method* caused
by a *bug in your plumbing*. In a study whose entire output is comparative conclusions, that class
of error is the most dangerous kind, because the result looks plausible and no test fails.

`speceval` pins it:

```python
def test_applies_the_query_prefix(self) -> None:
    # Asymmetric model: a query embedded without its prefix silently retrieves worse.
    self._retriever().search(QUERY, top_k=1)
    self.assertTrue(self.embedder.calls[0].startswith("search_query: "))
```

### Searching: brute force, and why that is correct here

With normalised vectors, search is:

1. Embed the query (a network call to the model server).
2. One matrix-vector product: `(19763, 768) @ (768,)` → 19,763 similarity scores.
3. Take the top *k*.

From `speceval/retrievers.py`:

```python
    def search_chunks(self, query: Query, depth: int) -> list[int]:
        with np.errstate(all="ignore"):
            similarities = self._matrix @ self._embed_query(query.text)
        depth = min(depth, similarities.shape[0])
        # argpartition finds the top `depth` without sorting all 19,763 scores, then only
        # those are sorted.
        candidates = np.argpartition(-similarities, depth - 1)[:depth]
        return candidates[np.argsort(-similarities[candidates])].tolist()
```

`np.argpartition` is worth knowing. Fully sorting 19,763 scores is `O(n log n)`; partitioning so
that the top `depth` are in front is `O(n)`, and only those few are then sorted. For `depth = 50`
out of 19,763 that is a real saving, and it costs one extra line.

Comparing every document to the query is called **brute-force** or **exact** search. Production
systems at scale use approximate nearest-neighbour indexes (HNSW, IVF, product quantisation) that
trade exactness for speed. `speceval` deliberately does not:

```python
    """Rung 2 -- embedding cosine similarity, brute force.

    Brute force is the correct choice at this scale: one (19763, 768) by (768,) matrix
    product is a few milliseconds, and an ANN index would add approximation error to a
    study whose entire subject is measurement accuracy.
    """
```

Two independent reasons, and the second is the stronger. An approximate index would introduce a
second source of error — you could no longer tell whether a missed document was the ranking's fault
or the index's. In a measurement study, exactness is not a luxury.

### One vector per corpus, cached

Embedding 19,763 chunks took roughly **17 minutes** on this machine, measured, at about 19 chunks
per second. That is far too slow to repeat on every run, so the vectors are cached to disk, keyed on
a fingerprint of everything that could change them:

```python
def cache_key(texts: list[str], model: str, prefix: str) -> str:
    """Fingerprint everything that affects the vectors.

    Model, prefix and the exact text of every chunk are all inputs -- so changing the
    chunking configuration or swapping the model produces a different key and the cache
    is rebuilt rather than silently reused.
    """
    digest = hashlib.sha256()
    digest.update(model.encode())
    digest.update(prefix.encode())
    digest.update(str(len(texts)).encode())
    for text in texts:
        digest.update(hashlib.sha256(text.encode()).digest())
    return digest.hexdigest()[:16]
```

The three inputs are exactly the three things that determine the output. Change the chunk size and
every chunk's text changes, so the key changes, so the cache rebuilds. Swap the model, same. Change
the prefix, same. A cache keyed on anything less — a file path, a corpus name, a timestamp — would
eventually serve vectors that do not correspond to the current chunks, and nothing would report an
error. Chapter 18 covers this module in full.

## Systems Perspective

Where the time actually goes, measured on this project:

| Stage | Cost | Note |
|---|---|---|
| Corpus embedding | ~17 min, once | 19,763 chunks at ~19/s, then cached |
| Query embedding | tens of ms | network round trip to the model server, dominates |
| Matrix product | ~few ms | 15M multiply-adds in BLAS |
| Top-k selection | sub-ms | `argpartition` |

Dense retrieval measures **p50 ≈ 37 ms** against BM25's **≈ 28 ms** on this corpus. Almost the whole
difference is the query-embedding round trip, not arithmetic. If you wanted dense retrieval to be
fast, the thing to remove is the network hop, not the matmul.

Memory: 19,763 × 768 float32 is about **58 MB**, small enough to keep resident. This is why brute
force is viable — the whole corpus is one array in RAM. At a hundred million vectors, it would not
be, and that is when ANN indexes stop being optional.

## Common Mistakes

**Skipping the asymmetric prefixes.** Degrades quality silently and gets misattributed to the
method.

**Using Euclidean distance on unnormalised vectors.** Magnitude tracks incidental properties like
length. Use cosine, or normalise and use dot product.

**Recomputing norms per query.** Normalise once at build time.

**No zero-vector guard.** `NaN` propagates silently through comparisons and corrupts rankings
without raising.

**Reaching for an ANN index by reflex.** Below a few hundred thousand vectors, brute force is
simpler, exact, and fast enough. In a measurement context it is also *more correct*.

**Caching embeddings on a weak key.** If the key does not cover the model, the prefix, and the
exact text, you will eventually compare a query against vectors of a different corpus.

## Interview Insight

> **"What is an embedding?"**

A learned mapping from text to a fixed-length vector, arranged so that semantically similar text
lands nearby, with similarity measured by cosine of the angle. It rests on the distributional
hypothesis — text appearing in similar contexts means similar things — and it solves vocabulary
mismatch, which lexical retrieval structurally cannot.

Then the sentence that shows real use: *the same property that makes it work also makes it confuse
things that appear in identical contexts but differ in status — which is exactly the failure this
project measures, where a superseded specification and its live replacement are near-indistinguishable
in embedding space.*

> **"Cosine similarity or Euclidean distance?"**

Cosine, because magnitude in embedding space tends to encode incidental properties like passage
length rather than meaning, and cosine divides magnitude out. In practice: normalise your vectors
once at build time and use a dot product, which is arithmetically identical to cosine and much
cheaper per query.

> **"When is brute force acceptable?"**

Up to a few hundred thousand vectors on one machine — 19,763 × 768 float32 is 58 MB and one matmul.
It is also strictly *preferable* when you are measuring something, because an approximate index adds
its own error and you lose the ability to attribute a miss to the ranking rather than the index.

## Debugging Tip

When dense retrieval underperforms, check these four before touching the model, in order:

1. **Are the prefixes applied, and the right way round?** By far the most common cause.
2. **Are the vectors normalised, and is the guard present?** Print `np.isnan(matrix).any()`.
3. **Does the cache key cover the model and prefix?** A stale cache is invisible.
4. **Sanity-check the geometry.** Embed a passage and compare it to itself: cosine must be 1.0.
   `speceval` tests exactly this (`test_self_similarity_is_one`), because if it is not 1.0 something
   upstream is broken and no retrieval result means anything.

## Summary

- Embeddings map text to vectors so that similar meanings are geometrically close, dissolving
  vocabulary mismatch rather than patching it.
- They rest on the distributional hypothesis, which also explains their weakness: things used in
  identical contexts sit together even when their *status* differs.
- Cosine similarity compares direction and ignores magnitude; normalise once and a dot product is
  identical and cheaper.
- Retrieval models are asymmetric; the query/document prefixes are mandatory and their absence is
  silent.
- Brute force over 19,763 × 768 is a few milliseconds and 58 MB, and is the *correct* choice in a
  measurement study because it introduces no approximation error.
- Corpus embedding is slow (~17 min here) and must be cached on a key covering model, prefix and
  exact text.

## Key Takeaways

1. Embeddings solve the problem BM25 cannot, and introduce one BM25 does not have.
2. Normalise once; then cosine is a dot product.
3. Get the asymmetric prefixes right or your conclusions will be about your plumbing.
4. Brute force is not a compromise at this scale — it is the rigorous choice.

## Why the Next Chapter Exists

We now have two scoring functions built on completely different principles. The obvious question is
which is better, and we have no way to answer it. "It looked better on the three queries I tried" is
not an answer, and it is how most retrieval systems are actually tuned.

Chapter 5 builds the measuring instrument: what relevance judgements are, how Recall@K and nDCG@K
are defined and what each ignores, and why a metric implementation must itself be tested before any
number it produces can be believed.
