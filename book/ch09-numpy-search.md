# Chapter 9 — Brute-Force Nearest Neighbours with NumPy

## Learning Objectives

- Explain why NumPy is fast and what "vectorised" actually means.
- Explain what **BLAS** is and why one line of Python can saturate your memory bandwidth.
- Explain `argpartition` and why it beats a full sort for top-k selection.
- Work through a real investigation of a floating-point warning: how to decide whether it is a bug or
  noise, and what to do either way.
- Explain when brute force stops being the right choice.

## Motivation

Chapter 4 established that dense retrieval is one matrix-vector product. This chapter is about that
product: why it is fast, how to extract a top-k from its output efficiently, and what to do when the
library underneath tells you something alarming.

That last part is the most useful thing in the chapter. While building this project, dense retrieval
began emitting `RuntimeWarning: divide by zero encountered in matmul` — in an operation containing no
division. The investigation that followed is a template for a situation you will meet repeatedly:
**something in a dependency looks wrong, and you must decide whether your results are affected.**

## First Principles

### Why loops are slow and arrays are fast

Compute 19,763 dot products in pure Python:

```python
scores = []
for row in matrix:                   # 19,763 iterations
    total = 0.0
    for a, b in zip(row, query):     # 768 iterations each
        total += a * b
    scores.append(total)
```

That is 15.2 million iterations of the interpreter loop. Each one boxes floats into Python objects,
dispatches `__mul__` and `__add__` through the type system, and allocates. Seconds, not milliseconds.

NumPy replaces it with:

```python
scores = matrix @ query
```

One line. Two things make it fast:

**Contiguous typed memory.** A NumPy array is a flat block of raw `float32` values, not a list of
pointers to Python objects. 19,763 × 768 float32 is one 58 MB block. The CPU reads it sequentially,
which is what hardware prefetchers are built for.

**The loop moves into compiled code.** The multiply-add loop runs in C, with no interpreter, using SIMD
instructions that process four or eight floats per instruction.

This is what **vectorised** means: not "no loop" — the loop still exists — but "the loop is not in
Python."

### BLAS

NumPy does not implement matrix multiplication itself. It delegates to **BLAS** (Basic Linear Algebra
Subprograms), a decades-old interface with heavily optimised implementations: OpenBLAS, Intel MKL,
Apple's Accelerate.

These are tuned to the specific CPU — cache blocking, register allocation, the right SIMD width. A good
BLAS is typically 10–100× faster than a naive C loop over the same data.

Which BLAS you have matters, and you can ask:

```python
>>> np.show_config()
blas:
    name: accelerate
```

On this machine it is **Apple Accelerate**. Remember that; it is the subject of the investigation
below.

### The operation, concretely

```
     _matrix                    query           similarities
   (19763, 768)               (768,)             (19763,)

   +-----------------+         +---+            +---+
   | chunk 0 ....... |         | . |            | . |   cosine(chunk 0, query)
   | chunk 1 ....... |    @    | . |     =      | . |   cosine(chunk 1, query)
   |      ...        |         | . |            | . |
   | chunk 19762 ... |         | . |            | . |
   +-----------------+         +---+            +---+
```

Because both sides were L2-normalised (chapter 4), each output element *is* a cosine similarity. About
15 million multiply-adds; a few milliseconds.

## Deep Explanation: top-k without sorting

We have 19,763 scores and want the highest 50.

The obvious approach sorts everything: `np.argsort(-similarities)[:50]`. That is `O(n log n)` — roughly
19,763 × 14.3 ≈ 283,000 comparisons — to obtain 50 values, discarding 19,713.

`np.argpartition` does better. It rearranges the array so that the k smallest are before position k and
the rest after, without ordering *within* either group. That is `O(n)` — one pass.

```python
        depth = min(depth, similarities.shape[0])
        # argpartition finds the top `depth` without sorting all 19,763 scores, then only
        # those are sorted.
        candidates = np.argpartition(-similarities, depth - 1)[:depth]
        return candidates[np.argsort(-similarities[candidates])].tolist()
```

Read it in three steps:

1. `np.argpartition(-similarities, depth - 1)` — negate so "largest" becomes "smallest", partition so
   the top `depth` indices occupy the first `depth` slots, in arbitrary order.
2. `[:depth]` — take those indices.
3. `np.argsort(-similarities[candidates])` — sort just those 50 properly.

Total: one `O(n)` pass plus an `O(k log k)` sort of 50 elements, instead of `O(n log n)` over 19,763.

```
  argsort:       [ sort all 19,763 ]  -> take 50            283,000 comparisons
  argpartition:  [ one pass, O(n) ]  -> sort 50            ~19,763 + 280
```

The `min(depth, similarities.shape[0])` guard matters: `argpartition` raises if the pivot index exceeds
the array length. Requesting more results than there are chunks would crash without it. Pinned:

```python
def test_top_k_larger_than_corpus_is_safe(self) -> None:
    ranked = self._retriever().search(QUERY, top_k=50)
    self.assertEqual(len(ranked), 3)
```

## Deep Explanation: the warning that was not ours

This section is a worked investigation. It is the most transferable content in the chapter.

### The symptom

Running dense retrieval produced:

```
speceval/retrievers.py:157: RuntimeWarning: divide by zero encountered in matmul
speceval/retrievers.py:157: RuntimeWarning: overflow encountered in matmul
speceval/retrievers.py:157: RuntimeWarning: invalid value encountered in matmul
```

Line 157 is `similarities = self._matrix @ self._embed_query(query.text)`.

Stop and notice what is strange: **matrix multiplication contains no division.** A "divide by zero" from
an operation with no division means either something upstream has produced non-finite values, or the
warning is misattributed.

The stakes are high. If the matrix contains `NaN`, every similarity score is garbage and every dense
result in the study is void. This is not a warning you can note and move past.

### Step 1: look for the obvious cause

`NaN` or `inf` in the data would explain it. Check directly:

```
raw   nan: 0  inf: 0
row norms: min 0.999999463558197 max 1.0000005960464478
zero rows: 0
norm  nan: 0  inf: 0
```

No `NaN`, no `inf`, no zero rows, and every row norm is 1.0 to within float32 precision — so
normalisation worked correctly. The inputs are clean.

And yet, with warnings promoted to errors, the multiply still raises:

```
matmul FloatingPointError: divide by zero encountered in matmul
```

Clean inputs, failing operation. So either NumPy is computing something wrong, or the *warning* is
wrong.

### Step 2: check the arithmetic against a reference

The only way to settle it is to compute the same thing a different way and compare. Use float64 — more
precision, and a different code path:

```
max abs diff matmul vs float64 loop: 9.667957090453427e-08
argsort identical: True
self-similarity (should be 1.0): 1.0
range: 0.3510034680366516 1.0
```

Read those four lines carefully, because together they are conclusive:

- Maximum deviation from the float64 reference is **9.7e-08** — the expected rounding error for
  float32, which carries about 7 decimal digits.
- The **top-20 ordering is identical**. Even if magnitudes differed slightly, ranking is what we
  actually use, and it is unaffected.
- **Self-similarity is exactly 1.0**, which is the strongest possible check: a normalised vector dotted
  with itself must be 1, and it is.
- The **range is [0.351, 1.0]**, sensible cosine territory with nothing anomalous.

The arithmetic is correct. The warning is spurious.

### Step 3: find the boundary

A spurious warning still has a cause. Try different matrix sizes:

```
  n=3:   no warning
  n=64:  divide by zero encountered in matmul
  n=100: divide by zero encountered in matmul
  n=633: divide by zero encountered in matmul
```

Nothing below 64 rows, everything at and above. That threshold is the signature of a **blocked BLAS
kernel**: below a size cutoff, a simple loop runs; above it, a tiled, SIMD-heavy path takes over. That
path evidently computes on padding lanes — values outside the real data — and those computations set
floating-point status flags. NumPy reads the flags afterwards and reports them, without knowing they
came from padding.

So it is an integration issue between NumPy 1.26.4 and Apple Accelerate. Not our bug, and not affecting
our numbers.

### Step 4: fix it narrowly, and pin the reasoning

```python
    def search_chunks(self, query: Query, depth: int) -> list[int]:
        # errstate: numpy 1.26.4 built against Apple Accelerate raises a spurious
        # "divide by zero encountered in matmul" for matrices above ~64 rows -- the FP
        # status flags are set by the blocked BLAS path, not by the arithmetic. Verified
        # harmless at the time: results matched a float64 reference to 1e-7 with identical
        # ordering. Not reproducible on numpy 2.5.2 (same Accelerate BLAS), so it was an
        # upstream integration bug, but the guard stays for anyone running 1.26.x.
        # Suppressed narrowly here rather than globally, so a genuine numerical fault
        # elsewhere still surfaces; TestPlatformNumerics pins the arithmetic either way.
        with np.errstate(all="ignore"):
            similarities = self._matrix @ self._embed_query(query.text)
```

Four deliberate properties:

**Narrow scope.** `np.errstate` is a context manager around one statement. A global `np.seterr(all=
"ignore")` would hide a real numerical fault anywhere else in the codebase.

**The reasoning is in the code**, not in a commit message nobody will find. Anyone encountering this
line learns why it exists and what was checked.

**Version-specific and honest.** A later check under numpy 2.5.2 — same Accelerate BLAS — showed the
warning **gone**, confirming an upstream bug since fixed. The comment says so rather than implying all
Accelerate builds are affected. The guard remains for anyone on 1.26.x.

**Backed by tests that would catch a real fault.** This is the part that makes the suppression
defensible rather than lazy:

```python
class TestPlatformNumerics(unittest.TestCase):
    """Guards the suppressed FP warning in DenseRetriever.search_chunks.
    ...
    The retriever suppresses it for anyone still on 1.26.x, so this test exists to make sure
    the suppression can only ever hide a cosmetic flag and not a real numerical fault: if
    the fast path stops matching a float64 reference, this fails regardless of numpy
    version or BLAS backend.
    """

    def test_matmul_matches_float64_reference(self) -> None: ...
    def test_ordering_is_unaffected(self) -> None: ...
    def test_self_similarity_is_one(self) -> None: ...
    def test_no_nan_or_inf_produced(self) -> None: ...
```

Four tests, version-independent and backend-independent. If a future NumPy, BLAS, or platform ever
computes this wrongly, they fail — even though the warning is suppressed.

### The generalisable procedure

1. **Read the warning literally.** "Divide by zero" in an operation with no division is information.
2. **Check the data** for the obvious cause.
3. **Compute the same thing a different way** and compare. This is the step that actually decides it.
4. **Find the boundary** — size, type, backend. A threshold identifies the mechanism.
5. **Fix narrowly, document the reasoning in the code, and add a test that would catch the real
   version of the problem.**

Step 3 is the one people skip, and it is the only one that converts a suspicion into a conclusion.

## Systems Perspective

Dense retrieval here is **memory-bandwidth bound**, not compute bound. Each of the 15 million
multiply-adds reads two floats and does one fused operation; modern CPUs can issue those far faster
than memory can supply 58 MB.

Consequences:

**`float32`, not `float64`.** Halves the bytes moved, therefore roughly halves the time, and chapter
4's precision check showed float32 is plenty — the ordering is identical to a float64 reference.

**SIMD helps less than you would hope.** If you are waiting on memory, processing eight floats per
instruction does not help. (A related measurement from the same author's ANN work: on an Apple M4,
single-thread memory bandwidth caps around 24 GB/s, so a linear scan is memory-bound and SIMD buys at
most ~1.19× there.)

**58 MB does not fit in cache** but is read sequentially, which prefetchers handle well. This is why
brute force is viable at all.

### When brute force stops being right

| Corpus size | Memory (768-dim float32) | Brute force |
|---|---|---|
| 20 K | 58 MB | milliseconds — ideal |
| 1 M | 2.9 GB | ~100 ms — borderline |
| 100 M | 290 GB | impossible on one machine |

Above roughly a million vectors you need an **approximate nearest neighbour** index — HNSW, IVF,
product quantisation — trading exactness for sublinear search.

`speceval` is at 20 K and explicitly declines to add one:

```
- **No vector database and no ANN index.** Brute-force cosine over a few thousand chunks is
  correct at this scale; an HNSW index here would be decoration.
```

Note the argument is not only "small enough". It is also that an approximate index adds its *own* error,
so a missed document could no longer be attributed to the ranking rather than the index. In a
measurement study, exactness is a requirement, not a luxury.

## Common Mistakes

**Suppressing a warning without investigating.** You may be hiding a real fault. Investigate, then
suppress narrowly with the reasoning recorded.

**Global `np.seterr`.** Silences the whole program, including the parts where you needed the warning.

**`argsort` when you want a top-k.** `O(n log n)` for `k` results.

**No bounds guard on `argpartition`.** Raises when `k` exceeds the array length.

**`float64` by default.** Doubles memory traffic in a bandwidth-bound operation for precision you do
not need.

**Reaching for an ANN index too early.** Below a million vectors it adds error, dependencies and
tuning for no benefit.

**Assuming your BLAS is the same everywhere.** `np.show_config()` differs across machines, and as this
chapter demonstrates, that difference can produce alarming output.

## Interview Insight

> **"Why is NumPy faster than a Python loop?"**

Contiguous typed memory instead of pointers to boxed objects, and the loop executes in compiled BLAS
code with SIMD instead of in the interpreter. For a 19,763 × 768 matrix that is 15 million multiply-adds
in a few milliseconds versus seconds in pure Python.

Worth adding: for this operation the bottleneck is memory bandwidth, not arithmetic, which is why
`float32` roughly halves the time and why SIMD helps less than you would expect.

> **"You see a `RuntimeWarning` from a library. What do you do?"**

This is a judgement question, and the strong answer is a procedure. Read the warning literally — "divide
by zero" from matrix multiplication is a contradiction worth noticing. Check the data for the obvious
cause. Then **compute the same result a different way and compare**, because that is the only step that
actually decides whether your numbers are affected. Find the boundary — here, matrices above 64 rows —
because a threshold identifies the mechanism, in this case a blocked BLAS kernel setting flags on
padding lanes.

Then fix narrowly with a context manager rather than globally, put the reasoning in the code, and add a
test comparing the fast path to a reference so that if it ever *is* a real fault, it fails loudly
despite the suppression.

> **"When would you use an ANN index?"**

Above roughly a million vectors, where brute force stops being interactive. Below that it adds
approximation error, a dependency, and tuning parameters for no gain — and if you are measuring
retrieval quality, that added error means you can no longer attribute a miss to the ranking rather than
the index.

## Performance Insight

`argpartition` before `argsort` is a free win whenever `k << n`, and the crossover is low — worth doing
for anything above a few hundred candidates. It is one extra line and it is the difference between
sorting 19,763 elements and sorting 50.

## Debugging Tip

To sanity-check any similarity computation in one line, dot a normalised vector with itself. The answer
must be 1.0:

```python
self.assertAlmostEqual(float(similarities[7]), 1.0, places=5)
```

If it is not, something upstream is wrong — normalisation, indexing, a stale cache — and no retrieval
result means anything until it is. This project keeps that assertion as a permanent test rather than a
one-off check.

## Summary

- NumPy is fast because of contiguous typed memory and compiled, SIMD-enabled loops. "Vectorised" means
  the loop is not in Python.
- Matrix multiplication is delegated to BLAS; which BLAS you have is observable and consequential.
- `argpartition` gets a top-k in `O(n)` instead of sorting in `O(n log n)`; guard the pivot against
  short arrays.
- A spurious `divide by zero` warning from `matmul` on NumPy 1.26.4 with Apple Accelerate was
  investigated rather than suppressed: inputs clean, output matching a float64 reference to 9.7e-08 with
  identical ordering, threshold at 64 rows indicating a blocked BLAS kernel. Fixed narrowly, documented
  in place, absent on NumPy 2.5.2, and pinned by four version-independent tests.
- The operation is memory-bandwidth bound, so `float32` matters more than SIMD.
- Brute force is right below roughly a million vectors — and in a measurement study it is *preferable*,
  because an approximate index adds error you cannot separate from the ranking's.

## Key Takeaways

1. Vectorise by moving the loop into compiled code, not by removing it.
2. `argpartition` then `argsort` whenever `k << n`.
3. Never suppress a numerical warning without computing the same result a different way first.
4. If you must suppress, do it narrowly, record why in the code, and add a test that would catch the
   real fault.

## Why the Next Chapter Exists

Part II is complete. We have the machinery: three ranking strategies, metrics, immutable records and a
strategy protocol, a model server with determinism and caching, and fast exact search.

Part III turns to what makes *this* project distinct. Chapter 10 examines the corpus itself — the PEP
series, its RFC-822 headers, its nine-value status enumeration and its supersession graph — because the
central finding of this study is only possible in a corpus whose documents record their own standing.
