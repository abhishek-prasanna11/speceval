# Chapter 7 — Python for Measurement Code

## Learning Objectives

- Explain what makes measurement code different from application code.
- Explain **dataclasses** and why `frozen=True` matters for experimental records.
- Explain **structural typing** via `typing.Protocol`, and why it is the right abstraction for a
  family of interchangeable strategies.
- Explain `field(init=False)` and `__post_init__`, and when derived state belongs in a constructor.
- Explain why `defaultdict` appears throughout the scoring code.
- Identify the specific choices in this codebase that make its results reproducible.

## Motivation

Measurement code has a requirement application code does not: **its output must be trustworthy in a
way you cannot check by looking at it.**

If a web handler is wrong, a page breaks and someone notices. If an evaluation harness is wrong, it
produces a number. The number is plausible. It goes in a table. The table goes in a README. Nobody
notices, possibly ever.

So measurement code is written defensively in a particular direction — not against crashes, but
against *silently producing a wrong number*. This chapter covers the Python features `speceval` uses
for that purpose, and why each was chosen.

## First Principles

### Records that cannot change under you

An experiment is a set of records: this query, these labels, this result. A record that is mutated
after creation is a record you can no longer reason about — if a query's relevance labels are
modified halfway through an evaluation loop, the first half of your average measured something
different from the second half, and nothing will report it.

Python's answer is the **dataclass**: a class whose fields are declared as annotations, with
`__init__`, `__repr__` and `__eq__` generated for you.

```python
from dataclasses import dataclass

@dataclass
class Pep:
    number: int
    title: str
    status: str
```

That alone gives you a constructor, printable output, and value equality. Adding `frozen=True` makes
instances immutable — assignment raises `FrozenInstanceError`:

```python
@dataclass(frozen=True)
class Pep:
    number: int
    ...
```

`speceval` freezes every record type that represents *data being studied*:

| Type | Module | Represents |
|---|---|---|
| `Pep` | `corpus.py` | One parsed document |
| `Chunk` | `chunking.py` | One retrievable fragment |
| `Query` | `retrievers.py` | One gold-set entry |

And deliberately does **not** freeze the types that represent *machinery*:

| Type | Module | Why mutable |
|---|---|---|
| `BM25Retriever` | `retrievers.py` | Builds an index in `__post_init__` |
| `DenseRetriever` | `retrievers.py` | Holds a query-embedding cache |
| `AuthorityReranker` | `rerank.py` | Configured per sweep point |

The distinction is worth stating as a rule: **freeze the data, not the apparatus.** A `Query` that
could change mid-run is a bug waiting to happen; a retriever that builds an index at construction
time is doing its job.

### Derived state: `field(init=False)` and `__post_init__`

A `BM25Retriever` needs an index. The index is *derived* from the chunks, so it should not be a
constructor argument — asking the caller to build the index and pass it in would let them pass an
index of different chunks.

Python's dataclass machinery has exactly this case covered:

```python
@dataclass
class BM25Retriever:
    """Rung 1 -- lexical retrieval."""

    chunks: list[Chunk]
    name: str = "BM25"
    chunk_depth_multiplier: int = CHUNK_DEPTH_MULTIPLIER
    index: BM25 = field(init=False)

    def __post_init__(self) -> None:
        self.index = BM25([chunk.indexed_text for chunk in self.chunks])
```

`field(init=False)` declares `index` as a real field that the generated `__init__` will *not* accept.
`__post_init__` runs immediately after construction and fills it in. The result: it is impossible to
construct a `BM25Retriever` whose index does not match its chunks.

`DenseRetriever` uses the same mechanism to enforce a stronger invariant:

```python
    def __post_init__(self) -> None:
        if self.vectors.shape[0] != len(self.chunks):
            raise ValueError(
                f"{self.vectors.shape[0]} vectors for {len(self.chunks)} chunks"
            )
        self._matrix = normalise(self.vectors.astype(np.float32))
        self.embedder = self.embedder or OllamaEmbedder()
```

Consider what happens without that check. Vectors and chunks are separate objects; the vectors come
from a cache on disk. If the cache is stale — a different chunking configuration, say — you get 19,000
vectors for 19,763 chunks. Every similarity score is then attributed to the wrong chunk. Retrieval
returns nonsense that *looks like* legitimate output: real chunk IDs, plausible scores, no error.

Three lines convert an invisible catastrophe into a startup failure. Pinned:

```python
def test_vector_count_mismatch_raises(self) -> None:
    # Silently mismatched vectors would misattribute every score to the wrong chunk.
    with self.assertRaises(ValueError):
        DenseRetriever(chunks=self.chunks, vectors=np.zeros((2, 2), ...), ...)
```

## Mental Model

Think of a **laboratory notebook** versus a **whiteboard**.

The notebook is written in pen. Entries are dated and never altered — if you were wrong, you write a
new entry. That is `frozen=True`: the record of what you measured cannot be retroactively edited.

The whiteboard is for working. It gets erased and rewritten constantly. That is the retriever: it
holds caches and indexes and intermediate state, and none of that is a claim about the world.

## Deep Explanation: structural typing with Protocol

This is the most important abstraction in the codebase, because it is what lets four different
retrieval strategies share one evaluation loop.

### The problem

We have four rungs. The evaluation loop should not care which it is running:

```python
for retriever in (lexical, dense, hybrid, reranker):
    result = evaluate(retriever, queries, k=10)
```

The traditional way to express "these are interchangeable" is inheritance — an abstract base class
that all four subclass. That works, and it has costs:

- Every retriever must import and inherit the base class, coupling all four to a shared parent.
- A test stub must also inherit it, which means test doubles carry production machinery.
- Adding a retriever means touching the hierarchy.

### The alternative

`typing.Protocol` expresses the requirement *structurally*: anything with a matching shape qualifies,
with no declaration and no inheritance.

```python
class Retriever(Protocol):
    name: str

    def search(self, query: Query, top_k: int) -> list[int]:
        """Return ranked, distinct PEP numbers, best first."""
        ...
```

That is the entire contract. Any object with a `name` attribute and a matching `search` method *is* a
`Retriever`, as far as the type checker is concerned. Nothing inherits from it. It is documentation
that a type checker can verify.

This is sometimes called **duck typing with a type checker attached** — Python always allowed the
former; `Protocol` adds static verification to it.

### What it buys, concretely

Six types in this codebase satisfy `Retriever` without any of them mentioning it:

```
   BM25Retriever        rung 1, real
   DenseRetriever       rung 2, real
   HybridRetriever      rung 3, real
   AuthorityReranker    rung 4, real, and lives in a different module
   OracleRetriever      synthetic, returns the ground truth
   RandomRetriever      synthetic, returns random documents
```

The last two matter most. Chapter 13 shows that validating the harness requires running *synthetic*
retrievers through the *real* evaluation loop. With inheritance, those test doubles would inherit
production machinery. With `Protocol`, `OracleRetriever` is nine lines and completely independent:

```python
@dataclass
class OracleRetriever:
    """Returns exactly the ground truth. Must score 1.0 on every metric."""

    name: str = "Oracle"

    def search(self, query: Query, top_k: int) -> list[int]:
        return sorted(query.relevant)[:top_k]
```

And `AuthorityReranker` lives in `rerank.py`, imports nothing from the retriever hierarchy, and drops
into the same loop.

There is a pleasing detail here: structural subtyping in Python was specified by **PEP 544**, which is
itself a document in this project's corpus, and gold query `q08` asks about it. The codebase uses the
feature the corpus specifies.

### The tests exploit it

Because the contract is structural, test doubles can be trivially small. From `tests/test_rerank.py`:

```python
@dataclass
class StubHybrid:
    """Stands in for HybridRetriever with a fixed pool ordering."""

    pool: list[int]
    chunks: list[Chunk] = field(default_factory=lambda: CHUNKS)
    chunk_depth_multiplier: int = 1
    name: str = "StubHybrid"

    def search_chunks(self, query: Query, depth: int) -> list[int]:
        return self.pool[:depth]
```

Fourteen lines replace a real hybrid retriever, so the reranker's logic can be tested against a known
input ordering with no embedding model, no index, no network, and no corpus. That is why 103 tests run
in **0.1 seconds** — measured — despite the system depending on two neural models.

## Deep Explanation: small choices that add up

### `defaultdict` in accumulation loops

Scoring accumulates into a mapping whose keys are not known in advance:

```python
scores: dict[int, float] = defaultdict(float)
...
scores[doc_index] += idf * (...)
```

Without `defaultdict`, every `+=` needs a membership check or a `.get(key, 0.0)`. With it, a missing
key materialises as `0.0` on first access. The same pattern appears in RRF's `fused` dict and in
BM25's per-document frequency counting.

This is not merely brevity — an explicit `if doc_index not in scores` on the hot path is one more
place for an error, in code whose correctness is the entire product.

### `frozenset` for label sets

```python
@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    category: str
    relevant: frozenset[int]
```

`frozen=True` prevents rebinding `query.relevant`, but if `relevant` were a plain `set`, its
*contents* could still be mutated — `query.relevant.add(999)` would work, silently changing ground
truth mid-run. `frozenset` closes that hole. Freezing a container's binding is not the same as
freezing its contents, and for ground-truth labels you need both.

### Keyword defaults that make behaviour explicit

```python
def authority_weight(
    pep: Pep | None,
    query: Query | None = None,
    known: set[int] | None = None,
    version_penalty: bool = False,
) -> float:
```

`version_penalty=False` is not just a default — it encodes a finding. Chapter 20 explains that the
naive version rule is *harmful*, measured. The default is off, the flag exists so the claim could be
tested rather than asserted, and the signature documents which is the ordinary path.

### Type hints as executable documentation

Every function in the codebase is annotated, and `from __future__ import annotations` appears at the
top of every module, allowing modern syntax (`str | None`, `list[int]`) regardless of interpreter
version.

The hints are not decoration. `def search(self, query: Query, top_k: int) -> list[int]` tells you the
return is document numbers, not chunk indices — a distinction that is invisible at runtime (both are
`list[int]`) and is the source of real confusion in retrieval code. Chapter 17 shows that this project
has *both* kinds of method and the annotation is often the fastest way to tell them apart.

## Systems Perspective

`frozen=True` has a cost: attribute assignment goes through `__setattr__`, and construction is
marginally slower than a plain class. For 19,763 `Chunk` objects created once, this is irrelevant. For
objects created in an inner loop it would not be. The rule is to freeze records, not hot-path
temporaries — and `speceval`'s hot paths (BM25 scoring, the matmul) operate on primitives and NumPy
arrays, not dataclasses.

`Protocol` has **zero** runtime cost. It is erased entirely; `isinstance` checks against it require
`@runtime_checkable`, which this project never needs. The verification happens in a type checker, not
in the interpreter.

## Common Mistakes

**Mutable default arguments.** `def f(items: list = [])` shares one list across all calls. Dataclasses
prevent it by raising at class-definition time; use `field(default_factory=list)`.

**`frozen=True` with a mutable field.** A frozen dataclass holding a `set` or `list` is only shallowly
immutable. Use `frozenset` and `tuple` for data that must not change.

**Reaching for an ABC when a Protocol fits.** Inheritance couples your implementations and forces test
doubles to carry production machinery.

**Accepting derived state as a constructor argument.** If the index can be passed in, it can be the
wrong index. Derive it in `__post_init__`.

**No invariant check on paired data.** Vectors and chunks arriving from different places must be
checked for agreement; the failure is silent otherwise.

**Skipping type hints in measurement code.** In retrieval especially, `list[int]` can mean two
completely different things and the annotation is the only signal.

## Interview Insight

> **"When would you use a Protocol instead of an abstract base class?"**

When you want to express "anything shaped like this works" rather than "everything must descend from
this". Protocols suit families of interchangeable strategies, especially when some implementations are
test doubles or live in modules that should not depend on the abstraction.

The concrete example: *this project has four real retrievers plus two synthetic ones used to validate
the metrics. With a base class, the synthetic ones would inherit production machinery. With a
Protocol, the oracle retriever is nine lines and imports nothing, so the entire test suite runs in a
tenth of a second despite the system depending on two neural models.*

> **"How do you make an experiment reproducible?"**

Four things, and this project does all four:

1. **Freeze the data.** Immutable records mean the second half of a run measured the same thing as the
   first half.
2. **Determinise every tiebreak.** Sorting by `(-score, index)` rather than by score alone; seeding
   the random retriever per query rather than once.
3. **Fix the sampling.** Temperature 0 and a fixed seed for generation — verified byte-identical
   across three runs.
4. **Pin the environment.** A project-local virtualenv plus a lockfile recording the exact versions
   that produced the reported numbers.

> **"Why `frozenset` rather than `set`?"**

Because `frozen=True` on a dataclass prevents rebinding the attribute, not mutating the object it
points to. `query.relevant.add(999)` would silently corrupt ground truth mid-evaluation. Freezing the
binding and freezing the contents are different problems and ground truth needs both.

## Debugging Tip

When an evaluation produces a suspicious number, check *identity* before checking logic. Print the
`repr` of the records involved — dataclasses give you a full field dump for free:

```
Query(qid='q16', text='how do I postpone the evaluation of annotations',
      category='availability', relevant=frozenset({649}), asked_version=None,
      trap=True, note="PEP 563 'Postponed Evaluation of Annotations' matches ...")
```

That single line answers "is this the query I think it is, with the labels I think it has?", which is
the actual cause of a surprising number more often than the metric being wrong.

## Summary

- Measurement code fails by producing plausible wrong numbers, so it is written to make silent error
  impossible rather than to avoid crashes.
- Freeze the data (`Pep`, `Chunk`, `Query`), not the apparatus (retrievers, rerankers).
- `frozenset` where a plain `set` would leave contents mutable inside a frozen record.
- Derive state in `__post_init__` with `field(init=False)`, so a mismatched index cannot be passed in.
- Check invariants between separately-sourced data; the vector/chunk count check turns an invisible
  misattribution into a startup error.
- `typing.Protocol` gives structural typing: six retrievers satisfy one contract with no inheritance,
  which keeps test doubles tiny and the suite fast.
- `defaultdict` keeps accumulation loops free of membership checks.
- Type hints disambiguate `list[int]` values that mean different things.

## Key Takeaways

1. Freeze the records; leave the machinery mutable.
2. Prefer `Protocol` over inheritance for interchangeable strategies — it is what makes cheap test
   doubles possible.
3. Derive state in the constructor and validate paired inputs, or accept silent misattribution.
4. Reproducibility is four separate disciplines: immutable data, deterministic tiebreaks, fixed
   sampling, pinned environment.

## Why the Next Chapter Exists

Chapter 4 said the query is embedded by a model, and chapter 5 assumed answers get generated, both
without saying how either actually happens.

Chapter 8 covers talking to a local model server over HTTP: batching, retries, the disk cache and its
fingerprint, and the single configuration choice that makes generated text reproducible — the property
without which none of Part V's answer-level numbers could be trusted.
