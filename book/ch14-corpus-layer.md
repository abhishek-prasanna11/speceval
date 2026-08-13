# Chapter 14 — The Corpus Layer

`speceval/corpus.py` (123 lines) · `speceval/chunking.py` (116 lines)

## Learning Objectives

- Trace the path from 734 files on disk to 19,763 in-memory chunks.
- Explain every field of `Pep` and `Chunk` and why it exists.
- Explain the three exits from the header-parsing loop.
- Explain the walrus operator in `load_corpus` and what it guards against.
- Explain why `load_corpus` raises on an empty result.
- Explain the section-detection predicate's four conditions, each ruling out a specific false positive.

## Motivation

This is the bottom of the system. Everything above it — BM25, embeddings, fusion, reranking, metrics —
consumes what this layer produces. A parsing mistake here does not cause a crash; it causes a `Status`
field to be empty, or a title to be truncated, and every number computed afterwards is subtly wrong with
no indication that anything happened.

It is also the layer with the least glamour and the most edge cases, which is the usual combination for
code that quietly determines whether a project works.

## The data model

Two frozen records (chapter 7 explained why frozen).

### `Pep`

```python
@dataclass(frozen=True)
class Pep:
    number: int
    title: str
    status: str
    pep_type: str
    python_version: str | None
    superseded_by: int | None
    replaces: tuple[int, ...]
    body: str
    path: Path

    @property
    def is_authoritative(self) -> bool:
        return self.status not in NON_AUTHORITATIVE
```

Field by field:

| Field | Type | Why it is here |
|---|---|---|
| `number` | `int` | Identity. Used as the label unit throughout |
| `title` | `str` | Prepended to indexed text (chapter 2); also the *cause* of the central failure mode |
| `status` | `str` | The authority signal. Kept as a string, not an enum — see below |
| `pep_type` | `str` | Standards Track / Informational / Process. Parsed but unused by the study |
| `python_version` | `str \| None` | The version dimension. `None` when absent |
| `superseded_by` | `int \| None` | The graph edge that `status` alone cannot express |
| `replaces` | `tuple[int, ...]` | The reverse edge. A tuple because frozen records need immutable containers |
| `body` | `str` | Everything after the header block |
| `path` | `Path` | Provenance — which file this came from, for debugging |

Two decisions worth defending.

**`status` is a `str`, not an `Enum`.** An enum would fail on an unrecognised value, and this corpus
contains `April Fool!` — a real status on a real PEP. More importantly, the corpus is external and
mutable: the PEP editors could add a status tomorrow. A string parses whatever is there; the *set* of
values that mean "not in force" is captured separately:

```python
# A PEP in one of these states must not be answered from as though it were current.
# This is the definition the superseded-citation-rate metric is built on, so it lives
# in one place rather than being re-spelled at each call site.
NON_AUTHORITATIVE = frozenset({"Rejected", "Withdrawn", "Superseded", "Deferred"})
```

An unknown status is then simply *not* in that set — it is treated as authoritative-by-default rather
than crashing the run. Chapter 20 assigns it a middling weight instead.

**`replaces` is a tuple, `superseded_by` is a scalar.** A PEP can replace several predecessors (PEP 600
replaces three manylinux tags), but is superseded by at most one. The types encode the corpus's actual
cardinality.

The `is_authoritative` property exists so that the negation is written once. Scattered
`pep.status not in NON_AUTHORITATIVE` checks would be four opportunities to write `in` by mistake.

### `Chunk`

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    pep_number: int
    pep_title: str
    section: str
    text: str

    @property
    def indexed_text(self) -> str:
        return f"{self.pep_title}\n{self.section}\n{self.text}"
```

`pep_number` and `pep_title` are **denormalised** — copied onto every chunk rather than looked up
through a reference. That is 19,763 duplicated title strings, which is trivial memory, and it buys
something valuable: a `Chunk` is self-describing. Any part of the system holding a chunk can attribute
it, format it for a prompt, or score it without needing access to the corpus. Chapter 19's prompt
builder relies on exactly this.

`chunk_id` is `f"pep-{pep.number:04d}#{index}"` — zero-padded so `pep-0008#3` sorts sensibly next to
`pep-0634#3`.

## Deep Explanation: header parsing

The heart of the module, and the place with the most edge cases (chapter 10 covered the format; this is
the implementation).

```python
def _parse_headers(lines: list[str]) -> tuple[dict[str, str], int]:
    """Return (headers, index of first body line).

    RFC-822 style: the block ends at the first blank line. A line starting with
    whitespace continues the previous header rather than beginning a new one.
    """
    headers: dict[str, str] = {}
    last_key: str | None = None
    body_start = 0

    for i, raw in enumerate(lines):
        if not raw.strip():
            body_start = i + 1
            break
        if raw[0] in " \t" and last_key is not None:
            headers[last_key] = f"{headers[last_key]} {raw.strip()}".strip()
            continue
        match = _HEADER_RE.match(raw)
        if match is None:
            # Not a header and not a continuation -- treat the block as finished.
            body_start = i
            break
        last_key = match.group(1)
        headers[last_key] = match.group(2).strip()
    else:
        body_start = len(lines)

    return headers, body_start
```

### The three exits

**Exit 1 — blank line.** The RFC-822 rule. `body_start = i + 1` skips the blank itself.

**Exit 2 — unparseable line.** Defensive. If a file has malformed headers, the block is treated as
finished and the rest becomes body. The alternative — raising — would mean one damaged file among 734
aborts the entire corpus load. Degrading gracefully is right here because the failure is *visible*: a
PEP with an empty `Status` shows up immediately in the status histogram that `run_phase1.py` prints
every run.

**Exit 3 — the `for...else`.** This is the least familiar Python construct in the codebase. The `else`
on a `for` loop runs **only if the loop completed without `break`**. So it handles a file consisting
entirely of headers with no body and no trailing blank line: `body_start = len(lines)`, giving an empty
body rather than an index error.

Without it, `body_start` would keep its initial value of `0` and the entire header block would be
re-emitted as body — meaning every header field would also be indexed as searchable text. Every chunk
would then contain `Status: Superseded`, which would leak the authority signal into the retrieval text
and confound the entire study. A one-line construct preventing a subtle, study-invalidating bug.

### The continuation branch

```python
        if raw[0] in " \t" and last_key is not None:
            headers[last_key] = f"{headers[last_key]} {raw.strip()}".strip()
            continue
```

Note `raw[0]` rather than `raw.startswith(" ")` — this is reached only after the blank-line check, so
`raw` is guaranteed non-empty and indexing is safe.

The `and last_key is not None` guard covers a file whose *first* line is indented. Without it, that
would be a `KeyError` on `headers[None]`.

### The two integer extractors

```python
def _first_int(value: str | None) -> int | None:
    if not value:
        return None
    match = _INT_RE.search(value)
    return int(match.group()) if match else None


def _all_ints(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(n) for n in _INT_RE.findall(value))
```

They differ because the fields differ. `Superseded-By: 634` is one number; `Replaces: 513, 571, 599`
may be several. Using `search` for the first and `findall` for the second matches the corpus's real
cardinality, and both return an empty-ish value rather than raising on a missing or malformed field.

## Deep Explanation: loading the corpus

```python
def load_corpus(root: Path | str = "peps/peps") -> list[Pep]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"PEP corpus not found at {root!r}. Run scripts/fetch_corpus.sh first."
        )

    peps = [pep for path in sorted(root.glob("pep-*.rst")) if (pep := load_pep(path))]
    if not peps:
        raise RuntimeError(f"No PEPs parsed from {root!r} -- has the layout changed?")
    return sorted(peps, key=lambda p: p.number)
```

Four deliberate properties.

**A missing corpus names the fix.** `Run scripts/fetch_corpus.sh first.` The corpus is gitignored, so a
fresh clone has no `peps/` directory, and this is the first error a new user will hit. An error message
that names the command is worth more than one that merely states the fact.

**The walrus operator does double duty.** `if (pep := load_pep(path))` assigns and tests in one
expression. `load_pep` returns `None` for a file with no usable PEP number — the corpus directory
contains `contents.rst`, `conf.py` and other non-PEP files matching neither the glob nor a valid header.
Those are skipped silently.

(There is a pleasing self-reference here: the walrus operator is PEP 572, which is in the corpus being
loaded, and is gold query `q02`.)

**Empty results raise rather than returning `[]`.** This is the important one. If the upstream
repository reorganises its layout — say `peps/peps/` becomes `peps/` — the glob matches nothing and
returns an empty list. Downstream, that produces zero chunks, an empty index, and a full evaluation run
reporting **Recall@10 = 0.000 for every system**.

That output looks like a catastrophic retrieval failure. You would go looking in the retriever. The
actual problem is a path. Raising converts a confusing wrong answer into an obvious one, and the message
points at the real cause: *has the layout changed?*

**Sorted output, twice.** `sorted(root.glob(...))` for deterministic file order (filesystem iteration
order is not guaranteed), then `sorted(peps, key=...)` for numerical order — `pep-0008` before
`pep-0634`, not lexicographic. Determinism again: two runs must produce identically ordered corpora, or
chunk indices shift between runs and cached embeddings no longer correspond.

## Deep Explanation: chunking

Chapter 2 covered the rationale. Here is the mechanism.

### Detecting a section title

```python
def _is_adornment(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 3 and len(set(stripped)) == 1 and stripped[0] in _ADORNMENT_CHARS
```

Three conditions: at least three characters, exactly one distinct character (`len(set(...)) == 1`), and
that character is a valid rST adornment. `===`, `---`, `~~~` qualify; `=-=-=` does not.

```python
        is_title = (
            title
            and not _is_adornment(line)
            and _is_adornment(next_line)
            and len(next_line.strip()) >= len(title)
        )
```

Each condition rules out a specific false positive:

| Condition | Rejects |
|---|---|
| `title` (non-empty) | A blank line above an adornment |
| `not _is_adornment(line)` | The *second* line of an over-and-under-lined title, which would otherwise be read as a title itself |
| `_is_adornment(next_line)` | Ordinary prose |
| `len(next_line) >= len(title)` | A short `---` divider following a paragraph |

That last one is the subtle one. reStructuredText requires the adornment to be at least as long as the
title; using that rule distinguishes a real heading from a horizontal rule that happens to follow text.

### Packing long sections

```python
def _split_long(text: str, limit: int) -> list[str]:
    """Pack paragraphs into windows of at most ``limit`` characters."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    windows: list[str] = []
    current: list[str] = []
    size = 0

    for paragraph in paragraphs:
        # A single oversized paragraph becomes its own chunk rather than being cut
        # mid-sentence; a handful of PEPs contain very long grammar blocks.
        if size and size + len(paragraph) > limit:
            windows.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph)

    if current:
        windows.append("\n\n".join(current))
    return windows
```

The `if size and ...` is the interesting condition. When `size == 0` — meaning the current window is
empty — the paragraph is added *regardless* of how large it is. So a 4,000-character grammar block
becomes one oversized chunk rather than being cut mid-sentence or, worse, dropped.

Overshooting the limit is a deliberate choice: a chunk slightly too large is a minor inefficiency, while
a chunk cut mid-sentence is unusable both for retrieval and for a prompt.

### Verified behaviour

Measured, live: **734 PEPs → 19,763 chunks**, mean 26.9 per PEP. Tests pin the structural properties:

```python
def test_sections_detected_by_adornment(self) -> None:
    titles = [title for title, _ in split_sections(self.pep.body) if title]
    self.assertEqual(titles, ["Abstract", "Syntax and Semantics"])

def test_adornment_lines_are_not_content(self) -> None:
    for _, text in split_sections(self.pep.body):
        self.assertNotIn("====", text)

def test_chunks_carry_provenance(self) -> None:
    chunks = chunk_pep(self.pep)
    for chunk in chunks:
        self.assertEqual(chunk.pep_number, 634)
        self.assertTrue(chunk.chunk_id.startswith("pep-0634#"))
```

The second is easy to overlook and matters: adornment runs must not survive into chunk text. If they
did, `====` would be tokenised away by BM25 (harmlessly) but would appear in prompts sent to the
language model (not harmlessly) and in the embedded text (adding noise to every vector).

## Systems Perspective

The whole layer is one pass over 734 files: read, regex the header block, split sections, pack
paragraphs. Sub-second, and it runs on every invocation of every driver rather than being cached.

That is a deliberate non-optimisation. Caching parsed corpora is a classic source of staleness bugs —
you change the chunker, forget to invalidate, and spend an afternoon confused. The parse is cheap enough
that re-doing it every run removes an entire category of error. What *is* cached is the expensive
downstream artefact (embeddings), and its key is derived from the chunk text itself (chapter 4), so a
chunker change invalidates it automatically.

Memory: 19,763 `Chunk` objects with denormalised titles, comfortably tens of megabytes. The embedding
matrix at 58 MB dominates.

## Common Mistakes

**Parsing headers with `line.split(":")`.** Truncates titles containing colons.

**Ignoring `for...else`.** Files that are entirely headers silently re-emit the header block as body,
leaking the authority signal into the searchable text.

**Returning `[]` when the corpus is missing.** Produces an evaluation reporting 0.000 everywhere and
sends you debugging the retriever.

**Unsorted globs.** Filesystem order is not guaranteed; chunk indices then differ between runs and
cached embeddings silently misalign.

**Enum for an externally-controlled field.** The corpus contains `April Fool!` and can gain new statuses
without warning.

**Normalising `pep_title` out of `Chunk`.** Denormalisation costs a few megabytes and makes every chunk
self-describing.

## Interview Insight

> **"Walk me through how you ingest your corpus."**

Clone the repository shallowly — history is not part of the study — glob 734 rST files in sorted order,
parse the RFC-822 header block with a regex that anchors the field name and captures everything after
the first colon so titles containing colons survive, handle indented continuation lines and empty
values, then split each body on reStructuredText section adornments and pack long sections into
1200-character windows on paragraph boundaries. That gives 19,763 chunks at a mean of 26.9 per document.

The part worth adding: **the loader raises if it parses zero PEPs.** If the upstream layout changes, the
glob matches nothing, and an empty corpus produces a full evaluation reporting Recall of 0.000 for every
system — which looks like total retrieval failure and sends you debugging the wrong component. Raising
turns a confusing wrong answer into an obvious one.

> **"Why is `status` a string rather than an enum?"**

Because the field is controlled by an external repository. The corpus contains a PEP whose status is
literally `April Fool!`, and the editors could add a status tomorrow. An enum would crash on it. The
*set* of values meaning "not in force" is defined once as a frozenset, and an unknown status is simply
not in it — treated as authoritative by default rather than aborting a run.

## Debugging Tip

Any change to this layer should be followed by comparing the status histogram to the previous run:

```
  Final         374
  Rejected      131  (trap)
  Withdrawn      70  (trap)
  Draft          48
  Active         38
  Deferred       36  (trap)
  Superseded     25  (trap)
  Accepted       11
  April Fool!     1
```

A parsing regression shows up here as a shifted distribution — a drop in `Final`, a surge of empty
statuses — immediately and unmistakably, long before it would surface as a strange retrieval result.
`run_phase1.py` prints it on every run for exactly this reason.

## Summary

- `corpus.py` turns 734 rST files into frozen `Pep` records; `chunking.py` turns those into 19,763
  `Chunk` records.
- `status` is a string because the field is externally controlled; the non-authoritative set is defined
  once, in one place.
- `replaces` is a tuple and `superseded_by` a scalar, matching the corpus's real cardinality.
- Header parsing has three exits — blank line, unparseable line, and `for...else` for all-header files —
  and the last prevents the header block leaking into searchable text.
- `load_corpus` raises on an empty result, converting a silent 0.000-everywhere evaluation into an
  obvious error, and sorts twice for determinism.
- Section detection has four conditions, each ruling out a specific false positive; oversized paragraphs
  overshoot the limit rather than being cut.
- `Chunk` denormalises title and PEP number so it is self-describing.

## Key Takeaways

1. Fail loudly when the input is missing or empty; a silent empty corpus looks like a broken retriever.
2. Use strings for externally-controlled vocabularies, and define your interpretation of them once.
3. Sort everything that feeds an index — cached artefacts depend on stable ordering.
4. `for...else` is obscure but occasionally exactly right.

## Why the Next Chapter Exists

We have chunks. Chapter 15 reads the first thing that ranks them — the hand-rolled BM25 implementation
— and examines the four decisions inside it that chapter 3's derivation did not require: the tokeniser,
the loop order, the deterministic tiebreak, and the choice to own rather than import.
