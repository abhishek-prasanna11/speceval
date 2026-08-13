# Chapter 2 — From Text to Something Searchable

## Learning Objectives

- Explain why raw text cannot be scored and what has to happen first.
- Define **tokenisation**, **vocabulary**, and **term frequency**, and describe the choices inside
  each.
- Explain the **vocabulary mismatch problem** — the fundamental limitation of all word-counting
  retrieval.
- Explain what an **inverted index** is, why it exists, and what it costs.
- Explain **chunking** and the tradeoff that governs chunk size.
- Read `speceval`'s tokeniser and chunker and say why each decision was made.

## Motivation

Chapter 1 concluded that we need `score(document, query) -> float`. We cannot write that function
yet. A document is a sequence of characters, and characters do not support the operations we need:
there is no meaningful way to multiply, average, or compare two strings such that the result tells
you about topical overlap.

So the first real engineering step in any retrieval system is a translation: **turn text into a
mathematical object.** Every decision in that translation silently determines what the finished
retriever can and cannot find. Get the translation wrong and no amount of clever scoring recovers
it.

## First Principles

### Step 1: tokenisation

A **token** is the atomic unit of text your system will reason about. **Tokenisation** is cutting a
string into tokens.

The naive version is to split on whitespace:

```python
"Python 3.10 added match/case".split()
# ['Python', '3.10', 'added', 'match/case']
```

Already broken. A query containing `match` will not match the token `match/case`, because string
equality is exact. So we normalise: lowercase everything (so `Python` and `python` unify), and
split on anything that is not a letter or digit.

That gives the standard approach, which is a regular expression describing what a token *is*
rather than what separates them:

```python
re.findall(r"[a-z0-9]+", text.lower())
```

### The choice that matters: what counts as part of a word?

Consider this line of Python:

```python
from __future__ import annotations
```

Under `[a-z0-9]+`, the token `__future__` becomes `['future']`. The underscores are gone. That
seems harmless until you realise `future` and `__future__` are completely different things — one
is an English word, the other is a specific Python module whose name is the *entire content* of
some queries.

The same problem hits `__init__`, `except*`, `:=`, and every dunder name in the language.

`speceval` therefore includes the underscore in what counts as a word. From
`speceval/bm25.py`:

```python
# Identifiers matter in this corpus -- `__future__`, `__init__`, `match` -- so the
# tokenizer keeps underscores rather than splitting on them.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
```

One character of difference in a regex — `_` inside the character class — and `__future__`
survives as a single token. This is pinned by a test, because it is the kind of thing a later
"cleanup" would silently break:

```python
def test_tokenizer_keeps_dunder_identifiers(self) -> None:
    # The reason for a custom tokenizer: `__future__` must survive intact.
    self.assertIn("__future__", tokenize("from __future__ import annotations"))
    self.assertEqual(tokenize("Match/Case!"), ["match", "case"])
```

The general lesson: **tokenisation is domain-specific.** A tokeniser tuned for news articles will
quietly destroy a corpus of code. There is no neutral choice.

### Step 2: the vocabulary and the term-document matrix

Collect every distinct token across the whole corpus. That set is the **vocabulary**, `V`.

Now each document can be described as a vector of counts — for each term in the vocabulary, how
many times does it appear in this document? That count is the **term frequency**, written
`tf(t, d)`.

```
              "walrus"  "operator"  "match"  "python"  ...
  chunk 1  [     3,         2,         0,        5,    ... ]
  chunk 2  [     0,         1,         7,        4,    ... ]
  chunk 3  [     0,         0,         0,        2,    ... ]
```

This is the **term-document matrix**, and it is the first mathematical object we have. It supports
arithmetic. We can finally write a scoring function.

It is also the point at which the model's central limitation becomes visible.

### The bag-of-words assumption

Representing a document as counts throws away word order. `"the cat sat on the mat"` and
`"the mat sat on the cat"` are *identical* under this representation.

This is called the **bag-of-words** assumption, and it sounds fatal but mostly is not: for judging
what a document is *about*, which words appear and how often turns out to carry most of the
signal. Word order matters enormously for meaning and rather little for topic.

But it does have a real cost, and chapter 4 exists because of it.

### The vocabulary mismatch problem

Here is the limitation you cannot engineer around while counting words.

Two documents in this corpus:

- **PEP 563**, titled *Postponed Evaluation of Annotations*
- **PEP 649**, titled *Deferred Evaluation Of Annotations Using Descriptors*

They describe the same feature area. PEP 563 has been superseded; PEP 649 is the current answer.
Now the query: *"how do I postpone the evaluation of annotations?"*

Count the shared tokens:

```
query:     how do i postpone the evaluation of annotations
PEP 563:   postponed evaluation of annotations        -> shares: evaluation, of, annotations
PEP 649:   deferred evaluation of annotations ...     -> shares: evaluation, of, annotations
```

The word `postpone` appears in the query. The token in PEP 563's title is `postponed` — a
different string, therefore a different token, therefore *no match at all* under exact equality.

So counting words cannot distinguish these two documents on the basis of the query's most
distinctive word, because that word matches neither of them exactly. And a human reading the query
would say *"postpone" obviously means "postponed"*.

This gap — the same idea expressed in different words — is the **vocabulary mismatch problem**. It
is the fundamental ceiling on all lexical retrieval. Techniques exist to soften it (stemming, which
would cut both `postpone` and `postponed` to `postpon`; synonym expansion), but they are patches on
a representation that has no concept of meaning.

Chapter 4 introduces a representation that does. Keep this example: it is query `q16` in the gold
set, and chapter 11 will show that the fix for vocabulary mismatch introduces a *new* failure of
its own on exactly this query.

### Step 3: the inverted index

Now a performance problem. Scoring a query against a corpus by walking the term-document matrix
means, for each of 19,763 chunks, looking up each query term. Most of those lookups return zero —
the matrix is overwhelmingly empty, because any given chunk contains a few hundred distinct words
out of a vocabulary of tens of thousands.

The fix is to store the matrix transposed and sparse. For each *term*, keep a list of the
documents that contain it, with counts:

```
FORWARD (what we had)              INVERTED (what we want)
chunk 1 -> {walrus:3, operator:2}  walrus   -> [(chunk 1, 3), (chunk 8, 1)]
chunk 2 -> {match:7, python:4}     operator -> [(chunk 1, 2), (chunk 5, 4)]
chunk 3 -> {python:2}              match    -> [(chunk 2, 7)]
                                   python   -> [(chunk 1, 5), (chunk 2, 4), (chunk 3, 2)]
```

Each list is called a **postings list**. To score a query you now touch only the documents that
contain at least one query term, which for a specific query is a tiny fraction of the corpus.

This is the data structure behind every lexical search engine ever built — Lucene, Elasticsearch,
PostgreSQL full-text search, and `speceval`'s forty-line BM25. Chapter 15 walks the implementation.

The cost is memory and build time: you pay once, up front, to make every subsequent query cheap.
In this project the index over 19,763 chunks builds in a couple of seconds.

## Mental Model

**Tokenisation** is deciding what counts as a word in your world. A chemist's index needs
`H2SO4` intact; a Python corpus needs `__future__` intact. There is no universal answer.

**The inverted index** is the index at the back of a textbook. You do not read all 700 pages
looking for "photosynthesis" — you look up the word and get a short list of page numbers. The
index took the publisher effort to build and takes up pages, and it makes every future lookup
nearly free.

## Deep Explanation: chunking

One more translation step, and it is the one with the most consequences for this project.

### Why documents get cut up

A PEP can be very long. `speceval` measures the real distribution: **734 PEPs become 19,763
chunks, a mean of 26.9 chunks per PEP.** So the average PEP is roughly 27 chunk-sized pieces of
text.

Three reasons not to retrieve whole documents:

**Dilution.** A 10,000-word document that mentions your topic in one paragraph looks, to a
word-counting scorer, like a document that is mostly about other things. The signal drowns.

**Generator budget.** The language model has a finite input size, and cost grows with input length.
You cannot paste three entire PEPs into a prompt.

**Precision of attribution.** "The answer is in PEP 634" is less useful than "the answer is in this
paragraph of PEP 634."

### The chunk size tradeoff

```
  small chunks                                     large chunks
  <---------------------------------------------------------->
  precise location                          full context preserved
  sharp lexical signal                      fewer boundary accidents
  context gets severed                      topic signal diluted
  more chunks to search                     fewer, fatter candidates
```

There is no correct answer, only a defensible one for your corpus.

### How `speceval` chunks

The corpus is reStructuredText, which marks section headings by underlining them with punctuation:

```
Abstract
========

This PEP provides the technical specification for the match statement.
```

That structure is a gift: section boundaries are *semantic* boundaries chosen by the document's
author. So `speceval` splits on them rather than on an arbitrary character count. From
`speceval/chunking.py`:

```python
# reStructuredText marks a section title by underlining it with a run of punctuation.
_ADORNMENT_CHARS = set("=-`:'\"~^_*+#<>")

# Longer chunks retrieve more context but blur the lexical signal; 1200 characters is
# roughly a screenful of prose and keeps most PEP sections whole.
MAX_CHUNK_CHARS = 1200
```

Detection requires care. A line is a section title only if it is non-blank, is not *itself* an
adornment run, and is followed by an adornment run at least as long as the title:

```python
is_title = (
    title
    and not _is_adornment(line)
    and _is_adornment(next_line)
    and len(next_line.strip()) >= len(title)
)
```

Each of those four conditions rules out a specific false positive. Dropping the "at least as long"
check, for instance, would let a line followed by a short `---` divider be mistaken for a heading.

Sections longer than 1200 characters are then packed into windows on **paragraph** boundaries, never
mid-sentence:

```python
def _split_long(text: str, limit: int) -> list[str]:
    """Pack paragraphs into windows of at most ``limit`` characters."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    ...
    for paragraph in paragraphs:
        # A single oversized paragraph becomes its own chunk rather than being cut
        # mid-sentence; a handful of PEPs contain very long grammar blocks.
        if size and size + len(paragraph) > limit:
            windows.append("\n\n".join(current))
            current, size = [], 0
```

### The decision that is easy to miss: what text gets indexed

A chunk from the middle of PEP 557 might never contain the phrase "data classes" — the title said
that once, at the top, and the body says "the decorator" thereafter. A query for *"data classes"*
would then score that chunk near zero.

So the indexed text is not the chunk body. It is the body **with the PEP title and section heading
prepended**:

```python
@property
def indexed_text(self) -> str:
    """What the retrievers actually see.

    The PEP title and section heading are prepended so a query matching a title
    can find the body under it -- without this, "Data Classes" would score poorly
    against a chunk that never repeats the phrase.
    """
    return f"{self.pep_title}\n{self.section}\n{self.text}"
```

This is a small line with large effects, and chapter 23 records one of them: it is a leading
candidate explanation for why a prediction this project made about dense retrieval turned out to be
wrong.

### Chunking is held constant, deliberately

One methodological point that matters more than any parameter choice. From the module docstring:

```python
"""Split PEP bodies into retrievable chunks along reStructuredText section boundaries.

Chunking quality is deliberately *not* a variable in this study -- all four retrieval
strategies see identical chunks, so any difference between them is attributable to the
strategy. The goal here is only to be reasonable and stable, not optimal.
"""
```

The chunker is not tuned, because tuning it would make it a confounding variable. All four rungs
see byte-identical chunks. Any difference between them therefore comes from the ranking, which is
the thing under study. Chapter 12 generalises this principle.

## Systems Perspective

The inverted index is a hash map from string to list. Two costs are worth knowing.

**Build cost** is one pass over the corpus, dominated by tokenisation — millions of regex matches
and dictionary increments. Seconds for this corpus.

**Query cost** is proportional to the total length of the postings lists for the query's terms. And
this is why common words are expensive: `python` appears in nearly every chunk, so its postings
list is nearly the whole corpus. A query containing it touches everything.

That observation is not just about speed. It is also the intuition behind the most important idea
in chapter 3: a term that appears in almost every document is doing almost no work in
distinguishing them, so it should be weighted down. The performance problem and the relevance
problem have the same root.

## Common Mistakes

**Using a default tokeniser on a specialised corpus.** The single highest-value line of code in
`speceval`'s lexical path is one underscore in a character class.

**Chunking on a fixed character count with no regard for structure.** Cutting mid-sentence
produces chunks that are hard to score and unreadable when shown to a user or a model.

**Indexing the chunk body only.** Titles carry the strongest topical signal in the document and
they appear exactly once.

**Tuning the chunker while comparing retrievers.** Then you no longer know which change caused the
difference. Fix it, document it, move on.

## Interview Insight

> **"How would you chunk documents for a RAG system?"**

The expected answer is a number, and a number alone is a weak answer. A strong answer names the
tradeoff (precision of location versus preservation of context), then says: prefer the document's
own structural boundaries over arbitrary character counts, because headings are semantic
boundaries an author already chose for you; fall back to paragraph packing with a cap for sections
that are too long; and prepend document and section titles to the indexed text so title-matching
queries can find body content.

Then add the part that distinguishes an engineer from a tutorial-follower: **if you are comparing
retrieval strategies, freeze the chunker.** Otherwise chunk size is a confounding variable and your
comparison measures nothing.

> **"What is the limitation of BM25 or any keyword search?"**

Vocabulary mismatch — "postpone" does not match "postponed", let alone "deferred". Word counting
has no model of meaning, only of string identity. This is the entire motivation for dense
retrieval. A good answer will note that the fix is not free: chapter 11 of this book documents a
failure mode that dense retrieval *introduces*.

## Debugging Tip

When a lexical retriever inexplicably misses an obvious document, **print the tokens** for both the
query and the document. Almost every such bug is a tokenisation mismatch — a hyphen, a case
difference, an underscore, a plural — and it is invisible until you look at the token lists side by
side.

## Summary

- Text must become a mathematical object before it can be scored. Tokenisation, then counting, then
  inversion.
- Tokenisation is domain-specific and consequential. `[a-z0-9_]+` rather than `[a-z0-9]+` preserves
  Python identifiers.
- The term-document matrix supports arithmetic but discards word order (bag-of-words), which costs
  little for topic and much for meaning.
- Vocabulary mismatch is the hard ceiling on word counting: identical ideas in different words score
  zero against each other.
- The inverted index makes queries cheap by touching only documents containing query terms.
- Chunking trades location precision against context. `speceval` splits on rST section headings, caps
  at 1200 characters, packs on paragraph boundaries, and prepends titles to the indexed text.
- The chunker is frozen across all four rungs so that differences are attributable to ranking.

## Key Takeaways

1. Your tokeniser defines what your retriever is able to find. Choose it for your corpus.
2. Word counting cannot bridge vocabulary mismatch. That limitation motivates all of chapter 4.
3. Prepend titles to indexed text; a title appears once but describes everything under it.
4. Freeze everything you are not currently measuring.

## Why the Next Chapter Exists

We now have an inverted index of term frequencies, which is enough to compute *something*. But
raw counts are a bad score, for reasons that are easy to demonstrate: they reward long documents
for being long and treat the word `python` as informative as the word `walrus` in a corpus about
Python.

Chapter 3 fixes both problems from first principles and arrives at BM25 — the formula `speceval`
implements in forty lines — showing where each of its two magic constants comes from and what it
controls.
