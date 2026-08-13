# Retrieval, Authority, and How to Know You Are Right

### A first-principles book built from the `speceval` codebase

---

## Preface

This book has two jobs.

The first is to teach **information retrieval** — how a computer finds documents that answer a
question — starting from nothing. Not "here is the library, call this function," but why the
problem is hard, what people tried, why it works, and how you would build it yourself. If you
can write a for-loop and know what a dictionary is, you have enough to start.

The second is to document **one real project** with total precision: `speceval`, a study of what
happens when a retrieval system is asked questions about documents that supersede one another.
Every claim in this book was checked against the source code as it exists today, and every
number was produced by running the code, not by copying a figure out of a README.

### Why those two jobs belong in one book

You cannot understand `speceval` without understanding retrieval, and you will not really
understand retrieval from a tutorial that only ever shows you the happy path. `speceval` exists
because of a failure mode that only becomes visible when you measure carefully: a retrieval
system will confidently hand you a document that is perfectly on-topic, well written, highly
ranked — and officially dead. Superseded. Replaced years ago by a document that says something
different.

That is not a bug in a library. It is a gap between what retrieval optimises for (*relevance*)
and what a user needs (*authority*). Teaching you to see that gap requires teaching you what
relevance actually is, mathematically, and that is most of a retrieval course.

### What this project is, in one paragraph

`speceval` takes the Python Enhancement Proposal (PEP) series — 734 documents describing changes
to the Python language — and builds four progressively more sophisticated retrieval systems over
it. It then measures, for each one, how often the answers it produces cite a PEP that has been
rejected, withdrawn, or superseded. The headline result is that a reranking step which reads the
PEPs' own status metadata reduces bad citations by roughly four-fifths **and improves retrieval
quality at the same time** — contradicting the hypothesis the project was built to test. Chapter
23 works out why, and where that result stops being true.

### What you will be able to do at the end

- Explain, from first principles, how BM25 and dense embedding retrieval work, and implement
  both from scratch.
- Explain what Recall@K and nDCG@K measure, what they miss, and why an evaluation harness must
  be tested before its numbers are believed.
- Read every line of `speceval` and say why it is written that way.
- Defend the project's design decisions, its measured results, and — importantly — its
  limitations, in a technical interview.

### How to read this book

Parts I to III are the domain. Part IV is the code. Part V is the synthesis and the interview
preparation.

If you are the author of this project returning to it after some months, you can start at
Part IV and use Parts I–III as reference. If you are new to retrieval, read in order. A concept
is explained exactly once, in the chapter where it is first needed, and referenced thereafter.

### A note on honesty

This book reports two predictions the project got wrong, one bug found by its own test suite, one
bug in a numerical library, one stale claim discovered in the project's own documentation during
the writing of this book, and a hand-tuned constant table that was never validated. Those are not
apologies — they are the most useful parts. A study that only records its successes has not
measured anything.

### Conventions

Canonical names used throughout, never varied:

| Term | Meaning |
|---|---|
| **corpus** | The 734 PEP documents |
| **chunk** | A retrievable fragment of one PEP (19,763 in total) |
| **rung** | One of the four retrieval strategies under comparison |
| **the ladder** | The four rungs taken as a progression |
| **gold set** | The 51 hand-labelled queries with verified answers |
| **trap query** | A gold query whose corpus contains a superseded predecessor likely to be retrieved instead of the answer |
| **authority** | Whether a document is currently in force, as opposed to merely on-topic |
| **λ (lambda)** | The reranker's strength parameter, 0 to 1 |

Code is quoted verbatim from the repository. Where a line is abridged, `...` marks the omission.

---

## Table of Contents

### Part I — Foundations: Information Retrieval From Zero

| Chapter | Title |
|---|---|
| 1 | [What Retrieval Actually Is](ch01-what-retrieval-is.md) |
| 2 | [From Text to Something Searchable](ch02-text-to-searchable.md) |
| 3 | [Lexical Ranking: Counting Words Properly](ch03-lexical-ranking-bm25.md) |
| 4 | [Meaning as Geometry: Embeddings](ch04-embeddings.md) |
| 5 | [Knowing Whether It Worked: Retrieval Metrics](ch05-retrieval-metrics.md) |

### Part II — Core Mechanisms

| Chapter | Title |
|---|---|
| 6 | [Combining Rankers: Reciprocal Rank Fusion](ch06-rank-fusion.md) |
| 7 | [Python for Measurement Code](ch07-python-for-measurement.md) |
| 8 | [Talking to a Local Model Server](ch08-model-server.md) |
| 9 | [Brute-Force Nearest Neighbours with NumPy](ch09-numpy-search.md) |

### Part III — The Domain of This Project

| Chapter | Title |
|---|---|
| 10 | [The PEP Corpus: Headers, Statuses, Supersession](ch10-pep-corpus.md) |
| 11 | [Authority Is Not Relevance](ch11-authority-vs-relevance.md) |
| 12 | [Designing an Evaluation You Can Trust](ch12-designing-evaluation.md) |
| 13 | [Measuring the Measurer](ch13-measuring-the-measurer.md) |

### Part IV — Building the System

| Chapter | Title |
|---|---|
| 14 | [The Corpus Layer](ch14-corpus-layer.md) |
| 15 | [The Lexical Retriever](ch15-lexical-retriever.md) |
| 16 | [The Metrics Layer](ch16-metrics-layer.md) |
| 17 | [The Retriever Ladder](ch17-retriever-ladder.md) |
| 18 | [The Embedding Layer](ch18-embedding-layer.md) |
| 19 | [The Generation Layer](ch19-generation-layer.md) |
| 20 | [The Reranker](ch20-reranker.md) |
| 21 | [The Gold Set and the Drivers](ch21-gold-set-and-drivers.md) |

### Part V — Understanding the Finished System

| Chapter | Title |
|---|---|
| 22 | [One Query, End to End](ch22-end-to-end-trace.md) |
| 23 | [The Findings, and How They Were Reached](ch23-findings.md) |
| 24 | [Interview Preparation](ch24-interview-prep.md) |

---

## Verification status of this book

Written after reading every source file in the repository. Specifically:

**Run live while writing:** the full test suite (103 tests, all passing), `scripts/verify_gold.py`,
and all four phase drivers. Every number in Part V was produced by those runs.

**Checked by reading code:** every function signature, constant, default, and control-flow claim
in Part IV.

**Taken on faith:** nothing about this repository. Two external facts are taken from their
sources rather than re-derived — the BM25 formulation and the reciprocal-rank-fusion constant
`k=60` are standard from the literature, and the `nomic-embed-text` prefix convention comes from
that model's documentation. Both are flagged where used.

**Known discrepancy found and fixed during writing:** the repository's README reported retrieval
numbers measured when the gold set held 45 queries, before six version-scoped queries were added.
Those numbers were no longer reproducible. They were re-measured and corrected; see chapter 23.
