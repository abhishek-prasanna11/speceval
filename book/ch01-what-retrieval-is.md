# Chapter 1 — What Retrieval Actually Is

## Learning Objectives

By the end of this chapter you will be able to:

- State the retrieval problem precisely, and say why it is not the same problem as searching for
  a substring.
- Explain the difference between a *boolean* match and a *ranked* match, and why ranking is the
  harder and more useful idea.
- Define **relevance** and explain why it is a judgement rather than a fact.
- Describe the two-stage shape (retrieve, then generate) that almost every modern question
  answering system has, and locate `speceval` inside it.

## Motivation

Suppose you have 734 documents and a question: *"how do I postpone the evaluation of
annotations?"* You want the answer.

The obvious approach is the one your text editor gives you: search for the words. `Ctrl-F` for
"postpone". This fails immediately and instructively, in four separate ways.

**It finds too much.** The word "postpone" might appear in fifty documents, most of them in
passing. You now have fifty results and no idea which to read.

**It finds too little.** A document that explains the feature perfectly might call it *deferred
evaluation* and never use the word "postpone" at all. You will never see it. (This is not a
hypothetical: it is exactly what happens in this corpus, and chapter 11 shows the case.)

**It has no notion of better.** `Ctrl-F` gives you matches in the order they appear on disk.
Position in a file has nothing to do with usefulness.

**It cannot tell current from obsolete.** Two documents may both discuss the feature, one written
in 2017 and formally replaced in 2024, the other the current specification. `Ctrl-F` treats them
identically. This one is the subject of the entire second half of this book.

Retrieval is the discipline of doing better than `Ctrl-F` on all four counts.

## First Principles

Start with nothing and build the problem up.

### The setup

You have:

- A **corpus**: a finite set of documents `D = {d₁, d₂, ..., dₙ}`. In this project, n = 734.
- A **query**: some text `q` expressing what the user wants.

You want a function. The question is what shape it should have.

### Attempt 1: the boolean predicate

The simplest possible function returns yes or no for each document:

```
match(d, q) -> bool
```

Return every document for which `match` is true. This is what `Ctrl-F` does, and what early
database-backed search did with `WHERE body LIKE '%postpone%'`.

The fatal problem is not accuracy — it is that **the output has no order**. If 50 documents
match, you have converted "find the answer among 734 documents" into "find the answer among 50
documents." That is progress, but it is not a solution, and it gets worse as the corpus grows.
Double the corpus and you double the matches.

### Attempt 2: the scoring function

Change the return type from a boolean to a number:

```
score(d, q) -> float
```

Now sort all documents by that score, descending, and hand back the top few. This one change is
the entire conceptual leap from *search* to *retrieval*.

Everything that follows in Part I is an answer to one question: **what should `score` be?**

Chapter 3 gives one answer built from word counts (BM25). Chapter 4 gives a completely different
answer built from geometry (embeddings). Chapter 6 combines them. Chapter 20 adds a third kind of
information that neither of them can see.

### Attempt 3: the ranked list, formally

In practice the interface is:

```
retrieve(q, k) -> [d, d, d, ...]      # length k, best first
```

Two parameters hide in there and both matter enormously.

**`k`, the cutoff.** You return the top `k`, not everything. A user reads three results; a
downstream program might consume five or ten. The choice of `k` is not cosmetic — chapter 5 shows
that a metric evaluated at `k=10` can rank two systems in the *opposite* order from the same
metric at `k=5`, and chapter 23 shows this actually happening in this project.

**The unit of retrieval.** What is a `d`? A whole document? A paragraph? A sentence? A whole PEP
can be 10,000 words, which is far too much to hand to anything. So documents get cut into
**chunks** (chapter 2), and retrieval operates on chunks while the *answer* is usually attributed
to the document a chunk came from. This mismatch between "the thing you rank" and "the thing you
report" is a recurring source of subtle error, and chapter 17 shows how this project handles it.

## Mental Model

Think of a **reference librarian**, not a filing cabinet.

A filing cabinet does boolean matching: the folder is either in drawer B or it is not. It has no
opinion.

A librarian does something else entirely. You describe what you want in your own words. They
translate that into what the collection actually calls it. They hand you a small stack, best
first, having judged which is most likely to help. And a *good* librarian does one more thing that
a filing cabinet never will: they say *"careful — that edition is out of date, here is the current
one."*

That last sentence is the whole of this project. The librarian's judgement about **currency** is
separate from their judgement about **topic**, and a system that only models topic will confidently
hand you the 1997 edition.

## Deep Explanation

### Relevance is a judgement, not a property

It is tempting to think of relevance as something a document *has*. It is not. Relevance is a
relation between a document, a query, and a person's intent — and the third term is the problem,
because intent is not in the query text.

Consider the query *"does Python have a switch statement?"* Which documents are relevant?

- A rejected proposal from 2006 arguing for a switch statement, explaining exactly why it was
  refused. **Relevant** if you want to know why Python lacks one.
- The current specification of `match`/`case`, added in 2021, which does the job a switch
  statement would have done. **Relevant** if you want to know what to write today.

Both are defensible. Neither is wrong. This is query `q06` in this project's gold set, and it is
labelled with *both* documents for exactly this reason:

```json
{"qid": "q06", "text": "does Python have a switch case statement",
 "category": "rationale", "relevant": [3103, 634], "trap": true,
 "note": "PEP 3103 is Rejected. The live answer is match/case via 634."}
```

Two consequences follow, and they shape everything in Part III.

**First: relevance must be recorded by a human, in advance.** There is no algorithm that derives
it, because it depends on intent. A set of `(query, relevant documents)` pairs recorded by hand is
called a **gold set** (also *ground truth*, or *relevance judgements* in the literature). Chapter
12 is about building one you can trust.

**Second: the gold set is the ceiling on the entire study.** If the labels are wrong, every number
computed from them is wrong, and — this is the dangerous part — *nothing downstream will tell
you*. The pipeline still runs. The numbers still look plausible. Chapter 13 is about the only
defence against this.

### Precision and recall: the two ways to be wrong

Any retrieval system fails in exactly two directions, and they trade against each other.

```
                     retrieved              not retrieved
                 +---------------------+---------------------+
   relevant      |   true positive     |   FALSE NEGATIVE    |
                 |   (good)            |   (a miss)          |
                 +---------------------+---------------------+
   not relevant  |   FALSE POSITIVE    |   true negative     |
                 |   (noise)           |   (correctly ignored)|
                 +---------------------+---------------------+
```

- **Recall** — of the documents that were relevant, what fraction did you return? Punishes misses.
- **Precision** — of the documents you returned, what fraction were relevant? Punishes noise.

Return the entire corpus and recall is a perfect 1.0, with useless precision. Return one document
you are certain of and precision is 1.0, with terrible recall. Neither extreme is a system.

Chapter 5 makes these precise and explains why this project reports Recall@10 but not precision —
there is a specific reason, and it is not laziness.

### The two-stage architecture

Almost every modern question-answering system has the same shape:

```
   query
     |
     v
+----------------+     top-k chunks     +------------------+
|   RETRIEVER    | -------------------> |    GENERATOR     |
|                |                      | (language model) |
| searches the   |                      | writes prose     |
| whole corpus   |                      | from the chunks  |
+----------------+                      +------------------+
     |                                          |
     | reads all 19,763 chunks                  | reads only the 5 it was given
     v                                          v
  cheap, dumb                              expensive, fluent
```

This arrangement is commonly called **RAG** — retrieval-augmented generation. The name is less
important than the division of labour:

- The **retriever** is fast, has no language ability, and sees everything. In this project it
  takes tens of milliseconds.
- The **generator** is slow, fluent, and sees only what the retriever hands it. In this project it
  takes 12 to 14 *seconds* — roughly a thousand times longer (measured; chapter 19).

That asymmetry has a hard consequence: **the generator cannot rescue a bad retrieval.** If the
right document is not in the top-k, no amount of fluency will produce the right answer. The
generator will instead write something confident and wrong from whatever it was given. Chapter 19
shows this happening verbatim.

### Where `speceval` sits

`speceval` is not a RAG application. It is a **measurement harness** that happens to contain a
minimal RAG pipeline as its subject.

```
        +--------------------------------------------------+
        |                  speceval                        |
        |                                                  |
        |  +--------------------------------------------+  |
        |  |   the system under test (deliberately dull)|  |
        |  |   retriever  ->  generator                 |  |
        |  +--------------------------------------------+  |
        |                      |                           |
        |                      v                           |
        |  +--------------------------------------------+  |
        |  |   the measurement apparatus (the project)  |  |
        |  |   gold set, metrics, controls, sweeps      |  |
        |  +--------------------------------------------+  |
        +--------------------------------------------------+
```

The pipeline is kept as simple as it can be, on purpose, because it is the *thing being measured*.
Every complication in it would be a confounding variable. The interesting engineering is in the
apparatus around it.

## Systems Perspective

It is worth knowing what the machine is actually doing, because the costs are wildly uneven.

**Lexical retrieval** (chapter 3) is a dictionary lookup and some arithmetic. For each word in the
query, look up a list of documents containing it, add a number to each. Memory-bound, no
floating-point heavy lifting, microseconds to low milliseconds.

**Dense retrieval** (chapter 4) is a matrix multiply. This project multiplies a 19,763 × 768
matrix by a 768-element vector: about 15 million multiply-adds, a few milliseconds in optimised
BLAS. But it is preceded by a network round trip to a model server to turn the query into a
vector, and *that* dominates — it is why dense retrieval measures ~37 ms p50 here while BM25
measures ~28 ms, and almost all of the difference is the round trip, not the arithmetic.

**Generation** is thousands of matrix multiplies through a neural network, once per output token,
each one dependent on the last. Nothing about it is parallel across tokens. Hence seconds.

The ratio — microseconds, milliseconds, seconds — is the single most important performance fact
in the entire architecture. It explains why this project times retrieval separately from
generation (chapter 16); measuring them together would have hidden every difference between the
retrieval strategies under the generator's noise.

## Common Mistakes

**Believing the retriever because the answer sounded good.** Fluency is generated by the language
model and is completely independent of whether the retrieved documents were right. A confident,
well-written, wrong answer is the *normal* failure mode, not an unusual one.

**Evaluating on the queries you built the system with.** If you tune your retriever until it
handles your ten favourite examples, you have measured nothing except your own memory.

**Assuming a boolean filter is a ranking.** "Only return documents with status Final" is a filter.
It cannot express "this one is better than that one," and it throws away documents you may need.
Chapter 20 is careful about this distinction; it *reranks* rather than filters, and chapter 23
shows that the difference is why the project's central hypothesis turned out to be wrong.

**Treating recall at one cutoff as recall in general.** Recall@5 and Recall@50 can rank systems
differently. State the cutoff every time.

## Interview Insight

> **"What is the difference between search and retrieval?"**

Search returns matches; retrieval returns a *ranking*. The moment your scoring function returns a
number instead of a boolean you have to decide what that number means, and every hard question in
the field — how to weight rare words, how to compare a lexical score against a cosine similarity,
how to fold in metadata — is a consequence of that one decision.

A strong answer then adds the part most candidates miss: **ranked retrieval creates an evaluation
problem that boolean matching does not have.** With booleans you can just check correctness. With
a ranking you need to decide whether being right at position 1 is worth more than being right at
position 5, which is what nDCG exists to answer, and you need human relevance judgements before
you can compute anything at all.

> **"Why can't the language model just fix a bad retrieval?"**

Because it never sees the corpus. It sees the k chunks it was handed. If the answer is not in
them, its options are to hallucinate or to decline — and models overwhelmingly prefer the former.
In this project the generator was given the top 5 chunks out of 19,763; it had no access to the
other 19,758 and no way to know they existed.

## Debugging Tip

When a RAG system gives a wrong answer, **always look at the retrieved chunks before looking at
anything else.** In practice the majority of "the model is bad" reports are retrieval failures
wearing a costume. Log the retrieved document IDs on every request; without them you are
debugging blind.

This project takes that seriously enough to make it a first-class output: `run_phase2.py` prints
the rank-1 document for every query and every strategy side by side, so a regression is visible at
a glance rather than inferred from an aggregate score.

## Summary

- Retrieval is ranking, not matching. The leap is returning a number instead of a boolean.
- The scoring function is the whole subject. Parts I and II are four different answers to what it
  should be.
- Relevance is a human judgement about intent, not a property of a document. It must be recorded
  in advance, and the recording is the ceiling on everything measured afterwards.
- Precision and recall are the two directions of failure and they trade off.
- The standard architecture is retrieve-then-generate, with a thousandfold cost asymmetry between
  the stages, and the generator cannot repair the retriever's mistakes.
- `speceval` is a measurement harness containing a deliberately boring pipeline.

## Key Takeaways

1. `score(document, query) -> float`, then sort. That is retrieval.
2. Always state your cutoff `k`. Conclusions can invert when it changes.
3. The gold set is the ceiling. Everything downstream inherits its errors silently.
4. Retrieval is milliseconds, generation is seconds. Time them separately or measure nothing.

## Why the Next Chapter Exists

We have decided we need `score(d, q) -> float`. We cannot write it yet, because we have no
representation to compute with: a document is currently an undifferentiated blob of text, and text
does not multiply, sort, or compare.

Chapter 2 turns text into a structure you can do arithmetic on — tokens, a vocabulary, and the
inverted index — and shows the choices in that translation which quietly determine what your
retriever will be good and bad at.
