# Chapter 11 — Authority Is Not Relevance

## Learning Objectives

- State the project's thesis precisely, in terms a metric can test.
- Explain why the most semantically relevant document is often the wrong answer.
- Explain the mechanism by which *better* semantic retrieval becomes *worse* at authority.
- Distinguish the two failure modes — superseded citation and version incorrectness — and say why they
  must be measured separately.
- Explain why this is not a corpus quirk but a general property of evolving documentation.

## Motivation

Everything so far has been about relevance: making `score(d, q)` reflect how well a document matches a
query. Chapters 3, 4 and 6 built three increasingly good answers.

This chapter argues that relevance was never the whole objective, and that a system optimising it alone
has a systematic failure — not an occasional one, a *systematic* one, whose rate this project measures
at roughly one answer in four.

## First Principles

### Two different questions

When a user asks *"how do I postpone the evaluation of annotations?"*, they are implicitly asking two
things:

1. **Which document is about this?** — a question about topic. This is relevance.
2. **Which document should I follow?** — a question about standing. Call it **authority**.

Retrieval, as built in Part I, answers only the first. Nothing in BM25's term statistics or in an
embedding's geometry has any concept of a document being in force. Both are functions purely of *text*.

For most corpora that is fine, because topic and standing are uncorrelated — a Wikipedia article about
photosynthesis is not competing with a superseded version of itself.

For an evolving specification corpus, it is not fine at all, because **the competing documents are the
same document at different points in time.**

### Why the wrong answer scores highest

Here is the specific case, and it is worth sitting with because the whole project follows from it.

Two documents in the corpus:

```
   PEP 563   Postponed Evaluation of Annotations                       Status: Superseded
   PEP 649   Deferred Evaluation Of Annotations Using Descriptors      Status: Final
```

The query: *"how do I postpone the evaluation of annotations"* (gold query `q16`).

Consider what each retriever sees.

**BM25** matches tokens. The query token `postpone` does not exactly match `postponed`, so neither
document gets credit for the query's most distinctive word (chapter 2's vocabulary mismatch). The
remaining tokens — `evaluation`, `annotations` — appear in both titles. BM25 has little to separate
them, and in the measured run it happened to rank PEP 649 first. **Correct, and partly by luck.**

**The embedding model** does much better at the semantic task. `postpone` and `postponed` are nearly
identical in vector space; so are `postpone` and `deferred`. It understands the query.

And it ranks **PEP 563 — the superseded one — first.**

Why? Because PEP 563's title is *Postponed Evaluation of Annotations*, which is a near-exact semantic
restatement of the query. PEP 649's title is *Deferred Evaluation Of Annotations Using Descriptors* —
also relevant, but with extra qualifying words that dilute the match.

The embedding did its job perfectly. Its job was the wrong job.

```
   query:  "how do I postpone the evaluation of annotations"
                            |
            +---------------+---------------+
            |                               |
            v                               v
   PEP 563  "Postponed Evaluation      PEP 649  "Deferred Evaluation Of
             of Annotations"                     Annotations Using Descriptors"
   cosine:  very high  <-- WINS         cosine:  high
   status:  Superseded  <-- WRONG       status:  Final  <-- the actual answer
```

### The uncomfortable generalisation

Now the part that is genuinely counterintuitive, and it follows directly from chapter 4's
distributional hypothesis.

A superseded document and its replacement:

- discuss the same topic,
- use the same technical vocabulary,
- were written by overlapping authors in the same house style,
- and often the superseded one has the *cleaner* title, because the replacement had to distinguish
  itself.

They are, in embedding space, almost the same point. And nothing about "same point in embedding space"
distinguishes the live one.

So: **the better a retriever becomes at semantic similarity, the more attractive a superseded
predecessor becomes.** Not despite being good at its job — *because* of it.

This is the project's thesis, and it is falsifiable. If it is true, improving retrieval quality should
not improve the authority error rate, and might worsen it.

### The prediction, and the measurement

That prediction was tested. Measured on the 51-query gold set, live:

| | Recall@10 | nDCG@10 | rank-1 correct | trap@1 |
|---|---|---|---|---|
| BM25 | 0.863 | 0.671 | 47.1% | 0.294 |
| Dense | **0.971** | **0.801** | **60.8%** | 0.255 |

Dense is dramatically better at retrieval — 60.8% versus 47.1% rank-1 accuracy. Its overall trap rate
is modestly better too (0.255 vs 0.294).

But split the queries into those where a superseded predecessor exists (**trap**, n=20) and those where
one does not (**ordinary**, n=31):

```
Retriever                  trap (n=20)           ordinary (n=31)
----------------------------------------------------------------
BM25              R 0.82 N 0.62 T 0.50      R 0.89 N 0.71 T 0.16
Dense             R 0.93 N 0.67 T 0.55      R 1.00 N 0.89 T 0.06
Hybrid            R 0.93 N 0.66 T 0.55      R 0.94 N 0.83 T 0.13
```

Read the `T` column:

- On **ordinary** queries dense is far better on authority: 0.06 versus 0.16 — it makes a quarter as
  many authority errors.
- On **trap** queries dense is **worse**: 0.55 versus 0.50. Eleven of twenty versus ten of twenty.

The entire advantage of the better retriever evaporates exactly where supersession is present. And it
tips very slightly negative.

### Honesty about the size of that effect

Eleven versus ten out of twenty is **one query**. That is not a finding, and the project says so
plainly:

> **Stated carefully: a one-query difference at n=18 is not a finding.** What the data supports is the
> weaker but still interesting claim that *dense retrieval's large advantage at finding the right
> document does not produce any advantage on the queries where supersession matters.*

(That quotation is from the original 45-query measurement; the 51-query re-run has 11 versus 10, and the
same caveat applies with the same force.)

The robust claim is not "dense is worse on traps". It is: **a large, unambiguous improvement in
retrieval quality bought no improvement at all on the queries where authority matters, and over half
of those queries still lead with a dead document regardless of which retriever runs.**

That is the gap chapter 20 exists to close, and it is measured rather than assumed.

### The mechanism, visible in the divergence

The subset numbers show *that* it happens. The divergence shows *why*.

BM25 and dense fall into different traps. Measured on the 51-query set:

```
   dense-only traps (BM25 got these right):   q16, q22, q24, q47
   BM25-only traps (dense got these right):   q02, q04, q15, q20, q28, q48
   shared:                                    9
```

The dense-only cases share one shape — the query paraphrases the *superseded* document's title almost
exactly:

| Query | Dense returns | Correct answer |
|---|---|---|
| q16 "how do I postpone the evaluation of annotations" | 563 *Postponed Evaluation of Annotations* `Superseded` | 649 `Final` |
| q22 "which manylinux platform tag should a wheel target" | 571 *manylinux2010* `Superseded` | 600 `Final` |
| q24 "is return allowed inside a finally block" | 601 `Rejected` | 765 `Final` |
| q47 "is zoneinfo available in Python 3.8" | 431 `Superseded` | 615 `Final` |

Four cases, one mechanism: semantic strength locking onto a dead document whose wording is closer to
the question.

Meanwhile BM25's traps are different in kind — its failures come from *weakness*, retrieving something
tangentially related because it had no strong lexical signal. For q02, *"what does the walrus operator
:= do"*, BM25's top result is PEP 622, `Superseded`, and about **pattern matching** — not even the right
topic. That is not the authority problem; it is ordinary retrieval failure that happens to land on a
non-authoritative document.

So the two retrievers produce authority errors for opposite reasons: **BM25 fails by being bad at
retrieval, dense fails by being good at it.** Only the second is the phenomenon this project is about,
and it is the one that will not be fixed by a better embedding model.

## Mental Model

A **medical librarian** and a **very well-read colleague**.

Ask both about a treatment. The colleague, who has read everything, immediately recalls a beautifully
written paper that describes exactly what you asked about — and was retracted in 2019. They recall it
*because* it is the clearest statement of the thing you described. Their fluency led them straight to
it.

The librarian is slower and less articulate but checks the standing of what they hand you.

Notice you cannot fix the colleague by making them better read. Retraction status is not in the prose.
It is metadata, and it must be consulted separately.

## Deep Explanation: two failure modes, not one

The thesis splits into two distinct errors. They must be measured separately because they move
independently.

### Failure mode 1: superseded citation

Answering from a `Rejected`, `Withdrawn`, `Superseded`, or `Deferred` document as though it were
current. Measured example from the Dense rung, verbatim:

> **q23** — *"how are Python package version numbers compared"* → *"According to PEP 386, Python
> package version numbers are compared using the standard schema specified in that…"*

PEP 386 is `Superseded` by 440. No hedge, no qualification — a dead specification presented as current.

### Failure mode 2: version incorrectness

A document that is entirely authoritative, describing a feature that does not exist in the version
asked about. Measured example:

> **q20** — *"can I run multiple interpreters from the standard library"* → *"Yes, you can run multiple
> interpreters from the standard library."* citing PEP 554.

PEP 554 is `Superseded` by 734, and the feature lands in 3.14. Both failures at once here, but the
second is separable: a `Final` PEP with `Python-Version: 3.14` cited in answer to a question about 3.9
is version-incorrect while being perfectly authoritative.

### Why separating them matters

They have different fixes. Superseded citation is addressed by reading the status and supersession
graph. Version incorrectness is addressed by comparing the version asked about to the version the
feature landed in — and chapter 20 shows that the obvious way to do *that* actively harms results.

Had they been collapsed into one metric, the second finding would have been invisible: the harmful
version rule leaves superseded citation completely unchanged (0.039 both ways) and damages only the
other columns. A combined metric would have shown "no effect" and the finding would have been lost.

## Systems Perspective

Authority is **metadata**, not text. That has an architectural consequence worth stating.

Relevance is computed *from the document body*. Authority is read from a *header field* and a *graph
edge*. They live in different places, are computed by different means, and — crucially — the authority
signal costs essentially nothing to consult: a dictionary lookup, no model call, no arithmetic.

```
   +----------------------+          +----------------------+
   |   TEXT               |          |   METADATA           |
   |  chunk bodies        |          |  Status              |
   |                      |          |  Superseded-By       |
   |  -> BM25 scores      |          |  Python-Version      |
   |  -> embeddings       |          |                      |
   |                      |          |  -> authority weight |
   |  expensive           |          |  free                |
   +----------------------+          +----------------------+
              \                          /
               \                        /
                v                      v
              +--------------------------+
              |   final ranking          |
              +--------------------------+
```

The asymmetry is the practical punchline. The expensive half of the system — an embedding model, a
matrix multiply — cannot see authority at all. The free half can. So the fix is cheap, which is one
reason chapter 23's result (that the fix costs nothing) is less surprising in hindsight than it seemed
in advance.

## Common Mistakes

**Assuming a better retriever fixes this.** Measured here: a 13-point rank-1 accuracy improvement
bought nothing on the trap subset.

**Assuming the generator will notice.** It sees only the chunks it is given, and in this project the
prompt deliberately withholds status (chapter 19) so that the retrieval comparison is not confounded.
Even with status visible, a model has no obligation to act on it.

**Collapsing the two failure modes into one metric.** The version finding would have been invisible.

**Filtering rather than reranking.** A rejected proposal is often the best available explanation of why
something is *not* in the language — gold query `q06` labels a `Rejected` PEP as relevant for exactly
that reason. Filtering it out would make that query unanswerable.

**Treating this as a corpus quirk.** Any corpus that evolves has it: API documentation across versions,
internal runbooks, policies with amendment histories, changelogs. The PEP corpus is unusual only in
*recording* it machine-readably, which is what makes it measurable.

## Interview Insight

> **"What was the actual problem you found?"**

Retrieval optimises relevance, but users need authority, and in a corpus where documents supersede one
another those two come apart. The superseded document and its replacement are near-identical in topic and
vocabulary, so they sit almost on top of each other in embedding space — and the superseded one often has
the *cleaner* title, because the replacement had to distinguish itself.

The concrete case: asked *"how do I postpone the evaluation of annotations"*, the embedding model ranks
PEP 563, titled *Postponed Evaluation of Annotations*, above PEP 649, the live specification. It matched
the wording of the dead document better. Measured, roughly one generated answer in four cited a
non-authoritative document, rising to about half on the queries where a superseded predecessor exists.

> **"So a better embedding model would fix it?"**

No, and that is the interesting part. I measured a retriever with a 13-point rank-1 accuracy advantage
and it bought *zero* improvement on the subset where supersession is present — 11 of 20 versus 10 of 20,
which is a one-query difference and therefore no difference. Its four unique failures were all cases
where it locked onto a superseded document whose title paraphrased the query more closely than the live
one did.

The reason is structural: authority is not in the prose. It is in a header field and a graph edge. No
amount of reading the text recovers it.

> **"How did you know it was the mechanism you claimed and not coincidence?"**

By looking at where the two retrievers *disagreed* rather than only at aggregates. Nine trap failures
were shared, but four were dense-only and six were BM25-only, and the two sets have different shapes —
dense's failures are all near-exact semantic matches to a superseded title, while BM25's are ordinary
retrieval misses that happened to land on a non-authoritative document. Same metric, two different
causes. The aggregate number cannot tell you that; the disagreement set can.

## Debugging Tip

When investigating an authority failure, print the **status of every retrieved document alongside its
rank**, not just the ranking. `speceval` builds this into its drivers:

```
q16  availability     649[Fina]*   563[Supe]    563[Supe]   [649]
```

Four columns: query, category, then rank-1 for BM25 / Dense / Hybrid with status abbreviated in
brackets, then the gold answer. A dead document at rank 1 is visible instantly. Without the status
annotation you would see three plausible PEP numbers and notice nothing.

## Summary

- A user asks two questions at once: what is this about (relevance) and what should I follow
  (authority). Retrieval answers only the first.
- In an evolving corpus the competing documents are the same document at different times, so they are
  near-identical in topic and vocabulary — and the superseded one often has the cleaner title.
- Therefore the better a retriever is at semantics, the more attractive a superseded predecessor
  becomes. This is the project's thesis.
- Measured: dense beats BM25 by 13 points of rank-1 accuracy and gains **nothing** on the trap subset
  (11 of 20 vs 10 of 20 — one query, so no real difference). Over half of trap queries lead with a dead
  document regardless of retriever.
- The divergence set shows the mechanism: dense's unique failures are near-exact semantic matches to
  superseded titles; BM25's are ordinary retrieval weakness. Same metric, opposite causes.
- Two separable failure modes — superseded citation and version incorrectness — with different fixes.
  Collapsing them would have hidden the version finding entirely.
- Authority is metadata, so it is free to consult and invisible to the expensive part of the system.

## Key Takeaways

1. Relevance and authority are different quantities; optimising one does not improve the other.
2. Better semantic retrieval can be *worse* at authority, because semantic strength locks onto
   near-duplicate dead documents.
3. Measure the disagreement set, not just aggregates — that is where mechanism lives.
4. Authority is metadata: cheap to read, impossible to infer from text.

## Why the Next Chapter Exists

We have a thesis and a claim that it is measurable. We do not yet have anything to measure it with. A
gold set of 51 queries has been referenced repeatedly without being justified: why 51, why those, why
40% of them traps, and how anyone can trust the labels.

Chapter 12 builds it — including the two decisions that made it affordable, and the control group
without which the reranker's benefit could only ever have looked good.
