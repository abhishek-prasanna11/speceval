# Chapter 12 — Designing an Evaluation You Can Trust

## Learning Objectives

- Explain why the gold set is the ceiling on every number a study produces.
- Explain the two decisions that made this gold set affordable, and what each cost.
- Explain how the corpus metadata **generates** hard test cases instead of them being invented.
- Explain why a control group of non-trap queries is mandatory, not optional.
- Explain why queries are tagged by category, and what aggregates hide.
- Explain the semantic bug in a field name that would have silently corrupted a metric.

## Motivation

Chapter 5 established that a metric needs relevance judgements. Chapter 11 established what we are
trying to detect. This chapter builds the instrument, and it is worth being blunt about the stakes.

**Everything a study reports is downstream of its gold set.** If a label is wrong, every metric
computed from it is wrong, the pipeline still runs, the number still looks plausible, and nothing
anywhere will tell you. There is no test for "my ground truth is mistaken."

That makes gold-set construction the highest-leverage work in the project and also the least
glamorous. It is the part most people skip, which is precisely why having done it is worth something.

## First Principles

### What a gold set entry has to contain

Minimally: a query, and the documents that answer it.

```json
{"qid": "q02", "text": "what does the walrus operator := do",
 "category": "identifier", "relevant": [572], "trap": false,
 "note": "PEP 572, Final, 3.8."}
```

Six fields, and every one earns its place:

| Field | Purpose |
|---|---|
| `qid` | Stable identifier, so a query can be discussed across runs and documents |
| `text` | The query as a user would type it |
| `category` | Query *type*, for disaggregation (below) |
| `relevant` | The labels — PEP numbers that legitimately answer it |
| `trap` | Whether a superseded predecessor exists to be retrieved instead |
| `note` | Why these labels, in prose, for a human re-checking them later |

The `note` field deserves a word. It is never read by code. It exists so that a person — including the
author six months later — can audit a label without re-deriving the reasoning. In a gold set, an
unexplained label is an unverifiable label.

### How large?

The measured gold set is **51 queries**. That number is a compromise, and the honest way to talk about
it is in terms of what it buys and what it does not.

At 51 queries with a metric that is a rate, one query is worth about 2 percentage points. So:

- A difference of 0.30 versus 0.05 (the headline result) is fifteen versus one — a real effect.
- A difference of 0.255 versus 0.294 (dense versus BM25) is thirteen versus fifteen — two queries. Not
  a finding, and the book says so wherever it appears.

The project states this constraint at every point where a number is reported. That discipline is more
valuable than a larger number would have been, because it means the reader can tell which claims to
lean on. Expanding to ~150 queries is listed in the README's future improvements as the single
highest-value addition.

### The first cost-saving decision: label documents, not chunks

Retrieval operates on 19,763 chunks. Labelling relevance per chunk would mean, for each query,
inspecting every chunk that might be relevant — thousands of judgements per query.

`speceval` labels at **document** level and collapses chunk rankings to document rankings before
scoring:

```python
"""Relevance is labelled at **PEP level**, not chunk level: a query's ground truth is the
set of PEP numbers that legitimately answer it. Retrievers return ranked chunks, which
are collapsed to ranked distinct PEPs before scoring. This is a deliberate choice --
labelling every relevant *chunk* by hand would cost several times more and would not
change which retrieval strategy wins.
"""
```

**What it costs:** resolution. You cannot ask "did it find the right *paragraph*?" And chapter 23
records that this loss mattered — the retrieval metrics were computed over documents while the
generator consumed chunks, and that mismatch inverted a conclusion.

The decision was still correct. A chunk-level gold set was not affordable, and an unaffordable gold set
is no gold set.

### The second cost-saving decision: metrics from metadata

The other expensive thing in an evaluation is grading *answers*. Was this generated paragraph correct?
That normally means human grading, or an LLM-as-judge, which then itself needs validating.

`speceval` avoids both on its primary path, because the corpus carries the answer:

```python
"""No LLM judge and no human grading on this path. That single constraint is what keeps the
project small, and it is possible only because the corpus carries machine-readable authority
(`Status`) and version (`Python-Version`) metadata.
"""
```

An answer cites PEP 386. Look up 386's status: `Superseded`. That is a superseded citation, determined
by a dictionary lookup. No judgement required.

**What it costs** is precision about what is being measured, and chapter 19 documents two cases where
the metric marks an answer wrong that a human would not. The response was to *narrow the claim* —
the README now says the metric measures citation hygiene, **not** answer truth — rather than to
add machinery. Narrowing a claim to match what you actually measured is usually the right move.

## Deep Explanation: letting the metadata generate the hard cases

This is the part that made a 51-query set with 20 trap cases feasible in an afternoon rather than a
week.

A trap query needs a superseded predecessor that a retriever will plausibly surface instead of the
answer. Inventing those requires knowing the corpus deeply. But the corpus *already knows* — every
`Superseded-By` edge is a candidate trap by construction.

So they were enumerated rather than imagined:

```python
for p in peps.values():
    if p.superseded_by and p.superseded_by in peps:
        n = peps[p.superseded_by]
        rows.append((p.number, p.title, p.status, n.number, n.title, n.status, ...))
```

That produced 31 usable pairs, and reading them was how the four structural cases in chapter 10 were
discovered — the both-`Final` pairs, the four-hop packaging chain, the manylinux convergence, and the
no-authoritative-answer case. None of those were anticipated; they were *found* by looking at what the
metadata said.

This is worth generalising: **when your corpus has machine-readable structure, use it to generate your
hard cases.** You get better coverage than intuition provides, and you discover structure you did not
know was there.

## Deep Explanation: the control group

Here is the methodological point that most distinguishes this evaluation.

The obvious way to build a gold set for this study is to fill it with trap queries — those are the
interesting ones, the ones the intervention targets. That would be a serious error.

If every query has a superseded predecessor, then an intervention that demotes superseded documents can
only help. There is no query on which it could do damage, so **you have made it impossible to measure
the cost.** You would report a large benefit and have no idea what you paid.

So the set is deliberately mixed: **20 trap queries (39%) and 31 ordinary ones (61%)**, verified live:

```
queries        51
categories     availability=21, identifier=15, rationale=15
trap cases     20 (39%)
ordinary       31
PEPs referenced 48
```

The reasoning is recorded in the code that reads the field:

```python
    # True when the corpus contains a superseded/rejected predecessor that a naive
    # retriever is likely to surface instead of the answer. Reported as its own subset:
    # authority reranking must be measurable on the queries it cannot help, or its benefit
    # is being scored only where it was designed to win.
    trap: bool = False
```

And the driver prints both subsets side by side, with the control named as such:

```
R = Recall@10, N = nDCG@10, T = trap@1. The ordinary subset is the control:
a reranker that only improves the trap column is buying its gain somewhere.
```

Chapter 23 shows this paying off in an unexpected direction: the intervention improved the ordinary
subset too. Without the control, that could not have been observed, and the result would have been
weaker.

## Deep Explanation: categories, and what aggregates hide

Queries are tagged with one of three types:

| Category | n | Shape | Example |
|---|---|---|---|
| `availability` | 21 | Does this exist, and when? | *"can I use structural pattern matching in Python 3.9"* |
| `identifier` | 15 | What is this specific thing? | *"what does the walrus operator := do"* |
| `rationale` | 15 | Why was this designed this way? | *"why was the print statement changed into a function"* |

These are not decorative. Measured per-category nDCG@10, live:

| Category | BM25 | Dense |
|---|---|---|
| availability | 0.62 | 0.75 |
| identifier | 0.77 | 0.86 |
| rationale | 0.64 | 0.81 |

The aggregate figures — 0.671 and 0.801 — hide the fact that BM25's weakness is concentrated in
`availability` and `rationale`, both of which ask about *discussion* rather than naming a thing, and
therefore offer less lexical signal. That is a mechanism, and it is only visible disaggregated.

Chapter 5 noted that scoring is free, so there is no cost to computing every subset you can think of.
The rule: **aggregate to report, disaggregate to understand.** Findings live in the breakdown.

One honesty note the project records: the categories are now unbalanced at 21/15/15, because the six
version-scoped queries added later are all availability questions. Per-category comparisons on that
axis are correspondingly weaker.

## Deep Explanation: a field name that was semantically wrong

This is a genuine bug, caught before it corrupted anything, and it is instructive because it was a
*naming* problem rather than a logic problem.

Queries originally carried a field called `python_version`. Its meaning drifted:

```
   q01  "can I use structural pattern matching in Python 3.9"    python_version: "3.9"
        ^ the version ASKED ABOUT (the answer is 3.10)

   q04  "which Python version added f-strings"                   python_version: "3.6"
        ^ the version of the ANSWER
```

Two different meanings in one field. Any metric reading it would be comparing incomparable things — for
q01 it would check whether the answer mentions 3.9, when the correct answer is precisely that the
feature is *not* in 3.9.

The fix was to give the field one meaning and rename it to say so. `asked_version` means **only** a
version named in the query text. Everything else moved into `note`, which no code reads.

Then the invariant was enforced in both directions:

```python
        # asked_version means exactly one thing: a Python version named in the query text.
        # It must not be confused with the version the *answer* involves -- conflating the
        # two silently corrupts the version metric, since a trap query deliberately asks
        # about a version predating the feature.
        version = record.get("asked_version")
        if version:
            if version not in record.get("text", ""):
                problems.append(
                    f"{qid}: asked_version {version!r} does not appear in the query text"
                )
            ...
        elif re.search(r"\b\d+\.\d+\b", record.get("text", "")):
            problems.append(
                f"{qid}: query text names a version but asked_version is unset"
            )
```

Both directions matter. The first catches a field set to something not in the text. The second catches
a query that names a version but forgot the field — which would silently exclude it from the version
metric.

With one consistent meaning, only q01 qualified out of the original 45 queries. That left the version
metric resting on a single query, which is no metric at all. So **six version-scoped queries were added
specifically to make it testable** — q46 through q51, each naming a version in its text:

```json
{"qid":"q46","text":"can I use the walrus operator in Python 3.7","relevant":[572],
 "asked_version":"3.7","note":"PEP 572 is 3.8, so the answer must say 3.8, not yes."}
```

Note the deliberate mix: q48 asks about a version where the answer genuinely *is* yes, so the metric is
not measuring a system that has learned to always say no.

Seven version-scoped queries is still thin, and the book says so at every point where that metric is
reported.

**The transferable lesson:** a field whose meaning drifts is worse than a missing field. Name fields for
exactly what they contain, and enforce the meaning in a check rather than in a comment.

## Mental Model

Building a gold set is **writing the exam before teaching the course**, and writing it so it cannot be
gamed.

Include questions the syllabus covers well and questions it covers badly. Include some where the
obvious answer is wrong, and some where it is right — otherwise a student who always distrusts the
obvious answer scores perfectly without understanding anything.

That last clause is the control group.

## Systems Perspective

The gold set is a 51-element JSON file — kilobytes, read once, negligible. Its cost is entirely human
time, and the two decisions above were about reducing that.

There is one operational hazard. The gold set is labelled against a corpus that is not vendored: the PEP
repository is cloned fresh and gitignored. So the corpus can change underneath the labels — a `Draft`
becomes `Final`, a `Final` gains a `Superseded-By`, a new PEP appears. Nothing about the gold set file
would change, and a label could quietly become wrong.

That hazard is why `scripts/verify_gold.py` exists and is wired to exit non-zero. Chapter 21 covers it
in detail; the principle is that **a gold set labelled against a moving corpus needs a tripwire**, and
the tripwire must fail the build rather than print a warning.

## Common Mistakes

**Filling the set with the interesting cases.** Then your intervention cannot lose and you have measured
a benefit with no cost.

**Labelling with no recorded reasoning.** An unexplained label cannot be audited, including by you.

**One field, two meanings.** Silently corrupts whatever reads it.

**Reporting only aggregates.** The mechanism is in the breakdown, and computing the breakdown is free.

**Building the set after seeing the results.** Then you have fitted your ground truth to your system.
The trap/ordinary split and the categories here were fixed before the reranker existed.

**Treating "n is small" as something to omit.** It is a property of the result and belongs next to it.

## Interview Insight

> **"How did you build your evaluation set?"**

Fifty-one queries with hand-verified labels, and three decisions worth naming.

Labels are at document level rather than chunk level — chunk-level labelling would have cost several
times more and would not change which strategy wins. The hard cases were *generated from the corpus
metadata* rather than invented: enumerating the 31 `Superseded-By` edges gave a candidate trap for each,
and reading that list is how I found the structural cases I had not anticipated — pairs where both
documents are `Final`, a four-hop chain, and one case where no authoritative answer exists at all.

And the set is only 39% trap cases. The other 61% are the control, because an intervention evaluated
only on queries it was designed to win cannot be measured for cost.

> **"Why not use an LLM to judge the answers?"**

Because the corpus carries the answer in machine-readable metadata, so citation checks are a dictionary
lookup. That kept the project small and removed a component that would itself need validating — a judge
you have not measured against human labels is just another unvalidated model in your pipeline.

The honest cost: it measures citation *hygiene*, not answer truth. I found two cases where a correct
answer scores as a failure — one where the model drew the right conclusion from a superseded source, and
one where it correctly hedged that nothing was settled. My response was to narrow the claim in the
documentation rather than to add machinery, and to list the validated judge as the highest-value future
addition.

> **"What is the weakest part of your evaluation?"**

Fifty-one queries. The headline effect is fifteen queries versus one, which is real, but several smaller
comparisons are one or two queries and I report them as not findings. Also, the version metric rests on
seven queries, which is anecdote — I say so wherever it appears.

## Debugging Tip

Before trusting any number from an evaluation, print the composition of the set: total, per category,
per subset, and how many distinct documents are referenced. `verify_gold.py` does this every run:

```
queries        51
categories     availability=21, identifier=15, rationale=15
trap cases     20 (39%)
ordinary       31
PEPs referenced 48
```

Forty-eight distinct PEPs across 51 queries means the set is not concentrated on a handful of documents —
which would be an easy and invisible way for a gold set to be unrepresentative.

## Summary

- The gold set is the ceiling on every number downstream, and there is no test for a wrong label.
- 51 queries: enough for a large effect (fifteen queries versus one), not enough for a small one (two
  queries). The constraint is restated wherever a number appears.
- Labels are at document level — a deliberate cost saving whose price was resolution, and chapter 23
  records that price being paid.
- Answer metrics come from corpus metadata, not a judge — which kept the project small and narrowed what
  can be claimed.
- Hard cases were *generated* from the 31 `Superseded-By` edges, which is how the four structural cases
  were discovered rather than invented.
- 39% traps and 61% control, because an intervention measured only where it was designed to win has no
  measurable cost.
- Three query categories, because the aggregate hides that BM25's weakness is concentrated in the
  categories with least lexical signal.
- A field with two meanings (`python_version`) was renamed to one (`asked_version`) and the meaning
  enforced in both directions; six queries were then added so the version metric was testable at all.

## Key Takeaways

1. Include a control group. Without it, benefit is unfalsifiable.
2. Generate hard cases from metadata where you can — better coverage, and you find structure you did
   not know about.
3. One field, one meaning, enforced by a check rather than a comment.
4. Aggregate to report, disaggregate to understand.

## Why the Next Chapter Exists

We have a gold set and metrics. There is one gap left, and it is the one that would invalidate
everything else: **how do you know the metric implementation is correct?**

A metric bug produces plausible numbers. No test elsewhere in the system detects it. Chapter 13 is about
the only real defence — running retrievers whose scores are known in advance through the real evaluation
loop — and it includes two measured results that show the technique catching things nobody was looking
for.
