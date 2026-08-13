# Chapter 19 — The Generation Layer

`speceval/generate.py` (207 lines) · `speceval/answer_metrics.py` (161 lines)

## Learning Objectives

- Explain what the prompt contains, what it withholds, and why withholding is load-bearing.
- Explain the citation regex and the two extraction functions.
- Explain the four answer-level metrics and how each is computed from corpus metadata.
- Explain the bug the test suite caught in `build_context`.
- Explain the two cases where a correct answer scores as a failure, and what follows from them.

## Motivation

Everything so far ranks documents. This layer turns the top-ranked chunks into prose and then measures
whether that prose cited documents that are still in force.

Two things here are more consequential than they look: what the prompt *omits*, and the precise wording
of what the metrics claim to measure.

## Deep Explanation: the prompt

```python
PROMPT_TEMPLATE = """You are answering questions about Python using only the excerpts below.

Rules:
- Answer in at most three sentences.
- Cite every PEP you rely on, in the form "PEP 634".
- If the excerpts do not settle the question, say so.

Excerpts:
{context}

Question: {question}

Answer:"""
```

Three instructions, each doing specific work.

**"at most three sentences"** bounds output length, which bounds generation time (each token is a
forward pass) and keeps answers short enough that citations are easy to locate. It is belt-and-braces
with `MAX_TOKENS = 220`.

**"Cite every PEP … in the form `PEP 634`"** specifies a format the extractor can parse. Without a
stated format the model would produce *PEP-634*, *pep 634*, *Python Enhancement Proposal 634* and
prose references — and the citation metrics depend entirely on finding them.

**"If the excerpts do not settle the question, say so"** gives the model an exit other than
fabrication. This matters for gold query `q30`, where no authoritative answer exists (chapter 10). The
model took the exit, verbatim:

> *"There is no definitive answer based on the provided excerpts. PEP 543 and PEP 748 both propose a
> unified TLS API…"*

That is exactly the desired behaviour, and — as the last section of this chapter shows — the metric
penalises it anyway.

### What the prompt withholds

```python
def build_context(chunks: list[Chunk], max_chars: int = 6000) -> str:
    """Format retrieved chunks for the prompt.

    Includes PEP number, title and section so the model can cite precisely -- and nothing
    about status or version, so the generator cannot compensate for bad retrieval.
    """
```

Each chunk becomes:

```
[PEP 634] Structural Pattern Matching: Specification -- Abstract
This PEP provides the technical specification for the match statement...
```

Number, title, section, text. **No `Status`. No `Python-Version`.**

This is a deliberate experimental control, and it is the most important design decision in the module:

```python
"""**The context deliberately omits each PEP's status.** Showing `Status: Superseded` would let
the generator route around a bad retrieval, which is a *prompt-level* intervention and would
confound the comparison between retrieval strategies. The generator is held constant and
uninformed on purpose; making it authority-aware is a separate experiment, noted in the
README as future work.
"""
```

The reasoning: the study compares four *retrieval* strategies. If the generator could see that a chunk
came from a superseded document, it might decline to cite it — and the measured superseded-citation
rate would then reflect the model's caution rather than the retriever's ranking. Every rung would
improve, by an amount depending on how compliant the model happened to be, and the comparison would
measure nothing.

So the generator is held constant and uninformed. Its behaviour is a fixed function of the chunks it
receives, which makes any difference in the answer metrics attributable to which chunks arrived.

Pinned by a test that will fail if anyone ever adds status to the context:

```python
def test_context_never_leaks_status(self) -> None:
    # The whole comparison depends on the generator being unable to route around a bad
    # retrieval. If status reached the prompt, that would be a different experiment.
    context = build_context(self.chunks)
    for status in ("Superseded", "Rejected", "Withdrawn", "Status"):
        self.assertNotIn(status, context)
```

Making the generator authority-aware is a genuinely interesting *separate* experiment — does telling
the model a document is superseded work as well as reranking? — and it is listed as future work rather
than smuggled in.

### The bug the tests caught

The original budget loop was:

```python
        if used + len(block) > max_chars:
            break
```

If the *first* block exceeded `max_chars`, the loop broke immediately with `parts` empty and returned
`""`. The model would receive a prompt with no excerpts at all and answer from parametric memory.

The failure mode is what makes this worth recording. The run does not crash. The model produces a
fluent answer. The citations are extracted normally. The numbers land in the table — and they look like
*a retrieval failure*, because the answer cites nothing relevant. You would go looking at the retriever.

The fix:

```python
        if used + len(block) > max_chars:
            # Never return an empty context: if the first block alone exceeds the budget,
            # truncate it instead of dropping it. An empty context would send the model to
            # answer from parametric memory with nothing to cite, and the resulting numbers
            # would look like a retrieval failure rather than a prompt-construction bug.
            if not parts:
                parts.append(block[:max_chars])
            break
```

At the production budget (6000 characters, chunks capped at 1200) this cannot trigger. It was found by
a test written to check the budget was respected at all, using `max_chars=60`. A test of an ordinary
property found a bug in an extraordinary case — which is the usual way it happens.

## Deep Explanation: citation extraction

```python
# Matches "PEP 634", "PEP-634", "pep634". Four digits max: PEP numbers top out in the
# thousands, and a looser pattern would capture years and version strings.
_CITATION_RE = re.compile(r"\bPEP[\s\-]?(\d{1,4})\b", re.IGNORECASE)
```

Three deliberate constraints. `\b` word boundaries prevent matching inside a longer token. `{1,4}`
bounds the number — without it, a year or a long numeric string could be captured. Case-insensitive,
because models produce `pep` and `PEP` interchangeably.

Two extraction functions, and the split is what makes one of the metrics possible:

```python
def extract_citations(answer: str, known: set[int] | None = None) -> list[int]:
    """PEP numbers cited in an answer, in order of first appearance, deduplicated.

    When `known` is given, numbers outside it are dropped: the model occasionally invents a
    plausible-looking PEP number, and counting those as citations would corrupt the metrics.
    Whether it invents them at all is measured separately as the hallucination rate.
    """


def extract_all_cited_numbers(answer: str) -> list[int]:
    """Every PEP-shaped number in an answer, including ones absent from the corpus."""
```

The filtered version feeds the authority metrics — a citation to a nonexistent PEP has no status and
would corrupt the rate. The unfiltered version feeds the hallucination metric. Same regex, two
consumers, and separating them is what lets "cited a dead PEP" and "cited a PEP that does not exist" be
different measurements.

## Deep Explanation: the four metrics

All computed from corpus metadata. No judge, no human grading (chapter 12 explained why).

```python
def score_answer(...) -> AnswerRecord:
    cited_superseded = [
        number
        for number in citations
        if number in peps and peps[number].status in NON_AUTHORITATIVE
    ]
    hallucinated = [number for number in all_cited if number not in peps]
    ...
    return AnswerRecord(
        ...
        cited_authoritative=bool(set(citations) & relevant),
        version_correct=version_correct,
        ...
    )
```

| Metric | Computation | Direction |
|---|---|---|
| **superseded-citation** | any cited PEP whose status is in `NON_AUTHORITATIVE` | lower better |
| **authoritative-citation** | any cited PEP in the gold labels | higher better |
| **version-correct** | on version-scoped queries only, does the answer mention the release the feature actually landed in | higher better |
| **hallucinated-citation** | any cited number absent from the corpus | lower better |

### The version metric, and how narrow it is

```python
    version_correct: bool | None = None
    if asked_version:
        # The release in which the feature actually landed, taken from the labelled PEPs.
        expected = next(
            (
                version
                for number in sorted(relevant)
                if number in peps and (version := release_version(peps[number].python_version))
            ),
            None,
        )
        if expected is not None:
            version_correct = mentions_version(answer, expected)
```

Three-state, not boolean. `None` means *not scored* — either the query names no version, or no labelled
PEP has a parseable release (chapter 10's `3.x` case). Chapter 16's `None`-versus-`0.0` discipline again:
a query that could not be scored must not be averaged in as a failure.

And the claim is narrow on purpose. From the module docstring:

```python
"""- **version-correct rate** -- on version-scoped queries only, did the answer surface the
  release in which the feature actually landed? Deliberately narrow: it checks that the right
  version *appears*, not that the surrounding claim is true. Full version correctness needs
  the deferred judge, and the docs say so rather than implying otherwise.
"""
```

Asked *"can I use the walrus operator in Python 3.7?"*, the correct answer must surface **3.8**. The
metric checks that `3.8` appears. It does not check that the sentence around it says "no" rather than
"yes". That is a real limitation, stated where the metric is defined rather than discovered later.

### `mentions_version` is exact

```python
def mentions_version(answer: str, version: str) -> bool:
    return version in set(_RELEASE_RE.findall(answer))
```

Extracting all `major.minor` tokens and testing set membership, rather than a substring search. A
substring search for `"3.1"` would match `"3.10"` and `"3.11"`. Pinned:

```python
def test_mentions_version_is_exact(self) -> None:
    self.assertTrue(mentions_version("added in 3.10", "3.10"))
    self.assertFalse(mentions_version("added in 3.1", "3.10"))
    self.assertFalse(mentions_version("added in 3.100", "3.10"))
```

## Deep Explanation: two places the metric marks a correct answer wrong

These were found by reading the failures rather than the aggregates, and both are recorded as
limitations rather than fixed.

### A correct conclusion from a dead source

**q47** — *"is zoneinfo available in Python 3.8"* → measured answer:

> *"No, zoneinfo is not available in Python 3.8. PEP 431 states…"*

The conclusion is right. The citation is `Superseded` (PEP 431, replaced by 615). The metric records a
superseded citation.

Is that wrong? No — but it means the metric measures **citation hygiene, not answer truth**. Those are
different properties and only one of them is being reported. The README was corrected during the
writing of this book to say so explicitly, because a metric called "superseded-citation rate" is easy to
read as "wrongness rate".

### A correct hedge, penalised

**q30** — the deliberate no-authoritative-answer case:

> *"There is no definitive answer based on the provided excerpts. PEP 543 and PEP 748 both propose a
> unified TLS API…"*

This is precisely the behaviour the prompt asked for and precisely what a careful system should do. It
still scores as a superseded citation, because PEP 543 is `Withdrawn`.

The metric cannot distinguish *citing a dead document as authority* from *citing a dead document while
explaining that nothing is settled*. A regex cannot; a validated judge could.

### What follows

Both cases argue for the same thing: the LLM-as-judge that this project deliberately did not build. It
is listed first in the README's future improvements, and the argument for it is now concrete — two
identified cases where the automatic metric is wrong in a specific, characterised way — rather than
"a judge would be more thorough."

That is the useful form of a limitation: **not "this could be better" but "here are the two cases where
it is wrong, and here is the component that would fix them."**

## Measured results

Live, 51 queries, three rungs, 153 answers:

```
Retriever     superseded authoritative     version    hallucin.
--------------------------------------------------------------
BM25               0.275         0.686       0.429        0.000
Dense              0.235         0.686       0.429        0.000
Hybrid             0.235         0.765       0.571        0.000
```

**Zero hallucinated citations across all 153 answers.** Every citation resolved to a real PEP. Grounding
is not the failure mode in this system; authority is. The model does not invent sources — it faithfully
cites dead ones.

## Systems Perspective

Generation dominates everything: **~8 seconds per answer**, so 51 queries is about 7 minutes per
configuration and the 153-answer run took roughly 20 minutes. Chapter 8's answer cache is what makes
chapter 20's five-configuration sweep affordable.

The cache also makes a warm re-run report end-to-end p95 of **1 ms**, which is a disk read. Any latency
figure from this layer must state whether the cache was cold.

## Common Mistakes

**Letting the generator see metadata you are trying to measure the retriever on.** Confounds the entire
comparison.

**Not specifying a citation format.** Extraction then depends on whatever the model felt like.

**One extraction function.** You lose the ability to distinguish a dead citation from an invented one.

**Substring matching version numbers.** `"3.1"` matches `"3.10"`.

**Boolean where three states are needed.** "Not scored" averaged in as "failed".

**A prompt with no exit.** Without "say so if the excerpts do not settle it", the model fabricates.

**Describing a citation metric as a correctness metric.** They differ, and the difference is visible in
real answers.

## Interview Insight

> **"How did you evaluate the generated answers without human grading?"**

The corpus carries the answer. An answer cites PEP 386; PEP 386's header says `Superseded`; that is a
superseded citation, determined by a dictionary lookup. Four metrics on that basis — superseded
citation, authoritative citation, version correctness on the version-scoped subset, and hallucinated
citations as a grounding check.

The honest cost is that it measures citation hygiene, not answer truth, and I can name the two cases
where they diverge: one answer drew the *right* conclusion from a superseded source, and one correctly
hedged that nothing was settled while citing a withdrawn document. Both score as failures. That is the
concrete argument for adding a validated judge, and it is why I state what the metric measures rather
than letting the name imply more.

> **"Why doesn't the prompt include each document's status?"**

Because it would confound the experiment. The study compares four retrieval strategies; if the generator
could see that a chunk came from a superseded document, the measured citation rate would reflect the
model's caution rather than the retriever's ranking, and every rung would improve by an amount depending
on how compliant the model happened to be.

So the generator is held constant and uninformed, and there is a test asserting that status never
reaches the prompt. Making it authority-aware is a genuinely interesting separate experiment — I list it
as future work rather than folding it in.

> **"Did the model hallucinate?"**

Not once in 153 answers — every citation resolved to a real PEP. That was worth measuring precisely
because it *localises the problem*: grounding is fine, authority is not. The model does not invent
sources, it faithfully cites dead ones, which is a harder failure to notice and the one the project is
about.

## Debugging Tip

When an answer metric is surprising, print the answer next to its citations and their statuses. The
driver does this for the rung under investigation:

```
q23 [rationale/trap] -> PEP 386 [Superseded]
     According to PEP 386, Python package version numbers are compared using the standard...
```

One line of answer text plus the resolved status turns an aggregate rate into a readable failure. This
is how both metric limitations above were found — not by looking at 0.235, but by reading the twelve
answers behind it.

## Summary

- The prompt bounds length, specifies a citation format, and offers an explicit exit from fabrication.
- It deliberately withholds `Status` and `Python-Version` so the generator cannot compensate for bad
  retrieval, which would confound the comparison. A test enforces this.
- `build_context` had a bug returning an empty context when the first chunk exceeded the budget — found
  by an ordinary budget test, and its failure mode was to look like a retrieval failure.
- Two extraction functions from one regex: filtered for authority metrics, unfiltered for hallucination.
- Four metrics from corpus metadata; version correctness is three-state and deliberately narrow.
- Two identified cases where a correct answer scores as a failure — a right conclusion from a dead
  source, and a correct hedge — which is the concrete argument for the deferred judge.
- Zero hallucinated citations in 153 answers: grounding is not the failure mode, authority is.

## Key Takeaways

1. Withhold from the generator anything you are trying to attribute to the retriever.
2. Specify the output format you intend to parse.
3. State what a metric measures, not what its name suggests.
4. A useful limitation names the failing cases and the component that would fix them.

## Why the Next Chapter Exists

Everything is now in place: four metrics on generated answers, and a measured baseline where roughly one
answer in four cites a dead specification.

Chapter 20 reads the intervention — the authority reranker — including the two signals it combines, the
strength parameter that makes the result a curve rather than two points, and the version rule that was
implemented specifically so it could be shown to be harmful.
