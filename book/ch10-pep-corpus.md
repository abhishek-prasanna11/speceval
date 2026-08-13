# Chapter 10 — The PEP Corpus: Headers, Statuses, Supersession

## Learning Objectives

- Explain what a PEP is and how the series is organised.
- Read **RFC-822 header** syntax, including continuation lines, and explain why parsing it naively
  fails.
- Explain the nine-value `Status` enumeration and which values mean "do not answer from this".
- Explain the **supersession graph** and why it is a signal distinct from `Status`.
- Describe the four structural cases in this corpus that a status-only rule cannot handle.
- Explain why this corpus was chosen over the alternatives.

## Motivation

The finding at the centre of this project is only measurable in a corpus whose documents **record
their own standing**. That is rare. Most corpora — web pages, wiki articles, support tickets, product
documentation — have no machine-readable notion of "this document has been replaced by that one."
Without it, you can observe that a retrieval system returned something unhelpful, but you cannot
*automatically* say it returned something obsolete, and therefore you cannot measure the rate at which
it happens.

The Python Enhancement Proposal series does record it. This chapter is about exactly what it records,
and about the four structural situations in it that make the problem interesting rather than trivial.

## First Principles

### What a PEP is

A **Python Enhancement Proposal** is a design document proposing a change to Python — a language
feature, a standard-library addition, or a process. The series began in 2000 and now numbers in the
hundreds. Some became the language; some were argued over and refused; some were replaced by better
versions of themselves.

The corpus is a git repository (`python/peps`), one reStructuredText file per proposal:

```
peps/peps/pep-0008.rst      Style Guide for Python Code
peps/peps/pep-0498.rst      Literal String Interpolation      (f-strings)
peps/peps/pep-0634.rst      Structural Pattern Matching: Specification
```

Fetched by `scripts/fetch_corpus.sh`, which shallow-clones because history is not part of the study:

```bash
# Fetch the PEP corpus. Shallow clone -- history is not part of the study.
# The corpus is gitignored: it is input data, not source.
```

Measured, live: **734 PEP files**, becoming **19,763 chunks** at a mean of 26.9 chunks each.

### The header block

Each file opens with a block of `Name: value` lines — the **RFC-822** format, the same shape as email
headers. Real example, PEP 634:

```
PEP: 634
Title: Structural Pattern Matching: Specification
Author: Brandt Bucher <brandt@python.org>,
        Guido van Rossum <guido@python.org>
BDFL-Delegate:
Discussions-To: python-dev@python.org
Status: Final
Type: Standards Track
Created: 12-Sep-2020
Python-Version: 3.10
Post-History: 22-Oct-2020, 08-Feb-2021
Replaces: 622
Resolution: https://mail.python.org/archives/...

Abstract
========

This PEP provides the technical specification for the match ...
```

The fields this project uses:

| Field | Meaning | Present in |
|---|---|---|
| `PEP` | The number | all |
| `Title` | Human title | all |
| `Status` | Current standing — the authority signal | all |
| `Type` | Standards Track / Informational / Process | all |
| `Python-Version` | Release the change landed in | 519 (71%) |
| `Superseded-By` | The PEP that replaced this one | 31 |
| `Replaces` | The PEP this one replaced | 36 |

### Why parsing it naively fails

The format has three features that break the obvious implementation, and all three occur in this
corpus.

**Continuation lines.** The `Author` field above spans two lines, the second indented. A parser
splitting on newlines and then on the first colon would read `guido@python.org>` as a header named
`Guido van Rossum <guido`.

**Values containing colons.** `Title: Structural Pattern Matching: Specification` has two colons.
Splitting on *every* colon truncates the title to "Structural Pattern Matching" — which, as chapter 11
shows, happens to be the exact title of the superseded predecessor. A parsing bug here would silently
merge the two documents this project is chiefly about.

**Empty values.** `BDFL-Delegate:` has no value. A parser requiring a non-empty value would either
crash or, worse, treat the next line as this field's value.

Plus one real-world wrinkle: some files have irregular whitespace after the colon. Grepping the corpus
finds `Status:            Final` with ten spaces alongside the usual single space.

`speceval/corpus.py` handles all of it:

```python
# "Header-Name: value", with continuation lines indented (the Author field wraps).
_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z-]*):[ \t]*(.*)$")
```

The regex anchors the name to the line start, permits only letters and hyphens in it, and captures
*everything* after the first colon-plus-whitespace as the value. Colons inside the value are safe
because the capture is greedy to end of line.

The block-level loop handles continuations and termination:

```python
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
```

Three exits: a blank line ends the block (the RFC-822 rule), an indented line appends to the previous
field, and anything else unparseable ends the block defensively rather than raising.

Each behaviour is pinned by a test built from a sample that reproduces all three wrinkles at once:

```python
def test_title_keeps_internal_colon(self) -> None:
    # Splitting on every colon would truncate this to "Structural Pattern Matching".
    self.assertEqual(self.pep.title, "Structural Pattern Matching: Specification")

def test_extra_whitespace_after_colon_is_stripped(self) -> None:
    self.assertEqual(self.pep.status, "Final")

def test_wrapped_author_is_consumed_as_a_continuation(self) -> None:
    # The indented line must be folded into the Author header, not leak into the
    # body and not be misread as a header of its own.
    self.assertNotIn("Guido van Rossum", self.pep.body)
    self.assertNotIn("brandt@python.org", self.pep.body)
```

## Deep Explanation: the status enumeration

`Status` is the primary authority signal. Measured across the corpus, live:

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

Nine values (one of them a joke — PEP 401 and friends). Their meanings:

| Status | Meaning | Standing |
|---|---|---|
| `Final` | Accepted and implemented | authoritative |
| `Active` | Ongoing process/informational PEP, in force | authoritative |
| `Accepted` | Approved, implementation pending | authoritative |
| `Provisional` | Accepted with a trial period | authoritative-ish |
| `Draft` | Under discussion | not yet anything |
| `Deferred` | Dormant — no decision, not refused | **not in force** |
| `Superseded` | Replaced by a later PEP | **not in force** |
| `Rejected` | Considered and refused | **not in force** |
| `Withdrawn` | Retracted by its author | **not in force** |

The four marked as not in force are what this project calls **non-authoritative**, defined in one place:

```python
# A PEP in one of these states must not be answered from as though it were current.
# This is the definition the superseded-citation-rate metric is built on, so it lives
# in one place rather than being re-spelled at each call site.
NON_AUTHORITATIVE = frozenset({"Rejected", "Withdrawn", "Superseded", "Deferred"})
```

**262 of 734 PEPs (36%) are non-authoritative.** That is the *trap surface* — over a third of the
corpus consists of documents that read like specifications, are indexed like specifications, and are
not specifications any more.

That 36% figure recurs throughout Part V, and chapter 13 shows it doing something rather elegant: a
retriever returning random documents lands at a trap rate of 0.333, almost exactly the base rate, which
is strong evidence the metric measures what it claims.

### Why `Status` alone is not enough

If authority were just `Status`, the intervention in chapter 20 would be a one-line filter and the
result would be foregone. Four structural cases in this corpus rule that out.

### Case 1 — the supersession edge exists independently of status

`Superseded-By` is a *separate* field from `Status`. They can disagree.

Enumerated from the corpus (all 31 edges were dumped during development; these are real):

```
   333  Python Web Server Gateway Interface v1.0   [Final] -> 3333  WSGI v1.0.1        [Final]
   409  Suppressing exception context              [Final] ->  415  Implement context…  [Final]
   248  Python Database API Specification v1.0     [Final] ->  249  DB API v2.0         [Final]
```

PEP 333 is `Final` **and** carries `Superseded-By: 3333`. Both members of the pair read `Final`. A
status-only rule cannot separate them — and answering a question about WSGI from PEP 333 rather than
3333 is exactly the error this project measures.

So authority requires reading the *graph*, not just the flag. Chapter 20 does precisely that, weighting
by status and then applying a separate penalty for having a live successor.

### Case 2 — multi-hop chains

Supersession is transitive, and it goes several steps:

```
   241  Metadata for Python Software Packages      [Superseded]
     |
     v
   314  Metadata … 1.1                             [Superseded]
     |
     v
   345  Metadata … 1.2                             [Superseded]
     |
     v
   566  Metadata … 2.1                             [Final]      <- the answer
```

Three superseded hops before reaching the live document. A single-step check that only asks "is this
PEP superseded?" happens to work here because every intermediate is *also* marked `Superseded` — but
the case demonstrates that the relation is a graph with depth, not a pairwise flag. This is gold query
`q21`.

### Case 3 — convergence

Three separate superseded PEPs point at one successor:

```
   513  manylinux1  [Superseded] ─┐
   571  manylinux2010 [Superseded]─┼──> 600  Future manylinux tags  [Final]
   599  manylinux2014 [Superseded]─┘
```

A query about platform tags can retrieve any of three dead documents. Gold query `q22`.

### Case 4 — no authoritative answer exists

```
   543  A Unified TLS API for Python  [Withdrawn] ──> 748  A Unified TLS API  [Draft]
```

The old one is withdrawn; its replacement is only a `Draft`. **Neither is authoritative.** The correct
behaviour for a system asked about this is to decline to present either as settled.

This is gold query `q30`, included deliberately, and it has a consequence worth noting now: it makes
the trap metric's floor non-zero. Even a perfect retriever registers one trap, because the ground truth
itself includes a withdrawn document. Chapter 13 returns to this.

### The version dimension

`Python-Version` records the release a change landed in. 519 PEPs carry it. Header formats are not
uniform:

```
   Python-Version: 3.10
   Python-Version: 3.11, 3.12
   Python-Version: 3.x
```

The last is unparseable as a concrete release. `speceval` handles this by returning `None` rather than
inventing a value:

```python
def release_version(python_version: str | None) -> str | None:
    """First concrete `major.minor` in a Python-Version header.

    Headers are not uniform: "3.10", "3.11, 3.12" and "3.x" all occur. "3.x" yields None,
    which excludes that PEP from version scoring rather than inventing a release for it.
    """
```

Excluding a case you cannot score is more honest than guessing at it — a guessed release would enter
the metric as if it were data.

Why the version dimension matters: **it makes correctness conditional.** PEP 634 is `Final`, so a
status-only view says "trustworthy". But structural pattern matching does not exist in Python 3.9. Ask
"can I use it in 3.9?" and the correct answer is *no*, and producing that answer requires the very
document whose version postdates the question. Chapter 20 shows that the obvious rule here is
**harmful**, and proves it by measurement.

## Mental Model

Think of **case law**. A judgment is a document with a citation, a date, and a standing: still good
law, overruled, distinguished, superseded by statute. A lawyer citing an overruled precedent as
current has made a serious error, even though the document is genuine, on-topic, and well argued.

Legal research tools exist largely to track that standing — the "is this still good law?" question is
separate from and harder than "is this about my topic?".

The PEP corpus is a small, clean, machine-readable version of the same structure. And a retrieval
system with no authority signal is a researcher who reads only for topic.

## Deep Explanation: why this corpus

Several corpora with supersession structure were considered. The requirements were: machine-readable
supersession, small enough to be tractable, technical enough that labels could be verified, and prose
substantial enough that chunking and retrieval are non-trivial.

| Corpus | Supersession signal | Why not chosen |
|---|---|---|
| IETF RFCs | `obsoletes` / `obsoleted-by` in `rfc-index.xml` | Strong candidate; larger and no cleaner |
| Kubernetes KEPs | `status` × `stage` × `milestone` in `kep.yaml` | Template boilerplate repeats near-verbatim across hundreds of documents, which would confound a 51-query study |
| TC39 proposals | Stage 0–4 in a README table | Documents too thin — mostly links, little prose |
| W3C specs | REC / superseded status | Messier metadata; one enormous HTML document |
| CVE/NVD advisories | Affected version ranges | Advisory text is short and formulaic, so chunking becomes trivial and it drifts toward a structured query problem |

PEPs won on a combination: `Status` is a nine-value graded enum rather than a boolean, `Python-Version`
adds a second orthogonal dimension, `Superseded-By`/`Replaces` give real graph edges, ingestion is one
`git clone`, and the prose is written to be read.

There was one genuine reversal during selection worth recording, because it shows the reasoning
working. The initial recommendation was KEPs, on the argument that PEP authority was a *flat* status
flag and would make the reranker a trivial filter with a foregone result. That was wrong: PEPs also
carry `Python-Version` and supersession edges, so authority is compound here too. Once corrected, PEPs
won on text quality and ingestion simplicity.

## Systems Perspective

Parsing 734 files is trivial — one `read_text` and one regex pass each, well under a second. Chunking
them into 19,763 pieces is likewise fast. The corpus layer is not a performance concern; it is a
*correctness* concern, because everything downstream inherits its parse.

One consequence of the corpus being external: the numbers in this book are tied to a snapshot. `python/peps`
gains PEPs and changes statuses over time — a `Draft` becomes `Final`, a `Final` gains a
`Superseded-By`. That is why `scripts/verify_gold.py` exists and why it is wired to fail loudly:
chapter 21 covers it, but the principle is that a gold set labelled against a moving corpus needs a
tripwire.

## Common Mistakes

**Splitting headers on every colon.** Truncates titles, and in this corpus truncates the one title that
matters most.

**Ignoring continuation lines.** Produces phantom header fields and leaks author names into the body.

**Treating `Status` as boolean.** Loses the distinction between `Deferred` (dormant) and `Rejected`
(refused), and misses the both-`Final` supersession pairs entirely.

**Ignoring `Superseded-By` because `Status` looks healthy.** The 333 → 3333 case is invisible otherwise.

**Assuming supersession is one hop.** The packaging chain is four documents deep.

**Coercing `3.x` into a release number.** Puts a guess into a metric as though it were data.

**Hard-coding the non-authoritative set at multiple call sites.** It is the definition the headline
metric rests on; it belongs in one place.

## Interview Insight

> **"Why did you choose this corpus?"**

Because the study needs documents that record their own standing, and most corpora do not. PEPs carry a
nine-value status enum, a `Python-Version` field that makes correctness version-conditional, and
explicit `Superseded-By`/`Replaces` edges — so the hard test cases can be *generated from the metadata*
rather than hand-invented, which is what made a 51-query gold set affordable.

Then the detail that shows you interrogated the choice: *and the structure turned out richer than
assumed. There are pairs where both documents are `Final` and only the supersession edge separates
them, a four-hop chain in the packaging metadata series, three documents converging on one successor,
and one case where the old document is withdrawn and its replacement is only a draft — so no
authoritative answer exists at all. A status-only rule handles none of those.*

> **"Why not just filter out non-current documents?"**

Three reasons. Status alone does not identify them — PEP 333 is `Final` and superseded. Filtering
discards documents you may need, since a rejected proposal is often the best explanation of *why*
something is not in the language. And a filter cannot express degree, whereas 36% of this corpus is
non-authoritative and those documents are not equally untrustworthy — `Deferred` is dormant, `Rejected`
is refused.

## Debugging Tip

When a header-parsing change is made, re-run the corpus statistics and compare the status histogram
against the previous run. A parsing regression shows up immediately as a shifted distribution — a
sudden drop in `Final` or an unexpected empty status — long before it shows up as a strange retrieval
result. `run_phase1.py` prints that histogram every run for exactly this reason.

## Summary

- 734 PEPs, 19,763 chunks, each file opening with an RFC-822 header block.
- Naive header parsing fails three ways in this corpus: continuation lines, colons inside values, and
  empty values — plus irregular whitespace. All are handled and tested.
- `Status` has nine values; four (`Rejected`, `Withdrawn`, `Superseded`, `Deferred`) mean not in force.
  **262 of 734, or 36%, are non-authoritative.**
- The supersession graph is a *separate* signal from `Status`: there are both-`Final` pairs, a four-hop
  chain, a three-into-one convergence, and one case where no authoritative answer exists.
- `Python-Version` adds an orthogonal dimension making correctness version-conditional; unparseable
  values yield `None` rather than a guess.
- The corpus was chosen for compound machine-readable authority, tractable size, verifiable labels, and
  readable prose.

## Key Takeaways

1. Authority in this corpus is compound — status, supersession edges, and version — not a flag.
2. Parse RFC-822 properly: continuations, internal colons, empty values.
3. 36% non-authoritative is the trap surface, and it reappears as a validation check in chapter 13.
4. Exclude what you cannot score rather than guessing it.

## Why the Next Chapter Exists

We now have a corpus that records which of its documents are still in force, and three retrieval
strategies that have no idea such a thing exists.

Chapter 11 puts those together and states the project's thesis precisely: relevance and authority are
different quantities, optimising the first does not improve the second, and — the part that is genuinely
counterintuitive — making a retriever *better* at relevance can make it *worse* at authority.
