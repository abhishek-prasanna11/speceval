# Chapter 21 — The Gold Set and the Drivers

`eval/queries_gold.json` · `scripts/verify_gold.py` (138 lines) · `run_phase1–4.py` (647 lines)

## Learning Objectives

- Explain each check in the gold-set verifier and the failure it prevents.
- Explain why the verifier has a documented escape hatch and what it is for.
- Explain what each of the four drivers produces and why they are separate programs.
- Explain the baseline mistake in the sweep and the measurement that revealed it.
- Explain the output-design choices that make a results table self-validating.

## Motivation

This chapter covers the code with no algorithms in it: the file of hand-written labels, the script that
checks them, and the four programs that turn everything into tables.

It is also where the study's most instructive near-miss lives. The sweep in chapter 20 has a baseline,
and choosing the *obvious* baseline would have attributed a pool-depth effect to reranking on 50 of 51
queries.

## Deep Explanation: the verifier

The gold set is labelled against a corpus that is not vendored — `peps/` is gitignored and cloned fresh.
So the corpus moves: a `Draft` becomes `Final`, a `Final` gains a `Superseded-By`, a PEP is added. The
labels do not move with it, and a label that silently became wrong would corrupt every number downstream
with nothing to indicate it.

`scripts/verify_gold.py` is the tripwire.

```python
"""Verify every label in the gold set against the corpus itself.

Label quality is the ceiling on this entire study: a brilliant harness over labels nobody
checked is worth nothing, and label rot is invisible once the corpus updates. So every
claim the gold set makes is re-derived from the PEP headers here rather than trusted.

Exits non-zero on any inconsistency, so it can gate a commit.
"""
```

### The checks, and what each prevents

| Check | Prevents |
|---|---|
| Duplicate `qid` | Two entries silently overwriting each other in a keyed report |
| Duplicate query text | The same query counted twice, inflating whatever it happens to score |
| Category in the known set | A typo creating a phantom fourth category with n=1 |
| Non-empty `relevant` | A query the metrics would raise on (chapter 5) |
| Every label exists in the corpus | A label pointing at a PEP that was renumbered or removed |
| A non-authoritative label requires `trap: true` | Accidentally treating a dead PEP as a normal answer |
| A `trap` query has a real trap available | A query marked trap with nothing to trap it |
| `asked_version` appears in the query text | The field drifting back to its old ambiguous meaning |
| A version-scoped query has a parseable release | A version metric that cannot be computed |
| Query text names a version but `asked_version` unset | Silent exclusion from the version metric |

The last two are the pair discussed in chapter 12 — the bidirectional check that pins a field to one
meaning.

### The trap check

```python
        if record.get("trap"):
            labelled = set(relevant)
            has_trap = any(
                pep.status in NON_AUTHORITATIVE
                and (
                    (pep.superseded_by in labelled)
                    or bool(labelled & set(pep.replaces))
                    or pep.number in labelled
                )
                for pep in peps.values()
            )
```

It scans the whole corpus for a non-authoritative PEP related to this query's answer in any of three
ways: it is superseded *by* the answer, it *replaces* the answer, or it *is* one of the labels (the
q06 case, where a `Rejected` proposal is legitimately relevant).

So `trap: true` is not a human assertion the verifier accepts — it is a claim re-derived from the
corpus graph. A trap query whose predecessor was later renumbered would fail here.

### The documented escape hatch

```python
            # q27 / q28 are the both-Final pairs: no non-authoritative PEP is involved, so
            # the trap is recency rather than status. Those declare it in the note.
            declares_hard_case = "both" in record.get("note", "").lower()
            if not has_trap and not declares_hard_case:
                problems.append(...)
```

Chapter 10's case 1 — PEP 333 and 3333 are both `Final` — has no non-authoritative document involved.
The trap is recency, expressed only through the supersession edge. The structural check cannot see it,
so those entries are exempted by declaring the case in their `note`:

> *"Hard case: PEP 333 and 3333 are BOTH Final. Authority here is recency, not status -- a status-only
> check cannot separate them."*

Escape hatches in validators are usually a smell. This one is defensible for two reasons: it requires an
explicit written declaration rather than a silent flag, and the declaration is prose a human must read
and agree with. It is worth being honest that a keyword match on `"both"` is a crude mechanism — a
dedicated field would be cleaner — but the property that matters is that skipping the check requires
saying why, in the file, where a reviewer sees it.

### Composition reporting

```
queries        51
categories     availability=21, identifier=15, rationale=15
trap cases     20 (39%)
ordinary       31
PEPs referenced 48

OK -- every label verified against the corpus
```

Printed every run. **48 distinct PEPs across 51 queries** is the line most worth watching: a gold set
concentrated on a handful of documents would be unrepresentative in a way no individual label check
would catch.

## Deep Explanation: four drivers, not one

```
   run_phase1.py   corpus statistics + BM25 baseline           (116 lines)
   run_phase2.py   the ladder, retrieval level                 (150 lines)
   run_phase3.py   the ladder, answer level                    (168 lines)
   run_phase4.py   rung 4 swept + ablation + per-query diff     (213 lines)
```

Separate programs rather than one with subcommands, and the reason is cost. `run_phase1` and
`run_phase2` are seconds. `run_phase3` is about twenty minutes cold. `run_phase4` is longer. Keeping
them separate means the cheap ones can be run freely while iterating, and it makes each one's output a
self-contained artefact.

### Progressive disclosure in the output

Each driver prints in the same order: **corpus/composition → controls → headline table → subsets →
per-query detail.** A reader can stop at any level.

`run_phase2` is representative:

```
THE LADDER  (k=10, 51 gold queries)
Retriever     Recall@10    nDCG@10   trap@1    p50 ms    p95 ms
---------------------------------------------------------------
Oracle            1.000      1.000    0.020      0.00      0.00
Random            0.000      0.000    0.333      0.02      0.03
BM25              0.863      0.671    0.294     27.88     42.55
Dense             0.971      0.801    0.255     36.78     45.22
Hybrid            0.931      0.765    0.294     28.29     39.08
```

The controls are rows one and two, and the driver says what they are for:

```
Oracle 1.000/1.000 and Random ~0.000 are the harness validating itself.
```

Chapter 13 argued for this: **the harness re-validates itself in front of you on every run**, so a
metric regression is visible before you read the rows you care about.

Then the subsets, with the control named:

```
Retriever                  trap (n=20)           ordinary (n=31)
----------------------------------------------------------------
BM25              R 0.82 N 0.62 T 0.50      R 0.89 N 0.71 T 0.16
Dense             R 0.93 N 0.67 T 0.55      R 1.00 N 0.89 T 0.06

R = Recall@10, N = nDCG@10, T = trap@1. The ordinary subset is the control:
a reranker that only improves the trap column is buying its gain somewhere.
```

Then per-query detail, with status annotations:

```
qid  cat                   BM25        Dense       Hybrid  want
q16  availability     649[Fina]*   563[Supe]    563[Supe]   [649]
```

Four characters of status in brackets is what turns three plausible PEP numbers into a visible failure.
Chapter 11's mechanism was found by reading this table, not by reading an aggregate.

### The smoke-test flag

```python
    parser.add_argument(
        "--limit", type=int, default=None, help="smoke-test on the first N PEPs only"
    )
    ...
    if args.limit:
        peps = peps[: args.limit]
        print(f"!! --limit {args.limit}: results are NOT comparable to the full corpus\n")
```

Chapter 18 explained why this exists instead of resume logic: it lets the whole pipeline be exercised on
30 PEPs in under a minute before committing to a seventeen-minute embedding run.

The warning line is not decoration. A limited run produces a complete, plausible results table with
wrong numbers. Printing the caveat *in the output* means a screenshot or a pasted table carries its own
disclaimer — the caveat cannot be separated from the numbers.

## Deep Explanation: the baseline mistake

This is the most instructive thing in the chapter.

Chapter 20's reranker wraps `HybridRetriever` and draws a pool **ten times** the requested depth before
reranking. Rung 3, measured in chapter 19, drew a pool equal to the requested depth.

The obvious baseline for the sweep is therefore Phase 3's hybrid row: same retriever, no reranking, a
number already computed. Using it would be wrong, and quantifiably so.

`run_phase4.py` says why in its docstring:

```python
"""The baseline is `strength = 0` **of this same pipeline**, not Phase 3's hybrid row. The
reranker draws a deeper candidate pool than rung 3 did, and RRF over a deeper pool can promote
different chunks, so only the in-pipeline zero point is a clean control. Phase 3's number is
printed alongside for reference, with the difference attributable to pool depth.
"""
```

The reasoning: RRF fuses whatever is in the pool. Fusing over 50 chunks per system produces a different
top-5 than fusing over 5, because documents ranked mid-list by both systems can outscore documents
ranked first by one (chapter 6's agreement property). Deeper pool, different fusion, different chunks.

**The measurement that settled it** came from the answer cache. Because prompts are keyed on their
content (chapter 8), an identical prompt is a cache hit. When λ=0 ran:

```
  lambda=0         (1 cached, 50 generated)
```

**One cache hit out of 51.** Fifty of 51 queries produced a *different prompt* than Phase 3's hybrid did
— different chunks in the context — purely from pool depth, with reranking doing nothing at λ=0.

Had Phase 3's hybrid row been used as the baseline, the reported improvement would have conflated two
effects on 50 of 51 queries. And the direction matters: the pool-depth effect alone improved the
superseded-citation rate from 0.235 to 0.157, so roughly **half** the apparent gain would have been
misattributed to reranking.

The driver prints both, labelled:

```
Phase 3 reference (rung 3, shallower pool): superseded 0.235, authoritative 0.765
The lambda=0 row above is the clean in-pipeline control; any gap from the Phase 3
row is pool depth, not reranking.
```

Two general lessons.

**Your baseline must differ from your treatment in exactly one respect.** "The same retriever without
the new thing" is not a baseline if enabling the new thing also changed something else — here, the pool
depth.

**Instrumentation intended for one purpose often answers a different question.** The cache-hit counter
exists to report cost. It happened to be a direct measurement of prompt overlap between two
configurations, which is exactly what was needed to size the confound.

## Deep Explanation: the per-query diff

`run_phase4` ends with something no aggregate provides:

```python
    fixed, broken = [], []
    for qid, before in base_records.items():
        after = full_records[qid]
        if before.cited_superseded and not after.cited_superseded:
            fixed.append((qid, before, after))
        elif not before.cited_superseded and after.cited_superseded:
            broken.append((qid, before, after))
```

```
FIXED (6) -- cited a dead PEP at lambda=0, clean at lambda=1
  q06/trap  dropped PEP 3103        q26/trap  dropped PEP 722
  q16/trap  dropped PEP 563         q31       dropped PEP 346
  q20/trap  dropped PEP 554         q47/trap  dropped PEP 431

BROKEN (0) -- clean at lambda=0, cited a dead PEP at lambda=1
```

A rate of 0.157 → 0.039 is consistent with many underlying patterns: eight fixed and two broken, six
fixed and none broken, or a churn of twenty changes netting six. Those have very different
implications, and the aggregate cannot distinguish them.

**Six fixed, zero broken** is the strongest available form of the claim, and — importantly — it does not
depend on rates at all. At n=51 a rate difference of 0.118 is six answers; stated that way it invites a
significance argument. Stated as "six improved, none regressed" it is a statement about individual
queries that survives the small-n objection.

## Systems Perspective

Runtime, measured:

| Driver | Cold | Warm |
|---|---|---|
| `run_phase1` | ~3 s | ~3 s |
| `run_phase2` | ~17 min (embedding) | ~5 s |
| `run_phase3` | ~20 min (153 generations) | ~5 s |
| `run_phase4` | ~30 min (204 more generations) | ~10 s |

The warm column is why the caches exist. Every number in Part V was reproduced from a warm cache while
writing this book, in seconds rather than an hour — with the caveat from chapter 8 that latency figures
must come from the cold run.

## Common Mistakes

**No verifier for a gold set labelled against an external corpus.** Label rot is silent.

**A validator with undocumented exemptions.** If a check can be skipped, skipping it should require
writing down why, in the file.

**One driver for everything.** Couples a three-second run to a thirty-minute one.

**Caveats outside the output.** A limited run's table looks exactly like a full one; print the warning
where it cannot be separated from the numbers.

**Comparing against the nearest available number.** A baseline must differ from the treatment in one
respect only.

**Reporting only rates for a small-n result.** "Six fixed, zero broken" is a stronger and more honest
claim than a rate delta.

## Interview Insight

> **"How do you keep your evaluation set from going stale?"**

The corpus is external and cloned fresh, so labels can silently become wrong — a `Draft` becomes
`Final`, a PEP gains a supersession edge. So every claim the gold set makes is re-derived from the
corpus headers by a script that exits non-zero: labels must exist, a query marked as a trap must have a
real superseded or rejected predecessor pointing at its answer, and a version-scoped query must name
that version in its own text.

It also prints the composition every run — 51 queries, 48 distinct PEPs — because a set concentrated on
a few documents is unrepresentative in a way no individual label check catches.

> **"How did you choose the baseline for your intervention?"**

This is the question I got wrong first and then measured. The obvious baseline was the unreranked hybrid
number I already had — but my reranker draws a candidate pool ten times deeper before reranking, and RRF
over a deeper pool promotes different chunks. So the "no reranking" configuration of my pipeline is not
the same system as the earlier rung.

I could size the confound because prompts are cached by content: at λ=0 exactly **one prompt in 51** hit
the cache, meaning 50 of 51 queries saw different context purely from pool depth. And that effect alone
accounted for roughly half the apparent improvement. So the baseline is λ=0 of the same pipeline, and the
earlier number is printed beside it labelled as pool depth.

The general rule: your baseline must differ from your treatment in exactly one respect, and "the same
system with the feature off" is only that if turning it on changed nothing else.

> **"Why report fixed-and-broken counts rather than the rate?"**

Because at n=51 a rate difference of 0.118 is six answers, and stated as a rate it invites an argument
about significance I cannot win. "Six queries improved, zero regressed" is a claim about individual
queries that does not depend on rates — and it distinguishes a clean improvement from a churn of twenty
changes that happened to net six, which the rate cannot.

## Debugging Tip

Run the verifier before trusting any result, and read the composition line rather than only the OK:

```bash
.venv/bin/python scripts/verify_gold.py
```

If `queries` or the category counts differ from what a stored result assumed, that result is not
reproducible — which is exactly the discrepancy recorded in this book's front matter. The README quoted
retrieval numbers measured at n=45 after the set had grown to 51, and it was the composition line that
made the mismatch obvious.

## Summary

- The verifier re-derives every gold-set claim from the corpus and exits non-zero, because label rot
  against an external corpus is silent.
- Its trap check scans the corpus graph rather than trusting the flag; its one escape hatch requires a
  written declaration in the entry's note.
- Four separate drivers because runtimes span seconds to half an hour; each prints composition,
  controls, headline, subsets, then per-query detail.
- Controls are printed on every run so the harness re-validates itself in front of the reader.
- `--limit` prints its own caveat into the output so the warning cannot be separated from the numbers.
- The sweep's baseline is λ=0 of the same pipeline, not the earlier hybrid row — a distinction sized by
  the cache-hit counter, which showed 50 of 51 prompts differed from pool depth alone and accounted for
  about half the apparent gain.
- The per-query diff reports six fixed and zero broken, a claim that does not rest on rates.

## Key Takeaways

1. A gold set labelled against a moving corpus needs a tripwire that fails, not warns.
2. Baseline and treatment must differ in exactly one respect — check that they do.
3. Print caveats inside the output they qualify.
4. Instrumentation built for one purpose often answers a different and more important question.

## Why the Next Chapter Exists

Part IV is complete: every module read, every decision accounted for. Part V now puts it together.

Chapter 22 traces a single query — `q16`, the one that produced the project's central finding — through
every layer, from raw text to generated answer, naming every function it passes through and every piece
of state it touches.
