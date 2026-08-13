# Chapter 16 — The Metrics Layer

`speceval/metrics.py` (56 lines) · `speceval/evaluate.py` (126 lines)

## Learning Objectives

- Explain the split between the two modules and why it matters.
- Explain `trap@1` — what it measures, why it exists at the retrieval level, and what it proxies for.
- Explain the timing decision and the measured ratio that justifies it.
- Explain nearest-rank percentiles and why interpolation was rejected.
- Explain how per-category disaggregation is computed and reported.

## Motivation

Chapter 5 defined Recall@K and nDCG@K. Chapter 13 validated them. This chapter is about the layer that
*applies* them: the loop that runs a retriever over a query set, times it, slices the results, and
formats them for a human.

That sounds like plumbing. Two decisions inside it are not.

## The split

Two modules, and the boundary is deliberate:

```
   metrics.py     pure functions on sequences and sets
                  no I/O, no timing, no domain types
                  recall_at_k, ndcg_at_k, mean

   evaluate.py    the loop: run a retriever, time it, aggregate,
                  disaggregate, format
```

`metrics.py` knows nothing about PEPs, queries, or retrievers. Its functions take `Sequence[Hashable]`
and `set[Hashable]`. That generality is what let chapter 13's tests exercise them with strings
(`["a", "b", "c"]`) rather than constructing PEP numbers — the metric under test is separated from the
domain it is used on.

## Deep Explanation: the evaluation loop

```python
def evaluate(
    retriever: Retriever,
    queries: list[Query],
    k: int = K,
    non_authoritative: set[int] | None = None,
) -> Result:
    recalls: list[float] = []
    ndcgs: list[float] = []
    latencies_ms: list[float] = []
    traps: list[float] = []
    by_category: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for query in queries:
        start = time.perf_counter()
        ranked = retriever.search(query, top_k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        relevant = set(query.relevant)
        recall = recall_at_k(ranked, relevant, k)
        ndcg = ndcg_at_k(ranked, relevant, k)

        recalls.append(recall)
        ndcgs.append(ndcg)
        by_category[query.category].append((recall, ndcg))

        if non_authoritative is not None:
            traps.append(1.0 if ranked and ranked[0] in non_authoritative else 0.0)
    ...
```

### The timed region

```python
        start = time.perf_counter()
        ranked = retriever.search(query, top_k=k)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
```

Exactly one call is inside the timer: `search`. Metric computation, list appends and category
bookkeeping are all outside it.

`time.perf_counter` rather than `time.time` — it is monotonic and has the highest resolution the
platform offers. `time.time` can go backwards (NTP adjustment) and would occasionally produce a
negative duration.

### Why retrieval is timed separately from generation

This is the decision chapter 5 flagged, and the measured numbers make the case better than the argument
does:

| Stage | p95 |
|---|---|
| Retrieval (BM25) | ~43 ms |
| Retrieval (Dense) | ~45 ms |
| **End to end, including generation** | **~11.7–13.8 s** |

Generation is roughly **300 times** the cost of the retrieval step being compared. An end-to-end
timing would show three effectively identical numbers — 13.7 s versus 13.8 s versus 11.7 s, dominated
by the model's own variance — and the 10 ms gap between BM25 and dense would be invisible.

So the two are measured in different places: retrieval here, end-to-end in `run_phase3.py` and
`run_phase4.py`. Reporting a single "latency" column would have hidden the only latency difference the
retrieval comparison actually has.

### `trap@1`

```python
        if non_authoritative is not None:
            traps.append(1.0 if ranked and ranked[0] in non_authoritative else 0.0)
```

One line, and it is the bridge between Part I's retrieval metrics and Part III's thesis.

```python
    # Fraction of queries whose rank-1 PEP is Rejected/Withdrawn/Superseded/Deferred.
    # This is a *retrieval-side proxy* for the superseded-citation rate that Phase 3 will
    # measure on generated answers -- available now, with no LLM in the loop, and it makes
    # the authority problem visible one rung at a time instead of only at the end.
    trap_at_1: float | None = None
```

Three properties worth drawing out.

**It requires no language model.** The authority problem could otherwise only be observed after
generation, which costs seconds per query. `trap@1` makes it visible at retrieval speed, which is what
allowed chapter 20's eleven-point strength sweep to be run for free.

**It is `None` when not requested**, not `0.0`. The parameter is optional, and a caller that does not
supply `non_authoritative` gets a column showing `-` rather than a spurious zero. Absence of a
measurement and a measurement of zero are different things, and conflating them in a results table is a
way to publish a number nobody computed.

**`ranked and ranked[0]`** guards the empty case. A retriever returning nothing (chapter 15: an
unanswerable query) is not a trap.

## Deep Explanation: aggregation and disaggregation

```python
    latencies_ms.sort()
    return Result(
        retriever=retriever.name,
        overall=Scores(mean(recalls), mean(ndcgs), len(queries)),
        per_category={
            category: Scores(
                mean([r for r, _ in pairs]), mean([n for _, n in pairs]), len(pairs)
            )
            for category, pairs in sorted(by_category.items())
        },
        latency_p50_ms=_percentile(latencies_ms, 0.50),
        latency_p95_ms=_percentile(latencies_ms, 0.95),
        trap_at_1=mean(traps) if traps else None,
    )
```

Per-category scores are computed in the same pass, from the `by_category` accumulator. Chapter 12
argued that findings live in the breakdown and chapter 5 noted that scoring is free, so there is no
reason to make disaggregation a separate opt-in step. It is always computed.

`sorted(by_category.items())` keeps category order stable across runs — the same determinism discipline
as everywhere else, applied here to output formatting so two runs' tables can be diffed.

`Scores` carries `n_queries` alongside the two metrics:

```python
@dataclass
class Scores:
    recall: float
    ndcg: float
    n_queries: int
```

That field is what lets a reader judge a number. `rationale 0.64` means something different at n=15
than at n=3, and the count travels with the score rather than having to be looked up. The drivers print
it.

## Deep Explanation: percentiles

```python
def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    # Nearest-rank percentile: adequate here, and it never interpolates between two
    # real measurements into a value that was never observed.
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]
```

**Nearest-rank**, deliberately. The common alternative — linear interpolation between the two
surrounding samples — reports a value that *was never measured*.

For 51 latency samples that distinction is small in magnitude and clear in principle: every number this
function returns is a real observation from a real run. In a measurement study, a reported figure that
no measurement produced is a small lie, even when it is a defensible estimate.

The `min(len - 1, ...)` guard prevents an index overflow at `fraction = 1.0`.

Why p50 and p95 rather than a mean? A mean hides the tail. One slow call — a cold model, a garbage
collection pause — moves a mean and does not move a median. Reporting both a middle and a tail figure
describes the distribution rather than a single point on it.

## Deep Explanation: formatting is part of the interface

Two formatting functions, and they are not incidental.

```python
def format_table(results: list[Result], k: int = K) -> str:
    lines = [
        f"{'Retriever':<12} {'Recall@' + str(k):>10} {'nDCG@' + str(k):>10} "
        f"{'trap@1':>8} {'p50 ms':>9} {'p95 ms':>9}",
        "-" * 63,
    ]
    for result in results:
        trap = "-" if result.trap_at_1 is None else f"{result.trap_at_1:.3f}"
        ...
```

The column header interpolates `k` — `Recall@10`, not `Recall` — so a table can never be read without
its cutoff. Chapter 5 argued that a metric reported without its `K` is not a claim; this makes it
impossible to produce one accidentally.

`trap = "-" if result.trap_at_1 is None` renders the absent case as a dash, carrying the
`None`-versus-`0.0` distinction all the way to the human reading the output.

Three decimal places on rates: at n=51, one query is 0.0196, so two decimals would round two adjacent
results to the same value. Three places make single-query differences visible — which matters, because
several differences in this study *are* single-query and the reader needs to see that.

## Systems Perspective

Scoring cost is negligible: two set operations and a short sum per query. For 51 queries the entire
metric layer is microseconds against retrieval's tens of milliseconds and generation's seconds.

That cheapness is what makes the design possible. Because computing a metric is free:

- disaggregation is always on rather than opt-in,
- the oracle and random controls run on every driver invocation (chapter 13),
- and chapter 20's retrieval sweep evaluates eleven reranker strengths in the time it takes to run
  retrieval eleven times, with the metric cost invisible.

A more expensive metric — anything involving a model — would have forced all three of those to become
occasional rather than routine, and each of them earned its place by being routine.

## Common Mistakes

**Timing more than the operation under test.** Metric computation inside the timer inflates the
measurement with work the system does not do in production.

**`time.time` instead of `time.perf_counter`.** Non-monotonic; can produce negative durations.

**Reporting one latency figure across stages with different orders of magnitude.** The slow stage hides
every difference in the fast one.

**Using `0.0` for "not measured".** Publishes a number nobody computed.

**Interpolated percentiles in a measurement study.** Reports a value never observed.

**Reporting a mean latency.** Hides the tail, which is usually the interesting part.

**Omitting `n` from a subgroup result.** A rate without a denominator is unreadable.

**Two decimal places when one query is worth two points.** Rounds real differences away.

## Interview Insight

> **"Why do you time retrieval and generation separately?"**

Because they differ by roughly 300×. Retrieval p95 is about 45 ms; end-to-end p95 with generation is
11.7 to 13.8 seconds. A single end-to-end number would have shown three effectively identical results
dominated by the model's own variance, and the 10 ms difference between lexical and dense retrieval —
which is a network round trip for the query embedding, not arithmetic — would have been invisible.

The general form: when stages differ by orders of magnitude, a combined measurement measures only the
slowest one.

> **"What is `trap@1` and why did you add it?"**

The fraction of queries whose top-ranked document is `Rejected`, `Withdrawn`, `Superseded` or
`Deferred`. It is a retrieval-side proxy for the citation metric that only becomes measurable after
generation.

The reason it exists is practical: generation costs about eight seconds per query, so measuring the
authority problem end-to-end costs seven minutes per configuration. `trap@1` measures it at retrieval
speed, which is what made an eleven-point strength sweep affordable — that sweep is free, and it is
where the shape of the result actually became visible.

> **"Why nearest-rank percentiles?"**

Because interpolation reports a value that was never measured. For latency in a study, I would rather
report a real observation slightly off the true quantile than a defensible estimate that corresponds to
nothing. It is a small point, but the whole value of the harness is that its numbers came from
somewhere.

## Debugging Tip

When a metric moves unexpectedly, check `n` before checking the metric. The `Scores.n_queries` field
and the drivers' `n per category` line exist for this:

```
n per category: availability=21, identifier=15, rationale=15
```

A category count that changed means the gold set changed, and a gold-set change explains a metric
change far more often than a code change does. This is exactly how the discrepancy documented in the
front matter was found: the README reported retrieval numbers measured at n=45, and the set had since
grown to 51.

## Summary

- `metrics.py` is pure and domain-free so it can be tested with strings; `evaluate.py` holds the loop,
  timing, aggregation and formatting.
- Only `search` is inside the timer, using monotonic `perf_counter`.
- Retrieval and generation are timed in different places because they differ by ~300×; a combined
  figure would hide the only difference retrieval has.
- `trap@1` brings the authority thesis into the retrieval-level metrics with no model in the loop,
  which is what made the eleven-point sweep free. It is `None`, not `0.0`, when not requested.
- Disaggregation is always computed because it is free, and `n` travels with every subgroup score.
- Nearest-rank percentiles report real observations rather than interpolated values; p50 and p95
  together describe a distribution where a mean would hide the tail.
- Formatting carries the cutoff into the header and the `None`/`0.0` distinction into the output.

## Key Takeaways

1. Time the operation, nothing else, with a monotonic clock.
2. Never combine timings across stages that differ by orders of magnitude.
3. "Not measured" and "measured zero" must render differently all the way to the reader.
4. Cheap metrics let disaggregation and controls be routine rather than occasional — which is most of
   why they got used.

## Why the Next Chapter Exists

We can score chunks and evaluate rankings. Chapter 17 reads the module that ties them together —
`retrievers.py`, where four strategies satisfy one protocol, chunk rankings collapse into document
rankings, and the interface decision that makes the reranker possible is made.
