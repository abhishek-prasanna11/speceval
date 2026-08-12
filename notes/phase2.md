# Phase 2 — dense retrieval, hybrid fusion, and the first real surprise

Rungs 1–3 of the ladder measured on the full 45-query gold set.

---

## 1. What this phase added

| File | Role |
|---|---|
| `speceval/embed.py` | Ollama embedding client, batched, with a fingerprinted disk cache |
| `speceval/retrievers.py` | `DenseRetriever` (rung 2), `HybridRetriever` (rung 3) |
| `eval/queries_gold.json` | 45 queries, 15 per category, 18 trap cases |
| `scripts/verify_gold.py` | Re-derives every label from the corpus; exits non-zero on drift |
| `tests/test_dense.py` | Fusion, cosine, cache and platform-numerics tests |

Corpus embedding took ~17 minutes for 19,763 chunks (~19/s) and is cached, so re-running
the evaluation is instant.

## 2. Decisions

**Reciprocal rank fusion, not a weighted score sum.** BM25 scores are unbounded and
corpus-dependent; cosine similarities live in [-1, 1]. Adding them requires a calibration
constant, and that constant would then sit unexamined underneath every result in the study.
RRF uses only ranks:

```
score(d) = SUM over systems s of  1 / (RRF_K + rank_s(d))
```

`RRF_K = 60` is the conventional value. It damps the top ranks — without it, rank 1 would
outweigh rank 2 by 2x.

**Asymmetric prefixes.** `nomic-embed-text` expects `search_document: ` on documents and
`search_query: ` on queries. Omitting them silently degrades dense retrieval, and the damage
would then be misattributed to dense retrieval rather than to the prefix.

**Brute-force cosine.** One `(19763, 768) @ (768,)` product is a few milliseconds. An ANN
index would inject approximation error into a study whose entire subject is measurement
accuracy.

**The gold set is 40% trap cases, not 100%.** The 27 ordinary queries are the control. A
reranker evaluated only on queries it was designed to win would have its benefit measured
where it cannot lose.

## 3. Results

```
Retriever     Recall@10    nDCG@10   trap@1    p50 ms    p95 ms
---------------------------------------------------------------
Oracle            1.000      1.000    0.022      0.00      0.00
Random            0.000      0.000    0.356      0.01      0.01
BM25              0.889      0.710    0.289     11.97     18.04
Dense             0.967      0.830    0.244     32.23     40.74
Hybrid            0.944      0.810    0.289     14.99     20.23
```

`trap@1` = fraction of queries whose rank-1 PEP is Rejected / Withdrawn / Superseded /
Deferred. A retrieval-side proxy for the citation metrics Phase 3 will measure on generated
answers.

Rank-1 accuracy, and how deep the first correct answer sits:

| Retriever | rank-1 correct | mean rank of first hit | never found in top 10 |
|---|---|---|---|
| BM25 | 23/45 (51.1%) | 1.98 | 4 |
| Dense | 30/45 (66.7%) | 1.68 | 1 |
| Hybrid | 29/45 (64.4%) | 1.63 | 2 |

### Two sanity checks worth noting

**Random's trap@1 is 0.356, and 36% of the corpus is non-authoritative.** The random
retriever recovers the corpus base rate almost exactly. That is strong evidence `trap@1` is
measuring what it claims rather than something incidental.

**Oracle's trap@1 is 0.022, not 0.000.** Exactly one query (q30, "is there a unified TLS API
in Python") deliberately has no authoritative answer: PEP 543 is `Withdrawn` and its
successor 748 is only a `Draft`. So even a perfect retriever registers one trap here. The
metric's floor is 1/45, not zero, and that is a property of the corpus rather than a bug.

## 4. Finding — better retrieval did not buy better authority

Dense beats BM25 on every retrieval metric: recall 0.967 vs 0.889, nDCG 0.830 vs 0.710,
rank-1 accuracy 66.7% vs 51.1%. It costs ~2.7x latency (32ms vs 12ms p50), essentially all
of it the Ollama round trip for the query embedding.

But splitting by subset shows the advantage does not carry over to authority:

```
Retriever                  trap (n=18)           ordinary (n=27)
----------------------------------------------------------------
BM25              R 0.81 N 0.61 T 0.50      R 0.94 N 0.78 T 0.15
Dense             R 0.92 N 0.68 T 0.56      R 1.00 N 0.93 T 0.04
Hybrid            R 0.92 N 0.67 T 0.56      R 0.96 N 0.90 T 0.11
```

On **ordinary** queries dense is clearly better on authority: 1 trap in 27 versus BM25's 4.
On **trap** queries the two are indistinguishable — 10 of 18 versus 9 of 18.

**Stated carefully: a one-query difference at n=18 is not a finding.** What the data
supports is the weaker but still interesting claim that *dense retrieval's large advantage
at finding the right document does not produce any advantage on the queries where
supersession matters.* Half of the trap queries still lead with a dead PEP no matter which
retriever is used. That is the gap Phase 4 exists to close, and it is now measured rather
than assumed.

### The mechanism, from the divergence

Dense and BM25 fall into *different* traps. Eight are shared; beyond those:

- **Dense-only traps (BM25 got these right): q16, q22, q24.**
- **BM25-only traps (dense got these right): q02, q04, q15, q20, q28.**

The dense-only cases share a shape. For q16, "how do I postpone the evaluation of
annotations", the superseded PEP 563 is titled *Postponed Evaluation of Annotations* — a
near-exact semantic match — while the live answer, PEP 649, is titled *Deferred Evaluation
Of Annotations Using Descriptors*. BM25 matched the live PEP; the embedding matched the dead
one. q22 is the same shape (571 `manylinux2010` over 600) and q24 likewise (601 `Rejected`
over 765).

So the better a retriever is at semantics, the more attractive a superseded predecessor
becomes: it is on-topic, often better-titled, and frequently better-written prose. **The
authority problem is not a retrieval-quality problem, and improving retrieval will not
solve it.** That is the argument for rung 4 existing at all.

## 5. Finding — hybrid did not beat dense

RRF fusion produced no improvement over dense alone: recall 0.944 vs 0.967, nDCG 0.810 vs
0.830, rank-1 29 vs 30, though mean rank of the first hit was marginally better (1.63 vs
1.68). Mixed and small.

**The honest statement is "no measurable gain", not "hybrid is worse".** Every gap here is
one or two queries out of 45.

The plausible mechanism: RRF weights both systems equally, and BM25 is substantially the
weaker of the two on this corpus. Fusing a strong ranker with a weak one at equal weight
drags the strong one toward the weak one, and RRF's rank-only formulation has no way to
express "trust dense more". A weighted variant would, at the cost of introducing exactly the
calibration constant RRF was chosen to avoid.

This matters for the résumé claim: the received wisdom is that hybrid retrieval beats either
component. It did not here, and the reason is specific to this corpus and this fusion rule.

### One fusion pathology, worth keeping

On q34 ("can I use builtin list and dict directly in type hints") hybrid returned PEP 637
(`Rejected`) at rank 1 — a PEP that *neither* BM25 nor dense ranked first. RRF can promote a
document ranked mid-list by both systems above a document ranked first by only one. Fusion
is not guaranteed to return something a component liked best.

## 6. The Phase 1 prediction was half wrong

Phase 1 predicted dense would beat BM25 on *rationale* queries and **lose** on *identifier*
queries, on the theory that embeddings handle paraphrase well and exact identifiers badly.

| Category | BM25 nDCG | Dense nDCG | Gap |
|---|---|---|---|
| rationale | 0.64 | 0.81 | +0.17 |
| identifier | 0.77 | 0.86 | +0.09 |
| availability | 0.72 | 0.82 | +0.10 |

The *direction* held — the gap is widest on rationale, exactly as predicted. The **loses on
identifier** half was wrong: dense won there too. Two plausible reasons, neither tested:
`indexed_text` prepends the PEP title and section heading, which gives the embedding a
strong clean signal for identifier-style questions; and `nomic-embed-text` was trained with
code and technical text in its mixture, so `__future__`-style tokens may be less alien to it
than assumed.

Recording the miss rather than quietly dropping it, since a prediction that only counts when
it succeeds is not a prediction.

## 7. Honest limitations

- **n=45 is small.** Differences of one or two queries are noise, and several comparisons in
  this document are exactly that size. This is the strongest argument for the deferred
  "expand to 150 queries" item.
- **No repeated runs.** BM25 and dense are deterministic given a fixed cache, so
  run-to-run variance is zero here — but that will stop being true in Phase 3, where
  generation is sampled. Variance calibration is deferred, and until it exists, small
  Phase 3 differences must not be read as real.
- **One embedding model.** Every dense result is `nomic-embed-text`. Whether the
  superseded-title attraction is a property of embeddings generally or of this model is
  untested.
- **Labels are mine.** `scripts/verify_gold.py` checks internal consistency against the
  corpus headers — that a labelled PEP exists, that a trap-marked query really has a
  superseded or rejected predecessor pointing at its answer — but it cannot check that the
  query text is best answered by the PEP I chose.

## 8. Next — Phase 3

Generation, and the two citation metrics computed automatically from PEP headers:
superseded-citation rate and version correctness. The baseline to beat is now established
and uncomfortable: roughly half of trap queries lead with a dead PEP regardless of retriever,
so a generator fed the top-k will cite dead specifications at a measurable rate.
