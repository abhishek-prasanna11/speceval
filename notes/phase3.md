# Phase 3 — generation, and where the retrieval metrics stop predicting anything

Answer-level measurement for rungs 1–3, scored automatically from PEP headers.

---

## 1. What this phase added

| File | Role |
|---|---|
| `speceval/generate.py` | Ollama generation, prompt construction, citation extraction, answer cache |
| `speceval/answer_metrics.py` | The four answer-level metrics |
| `run_phase3.py` | Driver: 3 rungs × 51 queries = 153 answers |
| `tests/test_answers.py` | 27 tests on extraction, prompt construction and scoring |

`llama3.2`, top-5 **chunks** per answer, ≤220 tokens, answers cached under
`.cache/answers/` keyed on model, options and prompt.

## 2. Two decisions that determine what the numbers mean

**Temperature 0, fixed seed.** Verified byte-identical across three repeated runs, so
run-to-run variance is zero and every difference between rungs is attributable to retrieval.
This is why the deferred variance-calibration item is not needed for these results — and why
it *would* be needed the moment sampling is enabled.

**The prompt hides each PEP's `Status`.** Showing `Status: Superseded` would let the
generator route around a bad retrieval, which is a prompt-level intervention and would
confound a comparison between retrieval strategies. The generator is held constant and
uninformed deliberately. Making it authority-aware is a separate experiment and is listed as
future work rather than quietly folded in here.

## 3. Results

```
Retriever     superseded authoritative     version    hallucin.     p95 ms
--------------------------------------------------------------------------
BM25               0.275         0.686       0.429        0.000      11701
Dense              0.235         0.686       0.429        0.000      13796
Hybrid             0.235         0.765       0.571        0.000      13674
```

- **superseded** — answers citing a Rejected / Withdrawn / Superseded / Deferred PEP
- **authoritative** — answers citing a gold-labelled PEP
- **version** — of the 7 version-scoped queries, those surfacing the real release
- **hallucin.** — answers citing a PEP number absent from the corpus

Trap versus ordinary:

```
Retriever     superseded authoritative     version
--------------------------------------------------
BM25/trap          0.500         0.450       0.667
BM25/ordinary      0.129         0.839       0.250
Dense/trap         0.400         0.500       0.667
Dense/ordinary     0.129         0.806       0.250
Hybrid/trap        0.450         0.600       0.667
Hybrid/ordinary    0.097         0.871       0.500
```

## 4. Finding — roughly one answer in four cites a dead specification

The headline baseline: **23.5%–27.5% of all answers cite a non-authoritative PEP**, rising to
**40%–50% on the trap subset** and falling to **10%–13% on ordinary queries**. A ~4x gap
between subsets, which is what the trap/ordinary split was built to expose.

This is the number Phase 4 has to move, and it is now measured rather than assumed.

Two representative failures, both from the Dense rung:

- **q23** — *"how are Python package version numbers compared"* → *"According to PEP 386,
  Python package version numbers are compared using the standard schema specified in that…"*
  PEP 386 is `Superseded` by 440. The answer presents a dead specification as current with no
  hedge at all.
- **q20** — *"can I run multiple interpreters from the standard library"* → *"Yes, you can run
  multiple interpreters from the standard library."* citing PEP 554, which is `Superseded` by
  734 and lands in 3.14. A confidently wrong availability claim sourced from a dead PEP.

## 5. Finding — the retrieval metrics did not predict answer quality

This is the most useful thing in the phase.

Phase 2 concluded that hybrid fusion bought **nothing** over dense: Recall@10 0.944 vs 0.967,
nDCG@10 0.810 vs 0.830, rank-1 29 vs 30. On those numbers, hybrid was the rung to drop.

At the answer level hybrid is the **best** rung:

| | authoritative | version | superseded |
|---|---|---|---|
| Dense | 0.686 | 0.429 | 0.235 |
| Hybrid | **0.765** | **0.571** | 0.235 |

Same corpus, same queries, opposite conclusion.

The mechanism is a measurement mismatch, not a mystery. Retrieval was scored on the top-10
**distinct PEPs**; generation consumes the top-5 **chunks**. Fusion reorders chunks, and its
chunk-level ordering evidently packs better evidence into a 5-chunk window even where its
PEP-level ranking at k=10 is marginally worse. Recall@10 over PEPs simply does not see that.

**The lesson generalises past this project:** a retrieval metric is a proxy for downstream
answer quality, and it is only as faithful as the match between what it scores and what the
generator actually consumes. Ours were mismatched in two ways at once — unit (PEPs vs chunks)
and depth (10 vs 5) — and the ranking of strategies inverted. Anyone tuning retrieval on
nDCG alone and shipping a generator over top-5 chunks is optimising the wrong quantity.

## 6. Finding — no hallucinated citations at all

**0 of 153 answers cited a PEP number absent from the corpus.** Every citation resolved to a
real PEP. Grounding is not the failure mode here; *authority* is. The model does not invent
sources, it faithfully cites dead ones.

## 7. Generation dominates latency by three orders of magnitude

End-to-end p95 is 11.7–13.8 **seconds**. Retrieval p95 from Phase 2 was 18–41 **milliseconds**.
Generation is ~1000x the cost of the retrieval step being compared.

That is the entire justification for timing retrieval separately: an end-to-end number would
have shown three effectively identical rungs and hidden the 2x retrieval difference between
BM25 and dense completely.

## 8. Two limitations the metric itself has — found by reading the failures

Both are cases where the automatic metric marks an answer wrong that a human would not, and
both argue specifically for the deferred judge rather than for more automation.

**A correct conclusion drawn from a dead source still counts as a failure.** q47 — *"is
zoneinfo available in Python 3.8"* → *"No, zoneinfo is not available in Python 3.8. PEP 431
states…"*. The conclusion is right; the citation is `Superseded`. The metric measures
citation hygiene, not answer truth. That is by design and defensible, but it must not be
described as measuring correctness.

**A correct hedge is penalised.** q30 is the deliberate no-authoritative-answer case (PEP 543
`Withdrawn`, successor 748 only a `Draft`). The model answered *"There is no definitive answer
based on the provided excerpts. PEP 543 and PEP 748 both propose a unified TLS API…"* — which
is exactly the desired behaviour. It still scores as a superseded citation, because the metric
cannot distinguish *citing a dead PEP as authority* from *citing a dead PEP while explaining
that nothing is settled*. A judge could. A regex cannot.

## 9. Honest limitations

- **51 queries.** The gap between Dense and BM25 on superseded-citation rate (0.235 vs 0.275)
  is **two answers**. It is not a finding. The subset gap (~0.45 vs ~0.11) and the
  retrieval-vs-answer inversion are large enough to take seriously; the margins between
  rungs are not.
- **The version metric rests on 7 queries.** 0.429 versus 0.571 is a difference of one
  answer. Treat it as anecdote until the gold set grows.
- **One generation model**, one prompt, one temperature. Whether the superseded-citation rate
  is a property of `llama3.2` or of the retrieved context is untested; a stronger model might
  notice staleness unprompted.
- **Categories are now unbalanced** — 21 availability, 15 identifier, 15 rationale — because
  the six version-scoped queries added in this phase are all availability questions. Per
  category comparisons are correspondingly weaker on that axis.

## 10. Next — Phase 4

Rung 4: rerank the hybrid candidate list by `Status` and `Python-Version`, with a **tunable
strength parameter** so the output is a tradeoff curve rather than two points. The baseline to
beat is now concrete: ~1 answer in 4 cites a dead PEP overall, ~1 in 2 on trap queries.

One design consequence of §5: Phase 4 must report answer-level metrics as primary. Retrieval
metrics have now demonstrably ranked the rungs in the wrong order once, so the reranker gets
judged on citations, not on nDCG.
