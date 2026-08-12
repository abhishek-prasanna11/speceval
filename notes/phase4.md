# Phase 4 — the tradeoff curve, and the hypothesis that turned out to be wrong

Rung 4 measured across a strength sweep, at both the retrieval and the answer level.

---

## 1. What this phase added

| File | Role |
|---|---|
| `speceval/rerank.py` | `AuthorityReranker` — graded status weights, supersession edges, tunable strength |
| `run_phase4.py` | Fine retrieval sweep (free) + answer sweep at four strengths + one ablation |
| `tests/test_rerank.py` | 19 tests, including the strength-0 invariant |

## 2. The reranker

```
final = base_rank_score * (1 - strength + strength * authority_weight)
```

Two signals feed `authority_weight`:

**Graded status.** The 9-value enum has real gradations — `Active` is as authoritative as
`Final`, a `Draft` is weaker without being dead, `Deferred` (dormant) outranks `Rejected`
(refused). Weights live in one table in `rerank.py` so they can be argued with rather than
buried in a conditional.

**The supersession edge, independently of status.** This is the signal that justified choosing
this corpus. PEP 333 is `Final` *and* carries `Superseded-By: 3333`; PEP 409 is `Final` *and*
superseded by 415. Status alone cannot separate either pair — both members read `Final`. So any
PEP whose `Superseded-By` points at a document that exists is down-weighted regardless of its
own status. This also handles the multi-hop packaging chain 241 → 314 → 345 → 566, where every
intermediate hop is itself superseded.

### The knob is sharply non-linear, and the grid had to account for it

RRF base scores at adjacent ranks are 1/61 and 1/62 — 1.6% apart — while authority weights
span 0.02 to 1.0. So flipping an *adjacent* pair takes only `strength ≈ 0.0165`, while
promoting a document from 20 ranks deeper takes `≈ 0.25`. Predicted from the arithmetic before
the sweep, then confirmed by it: nothing at all changes below λ=0.01, and the transition begins
at λ=0.02.

A uniform 0.25-step grid would have shown three indistinguishable points and made a genuinely
graded knob look like a switch.

### The baseline is λ=0 of this pipeline, not Phase 3's hybrid row

The reranker draws a pool ten times deeper than rung 3 before fusing, and RRF over a deeper
pool promotes different chunks. This is not a technicality: **λ=0 shared only 1 of 51 prompts
with Phase 3's hybrid**, so 50 of 51 queries saw different context. Using Phase 3's number as
the control would have attributed a pool-depth effect to reranking on almost every query.

Worth noting on its own: deepening the pool *by itself* improved the superseded-citation rate
from 0.235 (Phase 3 hybrid) to 0.157 (λ=0), with no reranking at all.

## 3. Retrieval-level sweep

```
  lambda   Recall@10   nDCG@10   trap@1   trap@1 (trap set)
------------------------------------------------------------
       0       0.951     0.771    0.294               0.550
   0.005       0.951     0.771    0.294               0.550
    0.01       0.951     0.771    0.294               0.550
    0.02       0.951     0.803    0.137               0.350
    0.05       0.971     0.813    0.137               0.350
     0.1       0.971     0.831    0.059               0.150
    0.15       0.971     0.841    0.039               0.100
    0.25       0.980     0.856    0.020               0.050
     0.5       0.980     0.852    0.000               0.000
    0.75       0.971     0.844    0.000               0.000
       1       0.961     0.840    0.000               0.000
```

## 4. Answer-level sweep

```
  lambda  superseded  authorit.  version  halluc.   trap: superseded
--------------------------------------------------------------------
       0       0.157      0.765    0.714    0.000              0.300
    0.05       0.235      0.725    0.857    0.000              0.450
    0.25       0.078      0.824    0.714    0.000              0.150
       1       0.039      0.863    0.714    0.000              0.050
```

In counts, since the rates hide how few answers move: superseded citations go from **8 of 51
to 2 of 51**, and on the trap subset from **6 of 20 to 1 of 20**.

Per query, λ=0 → λ=1: **6 fixed, 0 broken.** Five of the six are trap queries (q06 dropped
PEP 3103, q16 dropped 563, q20 dropped 554, q26 dropped 722, q47 dropped 431; q31 dropped 346).
Not one query that was clean at λ=0 regressed.

## 5. Finding — the tradeoff this project was built to measure does not exist

The premise was that authority correctness and retrieval recall trade off: filter hard enough
on authority and you starve the retriever of legitimate context. The README says so, and the
résumé bullet I drafted at the planning stage said *"at a measured cost of X% recall."*

**There is no such cost over the useful range.** Between λ=0 and λ=0.5:

| | λ=0 | λ=0.5 | direction |
|---|---|---|---|
| trap@1 | 0.294 | 0.000 | eliminated |
| Recall@10 | 0.951 | 0.980 | **improved** |
| nDCG@10 | 0.771 | 0.852 | **improved** |

Everything improves at once. Recall only turns back down at extreme strength (0.980 → 0.961 at
λ=1, which is one query), and even λ=1 still beats the λ=0 baseline on both retrieval metrics.
At the answer level the same pattern holds: superseded citations fall 0.157 → 0.039 while
authoritative citations *rise* 0.765 → 0.863.

**Why the premise was wrong.** The mental model was that authority reranking discards documents.
It does not — it reorders a fixed candidate pool. Demoting a superseded PEP does not remove
information, because the successor is almost always in the same pool: supersession pairs are
topically near-identical, so whatever retrieved 563 also retrieved 649. The intervention swaps
a dead document for its live twin rather than trading coverage for correctness.

That mechanism is specific and it predicts where the premise *would* have held: a corpus whose
superseded documents have no live successor, or where the successor is worded differently enough
to fall outside the pool. This corpus is not that, and the study can only speak for this corpus.

So the honest headline is stronger than the planned one: on this corpus, authority-aware
reranking is close to free, and λ ≈ 0.25–0.5 dominates the baseline on every metric measured.

## 6. Finding — partial reranking is worse than none

λ=0.05 is worse than λ=0 on both answer-level metrics that matter:

| | λ=0 | λ=0.05 | λ=0.25 |
|---|---|---|---|
| superseded-citation | 0.157 | **0.235** | 0.078 |
| authoritative-citation | 0.765 | **0.725** | 0.824 |

Non-monotonic, and it reverses cleanly by λ=0.25.

The mechanism follows from §2: at λ=0.05 the knob is past the ~0.0165 threshold where adjacent
pairs flip, but well short of the ~0.25 needed to promote a live document from deeper in the
pool. So it reshuffles the top of the list without being able to reach the document that would
fix it — enough force to disturb the ordering, not enough to repair it.

This is only visible because the grid was fine at the low end. A 0.25-step sweep would have
reported a clean monotonic improvement and missed a regime where the intervention actively
hurts. It is also a practical warning: a half-tuned authority reranker can be worse than none.

**Caveat, stated plainly:** λ=0.05 is 12 of 51 answers versus λ=0's 8. That is four answers.
The direction is consistent across both metrics and has a mechanism, but a four-answer
difference at n=51 is suggestive, not established.

## 7. Finding — the naive version rule is harmful, as predicted

The obvious version rule is to penalise any PEP whose `Python-Version` postdates the version
the query asks about. `rerank.py` argues this is wrong: answering *"can I use the walrus
operator in 3.7?"* correctly requires retrieving PEP 572, whose version is 3.8 — the very
document the rule demotes.

Implemented behind a flag and measured rather than asserted:

| variant | superseded | authoritative | version-correct |
|---|---|---|---|
| λ=1 | 0.039 | **0.863** | **0.714** |
| λ=1 + version penalty | 0.039 | 0.804 | 0.571 |

It changes nothing about authority and makes both other metrics worse. Version metadata tells
you what the answer must *say*, not which document to trust. Unlike the tradeoff hypothesis,
this prediction was made in advance and held.

## 8. Honest limitations

- **51 queries.** The headline is 8 answers → 2. Real, but small: three fewer fixes would halve
  the effect. The 6-fixed / 0-broken split is the most robust form of the claim, because it does
  not depend on rates.
- **The version metric rests on 7 queries.** Its apparent peak at λ=0.05 (0.857) is one answer
  and should be ignored.
- **The status weights are judgements, not measurements.** `Draft = 0.55` and
  `Deferred = 0.25` were chosen by hand. No sensitivity analysis was run on them, so an
  unknown part of the effect may be attributable to those constants rather than to the idea.
  This is the most substantive untested assumption in the study.
- **One embedding model, one generation model, one prompt, one corpus.** The §5 mechanism
  argues the result should *not* generalise to corpora where superseded documents lack a
  near-duplicate live successor, and that is untested.
- **Deterministic, but not repeated.** Temperature 0 with a fixed seed gives byte-identical
  runs, so there is no variance to report — which is not the same as having measured
  robustness to prompt or seed changes.

## 9. Where the study ended up

Four rungs, one metric set, 51 verified queries, ~460 generated answers, all metrics derived
from PEP headers with no judge and no hand-grading.

Two of the three substantive predictions made along the way were wrong, and both are recorded:
dense retrieval was predicted to lose on identifier queries and did not (Phase 2), and
authority reranking was predicted to cost recall and did not (Phase 4). The one that held was
the version rule being harmful. That ratio is the argument for having built the measurement
apparatus before forming opinions.
