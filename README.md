# speceval

**An evaluation study of the tradeoff between retrieval recall and version/authority
correctness in technical-document retrieval.**

Corpus: the Python Enhancement Proposal (PEP) series.

This is not a RAG application. The retrieval pipeline is deliberately modest and exists as the
*system under test*. The project is the measurement.

---

## The question

Retrieval systems optimise for **semantic relevance**. But in corpora where documents supersede
one another and correct answers are **version-conditional**, the most semantically relevant
chunk is frequently the wrong answer — it is well written, on topic, high scoring, and obsolete.

Two failure modes follow, and they are distinct:

1. **Superseded-document citation** — answering from a `Rejected`, `Withdrawn`, or `Superseded`
   PEP as though it were current.
2. **Version incorrectness** — answering with a feature that exists, but not in the version the
   question asked about.

So: **does version- and authority-aware reranking reduce those errors, and what does it cost in
retrieval recall?** The cost was expected to be the interesting half — filtering hard on
authority should eventually starve the retriever of legitimate context.

**It does not, and that is the study's main result.** On this corpus reranking eliminated
rank-1 retrieval of superseded documents while *improving* recall and nDCG. The premise was
wrong for a specific and checkable reason: reranking reorders a fixed candidate pool rather
than discarding documents, and supersession pairs are topically near-identical, so whatever
retrieved the dead PEP also retrieved its live successor. See
[`notes/phase4.md`](notes/phase4.md) §5 for where the premise *would* have held.

## Why PEPs

Every PEP carries machine-readable RFC-822 headers:

```
PEP: 634
Title: Structural Pattern Matching: Specification
Status: Final
Type: Standards Track
Created: 12-Sep-2020
Python-Version: 3.10
Superseded-By: <pep number>      # present on superseded PEPs
Replaces: <pep number>
```

Three properties make this corpus the right instrument:

- **`Status` is a 9-value enum** (`Draft`, `Deferred`, `Provisional`, `Accepted`, `Final`,
  `Active`, `Rejected`, `Withdrawn`, `Superseded`) — an authority signal with gradations, not a
  boolean.
- **`Python-Version` makes correctness version-conditional.** PEP 634 is `Final`, and structural
  pattern matching still does not exist in Python 3.9. Status alone gives a confidently wrong
  answer.
- **The headers let the hard test cases be generated, not hand-written.** Any PEP with
  `Superseded-By`, or with a `Python-Version` later than the version named in a query, is a trap
  case by construction. This is what keeps the golden-set work affordable.

The prose is also clean and written to be read, which matters: chunking quality is not the
variable under study, so it should not become a confound.

## The experiment

Four retrieval strategies, one metric set held constant across all of them:

| # | Strategy | What it adds |
|---|---|---|
| 1 | **BM25** | Lexical baseline. Strong on exact identifiers (`match`, `__future__`, PEP numbers) |
| 2 | **Dense** | Embedding cosine similarity. Strong on paraphrase, weak on identifiers |
| 3 | **Hybrid** | Rank fusion of 1 and 2 |
| 4 | **Hybrid + version/authority reranking** | Rescores by `Status` and `Python-Version` against the query |

**Design decision that the headline claim depends on:** the reranker's influence is a
**tunable strength parameter**, not an on/off switch. A boolean yields two data points and
supports only "it helped." A strength knob yields a curve, and the claim becomes *"authority
correctness rises to X before recall degrades past Y"* — which is the tradeoff this project
exists to measure.

## Metrics

| Metric | Definition |
|---|---|
| **Recall@K** | Fraction of a query's labelled-relevant chunks appearing in the top K. K = 10 |
| **nDCG@K** | Standard normalised DCG, binary relevance, log₂ discount. K = 10 |
| **Superseded-citation rate** | % of answers citing a PEP whose `Status` ∈ {`Rejected`, `Withdrawn`, `Superseded`, `Deferred`}. Measures citation hygiene, **not** answer truth — a correct conclusion drawn from a dead source still counts as a failure |
| **Authoritative-citation rate** | % of answers citing a gold-labelled PEP |
| **Version-correct rate** | Of the version-scoped queries, % whose answer surfaces the release the feature actually landed in. Deliberately narrow: it checks the right version *appears*, not that the surrounding claim is true |
| **Hallucinated-citation rate** | % of answers citing a PEP number absent from the corpus — a grounding check rather than an authority one |
| **Latency** | p50 / p95, **retrieval measured separately from end-to-end** — generation otherwise swamps every difference between the four strategies |

Every answer-level metric is computed **automatically from PEP headers**. No LLM judge and no
human grading on the primary path; that is the single choice that keeps this project small.

## Method notes

Three things separate measured results from asserted ones:

- **The harness is validated before it is trusted.** Two synthetic retrievers are run through it:
  an oracle returning exactly the ground truth (must score Recall@K = 1.0, nDCG = 1.0) and a
  random retriever (must score near chance). A metric implementation that cannot produce those
  two results is broken, and nothing else in the pipeline would reveal it.
- **Queries are tagged by category**, and every result is reported per category as well as in
  aggregate: *availability* ("can I use X in 3.9?"), *design rationale* ("why was X rejected?"),
  *exact identifier* ("what does `__future__` import do?"). Aggregates over ~50 queries hide
  effects; disaggregation is where findings live.
- **Error analysis on ~20 losses.** The cases where the best strategy loses are categorised by
  cause, so results come with a mechanism rather than only a number.

## Results

Rungs 1–3 measured on 51 gold queries, at both the retrieval and the answer level.

**Corpus:** 734 PEPs → 19,763 chunks. 519 (71%) carry `Python-Version`; 31 carry
`Superseded-By`. **262 PEPs (36%) are non-authoritative** — Rejected 131, Withdrawn 70,
Deferred 36, Superseded 25. That is the trap surface.

**Retrieval level** (51-query gold set; `trap@1` = fraction whose rank-1 PEP is
non-authoritative):

| Strategy | Recall@10 | nDCG@10 | trap@1 | rank-1 correct | Retrieval p95 |
|---|---|---|---|---|---|
| BM25 | 0.863 | 0.671 | 0.294 | 47.1% | 42.6 ms |
| Dense | **0.971** | **0.801** | **0.255** | **60.8%** | 45.2 ms |
| Hybrid | 0.931 | 0.765 | 0.294 | 56.9% | 39.1 ms |

**Answer level** (51 queries, 153 generated answers, temperature 0 and a fixed seed so
run-to-run variance is zero):

| Strategy | Superseded-cite ↓ | Authoritative ↑ | Version ↑ | Hallucinated ↓ | End-to-end p95 |
|---|---|---|---|---|---|
| BM25 | 0.275 | 0.686 | 0.429 | **0.000** | 11.7 s |
| Dense | 0.235 | 0.686 | 0.429 | **0.000** | 13.8 s |
| Hybrid | 0.235 | 0.765 | 0.571 | **0.000** | 13.7 s |
| **Hybrid + reranking** (λ=1) | **0.039** | **0.863** | 0.714 | **0.000** | — |

Harness validation, run as tests: Oracle scores 1.000/1.000, Random scores 0.000/0.000.

*End-to-end latency is measured on a cold answer cache. Re-running the drivers after
`.cache/answers/` is populated reports ~1 ms, which is cache-read time, not generation.*

**Rung 4, swept.** The reranker's strength is a knob, so the result is a curve. Retrieval level:

| λ | Recall@10 | nDCG@10 | trap@1 | trap@1 (traps only) |
|---|---|---|---|---|
| 0 | 0.951 | 0.771 | 0.294 | 0.550 |
| 0.02 | 0.951 | 0.803 | 0.137 | 0.350 |
| 0.10 | 0.971 | 0.831 | 0.059 | 0.150 |
| 0.25 | **0.980** | **0.856** | 0.020 | 0.050 |
| 0.50 | **0.980** | 0.852 | **0.000** | **0.000** |
| 1.00 | 0.961 | 0.840 | 0.000 | 0.000 |

Answer level, against the in-pipeline λ=0 control (**not** the rung-3 row above — the reranker
draws a 10x deeper pool before fusing, and λ=0 shared only 1 of 51 prompts with rung 3):

| λ | Superseded-cite ↓ | Authoritative ↑ | Trap subset: superseded ↓ |
|---|---|---|---|
| 0 | 0.157 | 0.765 | 0.300 |
| 0.05 | **0.235** | **0.725** | **0.450** |
| 0.25 | 0.078 | 0.824 | 0.150 |
| 1.00 | **0.039** | **0.863** | **0.050** |

**Finding — better retrieval did not buy better authority.** Dense beats BM25 on every
retrieval metric (rank-1 accuracy 60.8% vs 47.1%). But split by subset, that advantage does
not transfer: on the 31 *ordinary* queries dense takes trap@1 from 0.16 to 0.06, while on the
20 *trap* queries it is very slightly **worse** — 11 of 20 versus BM25's 10 of 20. Over half of
trap queries lead with a dead PEP no matter which retriever runs.

The mechanism shows in where they diverge. Dense and BM25 fall into *different* traps, and
the dense-only ones share a shape: for "how do I postpone the evaluation of annotations",
the superseded PEP 563 is titled *Postponed Evaluation of Annotations* while the live answer
PEP 649 is titled *Deferred Evaluation Of Annotations Using Descriptors*. BM25 matched the
live PEP; the embedding matched the dead one. **The better a retriever is at semantics, the
more attractive a superseded predecessor becomes** — it is on-topic, often better-titled,
and frequently better prose. So the authority problem is not a retrieval-quality problem,
which is the argument for rung 4 existing at all.

**Finding — hybrid did not beat dense.** RRF fusion produced no measurable gain over dense
alone, against the received wisdom that hybrid beats either component. The likely cause is
that RRF weights both systems equally while BM25 is substantially weaker here, and a
rank-only rule cannot express "trust dense more".

**Finding — roughly one answer in four cites a dead specification.** 23.5%–27.5% overall,
rising to **40%–50% on trap queries** and falling to 10%–13% on ordinary ones. That ~4x gap
is what the trap/ordinary split was built to expose, and it is the number rung 4 has to move.

**Finding — the retrieval metrics did not predict answer quality.** Phase 2 concluded hybrid
fusion bought nothing over dense and was the rung to drop. At the answer level hybrid is the
*best* rung: authoritative-citation 0.765 vs 0.686, version 0.571 vs 0.429. Same corpus, same
queries, opposite conclusion. The cause is a measurement mismatch, not a mystery — retrieval
was scored on the top-10 distinct *PEPs* while generation consumes the top-5 *chunks*, so
fusion's chunk-level reordering packs better evidence into the window that actually matters
and Recall@10 cannot see it. **The general lesson: a retrieval metric is only as faithful a
proxy as the match between what it scores and what the generator consumes.** Ours were
mismatched in unit and in depth, and the ranking of strategies inverted.

**Finding — zero hallucinated citations in 153 answers.** Every citation resolved to a real
PEP. Grounding is not the failure mode here; authority is. The model does not invent sources,
it faithfully cites dead ones.

**Finding — no tradeoff, over the whole useful range.** Between λ=0 and λ=0.5, trap@1 goes
0.294 → 0.000 while Recall@10 *rises* 0.951 → 0.980 and nDCG@10 *rises* 0.771 → 0.856.
Everything improves together. Recall only turns back down at extreme strength (one query), and
even λ=1 still beats the baseline on both retrieval metrics. At the answer level, superseded
citations fall 0.157 → 0.039 (**8 of 51 answers → 2**) while authoritative citations rise
0.765 → 0.863. Per query: **6 fixed, 0 broken**, five of the six on trap queries.

**Finding — partial reranking is worse than none.** λ=0.05 is worse than λ=0 on both answer
metrics (superseded 0.235 vs 0.157, authoritative 0.725 vs 0.765), reversing cleanly by λ=0.25.
The knob is past the ~0.0165 threshold where adjacent pairs flip but short of the ~0.25 needed
to promote a live document from deeper in the pool — enough force to disturb the ordering, not
enough to repair it. Only visible because the grid was fine at the low end; a uniform 0.25-step
sweep would have reported clean monotonic improvement and missed a regime where the intervention
actively hurts.

**Finding — the naive version rule is harmful, as predicted.** Penalising a PEP whose
`Python-Version` postdates the asked version changes authority not at all (0.039) and makes both
other metrics worse (authoritative 0.863 → 0.804, version 0.714 → 0.571). Answering "no, it
arrived in 3.8" requires retrieving the very document the rule demotes. Version metadata tells
you what the answer must *say*, not which document to trust.

**Stated carefully.** At n=51 the headline is 8 answers → 2, and several smaller gaps are one or
two answers: the Dense-vs-BM25 superseded gap is two, the λ=0.05 regression is four, and the
version metric rests on 7 queries. The 6-fixed / 0-broken split is the most robust form of the
main claim because it does not depend on rates. The status weights (`Draft = 0.55`,
`Deferred = 0.25`) are hand-chosen judgements with no sensitivity analysis — the most
substantive untested assumption in the study. Two of three predictions made along the way were
wrong and are recorded as wrong. Full analysis in
[`notes/phase2.md`](notes/phase2.md), [`notes/phase3.md`](notes/phase3.md) and
[`notes/phase4.md`](notes/phase4.md).

## Roadmap

- [x] **Phase 1** — Ingest `python/peps`, parse headers, chunk, hand-rolled BM25, Recall@10 and
      nDCG@10 harness. Retrieval only, no LLM. Includes the harness validation above.
      *(Done: baseline measured.)*
- [x] **Phase 2** — Dense retrieval (brute-force cosine) and hybrid fusion. Rungs 1–3 measured.
      *(Done: 57 tests, gold set expanded to 45 verified queries, 18 of them trap cases.)*
- [x] **Phase 3** — Generation over the retrieved context, and the answer-level citation
      metrics. Baseline error rates established.
      *(Done: 84 tests, 153 answers, gold set 51 queries incl. 7 version-scoped.)*
- [x] **Phase 4** — Version/authority reranking with tunable strength. Sweep it, produce the
      tradeoff curve and the results table. Error analysis on losses.
      *(Done: 103 tests, 11-point retrieval sweep, 4-point answer sweep, version-rule ablation.)*

## What this is not

Scope guards on **the study**, so it cannot quietly regrow:

- **The measured pipeline is not a chatbot.** No conversation, no memory, no per-query tuning.
  The deliverable is a results table. (A demo entry point exists — see *Try it* below — but it
  is a thin composition of the measured modules, adds no retrieval logic of its own, and no
  driver imports it.)
- **Not an agent.** No tool use, no planning loop.
- **No vector database and no ANN index.** Brute-force cosine over a few thousand chunks is
  correct at this scale; an HNSW index here would be decoration.
- **No fine-tuning and no model training.** Off-the-shelf embeddings and one local generation
  model throughout.
- **Not a novel retrieval algorithm.** BM25, dense, and rank fusion are standard on purpose —
  the contribution is the measured tradeoff, not the components.

## Future improvements

Deliberately cut to keep this finishable. Listed in the order they would be worth adding back:

1. **Validated LLM-as-judge** for answer correctness at full coverage — build the judge, then
   measure its agreement with the human labels (Cohen's κ) before trusting it, and report where
   it is unreliable. Almost nobody validates their grader; this is the highest-value addition.
2. **Expand the golden set** to ~150 queries, enough for per-category statistical confidence.
3. **CI regression gate** — fail the build when evaluation scores drop, with the threshold
   calibrated to *measured* run-to-run variance rather than a guessed constant.
4. **`FAILURES.md`** — a categorised failure taxonomy, each case mapped to its test, including
   the cases this design explicitly cannot handle.
5. **Cross-corpus replication** on Kubernetes Enhancement Proposals (`status` × `stage` ×
   `milestone`) or IETF RFCs (a true document-level obsolescence graph), testing whether the
   finding generalises beyond one corpus.
6. **Learned reranker** instead of a hand-designed authority score.

## Stack

Python. Local and free throughout: [Ollama](https://ollama.com) with `nomic-embed-text` for
embeddings and `llama3.2` for generation. BM25 implemented directly rather than pulled in as a
dependency — it is short, and it keeps the lexical baseline inspectable.

## Running it

Phase 1 is standard library only. Phase 2 onward needs `numpy` (brute-force cosine) and a
local Ollama server for embeddings.

```bash
python3 -m venv .venv                        # project-local, so versions are pinned here
.venv/bin/python -m pip install -r requirements.txt
./scripts/fetch_corpus.sh                    # shallow-clones python/peps (gitignored)

.venv/bin/python -m unittest discover -s tests   # 57 tests, incl. harness validation
.venv/bin/python run_phase1.py                   # corpus stats + BM25 baseline

ollama serve &                               # required from Phase 2: embeddings
.venv/bin/python scripts/verify_gold.py      # re-derive every gold label from the corpus
.venv/bin/python run_phase2.py               # the ladder, retrieval level
.venv/bin/python run_phase3.py               # the ladder, answer level (153 generations)
.venv/bin/python run_phase4.py               # rung 4: strength sweep + tradeoff curve
```

`requirements.txt` carries a loose floor; `requirements-lock.txt` pins the exact versions
used to produce the reported numbers (numpy 2.5.2, Python 3.14.3, macOS/arm64). The first
`run_phase2.py` embeds all 19,763 chunks — roughly 11 minutes — then caches the vectors
under `.cache/`, so later runs start instantly.

## Try it

The four drivers only run the 51 gold queries. To ask the finished system something else:

```bash
.venv/bin/python ask.py "how should I specify the build backend for a package"

# the same question without and with authority reranking, side by side
.venv/bin/python ask.py --compare "how do I postpone the evaluation of annotations"

.venv/bin/python serve.py          # minimal web UI on http://localhost:8000
```

Compare mode is the point: it shows what a conventional RAG pipeline returns next to what this
study's intervention returns. On the annotations query the left column cites PEP 563
(`Superseded`) and the right cites PEP 649 (`Final`).

`ask.py`, `serve.py` and `speceval/pipeline.py` are **not part of the study** — nothing in them
is measured and no driver imports them.

## Status

**All four phases complete (2026-08-13).** Four rungs, one metric set, 51 verified queries,
~460 generated answers, 103 tests. Every metric derived from PEP headers — no LLM judge and no
hand-grading anywhere on the primary path.

The headline: authority-aware reranking cut answers citing superseded specifications from
15.7% to 3.9% (8 of 51 → 2; 6 fixed, 0 broken) and eliminated rank-1 retrieval of superseded
documents entirely, while *improving* Recall@10 from 0.951 to 0.980 — the recall/authority
tradeoff the study was built to measure did not exist on this corpus, for a reason the notes
work out and bound.

Optional next steps are in Future improvements below; the study answers its question as it
stands.
