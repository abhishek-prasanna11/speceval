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
retrieval recall?** The cost is the interesting half. Filtering hard on authority must eventually
starve the retriever of legitimate context.

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
| **Superseded-citation rate** | % of answers citing a PEP whose `Status` ∈ {`Rejected`, `Withdrawn`, `Superseded`, `Deferred`} |
| **Version correctness** | % of version-scoped answers valid for the Python version named in the query, checked against `Python-Version` |
| **Answer correctness (sampled)** | Hand-graded on a ~30-query subset. Reported as a **sample**, not a full metric |
| **Latency** | p50 / p95, **retrieval measured separately from end-to-end** — generation otherwise swamps every difference between the four strategies |

The first four are computed **automatically from PEP headers**. No LLM judge and no human grading
on the primary path; that is the single choice that keeps this project small.

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

**Not yet measured.** No numbers appear in this README or on a résumé until they have been
produced on this machine.

| Strategy | Recall@10 | nDCG@10 | Superseded-cite | Version-correct | Retrieval p95 |
|---|---|---|---|---|---|
| BM25 | — | — | — | — | — |
| Dense | — | — | — | — | — |
| Hybrid | — | — | — | — | — |
| Hybrid + reranking | — | — | — | — | — |

## Roadmap

- [ ] **Phase 1** — Ingest `python/peps`, parse headers, chunk, hand-rolled BM25, Recall@10 and
      nDCG@10 harness. Retrieval only, no LLM. Includes the harness validation above.
- [ ] **Phase 2** — Dense retrieval (brute-force cosine) and hybrid fusion. Rungs 1–3 measured.
- [ ] **Phase 3** — Golden set of 40–60 categorised queries, generation, and the two
      automatic citation metrics. Baseline error rates established.
- [ ] **Phase 4** — Version/authority reranking with tunable strength. Sweep it, produce the
      tradeoff curve and the results table. Error analysis on losses.

## What this is not

Scope guards, so the project cannot quietly regrow:

- **Not a chatbot.** No UI, no conversation, no memory. The deliverable is a results table.
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

## Status

Planning complete as of 2026-08-12. No code yet; Phase 1 is next.
