# Phase 1 — corpus, BM25 baseline, and a validated harness

What this phase built, why each decision was made, and what the numbers actually say.

---

## 1. What exists now

| File | Role |
|---|---|
| `speceval/corpus.py` | Parse `pep-NNNN.rst` RFC-822 headers into `Pep` objects |
| `speceval/chunking.py` | Split bodies on reStructuredText section boundaries |
| `speceval/bm25.py` | Okapi BM25, hand-rolled, with an inverted index |
| `speceval/metrics.py` | `recall_at_k`, `ndcg_at_k` |
| `speceval/retrievers.py` | `BM25Retriever` + the synthetic `Oracle` / `Random` |
| `speceval/evaluate.py` | The evaluation loop, per-category breakdown, latency |
| `tests/` | 34 tests, including the harness validation |
| `eval/queries_seed.json` | 15 hand-labelled seed queries |

No third-party dependencies. Everything here is Python standard library.

## 2. BM25, and why it is the first rung

BM25 scores a document against a query by summing, over every query term, "how rare is
this term in the corpus" times "how often does it appear in this document":

```
score(d, q) = SUM over t in q of  idf(t) * tf_component(t, d)

              N - df(t) + 0.5
idf(t) = ln( ---------------- + 1 )
                df(t) + 0.5

                       f(t,d) * (k1 + 1)
tf_component(t,d) = --------------------------------
                     f(t,d) + k1 * (1 - b + b*|d|/avgdl)
```

Two constants carry all the judgement:

- **`k1 = 1.5` — term-frequency saturation.** A word appearing twenty times is not twenty
  times as relevant as appearing once. Without saturation, a document that repeats a term
  wins purely by repetition.
- **`b = 0.75` — length normalisation.** Long documents contain more words and would
  otherwise win by default. `b=1` normalises fully, `b=0` not at all; 0.75 is the standard
  compromise.

`idf` is why this baseline is strong on identifiers: `walrus` appears in a handful of PEPs
so it carries a large weight, while `python` appears in all 734 and carries almost none.

**Why implement it rather than `pip install rank_bm25`:** it is forty lines, and when the
results look surprising the lexical baseline is the one component whose behaviour must be
inspectable. That is not a general argument against dependencies — dense retrieval in
Phase 2 will use a real embedding model, because reimplementing that would be pointless.

### The tokenizer is not the default one

```python
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
```

Underscores are kept, so `__future__` and `__init__` survive as single tokens. A tokenizer
that split on punctuation would shred exactly the identifiers this corpus is full of, and
would quietly weaken the lexical baseline that dense retrieval is being compared against.

## 3. Two metrics, because they measure different failures

- **Recall@10 — "did we find it?"** Set overlap between the top 10 and the labelled
  relevant set. Position inside the top 10 is irrelevant here.
- **nDCG@10 — "did we rank it well?"** A hit at rank 1 is worth `1/log2(2) = 1.0`, at rank
  2 `1/log2(3) = 0.63`, at rank 5 `1/log2(6) = 0.39`. Normalised by the best possible
  ordering.

Keeping both apart matters, and Phase 1 already proves why — see §6.

**Recall must be set overlap, never rank-by-rank.** Comparing position-by-position against
a ground-truth list scores a provably correct retriever far below 1.0, because any tie
broken differently looks like an error. `test_is_set_overlap_not_positional` pins this.

## 4. Design decisions worth remembering

**Relevance is labelled at PEP level, not chunk level.** Ground truth is "which PEPs
legitimately answer this question". Labelling every relevant *chunk* by hand would cost
several times more and would not change which strategy wins.

**A PEP inherits the rank of its best-scoring chunk.** Retrieval runs over chunks, then
collapses to distinct PEPs. The alternative — concatenating each PEP into one long
document — would let BM25's length normalisation punish long PEPs for being thorough.

**Chunks are retrieved ten deep per PEP slot** (`chunk_depth_multiplier`). Several chunks
of the same PEP routinely occupy the top positions, so retrieving only 10 chunks would
often yield fewer than 10 distinct PEPs and cap recall artificially.

**The indexed text prepends PEP title and section heading.** Without it, a query matching
a PEP's title scores badly against the body underneath, which never repeats the phrase.

## 5. Validating the harness before trusting it

A broken metric is invisible from the outside: the pipeline still runs, the numbers still
look plausible, and the ranking between strategies may even survive. So two synthetic
retrievers are run through the *real* evaluation loop:

- **`OracleRetriever`** returns exactly the ground truth → must score **1.000 / 1.000**.
- **`RandomRetriever`** returns seeded random PEPs → must score **near zero** (roughly
  10 draws from 734).

If a metric implementation cannot produce those two results, it is wrong, and nothing else
in the pipeline would have said so. On top of that, both metrics are pinned to
hand-computed values (`test_two_hits_at_ranks_one_and_three` checks
`1.5 / (1 + 1/log2(3))`), and both raise on an empty relevant set — scoring an unlabelled
query as 0.0 or 1.0 would bury a golden-set bug inside a mean.

## 6. Measured results

Corpus: **734 PEPs → 19,763 chunks** (26.9 per PEP). 519 PEPs (71%) carry
`Python-Version`; 31 carry `Superseded-By`.

**262 PEPs (36%) are non-authoritative** — Rejected 131, Withdrawn 70, Deferred 36,
Superseded 25. That is the trap surface, and it is large.

| Retriever | Recall@10 | nDCG@10 | p50 | p95 |
|---|---|---|---|---|
| Oracle | 1.000 | 1.000 | 0.00 ms | 0.00 ms |
| Random | 0.000 | 0.000 | 0.01 ms | 0.01 ms |
| **BM25** | **0.933** | **0.644** | **12.13 ms** | **15.31 ms** |

Oracle and Random are the harness checking itself. BM25 is the only real measurement.

### Finding 1 — the headroom is in ordering, not in finding

Recall 0.933 against nDCG 0.644 is a wide gap. BM25 **finds** the right PEPs and **ranks
them badly**. That is encouraging for the experiment: it means authority-aware reranking
has room to improve ordering without needing to improve retrieval, and it sharpens the
Phase 4 question — the recall cost of reranking should be small at k=10, so the tradeoff
curve will be about *nDCG and citation correctness*, not recall collapse.

### Finding 2 — the trap fires in the baseline, unprompted

**7 of the 15 seed queries return a non-authoritative PEP at rank 1:**

| Query | Rank-1 result | Correct answer |
|---|---|---|
| pattern matching in 3.9 | 642 (Rejected) | 634 at rank 5 |
| walrus operator | 622 (Superseded) | 572 at rank 4 |
| f-strings version | 536 (Withdrawn) | 498 at rank 2 |
| switch/case | 275 (Rejected) | 3103 at rank 2 |
| template literal strings | 501 (Withdrawn) | 750 at rank 2 |
| function annotations | 563 (Superseded) | 484 at rank 3 |
| revised buffer protocol | 368 (Deferred) | 3118 at rank 3 |

The clearest case is the walrus operator: the top hit is PEP 622, *Superseded* by 634 and
about pattern matching, not assignment expressions. And for the pattern-matching query
itself, 622 outranks 634 — its title is literally "Structural Pattern Matching" while the
live specification is titled "Structural Pattern Matching: Specification".

This is the phenomenon the project exists to measure, present before any LLM is involved.

### Finding 3 — disaggregation already earns its keep

| Category | n | Recall@10 | nDCG@10 |
|---|---|---|---|
| availability | 4 | 1.00 | 0.72 |
| identifier | 6 | 1.00 | 0.66 |
| **rationale** | 5 | **0.80** | **0.56** |

Rationale questions ("why was X rejected", "why structural subtyping") are the weakest on
both metrics — plausible, since they ask about *discussion* rather than naming a feature,
so there is less lexical signal to match. The aggregate 0.933/0.644 hides this entirely.

## 7. Python notes

- `@dataclass(frozen=True)` for `Pep`, `Chunk`, `Query` — immutable value objects, free
  `__eq__`/`__repr__`, and a `TypeError` on accidental mutation.
- `typing.Protocol` for `Retriever` — structural typing, so a new retriever needs no base
  class, only a matching `search` method. (Fittingly, that is PEP 544, seed query q08.)
- The walrus operator in `load_corpus`:
  `[pep for path in ... if (pep := load_pep(path))]` — assign and test in one expression,
  which is PEP 572, seed query q02.
- `collections.defaultdict(list)` for postings and `defaultdict(float)` for score
  accumulation — avoids a `key in dict` check on every term.

## 8. Next — Phase 2

Add dense retrieval (`nomic-embed-text` via Ollama, brute-force cosine over the 19,763
chunks) and hybrid rank fusion, behind the same `Retriever` protocol. That completes rungs
1–3 and the first résumé bullet. The prediction to test: dense should beat BM25 on
*rationale* queries, where lexical signal is weakest, and lose on *identifier* queries.
