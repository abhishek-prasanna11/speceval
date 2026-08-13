# Chapter 8 — Talking to a Local Model Server

## Learning Objectives

- Explain what a local model server is and why the project uses one instead of a hosted API.
- Explain **batching** and why it matters for embedding throughput.
- Explain why a language model is normally nondeterministic, and exactly what makes it deterministic.
- Explain what **temperature** and **seed** control.
- Explain the two disk caches in this project, what each is keyed on, and why the keys are shaped
  that way.
- Explain the retry policy and why it is bounded.

## Motivation

Two stages in this project need a neural model: turning text into a vector (chapter 4) and turning
retrieved chunks into prose (chapter 19). Neither is implemented here. Both are external processes
reached over HTTP.

That introduces problems no pure-Python component has: the network can fail, calls are slow enough
that repeating them is unaffordable, and — most importantly for a study — **a language model's output
is random by default.** A measurement harness whose subject changes its answer between runs cannot
report a difference between two systems, because it cannot distinguish a real difference from
sampling noise.

This chapter is about making an unreliable, slow, nondeterministic dependency into something you can
measure against.

## First Principles

### What a local model server is

A model server is a long-running process that holds model weights in memory and answers HTTP requests.
`speceval` uses **Ollama**, running on `localhost:11434`.

```
   your Python process                    ollama serve
   +------------------+                  +----------------------+
   |                  |   HTTP POST      |  llama3.2 (2 GB)     |
   |  embed / generate| ---------------> |  nomic-embed-text    |
   |                  | <--------------- |  (274 MB)            |
   +------------------+   JSON            +----------------------+
                                          weights stay resident
```

Why a server rather than loading the model in-process? Because loading 2 GB of weights takes seconds
and would happen on every run. The server pays that cost once.

Why local rather than a hosted API? Three reasons, and they are worth separating:

**Cost.** This project generated roughly 460 answers across its phases. At hosted per-token prices
that is real money for a study you will re-run while developing.

**Reproducibility.** A hosted model is a moving target — providers update model versions without
notice. A local model file does not change unless you change it. For a study whose numbers must
reproduce, that matters more than quality.

**Determinism.** Covered below, and it is the decisive reason. Hosted APIs generally do not expose the
seed controls needed to make output repeatable.

### Two endpoints

```
POST /api/embed      {"model": ..., "input": [text, text, ...]}  -> {"embeddings": [[...], ...]}
POST /api/generate   {"model": ..., "prompt": ...}               -> {"response": "..."}
```

Verified against the running server while writing this book:

```
{"version":"0.9.0"}
n= 3 dim= 768 keys= ['model', 'embeddings', 'total_duration', 'load_duration', ...]
```

### Batching

`/api/embed` accepts a *list*. That matters enormously.

Each HTTP request has fixed overhead: connection, JSON parsing, model scheduling. Embedding 19,763
chunks one per request means paying that 19,763 times. Sending 64 per request means paying it 309
times.

Measured on this machine: **64 chunks of realistic length in 2.06 seconds**, about 32 ms per chunk
including all overhead. Extrapolating gave an estimate of ~10.6 minutes for the corpus; the actual run
took roughly **17 minutes at ~19 chunks/second**, so throughput degraded somewhat over a sustained
run — worth knowing, and a reminder that a short benchmark can overestimate sustained throughput.

`speceval/embed.py`:

```python
BATCH_SIZE = 64

    def embed(self, texts: list[str], prefix: str = "", progress: bool = False) -> np.ndarray:
        """Embed texts in batches. Returns an (n, dim) float32 array."""
        vectors: list[list[float]] = []
        total = len(texts)
        started = time.perf_counter()

        for start in range(0, total, self.batch_size):
            batch = [prefix + text for text in texts[start : start + self.batch_size]]
            vectors.extend(self._post(batch))
            if progress:
                done = min(start + self.batch_size, total)
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed else 0.0
                remaining = (total - done) / rate if rate else 0.0
                print(
                    f"\r  embedded {done}/{total} "
                    f"({done / total:.0%}, {rate:.0f}/s, ~{remaining / 60:.1f} min left)",
                    ...
                )
```

The progress reporting is not cosmetic. A silent seventeen-minute operation is indistinguishable from
a hang, and the natural response to an apparent hang is to kill it — losing the work. An ETA converts
"is this broken?" into "this has four minutes left."

Note the prefix is applied here, per batch, rather than by the caller. One place, so it cannot be
forgotten at a call site (chapter 4 explained why omitting it is silently damaging).

## Deep Explanation: determinism

This is the most consequential subject in the chapter.

### Why a language model is random by default

A language model does not output text. It outputs, for each position, a **probability distribution
over the whole vocabulary**:

```
   after "f-strings were added in Python 3."
        "6"   -> 0.71
        "7"   -> 0.12
        "8"   -> 0.06
        "12"  -> 0.03
        ... thousands more, each tiny
```

Something must turn that distribution into one token. **Sampling** draws randomly according to the
probabilities: usually "6", sometimes "7". This produces varied, natural-sounding text, which is what
you want in a chat product.

It is exactly what you do not want in a measurement harness. Run your evaluation twice and get two
different superseded-citation rates, with no way to tell whether a change you made helped or the dice
fell differently.

### Temperature

**Temperature** reshapes the distribution before sampling. It divides the model's raw scores (logits)
by a constant `T` before converting to probabilities:

```
   T = 1.0   ->  distribution as the model produced it
   T > 1.0   ->  flatter; unlikely tokens become more likely; more "creative", more erratic
   T < 1.0   ->  sharper; likely tokens dominate further
   T = 0.0   ->  degenerate: always take the single most likely token
```

At `T = 0` there is no randomness left. This is called **greedy decoding** — always take the
argmax — and it is what makes generation repeatable.

### Seed

Even at low but nonzero temperature you would want repeatability, so servers also accept a **seed** for
the pseudorandom generator. Fix the seed and the same "random" draws happen in the same order.

`speceval` sets both:

```python
SEED = 7
TEMPERATURE = 0.0
MAX_TOKENS = 220
```

Belt and braces: temperature 0 should be sufficient on its own, but the seed costs nothing and guards
against any residual nondeterminism in the sampling path.

### Verifying it rather than assuming it

The important part is that this was **checked, not trusted**. Three identical requests:

```
run1  3.52s  sha=617a6106376c  67 chars
run2  0.59s  sha=617a6106376c  67 chars
run3  0.57s  sha=617a6106376c  67 chars

identical across 3 runs: True
```

Byte-identical output, confirmed by hash. (The first run is slower because the model is being loaded
into memory; subsequent calls hit a warm server.)

This one verification is what licenses every answer-level number in Part V. From the module docstring:

```python
"""**Temperature 0 with a fixed seed.** Verified byte-identical across repeated runs, so
run-to-run variance is zero and differences between retrieval strategies are attributable
to retrieval. This is what makes the deferred variance-calibration item unnecessary for
these results -- and it stops being true the moment sampling is enabled.
"""
```

The last clause is the honest part. This is a property of *this configuration*, not of the system. Turn
sampling on and every reported difference would need a variance estimate before it meant anything.

### `MAX_TOKENS = 220`

A cap on output length, and it does three things at once: keeps generation fast (each token is a full
forward pass), keeps answers short enough to evaluate for citations, and bounds the worst case so one
runaway response cannot stall a 153-answer run.

The prompt asks for at most three sentences; the cap enforces it even if the model ignores the
instruction.

## Deep Explanation: the two caches

Both models are slow. Both results are cached to disk. The two caches are keyed differently, for
reasons specific to each.

### The embedding cache

Covered in chapter 4; the key fingerprints model, prefix, and the exact text of every chunk:

```python
def cache_key(texts: list[str], model: str, prefix: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode())
    digest.update(prefix.encode())
    digest.update(str(len(texts)).encode())
    for text in texts:
        digest.update(hashlib.sha256(text.encode()).digest())
    return digest.hexdigest()[:16]
```

It also writes a metadata sidecar, so a cache directory is readable months later:

```python
    meta_path.write_text(
        json.dumps({"model": ..., "prefix": ..., "n_texts": ..., "dim": ..., "key": key}, indent=2),
        encoding="utf-8",
    )
```

And it defends against a shape mismatch even on a key hit:

```python
    if vectors_path.exists():
        vectors = np.load(vectors_path)
        if vectors.shape[0] == len(texts):
            ...
            return vectors
        # Shape disagrees with the key, which should be impossible -- rebuild rather than
        # trust it.
```

That branch is unreachable if the key logic is correct. It exists because the consequence of being
wrong is silent misattribution of every score, and seventeen minutes of recomputation is a cheap
insurance premium.

### The answer cache

Generation is keyed per prompt rather than per corpus, because prompts vary per query *and* per
retrieval strategy:

```python
    def cache_key(self, prompt: str) -> str:
        digest = hashlib.sha256()
        for part in (
            self.model,
            str(self.seed),
            str(self.temperature),
            str(self.max_tokens),
            prompt,
        ):
            digest.update(part.encode())
        return digest.hexdigest()[:20]
```

Everything that could change the output is in the key: model, seed, temperature, token cap, prompt.
Change any and you get a new key.

```python
class CachedGenerator:
    """Wraps a generator with a disk cache keyed on model, options and prompt.

    Generation is the slow step and Phase 4 will re-run the whole grid repeatedly while only
    the reranker changes. Caching makes re-analysis free and keeps the numbers stable across
    runs of the driver.
    """
```

That second sentence describes a real economy. Chapter 20 sweeps the reranker across four strengths
plus an ablation — five configurations × 51 queries = 255 generations. But many reranker settings
produce the *same top-5 chunks*, therefore the same prompt, therefore a cache hit. The observed hit
counts across the sweep were 1, 27, 52, 97, 143 — by the last configuration, 143 of the prompts had
already been seen. Without the cache the sweep would have cost roughly 34 minutes; with it, far less.

Hit and miss counters are exposed so the driver can report them, which is how those numbers are known:

```python
        self.hits = 0
        self.misses = 0
```

### One number the cache makes misleading

A subtlety worth naming, because it caught this book's own verification pass. End-to-end latency
measured on a **warm** cache is meaningless — it measures a disk read, not generation. Re-running
`run_phase3.py` after the cache is populated reports p95 of **1 ms** instead of the true ~13.8 seconds.

The README now carries that warning explicitly. The general rule: **a cache changes what your timer
measures.** If you report latency, say whether the cache was cold.

## Deep Explanation: failure handling

The server can be down, slow, or return something unexpected. `_post` handles all three:

```python
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                embeddings = body.get("embeddings")
                if not embeddings or len(embeddings) != len(texts):
                    raise EmbeddingError(
                        f"expected {len(texts)} embeddings, got "
                        f"{len(embeddings) if embeddings else 0}"
                    )
                return embeddings
            except (urllib.error.URLError, TimeoutError, EmbeddingError) as error:
                last_error = error
                if attempt < self.retries - 1:
                    time.sleep(1.0 + attempt)
        raise EmbeddingError(
            f"embedding failed after {self.retries} attempts against {self.host}: "
            f"{last_error}. Is `ollama serve` running?"
        ) from last_error
```

Four properties:

**Bounded retries** — three attempts, not indefinite. An unbounded retry loop against a server that
is down is a hang, and a hang in a long batch job is worse than a failure because it wastes wall-clock
time silently.

**Linear backoff** — `1.0 + attempt` seconds. Enough to ride out a transient stall without
compounding delay.

**A count check, not just an HTTP check.** `len(embeddings) != len(texts)` catches a response that
succeeded at the protocol level but returned the wrong number of vectors. Without it, a short response
would misalign every subsequent chunk's vector — silent, and catastrophic.

**An actionable message.** `Is ollama serve running?` is the actual cause the overwhelming majority of
the time. An error message that names the likely fix saves more time than one that is merely accurate.

Note also `from last_error`: the original exception is chained, so the traceback retains the
underlying cause rather than replacing it.

Only `stdlib` is used — `urllib.request`, not `requests`. For four call sites, a dependency is not
worth the supply-chain surface, and `urllib` is entirely adequate.

## Systems Perspective

Where time goes, all measured on this machine:

| Operation | Cost | Note |
|---|---|---|
| Corpus embedding | ~17 min once | 19,763 chunks, ~19/s sustained |
| Single query embedding | tens of ms | dominated by round trip |
| One generation (cold) | ~8 s | 51 queries ≈ 7 min per configuration |
| One generation (cached) | ~1 ms | disk read |
| Model load (first call) | ~3 s | then resident |

The 8-second generation figure is why Part V's answer-level sweeps were run as background jobs and why
the cache exists at all.

## Common Mistakes

**Leaving temperature at its default in an evaluation.** Then every difference you measure is
confounded with sampling noise, and you will not know it.

**Assuming determinism instead of verifying it.** Hash the output of three identical calls. It takes a
minute.

**One request per item.** Batching cut per-item cost by roughly an order of magnitude here.

**Unbounded retries.** Converts a clear failure into a silent hang.

**Trusting HTTP 200.** Check the response *shape*. A successful call returning the wrong number of
vectors is worse than an error.

**A cache key missing an input.** Anything that affects the output must be in the key, or you will
eventually serve stale results with no error.

**Reporting latency from a warm cache.** You measured your filesystem.

## Interview Insight

> **"How do you make LLM output reproducible?"**

Temperature 0 (greedy decoding — always take the most likely token) plus a fixed seed, then *verify*
by hashing the output of repeated identical calls. In this project three runs were byte-identical,
which is what licenses treating any difference between retrieval strategies as real rather than as
sampling noise.

The point worth adding: this is a property of the configuration, not the model. Enable sampling and you
need a variance estimate before any difference means anything — which is why "we improved the metric by
2%" is not a claim unless you also state your run-to-run variance.

> **"Why cache, and what do you key on?"**

Because generation is seconds and embedding a corpus is minutes, and a sweep re-runs both. Key on
*everything that affects the output* — for embeddings: model, prefix, and the exact text of every
input; for generation: model, seed, temperature, token cap, and the full prompt. A key missing any
input eventually serves results that do not correspond to the current inputs, and nothing raises.

And know what the cache does to your measurements: end-to-end latency on a warm cache is a disk read.

> **"Local model or hosted API?"**

For a measurement study, local — because a hosted model version can change under you, and hosted APIs
generally do not expose the seed controls needed for determinism. For production, usually hosted, for
capability and operational reasons. The deciding question is whether you need reproducibility or
capability.

## Debugging Tip

When the model layer misbehaves, curl the endpoint directly before touching Python:

```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model":"nomic-embed-text","input":["hello","world"]}' | head -c 200
```

That distinguishes three failures immediately: server down (connection refused), model not pulled
(error naming the model), and a client-side bug (curl works, Python does not).

## Summary

- A local model server keeps weights resident so load cost is paid once; local is chosen for cost,
  reproducibility and — decisively — determinism.
- Batching is the difference between 309 requests and 19,763. Measured ~32 ms per chunk in a short
  batch; ~19/s sustained, so short benchmarks overestimate.
- Language models sample from a distribution and are therefore random by default. Temperature 0
  (greedy) plus a fixed seed removes it, and this project verified byte-identical output across three
  runs rather than assuming it.
- Two caches, both keyed on everything that affects their output; both are the reason a five-point
  sweep is affordable.
- A warm cache makes latency measurements meaningless; say whether the cache was cold.
- Retries are bounded with linear backoff, the response *shape* is validated, and errors name the
  likely fix.

## Key Takeaways

1. Temperature 0 plus a fixed seed, then verify by hashing repeated calls.
2. Batch anything that accepts a list.
3. A cache key must cover every input that changes the output — no exceptions.
4. Bounded retries. A hang is worse than a failure.

## Why the Next Chapter Exists

Chapter 4 described dense retrieval as a matrix multiply and moved on. That multiply is where the
project's remaining performance sits, and it is also where a genuine bug was found — not in this code,
but in the numerical library underneath it.

Chapter 9 covers NumPy for brute-force search: what makes the operation fast, how to select a top-k
without sorting everything, and a worked example of investigating a suspicious floating-point warning
rather than suppressing it.
