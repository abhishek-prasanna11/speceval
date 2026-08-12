"""Embeddings via a local Ollama server, with a disk cache.

Unlike BM25, this is not reimplemented: an embedding model is exactly the kind of
component where writing your own would teach nothing relevant to this study. What matters
here is that the calls are batched, cached, and that the cache invalidates when anything
that affects the vectors changes.

`nomic-embed-text` is an *asymmetric* model: it expects documents and queries to be
prefixed differently, so a query embedding lands near documents that answer it rather than
near documents that merely resemble it. Getting these prefixes wrong silently degrades
dense retrieval, which would then be blamed on dense retrieval rather than on the prefix.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

MODEL = "nomic-embed-text"
HOST = "http://localhost:11434"
BATCH_SIZE = 64
DIM = 768

# Documented prefixes for the nomic-embed-text family.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(
        self,
        model: str = MODEL,
        host: str = HOST,
        batch_size: int = BATCH_SIZE,
        timeout: float = 300.0,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
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
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        if progress:
            print(file=sys.stderr)

        return np.asarray(vectors, dtype=np.float32)


def normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so that a dot product is a cosine similarity.

    Normalising once at build time turns every later query into a single matrix-vector
    product, instead of recomputing magnitudes on each search.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against a zero vector, which would otherwise produce NaN and silently poison
    # every similarity score computed against it.
    norms[norms == 0] = 1.0
    return matrix / norms


def cache_key(texts: list[str], model: str, prefix: str) -> str:
    """Fingerprint everything that affects the vectors.

    Model, prefix and the exact text of every chunk are all inputs -- so changing the
    chunking configuration or swapping the model produces a different key and the cache
    is rebuilt rather than silently reused.
    """
    digest = hashlib.sha256()
    digest.update(model.encode())
    digest.update(prefix.encode())
    digest.update(str(len(texts)).encode())
    for text in texts:
        digest.update(hashlib.sha256(text.encode()).digest())
    return digest.hexdigest()[:16]


def embed_cached(
    texts: list[str],
    embedder: OllamaEmbedder | None = None,
    prefix: str = DOC_PREFIX,
    cache_dir: Path | str = ".cache",
    progress: bool = True,
) -> np.ndarray:
    """Embed texts, reusing a cached array when the fingerprint matches."""
    embedder = embedder or OllamaEmbedder()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = cache_key(texts, embedder.model, prefix)
    vectors_path = cache_dir / f"embeddings-{key}.npy"
    meta_path = cache_dir / f"embeddings-{key}.json"

    if vectors_path.exists():
        vectors = np.load(vectors_path)
        if vectors.shape[0] == len(texts):
            if progress:
                print(f"  cache hit: {vectors_path} {vectors.shape}", file=sys.stderr)
            return vectors
        # Shape disagrees with the key, which should be impossible -- rebuild rather than
        # trust it.
        print(
            f"  cache at {vectors_path} has {vectors.shape[0]} rows, expected "
            f"{len(texts)} -- rebuilding",
            file=sys.stderr,
        )

    if progress:
        print(
            f"  embedding {len(texts)} texts with {embedder.model} "
            f"(one-time, then cached)",
            file=sys.stderr,
        )
    vectors = embedder.embed(texts, prefix=prefix, progress=progress)
    np.save(vectors_path, vectors)
    meta_path.write_text(
        json.dumps(
            {
                "model": embedder.model,
                "prefix": prefix,
                "n_texts": len(texts),
                "dim": int(vectors.shape[1]),
                "key": key,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return vectors
