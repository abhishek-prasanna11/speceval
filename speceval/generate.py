"""Answer generation over retrieved context, via a local Ollama server.

Two decisions here shape every Phase 3 number.

**Temperature 0 with a fixed seed.** Verified byte-identical across repeated runs, so
run-to-run variance is zero and differences between retrieval strategies are attributable
to retrieval. This is what makes the deferred variance-calibration item unnecessary for
these results -- and it stops being true the moment sampling is turned on.

**The context deliberately omits each PEP's status.** Showing `Status: Superseded` would let
the generator route around a bad retrieval, which is a *prompt-level* intervention and would
confound the comparison between retrieval strategies. The generator is held constant and
uninformed on purpose; making it authority-aware is a separate experiment, noted in the
README as future work.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk

MODEL = "llama3.2"
HOST = "http://localhost:11434"
SEED = 7
TEMPERATURE = 0.0
MAX_TOKENS = 220

# Matches "PEP 634", "PEP-634", "pep634". Four digits max: PEP numbers top out in the
# thousands, and a looser pattern would capture years and version strings.
_CITATION_RE = re.compile(r"\bPEP[\s\-]?(\d{1,4})\b", re.IGNORECASE)

PROMPT_TEMPLATE = """You are answering questions about Python using only the excerpts below.

Rules:
- Answer in at most three sentences.
- Cite every PEP you rely on, in the form "PEP 634".
- If the excerpts do not settle the question, say so.

Excerpts:
{context}

Question: {question}

Answer:"""


class GenerationError(RuntimeError):
    pass


def build_context(chunks: list[Chunk], max_chars: int = 6000) -> str:
    """Format retrieved chunks for the prompt.

    Includes PEP number, title and section so the model can cite precisely -- and nothing
    about status or version, so the generator cannot compensate for bad retrieval.
    """
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        block = (
            f"[PEP {chunk.pep_number}] {chunk.pep_title}"
            f"{' -- ' + chunk.section if chunk.section else ''}\n{chunk.text}"
        )
        if used + len(block) > max_chars:
            # Never return an empty context: if the first block alone exceeds the budget,
            # truncate it instead of dropping it. An empty context would send the model to
            # answer from parametric memory with nothing to cite, and the resulting numbers
            # would look like a retrieval failure rather than a prompt-construction bug.
            if not parts:
                parts.append(block[:max_chars])
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[Chunk], max_chars: int = 6000) -> str:
    return PROMPT_TEMPLATE.format(
        context=build_context(chunks, max_chars=max_chars), question=question
    )


def extract_citations(answer: str, known: set[int] | None = None) -> list[int]:
    """PEP numbers cited in an answer, in order of first appearance, deduplicated.

    When `known` is given, numbers outside it are dropped: the model occasionally invents a
    plausible-looking PEP number, and counting those as citations would corrupt the metrics.
    Whether it invents them at all is measured separately as the hallucination rate.
    """
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if known is not None and number not in known:
            continue
        if number not in seen:
            seen.append(number)
    return seen


def extract_all_cited_numbers(answer: str) -> list[int]:
    """Every PEP-shaped number in an answer, including ones absent from the corpus."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


@dataclass
class OllamaGenerator:
    model: str = MODEL
    host: str = HOST
    seed: int = SEED
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    timeout: float = 300.0
    retries: int = 3

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_predict": self.max_tokens,
                },
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)["response"]
            except (urllib.error.URLError, TimeoutError, KeyError) as error:
                last_error = error
                if attempt < self.retries - 1:
                    time.sleep(1.0 + attempt)
        raise GenerationError(
            f"generation failed after {self.retries} attempts against {self.host}: "
            f"{last_error}. Is `ollama serve` running?"
        ) from last_error

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


class CachedGenerator:
    """Wraps a generator with a disk cache keyed on model, options and prompt.

    Generation is the slow step and Phase 4 will re-run the whole grid repeatedly while only
    the reranker changes. Caching makes re-analysis free and keeps the numbers stable across
    runs of the driver.
    """

    def __init__(
        self,
        generator: OllamaGenerator | None = None,
        cache_dir: Path | str = ".cache/answers",
    ) -> None:
        self.generator = generator or OllamaGenerator()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self.generator.model

    def generate(self, prompt: str) -> str:
        key = self.generator.cache_key(prompt)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["answer"]
        answer = self.generator.generate(prompt)
        self.misses += 1
        path.write_text(
            json.dumps({"answer": answer, "prompt": prompt}, indent=2), encoding="utf-8"
        )
        return answer
