"""BM25 -- the lexical baseline (rung 1 of the strategy ladder).

Implemented directly rather than pulled in as a dependency: it is about forty lines,
and the lexical baseline is the one component whose behaviour needs to be inspectable
when the results look surprising.

Okapi BM25 scores a document d against a query q as::

    score(d, q) = SUM over terms t in q of  idf(t) * tf_component(t, d)

               (N - df(t) + 0.5)
    idf(t) = ln ----------------- + 1
                 (df(t) + 0.5)

                          f(t, d) * (k1 + 1)
    tf_component(t, d) = ----------------------------------
                          f(t, d) + k1 * (1 - b + b * |d| / avgdl)

``k1`` controls how fast term-frequency saturates (a term appearing 20 times is not
20x as relevant as appearing once); ``b`` controls length normalisation, so a long
document does not win simply by containing more words.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

# Identifiers matter in this corpus -- `__future__`, `__init__`, `match` -- so the
# tokenizer keeps underscores rather than splitting on them.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """An inverted index over a fixed list of documents."""

    def __init__(self, documents: list[str], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)

        # term -> list of (doc_index, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: list[int] = []

        for doc_index, document in enumerate(documents):
            tokens = tokenize(document)
            self.doc_lengths.append(len(tokens))
            frequencies: dict[str, int] = defaultdict(int)
            for token in tokens:
                frequencies[token] += 1
            for term, frequency in frequencies.items():
                self.postings[term].append((doc_index, frequency))

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.n_docs if self.n_docs else 0.0
        )
        self.idf = {
            term: math.log((self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5) + 1.0)
            for term, posting in self.postings.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return the top_k (doc_index, score) pairs, highest score first.

        Scores are accumulated term-by-term over the postings lists, so only documents
        containing at least one query term are ever touched.
        """
        scores: dict[int, float] = defaultdict(float)

        for term in tokenize(query):
            posting = self.postings.get(term)
            if posting is None:
                continue
            idf = self.idf[term]
            for doc_index, frequency in posting:
                norm = 1.0 - self.b + self.b * (
                    self.doc_lengths[doc_index] / self.avg_doc_length
                )
                scores[doc_index] += idf * (
                    frequency * (self.k1 + 1.0) / (frequency + self.k1 * norm)
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]
