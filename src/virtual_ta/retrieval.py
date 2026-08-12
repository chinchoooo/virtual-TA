"""Retrievers. Each is fitted on the chunk list once and then answers ``search`` calls.

Register new ones with ``@RETRIEVERS.register("name")``. Constructors receive the
experiment's ``embedder`` plus any ``params`` from the config.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .documents import Chunk
from .embeddings import Embedder
from .registry import Registry


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class Retriever(ABC):
    chunks: list[Chunk]

    @abstractmethod
    def fit(self, chunks: list[Chunk], progress: bool = False) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int, terms: set[str] | None = None) -> list[ScoredChunk]:
        """Top-k chunks for ``query``; if ``terms`` is given, only chunks from those terms."""

    def _apply_terms(self, scores: np.ndarray, terms: set[str] | None) -> np.ndarray:
        if not terms:
            return scores
        mask = np.fromiter((c.term in terms for c in self.chunks), bool, len(self.chunks))
        return np.where(mask, scores, -np.inf)


RETRIEVERS: Registry[Retriever] = Registry("retriever")


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best finite scores, best first."""
    k = min(k, int(np.isfinite(scores).sum()))
    if k == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


@RETRIEVERS.register("dense")
class DenseRetriever(Retriever):
    """Cosine similarity over embeddings (exact search; fine up to ~1e5 chunks)."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.matrix: np.ndarray | None = None

    def fit(self, chunks: list[Chunk], progress: bool = False) -> None:
        self.chunks = chunks
        texts = [f"{c.title}\n\n{c.text}" for c in chunks]
        self.matrix = self.embedder.embed(texts, progress=progress)

    def search(self, query: str, k: int, terms: set[str] | None = None) -> list[ScoredChunk]:
        assert self.matrix is not None, "call fit() first"
        scores = self._apply_terms(self.matrix @ self.embedder.embed([query])[0], terms)
        return [ScoredChunk(self.chunks[i], float(scores[i])) for i in _top_k(scores, k)]


_STOPWORDS = frozenset(
    "a an and are as at be by can do for from has have how i if in is it its me my of on or "
    "so that the this to was we what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


@RETRIEVERS.register("bm25")
class BM25Retriever(Retriever):
    """Okapi BM25 over title + text."""

    def __init__(self, embedder: Embedder | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def fit(self, chunks: list[Chunk], progress: bool = False) -> None:
        self.chunks = chunks
        self.doc_tfs = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self.doc_lens = np.array([sum(tf.values()) for tf in self.doc_tfs], dtype=np.float32)
        self.avg_len = float(self.doc_lens.mean()) if len(chunks) else 0.0
        df: Counter[str] = Counter()
        for tf in self.doc_tfs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        # Inverted index: term -> (chunk indices, term frequencies)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, tf in enumerate(self.doc_tfs):
            for term, count in tf.items():
                ids, counts = postings.setdefault(term, ([], []))
                ids.append(i)
                counts.append(count)
        self.postings = {
            t: (np.array(ids), np.array(counts, dtype=np.float32))
            for t, (ids, counts) in postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_lens / max(self.avg_len, 1e-9))
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tf = self.postings[term]
            scores[ids] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[ids])
        return scores

    def search(self, query: str, k: int, terms: set[str] | None = None) -> list[ScoredChunk]:
        scores = self._apply_terms(self.scores(query), terms)
        return [
            ScoredChunk(self.chunks[i], float(scores[i]))
            for i in _top_k(scores, k)
            if scores[i] > 0
        ]


@RETRIEVERS.register("hybrid")
class HybridRetriever(Retriever):
    """Dense + BM25 fused with weighted Reciprocal Rank Fusion."""

    def __init__(
        self,
        embedder: Embedder,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = 60,
        pool: int = 50,
    ) -> None:
        self.dense = DenseRetriever(embedder)
        self.sparse = BM25Retriever()
        self.weights = (dense_weight, sparse_weight)
        self.rrf_k = rrf_k
        self.pool = pool

    def fit(self, chunks: list[Chunk], progress: bool = False) -> None:
        self.chunks = chunks
        self.dense.fit(chunks, progress=progress)
        self.sparse.fit(chunks)

    def search(self, query: str, k: int, terms: set[str] | None = None) -> list[ScoredChunk]:
        pool = max(self.pool, k)
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for weight, results in zip(
            self.weights,
            (self.dense.search(query, pool, terms), self.sparse.search(query, pool, terms)),
            strict=True,
        ):
            for rank, hit in enumerate(results):
                by_id[hit.chunk.id] = hit.chunk
                fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + weight / (
                    self.rrf_k + rank + 1
                )
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [ScoredChunk(by_id[cid], score) for cid, score in ranked]
