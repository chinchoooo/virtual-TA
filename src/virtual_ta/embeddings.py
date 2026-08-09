"""Embedding models. All embedders return L2-normalised float32 matrices.

Remote embeddings are cached on disk keyed by ``(model, sha256(text))``, so re-running
an experiment with a different retriever or reranker costs nothing, and changing the
chunker only pays for chunks that are actually new.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .llm import OpenAICompatClient
from .registry import Registry
from .settings import Settings


class Embedder(ABC):
    #: Identifier used in cache keys and run metadata.
    id: str

    @abstractmethod
    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray: ...


EMBEDDERS: Registry[Embedder] = Registry("embedder")


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


class EmbeddingCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            " model TEXT, text_hash TEXT, vector BLOB, PRIMARY KEY (model, text_hash))"
        )

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> dict[str, np.ndarray]:
        hashes = list({self.key(t) for t in texts})
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for i in range(0, len(hashes), 500):
                batch = hashes[i : i + 500]
                rows = self._con.execute(
                    f"SELECT text_hash, vector FROM embeddings WHERE model = ? "
                    f"AND text_hash IN ({','.join('?' * len(batch))})",
                    [model, *batch],
                )
                for h, blob in rows:
                    found[h] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, model: str, items: list[tuple[str, np.ndarray]]) -> None:
        with self._lock, self._con:
            self._con.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                [(model, self.key(t), v.astype(np.float32).tobytes()) for t, v in items],
            )


@EMBEDDERS.register("openai")
class OpenAIEmbedder(Embedder):
    """Any OpenAI-compatible ``/embeddings`` endpoint (configured via API_BASE_URL)."""

    def __init__(
        self,
        settings: Settings,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        max_chars: int = 24_000,
    ) -> None:
        self.model = model
        self.id = f"openai:{model}"
        self.batch_size = batch_size
        self.max_chars = max_chars  # stay under the 8k-token input limit
        self._settings = settings
        self._client: OpenAICompatClient | None = None
        self.cache = EmbeddingCache(settings.cache_dir / "embeddings.sqlite")

    @property
    def client(self) -> OpenAICompatClient:
        if self._client is None:
            self._client = OpenAICompatClient(self._settings)
        return self._client

    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray:
        texts = [t[: self.max_chars] for t in texts]
        cached = self.cache.get_many(self.id, texts)
        missing = list(dict.fromkeys(t for t in texts if EmbeddingCache.key(t) not in cached))
        batches = range(0, len(missing), self.batch_size)
        for i in tqdm(batches, desc=f"Embedding ({self.model})", disable=not progress):
            batch = missing[i : i + self.batch_size]
            vectors = [
                np.asarray(v, dtype=np.float32) for v in self.client.embed(batch, self.model)
            ]
            self.cache.put_many(self.id, list(zip(batch, vectors, strict=True)))
            cached.update((EmbeddingCache.key(t), v) for t, v in zip(batch, vectors, strict=True))
        return normalize(np.stack([cached[EmbeddingCache.key(t)] for t in texts]))


@EMBEDDERS.register("hashing")
class HashingEmbedder(Embedder):
    """Offline bag-of-words hashing embedder.

    Free, deterministic and dependency-free: useful as a lexical baseline and for tests.
    """

    def __init__(self, settings: Settings | None = None, dim: int = 1024) -> None:
        self.dim = dim
        self.id = f"hashing:{dim}"

    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
                matrix[row, h % self.dim] += 1.0 if (h >> 63) else -1.0
        return normalize(matrix)
