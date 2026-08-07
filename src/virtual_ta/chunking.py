"""Chunking strategies. Register new ones with ``@CHUNKERS.register("name")``.

Lengths are measured in whitespace-separated words, a cheap and dependency-free proxy
for tokens (~0.75 words per token for English prose).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .documents import Chunk, Document
from .registry import Registry


class Chunker(ABC):
    @abstractmethod
    def split(self, text: str) -> list[str]: ...

    def chunk(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in docs:
            pieces = [p for p in (s.strip() for s in self.split(doc.text)) if p]
            for i, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{i}",
                        doc_id=doc.id,
                        source=doc.source,
                        title=doc.title,
                        url=doc.url,
                        text=piece,
                        term=doc.term,
                    )
                )
        return chunks


CHUNKERS: Registry[Chunker] = Registry("chunker")


def _words(text: str) -> int:
    return len(text.split())


def _pack(units: list[str], max_words: int, sep: str) -> list[str]:
    """Greedily pack units into chunks of at most ``max_words`` (oversized units stand alone)."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        n = _words(unit)
        if current and size + n > max_words:
            chunks.append(sep.join(current))
            current, size = [], 0
        current.append(unit)
        size += n
    if current:
        chunks.append(sep.join(current))
    return chunks


@CHUNKERS.register("document")
class DocumentChunker(Chunker):
    """No splitting: one chunk per document (the original v0 behaviour)."""

    def split(self, text: str) -> list[str]:
        return [text]


@CHUNKERS.register("paragraph")
class ParagraphChunker(Chunker):
    """Pack blank-line separated paragraphs up to ``max_words``."""

    def __init__(self, max_words: int = 300) -> None:
        self.max_words = max_words

    def split(self, text: str) -> list[str]:
        return _pack(re.split(r"\n\s*\n", text), self.max_words, "\n\n")


@CHUNKERS.register("sliding_window")
class SlidingWindowChunker(Chunker):
    """Fixed-size word windows with overlap. Ignores document structure."""

    def __init__(self, window: int = 200, overlap: int = 50) -> None:
        if not 0 <= overlap < window:
            raise ValueError("overlap must be in [0, window)")
        self.window = window
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        words = text.split()
        step = self.window - self.overlap
        return [
            " ".join(words[i : i + self.window])
            for i in range(0, max(len(words) - self.overlap, 1), step)
        ]


@CHUNKERS.register("markdown")
class MarkdownSectionChunker(Chunker):
    """Split on markdown headings (and Discourse ``---`` post separators), then pack.

    Each chunk is prefixed with its heading so it stays self-describing.
    """

    _boundary = re.compile(r"^(?=#{1,6}\s)|^-{3,}\s*$", flags=re.MULTILINE)

    def __init__(self, max_words: int = 300, min_words: int = 20) -> None:
        self.max_words = max_words
        self.min_words = min_words

    def split(self, text: str) -> list[str]:
        sections = [s.strip() for s in self._boundary.split(text) if s and s.strip()]
        # Merge tiny sections into their successor so headings aren't orphaned.
        merged: list[str] = []
        carry = ""
        for section in sections:
            section = f"{carry}\n\n{section}".strip() if carry else section
            if _words(section) < self.min_words:
                carry = section
            else:
                merged.append(section)
                carry = ""
        if carry:
            if merged:
                merged[-1] = f"{merged[-1]}\n\n{carry}"
            else:
                merged.append(carry)

        out: list[str] = []
        for section in merged:
            if _words(section) <= self.max_words:
                out.append(section)
                continue
            heading, _, body = section.partition("\n")
            prefix = heading if heading.startswith("#") else ""
            body = body if prefix else section
            for piece in _pack(re.split(r"\n\s*\n", body), self.max_words, "\n\n"):
                out.append(f"{prefix}\n\n{piece}" if prefix else piece)
        return out
