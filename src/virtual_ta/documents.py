"""Corpus loading.

The corpus is ``data/corpus/<source>/<term>/**/*.md``. Each file may start with YAML front
matter (``title``, ``url``, ``term``, ...) written by the ingest step; anything missing is
inferred from the path and the first markdown heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .terms import is_term

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
# Navigation-only files that carry no answerable content.
_SKIP = {"_sidebar.md"}


@dataclass(frozen=True)
class Document:
    id: str  # "<source>/<term>/<path>", e.g. "course/2025-01/docker"
    source: str
    title: str
    url: str
    text: str
    term: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc_id>#<index>"
    doc_id: str
    source: str
    title: str
    url: str
    text: str
    term: str | None = None
    metadata: dict = field(default_factory=dict, compare=False, hash=False)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    return (meta, text[match.end() :]) if isinstance(meta, dict) else ({}, text)


def write_markdown(path: Path, body: str, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=1000)
    path.write_text(f"---\n{front}---\n\n{body.strip()}\n", encoding="utf-8", newline="\n")


def _first_heading(text: str, fallback: str) -> str:
    match = re.search(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def load_source(directory: Path, source: str) -> list[Document]:
    docs = []
    for path in sorted(directory.rglob("*.md")):
        if path.name in _SKIP:
            continue
        meta, text = split_front_matter(path.read_text(encoding="utf-8"))
        if meta.get("draft") or not text.strip():
            continue
        rel = path.relative_to(directory).with_suffix("")
        top = rel.parts[0] if len(rel.parts) > 1 else ""
        term = meta.get("term") or (top if is_term(top) else None)
        docs.append(
            Document(
                id=f"{source}/{rel.as_posix()}",
                source=source,
                title=str(meta.get("title") or _first_heading(text, rel.name.replace("-", " "))),
                url=str(meta.get("url") or ""),
                text=text,
                term=str(term) if term else None,
                metadata={k: v for k, v in meta.items() if k not in {"title", "url", "term"}},
            )
        )
    return docs


def load_corpus(
    corpus_dir: Path, sources: list[str], terms: list[str] | None = None
) -> list[Document]:
    """Load the given sources, optionally keeping only documents from ``terms``."""
    docs: list[Document] = []
    for source in sources:
        source_dir = corpus_dir / source
        if not source_dir.is_dir():
            available = sorted(p.name for p in corpus_dir.iterdir() if p.is_dir())
            raise FileNotFoundError(f"Corpus source '{source}' not found. Available: {available}")
        docs.extend(load_source(source_dir, source))
    if terms is not None:
        wanted = {str(t) for t in terms}
        docs = [d for d in docs if d.term in wanted]
    return docs
