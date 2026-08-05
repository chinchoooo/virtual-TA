"""Experiment configuration.

An experiment is a YAML file describing one point in the RAG design space. Every stage
is a ``{name, params}`` pair resolved through a registry, so new approaches never require
touching the pipeline code. See ``configs/`` for examples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Component(BaseModel):
    """A pluggable stage: a registry name plus keyword arguments for its constructor."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class CorpusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(
        default_factory=lambda: ["course", "discourse"],
        description="Subdirectories of data/corpus: course, discourse, live_sessions.",
    )
    terms: list[str] | None = Field(
        None, description="Only use these terms, e.g. ['2025-01']. null = all terms."
    )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    chunker: Component = Field(default_factory=lambda: Component(name="paragraph"))
    embedder: Component = Field(
        default_factory=lambda: Component(name="openai", params={"model": "text-embedding-3-small"})
    )
    retriever: Component = Field(default_factory=lambda: Component(name="dense"))
    query_transform: Component = Field(default_factory=lambda: Component(name="identity"))
    reranker: Component = Field(default_factory=lambda: Component(name="none"))
    generator: Component = Field(
        default_factory=lambda: Component(name="llm", params={"model": "gpt-4o-mini"})
    )
    top_k: int = Field(5, ge=1, description="Chunks passed to the generator.")
    candidate_k: int = Field(20, ge=1, description="Chunks retrieved before reranking.")
    max_links: int = Field(3, ge=0, description="Source links returned to the user.")
    term_filter: bool = Field(
        False,
        description="Restrict retrieval to the question's term when one is given "
        "(API `term` field / eval case `term`).",
    )
    image_model: str | None = Field(
        "gpt-4o-mini",
        description="Vision model used to caption attached images for retrieval (null = ignore).",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def fingerprint(self) -> str:
        """Short stable hash identifying this exact configuration."""
        payload = json.dumps(self.model_dump(exclude={"name", "description"}), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:10]
