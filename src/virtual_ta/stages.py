"""Query transforms, rerankers and generators.

Each stage has its own registry. Constructors receive ``settings`` plus the ``params``
from the experiment config.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from .documents import Chunk
from .llm import OpenAICompatClient, image_part
from .registry import Registry
from .retrieval import ScoredChunk
from .settings import Settings
from .terms import term_label

# --------------------------------------------------------------------------- #
# Query transforms: turn the user question into the string(s) used for retrieval.
# --------------------------------------------------------------------------- #


class QueryTransform(ABC):
    @abstractmethod
    def __call__(self, question: str) -> list[str]:
        """Return one or more retrieval queries. Results are fused by the pipeline."""


QUERY_TRANSFORMS: Registry[QueryTransform] = Registry("query_transform")


@QUERY_TRANSFORMS.register("identity")
class IdentityTransform(QueryTransform):
    def __init__(self, settings: Settings | None = None) -> None:
        pass

    def __call__(self, question: str) -> list[str]:
        return [question]


@QUERY_TRANSFORMS.register("hyde")
class HyDETransform(QueryTransform):
    """Hypothetical Document Embeddings: retrieve with a generated answer (Gao et al., 2022)."""

    PROMPT = (
        "Write a short passage (under 120 words) from the IITM 'Tools in Data Science' course "
        "notes or forum that would answer this question. Be specific.\n\nQuestion: {question}"
    )

    def __init__(
        self, settings: Settings, model: str = "gpt-4o-mini", keep_original: bool = True
    ) -> None:
        self.client = OpenAICompatClient(settings)
        self.model = model
        self.keep_original = keep_original

    def __call__(self, question: str) -> list[str]:
        passage = self.client.chat(
            [{"role": "user", "content": self.PROMPT.format(question=question)}],
            model=self.model,
            temperature=0,
        )
        return [question, passage] if self.keep_original else [passage]


@QUERY_TRANSFORMS.register("multi_query")
class MultiQueryTransform(QueryTransform):
    """Ask the LLM for paraphrases and retrieve with all of them (RAG-Fusion style)."""

    PROMPT = (
        "Generate {n} different search queries that would help find course material or forum "
        "posts answering the question below. Return one query per line, no numbering.\n\n"
        "Question: {question}"
    )

    def __init__(self, settings: Settings, model: str = "gpt-4o-mini", n: int = 3) -> None:
        self.client = OpenAICompatClient(settings)
        self.model = model
        self.n = n

    def __call__(self, question: str) -> list[str]:
        text = self.client.chat(
            [{"role": "user", "content": self.PROMPT.format(n=self.n, question=question)}],
            model=self.model,
            temperature=0,
        )
        variants = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
        return [question, *variants[: self.n]]


# --------------------------------------------------------------------------- #
# Rerankers: reorder the candidate pool before it is cut to top_k.
# --------------------------------------------------------------------------- #


class Reranker(ABC):
    @abstractmethod
    def rerank(self, question: str, hits: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


RERANKERS: Registry[Reranker] = Registry("reranker")


@RERANKERS.register("none")
class NoReranker(Reranker):
    def __init__(self, settings: Settings | None = None) -> None:
        pass

    def rerank(self, question: str, hits: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        return hits[:k]


@RERANKERS.register("llm")
class LLMReranker(Reranker):
    """Listwise LLM reranking: the model scores every candidate 0-10 in a single call."""

    PROMPT = (
        "Rate how useful each passage is for answering the question, from 0 (irrelevant) to "
        '10 (directly answers it). Respond with JSON only: {{"scores": [<int>, ...]}} with '
        "exactly {n} integers in passage order.\n\nQuestion: {question}\n\n{passages}"
    )

    def __init__(
        self, settings: Settings, model: str = "gpt-4o-mini", max_chars: int = 1200
    ) -> None:
        self.client = OpenAICompatClient(settings)
        self.model = model
        self.max_chars = max_chars

    def rerank(self, question: str, hits: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not hits:
            return hits
        passages = "\n\n".join(
            f"[{i}] {h.chunk.title}\n{h.chunk.text[: self.max_chars]}" for i, h in enumerate(hits)
        )
        text = self.client.chat(
            [
                {
                    "role": "user",
                    "content": self.PROMPT.format(
                        n=len(hits), question=question, passages=passages
                    ),
                }
            ],
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            scores = [float(s) for s in json.loads(text)["scores"]]
        except (ValueError, KeyError, TypeError):
            return hits[:k]  # fall back to retrieval order on malformed output
        if len(scores) != len(hits):
            return hits[:k]
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        return [ScoredChunk(hits[i].chunk, scores[i]) for i in order[:k]]


# --------------------------------------------------------------------------- #
# Generators: produce the final answer from the question and retrieved context.
# --------------------------------------------------------------------------- #


class Generator(ABC):
    @abstractmethod
    def generate(
        self, question: str, contexts: list[ScoredChunk], image: str | None = None
    ) -> str: ...


GENERATORS: Registry[Generator] = Registry("generator")

DEFAULT_SYSTEM_PROMPT = """\
You are a teaching assistant for the IIT Madras "Tools in Data Science" (TDS) course.
Answer the student's question using ONLY the numbered context passages below, which come
from the course notes and the course Discourse forum.

Rules:
- If the context does not contain the answer, say you don't know. Never invent dates,
  scores, deadlines or policies.
- Be concise and direct. Cite passages inline like [1], [2].
- Prefer official answers from the course team over other students' posts.
- Each passage is labelled with its term (e.g. "Jan 2025 term"). Policies, deadlines and
  tools change between terms: say which term an answer applies to when it matters."""


def format_context(contexts: list[ScoredChunk]) -> str:
    def header(chunk: Chunk) -> str:
        term = f"{term_label(chunk.term)} term, " if chunk.term else ""
        return f"{chunk.title} ({term}{chunk.url})"

    return "\n\n".join(
        f"[{i}] {header(h.chunk)}\n{h.chunk.text}" for i, h in enumerate(contexts, start=1)
    )


@GENERATORS.register("llm")
class LLMGenerator(Generator):
    def __init__(
        self,
        settings: Settings,
        model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_tokens: int | None = 600,
    ) -> None:
        self.client = OpenAICompatClient(settings)
        self.model = model
        self.system_prompt = system_prompt
        self.params: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            self.params["max_tokens"] = max_tokens

    def generate(self, question: str, contexts: list[ScoredChunk], image: str | None = None) -> str:
        user_text = f"Context:\n\n{format_context(contexts)}\n\nQuestion: {question}"
        content: Any = (
            [{"type": "text", "text": user_text}, image_part(image)] if image else user_text
        )
        return self.client.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            model=self.model,
            **self.params,
        )


@GENERATORS.register("retrieval_only")
class RetrievalOnlyGenerator(Generator):
    """Skip generation entirely. Use for fast, free retrieval-only experiments."""

    def __init__(self, settings: Settings | None = None) -> None:
        pass

    def generate(self, question: str, contexts: list[ScoredChunk], image: str | None = None) -> str:
        return ""
