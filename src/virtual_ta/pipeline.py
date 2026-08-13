"""The RAG pipeline: query transform -> retrieve -> (fuse) -> rerank -> generate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .chunking import CHUNKERS
from .config import ExperimentConfig
from .documents import Chunk, load_corpus
from .embeddings import EMBEDDERS
from .llm import OpenAICompatClient, image_part
from .retrieval import RETRIEVERS, Retriever, ScoredChunk
from .settings import Settings
from .stages import GENERATORS, QUERY_TRANSFORMS, RERANKERS, Generator, QueryTransform, Reranker

log = logging.getLogger(__name__)

IMAGE_CAPTION_PROMPT = (
    "A student attached this image to a question about a data science course. Describe the "
    "relevant content (text, error messages, scores, code) in under 80 words."
)


@dataclass
class Link:
    url: str
    text: str


@dataclass
class RAGResult:
    answer: str
    links: list[Link]
    contexts: list[ScoredChunk]
    queries: list[str]
    timings: dict[str, float] = field(default_factory=dict)


def rrf_fuse(result_lists: list[list[ScoredChunk]], k: int, rrf_k: int = 60) -> list[ScoredChunk]:
    if len(result_lists) == 1:
        return result_lists[0][:k]
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            chunks[hit.chunk.id] = hit.chunk
            scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1 / (rrf_k + rank + 1)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [ScoredChunk(chunks[cid], s) for cid, s in ranked]


def build_links(contexts: list[ScoredChunk], limit: int) -> list[Link]:
    links: list[Link] = []
    seen: set[str] = set()
    for hit in contexts:
        if hit.chunk.url in seen:
            continue
        seen.add(hit.chunk.url)
        snippet = " ".join(hit.chunk.text.split())
        links.append(Link(url=hit.chunk.url, text=f"{hit.chunk.title}: {snippet[:160]}"))
        if len(links) >= limit:
            break
    return links


class RAGPipeline:
    def __init__(
        self,
        config: ExperimentConfig,
        settings: Settings,
        retriever: Retriever,
        query_transform: QueryTransform,
        reranker: Reranker,
        generator: Generator,
    ) -> None:
        self.config = config
        self.settings = settings
        self.retriever = retriever
        self.query_transform = query_transform
        self.reranker = reranker
        self.generator = generator
        self._vision: OpenAICompatClient | None = None

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, settings: Settings | None = None, progress: bool = False
    ) -> RAGPipeline:
        settings = settings or Settings.from_env()
        docs = load_corpus(settings.data_dir / "corpus", config.corpus.sources, config.corpus.terms)
        if not docs:
            raise RuntimeError(
                f"No documents for sources={config.corpus.sources} terms={config.corpus.terms}"
            )
        chunker = CHUNKERS.build(config.chunker.name, **config.chunker.params)
        chunks = chunker.chunk(docs)
        log.info("Loaded %d documents -> %d chunks", len(docs), len(chunks))

        embedder = EMBEDDERS.build(
            config.embedder.name, settings=settings, **config.embedder.params
        )
        retriever = RETRIEVERS.build(
            config.retriever.name, embedder=embedder, **config.retriever.params
        )
        retriever.fit(chunks, progress=progress)

        def stage(registry, component):
            return registry.build(component.name, settings=settings, **component.params)

        return cls(
            config=config,
            settings=settings,
            retriever=retriever,
            query_transform=stage(QUERY_TRANSFORMS, config.query_transform),
            reranker=stage(RERANKERS, config.reranker),
            generator=stage(GENERATORS, config.generator),
        )

    def describe_image(self, image: str) -> str:
        if self._vision is None:
            self._vision = OpenAICompatClient(self.settings)
        assert self.config.image_model
        content = [{"type": "text", "text": IMAGE_CAPTION_PROMPT}, image_part(image)]
        return self._vision.chat(
            [{"role": "user", "content": content}], model=self.config.image_model, temperature=0
        )

    def retrieve(
        self, question: str, image: str | None = None, term: str | None = None
    ) -> tuple[list[ScoredChunk], dict]:
        """Run every stage except generation. Returns (contexts, trace)."""
        timings: dict[str, float] = {}
        t = time.perf_counter()
        query = question
        if image and self.config.image_model:
            query = f"{question}\n\n{self.describe_image(image)}"
            timings["image"] = time.perf_counter() - t
            t = time.perf_counter()

        queries = self.query_transform(query)
        terms = {term} if term and self.config.term_filter else None
        timings["query_transform"] = time.perf_counter() - t
        t = time.perf_counter()

        candidates = rrf_fuse(
            [self.retriever.search(q, self.config.candidate_k, terms) for q in queries],
            k=self.config.candidate_k,
        )
        timings["retrieve"] = time.perf_counter() - t
        t = time.perf_counter()

        contexts = self.reranker.rerank(question, candidates, self.config.top_k)
        timings["rerank"] = time.perf_counter() - t
        return contexts, {"queries": queries, "timings": timings}

    def answer(self, question: str, image: str | None = None, term: str | None = None) -> RAGResult:
        """``term`` (e.g. "2025-01") restricts retrieval when the config sets term_filter."""
        contexts, trace = self.retrieve(question, image, term)
        t = time.perf_counter()
        answer = self.generator.generate(question, contexts, image=image)
        trace["timings"]["generate"] = time.perf_counter() - t
        return RAGResult(
            answer=answer,
            links=build_links(contexts, self.config.max_links),
            contexts=contexts,
            queries=trace["queries"],
            timings=trace["timings"],
        )
