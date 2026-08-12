from virtual_ta.documents import Chunk
from virtual_ta.embeddings import HashingEmbedder
from virtual_ta.evaluation import retrieval_metrics
from virtual_ta.pipeline import rrf_fuse
from virtual_ta.retrieval import RETRIEVERS, ScoredChunk


def make_chunks(texts):
    return [
        Chunk(id=f"d{i}#0", doc_id=f"d{i}", source="s", title="", url=f"u{i}", text=t)
        for i, t in enumerate(texts)
    ]


CHUNKS = make_chunks(
    [
        "podman and docker containers",
        "uv runs python scripts",
        "marimo reactive notebooks",
    ]
)


def test_bm25_ranks_lexical_match_first_and_drops_zero_scores():
    r = RETRIEVERS.build("bm25")
    r.fit(CHUNKS)
    hits = r.search("how do I use podman?", k=3)
    assert [h.chunk.id for h in hits] == ["d0#0"]


def test_dense_with_hashing_embedder():
    r = RETRIEVERS.build("dense", embedder=HashingEmbedder(dim=256))
    r.fit(CHUNKS)
    assert r.search("reactive marimo", k=1)[0].chunk.id == "d2#0"


def test_hybrid_returns_k_unique_results():
    r = RETRIEVERS.build("hybrid", embedder=HashingEmbedder(dim=256))
    r.fit(CHUNKS)
    hits = r.search("python uv", k=2)
    assert hits[0].chunk.id == "d1#0"
    assert len({h.chunk.id for h in hits}) == len(hits) == 2


def test_rrf_fuse_rewards_agreement():
    a, b, c = CHUNKS
    fused = rrf_fuse(
        [[ScoredChunk(a, 1), ScoredChunk(b, 0.5)], [ScoredChunk(b, 1), ScoredChunk(c, 0.5)]], k=3
    )
    assert fused[0].chunk.id == b.id


def test_retrieval_metrics_collapse_duplicate_docs():
    m = retrieval_metrics(["x", "x", "y", "z"], expected=["z", "q"], k=3)
    assert m == {"hit@3": 1.0, "recall@3": 0.5, "mrr": 1 / 3}


def test_term_filter_restricts_every_retriever():
    chunks = [
        Chunk(id=f"d{i}#0", doc_id=f"d{i}", source="s", title="", url="", text=t, term=term)
        for i, (t, term) in enumerate(
            [("ga4 bonus marks", "2025-01"), ("ga4 bonus marks", "2025-05"), ("uv", "2025-05")]
        )
    ]
    for name in ("dense", "bm25", "hybrid"):
        r = RETRIEVERS.build(name, embedder=HashingEmbedder(dim=256))
        r.fit(chunks)
        hits = r.search("ga4 bonus", k=3, terms={"2025-05"})
        assert hits and all(h.chunk.term == "2025-05" for h in hits), name
        assert r.search("ga4 bonus", k=3, terms={"1999-01"}) == [], name
