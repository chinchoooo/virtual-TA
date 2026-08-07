import pytest

from virtual_ta.chunking import CHUNKERS, SlidingWindowChunker
from virtual_ta.documents import Document


def words(n: int, word: str = "w") -> str:
    return " ".join([word] * n)


def test_paragraph_packs_under_limit():
    text = "\n\n".join(words(40) for _ in range(5))
    pieces = CHUNKERS.build("paragraph", max_words=100).split(text)
    assert [len(p.split()) for p in pieces] == [80, 80, 40]


def test_sliding_window_overlap_covers_all_words():
    text = " ".join(str(i) for i in range(250))
    pieces = SlidingWindowChunker(window=100, overlap=20).split(text)
    assert pieces[0].split()[-20:] == pieces[1].split()[:20]
    assert pieces[-1].split()[-1] == "249"


def test_sliding_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        SlidingWindowChunker(window=10, overlap=10)


def test_markdown_splits_on_headings_and_keeps_heading_prefix():
    text = f"# Title\n\n{words(30, 'a')}\n\n## Second\n\n{words(30, 'b')}"
    pieces = CHUNKERS.build("markdown", max_words=100, min_words=5).split(text)
    assert len(pieces) == 2
    assert pieces[1].startswith("## Second")


def test_markdown_splits_oversized_section_with_heading_prefix():
    text = "## Big\n\n" + "\n\n".join(words(40) for _ in range(3))
    pieces = CHUNKERS.build("markdown", max_words=50, min_words=1).split(text)
    assert len(pieces) == 3
    assert all(p.startswith("## Big") for p in pieces)


def test_chunk_ids_and_metadata():
    doc = Document(id="course/x", source="course", title="X", url="u", text="a\n\nb")
    chunks = CHUNKERS.build("paragraph", max_words=1).chunk([doc])
    assert [c.id for c in chunks] == ["course/x#0", "course/x#1"]
    assert all(c.doc_id == "course/x" and c.url == "u" for c in chunks)
