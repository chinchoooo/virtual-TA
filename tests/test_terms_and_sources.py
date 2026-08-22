import json

import pytest

from virtual_ta.course import _site_url, term_pages
from virtual_ta.discourse import build_corpus, clean_markdown
from virtual_ta.documents import load_source
from virtual_ta.terms import term_for_date, term_from_tags, term_from_title, term_label


@pytest.mark.parametrize(
    "ts, term",
    [
        ("2025-01-01T00:00:00Z", "2025-01"),
        ("2025-04-30T23:59:59Z", "2025-01"),
        ("2025-05-01T00:00:00Z", "2025-05"),
        ("2025-12-31T00:00:00+05:30", "2025-09"),
    ],
)
def test_term_for_date(ts, term):
    assert term_for_date(ts) == term


def test_term_from_tags():
    assert term_from_tags(["announcement", "term1-2025"]) == "2025-01"
    assert term_from_tags(["t3-2024"]) == "2024-09"
    assert term_from_tags(["week-4"]) is None


@pytest.mark.parametrize(
    "title, term",
    [
        ("GA4 - Data Sourcing - Discussion Thread [TDS Jan 2025]", "2025-01"),
        ("TDS May 25 Project 1", "2025-05"),
        ("Project 2 [Sep 2024]", "2024-09"),
        ("Deadline is May 25th", None),  # a date, not a term
        ("Graded Assigments", None),
    ],
)
def test_term_from_title(title, term):
    assert term_from_title(title) == term


def test_term_label():
    assert term_label("2025-09") == "Sep 2025"


def test_clean_markdown_strips_mentions_emoji_and_image_urls():
    md = (
        "Hi [@bob](/u/bob) ![:grin:](https://emoji.x/grin.png)\n\n\n"
        "[![shot](https://x/t.png)shot 1920×1020 89 KB](https://x/full.png)"
    )
    assert clean_markdown(md) == "Hi  \n\n[image: shot]"


def _post(n, created, cooked="<p>text</p>", **kw):
    return {"post_number": n, "post_type": 1, "created_at": created, "cooked": cooked, **kw}


def test_build_corpus_splits_untagged_topics_by_post_date(tmp_path):
    raw = tmp_path / "raw"
    (raw / "topics").mkdir(parents=True)
    tagged = {
        "id": 1, "slug": "ga1-thread", "title": "GA1 [TDS Jan 2025]", "tags": [],
        "posts": [_post(1, "2025-01-10T00:00:00Z"), _post(2, "2025-06-01T00:00:00Z")],
    }  # fmt: skip
    evergreen = {
        "id": 2, "slug": "faq", "title": "FAQ", "tags": [],
        "posts": [
            _post(1, "2024-10-01T00:00:00Z"),
            _post(2, "2024-10-02T00:00:00Z", post_type=3),  # small action: dropped
            _post(3, "2025-02-01T00:00:00Z", user_title="Course_faculty", accepted_answer=True),
        ],
    }  # fmt: skip
    for topic in (tagged, evergreen):
        (raw / "topics" / f"{topic['id']}.json").write_text(json.dumps(topic))

    counts = build_corpus(raw, tmp_path / "out", base_url="https://d")
    assert counts == {"2024-09": 1, "2025-01": 2}

    docs = {d.id: d for d in load_source(tmp_path / "out", "discourse")}
    assert set(docs) == {
        "discourse/2025-01/ga1-thread-1",  # whole topic follows its title tag
        "discourse/2024-09/faq-2",
        "discourse/2025-01/faq-2",
    }
    later = docs["discourse/2025-01/faq-2"]
    assert later.url == "https://d/t/faq/2/3"  # links to the first post in that term
    assert later.term == "2025-01"
    assert "course team, accepted answer" in later.text
    assert later.metadata["course_team_posts"] == 1


def test_term_pages_resolve_shared_topics():
    tree = {"2025-05/README.md", "topics/uv.md", "2025-05/project-1.md"}
    sidebar = "- [Home](README.md)\n- [uv](uv.md)\n- [P1](project-1.md)\n- [Gone](gone.md)"
    assert term_pages("2025-05", sidebar, tree) == [
        ("README.md", "2025-05/README.md"),
        ("uv.md", "topics/uv.md"),
        ("project-1.md", "2025-05/project-1.md"),
    ]


def test_site_url():
    assert _site_url("2025-01", "docker.md") == "https://tds.s-anand.net/2025-01/docker/"
    assert _site_url("2026-09", "README.md") == "https://tds.s-anand.net/2026-09/"
    assert _site_url("2026-09", "week-1/README.md") == "https://tds.s-anand.net/2026-09/week-1/"
