"""Discourse scraping and term-by-term corpus building.

Scraping (``vta scrape discourse``) mirrors every topic of a category into
``data/raw/discourse/topics/<topic_id>.json``. It is incremental: a topic is re-fetched
only when its ``last_posted_at`` or ``posts_count`` changed since the last run.

Building (``vta ingest discourse``) splits each topic by academic term and writes
``data/corpus/discourse/<term>/<slug>-<topic_id>.md`` with YAML front matter. A topic's
term comes from its tags (``term1-2025``), then its title (``[TDS Jan 2025]``); topics
with neither (long-lived knowledge-base threads) are split post by post by date.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from .documents import write_markdown
from .terms import term_for_date, term_from_tags, term_from_title

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
DEFAULT_CATEGORY = "courses/tds-kb/34"
POST_BATCH = 20

# Post fields worth keeping. Usernames/avatars are dropped on purpose.
_POST_FIELDS = (
    "id",
    "post_number",
    "post_type",
    "created_at",
    "updated_at",
    "reply_to_post_number",
    "cooked",
    "staff",
    "admin",
    "moderator",
    "user_title",
    "primary_group_name",
    "trust_level",
    "accepted_answer",
    "reply_count",
    "score",
)
_COURSE_TEAM = re.compile(r"faculty|instructor|\bta\b|teaching|course.?team", re.IGNORECASE)


class DiscourseAuthError(RuntimeError):
    pass


class DiscourseClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        cookie: str | None = None,
        delay: float = 0.5,
        timeout: float = 30,
        max_retries: int = 5,
    ) -> None:
        cookie = cookie or os.getenv("DISCOURSE_COOKIE")
        if not cookie:
            raise DiscourseAuthError(
                "DISCOURSE_COOKIE is not set. Copy the `_t` and `_forum_session` cookies from "
                "your logged-in browser into .env as `DISCOURSE_COOKIE=_t=...; _forum_session=...`"
            )
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Cookie": cookie,
                "User-Agent": "virtual-ta-scraper/0.2 (+https://github.com/bored-arvi/virtual-TA)",
            }
        )

    def get(self, path: str, params: Any = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise
                wait = 2**attempt
                log.warning("%s failed (%s); retrying in %ss", url, e, wait)
            else:
                if resp.status_code in (401, 403):
                    raise DiscourseAuthError(
                        f"{resp.status_code} for {url}: session cookie missing or expired. "
                        "Refresh DISCOURSE_COOKIE from your browser."
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt == self.max_retries:
                        resp.raise_for_status()
                    wait = float(resp.headers.get("Retry-After", 2**attempt))
                    log.warning("%s returned %s; retrying in %ss", url, resp.status_code, wait)
                else:
                    resp.raise_for_status()
                    time.sleep(self.delay)
                    return resp.json()
            time.sleep(wait)
        raise AssertionError("unreachable")

    def current_user(self) -> str | None:
        user = self.get("session/current.json").get("current_user") or {}
        return user.get("username")

    def iter_category(self, category: str):
        """Yield topic summaries from every page of a category listing."""
        seen: set[int] = set()
        page = 0
        while True:
            data = self.get(f"c/{category}.json", params={"page": page})
            topics = data.get("topic_list", {}).get("topics", [])
            new = [t for t in topics if t["id"] not in seen]
            if not new:
                return
            for topic in new:
                seen.add(topic["id"])
                yield topic
            if not data["topic_list"].get("more_topics_url"):
                return
            page += 1

    def fetch_topic(self, topic_id: int) -> dict:
        data = self.get(f"t/{topic_id}.json")
        stream = data["post_stream"]["stream"]
        posts = {p["id"]: p for p in data["post_stream"]["posts"]}
        missing = [pid for pid in stream if pid not in posts]
        for i in range(0, len(missing), POST_BATCH):
            batch = missing[i : i + POST_BATCH]
            extra = self.get(f"t/{topic_id}/posts.json", params={"post_ids[]": batch})
            posts.update((p["id"], p) for p in extra["post_stream"]["posts"])
        return {
            "id": data["id"],
            "slug": data["slug"],
            "title": data["title"],
            "category_id": data.get("category_id"),
            "tags": [t["name"] if isinstance(t, dict) else t for t in data.get("tags", [])],
            "created_at": data["created_at"],
            "last_posted_at": data.get("last_posted_at"),
            "posts_count": data.get("posts_count"),
            "scraped_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "posts": [
                {k: posts[pid].get(k) for k in _POST_FIELDS} for pid in stream if pid in posts
            ],
        }


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def scrape(
    raw_dir: Path,
    category: str = DEFAULT_CATEGORY,
    base_url: str = DEFAULT_BASE_URL,
    since: datetime | None = None,
    until: datetime | None = None,
    force: bool = False,
    delay: float = 0.5,
) -> dict[str, int]:
    """Mirror every topic of ``category`` (optionally limited to a date window)."""
    client = DiscourseClient(base_url, delay=delay)
    user = client.current_user()
    if not user:
        raise DiscourseAuthError("Not logged in: refresh DISCOURSE_COOKIE from your browser.")
    log.info("Logged in as %s", user)

    topics_dir = raw_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    stats = {"listed": 0, "fetched": 0, "unchanged": 0, "out_of_range": 0, "failed": 0}
    bar = tqdm(client.iter_category(category), desc="Topics", unit="topic")
    for summary in bar:
        stats["listed"] += 1
        created, last = _parse(summary.get("created_at")), _parse(summary.get("last_posted_at"))
        if (since and (last or created) < since) or (until and created > until):
            stats["out_of_range"] += 1
            continue

        path = topics_dir / f"{summary['id']}.json"
        if path.exists() and not force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (cached.get("last_posted_at"), cached.get("posts_count")) == (
                summary.get("last_posted_at"),
                summary.get("posts_count"),
            ):
                stats["unchanged"] += 1
                continue
        try:
            topic = client.fetch_topic(summary["id"])
        except DiscourseAuthError:
            raise
        except requests.RequestException as e:
            log.error("Topic %s failed: %s", summary["id"], e)
            stats["failed"] += 1
            continue
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(topic, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        stats["fetched"] += 1
        bar.set_postfix(fetched=stats["fetched"], unchanged=stats["unchanged"])
    return stats


# --------------------------------------------------------------------------- #
# Corpus building
# --------------------------------------------------------------------------- #


def _is_course_team(post: dict) -> bool:
    if post.get("staff") or post.get("admin") or post.get("moderator"):
        return True
    group = f"{post.get('user_title') or ''} {post.get('primary_group_name') or ''}"
    return bool(_COURSE_TEAM.search(group))


def clean_markdown(md: str) -> str:
    md = re.sub(r"\[@[^\]]+\]\(/u/[^)]+\)", "", md)  # @mentions
    md = re.sub(r"!\[[^\]]*\]\([^)]*emoji[^)]*\)", "", md, flags=re.IGNORECASE)  # emoji images
    # Lightbox images "[![alt](thumb)alt 1920×1080 89 KB](full)" and plain images -> placeholder
    md = re.sub(r"\[!\[([^\]]*)\]\([^)]*\)[^\]]*\]\([^)]*\)", r"[image: \1]", md)
    md = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[image: \1]", md)
    md = re.sub(r"\n\s*\n", "\n\n", md)  # collapse blank lines
    return md.strip()


def _post_markdown(post: dict) -> str:
    import html2text
    from bs4 import BeautifulSoup

    converter = html2text.HTML2Text()
    converter.body_width = 0  # don't hard-wrap lines
    body = clean_markdown(converter.handle(str(BeautifulSoup(post["cooked"], "html.parser"))))
    labels = []
    if _is_course_team(post):
        labels.append("course team")
    if post.get("accepted_answer"):
        labels.append("accepted answer")
    if post.get("reply_to_post_number"):
        labels.append(f"reply to #{post['reply_to_post_number']}")
    suffix = f" ({', '.join(labels)})" if labels else ""
    return f"### Post {post['post_number']}{suffix} — {post['created_at'][:10]}\n\n{body}"


def topic_term(topic: dict) -> str | None:
    """The term a whole topic belongs to, or None if it should be split by post date."""
    return term_from_tags(topic.get("tags", [])) or term_from_title(topic["title"])


def build_corpus(raw_dir: Path, out_dir: Path, base_url: str = DEFAULT_BASE_URL) -> dict[str, int]:
    """Rebuild ``out_dir`` from scraped topics, one markdown file per (topic, term)."""
    topic_files = sorted((raw_dir / "topics").glob("*.json"))
    if not topic_files:
        raise FileNotFoundError(f"No scraped topics in {raw_dir / 'topics'}; run `vta scrape`.")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    per_term: dict[str, int] = defaultdict(int)
    for path in tqdm(topic_files, desc="Building discourse corpus"):
        topic = json.loads(path.read_text(encoding="utf-8"))
        fixed = topic_term(topic)
        groups: dict[str, list[dict]] = defaultdict(list)
        for post in topic["posts"]:
            if post.get("post_type") != 1 or not (post.get("cooked") or "").strip():
                continue  # skip whispers, small actions ("pinned", "closed"), empty posts
            groups[fixed or term_for_date(post["created_at"])].append(post)

        for term, posts in groups.items():
            first = posts[0]["post_number"]
            url = f"{base_url}/t/{topic['slug']}/{topic['id']}"
            if first > 1:
                url += f"/{first}"
            body = f"# {topic['title']}\n\n" + "\n\n---\n\n".join(_post_markdown(p) for p in posts)
            write_markdown(
                out_dir / term / f"{topic['slug'][:80]}-{topic['id']}.md",
                body,
                {
                    "title": topic["title"],
                    "url": url,
                    "term": term,
                    "topic_id": topic["id"],
                    "tags": topic.get("tags", []),
                    "created_at": posts[0]["created_at"],
                    "last_posted_at": posts[-1]["created_at"],
                    "posts": len(posts),
                    "course_team_posts": sum(_is_course_team(p) for p in posts),
                },
            )
            per_term[term] += 1
    return dict(sorted(per_term.items()))
