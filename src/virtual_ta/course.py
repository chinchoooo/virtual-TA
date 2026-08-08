"""Fetch course notes term by term from sanand0/tools-in-data-science-public.

The repository's ``main`` branch holds one folder per term (listed in ``terms.yml``) whose
``_sidebar.md`` defines that term's pages; 2025 terms link into the shared ``topics/``
folder. Older terms live on their own branches. FAQ summaries of recorded live sessions
(``live-sessions/YYYYMMDD-*.md``) are assigned to a term by date.

Output: ``data/corpus/course/<term>/<page>.md`` and
``data/corpus/live_sessions/<term>/<session>.md``, each with front matter.
"""

from __future__ import annotations

import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath

import requests
import yaml

from .documents import split_front_matter, write_markdown
from .terms import term_for_date

log = logging.getLogger(__name__)

REPO = "sanand0/tools-in-data-science-public"
SITE = "https://tds.s-anand.net"
# Terms that predate the per-term folders on main. No site archive exists for them.
LEGACY_BRANCHES = {"2024-05": "tds-2024-t2", "2024-09": "tds-2024-t3"}

_SIDEBAR_LINK = re.compile(r"\]\(([^)\s#]+\.md)\)")


class Repo:
    def __init__(self, repo: str = REPO) -> None:
        self.repo = repo
        self.session = requests.Session()

    def tree(self, ref: str) -> set[str]:
        url = f"https://api.github.com/repos/{self.repo}/git/trees/{ref}?recursive=1"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return {item["path"] for item in resp.json()["tree"] if item["type"] == "blob"}

    def raw(self, ref: str, path: str) -> str:
        resp = self.session.get(
            f"https://raw.githubusercontent.com/{self.repo}/{ref}/{path}", timeout=30
        )
        resp.raise_for_status()
        return resp.text

    def blob_url(self, ref: str, path: str) -> str:
        return f"https://github.com/{self.repo}/blob/{ref}/{path}"


def _site_url(term: str, page: str) -> str:
    """'week-1/README.md' -> https://tds.s-anand.net/<term>/week-1/"""
    p = PurePosixPath(page)
    stem = p.parent if p.stem.lower() in {"readme", "_index", "index"} else p.with_suffix("")
    path = "" if str(stem) == "." else f"{stem}/"
    return f"{SITE}/{term}/{path}"


def term_pages(term: str, sidebar: str, tree: set[str]) -> list[tuple[str, str]]:
    """Resolve a term's sidebar into ``(page path relative to the term, repo path)`` pairs."""
    pages: dict[str, str] = {}
    if f"{term}/README.md" in tree:
        pages["README.md"] = f"{term}/README.md"
    for link in _SIDEBAR_LINK.findall(sidebar):
        link = link.lstrip("./")
        for candidate in (f"{term}/{link}", f"topics/{link}", link):
            if candidate in tree:
                pages.setdefault(link, candidate)
                break
        else:
            log.warning("%s: sidebar link %s not found in repo", term, link)
    return list(pages.items())


def _write_page(out: Path, body_raw: str, meta: dict) -> bool:
    page_meta, body = split_front_matter(body_raw)
    if page_meta.get("draft") or not body.strip():
        return False
    if page_meta.get("title"):
        meta = {"title": str(page_meta["title"]), **meta}
    write_markdown(out, body, meta)
    return True


def fetch(corpus_dir: Path, terms: list[str] | None = None, workers: int = 8) -> dict[str, int]:
    """Download course pages (and live-session notes) for ``terms`` (default: all)."""
    repo = Repo()
    main_tree = repo.tree("main")
    config = yaml.safe_load(repo.raw("main", "terms.yml"))
    main_terms = [t["id"] for t in config["terms"]]
    wanted = set(terms) if terms else set(main_terms) | set(LEGACY_BRANCHES)

    # (output path, ref, repo path, metadata)
    jobs: list[tuple[Path, str, str, dict]] = []
    for term in sorted(wanted):
        target = corpus_dir / "course" / term
        if term in main_terms:
            sidebar = repo.raw("main", f"{term}/_sidebar.md")
            for page, path in term_pages(term, sidebar, main_tree):
                meta = {"url": _site_url(term, page), "term": term, "origin": f"main:{path}"}
                jobs.append((target / page, "main", path, meta))
        elif term in LEGACY_BRANCHES:
            branch = LEGACY_BRANCHES[term]
            for path in sorted(repo.tree(branch)):
                if path.endswith(".md") and PurePosixPath(path).name != "_sidebar.md":
                    meta = {
                        "url": repo.blob_url(branch, path),
                        "term": term,
                        "origin": f"{branch}:{path}",
                    }
                    jobs.append((target / path, branch, path, meta))
        else:
            log.warning(
                "Unknown term %s (known: %s)", term, sorted(main_terms + list(LEGACY_BRANCHES))
            )

    for path in sorted(
        p for p in main_tree if p.startswith("live-sessions/") and p.endswith(".md")
    ):
        name = PurePosixPath(path).name
        if not re.match(r"\d{8}-", name):
            continue
        term = term_for_date(datetime.strptime(name[:8], "%Y%m%d"))
        if term in wanted:
            meta = {
                "url": repo.blob_url("main", path),
                "term": term,
                "date": name[:8],
                "origin": f"main:{path}",
            }
            jobs.append((corpus_dir / "live_sessions" / term / name, "main", path, meta))

    for term in wanted:  # rebuild fetched terms from scratch
        for source in ("course", "live_sessions"):
            shutil.rmtree(corpus_dir / source / term, ignore_errors=True)

    counts: dict[str, int] = {}

    def run(job: tuple[Path, str, str, dict]) -> None:
        out, ref, path, meta = job
        if _write_page(out, repo.raw(ref, path), meta):
            key = f"{out.relative_to(corpus_dir).parts[0]}/{meta['term']}"
            counts[key] = counts.get(key, 0) + 1

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, jobs))
    return dict(sorted(counts.items()))
