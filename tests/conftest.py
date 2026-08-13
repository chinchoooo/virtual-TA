from pathlib import Path

import pytest

from virtual_ta.config import Component, ExperimentConfig
from virtual_ta.settings import Settings

# Corpus layout: <source>/<term>/<page>.md, with front matter as written by `vta ingest`.
CORPUS = {
    "course/2025-01/docker.md": "---\nurl: https://tds.s-anand.net/2025-01/docker/\n"
    "term: 2025-01\n---\n\n## Containers: Docker, Podman\n\nIn this course, we recommend "
    "Podman but Docker works in the same way.\n\n```bash\npodman machine init\n```\n",
    "course/2025-05/uv.md": "---\nurl: https://tds.s-anand.net/2025-05/uv/\nterm: 2025-05\n"
    "---\n\n## Python tools: uv\n\nRun scripts with a specific interpreter:\n\n"
    "uv run --python 3.11 script.py\n",
    "course/2025-05/_sidebar.md": "- [uv](uv.md)\n",
    "course/2025-05/hidden.md": "---\ndraft: true\n---\n\n# Unreleased\n",
    # No front matter: term comes from the folder, title from the heading.
    "discourse/2025-01/ga4-bonus-marks-165000.md": "# GA4 bonus marks\n\n### Post 1\n\n"
    "My dashboard shows 80 marks for GA4 after the bonus.\n\n---\n\n### Post 2\n\n"
    "You did not follow the instruction for the bonus mark.\n",
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    for rel, text in CORPUS.items():
        path = tmp_path / "data" / "corpus" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return Settings(
        api_key=None,
        api_base_url="http://invalid.local/v1",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )


@pytest.fixture
def offline_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        chunker=Component(name="markdown", params={"max_words": 50, "min_words": 3}),
        embedder=Component(name="hashing", params={"dim": 512}),
        retriever=Component(name="hybrid"),
        generator=Component(name="retrieval_only"),
        top_k=3,
        image_model=None,
    )
