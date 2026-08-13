from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from virtual_ta.api import create_app
from virtual_ta.config import ExperimentConfig
from virtual_ta.documents import load_corpus
from virtual_ta.evaluation import load_dataset, run_eval
from virtual_ta.pipeline import RAGPipeline
from virtual_ta.settings import PROJECT_ROOT


def test_corpus_loading(settings):
    docs = {d.id: d for d in load_corpus(settings.data_dir / "corpus", ["course", "discourse"])}
    assert set(docs) == {
        "course/2025-01/docker",
        "course/2025-05/uv",
        "discourse/2025-01/ga4-bonus-marks-165000",
    }  # _sidebar and draft pages are skipped
    docker = docs["course/2025-01/docker"]
    assert docker.url == "https://tds.s-anand.net/2025-01/docker/"
    assert docker.title == "Containers: Docker, Podman"
    assert docs["discourse/2025-01/ga4-bonus-marks-165000"].term == "2025-01"


def test_term_filter(settings, offline_config):
    corpus = settings.data_dir / "corpus"
    assert {d.term for d in load_corpus(corpus, ["course"], terms=["2025-05"])} == {"2025-05"}
    offline_config.corpus.terms = ["2025-01"]
    pipeline = RAGPipeline.from_config(offline_config, settings)
    assert {c.term for c in pipeline.retriever.chunks} == {"2025-01"}


def test_pipeline_end_to_end_offline(settings, offline_config):
    pipeline = RAGPipeline.from_config(offline_config, settings)
    result = pipeline.answer("Should I use podman or docker?")
    assert result.contexts[0].chunk.doc_id == "course/2025-01/docker"
    assert result.links[0].url == "https://tds.s-anand.net/2025-01/docker/"
    assert len({link.url for link in result.links}) == len(result.links)


def test_api(settings, offline_config):
    with TestClient(create_app(offline_config, settings)) as client:
        assert client.get("/health").json()["experiment"] == "test"
        resp = client.post("/api/", json={"question": "uv python version"})
        assert resp.status_code == 200
        assert resp.json()["links"][0]["url"].endswith("/2025-05/uv/")
        assert client.post("/api/", json={"question": ""}).status_code == 422


def test_run_eval_writes_run_dir(settings, offline_config, tmp_path):
    dataset = tmp_path / "ds.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {
                        "id": "a",
                        "question": "podman docker",
                        "expected_docs": ["course/2025-01/docker"],
                    },
                    {"id": "b", "question": "unknowable"},
                ]
            }
        )
    )
    summary = run_eval(offline_config, dataset, settings, retrieval_only=True)
    assert summary["retrieval"]["hit@3"] == 1.0
    assert summary["n_errors"] == 0
    run_dir = Path(summary["run_dir"])
    assert {p.name for p in run_dir.iterdir()} == {"config.yaml", "results.jsonl", "summary.json"}


def test_shipped_configs_and_dataset_are_valid():
    for path in sorted((PROJECT_ROOT / "configs").rglob("*.yaml")):
        ExperimentConfig.from_yaml(path)
    cases = load_dataset(PROJECT_ROOT / "eval" / "dataset.yaml")
    corpus = PROJECT_ROOT / "data" / "corpus"
    sources = [p.name for p in corpus.iterdir() if p.is_dir()]  # discourse is local-only
    corpus_ids = {d.id for d in load_corpus(corpus, sources)}
    expected = {doc for c in cases for doc in c.expected_docs if doc.split("/")[0] in sources}
    missing = expected - corpus_ids
    assert not missing, f"dataset references unknown docs: {missing}"
