"""Offline evaluation: retrieval metrics against labelled documents, plus an optional
LLM-as-judge pass over per-question rubrics. Every run is written to ``runs/`` so that
experiments can be compared later with ``vta compare``.
"""

from __future__ import annotations

import base64
import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from .config import ExperimentConfig
from .llm import OpenAICompatClient
from .pipeline import RAGPipeline
from .settings import Settings


@dataclass
class EvalCase:
    id: str
    question: str
    expected_docs: list[str] = field(default_factory=list)
    rubric: list[str] = field(default_factory=list)
    image: str | None = None  # path relative to the dataset file
    tags: list[str] = field(default_factory=list)
    term: str | None = None  # the term the question was asked in, e.g. "2025-01"


def load_dataset(path: Path) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cases = [EvalCase(**item) for item in raw["cases"]]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case ids in {path}")
    return cases


def _load_image(case: EvalCase, dataset_dir: Path) -> str | None:
    if not case.image:
        return None
    return base64.b64encode((dataset_dir / case.image).read_bytes()).decode()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def retrieval_metrics(retrieved_docs: list[str], expected: list[str], k: int) -> dict[str, float]:
    """Document-level metrics; duplicate chunks of the same doc are collapsed first."""
    ranked = list(dict.fromkeys(retrieved_docs))[:k]
    expected_set = set(expected)
    hits = [d for d in ranked if d in expected_set]
    first = next((i for i, d in enumerate(ranked) if d in expected_set), None)
    return {
        f"hit@{k}": float(bool(hits)),
        f"recall@{k}": len(hits) / len(expected_set),
        "mrr": 0.0 if first is None else 1 / (first + 1),
    }


JUDGE_PROMPT = """\
You are grading an answer from a course teaching assistant bot.

Question: {question}

Answer: {answer}

Criterion: {criterion}

Does the answer satisfy the criterion? Respond with JSON only:
{{"pass": true|false, "reason": "<one sentence>"}}"""


class Judge:
    def __init__(self, settings: Settings, model: str) -> None:
        self.client = OpenAICompatClient(settings)
        self.model = model

    def grade(self, question: str, answer: str, criterion: str) -> dict:
        text = self.client.chat(
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, answer=answer, criterion=criterion
                    ),
                }
            ],
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            verdict = json.loads(text)
            return {
                "criterion": criterion,
                "pass": bool(verdict["pass"]),
                "reason": verdict.get("reason", ""),
            }
        except (ValueError, KeyError):
            return {
                "criterion": criterion,
                "pass": False,
                "reason": f"unparseable judge output: {text[:200]}",
            }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_eval(
    config: ExperimentConfig,
    dataset_path: Path,
    settings: Settings | None = None,
    judge_model: str | None = None,
    retrieval_only: bool = False,
    output_dir: Path | None = None,
) -> dict:
    settings = settings or Settings.from_env()
    cases = load_dataset(dataset_path)
    pipeline = RAGPipeline.from_config(config, settings, progress=True)
    judge = Judge(settings, judge_model) if judge_model and not retrieval_only else None
    k = config.top_k

    results = []
    for case in tqdm(cases, desc=f"Evaluating {config.name}"):
        image = _load_image(case, dataset_path.parent)
        start = time.perf_counter()
        record: dict = {"id": case.id, "question": case.question, "tags": case.tags}
        try:
            if retrieval_only:
                contexts, trace = pipeline.retrieve(case.question, image, case.term)
                answer = ""
            else:
                result = pipeline.answer(case.question, image, case.term)
                contexts, answer = result.contexts, result.answer
                trace = {"queries": result.queries, "timings": result.timings}
        except Exception as e:  # keep evaluating other cases
            record.update(error=repr(e), latency=time.perf_counter() - start)
            results.append(record)
            continue

        record.update(
            answer=answer,
            latency=time.perf_counter() - start,
            queries=trace["queries"],
            timings=trace["timings"],
            retrieved=[{"chunk": h.chunk.id, "score": h.score} for h in contexts],
        )
        if case.expected_docs:
            record["retrieval"] = retrieval_metrics(
                [h.chunk.doc_id for h in contexts], case.expected_docs, k
            )
        if judge and case.rubric:
            grades = [judge.grade(case.question, answer, c) for c in case.rubric]
            record["judge"] = grades
            record["rubric_score"] = sum(g["pass"] for g in grades) / len(grades)
        results.append(record)

    summary = summarize(config, dataset_path, results, k, judge_model if judge else None)
    out = output_dir or settings.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{config.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False), encoding="utf-8"
    )
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["run_dir"] = str(out)
    return summary


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(
    config: ExperimentConfig, dataset: Path, results: list[dict], k: int, judge_model: str | None
) -> dict:
    with_retrieval = [r["retrieval"] for r in results if "retrieval" in r]
    metrics = {
        name: _mean([m[name] for m in with_retrieval])
        for name in (f"hit@{k}", f"recall@{k}", "mrr")
    }
    latencies = [r["latency"] for r in results if "error" not in r]
    return {
        "experiment": config.name,
        "fingerprint": config.fingerprint(),
        "dataset": str(dataset),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(results),
        "n_errors": sum("error" in r for r in results),
        "k": k,
        "retrieval": metrics,
        "rubric_score": _mean([r["rubric_score"] for r in results if "rubric_score" in r]),
        "judge_model": judge_model,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_max": round(max(latencies), 3) if latencies else None,
    }


def load_summaries(runs_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(runs_dir.glob("*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["run_dir"] = path.parent.name
        summaries.append(data)
    return summaries


def format_table(summaries: list[dict]) -> str:
    headers = ["run", "fingerprint", "n", "err", "hit@k", "recall@k", "mrr", "rubric", "p50 s"]
    rows = []
    for s in summaries:
        r = s["retrieval"]
        k = s["k"]
        rows.append(
            [
                s["run_dir"],
                s["fingerprint"],
                s["n_cases"],
                s["n_errors"],
                r.get(f"hit@{k}"),
                r.get(f"recall@{k}"),
                r.get("mrr"),
                s.get("rubric_score"),
                s.get("latency_p50"),
            ]
        )
    cells = [[("-" if v is None else str(v)) for v in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
