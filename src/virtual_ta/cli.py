"""Command line interface: ``vta <command> --help``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from .config import ExperimentConfig
from .settings import PROJECT_ROOT, Settings

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "dataset.yaml"


def cmd_ask(args: argparse.Namespace) -> None:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline.from_config(ExperimentConfig.from_yaml(args.config), progress=True)
    result = pipeline.answer(args.question)
    print(f"\n{result.answer}\n")
    if args.verbose:
        print("Queries:", *result.queries, sep="\n  ")
        print("\nContexts:")
        for hit in result.contexts:
            print(f"  {hit.score:.4f}  {hit.chunk.id}")
        print("\nTimings:", json.dumps({k: round(v, 3) for k, v in result.timings.items()}))
    print("Sources:")
    for link in result.links:
        print(f"  - {link.url}")


def cmd_index(args: argparse.Namespace) -> None:
    """Build the index for each config, warming the embedding cache."""
    from .pipeline import RAGPipeline

    for path in args.configs:
        config = ExperimentConfig.from_yaml(path)
        pipeline = RAGPipeline.from_config(config, progress=True)
        chunks = pipeline.retriever.chunks
        words = sorted(len(c.text.split()) for c in chunks)
        print(
            f"{config.name}: {len(chunks)} chunks, "
            f"words/chunk median={words[len(words) // 2]} max={words[-1]}"
        )


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluation import format_table, run_eval

    summaries = []
    for path in args.configs:
        summary = run_eval(
            ExperimentConfig.from_yaml(path),
            Path(args.dataset),
            judge_model=None if args.no_judge else args.judge_model,
            retrieval_only=args.retrieval_only,
        )
        summary["run_dir"] = Path(summary["run_dir"]).name
        summaries.append(summary)
    print()
    print(format_table(summaries))


def cmd_compare(args: argparse.Namespace) -> None:
    from .evaluation import format_table, load_summaries

    summaries = load_summaries(Settings.from_env().runs_dir)
    if args.filter:
        summaries = [s for s in summaries if args.filter in s["run_dir"]]
    if not summaries:
        print("No runs found. Run `vta eval` first.")
        return
    print(format_table(summaries[-args.last :] if args.last else summaries))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    app = create_app(ExperimentConfig.from_yaml(args.config))
    uvicorn.run(app, host=args.host, port=args.port)


def _date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def cmd_scrape(args: argparse.Namespace) -> None:
    from . import discourse

    stats = discourse.scrape(
        Settings.from_env().data_dir / "raw" / "discourse",
        category=args.category,
        base_url=args.base_url,
        since=args.since,
        until=args.until,
        force=args.force,
        delay=args.delay,
    )
    print(json.dumps(stats))
    if not args.no_build:
        cmd_ingest(argparse.Namespace(source="discourse", terms=None, base_url=args.base_url))


def cmd_ingest(args: argparse.Namespace) -> None:
    data = Settings.from_env().data_dir
    if args.source == "course":
        from . import course

        counts = course.fetch(data / "corpus", terms=args.terms)
    else:
        from . import discourse

        counts = discourse.build_corpus(
            data / "raw" / "discourse", data / "corpus" / "discourse", base_url=args.base_url
        )
    for key, n in counts.items():
        print(f"  {key:24} {n:5} docs")


def cmd_terms(args: argparse.Namespace) -> None:
    from collections import Counter

    from .documents import load_corpus

    corpus = Settings.from_env().data_dir / "corpus"
    sources = sorted(p.name for p in corpus.iterdir() if p.is_dir())
    counts = Counter((d.term or "-", d.source) for d in load_corpus(corpus, sources))
    terms = sorted({t for t, _ in counts})
    print(f"{'term':10}" + "".join(f"{s:>15}" for s in sources))
    for term in terms:
        print(f"{term:10}" + "".join(f"{counts[term, s]:>15}" for s in sources))


def cmd_components(args: argparse.Namespace) -> None:
    from .chunking import CHUNKERS
    from .embeddings import EMBEDDERS
    from .retrieval import RETRIEVERS
    from .stages import GENERATORS, QUERY_TRANSFORMS, RERANKERS

    for registry in (CHUNKERS, EMBEDDERS, RETRIEVERS, QUERY_TRANSFORMS, RERANKERS, GENERATORS):
        print(f"{registry.kind:16} {', '.join(registry.names())}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vta", description="Virtual TA RAG lab")
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="Answer one question")
    p.add_argument("question")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("index", help="Chunk + embed the corpus (warms the embedding cache)")
    p.add_argument("configs", nargs="*", default=[DEFAULT_CONFIG])
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("eval", help="Evaluate one or more experiment configs")
    p.add_argument("configs", nargs="+")
    p.add_argument("-d", "--dataset", default=DEFAULT_DATASET)
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--no-judge", action="store_true", help="Skip LLM-judged rubric scoring")
    p.add_argument("--retrieval-only", action="store_true", help="Skip generation (fast and cheap)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("compare", help="Tabulate saved evaluation runs")
    p.add_argument("--filter", help="Only runs whose directory name contains this string")
    p.add_argument("--last", type=int, help="Only the N most recent runs")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("serve", help="Run the HTTP API")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser(
        "scrape", help="Mirror a Discourse category (needs DISCOURSE_COOKIE), then rebuild corpus"
    )
    p.add_argument("source", choices=["discourse"])
    p.add_argument("--category", default="courses/tds-kb/34")
    p.add_argument("--base-url", default="https://discourse.onlinedegree.iitm.ac.in")
    p.add_argument("--since", type=_date, help="Only topics active on/after YYYY-MM-DD")
    p.add_argument("--until", type=_date, help="Only topics created on/before YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="Re-fetch unchanged topics")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between requests")
    p.add_argument("--no-build", action="store_true", help="Skip rebuilding the corpus")
    p.set_defaults(func=cmd_scrape)

    p = sub.add_parser("ingest", help="Rebuild a corpus source, split by term")
    p.add_argument("source", choices=["course", "discourse"])
    p.add_argument("--terms", nargs="+", help="course only: e.g. 2025-01 2025-05 (default all)")
    p.add_argument("--base-url", default="https://discourse.onlinedegree.iitm.ac.in")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("terms", help="Document counts per term and source")
    p.set_defaults(func=cmd_terms)

    p = sub.add_parser("components", help="List registered pipeline components")
    p.set_defaults(func=cmd_components)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
