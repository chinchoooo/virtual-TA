# Status and roadmap

_Last updated: 2026-10-12, branch `rag-lab`._

## Current results

Eval set: `eval/dataset.yaml`, 13 questions (all Jan 2025 term), k = 5, retrieval metrics only.
Every run below is **offline**: no API key, hashing embeddings and/or BM25, no generation.
Reproduce with `vta eval <config> --retrieval-only`.

| config                | retrieval                          | term filter | hit@5    | recall@5 | MRR      |
| --------------------- | ---------------------------------- | ----------- | -------- | -------- | -------- |
| `bm25`                | BM25                               | no          | 0.83     | 0.79     | 0.56     |
| `offline`             | hashing-dense + BM25, RRF          | no          | 0.83     | 0.79     | 0.78     |
| `offline_term_filter` | hashing-dense + BM25, RRF          | yes         | **0.92** | **0.88** | **0.92** |

Corpus: 4,639 documents / 33,124 chunks (course + discourse, all terms 2021–2026).

How to read this:

- **n = 13 is small.** One question is 7.7 points of hit@5, so only gaps of several
  questions mean anything. Growing the eval set is item 1 below.
- **Term filtering is the largest effect so far.** Without it, most misses are the same
  course page from another term (e.g. `course/2025-09/fastapi` instead of
  `course/2025-01/fastapi`). The eval counts that as a miss, which is partly real
  cross-term confusion and partly eval strictness (see item 2).
- **The dense half of the hybrid mainly improves ranking.** It doesn't change hit@5 over
  BM25, but MRR rises from 0.56 to 0.78.
- `expose-local-server` (expected doc: `course/2025-01/ngrok`) fails in every offline config.
- **Not yet measured:** OpenAI-embedding configs (`baseline`, `hybrid`, `hyde`,
  `hybrid_rerank`, `multi_query_hybrid`) and answer quality (LLM-judge rubric scores).
  Both need an API key. Embedding the full corpus is roughly 5M tokens, about $0.10 with
  `text-embedding-3-small`, cached after the first run.

## What has been done

| area           | before                                                              | now                                                                                             |
| -------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Answering      | LLM was given only top-3 **filenames**, never the text               | Retrieved passages with titles, terms and URLs; inline citations                               |
| Architecture   | 5 loose scripts, code ran at import                                 | `virtual_ta` package; every stage pluggable via registries + YAML configs                       |
| Approaches     | 1 (whole-doc dense)                                                 | 4 chunkers, 2 embedders, 3 retrievers, 3 query transforms, 2 rerankers, 2 generators            |
| Evaluation     | promptfoo smoke test (3 Qs)                                         | `vta eval` / `vta compare`: hit@k, recall@k, MRR, LLM judge, runs saved with config fingerprint |
| Forum scrape   | Jan–Apr 2025, first 20 posts per topic, no topic ids                | All 3,985 topics (2021–2026), every post, topic ids, incremental, retries                      |
| Course notes   | One snapshot (Jan 2025)                                             | 9 terms from the repo's per-term folders + 2024 branches; live-session notes                    |
| Term handling  | None                                                                | Every doc tagged by term (tags → title → date); optional term-filtered retrieval                |
| Links          | Built from post ids (wrong URLs)                                    | Real `/t/<slug>/<id>/<post>` and per-term course URLs (sampled: all 200)                        |
| Images         | First 200 chars of base64 sent to a text model                      | Captioned by a vision model for retrieval, passed to the generator                              |
| Cost           | Re-embedded everything each run                                     | Content-addressed embedding cache                                                               |
| Security       | API keys hardcoded; forum data public in repo                       | Secrets in `.env`; Discourse content gitignored (**history still needs rewriting**)            |
| Engineering    | No tests, `requirements.txt`, Windows-only DB path                  | 34 offline tests, ruff, CI, uv lockfile, non-root Docker image with healthcheck                 |

## Roadmap

Ordered by expected value. Effort: S = hours, M = a day or two, L = a week.

### 1. Measurement (do first: everything else depends on it)

| #   | upgrade                                                                                                                                                                                                                          | effort | why                                                                |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------ |
| 1.1 | **Mine an eval set from the forum.** Topics with an accepted answer or a course-team reply become (question = first post, expected doc = that topic, rubric = staff answer). Target 200+ questions across terms, with a frozen held-out split. | M      | n = 13 can't separate approaches; the metadata needed is already scraped |
| 1.2 | **Term-equivalent matching:** count any term's copy of the same course page (`origin` metadata) as a hit for term-agnostic questions; keep strict matching for term-specific ones                                                  | S      | Separates real term confusion from eval strictness                 |
| 1.3 | Bootstrap confidence intervals and per-tag breakdowns in `vta compare`                                                                                                                                                           | S      | Know when a gain is noise                                          |
| 1.4 | Run the API-backed configs once and record a real baseline with rubric scores                                                                                                                                                    | S      | ~$0.10 embeddings + judge calls                                    |
| 1.5 | Answer-level metrics: citation correctness (do cited passages support the claim?) and abstention accuracy on unanswerable questions                                                                                              | M      | Retrieval hits don't guarantee correct answers                     |

### 2. Retrieval

| #   | upgrade                                                                                                                                                                                            | effort | why                                                                       |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------- |
| 2.1 | **Post-aware Discourse chunking:** one chunk per post, prefixed with the topic title and the post it replies to (`reply_to_post_number` is stored)                                                  | M      | Threads reach 367 posts; answers live in individual replies               |
| 2.2 | **Authority boosting:** prior on course-team posts and accepted answers (already labelled in the corpus)                                                                                           | S      | Staff answers are the ground truth students need                          |
| 2.3 | **Soft term prior instead of a hard filter:** detect the question's term (API field, date asked, or LLM), down-weight other terms rather than excluding them                                     | M      | The hard filter drops useful cross-term material such as tool docs        |
| 2.4 | Collapse duplicate course pages across terms at retrieval time (same `origin`), keeping the closest term                                                                                           | S      | Frees top-k slots currently spent on copies                               |
| 2.5 | Local cross-encoder reranker (e.g. `bge-reranker-v2-m3`) vs the existing LLM listwise reranker                                                                                                     | M      | Cheaper and faster than LLM reranking, usually stronger than RRF alone    |
| 2.6 | Contextual chunk headers (LLM one-line summary of the parent doc prepended before embedding)                                                                                                       | M      | Known recall gains for short, context-poor chunks                         |
| 2.7 | float16 matrix or sqlite-vec/FAISS index; persist the fitted index                                                                                                                                  | S      | 33k × 1536 float32 ≈ 200 MB; the first scrape was killed for low memory   |

### 3. Generation

| #   | upgrade                                                                                                                       | effort | why                                                           |
| --- | ----------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------- |
| 3.1 | Structured output (`answer`, `citations[]`); build response links from **cited** passages rather than top-k                   | S      | Links currently reflect retrieval, not what the answer used   |
| 3.2 | Explicit abstention: confidence threshold on retrieval scores plus "I don't know" prompting, measured by 1.5                  | S      | Never invent dates or policies                                |
| 3.3 | Forum images: caption/OCR images inside posts (now `[image: …]` placeholders) at ingest                                       | M      | Many forum questions and staff answers are screenshots        |

### 4. Operations

| #   | upgrade                                                                                                                                                                | effort | why                                                             |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------- |
| 4.1 | **Rewrite git history** to purge the leaked AI Pipe key and the formerly committed forum data, then force-push; rotate the key                                         | S      | Both are still public in old commits                            |
| 4.2 | Scheduled incremental scrape on a local machine (cron / Task Scheduler). CI can't do it, because the forum needs a personal login                                       | S      | Keep the current term fresh                                     |
| 4.3 | Build the index at deploy time, not on first request; mount the embedding cache                                                                                        | S      | Cold start currently embeds/fits 33k chunks                     |
| 4.4 | Query logging (question, retrieved ids, latency) + a feedback endpoint; promote bad answers into the eval set                                                          | M      | Real questions are the best eval data                           |
| 4.5 | API auth + rate limiting before any public deployment                                                                                                                  | S      | The endpoint spends your API credits                            |

### Suggested order

1.1 → 1.2 → 1.4 → 2.2 → 2.1 → 2.4 → 3.1 → 2.5. Do 4.1 now, independently of the rest.
