# Haystack

**A semantic router for your markdown vault — built for AI agents, tuned by your own corrections.**

Static search engines index documents. Haystack also indexes *you*: every time
a query misses and you correct it, the correction becomes a permanent,
self-testing rule — so the agent stops re-reading the wrong files. All local,
zero cloud, one 35 MB embedding model.

## The problem it solves

If you run an AI agent over a personal knowledge base (a second brain, work
notes, project docs), the token bill has a shape most people don't notice:

| | Without haystack | With haystack |
|---|---|---|
| "that project about the batches" | agent greps the vault, guesses, opens 2–5 files, reads thousands of tokens of the wrong notes | one query → the right path + a 200-char snippet, ~10 ms |
| Same question tomorrow | the same exploration again | pinned by yesterday's correction — the miss can't happen twice |
| Did retrieval get worse? | nobody knows | `audit` runs your corrections as regression tests; any miss fails CI-style |
| Wrong answer from the agent | you fix the note, the pattern repeats | you say "that's not what I want" → `learn` → permanent + tested |

The author's whole agent stack (Hermes, flash-tier models, 76 sessions/month)
runs on **~66.9M tokens ≈ $1.85/month** — because the agent reads *the right
200 characters* instead of *the wrong 5 files*. Retrieval quality is what
keeps that number down. That's what haystack optimizes.

## How it works

- **Hybrid retrieval** — dense embeddings + corpus-IDF keyword scoring, fused
  with RRF (the ES/Qdrant default), plus a rare-token identifier bonus and a
  title-field boost.
- **Learned corrections** (`learn`) — your phrasing → right path, stored
  append-only outside the index (survives rebuilds). A correction only pins
  when it *beats the query's best corpus hit* — a scale-free rule, so it
  behaves identically under any embedder.
- **Cross-encoder rerank** — ms-marco MiniLM-L12 via llama.cpp `--reranking`;
  user-taught corrections pin *ahead* of the model's opinion (the teacher
  beats the model).
- **Self-audit** — every learned correction automatically becomes a QA
  regression pair. `audit` = reindex + run all pairs, exit 1 on any miss.
  Your corrections ARE the test suite; no manual test authoring.
- **Deterministic editing** — `append` / `frontmatter` / `new-note` with
  `--dry-run` everywhere, INDEX.md maintenance, edit log.

## Quickstart

```bash
./setup.sh --start          # model + config + embeddings server, one command
python3 haystack.py reindex
python3 haystack.py query "that thing about the invoices"
```

`--start` auto-detects `llama-server`, launches the embeddings server and
health-checks it; `--systemd` also installs a user unit. Any
OpenAI-compatible endpoint works instead (Ollama, LM Studio, remote API) —
point `[embeddings].url` at it. Python 3.11+, stdlib + numpy only.

## Measured

All numbers from the repo's own regression sets (docs/RESEARCH.md):

| Set | Recall@3 | MRR |
|---|---|---|
| English vault (68 pairs, nickname-style) | 1.000 | 0.917 |
| Portuguese vault (14 pairs, synthetic) | 1.000 | 1.000 |
| Foreign corpus (llama.cpp docs, 33 pairs) | 1.000 | 0.970 |
| Hand-written audit pairs | 8/8 | — |
| Learned corrections as regression | 7/7 | — |

Latency: ~10 ms/query without rerank, ~1 s with cross-encoder rerank
(~450 ms at `top=10`). Indexing: mtime-incremental, a few hundred notes in
seconds.

## Multilingual

English default: bge-small-en (llama.cpp GGUF, 35 MB, ~4 ms).
Multilingual: granite-embedding-97m (ONNX via `scripts/onnx-embed-server.py`,
200+ languages, Portuguese at parity, English −4pp) — point the config at it.
The embedder+reranker pair must match in language; the A/B recipes and the
trap list (broken static-int8 artifacts, GGUF metadata pitfalls) are in
docs/RESEARCH.md.

## Multi-corpus

Same engine, second index for external technical docs. Copy
`docs-router.toml.example`, set `[vault]` to the docs folder + a fresh index
db — e.g. llama.cpp questions answered from `~/src/llama.cpp/docs` directly.

## Design notes

- Vault-agnostic: your layout, your conventions, one `config.toml`. No schema.
- sqlite + numpy flat index. Edit log + optional INDEX.md maintenance on
  every write.
- Model-agnostic by contract; every adopted change measured against the QA
  gates (one variable at a time), rejected changes documented with numbers:
  [docs/RESEARCH.md](docs/RESEARCH.md).
- Failure catalog with architectural implications (each retrieval bug became a
  design rule): [docs/SHORTCOMINGS.md](docs/SHORTCOMINGS.md).
- HTTP API (`server.py`): /query /find /append /frontmatter /new-note /learn
  /review /unlearn /aliases /reindex /health — optional bearer token.
  `/review` is a headless two-call teach loop (candidates → teach by number
  or path), so remote agents can correct the index too.
- Everything on your disk. `learn` tables never leave your machine (they can
  contain note names — do not share them raw).

## Roadmap

- [x] RRF fusion (validated default: ES/Qdrant/SIGIR-2025 LiveRAG) + rare-token
      identifier bonus + title field boost — regression-tested on real QA pairs
- [x] Cross-encoder rerank (ms-marco-MiniLM-L12-v2 GGUF via llama.cpp --reranking;
      user-taught corrections pin ahead of it — the teacher beats the model)
- [x] Learned per-user term statistics (corpus-IDF stopwording, any language)
- [x] Alias mining (`aliases` command + /aliases endpoint): distinctive tokens
      per leaf folder, offered as `learn` candidates — nothing auto-taught
- [x] HTTP /review endpoint (headless two-call teach loop: list candidates,
      teach by n or path)
- [x] Multilingual embedder path (granite-97m via ONNX, A/B'd)

## License

MIT
