# notes-router

Local semantic router + deterministic editor for your markdown vault.

Static search engines index documents. notes-router also indexes **you**: every
time a query misses and you correct it, it learns — the correction becomes a
permanent boost for you and everyone phrasing things the way you do. All local,
zero cloud, one 35 MB embedding model.

## What it does

- **query** — semantic search over your vault (hybrid: embeddings + IDF keyword
  scoring + your learned corrections), top-N with path/heading/score/snippet.
- **find-file** — "the project about the batches and shit" → the one right path.
- **append / frontmatter / new-note** — deterministic, agent-safe note edits
  (dated lines, frontmatter keys, templated notes) with `--dry-run` everywhere.
- **learn** — teach a correction: `learn "that thing about X" path/to/note.md`.
- **review** — interactive feedback loop for any query.
- **audit** — reindex + run your ground-truth QA pairs; exit 1 on a miss, so a
  cron can watch quality regress. This is how the tool proves itself week after
  week.

## Design

- Vault-agnostic: your folder layout, your conventions — everything lives in
  one `config.toml`. No schema imposed on you.
- Model-agnostic: any OpenAI-compatible `/v1/embeddings` endpoint (llama.cpp,
  Ollama, LM Studio, remote API). Ships with bge-small (35 MB, runs on CPU).
- sqlite + numpy flat index; mtime-incremental; a few hundred notes reindex in
  seconds. Edit log + optional INDEX.md maintenance on every write.
- Everything on your disk. `learn` tables never leave your machine (they can
  contain note names — do not share them raw).

## Quickstart

```bash
./setup.sh                        # model + config template
# start an embeddings server (see setup.sh output)
python3 notes_router.py reindex
python3 notes_router.py query "that thing about the invoices"
```

Requires Python 3.11+ (stdlib + numpy only).

## Roadmap

- [x] RRF fusion (validated default: ES/Qdrant/SIGIR-2025 LiveRAG) + rare-token
      identifier bonus + title field boost — regression-tested on real QA pairs
- [x] HTTP API mode (`server.py`: /query /find /append /frontmatter /new-note
      /learn /reindex /health, optional bearer token)
- [ ] Cross-encoder rerank stage (biggest validated jump: Recall@5 0.695→0.816
      on T2-RAGBench; llama.cpp can host bge-reranker)
- [ ] Learned per-user term statistics (IDF from your vault)
- [ ] Alias mining (surface "the X thing" candidates from your own notes)
