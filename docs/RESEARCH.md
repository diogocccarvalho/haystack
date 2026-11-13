# RESEARCH.md — how notes-router could be better (evidence-based, Feb-2026 sources)

Method: one variable at a time. MemDelta (arXiv 2606.29914) showed that swapping
only the embedding model in an otherwise identical pipeline shifts accuracy
+6.2pp (p=0.004), and that "agent self-memory" can UNDERPERFORM plain retrieval
(42% vs 47%) — so every improvement below must be measured against the QA-pair
audit, not adopted because it sounds better.

## Adopted already (v0.5, 2026-11-13)

1. **Context-enriched chunks.** Every non-title chunk is prefixed with
   `[doc: rel/path | title]`. Evidence: Merola & Singh, "Reconstructing Context"
   (arXiv 2504.19754) — chunks enriched with document context + BM25 over both
   segment and context improved results; also validates that fixed-window vs
   semantic chunking barely differ (our simple chunker is fine, don't chase
   semantic chunking). Anthropic's full contextual retrieval (LLM rewriting
   each chunk) preserves coherence better but costs LLM calls per chunk —
   not worth it for ~100-file vaults.
2. **Learned store split from the index DB** (`learned.json`, append-only,
   survives index rebuilds) + **unlearn** (retractions). Evidence: Reglos
   (arXiv 2611.06049) — append-only memory with explicit supersession and
   forgetting outperformed Mem0/MemOS/EverOS on LongMemEval-S; multi-route
   recall fused by RRF is the winning architecture (we already do RRF).
   Counter-evidence honored: MemDelta showed free-form agent self-memory
   loses to plain retrieval — our learning is correction-driven, not
   free-form generation, which is the defensible variant.

## Prioritized backlog

**P1 — Cross-encoder rerank (biggest single quality jump).**
Evidence: Strich et al. (arXiv 2604.01733, T2-RAGBench, 23k queries): hybrid
RRF alone Recall@5 0.695; + cross-encoder rerank 0.816. llama.cpp serves
rerankers natively: `llama-server -m <reranker>.gguf --reranking` → `/rerank`
endpoint. GGUF options (sinjab/gguf-rerankers + llamacpp-reranking benchmark):
- `ms-marco-MiniLM-L12-v2` Q4: 28 MB, ~20ms, 100% top-1 accuracy in that bench
  (English-only) — best default for this tool.
- `bge-reranker-v2-m3` Q4: 418 MB, ~72ms, 100%, multilingual — for
  multilingual users.
Design: rerank top-20 fused candidates, keep score = RRF score (the reranker
only reorders). Effort: ~1 session.

**P2 — Synthetic query generation for the audit (scale ground truth from 8 pairs to the corpus).**
Evidence: "Can we Evaluate RAGs with Synthetic Data?" (arXiv 2508.11758) —
synthetic QA benchmarks reliably RANK retriever configurations (exactly our
use case: we tune retrieval, never the generator); RAGalyst (arXiv 2511.04502)
adds an Answerability LLM-judge filter that improves dataset quality.
Design: `eval-gen` command: for each (or sampled) note, generate 2 queries
(nickname-style + literal-style, no verbatim title reuse — the arXiv
2609.38473 protocol), require the note in top-3, report Recall@3/MRR per
folder. This converts the audit from hand-curated to corpus-wide. Effort: a
session, needs a local LLM endpoint.

**P3 — Better default embedding model.**
Evidence: Granite Embedding R2 (arXiv 2605.13521): granite-97m-multilingual-r2
scores 60.3 MTEB-v2 multilingual retrieval — highest open model <100M params,
+9.4 over multilingual-e5-small, 384-dim (same as ours), ONNX 98 MB. Bekko
(arXiv 2607.25180): bekko-a8m (7.7M active params) beats BGE-M3 on MMTEB
(56.2) at ~40x less compute; a25m hits 57.5. bge-small-en-v1.5 (our default)
is 2021-generation, English-only.
Constraint: llama.cpp GGUF availability is the practical filter (granite/bekko
ship ONNX, not GGUF). Because embeddings are endpoint-agnostic, the move is:
keep bge-small as the zero-dependency GGUF default, document granite-97m via
Ollama/ONNX endpoint for multilingual users, and re-audit when swapping.
Effort: docs + one benchmark run when switching.

**P4 — Learned per-user term statistics.** Replace hardcoded English STOP
with corpus-derived weights (current lexical_boost already uses corpus IDF;
STOP remains the weak part). Low value until non-English users exist.

**P5 — Time-aware recall** (Reglos: time/place/recency routes fused by RRF).
Only relevant if the vault accumulates time-sensitive notes (open-loops
already acts as the recency layer). Defer.

## Rejected (with reasons)
- Full Anthropic contextual retrieval (LLM rewrites every chunk): cost per
  chunk, marginal over title+path prefix at vault scale.
- Semantic chunking: no measurable gain over fixed/structural chunking
  (arXiv 2504.19754 Table 2).
- Fine-tuning the embedder on ~10s of user corrections: overfitting risk,
  contrastive training wants thousands of pairs.
- Graph memory (Zep/Graphiti): entity-graph value appears at multi-user,
  multi-session scale, not single-vault retrieval; MemDelta showed
  architecture gains are frequently embedding confounds.
- Rust rewrite: no latency problem to solve (whole query is one matmul).

## Measurement discipline (from MemDelta)
Every adopted change must: reindex from scratch, run `audit` (8/8 baseline),
run `tests/sim.py`, and report the QA pass rate delta. One variable per change.
