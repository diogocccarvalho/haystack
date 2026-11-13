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

**P4 — Learned per-user term statistics.** ADOPTED v0.7: corpus-IDF stopwording
(words in >50% of the corpus are dropped, any language) replaces the hardcoded
English STOP once n>=50 chunks. Audit 8/8 + sim 48/48 after the change.

## Runtime profiling (2026-11-13, Diogo's rule: >1s ops must be optimized or researched)
Measured (107 files, 1476 chunks, 6 learned entries): embed 1 query 39ms,
lexical_boost 55ms, learn_lookup embed 61ms, alias mining 14ms, reindex
incremental ~0.2s. The ONLY >1s operation was the cross-encoder rerank HTTP
call: 1457ms for 20 docs x 1000 chars (CPU, 4 threads). Optimizations applied:
- doc text capped at 600 chars for the reranker (config [rerank] max_chars):
  1457 -> 977ms. Rerankers judge relevance from the lead; no QA regression.
- [rerank] top=20 -> top=10 documented as the interactive-speed knob: ~456ms.
Research note: MiniLM-L6 (6 layers, ~2x faster, same accuracy class per
MO7YW4NG eval tables: LitSearch R@5 0.4426 fp32) has NO usable GGUF today —
cstr's conversion lacks token-type metadata ("bert model needs to define
token type count" on llama.cpp 0.8.0), sinjab's "L6" repo actually contains
an L12 model, everything else is bi-encoders (wrong class for rerank).
Converting locally needs torch (~2GB) — not worth it for 977ms -> ~500ms.
Revisit when a vetted L6 cross-encoder GGUF exists. Remaining honest latency
statement: ~1.0-1.4s per ranked query with rerank on; ~0.15s with rerank off.

**P1 addendum (2026-11-13): L6 g7c tested and REJECTED.** cstr's repo now
ships corrected `-g7c` files (pooler folded into dense→tanh→out_proj,
2026-08-05) but they still lack `tokenizer.ggml.token_type_count` and use
HF-style tensor names. Patched by hand with gguf-py (add token_type_count=2;
rename classifier.dense.*→cls.*, classifier.out_proj.*→cls.output.*; patch
script outcome recorded here so it isn't re-derived). Serves fine on
llama.cpp --reranking, wide score range (±11) as intended. A/B (same index,
same gates):
- Personal vault: L12 Recall@3 1.000 / MRR 0.926; L6 0.838 / 0.708.
- Foreign (llama.cpp docs): L12 1.000 / 0.970; L6 1.000 / 0.869.
- Latency (20 docs x 600 chars, 4 threads CPU): L12 ~830ms, L6 ~425ms.
Conclusion: L6 is exactly 2x faster and materially worse; on the corpus the
tool actually serves it drops 16pp of Recall@3. Keep ms-marco-MiniLM-L12-v2
Q8_0. Next lever if latency ever matters: [rerank] top=10 (~450ms at L12
speed). The MO7YW4NG "same accuracy class" claim does not transfer to
long-prose note retrieval.

**P3 benchmark (2026-11-13): granite via Ollama — REJECTED on this vault.**
Ollama's `granite-embedding` (30M params, 384-dim — drop-in for bge-small's
384, same /v1/embeddings shape). A/B, same index pipeline, fresh DBs:
- Personal vault: bge 1.000 Recall@3 / 0.926 MRR; granite 0.926 / 0.858.
- Foreign (llama.cpp docs): bge 1.000 / 0.970; granite 1.000 / 0.955.
- Embed latency: identical (~4ms/query both).
bge-small stays the default. Caveat: this is granite-30M-english-R2, NOT the
granite-97m-multilingual-r2 from the research above (no Ollama library
entry; needs an ONNX/Ollama-compatible host that doesn't exist today). The
multilingual headroom question stays open only for non-English vaults.

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
