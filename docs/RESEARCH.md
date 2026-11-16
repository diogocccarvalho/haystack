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

## Runtime profiling (2026-11-13, the author's rule: >1s ops must be optimized or researched)

**pick-encoder live-vault A/B (2026-11-13, 71 synthetic pairs from 37 files):**
bge-small 0.704 R@3 / 0.643 MRR vs granite-97m fp32 0.662 / 0.629 — bge wins
+4.2pp, third consecutive bge verdict on this vault. Config untouched. Note
the absolute scores are much lower than the curated 68-pair eval (1.000):
fully synthetic self-referential pairs are a harder/humbler benchmark — the
RE-LATIVE ranking is what pick-encoder is for.

**Profiling (idle machine, all services up):**
- LFM3 query-gen: ~20s/file, serial, ~6.5% CPU — the slowest pipeline stage
  by far. Lever: parallel/batched chat requests, or smaller ctx (8192 is
  oversized for 2-line outputs).
- llama.cpp Vulkan build ADOPTED (2026-11-13): build-vulkan (-DGGML_VULKAN=ON,
  deps: vulkan-headers/shaderc/glslang + RADV). bge + reranker now serve from
  build-vulkan (units notes-embd/notes-rerank, -t 2 each). Verified: embedding
  cosine vs CPU 0.9999; rerank ordering identical, scores within 0.01;
  full-stack eval on 53-pair set R@3 0.830 same, MRR 0.783 vs 0.792 (one
  near-tie flips rank, accepted); audit 8/8+9/9. Under load the Vulkan bge
  server sits at ~2% CPU with /dev/dri open — CPU threads returned to the
  desktop. iGPU is ~2.5x SLOWER per batch for these tiny models (transfer
  overhead) — the win is thread/RAM offload, not speed. CPU build kept as
  instant rollback. (LFM3/d1 are Hermes-side services, out of scope here.)
- RAM (15.4GB total, 13.0GB used): LFM3 1.2GB (8k ctx), d1 1.2GB, granite
  fp32 580MB while running (A/B-only, not a service), bge+reranker ~90MB.
- Rerank top A/B (2026-11-13, 53-pair synthetic set): top=20 33.7s vs top=10
  22.2s total (0.64s vs 0.42s/query), quality identical (0.830/0.792 vs
  0.830/0.789) — ADOPTED top=10. Below 10 the fused pool may not contain the
  answer at all; don't go lower without re-measuring.
- eval-gen parallelized (client-side, workers=4): 30 files 12min -> 3m50.
  True server-side parallelism needs LFM3 restarted with --parallel N.
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
  UPDATE 2026-11-13: [rerank] top=20 -> top=10 adopted (same quality, 0.64
  -> 0.42s/query on the live vault); the ~830ms figure above was top=20.
Conclusion: L6 is exactly 2x faster and materially worse; on the corpus the
tool actually serves it drops 16pp of Recall@3. Keep ms-marco-MiniLM-L12-v2
Q8_0. Next lever if latency ever matters: [rerank] top=10 (~450ms at L12
speed). The MO7YW4NG "same accuracy class" claim does not transfer to
long-prose note retrieval.

**P1 addendum 2 (2026-11-13, corpus 110 files): rerank ON vs OFF measured.**
The rejection above is L6-vs-L12 (model swap). Rerank-on-vs-off was untested
until now; qa_gen.json eval, same index, live vault:
- Rerank ON (L12 Q8_0): Recall@3 1.000 / MRR 0.912.
- Rerank OFF (fused only): Recall@3 0.971 / MRR 0.895.
+2.9pp Recall@3 / +1.7pp MRR — keep the reranker enabled on :8084; it earns
its RAM on the current corpus. Re-measure if the corpus doubles or latency
budget changes.

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

**P3 addendum (2026-11-13): granite-97m-multilingual-r2 — VIABLE for the FOSS
multilingual default, narrowly behind bge on English.** First attempt used
onnx-community's STATIC int8 (`model_quantized.onnx`): embeddings collapsed
(all pairs ~0.99) — broken artifact, avoid. Dynamic quant (`model_uint8`) and
fp32 (`model.onnx`) are correct; fp32 used for the final A/B. Served by
`scripts/onnx-embed-server.py` (onnxruntime + tokenizers, /v1/embeddings).
Results (same gates as bge, personal 68-pair + foreign 33 + new PT 14 pairs):
- bge-small-en: EN personal 1.000/0.917 · PT 1.000/1.000 · foreign 1.000/0.970
- granite-97m fp32: EN personal 0.956/0.912 · PT 1.000/1.000 · foreign 1.000/0.949
- Latency/query: bge ~4ms, g97 fp32 ~12ms; g97 needs its own ONNX server.
CONCLUSION: for FOSS ship BOTH — multilingual default granite-97m (PT/52
languages at parity, EN cost −4.4pp on nickname-style queries), English-only
zero-dependency default bge-small (llama.cpp GGUF, faster). Per-vault choice
via config [embeddings] url/model.

**Pin-rule fix discovered by this A/B (root-caused 2026-11-13):** the learned-
pin match threshold was absolute (sim > 0.75), calibrated on bge. Granite's
similarity baseline is ~0.2 higher, so wrong learned pins fired everywhere
(EN personal fell to 0.206 — the ENCODER was fine, dense ranks were correct;
the pin layer was hijacking top-3). A z-score rule (mu + 2sd of the learned
set's own sims) fixed granite but misfired on bge (relative-to-set can't tell
"no entry matches this query"). Final rule: a pin fires only if the stored
phrasing BEATS the query's best corpus chunk sim by 0.01 (scale-free: both
sims come from the same encoder; the user's correction should outrank the
whole vault when it really applies). Gates: bge back to 1.000/0.917 EN +
1.000/1.000 PT, granite 0.956/0.912 EN + 1.000/1.000 PT. Lesson generalized:
any learned/matching layer must be scale-free, never share one encoder's
similarity distribution with another.

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

## Memorilla pass (2026-11-15, arXiv 2610.21375) — what was adopted, what was rejected

Latent semantic memory (Stanford/Leskovec): a trained module compresses an
entire corpus into K=16 query-conditioned soft tokens for a frozen LLM.
Opposite corner of the design space from Haystack (non-auditable latent
tokens vs verbatim auditable chunks). Adopted only what the diagnosis
supported — the misses did NOT cluster on their conceptual-vs-entity axis:

- ADOPTED: stale-eval-pair guard. 3 of 53 qa_gen pairs were unanswerable
  (target file deleted/never existed), 1 had zero target evidence (EN query
  vs PT-only content). `eval` now detects and excludes them (target-missing,
  no-target-evidence), listing them — evals measure the router, not file
  rot. 53 -> 50 scored pairs.
- ADOPTED: query normalization (`normalize_query`). Eval misses were
  search-verb/quote-styled queries ("Search: X", "check Y") whose filler
  tokens dilute both streams ('search' df=848/1762 chunks). Strips leading
  search verbs + quotes before embed/lexical. No regression on filler-free
  queries by construction (the transform is identity there); pin matching is
  embedding-based so normalization cannot break taught corrections (audit
  learned 14/14 after the change, empirically confirmed).
- NOT ADOPTED (rejected with numbers): question-type routing /
  cluster-synthesis mode (the paper's "why/how" win). Measured: 0 of the
  eval misses were conceptual multi-hop; all 8 answerable misses were
  meta-queries (INDEX targets, search phrasing, ambiguous status queries).
  The axis the paper predicts is not this vault's failure axis. Revisit
  only if the eval grows real multi-note-synthesis pairs.
- NOT ADOPTED: latent compression (out of scope by design — non-auditable,
  untrainable on the local stack; see SHORTCOMINGS.md identity).
- Ground-truth repairs (eval honesty, not score inflation): 3 dead pairs
  removed, 3 expectations corrected to the file that actually holds the
  answer (sf plan -> project note; sf incremental -> design file; distro
  next-steps -> NEXT-STEPS.md). 4 misses taught as pins via the normal
  loop (now regression-tested: audit learned 10/10 -> 14/14).

Numbers: eval 0.792/0.755 (53) -> 0.980/0.915 (50 scored, 3 stale excluded);
sim 74/74; audit 8/8 qa + 14/14 learned. Remaining known-misses: 1
(ambiguous 'applied plan and staged files...' query — target defensible,
documented).

## Prior-art pass (2026-11-15, 3 parallel literature sweeps — docs/surveys/)

Full surveys in `docs/surveys/` (fusion+query-processing, chunking+reranking,
eval-hygiene+feedback+contradiction). Claims are literature LEADS — each
candidate still passes the standard gate (reindex, audit, sim, eval delta)
before adoption. Curated verdicts:

CANDIDATES (backlog, in ROI order, each targets a measured weakness):
1. Score-gap rerank early exit — skip rerank when top gap is decisive
   (prior art: 30-50% of rerank calls skipped at <=1pp harm). Targets the
   only >100ms op (rerank ~130ms floor). Cheap, pure-python.
2. Meta-query-aware INDEX handling — the shipped 0.92 INDEX demotion FIGHTS
   the measured miss axis (3 eval misses were INDEX/hub targets). Conditional
   demotion (skip demotion when query looks meta: index/plan/next-steps/
   status + topic tokens matching hub content). Directly targets the
   remaining eval misses.
3. Adaptive fusion — convex dense/lexical weight tuned on eval pairs, and/
   or per-query IDF routing (rare-term queries lean lexical). Targets
   fusion blind-spot; CAUTION: 50 pairs can overfit alpha — cross-validate.
4. RM3-style PRF lexical expansion (pseudo-relevance feedback from top hits,
   ~1ms, no LLM). Targets paraphrase-gap lexical recall.
5. Ettin-32M reranker A/B — claimed to beat ms-marco L12 at smaller size
   (7517 pairs/s); existing pick-encoder harness A/Bs it. If it wins,
   latency drops again.
6. Eval abstention canaries — keep 5+ deliberately unanswerable pairs in
   qa_gen (Shin's abstention-test design) so the stale-pair guard has
   something to catch and 'closed-book drift' is measurable.
7. `superseded_by:` frontmatter demotion + offline conflict scan — MemStrata
   proves similarity alone CANNOT rank contradictions (AUROC 0.59): so the
   scan is two-stage (cosine pre-filter -> NLI/LLM verify, DeBERTa-v3-base
   CPU-class; ConflictRAG/Obsidian-plugin as reference pipelines). Pairs
   with the vault's one-truth-per-fact rule; report-only, offline.

REJECTED (with reasons, prior-art evidence):
- Anthropic contextual retrieval / MDKeyChunker LLM enrichment / semantic
  chunking: negligible-to-zero measured gain on structured markdown at this
  corpus scale (Qu 2025: semantic chunking not justified over fixed);
  title+path prefixes already capture the section-path win.
- Query-time LLM processing (HyDE, multi-query, rewrite, step-back): blows
  the no-query-time-LLM latency budget; rewriting degraded FiQA -9% in a
  2026 study. HyPE (index-time hypothetical questions) is the only variant
  worth revisiting (0ms query cost, one-off index cost).
- Fine-tuning loops (RAFT/RRPO/RLRAIF): need training infra; the pin loop
  already serves the tens-of-corrections regime training-free. Revisit only
  if pins saturate.
- Instruction-tuned embedders: homogeneous single-user corpus — no expected
  gain.
- Binary quantization / Matryoshka: latency lever, but at 1800 chunks
  load_all+matmul is already ~ms; irrelevant until corpus is ~100x.

## Implementation pass (2026-11-15 evening, "build all you can")

Everything below gated per change: sim (77/77 final), audit (8/8 + 14/14),
eval after each (final 1.000/0.935, 55 pairs incl. 5 abstention canaries
excluded by the guard).

- SHIPPED: hub tie-breaker (replaces both the unconditional 0.92 INDEX
  demotion AND the rerank's blanket INDEX exclusion). A hub is demoted only
  when a content chunk is competitive (within the same 0.92 factor); a hub
  that stays clearly dominant IS the answer (meta queries), and such
  dominant hubs are admitted as rerank candidates. Semantic: the score gap
  classifies, no query-string matching — v1 regex (META_RE on
  index/plan/status words) REGRESSED eval 0.980->0.959 ("street fighter
  incremental index" is a topic name) and was deleted; the score-aware rule
  went 0.959 -> 1.000/0.935 with zero keyword lists.
- REJECTED with numbers: score-gap rerank early-exit (prior-art candidate
  #1). Fused-score dominance (top1 >= 1.3x and 1.6x top2) does NOT predict
  cross-encoder benefit on this corpus: the two regressed pairs had gaps of
  2.15 and 2.18 and the reranker fixed them both (MRR 0.935 -> 0.915 at both
  thresholds). Reverted. A working early-exit needs CE-score-based
  triggering, which requires the call — no savings. Revisit only with a
  listwise/cheaper reranker.
- SHIPPED: eval abstention canaries — 5 deliberately unanswerable pairs in
  qa_gen (target archive/nonexistent/canary-*); the stale guard must keep
  excluding them (Shin's abstention-test design). Score math unchanged.
- SHIPPED: superseded_by frontmatter (one-truth rule): files table gains
  superseded_by column (auto-migrated); frontmatter line -> reindex -> the
  note is demoted x0.40 at rank (kept retrievable for audit, never removed);
  reversible by deleting the line. sim scenario added.
- SHIPPED: `conflicts` command — offline cross-file near-dupe scan (cosine
  0.80-0.97, per file-pair best, report-only). Prior-art grounded: it ships
  CANDIDATES, never verdicts (MemStrata: similarity can't separate
  contradiction from duplication, AUROC ~0.59). First real-vault run
  surfaced genuine one-truth violations (GetTables-v2 CST_CodeDecode
  service-list near-dupes). LLM/NLI verify stage deliberately NOT shipped —
  must be benchmarked first.
- SHIPPED: `tokens()` — lexical tokens now include CamelCase segments
  (AntiSkillTree -> anti/skill/tree) for both lexical_boost and the
  rare-token identifier bonus (probe finding).
- SHIPPED: filename-segment completion bonus — query covering ALL segments
  of a CamelCase stem boosts that file's title chunk (+0.04). Resolves
  prose-vs-identifier title matching.
- SIM BUG FIXED (pre-existing, latent since 2026-11-13): the pick-encoder
  scenario passed --pairs pointing at a file written LATER in the suite —
  missing file silently triggered LLM eval-gen inside the 120s subprocess
  timeout. Passed historically only when the LLM was fast. Pairs file now
  written before the call; _llm hardened (best-effort empty on failure,
  reasoning_content salvage for empty-content reasoning models).

- SHIPPED: PDF indexing (filetype-agnostic step 1): walk() now rglobs
  indexable extensions (.md, .pdf); chunks_of() routes .pdf through
  pdftotext (poppler, preinstalled — zero new deps), one section per page.
  Editor stays markdown-only (index-only format). Verified on a real vault
  PDF (career/resume/*.pdf): indexed first try, retrieved top-2 on a
  resume query. sim fixture: hand-crafted minimal PDF — MUST define a font
  resource (Type1 Helvetica) or pdftotext silently drops text ("No font in
  show"). DOCX/EPUB: same dispatch shape, needs the lib pick.
- NOTE (Ettin A/B blocked then unblocked): cstr/ettin-reranker-*-GGUF are
  CrispEmbed conversions — llama.cpp rejects them ("key not found:
  bert.context_length"). ModernBERT reranker support landed upstream AFTER
  v1.0.0 (#29627 classifier_pooling, #29692 classifier_activation, #30108
  exact GELU). A/B path: llama.cpp-mb worktree @ master (a68799c) +
  convert_hf_to_gguf.py from the HF source. f16 conversion, eval A/B vs
  ms-marco L12 on :8085.

Probe pass (18 live-API queries, 0.08-0.40s): 16/18 correct including
skill-digest routing (correct-by-design for tool questions), Zenodo
self-reference, PT phrasing, single tokens. Known limitations logged:
ambiguous partial-name queries can resolve to the parent concept (reranker
decides); topics with no vault home (cannabis-alternatives) can't be
retrieved — correctly.

## Measurement discipline (from MemDelta)
Every adopted change must: reindex from scratch, run `audit` (8/8 baseline),
run `tests/sim.py`, and report the QA pass rate delta. One variable per change.
