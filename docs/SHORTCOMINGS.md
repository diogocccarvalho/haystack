# SHORTCOMINGS.md — failure catalog of the plugins

Purpose (the author, 2026-11-13): every shortcoming found while dogfooding the
plugins (notes-router now, crystallization later) becomes a spec line for the
from-scratch architecture. Each entry: what fails, why, the plugin patch, and
the architectural implication. The plugin patches are mortised wood; the
implications are the design.

## R1. Reranker overrode user corrections (found live, 2026-11-13)
- What: query "deploying webmethods packages how" — the taught file dropped
  out of top-3 BECAUSE the cross-encoder reordered it away. The learned
  boost fed the fusion scores, but rerank then ignored scores.
- Plugin patch: pinned paths — learned corrections are placed ahead of
  rerank's order ("teacher beats model").
- Architectural implication: user-acquired constraints must be a HARD layer,
  not a soft score any learned component can override. In a from-scratch
  design, "what the user taught" is an invariant block (non-learned), not a
  prior. Selection order: invariant > learned-by-use > model opinion.

## R2. Pinned corrections vanish when rerank is down (found while fixing R1)
- What: the degraded-mode fallback (`return order`) silently dropped the
  pinning — the exact same query that worked with rerank up failed with it
  down. Two systems, one expectation, different results.
- Plugin patch: fallback returns pinned + order.
- Architectural implication: graceful degradation must degrade the CAPABILITY
  (skip reordering), never the CONSTRAINTS (ordering rules). Layers must be
  composable in any availability state. A "just do" step (crystallized op)
  can't depend on which optional components happen to be alive.

## R3. Cross-encoder latency ~1s (the only >1s op in the whole product)
- What: 20 docs x 600 chars through 33M params on CPU = ~1s; everything else
  in the pipeline is 10-60ms. The smartest component is also 20x the cost.
- Plugin patch: input cap 600 chars, optional top=10 (~450ms). L6 variant
  unusable (no valid GGUF; see RESEARCH.md).
- Architectural implication: relevance judgment at query time is the
  bottleneck of retrieval-augmented everything. The endgame (no encoders,
  crystallized selection) must make "is this context good" O(1)-ish or make
  the judgment incremental (judge once, remember the verdict), not re-judge
  every candidate on every query.

## R4. Learned memory lived inside the cache (fixed 2026-11-13, the costly one)
- What: an index rebuild deleted every learned correction — the system's
  improvement state was a casualty of its own maintenance. Found because a
  QA pair regressed right after a rebuild.
- Plugin patch: learned.json, append-only, outside the index DB.
- Architectural implication: crystallized knowledge (verified ops, taught
  mappings) must be an append-only substrate that all derived structures
  (indexes, caches, weights) are REBUILDABLE FROM. Cache may die; crystal
  may not. This is the storage law for the from-scratch architecture.

## R5. Hub files outrank the content they point to
- What: "the incremental game idea" returned design/INDEX.md (a table of
  contents) instead of StreetFighterIncremental.md (the note). INDEX files
  mention everything, so lexical + title boosts favor them.
- Plugin patch: 0.92 demotion on /INDEX.md paths.
- Architectural implication: navigational artifacts and content artifacts
  need different retrieval classes. A from-scratch architecture should index
  the graph (structure) separately from the text (content) and rank content
  first by default — "pointers serve, never replace."

## R6. Sibling ambiguity — multiple valid answers, one ground truth
- What: distro family has 6 sibling files; "hopping to another distro" can
  legitimately hit several. Any single-file ground truth penalizes correct
  behavior.
- Plugin patch: eval expectations at folder/family level where the user's
  intent really is family-level.
- Architectural implication: correctness of retrieval is often set-valued,
  not scalar. An architecture whose learning signal can't represent "this
  family, any member" will chase noise. Learned corrections should record
  granularity (file / folder / family), not just a path.

## R7. Small local models burn their budget on reasoning before answering
- What: eval-gen with LFM3-1.7B produced empty content — 4271 chars of
  reasoning, zero answer, at max_tokens=400. the author's call: don't use a small
  model during development at all; use the agent (me) as generator, then
  audit small-model failures later.
- Plugin patch: enable_thinking=False works; agent-generated eval set is the
  dev standard (qa_gen.json, 68 pairs).
- Architectural implication (crystallization): "think then do" is the wrong
  shape for bounded tasks — the thinking cost exceeds the task. Crystallized
  steps should have a think-budget of ZERO by default and earn thinking only
  when a validator says the cheap path failed. (This is the overthinking
  paper observation from the crystallization README, met in production.)

## R9. Known-but-unfixed (tracked, not hidden)
- ~~Chunker heuristics tuned on ONE vault (the author's); no evidence they
  generalize.~~ → VALIDATED 2026-11-13: foreign-vault known-answer eval on
  llama.cpp docs/ (55 files): 33 agent-authored owner-style pairs with
  verified targets (nr-foreign-qa.json, scratch), Recall@3 1.000 MRR 0.970.
  One pair needed disambiguation — ops.md is reachable via its distinctive
  intro wording but loses rerank to any backend doc's "Supported Operations"
  section on generic phrasing (same class as R6: set-valued ground truth).
  Caveats: one foreign corpus only; pairs authored by the agent (circularity
  below still applies).
- QA ground truth written by the same agent that built the retriever →
  circularity risk. eval-gen's small-model queries partially break the
  circularity; keep both sources.
- 8 hand QA pairs + 68 agent pairs (+33 foreign) is still a small eval;
  recall numbers are directionally useful, not proof.
- bge-small embedder is 2021-generation (see RESEARCH.md P3) — swap gated on
  multilingual needs and GGUF availability.

## R10. Foreign corpus killed the reindexer (found 2026-11-13, foreign-vault test)
- What: llama.cpp's docs/ (55 files, URL/code-dense markdown) — one chunk
  tokenized denser than the embedder's 512-token context; llama-server
  returned 500 for the WHOLE batch and reindex died at file 1 of 55. The
  1000-char cap was tuned on prose (the author's vault) and silently assumed
  tokens ~= chars/4.
- Plugin patch: embed_resilient — split-on-failure down to single chunks;
  over-context chunks are skipped + logged, never fatal.
- Architectural implication: any "context window" assumption in the
  from-scratch design must be enforced at the boundary (token-count check
  before ingest), not tuned per-corpus by character heuristics. Length is a
  property of the TOKENIZER, not the text; only the tokenizer can validate it.

## Pass results (2026-11-13 03:15, post-fix)
- Personal gates: audit 8/8, eval 68/68 Recall@3 (MRR 0.926), sim 48/48.
- Foreign-vault (llama.cpp docs/): reindex 55/55 with resilient embeds;
  spot-checks: build-with-cuda→build.md ✓, android→android.md ✓;
  2 other spot-checks had NO true target in the corpus (unfair queries).
  Honest status: robustness validated; retrieval-quality generalization
  still unproven (needs a corpus with known-answer questions).

## Running log (append on every new failure, date + entry)
- 2026-11-13: R1, R2, R4 found live; R3, R5, R6 found in the 68-pair eval
  (63/68 → 5 misses diagnosed to root causes);  R7 found building eval-gen; R10 found in the foreign-vault robustness test.
- Fix status: R1 ✓ (pins), R2 ✓ (fallback keeps pins), R4 ✓ (learned.json),
  R5 ✓ (INDEX excluded from rerank), R6 ✓ (family-level eval truth),
  R7 ✓ (enable_thinking=False; agent-as-generator per the author), R3 partial
  (cap 600 chars; L6 GGUF tested 2026-11-13: 2x faster, -16pp Recall@3 on
  personal vault → REJECTED, see RESEARCH.md P1 addendum; knob is top=10),
  R8 removed by the author's call (2026-11-13): agent-in-the-loop is by design,
  not a shortcoming,
  R9 mostly closed (foreign-vault quality VALIDATED 1.000/0.970; residual:
  circularity, eval size, embedder generation), R10 ✓ (embed_resilient).
