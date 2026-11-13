# SHORTCOMINGS.md — failure catalog of the plugins

Purpose (the author, 2026-11-13): catalog of failures found while dogfooding
the tool. Each entry: what fails, why, and the plugin patch. The design
implications distilled from these failures moved to the author's private
crystallization repo (docs/design.md) — haystack keeps the observations, the
from-scratch architecture gets the rules.

## R1. Reranker overrode user corrections (found live, 2026-11-13)
- What: query "deploying packages how" (a real taught-route miss) — the file dropped
  out of top-3 BECAUSE the cross-encoder reordered it away. The learned
  boost fed the fusion scores, but rerank then ignored scores.
- Plugin patch: pinned paths — learned corrections are placed ahead of
  rerank's order ("teacher beats model").
- Design rule: D1 (crystallization repo, docs/design.md).

## R2. Pinned corrections vanish when rerank is down (found while fixing R1)
- What: the degraded-mode fallback (`return order`) silently dropped the
  pinning — the exact same query that worked with rerank up failed with it
  down. Two systems, one expectation, different results.
- Plugin patch: fallback returns pinned + order.
- Design rule: D2 (crystallization repo, docs/design.md).

## R3. Cross-encoder latency ~1s (the only >1s op in the whole product)
- What: 20 docs x 600 chars through 33M params on CPU = ~1s; everything else
  in the pipeline is 10-60ms. The smartest component is also 20x the cost.
- Plugin patch: input cap 600 chars, optional top=10 (~450ms). L6 variant
  unusable (no valid GGUF; see RESEARCH.md).
- Design rule: D3 (crystallization repo, docs/design.md).

## R4. Learned memory lived inside the cache (fixed 2026-11-13, the costly one)
- What: an index rebuild deleted every learned correction — the system's
  improvement state was a casualty of its own maintenance. Found because a
  QA pair regressed right after a rebuild.
- Plugin patch: learned.json, append-only, outside the index DB.
- Design rule: D4 (crystallization repo, docs/design.md).

## R5. Hub files outrank the content they point to
- What: "the incremental game idea" returned design/INDEX.md (a table of
  contents) instead of StreetFighterIncremental.md (the note). INDEX files
  mention everything, so lexical + title boosts favor them.
- Plugin patch: 0.92 demotion on /INDEX.md paths.
- Design rule: D5 (crystallization repo, docs/design.md).

## R6. Sibling ambiguity — multiple valid answers, one ground truth
- What: distro family has 6 sibling files; "hopping to another distro" can
  legitimately hit several. Any single-file ground truth penalizes correct
  behavior.
- Plugin patch: eval expectations at folder/family level where the user's
  intent really is family-level.
- Design rule: D6 (crystallization repo, docs/design.md).

## R7. Small local models burn their budget on reasoning before answering
- What: eval-gen with LFM3-1.7B produced empty content — 4271 chars of
  reasoning, zero answer, at max_tokens=400. the author's call: don't use a small
  model during development at all; use the agent (me) as generator, then
  audit small-model failures later.
- Plugin patch: enable_thinking=False works; agent-generated eval set is the
  dev standard (qa_gen.json, 68 pairs).
- Design rule: D7 (crystallization repo, docs/design.md) — the overthinking
  paper observation, met in production.

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
- Design rule: D8 (crystallization repo, docs/design.md).

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
