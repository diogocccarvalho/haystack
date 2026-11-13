# SHORTCOMINGS.md — failure catalog of the plugins

Purpose (Diogo, 2026-11-13): every shortcoming found while dogfooding the
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
  reasoning, zero answer, at max_tokens=400. Diogo's call: don't use a small
  model during development at all; use the agent (me) as generator, then
  audit small-model failures later.
- Plugin patch: enable_thinking=False works; agent-generated eval set is the
  dev standard (qa_gen.json, 68 pairs).
- Architectural implication (crystallization): "think then do" is the wrong
  shape for bounded tasks — the thinking cost exceeds the task. Crystallized
  steps should have a think-budget of ZERO by default and earn thinking only
  when a validator says the cheap path failed. (This is the overthinking
  paper observation from the crystallization README, met in production.)

## R8. The learning loop has no eyes without the agent
- What: nothing in the product notices a wrong answer by itself; misses are
  only caught because Diogo (or the agent, via QA pairs) says so. The
  feedback loop's sensor is a human or an agent turn.
- Plugin patch: skill rule — agent auto-learns on correction; audit cron.
- Architectural implication: self-improvement needs a standing validator,
  not an occasional teacher. The from-scratch architecture's "auditor blocks"
  must run continuously against replayable ground truth, or crystallization
  crystallizes mistakes.

## R9. Known-but-unfixed (tracked, not hidden)
- Chunker heuristics tuned on ONE vault (Diogo's); no evidence they
  generalize. → needs a foreign-vault validation run before publish.
- QA ground truth written by the same agent that built the retriever →
  circularity risk. eval-gen's small-model queries partially break the
  circularity; keep both sources.
- 8 hand QA pairs + 68 agent pairs is still a small eval; recall numbers are
  directionally useful, not proof.
- bge-small embedder is 2021-generation (see RESEARCH.md P3) — swap gated on
  multilingual needs and GGUF availability.

## Running log (append on every new failure, date + entry)
- 2026-11-13: R1, R2, R4 found live; R3, R5, R6 found in the 68-pair eval
  (63/68 → 5 misses diagnosed to root causes); R7 found building eval-gen;
  R8 is standing state.
