# Prior-art survey: eval hygiene, small-scale feedback loops, contradiction detection
Compiled 2026-11-15 by research subagent (task 3 of the Memorilla follow-up
prior-art pass). Claims are LEADS from literature search — verify at the
implementation gate, same rule as every adopted change. Not independently
re-verified by the maintainer yet.

## (a) Eval hygiene

Vectors: (1) corpus-eval overlap, (2) pretrain contamination, (3) stale-
document poisoning. Prior art found:

- NeoQA (2024): fictional news timelines so answers can't be memorized.
- LatestEval (AAAI 2025): recent-window texts, answers deleted from context.
- RefusalBench (2025): programmatic fresh unanswerable instances.
- RePCS (2025): KL-divergence between RAG output and parametric-only output
  detects when retrieval isn't actually used; ~one extra forward pass,
  ROC-AUC 0.918 claimed.
- Stale-Document Poisoning benchmark (2025): 317 verified knowledge
  reversals; outdated retrieval flips 30-37% of answers.
- MemStrata (2025): cosine-similarity AUROC ~0.59 separating contradictions
  from duplicates — similarity CANNOT detect staleness/contradiction;
  needs deterministic (subject, relation, object) supersedence rules or
  temporal metadata. "Fixed recency-aware hybrid re-ranker reduces
  poisoning by 4.6-10.0 points."
- LayerRAG-Bench (2025): 240 tasks x 9 fault scenarios incl. stale_index
  and conflicting_evidence.
- Corpus-Provenance Gap (Shin 2025, blog): document-level content hashing
  between eval-QA sources and corpus manifest; abstention tests with 10-20
  known-unanswerable queries.

## (b) Feedback loops with TENS of corrections

- REPLUG + LSR (2023): frozen-LM perplexity delta as per-pair retriever
  signal; no batch needed.
- RAFT (2024): SFT on (query, golden docs, distractors, CoT); trains
  distractor-ignoring. Needs training infra (LoRA).
- RRPO (2026): reranking as RL with generation-quality reward; 300M reranker.
- RLRAIF (2024): RL-to-rank from AI feedback, designed for <1% feedback
  budgets.
- CRAG (2024): inference-time retrieval evaluator -> correct/incorrect/
  ambiguous gates, no training.
- CAR (2026): confidence-aware reranking from answer-stability deltas;
  training-free, black-box.
- Q-REACT (2026): sparse reranker feedback -> low-rank query-side residuals.

Pattern: use the model as a differentiable signal (perplexity, stability,
preference), not an oracle. haystack's pins already occupy the
tens-of-corrections regime without training; these matter only if pins
saturate.

## (c) Contradiction detection (local, cheap)

- GVD (Governed Versioning): embed -> cosine pre-filter -> Counterfactual
  Span Probing classification; F1 0.94 rule-level consistency claimed; no GPU.
- ConflictRAG (IEEE 2026): two-stage — embedding-MLP conflict classifier
  (interaction features [e_i; e_j; |e_i-e_j|; e_i*e_j], 88.7% F1 claimed),
  selective LLM only for uncertain pairs; 62% cost cut vs pure-LLM.
- Obsidian Contradiction Finder (plugin): the reference pipeline — embed
  note title + first ~1500 chars, cosine pairs >0.80, LLM verdict per pair
  via any OpenAI-compatible endpoint (incl. local Ollama).
- Pragmatic pipeline: chunk+embed -> cosine pre-filter -> NLI/small-LLM
  inversion. Failure modes: cosine blind to negation; contradicting
  passages may share zero content words. Suggested NLI:
  MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli.
- ContraDoc (NAACL 2024): pairwise NLI maxes at AUROC 0.666 whole-doc
  because candidate retrieval misses pairs (59.5% at K=20 — contradicting
  sentences share no content words); whole-document judges reach 0.854.
- CPU NLI: DeBERTa-v3-base (~184M) is the laptop-CPU winner in the 2026
  HaluEval CPU benchmark; NLI+similarity ensembles win.
- casual-memory pipeline: NLI pre-filter (92%+ accuracy) removes 70-85%
  of cases before LLM.

Consensus: cosine pairing is necessary but insufficient (blind to negation);
pairwise classification under-recovers at document level; two-stage
(pre-filter + selective verification) is the standard cheap design.
