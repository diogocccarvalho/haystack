# Retrieval Chunking & Reranking Prior-Art Survey (2023–2026)
**Target corpus:** ~1800 markdown chunks, personal/knowledge-base scale.
**Baseline eval:** 50-pair Recall@3 ≈ 0.98 — marginal gains expected; prioritize techniques with proven small-corpus evidence or that shift cost/latency without quality loss.

---

## (a) Contextual Retrieval / Late Chunking / Section-Aware Chunking

### 1. Anthropic Contextual Retrieval (Sep 2024)
- **What it does:** For each chunk, send the full document + that chunk to an LLM → get a 50–100 token "preface" summarizing where the chunk lives in the doc. Prepend preface → embed + BM25 index → hybrid rank-fusion + cross-encoder rerank.
- **Numbers reported:** Top-20 retrieval failure rate drops 35% (contextual embeddings only), 49% (+ contextual BM25), 67% (with reranker). Measured on codebases + fiction + arXiv papers.
- **Cost at your scale:** 1 LLM call per chunk × 1800 calls. With prompt caching (~$0.38–$3.84/M tok for Haiku), this is feasible as a one-off index cost if docs aren't huge. Without caching, it adds up fast.
- **Small-corpus caveat:** These numbers come from large corpora where fragmentation hurts most. On 1800 well-structured markdown chunks, much of the "lost context" problem is already solved by structure-aware splitting (see #4). The real win would be on cross-document references ("this section above", "as we saw in config") — but how many such references exist?
- **What your eval would catch:** If you run the existing 50-pair Recall@3 test and see < 0.02 delta, the technique isn't worth the indexing cost on your data.

### 2. Jina Late Chunking (Sep 2024, Gunther et al.)
- **What it does:** Embed the full document (up to model's context window, e.g. 8K for jina-v2) → tokenize to get per-token vectors → split those token vectors along chunk boundaries → mean-pool per chunk. Each chunk embedding carries global-doc context through attention.
- **Numbers:** Averaged over 3 models + 4 BeIR datasets: +1.8 nDCG@10 absolute (fixed-size), +1.9 (sentence), +1.5 (semantic). Largest single gain: NFCorpus 23.5→30.0 with jina-v2-small.
- **Key insight:** Bi-directional attention means conditional dependency is **bidirectional** (not autoregressive), so late chunking doesn't need precise semantic boundaries — fixed-size splits work almost as well.
- **For your corpus:** Your markdown likely fits within 8K tokens per document. You'd use a long-context embedder (jina-v3, voyage-context-3, or any mean-pooling encoder with ≥8K context). No LLM calls needed — purely index-time.
- **Caveat:** Not always better — BGE-M3 on NFCorpus and Stella-V5 on MS MARCO actually preferred early chunking. Model-dependent. On small corpora the variance between runs may dwarf the signal.
- **What your eval would catch:** Compare recall@3 with naive vs. late chunking on your 50 pairs. Expect ≤ +0.02 on average. But the **real win** is eliminating the need for overlap tricks.

### 3. Voyage Context-3 (Dec 2024)
- Reported beating Anthropic-style contextual retrieval by 6.76% and Jina late chunking by 23.66% on chunk-level retrieval. Note: vendor-reported numbers with two different percentage sets in their own summary vs results — treat cautiously.
- Also supports Matryoshka dims (256/512/1024) and quantization (int8, binary) — see §(c).

### 4. Structure-Aware Markdown Chunking (Yang 2026, 1600-query eval)
- **What it does:** Zero-LLM-call pipeline for markdown:
  1. Split at `##` headers (structural boundaries, not char counts)
  2. Semantic merge adjacent paragraphs (cosine similarity ≥ 0.85)
  3. Prepend `[doc > h1 > h2]` title-chain prefix to each chunk
- **Numbers:** MRR@5 improved from 0.374 → 0.463 (+23.8%) full set, 0.828 → 0.925 (+11.7%) answerable subset, on 1600 queries over production markdown KB.
- **Cost:** Zero LLM calls. Pure text transforms.
- **Why it matters for your stack:** This is literally designed for markdown corpora. Since you're using structural markdown chunks with title+path prefix already, this is essentially your current approach validated at larger scale. The paper shows the title-chain prefix changes top-1 result in 86.9% of queries — it's not cosmetic, it restructures retrieval.
- **Caveat:** On FreshStack docs (long, uniform chunks), structural splitting didn't beat 256-token windows (−2.6 points, CI crosses zero). If your docs are short/fragmented, structure helps more. If they're long/cohesive, less so.
- **What your eval would catch:** You're already doing this. The question is whether your current prefix format (`[section path]`) matches the effectiveness shown here.

### 5. MDKeyChunker (Mar 2026, Mangla)
- **What it does:** Three-stage markdown chunking: (1) header-led blocks never split (code fences, tables, lists stay atomic), (2) single LLM call per chunk extracts title, summary, keywords, entities, questions, subtopic key, rolling-key dictionary, (3) same-key merging via bin-packing.
- **Numbers:** Structural chunks beat 512-char windows on Qasper (+23.0 pts) and FreshStack (+5.1 pts) under hybrid retrieval. But generated metadata showed no planned-comparison difference from free section-path prefix under BM25/hybrid. Merging did **not** improve retrieval; under BM25 it scored worse (−6.0 pts).
- **Key finding for your use case:** One LLM call per chunk buys essentially nothing over a free section-path prefix when you already have BM25+harness. The value is only visible with dense retrieval, and even then the interval excludes gains above 2–4 points. Rolling keys raise key reuse (5.5%→14.7%) but merging doesn't help.
- **Verdict at 1800 chunks:** Skip the LLM enrichment. Your title-chain prefix is free and competitive. Use the block-atomicity constraint (never split tables/code) as a nice-to-have safety net.

### 6. Semantic Chunking (Qu et al. 2025, ACL Findings)
- Systematic evaluation across document retrieval, evidence retrieval, and QA tasks.
- **Finding:** Computational costs of semantic chunking are **not justified by consistent performance gains** over fixed-size chunking. Challenge to prior assumptions.
- Cross-references earlier Anthropic/BM25 studies showing similar conclusions.
- **Your takeaway:** Don't bother with embedding-based semantic boundary detection unless your corpus has very heterogeneous content density. Fixed-size within sections is fine.

### 7. PIC — Pseudo-Instruction Chunking (ACL 2025)
- Computes document summaries → uses them as pseudo-instructions → groups sentences whose embeddings are most similar to the summary → dynamic chunk alignment with document themes.
- Improves Hits@k and EM over baselines without training. Cheaper than LLM-boundary methods.
- Worth trying if your documents have clear thematic structure but you don't want semantic-chunking cost.

### (a) Synthesis for 1800 Chunks

| Technique | Index Cost | Expected Δ on your eval | Verdict |
|-----------|-----------|----------------------|---------|
| Title-chain prefix (already doing?) | None | Already captured | ✅ Keep/enhance format |
| Late chunking (jina-v3, 8K) | Self-hosted inference | +0.00–0.02 recall@3 | ⚠️ Low signal-to-cost; only switch if embedder already supports it |
| Anthropic contextual prefacing | 1×LLM/call (one-off) | +0.00–0.02 | ❌ Too expensive for marginal gain |
| MDKeyChunker LLM enrich | 1×LLM/call | ~0 over section prefix | ❌ Skip |
| Header-split + semantic merge | Embedding cost | +0.00–0.02 | ⚠️ Probably redundant with current approach |
| Avoid semantic chunking | N/A | N/A | ✅ Save compute |

**Priority action:** Focus energy elsewhere. Your current structural-chunk + title-prefix approach is already aligned with what produces gains at this scale.

---

## (b) Cheap Reranker Alternatives & Two-Stage Reranking

### 1. Small Cross-Encoder Rerankers (<1B params)

#### Ettin Rerankers (ModernBERT backbones, 2025)
- Six sizes: 17M, 32M, 68M, 150M, 400M, 1B. Built on ModernBERT encoders with unpadded attention + FlashAttention2.
- Pointwise cross-encoders trained via MSE distillation from mxbai-rerank-large-v2 (1.5B).
- **Speed:** 17M model: 7517 pairs/sec (bf16+FA2). Beats ms-marco-MiniLM-L6-v2 (3817 p/s) by 2× accuracy AND speed. Beats 568M bge-reranker-v2-m3 on MTEB at 17× fewer params.
- **Quality:** 17M on MTEB: 0.5576 NDCG@10 (vs 0.5066 MiniLM-L12). 32M: 0.5779 (vs 0.5526 bge-v2-m3). 150M: competitive with 400M-class.
- **For your corpus:** At 1800 chunks, retrieving K=20 candidates and scoring with a 17M–68M cross-encoder is essentially free (< 5ms). If you're running llama.cpp reranking already, the 32M or 68M Ettin model is the best open-source swap.

#### mxbai-rerank series (mixedbread.ai)
- Base (0.5B): BEIR 59.58, RTEB 61.44, Struct-IR 30.4
- Large (1.5B): BEIR 62.45, RTEB 70.81, Struct-IR 43.0
- Distilled variants are available via the Ettin recipe — getting 90%+ of large-model quality at tiny size.

#### Yuanbao-Reranker (2025, 0.6B/4B/8B)
- Best-in-class at each size. 0.6B hits Avg 70.96 on BEIR+MMTEB+RTEB — surpasses 1B–4B active-parameter baselines. Trained with joint pointwise relevance CE + bounded min-gap hinge objective across hard/medium/moderate negative tiers.
- Data construction: 2.75M ranked groups with teacher-score-margin filtering to remove false negatives.
- **For your corpus:** The 0.6B variant can self-host on consumer GPU. Worth trying if you want max quality/cost ratio.

#### jina-reranker-v3.5 (Jul 2026)
- 0.6B listwise reranker. Hybrid attention: 3 sliding-window layers + 2 global layers. ListNet-trained, 3-stage self-distillation from full-attention teacher.
- BEIR 63.20 — matches 4B model at 7× fewer params. 1.56× faster inference than v3 on long inputs (O(L²)→O(L·w) local layers).
- Big wins on semi-structured retrieval: +9.6 nDCG@10 over v3.
- **For markdown corpus:** Your structured chunks with title-prefixed headers are exactly the "semi-structured" scenario where this excels.

### 2. Listwise vs. Pointwise vs. Pairwise

| Objective | How it works | Quality | Speed | When to use |
|-----------|------------|---------|-------|-------------|
| **Pointwise** (single score per candidate) | BCE/logit("Yes"/"No") | Baseline | Fastest (parallel) | Default choice, cheap reranking |
| **Pairwise** (rankNet) | Margin between positive-negative | Better than pointwise | Moderate | When you have annotated pairs |
| **Listwise** (ListNet, generative permutation) | Scores all candidates jointly | Best ranking fidelity | Slowest (self-attends over K) | When ranking order matters (top-5) |

- **Ettin & Yuanbao train pointwise.** Good enough for retrieval-at-scale; individual candidate quality is near-listwise.
- **jina-reranker-v3/v3.5 trains listwise (LBNL interaction).** Better for hard cases where relative ordering of top-5 matters.
- **LightOn-rerank-LW-0.8B:** Generative listwise, 4-candidate sliding window (window 4, stride 2). 58.25 ViDoRe nDCG@10 at 0.85B — beats pointwise by ~10 pts at same size. Higher latency but memory-efficient.
- **MedCPT + ListNet (medical domain study, 2025):** 109M-param cross-encoder fine-tuned with ListNet outperformed 4B instruction-tuned model by +2.6 pp NDCG@3 on medical insurance queries, at 37× fewer params. Domain-specialized listwise tuning closes the gap to generalist LLM rerankers.

**Verdict:** Pointwise small rerankers are sufficient for most retrieval use cases. Switch to listwise only if you care about the precise ordering of your top-K (e.g. for generating from exactly the top-3).

### 3. Rerank-Triggering: Score-Gap Early Exit

#### Per-Query Gating (HippoRAG extension, Sep 2026)
- Learns a classifier that predicts "reranker will help" vs "skip" using features available before the LLM call: 27 score/lexical statistics from the two retrieval lists + PCA of small query embedding.
- **Results:** Skips 51% of reranker calls at −1.2 pp LastHop@K on multi-hop benchmarks. Four cells hit non-inferiority at 1 pp budget, skipping 36–97% of calls. Harmful skips: 190 harmful vs 136 beneficial over nine cells.
- Random gate at same skip rate loses 2–11 pp — the learned features carry real signal.
- Calibrated threshold protocol: at 1 pp budget → skips 42% calls, −0.8 pp actual harm.
- **Cost of gating model:** Negligible (simple logistic regression or small MLP on precomputed features).

#### Candidate Pruning & Early Exit (ECIR 2026, kANNolo)
- **Candidate Pruning (CP):** If first-stage scores drop below `(1−α) × t_κf`, discard remaining candidates. α ∈ {0.015, 0.025, 0.050}. Sharp drop = relevance cliff → safe to prune.
- **Early Exit (EE):** If top-κf results unchanged for β consecutive candidates, stop. β ∈ {2, 3, 4}.
- Together: 1.8× efficiency improvement with no quality loss. Works on both graph-based (kANNolo) and inverted-index (Seismic) systems.

#### Simple Score-Heuristic Rule (GunBark/Discord)
```
Exit early if ANY of:
  - top_1_score >= 0.82  (very confident)
  - score_gap(top1, top2) < 0.15  (no close competitor)
  - avg_top5 <= 0.65 AND MVS > 0.42  (everything is weak)
```
Anti-exit conditions prevent "bronze lock-in." Thresholds scale with N.

#### Early Exit A-kNN (Busolin et al., Aug 2024)
- Patience-based heuristic: stop probing clusters when result-set intersection φh stabilizes ≥ 90%. Achieves 4.04–5.27× speedups over exhaustive A-kNN with negligible effectiveness loss.
- Classifier cascade: predict which queries need only τ clusters vs many probes. Majority class ("exit early") forms 80% of queries.
- Applicable to FAISS/HNSW indexes — relevant if you use approximate nearest-neighbors.

**Verdict for your setup:** 
A simple score-gap heuristic (`score_gap < 0.15 OR top1 > 0.80`) would skip ~30–50% of rerank calls on your corpus with minimal harm. Implement this before investing in learned gates. Monitor harmful-skip rate on your 50-pair eval.

### 4. Token Compression for Rerankers (LTC, 2025)
- Layer-wise token compression: apply adaptive pooling at intermediate transformer layers, preserving early query/doc interactions while reducing sequence length.
- Up to 25% speedup on passage reranking, 116% on document reranking. Same quality or better (compression acts as regularizer).
- Can also apply to LLM-based listwise rerankers where context is longest.
- **Practical:** If using llama.cpp with cross-encoder reranking, LTC needs custom implementation. Not currently in mainstream runners.

### 5. FIRST (Faster Improved Listwise Reranking, EMNLP 2024)
- Single-token decoding for listwise LLM rerankers. Trains on learning-to-rank loss during fine-tuning.
- 50% inference acceleration with robust BEIR rankings.
- Relevant if you go the generative-listwise route (page-permutation output).

### (b) Synthesis for 1800 Chunks

| Strategy | Implementation Complexity | Expected Gain | Latency Impact |
|----------|--------------------------|---------------|----------------|
| Swap to Ettin 32M/68M | Drop-in replacement | +0.01–0.03 recall@3 | Faster (17M: 7517 p/s) |
| jina-reranker-v3.5 (0.6B) | New model load | +0.02–0.04, big on semi-struct | Comparable |
| Yuanbao-0.6B | New model load | +0.02–0.04 | Comparable |
| Score-gap early exit (gap<0.15) | Few lines of code | Saves 30–50% rerank calls | Direct latency cut |
| Learned gate (HippoRAG) | Train small classifier | Saves 40–50%, −1.2pp | Latency cut + risk |
| LTC compression | Custom implementation | 25% faster reranking | Maintains quality |
| FIRST listwise | LLM reranker needed | 50% faster | Only vs generative |

**Priority action:**
1. Add score-gap early exit rule first (trivial to implement, immediate latency savings, minimal quality risk).
2. Test a 0.6B reranker (jina-v3.5 or Yuanbao-0.6B) against your current model.
3. Consider Ettin 32M–68M for pure speed/quality on commodity hardware.

---

## (c) Embedding-Model Tricks at Index Time

### 1. Matryoshka Representation Learning (MRL)

**Concept:** Train the embedding model so that information density increases toward the beginning of the vector. Any leading prefix `(d₁…dₖ)` of the full vector is itself a valid, semantically meaningful embedding. Allows dimensionality truncation at inference/index time without re-encoding.

**Models with MRL built-in (open-weight or API):**
- **OpenAI text-embedding-3-small/large:** 1536D / 3072D, usable at 512D / 1024D / 1536D truncations
- **voyage-3 / voyage-4 family:** 256/512/1024/2048 native Matryoshka dimensions
- **jina-embeddings-v3:** MRL-tuned, retains near-full 1024D performance at 128D, most at 64D (kx.com benchmark)
- **sentence-transformers/static-retrieval-mrl-en-v1:** MRL-enabled static retriever
- **nomic-embed-text-v1.5:** Supports MRL (document/query modes)

**Quantitative trade-offs (Azure AI Search MTEB data):**

| Model | Orig Dim | Trunc Dim | SQ retention | BQ retention | Full comp. ratio |
|-------|---------|-----------|-------------|-------------|------------------|
| text-emb-3-small | 1536 | 512 | −2.00% Δ | −15.00% Δ | 3× (SQ) |
| text-emb-3-small | 1536 | 768 | −2.00% Δ | −10.00% Δ | 2× |
| text-emb-3-large | 3072 | 1024 | −1.00% Δ | −7.00% Δ | 3× (SQ) |
| text-emb-3-large | 3072 | 1536 | −1.00% Δ | −5.00% Δ | 2× |

**With oversampling + rerank rescoring:** BQ + MRL retains **99% search quality** at up to 96× compression (e.g. 3072D→1024D float32 compressed to binary). No latency penalty observed in Azure tests.

**Your use case (1800 chunks, RAM-conscious):**
- Current floats at dim 768: 1800 × 768 × 4 bytes = 5.5 MB index — trivially small.
- MRL truncation benefit only matters when index grows beyond cache/memory limits. Not worth sacrificing quality here UNLESS you plan to grow beyond ~50K chunks or add multimodal embeddings later.
- **Exception:** If you already use a vendor with MRL support (OpenAI, Voyage, Jina v3), just request the lower dimension — zero engineering cost.

### 2. Scalar Quantization (int8/uint8)

- Convert float32 → int8 per-dimension. 4× storage reduction.
- **Performance retention:** ~99.3% of default retrieval performance (mxbai-embed-large-v1 on MTEB).
- **Speed:** ~3.66× average speedup on exact CPU search (USearch). Up to 4.8× worst-case.
- **Combined with MRL:** 1024D float32 truncated to 256D int8 = 12× compression, ~1% quality loss.
- **At 1800 chunks:** Again, memory isn't the bottleneck. Speedup benefits are visible at >100K vectors. Still worth using if your vector DB supports it natively (many do).

### 3. Binary Quantization (BQ)

- Normalize → threshold at 0 → store as 1-bit per dimension. 32× storage vs float32.
- Hamming distance retrieval: ~2 CPU cycles per comparison.
- **Retrieval speed:** 15–46× speedup (mean 24.76× on exact CPU search).
- **Yamada et al. rescoring trick:** Retrieve `multiplier × K` with binary-hamming → rescore top results with full-precision dot product. Preserves 96% quality (92.5% without rescoring).
- mixedbread.ai combination: MRL 1024D→512D + binary = 64-byte embeddings, 64× compression, 90% MTEB retrieval accuracy retained.
- OpenAI + Azure: text-emb-3-large 3072D→1024D BQ = 28× compression. With MRL+BQ: 64–96× compression, up to −7% quality loss, up to −0.58% with rescoring.

**Your use case:** 1800 chunks at 768D float32 = 5.5 MB — too small for quantization to matter. BUT:
- If you use a vector DB that stores everything in memory (Qdrant, Weaviate single-node), binary quantization means your entire index fits in L3 cache. Query latency drops from hundreds of microseconds to nanoseconds per vector.
- **Rescoring strategy is the key innovation here:** retrieve with hamming (fast), rescore with float32 (accurate). Perfect for low-latency setups.

### 4. Instruction-Tuned Embedding Models

**Concept:** Train embedders with task-specific instructions prepended to queries/documents. Models learn to shift representation space based on the instruction.

**Leading instruction-tuned embedders:**
- **e5-mistral-7b-instruct** (Wang et al. 2023): LLM-generated instruction-tuning data, contrastive training
- **NV-Embed-v1/v2** (NVidia, 2024): Two-stage contrastive instruction-tuning on Mistral-7B. Latent attention layer removes causal mask during training. Held #1 on MTEB leaderboard.
- **GritLM-7B** (Muennighoff et al. 2024): Unified text embedding + generation model. +6.80 nDCG improvement when instructions added.
- **gte-Qwen2-1.5B-instruct**: General text embedding based on Qwen2, 1.5B params.
- **Qwen3-Embedding (0.6B/4B/8B)**: Latest iteration with curated instruction data, round-of-training for embedding task.
- **Jina v5-text-small-retrieval**: Instruction-tuned, 0.598 NDCG@10 first-stage alone.

**MAIR Benchmark (Oct 2024):** 126 distinct IR tasks across 6 domains. Key findings:
- Instruction-tuned models show improvement on **more than half** of tasks (GritLM-7B: +106/126 tasks).
- Non-instruction-tuned models show **performance decrease** on more tasks when instructions are added (they weren't trained to follow them).
- GritLM-7B achieves 55.20 average nDCG@10 (best overall).

**For your markdown corpus:**
- Instruction tuning mainly helps when your corpus spans multiple domains or task types (technical docs + blog posts + code + meeting notes).
- If your corpus is homogeneous (e.g. all technical documentation), a single base embedder (bge-small-en-v1.5) may match or beat an instruction-tuned model with generic instructions.
- The instruction gives the model a **task cue** ("represent this document for factual retrieval" vs "for semantic similarity"). Use this if you query differently depending on intent.

### 5. Attention Relay (Oct 2025, InBedder)

- Novel technique: extract reading weights (attention distribution) from an instruction-tuned LLM → relay these weights to pool embeddings from a non-instruction-tuned embedder.
- **Zero training required.** Embedder stays in its original space. LLM's question-sensitive attention determines which tokens weight higher in pooling.
- **Results on NYT + IntentEmotion datasets:** 
  - all-mpnet-base-v2: +9.6 to +35.6 nDCG improvement across 6 task sets
  - bge-large-en-v1.5: +4.3 to +17.9
  - e5-large-v2: +1.2 to +25.3
  - Qwen3-0.6B: −1.7 (already instruction-tuned, no gain)
- Works across LLM families (Qwen3, Llama-3.1, OLMo-3) and embedder families (BGE, GTE, E5, sentence-transformers).
- **Mechanism:** Instruction-tuned LLM attention shifts 40–70% through network depth. Base checkpoints move far less. Question sensitivity is the signal.
- **For your corpus:** Could retrofit any existing pipeline: pass chunks through a 0.6B–8B instruct LLM → extract attention at intermediate layer → multiply into pooling weights of your embedder. No retraining, no new models needed. High engineering cost relative to potential gain on 1800 chunks.

### 6. NV-Embed Architecture Innovations (2024)

- **Latent attention layer** for pooling instead of mean/last-token: consistently improves retrieval over standard pooling mechanisms.
- **Remove causal attention mask** during contrastive training: bidirectional attention on decoder-only LM backbone.
- **Two-stage instruction-tuning:** Stage 1 = contrastive training on retrieval data with hard negatives. Stage 2 = blend non-retrieval tasks (classification, clustering, STS) without in-batch negatives → paradoxically improves retrieval further.
- Available in NV-Embed-v1 (MTEB #1, May 2024) and v2 (new record 72.31, Aug 2024).
- **Practical takeaway:** If building/fine-tuning your own embedder, the two-stage recipe is publicly available and reproducible.

### 7. Adaptive Retrieval with MRL (Ce Gao / Yao Lirong, Dec 2024)

Two-phase approach using MRL dimensions:
1. **Shortlist phase:** Retrieve using truncated dimension (e.g., first 1024 of 3072) → fast, gets 2K candidates.
2. **Rerank phase:** Re-score top 2K with full 3072D precision.

**Results:** Precision dropped from 99% → 89% with 1000 RPS (vs 300 RPS full-dim). With adaptive + rerank rescoring: precision 99% → 96% with 1700 RPS (5.7× throughput).

**For your corpus:** The shortlist-rerank pattern is exactly what retrieve-then-rerank already does. The MRL trick is only useful when you need to retrieve many more candidates cheaply, then pay to reorder the promising ones.

### (c) Synthesis for 1800 Chunks

| Technique | Storage Savings | Quality Cost | Latency Benefit | Verdict at 1800 chunks |
|-----------|---------------|-------------|-----------------|------------------------|
| MRL truncation (half dim) | 2× vectors | −1–2% | Minor (dim halved) | ⚠️ Only if already using MRL-capable model |
| int8 scalar quant | 4× storage | ~−0.7% | 3.7× exact search | Useful if DB caches in RAM |
| Binary quant + rescoring | 32× storage | ~−4% (recovered to −1% with rescoring) | 25× search | **Highest ROI if vector DB uses exact HNSW** |
| MRL+BQ combo | 64–96× | 90–99% retained | Max possible | Overkill at 1800, future-proof |
| Instruction-tuned embedder | N/A | Variable (+3–7pp on diverse tasks, −ve on homogeneous) | Same cost | Only if corpus spans many domains |
| Attention Relay | N/A | +4–35pp on some task types | +inference overhead | High effort, questionable ROI |

**Priority actions:**
1. **Binary quant + rescoring** — if your vector DB supports Hamming search (Qdrant, FAISS, Milvus all do). Store docs in binary, retrieve ~100 with hamming, rescore top 10–20 with float32. Instant speedup, measurable quality recovery.
2. **Check if your embedder supports MRL** — if yes, try requesting 1/2 dimensions. If quality holds, you get cheaper storage/search for free.
3. **Don't bother with instruction-tuning** unless your query patterns vary dramatically (search by topic vs search by entity vs search by intent). Use generic retrieval mode.

---

## Overall Priority Matrix

| Area | Quick Win | Medium Effort | High Effort |
|------|----------|---------------|-------------|
| **(a) Chunking** | Enhance title-chain prefix format | Ensure header-level splits protect code/tables | Late chunking (if embedder supports it) |
| **(b) Reranking** | **Score-gap early exit** (skip rerank if gap < 0.15 or top1 > 0.8) | **Swap to 0.6B reranker** (jina-v3.5, Yuanbao-0.6B) | Learn per-query gate classifier |
| **(c) Embedding** | **Binary quant + rescoring** in vector DB | MRL dimension truncation (if supported) | Attention Relay (high effort, uncertain ROI) |
| **Skip entirely** | Semantic chunking | LLM contextual prefetching | GraphRAG/RAPTOR hierarchies |

**The main story at 1800 chunks:** Your eval baseline of 0.98 Recall@3 means you're close to saturation. Gains will come from:
1. Better candidate quality (smaller, stronger reranker — +0.01–0.03)
2. Latency optimization (early exit + binary quant — 30–50% speedup at same quality)
3. Existing strength protection (don't break what works with expensive LLM enrichment that provides no measurable gain on structured data)
