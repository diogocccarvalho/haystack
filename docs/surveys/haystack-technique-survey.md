# Hybrid Retrieval & Query Processing Survey (2023–2026)

For haystack (~1800 chunks, single-user, ~4ms embedding + ~130ms cross-encoder rerank, RRF 0.7/0.3). No query-time LLMs; local 1.7B models available as separate services. Budget: ~10 ms/query extra.

---

## (A) FUSION IMPROVEMENTS OVER RRF 0.7/0.3

### 1. Learned convex combination (TM2C2) — Bruch et al. SIGIR 2023
**What:** Tune a single α parameter on theoretical min-max normalized scores: `score = α·s_dense_norm + (1−α)·s_lexical_norm`. Beats equal-RRF in-domain and out-of-domain. Best variant TM2C2 (min-max over theoretical bounds, α tuned) hit 0.454 nDCG@1000 vs 0.425 for RRF k=60 on MS MARCO, 0.699 vs 0.675 on HotpotQA.
**Evidence:** Bruch et al., "An Analysis of Fusion Functions for Hybrid Retrieval", ACM TOIS 2023 / SIGIR. Tested across multiple datasets.
**Cost:** One hyperparameter. Train on ~50–100 human-labeled queries per domain (or use personal eval pairs already available). Implementation: ~10 lines of Python. Score normalization must be consistent (theoretical min-max recommended over within-batch).
**Worth it?** Yes. This directly addresses your 0.7/0.3 weight being fixed. The gap between RRF and tuned CC is small but consistent. With 50 eval pairs you can grid-search α once and lock it.

### 2. Rank–Score Mix (α, λ, κ parameterized) — Sept 2026
**What:** A unified fusion family combining score-based mixing (`α·s̃_d + (1−α)·s̃_s`) with rank-kernel mixing (`αρ(r_d) + (1−α)ρ(r_s)`), controlled by three parameters: dense prior α ∈ [0,1], score-vs-rank mix λ ∈ [0,1], RRF smoother κ. Special cases include pure RRF (λ=0) and pure CombSUM (λ=1).
**Evidence:** "Parameterized Dense-Sparse Fusion", Sept 2026. Grid search on SciFact: optimal (α=0.8, λ=0.75, κ=20) → nDCG@10 = 0.753 vs dense BGE 0.742 and RRF 0.707. Also tested on all 20 BEIR indexed units: beats equal RRF 20/20, beats dense 16/20. List-conditioned α adds +0.0006 nDCG (marginal). Router that skips sparse when query is "dense-only" was **rejected**.
**Cost:** Three hyperparameters. Grid search needs labeled queries but the search space is tiny (3 values × 3 values × 3 values = 27 combos). Implementation: minimal Python.
**Worth it?** Marginal win (+0.011 nDCG@10 over dense, +0.046 over equal RRF). Given your stack already has bge-small + BM25, try a simple tuned CC first (cheaper). Upgrade to rank-score mix only if you want maximal optimization.

### 3. Per-query adaptive IDF weighting on RRF — vstash (local-first, Feb 2026)
**What:** Compute mean IDF of stemmed query terms; map through sigmoid to get w_v/w_f per query. Rare/technical terms → higher lexical weight. Common vocabulary → higher vector weight. Uses existing inverted index vocab — no extra model needed.
**Evidence:** vstash paper, Feb 2026. +21.4% nDCG@10 on ArguAna, improved on all 5 BEIR datasets vs fixed weights. Self-supervised embedding refinement via disagreement signals also reported (+19.5% NFCorpus).
**Cost:** Very low. You already have BM25/tokenization. Extract query term IDs from FTS or jieba, look up IDF from the same table BM25 uses, apply sigmoid: `w_v = σ(c − mean_idf)`. Pure Python, O(num_query_terms).
**Worth it?** High signal-to-cost ratio. Directly implements the routing intuition that product-name/scode queries should lean lexical while conceptual questions should lean dense. Zero new dependencies.

### 4. The "Fusion Tax" caveat — Oct 2026
**What:** When hybrid retrieval actually *hurts* individual queries. Even best method (dense-weighted interpolation) degrades below the better component on 13–48% of queries and below dense-only on 8–21%. The tax is mostly ranking-level (not recall) and concentrates on queries only one retriever covers. RRF carries the highest tax where dense is clearly stronger. Cross-encoder recovery is limited.
**Evidence:** "The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts", Oct 2026. Seven BEIR corpora. Tested Anserini BM25 + five dense encoders + SPLADE + cross-encoder second stage.
**Implication for you:** If your bge-small dominates (which your Recall@3 0.98 suggests), the marginal gain from adding BM25 fusion might be <1% and could hurt some queries. Consider measuring per-query fusion benefit on your 50 eval pairs before optimizing further.

### 5. XViewRank — Overlap-conditioned MLP fusion (Oct 2026)
**What:** Keeps view-specific BM25 scores and result-set overlaps per query; trains a tiny nonlinear scorer (128→64 hidden MLP) to fuse based on both. For multi-view queries (original + hypothetical + step-back views). Improves MAP statistically significantly on TREC DL2019.
**Evidence:** "XViewRank", Oct 2026. Beats linear scores with overlap by +0.012 MAP, beats LambdaMART on same features.
**Cost for you:** Needs labeled training data and an MLP (numpy implementation trivial). Not worth it standalone if you're not doing multi-view expansion (section B). But the **overlap-aware concept** transfers: if you expand a query into variants, tracking which docs appear in how many variants is a free signal that could conditionally boost fusion weights.

---

## (B) QUERY PROCESSING

### 1. HyDE (Hypothetical Document Embeddings) — Gao & Callan ACL 2023
**What:** Generate a fake document answering the query with an LLM → embed it → use for dense search. Replaces original query embedding entirely.
**Evidence:** Original paper outperformed Contriever zero-shot across 11 query sets (web search, QA, fact verification, non-English). Follow-ups confirm.
**Latency cost:** Requires an LLM call at query time. Using a local 1.7B model: ~100–500ms depending on length/context window. **This alone blows your 10ms budget by 10–50×.**
**When it works:** Semantic gap scenarios — short informal query vs formal documentation. Elasticsearch internal test: +50% precision/recall on short casual queries against formal papers.
**When it fails:** Exact keyword lookups ("A17 Pro"). A recent analysis found HyDE **dilutes** embeddings for queries that already have precise identifiers (product names, error codes, proprietary facts). It spreads attention across too many concepts.
**Mitigation — HyDE Vector Mix:** UTokyo at TREC RAG 2025 weighted average: `(1−α)·emb(query) + α·emb(hypothetical)`, α=0.7 empirically optimal. Preserves original query signal while adding semantic depth. nDCG@10: 0.493 → 0.564 (+14.4%) with BGE-small. Still requires LLM call.
**HyPE (index-time alternative):** Jul 2026. Precomputes hypothetical prompts per chunk during indexing. **Zero query-time LLM cost.** Improves context precision +42pp. For you: would require re-indexing but eliminates runtime cost entirely. Worth investigating.
**Verdict:** Requires LLM at query time. Budget: ~100–500ms. Only worth it if your eval pairs show systematic semantic-gap failures (short/informal queries missing formal docs). HyPE is the zero-cost alternative.

### 2. Multi-Query Expansion — Wang et al. (Query2doc/RAG-Fusion lineage)
**What:** LLM generates 3–5 query variants from different angles → parallel retrieve → RRF merge. Documents appearing across multiple variants rank higher.
**Evidence:** Consistently shown to increase recall. RAG-Fusion paper shows consensus-building effect reduces hallucination. However, one study ("Not All Queries Need Rewriting", Mar 2026) found single-step rewriting **degraded** FiQA nDCG@10 by 9.0% because rewritten queries substituted away domain-specific terms. Improvement on TREC-COVID (+5.1%, p=0.024). Mechanism: rewriting substitutes domain terms in 95% of cases; direction determines effectiveness.
**Latency cost:** LLM call (same as HyDE: ~100–500ms with local 1.7B). Plus 3–5× more retriever calls (mitigated by fan-out).
**Verdict:** High latency, questionable ROI at ~10ms budget. Could improve recall but may degrade precision on well-formed domain queries. Only worth it if you have clear evidence of vocabulary-mismatch failures in your eval set.

### 3. Rewrite-Retrieve-Read — Ma et al. NeurIPS 2023
**What:** LLM rewrites user query into cleaner/more specific form → retrieve → answer from originals. Single rewrite.
**Evidence:** Improves retrieval when original query is noisy/informal. Same degradation risk as multi-query on well-formed domain queries (Mar 2026 finding above: 95% of rewrites substitute terms).
**Latency cost:** Single LLM call. ~50–200ms local 1.7B.
**Verdict:** Same LLM requirement. Low-latency local model might make it feasible (~100ms), but still exceeds 10ms budget. Usefulness depends on your query patterns.

### 4. Step-Back Prompting — Zheng et al. arXiv 2023
**What:** Generate a broader, more abstract question → retrieve with both step-back and original → combine results. E.g., "What timeout for payment webhooks?" → "How does payment webhook configuration work?"
**Evidence:** Retrieves foundational/conceptual context that complements granular snippets. Shown helpful for technical documentation retrieval where users ask specific config questions but need architecture context.
**Latency cost:** One LLM call (~50–200ms with local 1.7B).
**Verdict:** Same LLM requirement. Only useful if your users ask narrow operational questions that need broader conceptual grounding.

### 5. STORM — Stepwise Token Optimization for Lexical Expansion (Nov 2025)
**What:** Trains a 0.6B–8B LM to generate query expansions via reward-guided beam search, using BM25 retrieval feedback as the reward signal. At inference: rewrites query for BM25 matching. Retrieves as fast as plain BM25.
**Evidence:** Matches or beats competitive LLM rewriters on TREC DL and BEIR. Transfers zero-shot to 18 languages. Key difference: generation is guided by retrieval feedback at token level, not decoupled prompting.
**Latency cost:** Inference: one forward pass through 0.6B–8B model (~10–100ms depending on size). Much cheaper than HyDE/multi-query since output is just an expanded query string, not a paragraph.
**Verdict:** The most promising technique for sub-200ms budget without blowing past 10ms dramatically. Would need fine-tuning on a local backbone (possibly yours), but at inference it's essentially just generating a few extra terms. Worth prototyping if vocabulary mismatch is a known failure mode.

### 6. RM3 Pseudo-Relevance Feedback (classic, still relevant)
**What:** Take top-K retrieved documents, extract expansion terms via BM25/MF scoring, append to query, re-run BM25.
**Evidence:** Decades-old IR technique, still baseline for PRF. Improves BM25 typically 5–15% nDCG.
**Latency cost:** Negligible. Already running BM25. Just two passes over top-K results + re-extract. Pure Python/numpy.
**Verdict:** Worth trying. Zero new dependencies, fits budget perfectly. Often overlooked in neural-era papers but remains effective for lexical expansion. Your BM25 already handles this.

### Summary Table — Query Processing Techniques

| Technique | LLM required? | Latency (est.) | Fits 10ms? | ROI potential |
|-----------|--------------|----------------|------------|---------------|
| RM3 PRF | No | ~1ms | ✅ Yes | Medium (lexical) |
| Adaptive IDF RRF | No | ~1ms | ✅ Yes | Medium-High (routing) |
| STORM (fine-tuned) | Optional (training only) | ~10–100ms | ⚠️ Borderline | Medium-High |
| Rewrite-Retrieve-Read | Yes (query-time) | ~50–200ms | ❌ No | Variable |
| Step-Back Prompting | Yes (query-time) | ~50–200ms | ❌ No | Niche |
| Multi-Query Expand | Yes (query-time) | ~100–500ms | ❌ No | Risky (can degrade) |
| HyDE (original) | Yes (query-time) | ~100–500ms | ❌ No | Good when gap exists |
| HyPE (index-time) | No (precomputed) | 0ms | ✅ Yes | High (zero runtime cost) |

---

## (C) META-QUERY HANDLING (QUERIES ABOUT THE INDEX ITSELF)

### 1. Metadata extraction from queries (self-querying) — widely documented 2023–2025
**What:** Parse user query to extract structured filter values (date range, source, author, category) → apply as metadata filter before retrieval. Examples: LangChain SelfQueryRetriever, Elasticsearch metadata filtering.
**Evidence:** Systematic study ("Utilizing Metadata for Better RAG", Jan 2026) showed prefixing metadata improves plain-text baselines consistently, unified dual-encoder sometimes better than prefixing, lateral-fusion retrieval matches accuracy with easier maintenance. Structured metadata filters narrow candidate space effectively.
**Cost:** Requires you to attach meaningful metadata to each chunk at ingest time. Extraction can be done offline (file-based metadata, directory structure parsed) or via lightweight regex/pattern matching. No LLM needed.
**Verdict:** Very high ROI. Your chunks likely come from files/folders/dates anyway. Just add these as SQLite columns (already using SQLite) and filter early. For example: if user asks "notes about meeting X in December", filter by `source LIKE '%meeting%' AND month = 'december'` before any retrieval. Cuts candidate pool dramatically.

### 2. Directory-scoped hierarchical retrieval (TrieHI) — Jun 2026
**What:** Native trie index over directory paths enabling recursive subtree search without query-time path expansion. DSQ (directory-semantic query) supports confined/restricted scope searches.
**Evidence:** ByteDance Viking engine. Shows flat metadata filtering can't express hierarchical scopes efficiently; trie-based traversal is O(path_length) vs O(expansion) for naive approaches.
**Relevance to you:** Your notes are presumably organized hierarchically (folders, tags, sources). You don't need a trie implementation; you already have tree structure via filesystem or DB parent-child relationships. The key insight: **scoping retrieval to a subtree before searching is cheap and correct**. Filter first, search later.

### 3. Knowledge Graph neighbour expansion — DocuSearch (Sept 2026)
**What:** Add a graph/edge layer where nodes link related chunks. Search pulls KG neighbors as a third signal in RRF.
**Evidence:** Telecom production system: Precision@10 0.69, Recall@10 0.79, grounding rate 89.6%. KG signal modest but consistent — surfaces structural neighbours neither cosine nor BM25 ranks highly.
**Cost for you:** Requires building and maintaining edges. Could be done via co-reference resolution (shared entities across chunks) or document relationship metadata (e.g., "this note references that meeting"). Non-trivial upfront but reusable.
**Verdict:** Only worth it if your knowledge base has explicit cross-references. Less relevant for personal note-taking unless you manually link things.

### 4. HySE (Hypothetical Schema Embeddings) — Oct 2026
**What:** Analogous to HyDE but for tabular/metadata retrieval. Generates hypothetical table schemas from task descriptions → embed → retrieve datasets. Used alongside FlexPrune (NL→SQL metadata filter).
**Evidence:** KaggleDS corpus, +6pp Recall@10 over dense baselines.
**Relevance to you:** Conceptually useful: if your index has typed metadata fields, treat the query's mention of those fields as self-querying rather than letting them get absorbed into the embedding. Always check whether a query portion maps to a metadata field BEFORE embedding.

### Recommended Meta-Query Strategy
1. **Always run metadata filter first** if query contains identifiable entity/date/source. Regex-based extraction of common patterns (dates, project names, file extensions) costs <1ms.
2. **Don't embed meta-terms.** Strip date/entity tokens from the query text before embedding — let BM25 handle exact match and embedding handle the semantic remainder. This prevents the embedding from being diluted with noise like "in 2024" or "about file X".

---

## RECOMMENDATIONS FOR HAYSTACK (prioritized by cost/benefit)

### Tier 1 — Implement now (low cost, proven gain)
1. **Tune α on convex combination** using your 50 eval pairs. Replace fixed 0.7/0.3 with learned α. Cost: minutes of code. Expected gain: +1–3% NDCG.
2. **Adaptive IDF weighting** on RRF. Weight rare-term-heavy queries toward BM25, common-vocab toward dense. Cost: ~20 lines. Expected gain: +1–2% NDCG on diverse queries.
3. **Metadata pre-filtering**. Strip identifiable metadata tokens from query before embedding; use them as SQL filters. Cost: parsing logic. Expected gain: large for meta-queries, negligible for content queries.

### Tier 2 — Investigate next (medium effort, potential significant gain)
4. **RM3 pseudo-relevance feedback** on BM25. Two-pass BM25. Cost: 30 lines. Expected gain: +5–10% on lexical queries.
5. **HyPE index-time approach**. During chunking, generate 2–3 hypothetical questions per chunk offline. Store those embeddings alongside chunk embeddings. Query always stays the same. Cost: re-indexing + storage (~10ms extra per chunk during build, 0ms at query time). Potential gain: major on semantic-gap queries.

### Tier 3 — LLM-dependent, evaluate case-by-case
6. **HyDE with vector mix** — only if your eval shows systematic short-query failures. Local 1.7B model available, so ~100ms budget. Measure on your 50 pairs first.
7. **Rewrite-Retrieve-Read** — same evaluation criteria. Be aware of degradation risk on well-formed queries (Mar 2026 paper).
8. **STORM fine-tuning** — interesting long-term. Fine-tune a small LM on query-expansion using your eval pairs as reward signal. Deployment would be ~10ms inference.

### What NOT to do right now
- **Cross-encoder reranking** is already your pipeline (~130ms). Don't add more post-fusion stages.
- **MMR deduplication** after RRF failed to improve in vstash benchmark experiments. Skip.
- **Frequency+decay scoring** on top of RRF failed to improve NDCG (-1.6%). Skip.
- **Router that skips one retriever** was rejected by train data (skipping ~50% queries lost nDCG). Keep both retrievers always running.

---

## KEY PAPERS REFERENCED

1. Bruch et al. "An Analysis of Fusion Functions for Hybrid Retrieval." ACM TOIS 2023.
2. Gao & Callan. "Precise Zero-Shot Dense Retrieval Without Relevance Labels." ACL 2023 (HyDE).
3. Zhang et al. "At TREC RAG 2025: HyDE Vector Mix." TREC 2025.
4. "The Fusion Tax." Oct 2026. (arXiv 2610.30612)
5. "Parameterized Dense-Sparse Fusion." Sept 2026. (arXiv 2609.22770)
6. vstash. "Local-First Hybrid Retrieval with Adaptive Fusion." Feb 2026.
7. "Not All Queries Need Rewriting." Mar 2026. (arXiv 2603.13301)
8. Ma et al. "Is Reciprocal Rank Fusion All You Need?" 2024.
9. Lin & Lin. "A Dense Representation Framework for Lexical and Semantic Matching." 2023 (DLR/DHR).
10. XViewRank. "Overlap-Conditioned Score Fusion." Oct 2026.
11. STORM. "Stepwise Token Optimization." Nov 2025.
12. HyPE. "Hypothetical Prompt Embeddings." Jul 2026.
13. Yousuf et al. "Utilizing Metadata for Better RAG." Jan 2026.
14. TrieHI. "Directory-Aware Query and Maintenance." Jun 2026.
15. XViewRank. "Overlap-Conditioned Score Fusion." Oct 2026.
