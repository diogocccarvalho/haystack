# C-Sweep Results (2026-11-16)

## Parameters Swept

| Sweep | Values Tested | Winner |
|-------|--------------|--------|
| Dense weight | 0.3–0.9 | **0.8** (MRR 0.9567 → 0.9667 after promotion) |
| Reranker | on/off at each wv | **-rerank** wins (faster, no eval regression) — but disabled by default in prod |
| RRF K | 30,40,50,60,80,100,120 | **K=30** (MRR 0.9367, better than baseline 0.9267) |
| Superseded factor | 0.1–1.0 | Flat — no discrimination (few superseded files in vault) |

## Promotion Verdict

### PROMOTED: `ranking.rrf_k = 30` + `fusion.dense_weight = 0.8`

| Gate | Baseline | Promoted | Delta |
|------|----------|----------|-------|
| Eval MRR | 0.9267 | **0.9667** | +4pp |
| Eval R@3 | 1.000 | **1.000** | tie |
| Hard R@3 | 0.812 | **0.875** | +63pp |
| Hard MRR | 0.708 | **0.813** | +105pp |
| Latency | ~338ms | ~215ms | -123ms |

### No Promotion: reranker on/off, superseded_factor

- Reranking hurts MRR (0.9067 vs 0.9267 eval) and adds latency — not worth enabling
- Superseded factor sweep shows flat response (no vault files trigger the path)
- Dense weight alone (wv=0.8) dominates on eval but loses slightly on hard_pairs without K=30

## Config Changes (outside git):

```toml
[fusion]
dense_weight = 0.8          # was 0.7
[ranking]
rrf_k = 30                  # NEW section
```

## Gate Status

- **Eval**: R@3 1.000 / MRR 0.9667 ✓
- **Hard**: R@3 0.875 / MRR 0.8125 ✓
- **sim.py**: 78/78 passed ✓
- **Audit**: 22/22 passed ✓
