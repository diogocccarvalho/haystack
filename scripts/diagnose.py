#!/usr/bin/env python3
"""Diagnose distro-cluster nickname-query retrieval failure."""
import json, re, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from haystack import *

con = connect()

# The two distro-cluster queries from hard_pairs.json
distro_queries = [
    ("dude whats the fastest setup for this laptop again", "work/research/distro-alt-NEXT-STEPS.md"),
    ("that distro comparison thing, the v2 deep dive", "work/research/distro-alt-v2-cli-desktop.md"),
]

print("=" * 80)
print("DIAGNOSIS: Distro-cluster nickname-query retrieval failure")
print("=" * 80)

for query_text, expected_path in distro_queries:
    print(f"\n--- Query: '{query_text}' ---")
    print(f"Expected: {expected_path}")
    
    # Token analysis of original query
    orig_tokens = set(tokens(query_text)) - STOP
    norm_query = normalize_query(query_text)
    norm_tokens = set(tokens(norm_query)) - STOP
    
    print(f"Original tokens ({len(orig_tokens)}): {orig_tokens}")
    print(f"After normalize_query: '{norm_query}'")
    print(f"Normalized tokens ({len(norm_tokens)}): {norm_tokens}")
    
    # What filler words are being dropped?
    filler_words = ['dude', 'whats', 'again', 'that', 'the']
    found_fillers = [w for w in filler_words if w.lower() in query_text.lower().split()]
    print(f"Filler/corpus-common words present: {found_fillers}")
    
    # Run query
    result = rank(con, query_text, k=10)
    top_indices, all_paths, all_heads, all_texts, all_scores = result
    
    # Get actual returned results
    ranked_paths = [all_paths[i] for i in top_indices]
    
    print(f"\nTop 10 results:")
    for i, idx in enumerate(top_indices[:10]):
        p = all_paths[idx]
        s = float(all_scores[idx])
        snip = all_texts[idx][:80].replace('\n', ' ')[:50]
        print(f"  #{i+1}: {p}  score={s:.4f}  [{snip}...]")
    
    # Check if target is in top 3
    ranked_list = [all_paths[i] for i in top_indices]
    hit = any(p.startswith(expected_path) or p == expected_path for p in ranked_list[:3])
    print(f"\nHit in top 3: {hit}")
    if not hit:
        print(f"  Best position for '{expected_path}': index #{ranked_list.index(expected_path)+1 if expected_path in ranked_list else 'NOT FOUND'}")
    
    # Lexical overlap with target
    target_path = Path.home() / expected_path.lstrip('~')
    if target_path.exists():
        target_text = target_path.read_text(errors="replace").lower()
        tgt_toks = set(re.findall(r'\w{4,}', target_text))
        
        lex_overlap = orig_tokens & tgt_toks
        norm_overlap = norm_tokens & tgt_toks
        
        print(f"\nTarget file tokens (sample): listing distinctive 4+ token count...")
        # Count how many distinct 4+ token words are in target
        all_tgt_toks = set(re.findall(r'\w{4,}', target_text))
        print(f"Total distinct 4+ tokens in target: {len(all_tgt_toks)}")
        print(f"Orig query tokens intersecting target: {lex_overlap}")
        print(f"Norm query tokens intersecting target: {norm_overlap}")
        
        # Key question: does the target contain words like 'fastest', 'setup', 'laptop' etc?
        key_terms = ['fastest', 'setup', 'laptop', 'speed', 'tuning', 'distro', 
                     'comparison', 'deep', 'diver', 'v2', 'cachyos', 'ryzen']
        for kt in key_terms:
            found_in_orig = kt in query_text.lower()
            found_in_norm = kt in norm_query.lower()
            found_in_target = kt in target_text
            print(f"  '{kt}': in_query={found_in_orig}, after_norm={found_in_norm}, in_target={found_in_target}")
    else:
        print(f"WARNING: target file not found at {target_path}")
    
    # Dense similarity check (recompute from load_all)
    _, _, mat_all, texts_all_full = load_all(con)
    norm_text = normalize_query(query_text)
    try:
        q_vec = embed([norm_text])[0]
        dense_sim = mat_all @ q_vec
        
        # Find max dense sim
        max_sim_idx = int(np.argmax(dense_sim))
        max_dense_sim = float(dense_sim[max_sim_idx])
        print(f"\nMax dense similarity: {max_dense_sim:.4f} (path: {all_paths[max_sim_idx]})")
        
        # Dense rank of expected path
        if expected_path in all_paths:
            exp_idx = all_paths.index(expected_path)
            exp_dense = float(dense_sim[exp_idx])
            print(f"Dense sim for expected path '{expected_path}': {exp_dense:.4f}")
        
        # How many chunks have dense sim above 0.7 threshold?
        high_sims = np.sum(dense_sim > 0.7)
        print(f"Chunks with dense sim > 0.7: {high_sims}")
        print(f"Chunks with dense sim > 0.8: {np.sum(dense_sim > 0.8)}")
    except Exception as e:
        print(f"\nDense sim failed: {e}")

print("\n" + "=" * 80)
print("COMPARISON: normal literal query vs nickname query")
print("=" * 80)

# A normal query to compare
normal_query = "distro tuning next steps"
print(f"\nNormal query: '{normal_query}'")
top2, paths2 = query(con, normal_query, k=10)
