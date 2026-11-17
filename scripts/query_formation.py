#!/usr/bin/env python3
"""Query-formation pass: deterministic rewrite operators before haystack.query().

Operators form a chain: text -> op1 -> op2 -> ... -> rewritten query.
Configurable via ~/.config/haystack/config.toml [query_formation] section."""

import re
import json
import sys
import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Fillers: discourse markers carrying no retrieval semantics
# ---------------------------------------------------------------------------
FILLER_WORDS = frozenset({
    'dude', 'bro', 'man', 'mate', 'hey', 'yo', 'eh',
    'whats', 'wont', 'cant', 'didnt', 'doesnt', 'isnt',
    'arent', 'couldnt', 'wouldnt', 'shouldnt',
    'gonna', 'wanna', 'gotta',
    'thats', 'heres', 'theres', 'wheres', 'whos', 'id',
    'ive', 'im', 'its',
    'again', 'already', 'still', 'yet', 'also',
    'stuff', 'things', 'bit', 'sort', 'type',
})

# ---------------------------------------------------------------------------
# Synonym expansion: colloquial -> technical synonyms (union, not replace)
# ---------------------------------------------------------------------------
SYNONYM_MAP = {
    'fastest': {'performance'},
    'speed': {'performance'},
    'slow': {'laggy', 'latency'},
    'setup': {'configuration', 'config', 'tune', 'tuning'},
    'configure': {'configuration', 'config', 'tune', 'tuning'},
    'tune': {'configuration', 'config', 'tune', 'tuning'},
    'comparison': {'compare', 'versus', 'contrast'},
    'compare': {'compare', 'versus', 'contrast'},
    'deep': {'thorough', 'extensive', 'comprehensive', 'detail'},
    'diver': {'thorough', 'extensive', 'comprehensive', 'detail'},
    'dive': {'thorough', 'extensive', 'comprehensive', 'detail'},
    'overview': {'summary', 'survey'},
}

# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------

def apply_filler_strip(text):
    """Remove filler words from query text. Expands contractions first."""
    t = text.lower()
    contraction_map = [
        (r'\bwhats\b', 'what'), (r"\bwont\b", "won't"),
        (r"\bcant\b", "can't"), (r"\bdidnt\b", "didn't"),
        (r"\bdoesnt\b", "doesn't"), (r"\bisnt\b", "isn't"),
        (r"\barent\b", "aren't"), (r"\bcouldnt\b", "couldn't"),
        (r"\bwouldnt\b", "wouldn't"), (r"\bshouldnt\b", "shouldn't"),
        (r"\bgonna\b", "going-to"), (r"\bwanna\b", "want-to"),
        (r"\bgotta\b", "got-to"), (r"\bthats\b", "that's"),
        (r"\bheres\b", "here's"), (r"\btheres\b", "there's"),
        (r"\bwheres\b", "where's"), (r"\bwhos\b", "who's"),
        (r"\bim\b", "I'm"), (r"\bive\b", "I've"), (r"\bits\b", "it's"),
    ]
    for pat, repl in contraction_map:
        t = re.sub(pat, repl, t)
    result_words = []
    for token in re.findall(r"[^\s]+", text):
        token_lower = token.lower()
        stripped = token_lower.rstrip(',.!?;:\'')
        if not stripped or stripped == token_lower:
            result_words.append(token)
            continue
        if stripped in FILLER_WORDS:
            continue
        prefix = token[:len(token) - len(stripped)]
        suffix = token[len(stripped):]
        result_words.append(prefix + stripped + suffix)
    return ' '.join(result_words).strip()


def apply_synonym_expansion(text):
    """Append synonym expansions as additional search tokens."""
    lower = text.lower()
    extra_terms = []
    for orig_key, expansions in SYNONYM_MAP.items():
        if orig_key in lower:
            extra_terms.extend(expansions)
    if not extra_terms:
        return text
    seen = set()
    unique_extra = []
    for t in extra_terms:
        if t not in seen:
            seen.add(t)
            unique_extra.append(t)
    return text + ' ' + ' '.join(unique_extra)


# ---------------------------------------------------------------------------
# Chain registry: name -> list of operator functions
# ---------------------------------------------------------------------------
# NOTE (2026-11-17, Diogo's design principle): the Populist is SEMANTIC-ONLY.
# No hand-authored regex ops or maps — everything semantic is discovered from
# data (vault co-occurrence mining, call logs, correction tables) and frozen
# only after measured promotion gates (DiamondHead crystallization). Regex
# ops written by hand = miscop; the semantic-only redesign is in
# diamondhead/research/p3-prior-art.md once the focused sweep lands.


def rewrite_query(text, chain_name='standard'):
    """Apply a named rewrite chain to query text."""
    ops = CHAINS.get(chain_name, CHAINS['standard'])
    result = text
    for op_fn in ops:
        result = op_fn(result)
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_ops(hm, con, pairs, ops):
    """Evaluate a list of operators on a pair list. Returns (r@3, mrr)."""
    recall = mrr_val = 0
    scored = 0
    for qtext, expect in pairs:
        rewritten = qtext
        for op_fn in ops:
            rewritten = op_fn(rewritten)
        top_indices, paths = hm.query(con, rewritten, k=5)
        ranked = [paths[i] for i in top_indices]
        scored += 1
        for rank_i, p in enumerate(ranked[:3]):
            if p.startswith(expect) or p == expect:
                recall += 1
                mrr_val += 1.0 / (rank_i + 1)
                break
    n = max(1, scored)
    return recall / n, mrr_val / n


# ---------------------------------------------------------------------------
# Main: genome sweep
# ---------------------------------------------------------------------------

def main():
    HM_PATH = Path('/home/diogo/Documents/code/haystack/haystack.py')
    vault_cfg = Path.home() / '.config/haystack/config.toml'

    cfg_text = vault_cfg.read_text()
    qa_match = re.search(r'qa_file\s*=\s*"([^"]*)"', cfg_text)
    qa_gen_file = Path(qa_match.group(1)).expanduser() if qa_match else None
    vault_match = re.search(r'path\s*=\s*"([^"]*)"', cfg_text)
    idx_match = re.search(r'db\s*=\s*"([^"]*)"', cfg_text)

    spec = importlib.util.spec_from_file_location("hm", HM_PATH)
    hm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hm)
    if vault_match:
        hm.VAULT = Path(vault_match.group(1)).expanduser()
    if idx_match:
        hm.DB = Path(idx_match.group(1)).expanduser()
    con = hm.connect()

    hard_path = Path(__file__).parent / 'hard_pairs.json'
    with open(hard_path) as f:
        hard_pairs = json.load(f)['pairs']

    eval_pairs = []
    if qa_gen_file and qa_gen_file.exists():
        with open(qa_gen_file) as f:
            eval_pairs = json.load(f).get('pairs', [])

    print("=" * 70)
    print("QUERY-FORMATION PASS: Genome sweep")
    print("=" * 70)
    print(f"Hard pairs: {len(hard_pairs)} | Eval pairs: {len(eval_pairs)}")

    results = {}
    base_ops = CHAINS['no_op']
    br3, bmrr = evaluate_ops(hm, con, hard_pairs, base_ops)
    print(f"\nBASELINE (no rewrite): hard R@3={br3:.4f}, MRR={bmrr:.4f}")
    results['baseline'] = [br3, bmrr]

    er3_b, emrr_b = 0, 0
    if eval_pairs:
        er3_b, emrr_b = evaluate_ops(hm, con, eval_pairs, base_ops)
        print(f"BASELINE eval: R@3={er3_b:.4f}, MRR={emrr_b:.4f}")
        results['baseline'].extend([er3_b, emrr_b])

    # Single-operator sweep
    print("\n--- SINGLE OPERATOR SWEEP ---")
    single_ops = [('filler_only', [apply_filler_strip]), ('synonym_only', [apply_synonym_expansion])]
    for op_name, ops in single_ops:
        r3, mrr = evaluate_ops(hm, con, hard_pairs, ops)
        delta_h = r3 - br3
        status = "OK"
        if eval_pairs:
            er3, emrr = evaluate_ops(hm, con, eval_pairs, ops)
            delta_e = er3 - er3_b
            if delta_e < -0.01:
                status = "REGRESSION!"
            print(f"  {op_name}: hard R@3={r3:.4f} MRR={mrr:.4f} (ΔR@3={delta_h:+.4f}) "
                  f"| eval R@3={er3:.4f} MRR={emrr:.4f} (Δ={delta_e:+.4f}) [{status}]")
            results[op_name] = [r3, mrr, er3, emrr]
        else:
            print(f"  {op_name}: hard R@3={r3:.4f} MRR={mrr:.4f} (ΔR@3={delta_h:+.4f}) [{status}]")
            results[op_name] = [r3, mrr]

    # Combo sweep
    print("\n--- COMBO SWEEP ---")
    combos = [('filler+synonym', [apply_filler_strip, apply_synonym_expansion])]
    best_overall = None
    best_score = (-999, -999)

    for cname, ops in combos:
        r3, mrr = evaluate_ops(hm, con, hard_pairs, ops)
        delta_h = r3 - br3
        status = "OK"
        er3, emrr = 0, 0
        delta_e_val = 0
        if eval_pairs:
            er3, emrr = evaluate_ops(hm, con, eval_pairs, ops)
            delta_e_val = er3 - er3_b
            if delta_e_val < -0.01:
                status = "REGRESSION!"
        print(f"  {cname}: hard R@3={r3:.4f} MRR={mrr:.4f} (ΔR@3={delta_h:+.4f}) "
              f"| eval R@3={er3:.4f} MRR={emrr:.4f} (Δ={delta_e_val:+.4f}) [{status}]")
        results[cname] = [r3, mrr, er3, emrr]
        if status != "REGRESSION" and (r3 + er3) > best_score[0]:
            best_score = (r3 + er3, mrr + emrr)
            best_overall = (cname, ops)

    # Detailed per-query analysis for best chain
    if best_overall:
        print("\n" + "=" * 70)
        print("BEST CHAIN DETAIL")
        print("=" * 70)
        cname, cops = best_overall
        print(f"Chain: {cname}")
        print(f"Ops: {[op.__name__ for op in cops]}")

        br3_v, bmrr_v = evaluate_ops(hm, con, hard_pairs, base_ops)
        cr3, cmrr = evaluate_ops(hm, con, hard_pairs, cops)

        er3_bv, _ = (0, 0)
        cr3_e, cmrr_e = (0, 0)
        if eval_pairs:
            er3_bv, _ = evaluate_ops(hm, con, eval_pairs, base_ops)
            cr3_e, cmrr_e = evaluate_ops(hm, con, eval_pairs, cops)

        print(f"Hard:   R@3={cr3:.4f}, MRR={cmrr:.4f}  (baseline {br3_v:.4f}/{bmrr_v:.4f}, ΔR@3={cr3-br3_v:+.4f})")
        if eval_pairs:
            print(f"Eval:   R@3={cr3_e:.4f}, MRR={cmrr_e:.4f}  (baseline {er3_bv:.4f}, ΔR@3={cr3_e-er3_bv:+.4f})")

        # Show per-query detail
        print("\nPer-query improvement detail:")
        for qtext, expect in hard_pairs:
            old_top, old_paths = hm.query(con, qtext, k=5)
            new_top, new_paths = hm.query(con, rewrite_query(qtext, cname), k=5)

            old_ranked = [old_paths[i] for i in old_top]
            new_ranked = [new_paths[i] for i in new_top]

            old_hit = any(p.startswith(expect) or p == expect for p in old_ranked[:3])
            new_hit = any(p.startswith(expect) or p == expect for p in new_ranked[:3])
            old_pos = next((i+1 for i, p in enumerate(old_ranked) if p.startswith(expect) or p == expect), 'N/A')
            new_pos = next((i+1 for i, p in enumerate(new_ranked) if p.startswith(expect) or p == expect), 'N/A')

            marker = ""
            if not old_hit and new_hit:
                marker = " FIXED"
            elif old_hit and not new_hit:
                marker = " REGRESSED"
            elif old_hit and new_hit:
                marker = " OK unchanged"
            elif not old_hit and not new_hit:
                marker = " MISS unchanged"

            print(f"  '{qtext}'")
            print(f"    want={expect}")
            print(f"    old #{old_pos} ({'HIT' if old_hit else 'MISS'}) -> new #{new_pos} ({'HIT' if new_hit else 'MISS'}){marker}")

    # Save results
    out_path = Path(__file__).parent / 'rewrite_results.json'
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
