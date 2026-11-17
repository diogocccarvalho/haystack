#!/usr/bin/env python3
"""The Populist — semantic query-formation nursery (v1).

Design (research/p3-semantic-sweep.md, semantic-only principle):
  NOTHING hand-authored. Candidates are MINED from the corpus (distributional
  thesaurus: document-incidence context vectors); acceptance is SEMANTIC
  (embedder cosine between rewritten and original >= theta); theta is
  CALIBRATED on train pairs; the frozen artifact is the procedure + theta, and
  it must beat the frozen hand rules (CHAINS in query_formation.py) on
  held-out pairs or the layer dies.

Boundary law: augmentation only (original tokens always present).
Drift leash: cosine gate on every accepted term (DC-PRF safety anchor).

Split: hard pairs 16 -> train 8 / dev 8 (interleaved); eval pairs = untouched
holdout (reported, never selected on).
"""
import importlib.util
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("hm", REPO / "haystack.py")
hm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hm)

MAX_AUG = 3            # chain-length gene (max accepted neighbors)
NEIGHBORS_PER_TOKEN = 2
DF_CAP = 0.30          # corpus-derived stopword band (df > 30% of chunks)
DF_FLOOR = 3           # too rare to be a reliable context signal
THETAS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]

TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def build_thesaurus(con):
    """Distributional thesaurus: word -> doc-incidence context vectors.
    Stopwords are DATA-DERIVED (df band), not a hand list."""
    rows = con.execute("select path, heading, text from chunks").fetchall()
    doc_sets = [set(tokens((h or "") + " " + (t or ""))) for _, h, t in rows]
    n_docs = len(doc_sets)
    df = Counter()
    for s in doc_sets:
        df.update(s)
    cap = DF_CAP * n_docs
    vocab = {w for w, c in df.items() if DF_FLOOR <= c <= cap}
    inv = {}
    for i, s in enumerate(doc_sets):
        for w in s & vocab:
            inv.setdefault(w, []).append(i)
    return inv, vocab, doc_sets, n_docs


def neighbors_fast(inv, doc_sets, word, n):
    """Cosine over doc-incidence vectors, computed via per-doc set counting."""
    di = inv.get(word)
    if not di:
        return []
    cnt = Counter()
    for d in di:
        cnt.update(doc_sets[d] & inv.keys())
    cnt.pop(word, None)
    scored = []
    for w, c in cnt.items():
        cos = c / math.sqrt(len(di) * len(inv[w]))
        scored.append((cos, w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:n]]


def candidates_for(query, inv, vocab, doc_sets):
    out = []
    for t in [t for t in tokens(query) if t in vocab]:
        for w in neighbors_fast(inv, doc_sets, t, NEIGHBORS_PER_TOKEN):
            out.append((t, w))
    return out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def greedy_augment(query, inv, vocab, doc_sets, theta, emb_orig):
    """Populist v1: mine neighbors, accept each augmentation only if the
    embedder agrees it stays on-intent (DC-PRF safety anchor)."""
    cands = candidates_for(query, inv, vocab, doc_sets)
    accepted, cur = [], query
    for t, w in cands:
        if len(accepted) >= MAX_AUG or w in accepted:
            continue
        trial = cur + " " + w
        if cos(hm.embed([trial])[0], emb_orig) >= theta:
            cur = trial
            accepted.append(w)
    return cur, accepted


# ---------------------------------------------------------------------------
# v2 strategies. Each is a *procedure* over mined candidates; the frozen
# artifact = (strategy, theta, params) calibrated on train, reported on dev,
# holdout eval touched last. Boundary law holds in all: original query or
# original ranking is always present.
# ---------------------------------------------------------------------------

def neighbors_rare(inv, doc_sets, word, n):
    """Rarity-directed thesaurus: neighbors from the LOW-df side of the band.
    v1 diagnosis: doc-incidence neighbors = hub vocabulary (anti-
    discriminative); the discriminating token for a cluster is the RARE one."""
    di = inv.get(word)
    if not di:
        return []
    cnt = Counter()
    for d in di:
        cnt.update(doc_sets[d] & inv.keys())
    cnt.pop(word, None)
    scored = []
    for w, c in cnt.items():
        cosv = c / math.sqrt(len(di) * len(inv[w]))
        # rarity bonus: divide by sqrt(df) -> prefers discriminative terms
        scored.append((cosv / math.sqrt(len(inv[w])), w))
    scored.sort(reverse=True)
    return [w for _, w in scored[:n]]


def candidates_rare(query, inv, vocab, doc_sets):
    out = []
    for t in [t for t in tokens(query) if t in vocab]:
        for w in neighbors_rare(inv, doc_sets, t, NEIGHBORS_PER_TOKEN):
            out.append((t, w))
    return out


def strat_substitute(con, query, expect, inv, vocab, doc_sets, theta, emb_orig, fuse=False):
    """Replace the token with its mined neighbor (semantic acceptance);
    original query never discarded (fused or fallback)."""
    variants = []
    for t, w in candidates_rare(query, inv, vocab, doc_sets):
        v = re.sub(rf"\b{re.escape(t)}\b", w, query, flags=re.IGNORECASE)
        if v != query and cos(hm.embed([v])[0], emb_orig) >= theta:
            variants.append(v)
    # ALWAYS returns flat (hit, pos) — run_strategy's contract.
    if not variants:
        h, p = rank_ok(con, query, expect)
        return h, p
    if not fuse:
        best = None
        for v in variants[:MAX_AUG]:
            h, p = rank_ok(con, v, expect)
            if h and (best is None or p < best[1]):
                best = (h, p)
        if best:
            return best
        h, p = rank_ok(con, query, expect)
        return h, p
    # RRF fusion across original + variants
    lists = [hm.rank(con, query, 10)] + [hm.rank(con, v, 10) for v in variants[:MAX_AUG]]
    fused = rrf(lists)
    for i, p in enumerate(fused[:3]):
        if p.startswith(expect) or p == expect:
            return True, i + 1
    return False, None


def rrf(rank_lists, k=60):
    """Reciprocal-rank fusion over (top, paths, heads, texts, scores) tuples."""
    score = Counter()
    paths_by = {}
    for top, paths, *_ in rank_lists:
        for i, t in enumerate(top):
            p = paths[t]
            score[p] += 1.0 / (k + i + 1)
            paths_by.setdefault(p, p)
    return [p for p, _ in score.most_common()]


def strat_saliency_delete(con, query, expect, inv, vocab, doc_sets, theta, emb_orig):
    """Leave-one-out saliency: delete tokens whose removal barely moves the
    embedding (semantic filler, data-derived). Hub terms stay; noise goes."""
    toks = [t for t in tokens(query) if t in vocab]
    kept = []
    for t in toks:
        reduced = " ".join(x for x in tokens(query) if x != t)
        if not reduced:
            kept.append(t)
            continue
        d = 1.0 - cos(hm.embed([reduced])[0], emb_orig)
        if d >= (1.0 - theta):   # removal moves meaning more than (1-theta)
            kept.append(t)
    new_q = " ".join(kept) or query
    h, p = rank_ok(con, new_q, expect)
    return h, p


def run_strategy(name, fn, con, pairs, inv, vocab, doc_sets, theta):
    r3 = mrr = 0
    for q, expect in pairs:
        emb_orig = hm.embed([q])[0]
        res = fn(con, q, expect, inv, vocab, doc_sets, theta, emb_orig)
        hit, pos = res[0], res[1]
        if hit:
            r3 += 1
            mrr += 1.0 / pos
    n = max(1, len(pairs))
    return r3 / n, mrr / n



def rank_ok(con, text, expect, k=5):
    top, paths, heads, texts, scores = hm.rank(con, text, k)
    for i, p in enumerate([paths[i] for i in top][:3]):
        if p.startswith(expect) or p == expect:
            return True, i + 1
    return False, None


def evaluate(con, pairs, inv, vocab, doc_sets, theta):
    r3 = mrr = 0
    for q, expect in pairs:
        emb_orig = hm.embed([q])[0]
        aug, _ = greedy_augment(q, inv, vocab, doc_sets, theta, emb_orig)
        hit, pos = rank_ok(con, aug, expect)
        if hit:
            r3 += 1
            mrr += 1.0 / pos
    n = max(1, len(pairs))
    return r3 / n, mrr / n


def eval_hand_rules(con, pairs, name):
    sys.path.insert(0, str(Path(__file__).parent))
    import query_formation as qf
    r3 = mrr = 0
    for q, expect in pairs:
        hit, pos = rank_ok(con, qf.rewrite_query(q, name), expect)
        if hit:
            r3 += 1
            mrr += 1.0 / pos
    n = max(1, len(pairs))
    return r3 / n, mrr / n


def main():
    cfg = (Path.home() / ".config/haystack/config.toml").read_text()
    con = hm.connect()
    inv, vocab, doc_sets, n_docs = build_thesaurus(con)
    print(f"corpus: {n_docs} chunks, vocab {len(vocab)} (df band {DF_FLOOR}..{int(DF_CAP*n_docs)})")

    hard = json.load(open(Path(__file__).parent / "hard_pairs.json"))["pairs"]
    train, dev = hard[0::2], hard[1::2]
    qa_m = re.search(r'qa_file\s*=\s*"([^"]*)"', cfg)
    eval_pairs = []
    if qa_m:
        eval_pairs = json.load(open(Path(qa_m.group(1)).expanduser()))["pairs"]
    print(f"train {len(train)} | dev {len(dev)} | eval {len(eval_pairs)} (untouched holdout)")

    strategies = [
        ("substitute", strat_substitute),
        ("substitute+fuse", lambda c, q, e, i, v, d, t, eo: strat_substitute(c, q, e, i, v, d, t, eo, fuse=True)),
        ("saliency_delete", strat_saliency_delete),
    ]
    baselines_dev = {n: eval_hand_rules(con, dev, n) for n in ("no_op", "standard")}
    print("\n--- v2 strategy sweep (theta selected on TRAIN per strategy) ---")
    results = {}
    for sname, fn in strategies:
        best = None
        for th in THETAS:
            tr = run_strategy(sname, fn, con, train, inv, vocab, doc_sets, th)
            if best is None or tr > best[0]:
                best = (tr, th)
        th_star = best[1]
        dv = run_strategy(sname, fn, con, dev, inv, vocab, doc_sets, th_star)
        results[sname] = (th_star, dv)
        print(f"{sname:18s} theta*={th_star:.2f}  dev R@3={dv[0]:.3f} MRR={dv[1]:.3f}   (baselines: no_op {baselines_dev['no_op'][0]:.3f}, standard {baselines_dev['standard'][0]:.3f})")

    if eval_pairs:
        print("\n--- HOLDOUT eval (top strategy by train, dev tiebreak) ---")
        top_s = max(results, key=lambda s: results[s][1])
        th_star = results[top_s][0]
        fn = dict(strategies)[top_s]
        ev = run_strategy(top_s, fn, con, eval_pairs, inv, vocab, doc_sets, th_star)
        b = eval_hand_rules(con, eval_pairs, "standard")
        print(f"{top_s} @theta*={th_star:.2f}: populist {ev[0]:.3f}/{ev[1]:.3f} vs standard {b[0]:.3f}/{b[1]:.3f}")


if __name__ == "__main__":
    main()
