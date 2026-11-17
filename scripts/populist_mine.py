#!/usr/bin/env python3
"""The Populist — pattern miner (v2.5, Diogo's design).

'Send them all and find the pattern that wins': enumerate a large
rearrangement space per training query (single-token substitutions to BOTH
sides of the thesaurus, token deletions, original), score every variant
against the expected path, then MINE the outcome table: which transformation
features separate winners from losers? The induced pattern is then frozen as
a selector and validated on dev (never seen during induction) and eval
(untouched holdout).

Features logged per variant (all data-derived, nothing hand-authored):
  op (substitute/delete/original), token_df, neighbor_df, df_ratio,
  cos (rewrite vs original), rel_pos (token position / len), q_len.
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

TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")
NEIGHBORS = 3            # both directions
DF_CAP = 0.30
DF_FLOOR = 3


def tokens(text):
    return TOKEN_RE.findall(text.lower())


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def build(con):
    rows = con.execute("select path, heading, text from chunks").fetchall()
    doc_sets = [set(tokens((h or "") + " " + (t or ""))) for _, h, t in rows]
    df = Counter()
    for s in doc_sets:
        df.update(s)
    vocab = {w for w, c in df.items() if DF_FLOOR <= c <= DF_CAP * len(doc_sets)}
    inv = {}
    for i, s in enumerate(doc_sets):
        for w in s & vocab:
            inv.setdefault(w, []).append(i)
    return inv, vocab, doc_sets, df


def neighbors_both(inv, doc_sets, word, n=NEIGHBORS):
    """Two directions: common-side (high cos) and rare-side (cos / sqrt(df))."""
    di = inv.get(word)
    if not di:
        return {}
    cnt = Counter()
    for d in di:
        cnt.update(doc_sets[d] & inv.keys())
    cnt.pop(word, None)
    scored = []
    for w, c in cnt.items():
        scored.append((c / math.sqrt(len(di) * len(inv[w])), len(inv[w]), w))
    common = sorted(scored, reverse=True)[:n]
    rare = sorted(scored, key=lambda x: -(x[0] / math.sqrt(x[1])))[:n]
    out = {}
    for _, dfw, w in common + rare:
        out.setdefault(w, dfw)
    return out


def mine(con, pairs, inv, vocab, doc_sets, df):
    rows = []
    cache = {}
    for q, expect in pairs:
        toks = tokens(q)
        emb_o = hm.embed([q])[0]
        base_hit, base_pos = None, None
        top, paths, *_ = hm.rank(con, q, 5)
        for i, p in enumerate([paths[t] for t in top][:3]):
            if p.startswith(expect) or p == expect:
                base_hit, base_pos = True, i + 1
                break
        rows.append(dict(q=q, op="original", cos=1.0, token_df=0, neighbor_df=0,
                         df_ratio=1.0, rel_pos=0.0, q_len=len(toks),
                         hit=bool(base_hit), pos=base_pos or 99, base_hit=base_hit))
        seen = set()
        for i, t in enumerate(toks):
            if t not in vocab:
                continue
            # deletion
            reduced = " ".join(x for j, x in enumerate(toks) if j != i)
            if reduced and reduced not in seen:
                seen.add(reduced)
                v = hm.embed([reduced])[0]
                top, paths, *_ = hm.rank(con, reduced, 5)
                hit = pos = None
                for j, p in enumerate([paths[t2] for t2 in top][:3]):
                    if p.startswith(expect) or p == expect:
                        hit, pos = True, j + 1
                        break
                rows.append(dict(q=q, op="delete", cos=cos(v, emb_o),
                                 token_df=df.get(t, 1), neighbor_df=0, df_ratio=0.0,
                                 rel_pos=i / max(1, len(toks)), q_len=len(toks),
                                 hit=bool(hit), pos=pos or 99, base_hit=base_hit))
            # substitutions, both directions
            for w, dfw in neighbors_both(inv, doc_sets, t).items():
                variant = re.sub(rf"\b{re.escape(t)}\b", w, q, flags=re.IGNORECASE)
                if variant in seen or variant == q:
                    continue
                seen.add(variant)
                v = hm.embed([variant])[0]
                top, paths, *_ = hm.rank(con, variant, 5)
                hit = pos = None
                for j, p in enumerate([paths[t2] for t2 in top][:3]):
                    if p.startswith(expect) or p == expect:
                        hit, pos = True, j + 1
                        break
                rows.append(dict(q=q, op=f"sub:{t}->{w}", cos=cos(v, emb_o),
                                 token_df=df.get(t, 1), neighbor_df=dfw,
                                 df_ratio=dfw / max(1, df.get(t, 1)),
                                 rel_pos=i / max(1, len(toks)), q_len=len(toks),
                                 hit=bool(hit), pos=pos or 99, base_hit=base_hit))
    return rows


def analyze(rows):
    print(f"\n=== outcome table: {len(rows)} variants ===")
    base = [r for r in rows if r["op"] == "original"]
    subs = [r for r in rows if r["op"].startswith("sub:")]
    dels = [r for r in rows if r["op"] == "delete"]
    print(f"originals: hit {sum(r['hit'] for r in base)}/{len(base)}")
    print(f"deletions: hit {sum(r['hit'] for r in dels)}/{len(dels)}")
    print(f"substitutions: hit {sum(r['hit'] for r in subs)}/{len(subs)}")

    # flips: variants that hit where original missed (the valuable class)
    flips = [r for r in subs if r["hit"] and not r["base_hit"]]
    breaks = [r for r in subs if not r["hit"] and r["base_hit"]]
    print(f"\nFLIPS (fixed a miss): {len(flips)} | BREAKS (broke a hit): {len(breaks)}")
    if flips:
        print("\n-- what do the WINNING substitutions look like? --")
        for r in flips:
            print(f"  {r['op']:34s} cos={r['cos']:.3f} df_ratio={r['df_ratio']:.2f} pos={r['pos']}  q='{r['q'][:44]}'")
    if breaks:
        print("\n-- what do the BREAKING substitutions look like? (avoid these) --")
        for r in breaks:
            print(f"  {r['op']:34s} cos={r['cos']:.3f} df_ratio={r['df_ratio']:.2f}  q='{r['q'][:44]}'")

    # feature separation: winning vs losing substitution cos / df_ratio
    wins = [r for r in subs if r["hit"]]
    losses = [r for r in subs if not r["hit"]]
    for name, feat in (("cos", "cos"), ("df_ratio", "df_ratio")):
        wv = [r[feat] for r in wins] or [0]
        lv = [r[feat] for r in losses] or [0]
        print(f"\n{feat}: wins mean {sum(wv)/len(wv):.3f} | losses mean {sum(lv)/len(lv):.3f}")
    return flips, breaks


def main():
    cfg = (Path.home() / ".config/haystack/config.toml").read_text()
    con = hm.connect()
    inv, vocab, doc_sets, df = build(con)
    hard = json.load(open(Path(__file__).parent / "hard_pairs.json"))["pairs"]
    train, dev = hard[0::2], hard[1::2]
    print(f"mining on TRAIN ({len(train)} pairs); dev/eval untouched until pattern frozen")

    rows = mine(con, train, inv, vocab, doc_sets, df)
    flips, breaks = analyze(rows)

    out = Path(__file__).parent / "populist_outcomes.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\noutcome table -> {out}")


if __name__ == "__main__":
    main()
