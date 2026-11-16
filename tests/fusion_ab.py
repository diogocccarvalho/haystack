import json, os, sys
sys.path.insert(0, os.path.dirname(__file__) + "/..")
import haystack as H

pairs_file = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    H._cfg.get("audit", {}).get("qa_gen_file", "~/.config/haystack/qa_gen.json"))
pairs = json.load(open(pairs_file))["pairs"]
con = H.connect()

def score(pairs_list):
    recall = mrr = scored = 0
    per = []
    for qtext, expect in pairs_list:
        paths, heads, mat, texts = H.load_all(con)
        why = H.stale_pair(con, qtext, expect, paths, texts)
        if why:
            continue
        scored += 1
        top, qp = H.query(con, qtext, k=10)
        ranked = [qp[i] for i in top]
        r = 0.0
        for i, p in enumerate(ranked[:3]):
            if p.startswith(expect) or p == expect:
                r = 1.0 / (i + 1)
                break
        recall += 1 if r else 0
        mrr += r
        per.append((qtext, r))
    return recall / scored, mrr / scored, per

print(f"{len(pairs)} pairs (stale excluded per config)")
results = {}
for wv in (0.0, 0.3, 0.5, 0.7, 0.9, 1.0):
    H._cfg.setdefault("fusion", {})["dense_weight"] = wv
    rec, mrr, per = score(pairs)
    results[wv] = (rec, mrr, per)
    print(f"dense_weight={wv:.1f}  Recall@3 {rec:.3f}  MRR {mrr:.3f}")

# adaptive IDF
H._cfg["fusion"]["adaptive_idf"] = True
rec, mrr, _ = score(pairs)
print(f"adaptive_idf       Recall@3 {rec:.3f}  MRR {mrr:.3f}")

# per-query fusion tax: dense-only (1.0) vs incumbent (0.7)
dense_only = {q: r for q, r in results[1.0][2]}
hyb = dict(results[0.7][2])
worse = [q for q in hyb if hyb[q] < dense_only.get(q, 0)]
better = [q for q in hyb if hyb[q] > dense_only.get(q, 0)]
print(f"\nfusion tax (0.7 vs dense-only 1.0): {len(worse)} queries worse, {len(better)} better")
for q in worse:
    print(f"  worse: {q[:70]!r}")
