"""Fusion policy sweep — the eval gate for the dense_weight crystal.

Grid over fusion.dense_weight + adaptive_idf against the qa_gen pairs;
measures Recall@3 / MRR / mean latency. Mutates hs._cfg in-process and
restores it (the TOML on disk is never touched). Winner must beat the
incumbent on MRR AND latency to justify a config change.
Run:  python3 scripts/sweep_fusion.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))
import haystack as hs  # noqa: E402

PAIRS = Path.home() / ".config" / "haystack" / "qa_gen.json"
GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def fresh_con():
    return hs.connect()


def evaluate(con, pairs):
    r3 = 0
    rr = 0.0
    lat = 0.0
    n = 0
    import io
    import contextlib
    for q, expect_rel, expect_abs in pairs:
        expected = expect_rel
        if not expect_abs.exists():
            continue  # stale pair (target missing), same as eval's guard
        n += 1
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _top, qpaths = hs.query(con, q, k=10)
        except Exception:
            lat += time.perf_counter() - t0
            continue
        lat += time.perf_counter() - t0
        ranked = [qpaths[i] for i in _top]  # same semantics as eval_pairs
        for rank_i, p in enumerate(ranked[:3]):
            if p.startswith(str(expected)) or p == str(expected):
                r3 += 1
                rr += 1.0 / (rank_i + 1)
                break
    return r3 / max(n, 1), rr / max(n, 1), lat / max(n, 1)


def set_cfg(key, sub, value):
    hs._cfg.setdefault(key, {})[sub] = value


def main():
    con = fresh_con()
    vault = hs.VAULT
    pairs = [(q, p, vault / p) for q, p in
             json.loads(PAIRS.read_text())["pairs"]]
    rows = []
    incumbent = hs._cfg.get("fusion", {}).get("dense_weight", 0.7)
    print(f"incumbent dense_weight={incumbent}")
    try:
        for wv in GRID:
            set_cfg("fusion", "dense_weight", wv)
            set_cfg("fusion", "adaptive_idf", False)
            r3, mrr, lat = evaluate(con, pairs)
            rows.append({"dense_weight": wv, "adaptive_idf": False,
                         "recall3": round(r3, 3), "mrr": round(mrr, 4),
                         "lat_ms": round(lat * 1000, 1)})
            print(f"wv={wv:.1f}  Recall@3 {r3:.3f}  MRR {mrr:.4f}  "
                  f"{lat * 1000:.1f}ms")
        set_cfg("fusion", "adaptive_idf", True)
        r3, mrr, lat = evaluate(con, pairs)
        rows.append({"adaptive_idf": True, "recall3": round(r3, 3),
                     "mrr": round(mrr, 4), "lat_ms": round(lat * 1000, 1)})
        print(f"adaptive  Recall@3 {r3:.3f}  MRR {mrr:.4f}  "
              f"{lat * 1000:.1f}ms")
    finally:
        set_cfg("fusion", "dense_weight", incumbent)
        set_cfg("fusion", "adaptive_idf",
                json.loads(json.dumps(
                    hs._cfg.get("fusion", {}).get("adaptive_idf", False))))
        # restore from disk to be exact
        import tomllib
        with open(Path.home() / ".config" / "haystack" / "config.toml", "rb") as f:
            disk = tomllib.load(f)
        hs._cfg["fusion"] = disk.get("fusion", {"dense_weight": 0.7})
    out = HERE / "scripts" / "sweep_fusion_results.json"
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print("saved ->", out)


if __name__ == "__main__":
    main()
