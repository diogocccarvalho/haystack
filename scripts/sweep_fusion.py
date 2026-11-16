#!/usr/bin/env python3
"""Fusion/ranking parameter sweep — grid over configurable params against eval
pairs and the 16-pair hard adversarial set.

Modes (auto-selected):
  dense_weight grid       — default path (dense_weight + adaptive_idf)
  reranker                — rerank on/off at every dense_weight setting
  K                       — rrf_k constant sweep (ranking.reranker_k)
  superseded_factor       — demotion factor for tombstoned notes

Writes results to scripts/sweep_fusion_results.json.
Run:  python3 scripts/sweep_fusion.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))
import haystack as hs  # noqa: E402

QA_PATH = Path.home() / ".config" / "haystack" / "qa_gen.json"
HARD_PATH = HERE / "scripts" / "hard_pairs.json"


# --------------------------------------------------------------------------- helpers
def fresh_con():
    return hs.connect()


def _set_cfg(key, sub, value):
    """Set a config sub-key inside hs._cfg."""
    hs._cfg.setdefault(key, {})[sub] = value


def _unset_subsection(section):
    """Remove all keys from a dict; if empty, delete it entirely."""
    d = hs._cfg.get(section)
    if isinstance(d, dict):
        list(d.keys()).clear() if False else [d.pop(k) for k in list(d)]
    elif section in hs._cfg:
        del hs._cfg[section]


def evaluate(con, pairs):
    """Recall@3, MRR, mean latency over *pairs* list.

    Each item is (query_text, expect_rel_path, expected_abs_Path).
    Stale pairs (target missing) are skipped.
    """
    r3 = rr = lat = n = 0
    import io, contextlib
    for q, rel, absp in pairs:
        if not absp.exists():
            continue
        n += 1
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                _top, qpaths = hs.query(con, q, k=10)
        except Exception:
            lat += time.perf_counter() - t0
            continue
        lat += time.perf_counter() - t0
        ranked = [qpaths[i] for i in _top[:10]]
        for ri, p in enumerate(ranked[:3]):
            if p.startswith(rel) or p == rel or p.startswith(str(absp)):
                r3 += 1
                rr += 1.0 / (ri + 1)
                break
    return r3 / max(n, 1), rr / max(n, 1), lat / max(n, 1)


def load_pairs(path, vault_base):
    raw = json.loads(path.read_text())["pairs"]
    return [(q, p, vault_base / p) for q, p in raw]


# --------------------------------------------------------------------------- experiments
def exp_dense_weight(con, pairs, incumbent_wv, incumbent_adidf, baseline_rank):
    """Grid over dense_weight × adaptive_idf."""
    _unset_subsection("fusion")
    hs._cfg["fusion"] = {}
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _set_cfg("fusion", "adaptive_idf", incumbent_adidf)
    rows = []
    print("\n=== Dense weight sweep ===")
    for wv in grid:
        _set_cfg("fusion", "dense_weight", wv)
        r3, mrr, lat = evaluate(con, pairs)
        _set_cfg("fusion", "dense_weight", incumbent_wv)
        _set_cfg("fusion", "adaptive_idf", incumbent_adidf)
        rec = {"dense_weight": wv, "adaptive_idf": incumbent_adidf,
               "recall3": round(r3, 3), "mrr": round(mrr, 4),
               "lat_ms": round(lat * 1000, 1)}
        rows.append(rec)
        print(f"wv={wv:.1f}  R@3 {r3:.3f}  MRR {mrr:.4f}  {lat*1000:.1f}ms")
    # adaptive_idf True probe
    _set_cfg("fusion", "dense_weight", incumbent_wv)
    _set_cfg("fusion", "adaptive_idf", True)
    r3, mrr, lat = evaluate(con, pairs)
    rec = {"dense_weight": incumbent_wv, "adaptive_idf": True,
           "recall3": round(r3, 3), "mrr": round(mrr, 4),
           "lat_ms": round(lat * 1000, 1)}
    rows.append(rec)
    print(f"adaptive=True  R@3 {r3:.3f}  MRR {mrr:.4f}  {lat*1000:.1f}ms")
    _set_cfg("fusion", "adaptive_idf", incumbent_adidf)
    return rows


def exp_reranker(con, pairs, incumbent_wv, incumbent_adidf, baseline_rank):
    """Reranker on/off at incumbent dense_weight, plus a few neighbours."""
    url_key = "url"
    old_renk = hs._cfg.get("rerank", {}).copy() if hs._cfg.get("rerank") else {}
    _unset_subsection("rerank")
    hs._cfg["fusion"] = hs._cfg.get("fusion", {})
    rows = []
    print("\n=== Reranker sweep ===")
    combos = [
        (incumbent_wv, incumbent_adidf, False),
        (incumbent_wv, incumbent_adidf, True),
        (0.6, incumbent_adidf, True),
        (0.8, incumbent_adidf, True),
        (0.5, incumbent_adidf, True),
    ]
    for wv, adidf, on in combos:
        _set_cfg("fusion", "dense_weight", wv)
        _set_cfg("fusion", "adaptive_idf", adidf)
        if on:
            _set_cfg("rerank", url_key, "http://127.0.0.1:8084")
        else:
            hs._cfg.setdefault("rerank", {}).pop(url_key, None)
        r3, mrr, lat = evaluate(con, pairs)
        if on:
            _set_cfg("rerank", url_key, "")
        else:
            _unset_subsection("rerank")
        label = f"{'+rerank' if on else '-rerank'}"
        rec = {"label": label, "dense_weight": wv, "adaptive_idf": adidf,
               "recall3": round(r3, 3), "mrr": round(mrr, 4),
               "lat_ms": round(lat * 1000, 1)}
        rows.append(rec)
        print(f"{label} wv={wv:.1f}  R@3 {r3:.3f}  MRR {mrr:.4f}  {lat*1000:.1f}ms")
    # restore
    _set_cfg("fusion", "dense_weight", incumbent_wv)
    _set_cfg("fusion", "adaptive_idf", incumbent_adidf)
    hs._cfg.pop("rerank", None)
    for k, v in old_renk.items():
        _set_cfg("rerank", k, v)
    return rows


def exp_K(con, pairs, incumbent_wv, incumbent_adidf, baseline_rank):
    """Sweep RRF K constant via ranking.rrf_k config entry."""
    old_val = float(_cfg_read("ranking", "rrf_k", 60.0))
    _unset_subsection("ranking")
    hs._cfg["fusion"] = hs._cfg.get("fusion", {})
    grid = [30, 40, 50, 60, 80, 100, 120]
    rows = []
    print("\n=== RRF K sweep ===")
    for k in grid:
        _set_cfg("ranking", "rrf_k", k)
        r3, mrr, lat = evaluate(con, pairs)
        rows.append({"rrf_k": k, "recall3": round(r3, 3),
                     "mrr": round(mrr, 4), "lat_ms": round(lat * 1000, 1)})
        print(f"K={k:<4}  R@3 {r3:.3f}  MRR {mrr:.4f}  {lat*1000:.1f}ms")
    _set_cfg("ranking", "rrf_k", old_val)
    return rows


def exp_superseded_factor(con, pairs, incumbent_wv, incumbent_adidf, baseline_rank):
    """Sweep demotion factor for superseded (tombstoned) notes."""
    old_val = float(_cfg_read("ranking", "superseded_factor", 0.40))
    _unset_subsection("ranking")
    hs._cfg["fusion"] = hs._cfg.get("fusion", {})
    grid = [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0]
    rows = []
    print("\n=== Superseded demotion factor sweep ===")
    for f_ in grid:
        _set_cfg("ranking", "superseded_factor", f_)
        r3, mrr, lat = evaluate(con, pairs)
        rows.append({"superseded_factor": f_, "recall3": round(r3, 3),
                     "mrr": round(mrr, 4), "lat_ms": round(lat * 1000, 1)})
        print(f"sf={f_:>.2f}  R@3 {r3:.3f}  MRR {mrr:.4f}  {lat*1000:.1f}ms")
    _set_cfg("ranking", "superseded_factor", old_val)
    return rows


def _cfg_read(section, key, default):
    try:
        return float(hs._cfg.get(section, {}).get(key, default))
    except (TypeError, ValueError):
        return float(default)


def run_hard_gate(rows):
    """Evaluate each row's effective settings against hard_pairs."""
    vault = Path(hs._cfg["vault"]["path"]).expanduser()
    pairs = load_pairs(HARD_PATH, vault)
    con = fresh_con()
    hard_rows = []
    best_mrr_h = best_n = 0
    for i, rec in enumerate(rows):
        # Restore incumbent defaults
        _unset_subsection("fusion")
        _unset_subsection("rerank")
        _unset_subsection("ranking")

        # Apply this row's config
        _set_cfg("fusion", "dense_weight", rec.get("dense_weight", 0.7))
        _set_cfg("fusion", "adaptive_idf", rec.get("adaptive_idf", False))
        if rec.get("label") and "+rerank" in str(rec.get("label", "")):
            _set_cfg("rerank", "url", "http://127.0.0.1:8084")
        if "rrf_k" in rec:
            _set_cfg("ranking", "rrf_k", rec["rrf_k"])
        if "superseded_factor" in rec:
            _set_cfg("ranking", "superseded_factor", rec["superseded_factor"])

        r3, mrr, lat = evaluate(con, pairs)
        out = dict(rec)
        out["hard_recall3"] = round(r3, 3)
        out["hard_mrr"] = round(mrr, 4)
        out["hard_n"] = len(pairs)
        hard_rows.append(out)
        print(f"  Row {i}: hard_R@3 {r3:.3f}  hard_MRR {mrr:.4f}")
        if mrr > best_mrr_h:
            best_mrr_h = mrr
            best_n = i

        # Reset before next iteration
        _unset_subsection("fusion")
        _unset_subsection("rerank")
        _unset_subsection("ranking")

    print(f"\nHard gate best: row {best_n} (MRR {best_mrr_h:.4f})")
    return hard_rows


def main():
    con = fresh_con()
    vault = Path(hs._cfg["vault"]["path"]).expanduser()
    qa_pairs = load_pairs(QA_PATH, vault)
    qa_n = sum(1 for _, _, p in qa_pairs if p.exists())
    print(f"Eval pairs (exists target): {qa_n}")

    incumbent_wv = float(hs._cfg.get("fusion", {}).get("dense_weight", 0.7))
    incumbent_adidf = bool(hs._cfg.get("fusion", {}).get("adaptive_idf", False))
    print(f"Incumbent: dense_weight={incumbent_wv}, adaptive_idf={incumbent_adidf}")

    # Baseline eval first
    r3, mrr, lat = evaluate(con, qa_pairs)
    baseline = {"label": "baseline", "dense_weight": incumbent_wv,
                "adaptive_idf": incumbent_adidf,
                "recall3": round(r3, 3), "mrr": round(mrr, 4),
                "lat_ms": round(lat * 1000, 1)}
    print(f"Baseline → R@3 {r3:.3f}  MRR {mrr:.4f}  {lat*1000:.1f}ms")

    all_rows = [baseline]

    # ---- Run experiments ----
    rows_dw = exp_dense_weight(con, qa_pairs, incumbent_wv, incumbent_adidf, baseline)
    all_rows.extend(rows_dw)

    rows_rk = exp_reranker(con, qa_pairs, incumbent_wv, incumbent_adidf, baseline)
    all_rows.extend(rows_rk)

    rows_K = exp_K(con, qa_pairs, incumbent_wv, incumbent_adidf, baseline)
    all_rows.extend(rows_K)

    rows_sf = exp_superseded_factor(con, qa_pairs, incumbent_wv, incumbent_adidf, baseline)
    all_rows.extend(rows_sf)

    # ---- Hard gate ----
    print("\n========== HARD GATE ==========")
    hard_rows = run_hard_gate(all_rows)

    # Combine results
    combined = []
    for i, ar in enumerate(all_rows):
        hr = hard_rows[i] if i < len(hard_rows) else {}
        row = {**ar, **{k: hr[k] for k in ("hard_recall3", "hard_mrr", "hard_n")}}
        combined.append(row)

    # ---- Print table ----
    print("\n\n============ FULL RESULTS TABLE =============")
    hdr = f"{'label':<20} {'wv':>5} {'adidf':>5} {'K':>4} {'sf':>5} {'+rnkr':>6} | qR@3 qMRR qLat | hR@3 hMRR (n={hr['hard_n'] if hard_rows else '?'})"
    print(hdr)
    print("-" * len(hdr))
    for r in combined:
        rnkr = '+' in str(r.get('label', ''))
        lbl = str(r.get('label', f"dense_weight={r.get('dense_weight','')}"))
        print(f"{lbl:<20} {r.get('dense_weight',0):>5.1f} {str(r.get('adaptive_idf',False)):>5} "
              f"{r.get('rrf_k',60):>4} {r.get('superseded_factor',0.40):>5.2f} {'YES' if rnkr else 'NO':>6} | "
              f"{r.get('recall3',0):.3f} {r.get('mrr',0):.4f} {r.get('lat_ms',0):.0f} | "
              f"{r.get('hard_recall3',0):.3f} {r.get('hard_mrr',0):.4f}")

    # ---- Verdict ----
    print("\n============ VERDICT =============")
    for i, r in enumerate(combined):
        q_ok = r.get('mrr', 0) >= baseline['mrr'] and r.get('lat_ms', 9999) <= baseline['lat_ms'] * 1.1
        h_ok = r.get('hard_mrr', 0) >= baseline.get('hard_mrr', 0)
        status = "PROMOTE" if (q_ok and h_ok) else ("— no dominance")
        if status == "PROMOTE":
            print(f"✓ ROW {i} ({r}): PROMOTED — dominates BOTH gates")
        else:
            reason = []
            if not q_ok:
                reason.append(f"eval worse (MRR {r.get('mrr',0):.4f} vs {baseline['mrr']:.4f})")
            if not h_ok:
                reason.append(f"hard worse (MRR {r.get('hard_mrr',0):.4f} vs {baseline.get('hard_mrr',0):.4f})")
            print(f"✗ ROW {i} ({r}): {'; '.join(reason)}")

    # Save results
    out = HERE / "scripts" / "sweep_fusion_results.json"
    out.write_text(json.dumps(combined, indent=1) + "\n")
    print(f"\nResults saved -> {out}")

    # Restore exact disk state
    import tomllib
    cfg_disk = Path.home() / ".config" / "haystack" / "config.toml"
    if cfg_disk.exists():
        with open(cfg_disk, "rb") as f:
            disk = tomllib.load(f)
        hs._cfg["fusion"] = disk.get("fusion", {"dense_weight": 0.7})
        hs._cfg["ranking"] = disk.get("ranking", {})
        rs = disk.get("rerank", {})
        if rs:
            hs._cfg["rerank"] = rs
        elif "rerank" in hs._cfg:
            del hs._cfg["rerank"]


if __name__ == "__main__":
    main()
