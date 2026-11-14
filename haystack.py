#!/usr/bin/env python3
"""haystack — semantic router + deterministic editor for a local markdown vault.

Local, zero-API: points at ANY OpenAI-compatible /v1/embeddings endpoint
(llama.cpp, Ollama, LM Studio, remote). Learns from your corrections.

Usage:
  haystack.py reindex                  # incremental reindex
  haystack.py query <text> [-k N]      # top-N: path<TAB>heading<TAB>score<TAB>snippet
  haystack.py find-file <words>        # best path only
  haystack.py append <file> <heading> <text> [--dry-run]
  haystack.py frontmatter <file> k=v [--dry-run]
  haystack.py new-note <dir> <name> ["Title"] [--dry-run]
  haystack.py learn "<query>" <path>   # teach a correction
  haystack.py review <text>            # interactive feedback loop
  haystack.py aliases                  # mine nickname candidates per folder
  haystack.py audit                    # reindex + run QA pairs (exit 1 on miss)
  haystack.py pick-encoder <url> [--pairs f] [--limit N] [--yes]
                                       # A/B embeddings endpoints on THIS vault,
                                       # --yes writes the winner to config + rebuilds

Config: ~/.config/notes-router/config.toml (override with NR_CONFIG).
See config.toml in this repo for every option.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from pathlib import Path

import numpy as np

def _default_cfg():
    # New installs use ~/.config/haystack (setup.sh); legacy installs keep
    # ~/.config/notes-router working. First existing wins.
    for d in ("haystack", "notes-router"):
        p = Path.home() / ".config" / d / "config.toml"
        if p.exists():
            return p
    return Path.home() / ".config/haystack/config.toml"


CFG_PATH = Path(os.environ.get("NR_CONFIG") or _default_cfg())
if not CFG_PATH.exists():
    sys.exit(f"no config at {CFG_PATH} — run setup.sh, or copy config.toml from the repo and edit vault.path")

with open(CFG_PATH, "rb") as f:
    _cfg = tomllib.load(f)

VAULT = Path(_cfg["vault"]["path"]).expanduser()
INDEX_CONV = _cfg["vault"].get("index_convention", "list")
SKIP_PARTS = tuple(_cfg["vault"].get("exclude_dirs", []))
SKIP_FILES = set(_cfg["vault"].get("exclude_files", []))
EXTERNAL = {Path(s).expanduser(): "external/" + Path(s).name
            for s in _cfg["vault"].get("external", [])}
DB = Path(_cfg["index"]["db"]).expanduser()
EMBD_URL = _cfg["embeddings"]["url"]
EMBD_MODEL = _cfg["embeddings"].get("model", "")
QA_FILE = Path(_cfg["audit"]["qa_file"]).expanduser()
LOG_DIR = DB.parent
DATE = time.strftime("%Y-%m-%d")


def embed(texts):
    payload = {"input": texts}
    if EMBD_MODEL:
        payload["model"] = EMBD_MODEL
    req = urllib.request.Request(
        EMBD_URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return np.array([d["embedding"] for d in data["data"]], dtype=np.float32)


def embed_resilient(texts):
    """R10: a chunk denser than the embedder's context 500s the whole batch —
    split-on-failure and skip the offending chunk instead of dying (a foreign
    corpus with URLs/code tokens hit this on an otherwise healthy server).
    Returns (vecs, [kept indices])."""
    try:
        return embed(texts), list(range(len(texts)))
    except urllib.error.HTTPError:
        if len(texts) == 1:
            log_edit("embed-skip", f"chunk exceeds ctx, skipped: {texts[0][:60]!r}")
            return np.zeros((0, 384), dtype=np.float32), []
        mid = len(texts) // 2
        v1, k1 = embed_resilient(texts[:mid])
        v2, k2 = embed_resilient(texts[mid:])
        k2 = [i + mid for i in k2]
        return np.vstack([v1, v2]), k1 + k2


def chunks_of(path, rel=None):
    """[(heading, text)] per file: title blob + one chunk per ## section.

    Every non-title chunk is prefixed with the document title + relative path
    (poor-man's contextual retrieval: arXiv 2504.19754 found context-enriched
    chunks + BM25-over-context beat plain chunks; this gives BM25/IDF and the
    dense model the document identity in every chunk)."""
    t = path.read_text(errors="replace")
    lines = t.split("\n")
    title_blob = []
    for ln in lines[:10]:
        if ln.startswith(("title:", "# ")) and len(ln) > 4:
            title_blob.append(ln)
    title_blob_text = "\n".join(title_blob or lines[:3])
    rel = rel or path.name
    ctx = f"[doc: {rel} | {title_blob_text.splitlines()[0] if title_blob_text else ''}]"
    out, cur_head, cur = [], "(top)", []
    out.append(("(title)", title_blob_text))
    for ln in lines:
        if ln.startswith("## "):
            if cur:
                out.append((cur_head, "\n".join(cur)))
            cur_head, cur = ln[3:].strip(), []
        else:
            cur.append(ln)
    if cur:
        out.append((cur_head, "\n".join(cur)))
    # list-heavy files: one chunk per ~6 lines — big windows dilute
    n_list = sum(1 for ln in lines if ln.strip().startswith(("- ", "* ")))
    if n_list > 20 and len(lines) > 60:
        out = [("(title)", title_blob_text)]
        buf, head_now = [], cur_head
        for ln in lines:
            if ln.startswith("## "):
                if buf:
                    out.append((head_now or "(list)", title_blob_text + "\n" + "\n".join(buf)))
                head_now, buf = ln[3:].strip(), [ln]
            else:
                buf.append(ln)
                if len([b for b in buf if b.strip()]) >= 6:
                    out.append((head_now or "(list)", title_blob_text + "\n" + "\n".join(buf)))
                    buf = []
        if buf and len([b for b in buf if b.strip()]):
            out.append((head_now or "(list)", title_blob_text + "\n" + "\n".join(buf)))
    else:
        # sections longer than ~1 window: slide windows so deep lines surface
        long = []
        for h, txt in out:
            if len(txt) > 1000:
                step, win = 800, 1000
                i = 0
                while i < len(txt):
                    seg = txt[i:i + win].strip()
                    if len(seg) > 200:
                        long.append((f"{h} · L{i + 1}", seg))
                    i += step
            else:
                long.append((h, txt))
        out = long
    final = []
    for h, txt in out:
        txt = txt.strip()
        if not txt:
            continue
        if h != "(title)":
            txt = (ctx + "\n" + txt)[:1000]
        final.append((h, txt))
    return final  # BGE ctx=512 tok


def real_path(rel):
    for src, name in EXTERNAL.items():
        if name == rel:
            return src
    return VAULT / rel


def rel_of(p):
    for src, name in EXTERNAL.items():
        if src == p:
            return name
    try:
        return str(p.relative_to(VAULT))
    except ValueError:
        return "external/" + p.name


def walk():
    for p in EXTERNAL:
        if p.exists():
            yield p
    for p in sorted(VAULT.rglob("*.md")):
        if not p.is_file():  # a directory named "x.md" would crash chunks_of
            continue
        rel = str(p.relative_to(VAULT))
        if rel in SKIP_FILES or any(part in rel for part in SKIP_PARTS):
            continue
        yield p


LEARNED_FILE = CFG_PATH.parent / "learned.json"


def _load_learned():
    if LEARNED_FILE.exists():
        try:
            return json.loads(LEARNED_FILE.read_text())
        except json.JSONDecodeError:
            sys.exit(f"{LEARNED_FILE}: invalid JSON — fix or delete it")
    return []


def learn_lookup(con, text, paths, q, dense=None):
    """Past corrections boost identical/similar queries (and their folder).
    Stored OUTSIDE the index DB (learned.json) so index rebuilds never wipe
    learned ground truth (append-only; Reglos-style persistence)."""
    rows = _load_learned()
    if not rows:
        return np.zeros(len(paths))
    qs = embed([r["qtext"] for r in rows])
    sims = qs @ q
    # Model-agnostic match rule: a taught correction deserves the pin only if
    # it BEATS the corpus's own best hit for this query. Embedders differ in
    # similarity baseline (bge unrelated ~0.5, granite ~0.7+), so any fixed
    # absolute threshold misfires on some models; but max(chunk-sim) for THIS
    # query comes from the SAME encoder, so "entry > corpus best" is
    # scale-free. Rationale: the user's phrasing, matched against its own
    # correction, should outrank everything in the vault — a near-tie means
    # the query is actually about something else. Floor 0.60 guards
    # degenerate corpora.
    corpus_best = float(dense.max()) if dense is not None and dense.size else 0.0
    boost = np.zeros(len(paths))
    for row, sim in zip(rows, sims):
        if sim > 0.60 and sim >= corpus_best + 0.01:
            folder = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else row["path"]
            strength = 0.10 * sim * min(row["hits"], 3)
            for i, p in enumerate(paths):
                if p == row["path"] or p == folder:
                    boost[i] += strength  # exact target (file or folder pin)
                elif p.startswith(folder + "/"):
                    boost[i] += strength / 2  # folder fallback: strictly weaker,
                    # else a pin in a busy folder can't outrank its own siblings
    return boost


def connect():
    import sqlite3
    Path(DB).parent.mkdir(parents=True, exist_ok=True)  # config may point anywhere
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        path TEXT, heading TEXT, hash TEXT, vec BLOB, text TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS files(path TEXT, mtime REAL, hash TEXT)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_c ON chunks(path)")
    return con


def reanchor(con, moved=False):
    """Stale-pin repair (reidmarlow's edge case): a pin whose target path left
    the index (rename/move) is re-anchored onto the same note instead of
    surfacing a dead path. Identity check order: unique content-hash match
    (free, exact — catches rename-only moves) -> semantic match of the stored
    title anchor against current title chunks (survives edits too), gated by
    the scale-free margin: best title hit must clear 0.60 AND beat the
    runner-up by 0.05. Semantics only run when THIS reindex saw a plausible
    move (files added AND removed together) — a deletion-only pass must never
    re-anchor, or a near-duplicate decoy file silently inherits the pin. No
    confident match = pin stays stale and audit keeps flagging it — a wrong
    auto-reanchor lies, a stale pin fails loudly."""
    if not moved:
        return
    rows = _load_learned()
    live = {r[0]: r[1] for r in con.execute("SELECT path, hash FROM files")}
    titles = list(con.execute("SELECT path, text FROM chunks WHERE heading='(title)'"))
    changed = 0
    for row in rows:
        old = row.get("path", "")
        if old in live or not (row.get("hash") or row.get("anchor")):
            continue  # healthy pin, folder pin, or legacy row without anchors
        new_path = None
        if row.get("hash"):
            hits = [lp for lp, lh in live.items() if lh == row["hash"]]
            if len(hits) == 1:
                new_path = hits[0]
        if new_path is None and row.get("anchor") and titles:
            av = embed([row["anchor"]])[0]
            tv = embed([t for _, t in titles])
            sims = tv @ av
            order = np.argsort(-sims)
            best = float(sims[order[0]])
            second = float(sims[order[1]]) if len(order) > 1 else 0.0
            if best > 0.60 and (len(order) == 1 or best - second >= 0.05):
                new_path = titles[int(order[0])][0]
        if new_path and new_path != old:
            row.setdefault("history", []).append(old)
            row["path"] = new_path
            changed += 1
            log_edit("reanchor", f"{old} -> {new_path}")
    if changed:
        LEARNED_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(f"reanchored {changed} stale pin(s) (see {LOG_DIR.name}/notes-editor.log)")


def reindex(con):
    db_files = {r[0]: (r[1], r[2]) for r in con.execute("SELECT path, mtime, hash FROM files")}
    cur = con.cursor()
    n_new = n_del = 0
    seen = set()
    for p in walk():
        rel = rel_of(p)
        seen.add(rel)
        mt, h0 = db_files.get(rel, (0, None))
        raw = p.read_bytes()
        h = hashlib.sha256(raw).hexdigest()[:16]
        if h0 == h:
            continue
        cur.execute("DELETE FROM chunks WHERE path=?", (rel,))
        cur.execute("DELETE FROM files WHERE path=?", (rel,))
        cs = chunks_of(p, rel)
        if cs:  # empty/near-empty files: no chunks to embed (embed([]) 400s)
            vecs, kept = embed_resilient([c[1] for c in cs])
            for pos, i in enumerate(kept):
                head, txt = cs[i]
                cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?)",
                            (rel, head, h, vecs[pos].astype(np.float32).tobytes(), txt))
        cur.execute("INSERT INTO files VALUES(?,?,?)", (rel, p.stat().st_mtime, h))
        n_new += 1
    for rel in set(db_files) - seen:
        cur.execute("DELETE FROM chunks WHERE path=?", (rel,))
        cur.execute("DELETE FROM files WHERE path=?", (rel,))
        n_del += 1
    con.commit()
    reanchor(con, moved=(n_new > 0 and n_del > 0))
    print(f"indexed: {n_new} updated, {n_del} removed, {sum(1 for _ in walk())} files")


STOP = set("the a an of about my i me that this thing stuff should where do is are for on in to with and or".split())
if "urllib.request" not in sys.modules:
    import urllib.request
    import urllib.error


def lexical_boost(text, texts):
    lowered = [t.lower() for t in texts]
    n = len(texts)
    raw = re.findall(r"\w{4,}", text.lower())
    if n >= 50:
        # learned stopwording: words present in most of the corpus are
        # non-discriminative in ANY language (IDF floor, replaces the
        # hardcoded English STOP once the corpus is big enough)
        toks = [w for w in raw if w not in STOP]
        for w in set(raw):
            if sum(1 for t in lowered if w in t) > 0.5 * n:
                toks = [t for t in toks if t != w]
    else:
        toks = [w for w in raw if w not in STOP]
    if not toks:
        return np.zeros(len(texts))
    # IDF weighting: common words count little, distinctive ones count
    boost = np.zeros(n, dtype=np.float32)
    for w in set(toks):
        df = sum(1 for t in lowered if w in t) or 1
        idf = 0.25 * min(3.0, math.log(n / df) + 0.5)
        tf = np.minimum(np.array([t.count(w) for t in lowered], dtype=np.float32), 2.0)
        boost += idf * tf
    return 0.10 * boost


def load_all(con):
    rows = list(con.execute("SELECT path, heading, vec, text FROM chunks"))
    paths = [r[0] for r in rows]
    heads = [r[1] for r in rows]
    if rows:
        mat = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    else:
        mat = np.zeros((0, 384), dtype=np.float32)
    texts = [r[3] for r in rows]
    return paths, heads, mat, texts


def rank(con, text, k=5):
    """RRF fusion (Cormack 2009; k=60 is the ES/Qdrant default) of the dense
    ranking and the lexical ranking, plus additive learned-correction boost.
    Rank-based fusion avoids the cosine-vs-BM25 scale-calibration problem."""
    paths, heads, mat, texts = load_all(con)
    if not paths:
        return np.array([], dtype=int), paths, heads, texts, np.array([])
    q = embed([text])[0]
    n = len(paths)
    dense = mat @ q
    lex = lexical_boost(text, texts)
    # field boost: title/top chunks dominate in both streams (WANDS: biggest gain)
    title_mask = np.array([1.15 if h in ("(title)", "(top)") else 1.0 for h in heads])
    dense_rank = np.empty(n, dtype=int)
    dense_rank[np.argsort(-(dense * title_mask))] = np.arange(n)
    lex_rank = np.empty(n, dtype=int)
    lex_rank[np.argsort(-(lex * title_mask))] = np.arange(n)
    K = 60.0
    scores = (0.7 / (K + dense_rank + 1) + 0.3 / (K + lex_rank + 1)).astype(np.float32)
    # rare-token exact hit: identifiers win (financial-docs 2026: BM25 beats dense here)
    lowered = [t.lower() for t in texts]
    for w in set(re.findall(r"\w{4,}", text.lower())) - STOP:
        df = sum(1 for t in lowered if w in t)
        if 0 < df <= 3:
            scores += np.array([0.03 if w in t else 0.0 for t in lowered], dtype=np.float32)
    lboost = learn_lookup(con, text, paths, q, dense=dense)
    scores += lboost  # additive post-fusion
    # INDEX.md files are pointer hubs; the note they point to is the content.
    # Small demotion (NOT removal) so hub files stop outranking their targets.
    for i, p in enumerate(paths):
        if p.endswith("/INDEX.md") or p == "INDEX.md":
            scores[i] *= 0.92
    order = list(np.argsort(-scores))
    # teacher beats model: user-taught paths survive the cross-encoder's reordering
    # (pinned must be score-ordered or the "best" correction is a random chunk)
    pinned = sorted((i for i in range(len(paths)) if lboost[i] > 0), key=lambda i: -scores[i])
    order = rerank(con, text, order, texts, pinned=pinned, paths_rel=paths)
    top = order[:k]
    return top, paths, heads, texts, scores


def rerank(con, text, order, texts, pinned=(), paths_rel=()):
    """Cross-encoder rerank of the fused top (T2-RAGBench: Recall@5
    0.695 -> 0.816). Optional [rerank] url/model/top in config; the reranker
    only REORDERS (fusion scores stay the reported scores), and learned
    paths are pinned ahead of it — user corrections outrank any model. Any
    failure falls back to the fused order — retrieval must never hard-depend
    on a second server being up."""
    url = _cfg.get("rerank", {}).get("url", "")
    if not url or len(order) < 2:
        return order
    topn = int(_cfg.get("rerank", {}).get("top", 20))
    max_chars = int(_cfg.get("rerank", {}).get("max_chars", 600))
    pinned = [i for i in pinned if i in order[:topn]]
    order = [i for i in order if i not in pinned]
    # hub files (INDEX) mention everything, so the cross-encoder ranks them
    # highly on name-recognition alone — excluded from candidates (R5: the
    # content file they point to must be the answer)
    cand = [i for i in order[:topn * 2]
            if not paths_rel[i].endswith("INDEX.md") or paths_rel[i].count("/") == 0][:topn] if paths_rel else order[:topn]
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/rerank",
            json.dumps({"query": text, "documents": [texts[i][:max_chars] for i in cand]}).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)["results"]
    except Exception as e:  # degraded mode: fused order, but pins survive (R2)
        log_edit("rerank-fallback", str(e)[:80])
        return pinned + [i for i in order if i not in pinned]
    ranked = [cand[r["index"]] for r in res if r["index"] < len(cand)]
    tail = [i for i in order if i not in cand]
    return pinned + ranked + tail  # pinned = learned corrections, always first


def query(con, text, k=5):
    top, paths, heads, texts, scores = rank(con, text, k)
    for i in top:
        snip = re.sub(r"\s+", " ", str(texts[i])[:180])
        print(f"{paths[i]}\t{heads[i]}\t{scores[i]:.3f}\t{snip}")
    return top, paths


def find_file(con, words):
    top, paths, _, _, _ = rank(con, " ".join(words), k=1)
    if len(top):
        print(str(real_path(paths[int(top[0])])))


def log_edit(op, detail):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "notes-editor.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {op} {detail}\n")


def index_one(con, p):
    rel = rel_of(p)
    cur = con.cursor()
    cur.execute("DELETE FROM chunks WHERE path=?", (rel,))
    cur.execute("DELETE FROM files WHERE path=?", (rel,))
    cs = chunks_of(p, rel)
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    vecs, kept = embed_resilient([c[1] for c in cs]) if cs else (np.zeros((0, 384), dtype=np.float32), [])
    for pos, i in enumerate(kept):
        head, txt = cs[i]
        cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?)",
                    (rel, head, h, vecs[pos].astype(np.float32).tobytes(), txt))
        cur.execute("INSERT INTO files VALUES(?,?,?)", (rel, p.stat().st_mtime, h))
    con.commit()


def update_index(fpath):
    """Ensure INDEX.md has a line for fpath (only when index_convention=list)."""
    if INDEX_CONV != "list" or rel_of(fpath).startswith("external/"):
        return
    idx = VAULT / "INDEX.md"
    if not idx.exists():
        idx.write_text("# INDEX\n")
    rel = str(fpath.relative_to(VAULT))
    t = idx.read_text(errors="replace")
    if rel in t:
        return
    title = ""
    txt = fpath.read_text(errors="replace")
    if txt.startswith("---"):
        m = re.search(r'title:\s*"?([^"\n]+?)"?\s*$', txt.split("---")[1], re.M)
        title = m.group(1) if m else fpath.stem
    line = f"- [[{rel}]] — {title or fpath.stem} (auto-added {DATE})"
    idx.write_text(t.rstrip() + "\n" + line + "\n")


def append(file_ref, heading, text, dry=False):
    p = real_path(file_ref)
    if not p.exists():
        sys.exit(f"{file_ref}: not found")
    t = p.read_text(errors="replace")
    line = f"- {DATE} {text.strip()}\n"
    if heading == "(top)":
        new = t.rstrip() + "\n" + line
    else:
        m = (re.search(rf"^## {re.escape(heading)}\n", t, re.M)
             or re.search(rf"^# {re.escape(heading)}\n", t, re.M))
        if not m:
            sys.exit(f"{file_ref}: heading '{heading}' not found")
        new = t[:m.end()] + line + t[m.end():]
    if dry:
        print(f"[dry-run] would append under '## {heading}' in {p}:\n  {line.strip()}")
        return
    p.write_text(new)
    update_index(p)
    log_edit("append", f"{file_ref}#{heading} {text[:60]}")
    index_one(connect(), p)
    print(f"appended -> {file_ref} #{heading}")


def set_frontmatter(file_ref, kv, dry=False):
    p = real_path(file_ref)
    key, val = kv.split("=", 1)
    t = p.read_text(errors="replace")
    if not t.startswith("---"):
        new = f"---\n{key}: {val}\n---\n\n" + t
    else:
        head, rest = t.split("---", 2)[1], "---".join(t.split("---")[2:])
        if re.search(rf"^{key}:", head, re.M):
            head = re.sub(rf"^{key}:.*$", f"{key}: {val}", head, flags=re.M)
        else:
            head = head.rstrip() + f"\n{key}: {val}\n"
        new = "---" + head + "---" + rest
    if dry:
        print(f"[dry-run] would set {key}={val} in {p}")
        return
    p.write_text(new)
    update_index(p)
    log_edit("frontmatter", f"{file_ref} {key}={val}")
    index_one(connect(), p)
    print(f"{file_ref}: {key}={val}")


def new_note(dirrel, name, title=None, dry=False):
    if "/" in name or "\\" in name or ".." in name or not name.strip():
        sys.exit(f"invalid note name: {name!r}")
    d = VAULT / dirrel
    p = d / f"{name}.md"
    if dry:
        print(f"[dry-run] would create {p} (templated frontmatter + INDEX line)")
        return
    d.mkdir(parents=True, exist_ok=True)
    if p.exists():
        sys.exit(f"{p} exists")
    p.write_text(f'---\ntitle: "{title or name}"\ndate: {DATE}\nstatus: active\n---\n\n# {title or name}\n\n')
    update_index(p)
    log_edit("new-note", str(p))
    index_one(connect(), p)
    print(p)


def learn(con, qtext, path, commit=True):
    rp = real_path(path)
    # Containment: the resolved path must live inside the vault, or be one of
    # the config's explicitly listed EXTERNAL files. Existence alone is not
    # enough — VAULT/"../../etc/passwd" resolves to a real system file.
    allowed_external = rp in EXTERNAL.values() if EXTERNAL else False
    if not allowed_external:
        try:
            rp.resolve().relative_to(VAULT.resolve())
        except ValueError:
            sys.exit(f"{path}: not a vault path")
    if not (rp.is_file() or rp.is_dir()):
        sys.exit(f"{path}: not a vault path")
    # Identity snapshot for stale-pin re-anchoring: content hash (rename-only
    # moves) + title anchor text (semantic fallback that survives edits).
    fhash = fanchor = None
    if rp.is_file():
        fhash = hashlib.sha256(rp.read_bytes()).hexdigest()[:16]
        cs = chunks_of(rp)
        if cs and cs[0][0] == "(title)" and cs[0][1].strip():
            fanchor = cs[0][1]
    rows = _load_learned()
    for row in rows:
        if row["qtext"] == qtext and row["path"] == path:
            row["hits"] = min(row["hits"] + 1, 3)
            break
    else:
        rows.append({"qtext": qtext, "path": path, "hits": 1,
                     "hash": fhash, "anchor": fanchor})
    LEARNED_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    log_edit("learn", f"{qtext[:60]} -> {path}")
    print(f"learned: {qtext[:60]!r} -> {path}")


def unlearn(con, qtext, path):
    """Negative evidence: remove a wrong learned mapping (append-only record)."""
    rows = _load_learned()
    kept = [r for r in rows if not (r["qtext"] == qtext and r["path"] == path)]
    if len(kept) == len(rows):
        sys.exit(f"no learned entry for {qtext[:40]!r} -> {path}")
    LEARNED_FILE.write_text(json.dumps(kept, ensure_ascii=False, indent=1))
    log_edit("unlearn", f"{qtext[:60]} -> {path}")
    print(f"unlearned: {qtext[:60]!r} -> {path}")


def load_qa():
    if not QA_FILE.exists():
        sys.exit(f"no QA pairs yet ({QA_FILE}). Seed them: haystack.py learn "
                 "\"<query>\" <path>, or copy a qa.json template into that file.")
    try:
        data = json.loads(QA_FILE.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{QA_FILE}: invalid JSON ({e})")
    return data.get("pairs", [])


def audit(con):
    # Learned corrections ARE ground truth: every `learn` (i.e. every real
    # "that's not what I want") automatically becomes a QA pair here — no
    # manual qa.json entry needed. qa.json stays as the seeded extra layer.
    pairs = load_qa() if QA_FILE.exists() else []
    learned = [(r["qtext"], r["path"]) for r in _load_learned()]
    if not pairs and not learned:
        sys.exit("no ground truth: qa.json empty and learned.json empty.")
    reindex(con)
    passed = failed = 0
    lpassed = lfailed = 0
    misses = []
    for qtext, expect in pairs:
        top, paths = query(con, qtext, k=5)
        ranked = [paths[i] for i in top]
        if any(p.startswith(expect) or p == expect for p in ranked[:3]):
            passed += 1
        else:
            failed += 1
            misses.append((qtext, expect, ranked[0]))
    for qtext, expect in learned:
        top, paths = query(con, qtext, k=5)
        ranked = [paths[i] for i in top]
        if any(p.startswith(expect) or p == expect for p in ranked[:3]):
            lpassed += 1
        else:
            lfailed += 1
            misses.append((qtext, expect, ranked[0]))
    line = (f"{time.strftime('%Y-%m-%d %H:%M')} audit: "
            f"{passed}/{passed + failed} qa.json, "
            f"{lpassed}/{lpassed + lfailed} learned passed")
    if misses:
        line += " | misses: " + "; ".join(
            f"{q[:40]!r} want {e[:50]} got {g}" for q, e, g in misses)
    with open(LOG_DIR / "audit.log", "a") as f:
        f.write(line + "\n")
    print(line)
    sys.exit(1 if misses else 0)


def _llm(prompt, max_tokens=200):
    """Query-generation backend: [eval] url/model in config (default
    127.0.0.1:8081 lfm3 — a 1.7B reasoning model is enough to paraphrase)."""
    sec = _cfg.get("eval", {})
    url = sec.get("url", "http://127.0.0.1:8081").rstrip("/")
    model = sec.get("model", "lfm3")
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens, "temperature": 0.7,
                    # reasoning models burn the whole budget thinking; eval-gen
                    # wants the answer, not the chain of thought
                    "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def eval_gen(con, limit=0, dest=None, workers=4):
    """P2: synthetic QA — for each (or N sampled) note, generate 2 natural
    queries (nickname-style + literal-style, no verbatim title reuse; arXiv
    2609.38473 protocol) -> qa_gen.json. Synthetic benchmarks reliably RANK
    retriever variants (arXiv 2508.11758) — exactly what the audit needs.
    Concurrent requests (workers=N): the server is the bottleneck either way,
    but overlapping streams cuts wall-clock when it has slots (-np)."""
    from concurrent.futures import ThreadPoolExecutor
    out = []
    files = [p for p in walk() if rel_of(p) not in SKIP_FILES]
    if limit:
        files = files[::max(1, len(files) // limit)]
    print(f"generating queries for {len(files)} files ({workers} workers, needs the LLM endpoint)...")

    def gen_one(p):
        head = "\n".join(p.read_text(errors="replace").split("\n")[:15])
        txt = _llm(
            "You generate test queries for a personal-notes search engine. "
            "Given a note's start, write 2 search queries the note's OWNER would type: "
            "one vague/nickname-style, one literal-style. NEVER reuse words from the title verbatim. "
            "Reply with ONLY the two queries, one per line.\n\n" + head, max_tokens=1024)
        return [x.strip("-*.0123456789. ") for x in txt.strip().split("\n") if x.strip()][:2]

    done = [0]
    def collect(fut):
        done[0] += 1
        if done[0] % 10 == 0:
            print(f"  {done[0]}/{len(files)} files, {len(out)} queries")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(gen_one, p): p for p in files}
        for f in futs:
            f.add_done_callback(collect)
        for f, p in futs.items():
            try:
                for q in f.result():
                    if len(q.split()) >= 2:
                        out.append([q, rel_of(p)])
            except Exception as e:
                print(f"  skip {rel_of(p)}: {e}")
    gen_file = Path(str(dest) if dest else _cfg.get("audit", {}).get("qa_gen_file", str(CFG_PATH.parent / "qa_gen.json")))
    gen_file.write_text(json.dumps({"pairs": out}, ensure_ascii=False, indent=1))
    print(f"wrote {len(out)} synthetic pairs -> {gen_file}")


def eval_pairs(con, pairs):
    """R@3 + MRR for a pairs list. Shared by `eval` and `pick-encoder`."""
    recall = mrr = 0.0
    for qtext, expect in pairs:
        top, paths = query(con, qtext, k=10)
        ranked = [paths[i] for i in top]
        for rank_i, p in enumerate(ranked[:3]):
            if p.startswith(expect) or p == expect:
                recall += 1
                mrr += 1.0 / (rank_i + 1)
                break
    n = max(1, len(pairs))
    return recall / n, mrr / n


def eval_run(con, qa_file=None):
    """Recall@3 + MRR over a synthetic (or any) pairs file. Diagnostic, not a gate."""
    f = Path(qa_file or _cfg.get("audit", {}).get("qa_gen_file", str(CFG_PATH.parent / "qa_gen.json")))
    if not f.exists():
        sys.exit(f"no {f} — run eval-gen first")
    pairs = json.loads(f.read_text()).get("pairs", [])
    recall, mrr = eval_pairs(con, pairs)
    print(f"eval {f.name}: Recall@3 {recall:.3f}  MRR {mrr:.3f}  ({len(pairs)} pairs)")


def pick_encoder(args):
    """A/B encoder candidates on THIS vault: score each on a throwaway DB over
    synthetic (or --pairs) queries, print the table, --yes swaps config and
    rebuilds. Encoder choice is corpus-shaped (same models, opposite verdicts
    on personal vs foreign vaults — RESEARCH.md), so candidates are measured,
    never predicted from model cards."""
    urls, pairs_file, limit, yes = [], None, 30, "--yes" in args
    rest = [x for x in args if x not in ("--yes",)]
    i = 0
    while i < len(rest):
        if rest[i] == "--pairs" and i + 1 < len(rest):
            pairs_file = rest[i + 1]
            i += 2
        elif rest[i] == "--limit" and i + 1 < len(rest):
            limit = int(rest[i + 1])
            i += 2
        else:
            urls.append(rest[i])
            i += 1
    if not urls:
        sys.exit("pick-encoder: give at least one embeddings url (current first)")
    # default candidate list: keep current first so ties keep the incumbent
    if EMBD_URL not in urls:
        urls = [EMBD_URL] + urls
    if not (pairs_file and Path(pairs_file).exists()):
        gen = Path(pairs_file or (Path(tempfile.mkdtemp(prefix="hay-pick-")) / "qa_gen.json"))
        print(f"no pairs file — generating ({limit} files, needs the LLM endpoint)...")
        eval_gen(connect(), limit=limit, dest=gen)
        pairs_file = str(gen)
    pairs = json.loads(Path(pairs_file).read_text()).get("pairs", [])
    if len(pairs) < 5:
        sys.exit(f"only {len(pairs)} pairs — too few to rank encoders; raise --limit")
    rows = []
    for url in urls:
        old = (EMBD_URL, EMBD_MODEL, DB)
        tmpdir = tempfile.mkdtemp(prefix="hay-pick-")
        try:
            globals()["EMBD_URL"], globals()["EMBD_MODEL"] = url, ""
            globals()["DB"] = Path(tmpdir) / "pick.sqlite"
            c = connect()
            reindex(c)
            r, m = eval_pairs(c, pairs)
            rows.append((url, r, m, len(pairs)))
            print(f"  {url}: Recall@3 {r:.3f}  MRR {m:.3f}")
        except Exception as e:
            print(f"  {url}: FAILED ({str(e)[:80]})")
        finally:
            globals()["EMBD_URL"], globals()["EMBD_MODEL"], globals()["DB"] = old
    if not rows:
        sys.exit("pick-encoder: every candidate failed")
    rows.sort(key=lambda x: (-x[1], -x[2]))
    print(f"\n{'url':<50} Recall@3   MRR")
    for url, r, m, n in rows:
        print(f"{url:<50} {r:.3f}      {m:.3f}   ({n} pairs)")
    win = rows[0]
    print(f"\nwinner: {win[0]}  (+{100 * (win[1] - rows[-1][1]):.1f}pp R@3 over worst)")
    if not yes:
        print("dry run — re-run with --yes to switch config and rebuild the index")
        return
    if win[0] == EMBD_URL:
        print("winner is the current encoder — config untouched")
        return
    t = CFG_PATH.read_text()
    m = re.search(r"^\[embeddings\]\n(?:(?!\[).*\n)*", t, re.M)
    if not m:
        sys.exit("no [embeddings] section in config")
    seg = m.group(0)
    new_seg = re.sub(r'^url\s*=\s*".*"$', f'url = "{win[0]}"', seg, flags=re.M)
    new_seg = re.sub(r'^model\s*=\s*".*"$', 'model = ""', new_seg, flags=re.M)
    CFG_PATH.write_text(t.replace(seg, new_seg))
    DB.unlink(missing_ok=True)  # embeddings swapped: full rebuild (learned pins live outside the cache)
    reindex(connect())
    print(f"switched to {win[0]}, index rebuilt ({CFG_PATH})")


def mine_aliases(con, min_folder_df=2, max_corpus_df=3, top=5):
    """Alias mining: tokens distinctive to one folder (frequent inside, rare
    outside) are likely the user's nicknames for that project. Candidate list
    for `learn` — nothing is auto-taught."""
    rows = list(con.execute("SELECT path, heading, text FROM chunks WHERE heading='(title)'"))
    corpus = [t.lower() for _, _, t in rows]
    n = len(corpus)
    leaf = {}
    for path, _, text in rows:
        folder = "/".join(path.split("/")[:-1]) if "/" in path else path
        leaf.setdefault(folder, []).append((path, text.lower()))
    out = []
    for folder, items in sorted(leaf.items()):
        nfiles = len(items)
        if nfiles < min_folder_df:
            continue
        cands = []
        for w in {x for _, t in items for x in re.findall(r"\w{4,}", t) if x not in STOP}:
            in_folder = sum(1 for _, t in items if w in t)
            df = sum(1 for t in corpus if w in t)
            # distinctive = repeated inside the folder, rare outside it
            if in_folder >= 2 and df <= max_corpus_df:
                cands.append((w, in_folder, df))
        cands.sort(key=lambda x: (-x[1], x[2]))
        if cands:
            out.append((folder, cands[:top]))
    for folder, cands in out:
        print(folder + "\t" + ", ".join(f"{w} (x{fi}, df{df})" for w, fi, df in cands))
    return out


def review(con, text):
    """Interactive feedback loop: rank, accept or teach."""
    top, paths, heads, texts, scores = rank(con, text, k=5)
    for n, i in enumerate(top, 1):
        snip = re.sub(r"\s+", " ", str(texts[i])[:100])
        print(f"{n}. {paths[i]}  ({heads[i]}, {scores[i]:.2f})\n   {snip}")
    print("Reply with the number that's right (1-5), 'q' to quit, or type the correct path.")
    try:
        ans = input("> ").strip()
    except EOFError:
        return
    if ans.isdigit() and 1 <= int(ans) <= len(top):
        learn(con, text, paths[int(top[int(ans) - 1])])
    elif ans and ans != "q":
        learn(con, text, ans)


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    cmd = a[0]
    dry = "--dry-run" in a
    a = [x for x in a if x != "--dry-run"]
    con = connect()
    if cmd == "reindex":
        reindex(con)
    elif cmd == "query":
        k = 5
        if "-k" in a:
            k = int(a[a.index("-k") + 1])
            a = a[:a.index("-k")] + a[a.index("-k") + 2:]
        query(con, " ".join(a[1:]), k)
    elif cmd == "find-file":
        find_file(con, a[1:])
    elif cmd == "append":
        append(a[1], a[2], " ".join(a[3:]), dry)
    elif cmd == "frontmatter":
        set_frontmatter(a[1], a[2], dry)
    elif cmd == "new-note":
        new_note(a[1], a[2], a[3] if len(a) > 3 and not a[3].startswith("-") else None, dry)
    elif cmd == "learn":
        learn(con, " ".join(a[1:-1]).strip('"'), a[-1])
    elif cmd == "unlearn":
        unlearn(con, " ".join(a[1:-1]).strip('"'), a[-1])
    elif cmd == "review":
        review(con, " ".join(a[1:]))
    elif cmd == "eval-gen":
        eval_gen(con, int(a[1]) if len(a) > 1 else 0,
                 workers=int(a[2]) if len(a) > 2 else 4)
    elif cmd == "pick-encoder":
        pick_encoder(a[1:])
    elif cmd == "eval":
        eval_run(con, a[1] if len(a) > 1 else None)
    elif cmd == "aliases":
        mine_aliases(con)
    elif cmd == "audit":
        audit(con)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
