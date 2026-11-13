#!/usr/bin/env python3
"""notes-router — semantic router + deterministic editor for a local markdown vault.

Local, zero-API: points at ANY OpenAI-compatible /v1/embeddings endpoint
(llama.cpp, Ollama, LM Studio, remote). Learns from your corrections.

Usage:
  notes-router.py reindex                  # incremental reindex
  notes-router.py query <text> [-k N]      # top-N: path<TAB>heading<TAB>score<TAB>snippet
  notes-router.py find-file <words>        # best path only
  notes-router.py append <file> <heading> <text> [--dry-run]
  notes-router.py frontmatter <file> k=v [--dry-run]
  notes-router.py new-note <dir> <name> ["Title"] [--dry-run]
  notes-router.py learn "<query>" <path>   # teach a correction
  notes-router.py review <text>            # interactive feedback loop
  notes-router.py aliases                  # mine nickname candidates per folder
  notes-router.py audit                    # reindex + run QA pairs (exit 1 on miss)

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
import time
import tomllib
import urllib.request
from pathlib import Path

import numpy as np

CFG_PATH = Path(os.environ.get("NR_CONFIG", Path.home() / ".config/notes-router/config.toml"))
if not CFG_PATH.exists():
    sys.exit(f"no config at {CFG_PATH} — copy config.toml from the repo and edit vault.path")

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


def learn_lookup(con, text, paths, q):
    """Past corrections boost identical/similar queries (and their folder).
    Stored OUTSIDE the index DB (learned.json) so index rebuilds never wipe
    learned ground truth (append-only; Reglos-style persistence)."""
    rows = _load_learned()
    if not rows:
        return np.zeros(len(paths))
    qs = embed([r["qtext"] for r in rows])
    sims = qs @ q
    boost = np.zeros(len(paths))
    for row, sim in zip(rows, sims):
        if sim > 0.75:
            for i, p in enumerate(paths):
                if p == row["path"] or p.startswith(row["path"].rsplit("/", 1)[0]):
                    boost[i] += 0.10 * sim * min(row["hits"], 3)
    return boost


def connect():
    import sqlite3
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        path TEXT, heading TEXT, hash TEXT, vec BLOB, text TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS files(path TEXT, mtime REAL, hash TEXT)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_c ON chunks(path)")
    return con


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
            vecs = embed([c[1] for c in cs])
            for (head, txt), v in zip(cs, vecs):
                cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?)",
                            (rel, head, h, v.astype(np.float32).tobytes(), txt))
        cur.execute("INSERT INTO files VALUES(?,?,?)", (rel, p.stat().st_mtime, h))
        n_new += 1
    for rel in set(db_files) - seen:
        cur.execute("DELETE FROM chunks WHERE path=?", (rel,))
        cur.execute("DELETE FROM files WHERE path=?", (rel,))
        n_del += 1
    con.commit()
    print(f"indexed: {n_new} updated, {n_del} removed, {sum(1 for _ in walk())} files")


STOP = set("the a an of about my i me that this thing stuff should where do is are for on in to with and or".split())
if "urllib.request" not in sys.modules:
    import urllib.request
    import urllib.error


def lexical_boost(text, texts):
    toks = [w for w in re.findall(r"\w{4,}", text.lower()) if w not in STOP]
    if not toks:
        return np.zeros(len(texts))
    # IDF weighting: common words count little, distinctive ones count
    lowered = [t.lower() for t in texts]
    n = len(texts)
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
    scores += learn_lookup(con, text, paths, q)  # additive post-fusion
    order = list(np.argsort(-scores))
    order = rerank(con, text, order, texts)
    top = order[:k]
    return top, paths, heads, texts, scores


def rerank(con, text, order, texts):
    """Cross-encoder rerank of the fused top (T2-RAGBench: Recall@5
    0.695 -> 0.816). Optional [rerank] url/model/top in config; the reranker
    only REORDERS (fusion scores stay the reported scores). Any failure falls
    back to the fused order — retrieval must never hard-depend on a second
    server being up."""
    url = _cfg.get("rerank", {}).get("url", "")
    if not url or len(order) < 2:
        return order
    topn = int(_cfg.get("rerank", {}).get("top", 20))
    cand = order[:topn]
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/rerank",
            json.dumps({"query": text, "documents": [texts[i] for i in cand]}).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            res = json.load(r)["results"]
    except Exception as e:  # degraded mode: fused order only
        log_edit("rerank-fallback", str(e)[:80])
        return order
    ranked = [cand[r["index"]] for r in res if r["index"] < len(cand)]
    tail = [i for i in order if i not in cand]
    return ranked + tail


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
    vecs = embed([c[1] for c in cs]) if cs else []
    for (head, txt), v in zip(cs, vecs):
        cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?)",
                    (rel, head, h, v.astype(np.float32).tobytes(), txt))
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
        pat = rf"^(## {re.escape(heading)}\n)"
        m = re.search(pat, t, re.M)
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
    if not (real_path(path).is_file() or real_path(path).is_dir()):
        sys.exit(f"{path}: not a vault path")
    rows = _load_learned()
    for row in rows:
        if row["qtext"] == qtext and row["path"] == path:
            row["hits"] = min(row["hits"] + 1, 3)
            break
    else:
        rows.append({"qtext": qtext, "path": path, "hits": 1})
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
        sys.exit(f"no QA pairs yet ({QA_FILE}). Seed them: notes-router.py learn "
                 "\"<query>\" <path>, or copy a qa.json template into that file.")
    try:
        data = json.loads(QA_FILE.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"{QA_FILE}: invalid JSON ({e})")
    return data.get("pairs", [])


def audit(con):
    pairs = load_qa()
    if not pairs:
        sys.exit("qa.json has no pairs — audit needs ground truth to be meaningful.")
    reindex(con)
    passed = failed = 0
    misses = []
    for qtext, expect in pairs:
        top, paths = query(con, qtext, k=5)
        ranked = [paths[i] for i in top]
        if any(p.startswith(expect) or p == expect for p in ranked[:3]):
            passed += 1
        else:
            failed += 1
            misses.append((qtext, expect, ranked[0]))
    line = f"{time.strftime('%Y-%m-%d %H:%M')} audit: {passed}/{passed + failed} passed"
    if misses:
        line += " | misses: " + "; ".join(
            f"{q[:40]!r} want {e[:50]} got {g}" for q, e, g in misses)
    with open(LOG_DIR / "audit.log", "a") as f:
        f.write(line + "\n")
    print(line)
    sys.exit(1 if misses else 0)


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
    elif cmd == "aliases":
        mine_aliases(con)
    elif cmd == "audit":
        audit(con)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
