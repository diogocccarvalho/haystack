#!/usr/bin/env python3
"""Simulated-use bug hunt for notes-router → haystack (2026-11-13).

Builds a fresh adversarial vault in a temp dir, runs the CLI + API the way an
agent would (including hostile inputs), prints PASS/FAIL per scenario.
Exit 0 only if every scenario passes. Run:  python3 tests/sim.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "haystack.py"
EMBD = "http://127.0.0.1:8082/v1/embeddings"

results = []


def run_cli(cfg, *args, expect_rc=0):
    env = dict(os.environ, NR_CONFIG=str(cfg))
    r = subprocess.run(["python3", str(SCRIPT), *args], capture_output=True,
                       text=True, env=env, timeout=120)
    return r


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="nrsim-"))
    vault = tmp / "vault"
    (vault / "proj/alpha").mkdir(parents=True)
    (vault / "proj/beta").mkdir(parents=True)
    (vault / "misc").mkdir(parents=True)

    # adversarial corpus
    (vault / "proj/alpha/main.md").write_text(
        '---\ntitle: "Alpha invoices"\ndate: 2026-11-13\n---\n# Alpha\n\n## Status\nrunning\n'
        '## Next step\nship the parser\n')
    (vault / "proj/beta/note.md").write_text(
        '---\ntitle: "Beta drones"\ndate: 2026-11-12\n---\n# Beta\n\n## Status\nparked\n\n')
    (vault / "misc/plain.md").write_text("no frontmatter here\n## Only heading\n- a\n- b\n")
    (vault / "misc/empty.md").write_text("")
    (vault / "misc/nl.md").write_text("\n\n\n")
    (vault / "misc/big.md").write_text("# Big\n\n## Wall\n\n" + "lorem ipsum dolor sit amet. " * 400)
    (vault / "proj/alpha/tricky.md").write_text(
        '---\ntitle: "Tricky"\ndate: 2026-11-13\n---\n# T\n\n## Status (v2.0+beta)\nbody\n')
    (vault / "misc/emoji-🚀.md").write_text('---\ntitle: "Emoji file 🚀"\ndate: 2026-11-13\n---\n# E\n## S\nrocket stuff\n')
    (vault / "INDEX.md").write_text("# INDEX\n")

    db = tmp / "index.sqlite"
    cfg = tmp / "cfg.toml"
    cfg.write_text(f"""
[vault]
path = "{vault}"
index_convention = "list"
exclude_dirs = []
exclude_files = ["LOG.md"]
external = []
[index]
db = "{db}"
[embeddings]
url = "{EMBD}"
model = ""
[audit]
qa_file = "{tmp / 'qa.json'}"
""")
    (tmp / "qa.json").write_text(json.dumps({
        "pairs": [["alpha invoice parser thing", "proj/alpha"],
                  ["the drone project", "proj/beta/note.md"]]}))

    # --- reindex (twice: idempotency)
    r = run_cli(cfg, "reindex")
    check("reindex runs", r.returncode == 0, r.stderr[-200:])
    r2 = run_cli(cfg, "reindex")
    check("reindex idempotent", "0 updated" in r2.stdout, r2.stdout)

    # --- queries
    def top1(q):
        r = run_cli(cfg, "query", q, "-k", "1")
        return r.stdout.split("\t")[0] if r.stdout else ""

    check("query: invoices->alpha", top1("alpha invoice parser thing").startswith("proj/alpha"),
          top1("alpha invoice parser thing"))
    check("query: drones->beta", top1("the drone project").startswith("proj/beta"),
          top1("the drone project"))
    check("query: emoji file reachable", top1("rocket stuff") == "misc/emoji-🚀.md",
          top1("rocket stuff"))
    check("query: stopwords-only doesn't crash", run_cli(cfg, "query", "the a of the").returncode == 0)
    check("query: empty string doesn't crash", run_cli(cfg, "query", "").returncode == 0)
    check("query: -k 0 no crash", run_cli(cfg, "query", "alpha", "-k", "0").returncode == 0)

    # --- learned boost actually flips a query
    before = top1("the invoices thing")
    run_cli(cfg, "learn", "the invoices thing", "proj/alpha/main.md")
    after = top1("the invoices thing")
    check("learn flips query to alpha", after.startswith("proj/alpha"), f"{before} -> {after}")

    # --- editor ops
    r = run_cli(cfg, "append", "proj/alpha/main.md", "Status", "line one")
    check("append ok", r.returncode == 0 and "running" in (vault / "proj/alpha/main.md").read_text())
    r = run_cli(cfg, "append", "proj/alpha/tricky.md", "Status (v2.0+beta)", "tricky line")
    check("append regex-special heading", r.returncode == 0
          and "tricky line" in (vault / "proj/alpha/tricky.md").read_text(), r.stderr[-150:])
    r = run_cli(cfg, "append", "proj/alpha/main.md", "NoSuchHeading", "x")
    check("append bad heading fails cleanly", r.returncode != 0 and "not found" in (r.stderr + r.stdout))
    rd = run_cli(cfg, "append", "proj/alpha/main.md", "Status", "dry only", "--dry-run")
    check("append --dry-run writes nothing",
          rd.returncode == 0 and "dry only" not in (vault / "proj/alpha/main.md").read_text())
    # security: traversal
    r = run_cli(cfg, "append", "../../etc/hostname", "x", "hostile write")
    outside = Path("/etc/hostname").read_text()
    check("append blocks traversal", r.returncode != 0 and "hostile write" not in outside)
    r = run_cli(cfg, "learn", "hostile", "../../../etc/passwd")
    check("learn blocks traversal", r.returncode != 0, r.stdout + r.stderr)
    # frontmatter on fm-less file
    r = run_cli(cfg, "frontmatter", "misc/plain.md", "status=active")
    t = (vault / "misc/plain.md").read_text()
    check("frontmatter adds block", r.returncode == 0 and t.startswith("---\nstatus: active"), t[:40])
    # new-note dup
    run_cli(cfg, "new-note", "misc", "fresh", "Fresh")
    r = run_cli(cfg, "new-note", "misc", "fresh", "Fresh")
    check("new-note refuses dup", r.returncode != 0)
    # INDEX maintenance
    idx = (vault / "INDEX.md").read_text()
    check("INDEX covers created notes", "misc/fresh.md" in idx and "proj/alpha/main.md" in idx)

    # --- unintended-use battery (agent will do all of these eventually)
    (vault / "misc/crlf.md").write_bytes(b"---\r\ntitle: \"CRLF note\"\r\ndate: 2026-11-13\r\n---\r\n\r\n# C\r\n\r\n## Status\r\nwindows line endings\r\n")
    (vault / "misc/bom.md").write_bytes(b'\xef\xbb\xbf---\ntitle: "BOM"\ndate: 2026-11-13\n---\n\n# B\n\n## S\nbom content\n')
    (vault / "misc/dir.md").mkdir()  # a DIRECTORY named like a note
    r = run_cli(cfg, "reindex")
    check("reindex survives CRLF + BOM + .md-named-dir", r.returncode == 0, r.stderr[-200:])
    r = run_cli(cfg, "append", "misc/crlf.md", "Status", "into crlf")
    check("append into CRLF heading", r.returncode == 0
          and "into crlf" in (vault / "misc/crlf.md").read_bytes().decode(), r.stderr[-200:] + r.stdout[-100:])
    r = run_cli(cfg, "append", "proj/alpha/main.md", "## Inception", "meta")
    check("append heading containing '## ' fails cleanly", r.returncode != 0)
    r = run_cli(cfg, "append", "proj/alpha/main.md", "Status", "")
    check("append empty text -> empty line ok", r.returncode == 0)
    r = run_cli(cfg, "frontmatter", "proj/alpha/main.md", "tags=a=b=c,d=e")
    t = (vault / "proj/alpha/main.md").read_text()
    check("frontmatter value with '='", r.returncode == 0 and "tags: a=b=c,d=e" in t)
    r = run_cli(cfg, "frontmatter", "proj/alpha/main.md", "noequals")
    check("frontmatter missing '=' fails cleanly", r.returncode != 0)
    r = run_cli(cfg, "new-note", "misc", "../escape")
    check("new-note blocks ../ in name", r.returncode != 0
          and not (vault / "escape.md").exists())
    r = run_cli(cfg, "new-note", "misc", "sub/dir/name")
    check("new-note blocks / in name", r.returncode != 0)
    r = run_cli(cfg, "learn", "no args here")
    check("learn without path fails cleanly", r.returncode != 0)
    r = run_cli(cfg, "learn", "dup query", "misc/ghost.md")
    check("learn nonexistent path fails", r.returncode != 0)
    r = run_cli(cfg, "learn", "dir as target", "misc/dir.md")
    check("learn accepts .md-named dir", r.returncode == 0)
    r = run_cli(cfg, "unlearn", "dir as target", "misc/dir.md")
    check("cleanup dir learn", r.returncode == 0)
    r = run_cli(cfg, "learn", "folder-level alias", "proj/alpha")
    check("learn accepts folder (folder-scoped boost)", r.returncode == 0)
    r = run_cli(cfg, "unlearn", "folder-level alias", "proj/alpha")
    check("unlearn removes entry", r.returncode == 0)
    r = run_cli(cfg, "unlearn", "folder-level alias", "proj/alpha")
    check("unlearn twice fails cleanly", r.returncode != 0)
    run_cli(cfg, "learn", "quotes 'and' \"double\" path", "proj/beta/note.md")
    run_cli(cfg, "learn", "newline\nin query", "proj/beta/note.md")
    check("learn weird queries ok", run_cli(cfg, "query", "quotes and double path", "-k", "1").returncode == 0)
    (tmp / "bad.json").write_text("{not json")
    badcfg = tmp / "badqa.toml"
    badcfg.write_text(cfg.read_text().replace(str(tmp / "qa.json"), str(tmp / "bad.json")))
    r = run_cli(badcfg, "audit")
    check("audit with malformed qa.json fails cleanly", r.returncode != 0 and "Traceback" not in r.stderr)
    r = run_cli(cfg, "query", "alpha", "-k", "-5")
    check("query -k negative no crash", r.returncode in (0, 1), r.stderr[-100:])
    r = run_cli(cfg, "query", "🚀🚀🚀", "-k", "1")
    check("query emoji-only no crash", r.returncode == 0)
    r = run_cli(cfg, "aliases")
    check("aliases after weird corpus no crash", r.returncode == 0)
    # --- stale-pin re-anchoring (reidmarlow edge case): rename/move taught targets
    def learned_rows():
        return json.loads((tmp / "learned.json").read_text())

    # dedicated target so other pins/qa pairs stay untouched
    (vault / "misc/movable.md").write_text(
        '---\ntitle: "Gamma wallet"\ndate: 2026-11-13\n---\n# Gamma\n\n## Ledger\nwallet ledger entries\n')
    run_cli(cfg, "reindex")
    # fuzzy pin: only the pin carries this query (dense can't find it alone)
    r = run_cli(cfg, "learn", "where is my money thing", "misc/movable.md")
    check("learn stores anchor fields", "hash" in learned_rows()[-1] and learned_rows()[-1]["anchor"],
          json.dumps(learned_rows()[-1])[:120])
    check("pin-driven hit before move", top1("where is my money thing") == "misc/movable.md",
          top1("where is my money thing"))
    # rename-only: hash re-anchor path
    (vault / "misc/movable.md").rename(vault / "misc/movable-renamed.md")
    run_cli(cfg, "reindex")
    check("reanchored after rename", top1("where is my money thing") == "misc/movable-renamed.md",
          top1("where is my money thing"))
    check("rename keeps history", learned_rows()[-1]["history"] == ["misc/movable.md"],
          json.dumps(learned_rows()[-1])[:150])
    # cross-dir move, THEN edit (hash mismatch -> semantic fallback must carry it)
    (vault / "misc/movable-renamed.md").rename(vault / "proj/movable-moved.md")
    (vault / "proj/movable-moved.md").write_text(
        '---\ntitle: "Gamma wallet"\ndate: 2026-11-13\n---\n# Gamma\n\n## Ledger\nwallet ledger, edited after move\n')
    run_cli(cfg, "reindex")
    check("reanchored after move+edit (semantic)", top1("where is my money thing") == "proj/movable-moved.md",
          top1("where is my money thing"))
    check("history accumulates", learned_rows()[-1]["history"] == ["misc/movable.md", "misc/movable-renamed.md"],
          json.dumps(learned_rows()[-1])[:200])
    # deleted target stays stale EVEN with a near-duplicate decoy present:
    # a deletion-only pass must not re-anchor (decoy would inherit the pin)
    (vault / "misc/decoy.md").write_text(
        '---\ntitle: "Gamma wallet copy"\ndate: 2026-11-13\n---\n# Gamma\n\n## Ledger\nwallet ledger entries (near-duplicate decoy)\n')
    run_cli(cfg, "reindex")  # decoy exists while original still live
    ghost2 = [r0 for r0 in learned_rows() if r0["qtext"] == "where is my money thing"][0]
    check("decoy coexistence doesn't move pin", ghost2["path"] == "proj/movable-moved.md",
          json.dumps(ghost2)[:150])
    (vault / "proj/movable-moved.md").unlink()  # delete ONLY the pinned file
    run_cli(cfg, "reindex")
    ghost = [r0 for r0 in learned_rows() if r0["qtext"] == "where is my money thing"][0]
    check("deleted target stays stale (no wrong auto-fix)", ghost["path"] == "proj/movable-moved.md",
          json.dumps(ghost)[:150])
    # legacy rows without anchor fields must not crash reanchor
    legacy = learned_rows()
    legacy.append({"qtext": "legacy pin", "path": "misc/long-gone.md", "hits": 1})
    (tmp / "learned.json").write_text(json.dumps(legacy))
    r = run_cli(cfg, "reindex")
    check("reanchor skips legacy rows cleanly", r.returncode == 0, r.stderr[-150:])
    run_cli(cfg, "unlearn", "where is my money thing", "proj/movable-moved.md")
    run_cli(cfg, "unlearn", "legacy pin", "misc/long-gone.md")

    # --- audit
    r = run_cli(cfg, "audit")
    check("audit passes its own pairs", r.returncode == 0 and "2/2 qa.json" in r.stdout and "learned passed" in r.stdout,
          r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-150:])

    # --- pick-encoder: dry run measures candidates, --yes with winner==current
    # leaves config alone (single live endpoint can't exercise a real swap)
    (tmp / "pick-pairs.json").write_text(json.dumps({"pairs": [
        ["alpha invoice parser thing", "proj/alpha/main.md"],
        ["the drone project", "proj/beta/note.md"],
        ["rocket stuff emoji", "misc/emoji-🚀.md"],
        ["crlf windows endings", "misc/crlf.md"],
        ["bom marker note", "misc/bom.md"]]}))
    before_cfg = cfg.read_text()
    r = run_cli(cfg, "pick-encoder", EMBD, "--pairs", str(tmp / "pick-pairs.json"))
    check("pick-encoder dry run scores candidates", r.returncode == 0 and "winner:" in r.stdout
          and EMBD in r.stdout, (r.stdout + r.stderr)[-200:])
    check("pick-encoder dry run leaves config", cfg.read_text() == before_cfg)
    r = run_cli(cfg, "pick-encoder", "http://127.0.0.1:59999/v1/embeddings",
                "--pairs", str(tmp / "pick-pairs.json"), "--yes")
    check("pick-encoder dead candidate skipped, current wins, config untouched",
          r.returncode == 0 and "FAILED" in r.stdout and cfg.read_text() == before_cfg,
          (r.stdout + r.stderr)[-200:])
    nocfg = tmp / "nollm.toml"
    nocfg.write_text(cfg.read_text().replace("8082/v1/embeddings", "8082/v1/embeddings")
                     + '\n[eval]\nurl = "http://127.0.0.1:59998"\n')
    r = run_cli(nocfg, "pick-encoder", EMBD)
    check("pick-encoder missing pairs + no LLM fails cleanly", r.returncode != 0
          and "Traceback" not in r.stderr, (r.stdout + r.stderr)[-400:])

    # deleted file still served from index -> stale path reported
    (vault / "proj/beta/note.md").unlink()
    r = run_cli(cfg, "query", "the drone project", "-k", "1")
    check("query without reindex after delete: no crash", r.returncode == 0)
    # --- API
    api = subprocess.Popen(
        ["python3", str(REPO / "server.py"), "--port", "8099"],
        env=dict(os.environ, NR_CONFIG=str(cfg)), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    try:
        def post(path, payload):
            req = urllib.request.Request(f"http://127.0.0.1:8099{path}",
                                         json.dumps(payload).encode(), {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return resp.status, json.load(resp)
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)

        st, body = post("/query", {"q": "alpha invoice parser thing", "k": 2})
        check("api /query", st == 200 and body and body[0]["path"].startswith("proj/alpha"), str(body)[:80])
        st, body = post("/find", {"q": "rocket stuff"})
        check("api /find", st == 200 and body["path"].endswith("emoji-🚀.md"), str(body))
        st, body = post("/query", {"q": "  "})
        check("api empty q -> 400", st == 400)
        st, body = post("/append", {"file": "nope.md", "heading": "S", "text": "t", "dry_run": True})
        check("api missing file -> 4xx", st in (400, 500) and "error" in body, f"{st} {body}")
        st, body = post("/append", {"file": "../../etc/hostname", "heading": "S", "text": "h", "dry_run": True})
        check("api traversal blocked", st == 400 and "not found" in body.get("error", ""), f"{st} {body}")
        st, body = post("/learn", {"q": "rocket ship", "path": "misc/emoji-🚀.md"})
        check("api /learn", st == 200)
        st, body = post("/reindex", {})
        check("api /reindex", st == 200)
        st, body = post("/aliases", {})
        check("api /aliases", st == 200 and isinstance(body, list))
        # /review: two-call teach loop
        st, body = post("/review", {"q": "alpha invoice parser thing"})
        check("api /review candidates", st == 200 and body.get("candidates")
              and body["candidates"][0]["path"].startswith("proj/alpha"), str(body)[:100])
        cand_n = body["candidates"][0]["n"] if st == 200 and body.get("candidates") else "1"
        st, body = post("/review", {"q": "alpha invoice parser thing", "path": str(cand_n)})
        check("api /review teach by n", st == 200 and body.get("ok"), f"{st} {body}")
        st, body = post("/review", {"q": "alpha invoice parser thing", "path": "999"})
        check("api /review bad n -> 400", st == 400 and "out of range" in body.get("error", ""), f"{st} {body}")
        st, body = post("/review", {"q": "  "})
        check("api /review empty q -> 400", st == 400)
        st, body = post("/review", {"q": "teach traversal probe", "path": "../../etc/hostname"})
        check("api /review traversal blocked", st == 400 and "not a vault path" in body.get("error", ""), f"{st} {body}")
    finally:
        api.terminate()

    fails = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(fails)}/{len(results)} passed")
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
