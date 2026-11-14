#!/usr/bin/env python3
"""haystack HTTP API — same ops as the CLI, for agents and integrations.

Stdlib only. Endpoints:
  GET  /health                 -> {"ok": true, "chunks": N}
  POST /query   {"q": str, "k": int=5}      -> [{"path","heading","score","snippet"}]
  POST /find    {"q": str}                  -> {"path": "..."}
  POST /append  {"file","heading","text","dry_run":false}
  POST /frontmatter {"file","key","value","dry_run":false}
  POST /new-note {"dir","name","title":null,"dry_run":false}
  POST /learn   {"q","path"}
  POST /review  {"q"}                        -> {"candidates":[{n,path,heading,score,snippet}]}
  POST /review  {"q","path"}                 -> teach (learn + index log); path may be a
                                                candidate number from the first call
  POST /reindex {}
Auth: optional bearer token — set [api] token = "..." in config.toml.

Usage: python3 server.py [--port 8083]
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import haystack as nr

CFG = __import__("tomllib").load(open(nr.CFG_PATH, "rb"))
API_TOKEN = (CFG.get("api") or {}).get("token", "")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self):
        if not API_TOKEN:
            return True
        return self.headers.get("Authorization") == f"Bearer {API_TOKEN}"

    def do_GET(self):
        if self.path == "/health":
            con = nr.connect()
            n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            self._send(200, {"ok": True, "chunks": n})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth():
            self._send(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            con = nr.connect()
            if self.path == "/query":
                if not str(body.get("q", "")).strip():
                    self._send(400, {"error": "empty query"}); return
                k = int(body.get("k", 5))
                top, paths, heads, texts, scores = nr.rank(con, body["q"], k)
                if len(top):  # log for temporal asked-before (the agent's real query path)
                    nr.log_event(con, "query", paths[int(top[0])], body["q"])
                out = [{"path": paths[i], "heading": heads[i],
                        "score": round(float(scores[i]), 4),
                        "snippet": texts[i][:180]} for i in top]
                self._send(200, out)
            elif self.path == "/find":
                top, paths, _, _, _ = nr.rank(con, body["q"], k=1)
                self._send(200, {"path": str(nr.real_path(paths[int(top[0])])) if len(top) else None})
            elif self.path == "/append":
                nr.append(body["file"], body["heading"], body["text"],
                          dry=bool(body.get("dry_run")))
                self._send(200, {"ok": True})
            elif self.path == "/frontmatter":
                nr.set_frontmatter(body["file"], f"{body['key']}={body['value']}",
                                   dry=bool(body.get("dry_run")))
                self._send(200, {"ok": True})
            elif self.path == "/new-note":
                nr.new_note(body["dir"], body["name"], body.get("title"),
                            dry=bool(body.get("dry_run")))
                self._send(200, {"ok": True})
            elif self.path == "/learn":
                nr.learn(con, body["q"], body["path"])
                self._send(200, {"ok": True})
            elif self.path == "/reindex":
                nr.reindex(con)
                self._send(200, {"ok": True})
            elif self.path == "/review":
                # Two-call teach loop for headless agents:
                #  {"q"}            -> ranked candidates
                #  {"q","path"}     -> teach (path = vault path OR candidate n)
                qtext = str(body.get("q", "")).strip()
                if not qtext:
                    self._send(400, {"error": "empty query"}); return
                if "path" not in body or body["path"] in (None, ""):
                    top, paths, heads, texts, scores = nr.rank(con, qtext, k=5)
                    cands = [{"n": n + 1, "path": paths[i], "heading": heads[i],
                              "score": round(float(scores[i]), 4),
                              "snippet": re.sub(r"\s+", " ", str(texts[i]))[:140]}
                             for n, i in enumerate(top)]
                    self._send(200, {"q": qtext, "candidates": cands,
                                     "teach": {"path": "<candidate n or vault path>"}})
                    return
                p = str(body["path"])
                if p.isdigit():
                    top, paths, _, _, _ = nr.rank(con, qtext, k=5)
                    idx = int(p) - 1
                    if idx < 0 or idx >= len(top):
                        self._send(400, {"error": f"candidate {p} out of range (1-{len(top)})"}); return
                    p = paths[int(top[idx])]
                nr.learn(con, qtext, p)
                self._send(200, {"ok": True, "taught": {"q": qtext, "path": p}})
            elif self.path == "/aliases":
                out = nr.mine_aliases(con)
                self._send(200, [{"folder": f, "candidates": c} for f, c in out])
            else:
                self._send(404, {"error": "not found"})
        except SystemExit as e:          # nr uses sys.exit(str) for user errors
            self._send(400, {"error": str(e)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8083
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
