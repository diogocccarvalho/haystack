#!/usr/bin/env python3
"""notes-router HTTP API — same ops as the CLI, for agents and integrations.

Stdlib only. Endpoints:
  GET  /health                 -> {"ok": true, "chunks": N}
  POST /query   {"q": str, "k": int=5}      -> [{"path","heading","score","snippet"}]
  POST /find    {"q": str}                  -> {"path": "..."}
  POST /append  {"file","heading","text","dry_run":false}
  POST /frontmatter {"file","key","value","dry_run":false}
  POST /new-note {"dir","name","title":null,"dry_run":false}
  POST /learn   {"q","path"}
  POST /reindex {}
Auth: optional bearer token — set [api] token = "..." in config.toml.

Usage: python3 server.py [--port 8083]
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import notes_router as nr

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
