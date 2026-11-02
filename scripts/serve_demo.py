"""A loopback-only, single-session demonstration server."""

import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from src.appliances.config import ROOT, MODEL_DIR
from src.appliances.demo import Replay


def serve(port=8767, database=None, model_dir=MODEL_DIR):
    replay = Replay(database or ROOT / "outputs/appliances_demo.sqlite3", model_dir)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind="application/json; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return self.send(403, "{}")
            if self.path == "/":
                html = (
                    (ROOT / "web/index.html")
                    .read_text(encoding="utf-8")
                    .replace("__TOKEN__", token)
                )
                return self.send(200, html, "text/html; charset=utf-8")
            if self.path == "/api/state":
                return self.send(200, json.dumps(replay.dispatch(), allow_nan=False))
            self.send(404, "{}")

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed = (f"http://127.0.0.1:{port}", f"http://localhost:{port}")
            if (
                self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}")
                or origin not in allowed
                or self.headers.get("X-Session-Token") != token
            ):
                return self.send(403, json.dumps({"error": "Invalid local session"}))
            if self.path != "/api/action":
                return self.send(404, "{}")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 8192:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                result = replay.dispatch(body.get("action"), body.get("decision"))
                self.send(200, json.dumps(result, allow_nan=False))
            except (ValueError, TypeError, KeyError) as exc:
                self.send(400, json.dumps({"error": str(exc)}))

        def log_message(self, *args):
            pass

    print(f"Appliances demo http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--database", type=Path)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    a = p.parse_args()
    serve(a.port, a.database, a.model_dir)
