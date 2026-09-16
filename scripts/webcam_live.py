"""Local online RGB → ORCA simulation, with a synchronized browser preview.

Run: .venv/bin/python scripts/webcam_live.py
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from orca_feetech.live import LiveSession

ROOT = Path(__file__).resolve().parents[1]


def make_server(session, port=8767):
    page = (ROOT / "scripts/webcam_live.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, payload, code=200, content_type="application/json; charset=utf-8"):
            body = json.dumps(payload, ensure_ascii=False).encode() if isinstance(payload, dict) else payload
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(page, content_type="text/html; charset=utf-8")
            if url.path in {"/api/frame", "/api/state"}:
                after = parse_qs(url.query).get("after", [None])[0]
                data = session.snapshot(after, heartbeat=url.path == "/api/frame")
                if url.path == "/api/state":
                    data.pop("pair", None)
                return self.send(data)
            return self.send({"error": "Not found"}, code=404)

        def do_POST(self):
            actual_port = self.server.server_address[1]
            if self.headers.get("Origin") not in {f"http://127.0.0.1:{actual_port}", f"http://localhost:{actual_port}"}:
                return self.send({"error": "Origin rejected"}, code=403)
            if self.path == "/api/start":
                session.start()
            elif self.path == "/api/stop":
                session.stop()
            elif self.path == "/api/recalibrate":
                session.recalibrate()
            else:
                return self.send({"error": "Not found"}, code=404)
            return self.send({"accepted": True})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--rate", type=int, default=20)
    args = parser.parse_args()
    session = LiveSession(rate=args.rate, camera_index=args.camera)
    server = make_server(session, args.port)
    print(f"Live ORCA: http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        if session.worker:
            session.worker.join(timeout=15)
        server.server_close()


if __name__ == "__main__":
    main()
