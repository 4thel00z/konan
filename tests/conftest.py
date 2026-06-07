import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def _embed(text: str) -> list[float]:
    # Deterministic 2D embeddings: cat-topic vs everything else.
    return [1.0, 0.0] if "cat" in text.lower() else [0.0, 1.0]


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        data = [
            {"object": "embedding", "index": i, "embedding": _embed(t)}
            for i, t in enumerate(payload["input"])
        ]
        body = json.dumps(
            {"object": "list", "model": payload["model"], "data": data}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="session")
def embeddings_url():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
