import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# Models whose first request fails with a 500 (for retry tests). Maps
# model name -> True once the failure has been served.
_FAILED_ONCE: dict[str, bool] = {}


def _embed(text: str, dimensions: int | None) -> list[float]:
    # Deterministic 2D embeddings: cat-topic vs everything else,
    # zero-padded/truncated to `dimensions` when requested.
    vec = [1.0, 0.0] if "cat" in text.lower() else [0.0, 1.0]
    if dimensions is not None:
        vec = (vec + [0.0] * dimensions)[:dimensions]
    return vec


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        model = payload["model"]

        if model.startswith("fail-once") and not _FAILED_ONCE.get(model):
            _FAILED_ONCE[model] = True
            self._send(
                500,
                {"error": {"message": "boom", "type": "server_error"}},
            )
            return

        dimensions = payload.get("dimensions")
        data = [
            {"object": "embedding", "index": i, "embedding": _embed(t, dimensions)}
            for i, t in enumerate(payload["input"])
        ]
        self._send(
            200,
            {
                "object": "list",
                "model": model,
                "data": data,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
        )

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
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
