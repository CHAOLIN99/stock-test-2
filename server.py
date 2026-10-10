#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import json
import os
import posixpath
import socketserver
import sys
import webbrowser
from functools import partial
from urllib.parse import unquote, urlsplit

# Only the dashboard is served; .git/, .venv/, data/, saved_models/ etc. stay private.
STATIC_FILES = {"/dashboard.html"}
STATIC_DIRS = ("/css/", "/js/", "/results/")
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, loopback_only: bool, **kwargs):
        self._loopback_only = loopback_only
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        return

    def list_directory(self, path):
        self.send_error(404)
        return None

    def do_GET(self) -> None:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        if self._loopback_only and host not in LOOPBACK_HOSTS:
            # Blocks DNS-rebinding: a malicious site resolving its own hostname to 127.0.0.1.
            return self._send_json(403, {"error": "Forbidden host"})

        path = posixpath.normpath(unquote(urlsplit(self.path).path))
        if path.startswith("/api/"):
            return self._handle_api(path[len("/api/"):].strip("/"))
        if path == "/":
            self.send_response(302)
            self.send_header("Location", "/dashboard.html")
            return self.end_headers()
        if not (path in STATIC_FILES or path.startswith(STATIC_DIRS)):
            return self.send_error(404)
        super().do_GET()

    def do_HEAD(self) -> None:
        self.send_error(405)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_api(self, route: str) -> None:
        from config import TICKERS

        if route == "health":
            return self._send_json(200, {"ok": True})
        if route == "tickers":
            return self._send_json(200, {"tickers": sorted(TICKERS)})
        if route.startswith("predict/"):
            ticker = route.split("/", 1)[1].upper()
            if ticker not in TICKERS:
                return self._send_json(404, {"error": "Ticker is not covered", "tickers": sorted(TICKERS)})
            try:
                import predict_service

                return self._send_json(200, predict_service.predict_next(ticker))
            except Exception as e:
                return self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send_json(404, {"error": "Unknown API route"})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="server.py",
        description="Local server for the dashboard and the live prediction API.",
    )
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")), help="Bind port (default: 8080)")
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the dashboard in the default browser after the server starts.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    loopback_only = args.host in {"127.0.0.1", "localhost", "::1"}
    handler = partial(_Handler, loopback_only=loopback_only, directory=os.path.dirname(os.path.abspath(__file__)))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True

    url = f"http://{args.host}:{args.port}/dashboard.html"
    try:
        with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
            print(f"Dashboard: {url}", flush=True)
            print(f"Prediction API: http://{args.host}:{args.port}/api/predict/<TICKER>", flush=True)
            if args.open:
                webbrowser.open(url, new=2)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped.", flush=True)
        return 0
    except OSError as e:
        print(f"Failed to start server on {args.host}:{args.port}: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
