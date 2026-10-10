#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import webbrowser
from functools import partial
from urllib.parse import urlsplit


class _Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, cors: bool, **kwargs):
        self._cors = cors
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        if self._cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # quiet
        return

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self._handle_api(path[len("/api/"):].strip("/"))
        else:
            super().do_GET()

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
                return self._send_json(404, {"error": f"{ticker} is not covered", "tickers": sorted(TICKERS)})
            try:
                import predict_service

                return self._send_json(200, predict_service.predict_next(ticker))
            except Exception as e:
                return self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
        return self._send_json(404, {"error": f"Unknown API route: /api/{route}"})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="server.py",
        description="Local server for the dashboard and the live prediction API used by the Chrome extension.",
    )
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")), help="Bind port (default: 8080)")
    p.add_argument(
        "--no-cors",
        action="store_true",
        help="Disable CORS header. (CORS is safe here; server is local-only by default.)",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the dashboard in the default browser after the server starts.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    handler = partial(_Handler, cors=not args.no_cors, directory=os.path.dirname(__file__))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    socketserver.ThreadingTCPServer.daemon_threads = True

    url = f"http://{args.host}:{args.port}/dashboard.html"
    try:
        with socketserver.ThreadingTCPServer((args.host, args.port), handler) as httpd:
            print(f"Dashboard: {url}", flush=True)
            print(f"Prediction API: http://{args.host}:{args.port}/api/predict/<TICKER>", flush=True)
            if args.open:
                try:
                    webbrowser.open(url, new=2)
                except Exception:
                    pass
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

