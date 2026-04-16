#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import sys
import webbrowser
from functools import partial


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


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="server.py",
        description="Static server for the offline dashboard (no build step).",
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
    socketserver.TCPServer.allow_reuse_address = True

    url = f"http://{args.host}:{args.port}/dashboard.html"
    try:
        with socketserver.TCPServer((args.host, args.port), handler) as httpd:
            print(f"Dashboard: {url}", flush=True)
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

