"""Local web page for the equity calculator: ``python -m pokerlab serve``.

Standard library only. ``GET /`` returns the page, ``POST /api/analyze`` takes a spot as JSON
and returns :func:`pokerlab.advisor.to_json` of its analysis.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import advisor
from .cards import parse_cards
from .inference import MODELS
from .preflop import parse_range

PAGE = os.path.join(os.path.dirname(__file__), "static", "index.html")
MAX_TRIALS = 2_000_000
MIN_PRECISION = 0.0005          # +-0.05 percentage points: a few seconds at a full table


def _number(value):
    return None if value in (None, "") else float(value)


def analyze_request(body):
    """Request dict -> JSON-ready analysis. Raises ValueError (or KeyError) on bad input."""
    ranges, lines = [], []
    for opponent in body.get("opponents", []):
        kind, value = opponent.get("type"), str(opponent.get("value", "")).strip()
        if not value:
            continue
        if kind == "line":
            seat = str(opponent.get("position") or "").strip()
            stack = _number(opponent.get("stack"))
            prefix = " ".join(([seat] if seat else []) + ([f"{stack:g}bb"] if stack else []))
            lines.append(f"{prefix}: {value}" if prefix else value)
        elif kind == "range":
            ranges.append(parse_range(value))
        elif kind == "cards":
            cards = parse_cards(value)
            if len(cards) != 2:
                raise ValueError("상대 카드는 두 장이어야 합니다")
            ranges.append([tuple(cards)])
        else:
            raise ValueError(f"unknown opponent type {kind!r}")
    n_players = int(body.get("players", 2))
    if not 2 <= n_players <= 10:
        raise ValueError("인원은 2~10명이어야 합니다")
    if len(ranges) + len(lines) > n_players - 1:
        raise ValueError("상대 설정이 상대 수보다 많습니다")
    precision = _number(body.get("precision"))
    model = str(body.get("model") or advisor.DEFAULT_MODEL)
    if model not in MODELS:                 # names only: a request must not make the server open arbitrary files
        raise ValueError(f"모델은 {', '.join(MODELS)} 중 하나여야 합니다")
    analysis = advisor.analyze(
        body["hand"], body.get("board", ""), n_players, _number(body.get("pot")), _number(body.get("to_call")),
        ranges, trials=min(int(body.get("trials", 200_000)), MAX_TRIALS),
        precision=max(precision, MIN_PRECISION) if precision else None,
        bet=_number(body.get("bet")), lines=lines, profile=str(body.get("profile", "all")), model=model,
        position=str(body.get("position") or "") or None, table=int(body["table"]) if body.get("table") else None)
    return advisor.to_json(analysis)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            with open(PAGE, "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/analyze":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > 100_000:
            self._send(413, b"request too large", "text/plain")
            return
        try:
            result, status = analyze_request(json.loads(self.rfile.read(length))), 200
        except (ValueError, KeyError, TypeError) as error:
            result, status = {"error": str(error) or type(error).__name__}, 400
        self._send(status, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def log_message(self, format, *args):       # keep the terminal quiet
        pass


def serve(host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"pokerlab: http://{host}:{server.server_address[1]}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
