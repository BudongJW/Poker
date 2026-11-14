"""Local web page for the equity calculator: ``python -m pokerlab serve``.

Standard library only. ``GET /`` returns the page, ``POST /api/analyze`` takes a spot as JSON
and returns :func:`pokerlab.advisor.to_json` of its analysis. ``POST /api/recognize`` reads one
picture of a table into a snapshot, and ``POST /api/table`` analyses the hand that a list of
snapshots shows. ``POST /api/live`` runs the table at which the visitor plays the bots.
"""

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import advisor, live, table, vision
from .cards import card_str, parse_cards
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


def recognize_request(body):
    """{'image': base64 or a data URL, 'mime_type'} -> {'snapshot': ...}."""
    image, mime_type = str(body.get("image") or ""), str(body.get("mime_type") or "")
    if image.startswith("data:"):                       # data:image/png;base64,....
        header, _, image = image.partition(",")
        mime_type = header[5:].split(";")[0] or mime_type
    try:
        raw = base64.b64decode(image, validate=True)
    except ValueError:
        raise ValueError("이미지 데이터를 읽지 못했습니다") from None
    return {"snapshot": vision.recognize(raw, mime_type or "image/png")}


def table_request(body):
    """{'snapshots': [...], 'model', 'profile', 'precision', 'trials'} -> analysis plus the reconstructed table."""
    snapshots = body.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots or len(snapshots) > 12:
        raise ValueError("스냅샷은 1~12개여야 합니다")
    model = str(body.get("model") or advisor.DEFAULT_MODEL)
    if model not in MODELS:
        raise ValueError(f"모델은 {', '.join(MODELS)} 중 하나여야 합니다")
    precision = _number(body.get("precision"))
    analysis, hand = table.analyze(
        snapshots, model, str(body.get("profile", "all")), trials=min(int(body.get("trials", 200_000)), MAX_TRIALS),
        precision=max(precision, MIN_PRECISION) if precision else None)
    out = advisor.to_json(analysis)
    out["table"] = {
        "seats": table.describe(hand), "notes": hand["notes"], "pot": hand["pot"], "to_call": hand["to_call"],
        "big_blind": hand["big_blind"], "n_players": hand["n_players"], "live": len(hand["live"]),
        "hero": [card_str(c) for c in hand["hero"]], "board": [card_str(c) for c in hand["board"]],
        "lines": table.summary(hand)}
    return out


_LIVE = {"session": None}       # one table: this is a page for one person on their own machine
_LIVE_LOCK = threading.Lock()


def live_request(body):
    """{'do': 'new' | 'deal' | 'act' | 'state' | 'analysis', ...} -> the state of the table the visitor plays at.

    'new' takes 'opponents' (a list of bot names) or 'kind' and 'players', and 'stack_bb'; 'act' takes
    'action' ('fold', 'call', 'raise') and 'amount' (chips in front after a raise); 'analysis' takes
    what /api/table takes and answers {'analysis': ...} for the decision at hand.
    """
    do = body.get("do")
    with _LIVE_LOCK:
        session = _LIVE["session"]
        if do == "new":
            opponents = body.get("opponents") or [str(body.get("kind") or "human")] * (int(body.get("players", 6)) - 1)
            fresh = live.Session([str(kind) for kind in opponents], int(body.get("stack_bb", 100)))
            if session:
                session.close()
            _LIVE["session"] = fresh
            return fresh.deal()
        if session is None:
            return {"phase": "none"}
        if do == "deal":
            return session.deal()
        if do == "act":
            return session.act(str(body.get("action")), body.get("amount"))
        if do == "analysis":
            model = str(body.get("model") or advisor.DEFAULT_MODEL)
            if model not in MODELS:
                raise ValueError(f"모델은 {', '.join(MODELS)} 중 하나여야 합니다")
            precision = _number(body.get("precision"))
            return {"analysis": session.analysis(
                model, str(body.get("profile", "all")), trials=min(int(body.get("trials", 200_000)), MAX_TRIALS),
                precision=max(precision, MIN_PRECISION) if precision else None)}
        if do == "state":
            return session.state()
    raise ValueError(f"unknown request {do!r}")


ROUTES = {      # path -> (handler, largest request body in bytes)
    "/api/analyze": (analyze_request, 100_000),
    "/api/live": (live_request, 10_000),
    "/api/table": (table_request, 1_000_000),
    "/api/recognize": (recognize_request, 17_000_000),
}


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
        if self.path not in ROUTES:
            self._send(404, b"not found", "text/plain")
            return
        handle, limit = ROUTES[self.path]
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            self._send(413, json.dumps({"error": "요청이 너무 큽니다"}).encode(), "application/json; charset=utf-8")
            return
        try:
            result, status = handle(json.loads(self.rfile.read(length))), 200
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
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
