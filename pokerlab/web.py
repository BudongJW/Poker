"""Local web page: ``python -m pokerlab serve``.

Standard library only. ``GET /`` returns the page and ``GET /api/info`` says which service reads
the pictures. ``POST /api/capture`` takes the frames of the
windows the visitor has picked as sources, one source per program, and answers with the table
each one shows and the analysis of the hand being played there. ``POST /api/analyze`` takes a
spot typed in by hand and returns :func:`pokerlab.advisor.to_json` of its analysis.
"""

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import advisor, follow, vision
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


_SOURCES = {}       # captured windows by the name the page gives them -> the follower of each
_SOURCES_LOCK = threading.Lock()
MAX_SOURCES = 8
RECORD = None       # a file to which every hand that was followed is appended (set by ``serve``)


def _picture(body):
    """The picture in a request as (bytes, mime type)."""
    image, mime_type = str(body.get("image") or ""), str(body.get("mime_type") or "")
    if image.startswith("data:"):                       # data:image/png;base64,....
        header, _, image = image.partition(",")
        mime_type = header[5:].split(";")[0] or mime_type
    try:
        return base64.b64decode(image, validate=True), mime_type or "image/png"
    except ValueError:
        raise ValueError("이미지 데이터를 읽지 못했습니다") from None


def capture_request(body):
    """One frame of a captured window -> what that source shows now, with the analysis of the hand.

    {'source': a name, 'image': a frame (or 'snapshot': values already read), 'notes': what is
    particular about the program, 'at': where the frame is in a recording, 'model', 'profile',
    'precision', 'trials'}. {'do': 'state'} asks without a frame, 'reset' forgets the hand being
    followed, 'remove' drops the source.
    """
    do, name = body.get("do", "frame"), str(body.get("source") or "1")[:60]
    snapshot = None
    if do == "frame":                                   # reading takes seconds: not while holding the lock
        snapshot = body.get("snapshot")
        if snapshot is None:
            snapshot = vision.recognize(*_picture(body), notes=str(body.get("notes") or "")[:2000] or None)
    with _SOURCES_LOCK:
        if do == "remove":
            follower = _SOURCES.pop(name, None)
            if follower:
                follower.close()
            return {"removed": name}
        if name not in _SOURCES:
            if len(_SOURCES) >= MAX_SOURCES:
                raise ValueError(f"소스는 {MAX_SOURCES}개까지 둘 수 있습니다")
            _SOURCES[name] = follow.Follower(name, RECORD)
        follower = _SOURCES[name]
        if body.get("label"):
            follower.name = str(body["label"])[:60]         # what the visitor calls the program; goes into the record
        if do == "reset":
            follower.close()
        elif do == "frame":
            follower.push(snapshot, at=str(body["at"])[:20] if body.get("at") not in (None, "") else None)
        elif do != "state":
            raise ValueError(f"unknown request {do!r}")
        out = follower.state()
        if follower.hand is not None:
            model = str(body.get("model") or advisor.DEFAULT_MODEL)
            if model not in MODELS:
                raise ValueError(f"모델은 {', '.join(MODELS)} 중 하나여야 합니다")
            precision = _number(body.get("precision"))
            out["analysis"] = follower.analysis(
                model, str(body.get("profile", "all")), trials=min(int(body.get("trials", 200_000)), MAX_TRIALS),
                precision=max(precision, MIN_PRECISION) if precision else None)
        return out


ROUTES = {      # path -> (handler, largest request body in bytes)
    "/api/analyze": (analyze_request, 100_000),
    "/api/capture": (capture_request, 17_000_000),
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
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(PAGE, "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif path == "/api/info":
            try:
                result, status = {"vision": vision.info()}, 200
            except ValueError as error:
                result, status = {"error": str(error)}, 400
            self._send(status, json.dumps(result, ensure_ascii=False).encode(), "application/json; charset=utf-8")
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


def serve(host="127.0.0.1", port=8765, record=None):
    """Run the page. record: a file to which every hand followed in a window is appended."""
    global RECORD
    RECORD = record
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"pokerlab: http://{host}:{server.server_address[1]}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
