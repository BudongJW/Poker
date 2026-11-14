"""Read a poker table off a picture.

:func:`recognize` sends one image to a vision model and returns a snapshot for
:mod:`pokerlab.table`: who sits where, stacks, the chips in front of each player, who still holds
cards, the dealer button, the board, the pot and the hero's cards. Only what is visible is read.
Positions, table size, who did what, bet sizes, all-ins and stack depths are worked out afterwards
from the snapshots, not asked of the model.

The model is Google's Gemini, called over its Interactions API with the standard library only.
The key is read from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``); ``POKERLAB_VISION_MODEL`` names
another model. The picture leaves this machine: it is sent to Google, with ``store`` off.
"""

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
DEFAULT_MODEL = "gemini-4-hydrogen"
MAX_IMAGE_BYTES = 12_000_000
MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif")

PROMPT = """You are reading a screenshot of an online Texas Hold'em table. Report only what is visible in the picture. \
Never guess a card or a number you cannot read: use null (or leave the list short) instead.

seats: one entry for every seat that has a player (a name plate or avatar), including players who have folded and \
players who are sitting out. Skip empty seats.
- x, y: the centre of the player's name plate as fractions of the image width and height (0 to 1, y measured from the top).
- name: the name on the plate, or null.
- stack: the chips the player has left, as shown on the plate. If the plate says the player is all-in, stack is 0 and \
all_in is true.
- bet: the chips lying in front of the player for the current betting round (posted blinds count). 0 if there are none. \
The pot in the middle is not a bet.
- in_hand: true if the player still holds hole cards (card backs or face-up cards next to the plate); false if the \
player has folded or was not dealt in.
- is_hero: true for the one player whose two hole cards are face up (normally at the bottom of the table).
- has_button: true for the one player who has the dealer button (a disc marked D) next to them.
- sitting_out: true if the plate says the player is sitting out or away.

hero_cards: the two face-up hole cards of the hero. board: the community cards in the middle, left to right \
(0, 3, 4 or 5 cards). Write each card as rank then suit: ranks 2 3 4 5 6 7 8 9 T J Q K A, suits s h d c \
(for example "As", "Td", "9h"). In four-colour decks blue is diamonds and green is clubs.

pot: the number the table shows for the pot (prefer the one labelled total pot). pot_includes_bets: true if that number \
already includes the chips lying in front of the players, false if it only counts chips already gathered in the middle, \
null if you cannot tell.
big_blind: the big blind, from the stakes in the window title or table name if they are visible (for "$0.50/$1.00" \
answer 1.0), else null.

Numbers: plain numbers in the unit the table uses, without currency signs or thousands separators ("1.2K" is 1200). \
If the table shows amounts in big blinds ("98.5 BB"), report those numbers and answer 1 for big_blind.
is_poker_table: false if the picture does not show a hold'em table; then leave everything else empty."""

_NUMBER = {"type": ["number", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "is_poker_table": {"type": "boolean"},
        "hero_cards": {"type": "array", "items": {"type": "string"}},
        "board": {"type": "array", "items": {"type": "string"}},
        "pot": _NUMBER,
        "pot_includes_bets": {"type": ["boolean", "null"]},
        "big_blind": _NUMBER,
        "seats": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]}, "x": {"type": "number"}, "y": {"type": "number"},
                "stack": _NUMBER, "bet": _NUMBER, "in_hand": {"type": "boolean"}, "is_hero": {"type": "boolean"},
                "has_button": {"type": "boolean"}, "sitting_out": {"type": "boolean"}, "all_in": {"type": "boolean"}},
            "required": ["x", "y", "stack", "bet", "in_hand", "is_hero", "has_button", "sitting_out", "all_in"]}},
    },
    "required": ["is_poker_table", "hero_cards", "board", "pot", "seats"],
}


class VisionError(ValueError):
    """The picture could not be read: no key, the service refused, or the answer made no sense."""


def _card(text):
    """'10h', 'ah', 'A♥' -> 'Th', 'Ah', 'Ah' (left as it is when it is not a card: parsing reports it later)."""
    text = str(text).strip().replace("10", "T")
    for symbol, letter in (("♠", "s"), ("♥", "h"), ("♦", "d"), ("♣", "c")):
        text = text.replace(symbol, letter)
    return text[:1].upper() + text[1:2].lower() if len(text) == 2 else text


def to_snapshot(answer):
    """The model's answer -> a snapshot as :func:`pokerlab.table.normalize` expects it."""
    if not isinstance(answer, dict) or not isinstance(answer.get("seats"), list):
        raise VisionError("이미지 인식 결과의 형식이 올바르지 않습니다")
    if answer.get("is_poker_table") is False:
        raise VisionError("홀덤 테이블 이미지가 아닌 것 같습니다")
    seats = []
    for seat in answer["seats"]:
        if not isinstance(seat, dict):
            continue
        stack = seat.get("stack")
        seats.append({"name": seat.get("name"), "x": seat.get("x"), "y": seat.get("y"),
                      "stack": 0.0 if stack is None and seat.get("all_in") else stack, "bet": seat.get("bet") or 0.0,
                      "in_hand": bool(seat.get("in_hand")), "hero": bool(seat.get("is_hero")),
                      "button": bool(seat.get("has_button")), "sitting_out": bool(seat.get("sitting_out"))})
    includes = answer.get("pot_includes_bets")
    return {"hero": "".join(_card(c) for c in answer.get("hero_cards") or []),
            "board": "".join(_card(c) for c in answer.get("board") or []), "pot": answer.get("pot"),
            "pot_includes_bets": True if includes is None else bool(includes), "big_blind": answer.get("big_blind"),
            "seats": seats}


def _output_text(reply):
    """The text the model wrote, from an Interactions API reply."""
    if isinstance(reply.get("output_text"), str):
        return reply["output_text"]
    texts = [part.get("text", "") for step in reply.get("steps") or [] if step.get("type") == "model_output"
             for part in step.get("content") or [] if part.get("type") == "text"]
    if not texts:
        raise VisionError(f"이미지 인식 서비스가 답을 주지 않았습니다 (상태: {reply.get('status', '알 수 없음')})")
    return "".join(texts)


def recognize(image, mime_type="image/png", model=None, key=None, timeout=90):
    """One picture of a table (bytes) -> snapshot dict. Raises :class:`VisionError`."""
    if mime_type not in MIME_TYPES:
        raise VisionError(f"지원하지 않는 이미지 형식입니다 ({mime_type}). PNG, JPEG, WebP를 쓸 수 있습니다")
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise VisionError("이미지가 비어 있거나 너무 큽니다 (12MB 이하)")
    key = key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    body = {
        "model": model or os.environ.get("POKERLAB_VISION_MODEL") or DEFAULT_MODEL,
        "input": [{"type": "text", "text": PROMPT},
                  {"type": "image", "data": base64.b64encode(image).decode(), "mime_type": mime_type}],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA},
        "store": False,
    }
    request = urllib.request.Request(URL, json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "Api-Revision": API_REVISION,
        # without a key of our own a placeholder goes out: a proxy that holds the real one swaps it in
        "x-goog-api-key": key or "missing"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        if error.code in (400, 401, 403) and not key:
            raise VisionError("Gemini API 키가 없습니다. 환경 변수 GEMINI_API_KEY에 키를 넣어 주세요") from None
        if error.code in (401, 403):
            raise VisionError("Gemini API 키가 거부됐습니다. 키와 사용 권한을 확인해 주세요") from None
        if error.code == 429:
            raise VisionError("Gemini API 사용량 한도에 걸렸습니다. 잠시 뒤 다시 시도해 주세요") from None
        raise VisionError(f"이미지 인식 서비스 오류 {error.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise VisionError(f"이미지 인식 서비스에 연결하지 못했습니다: {error}") from None
    try:
        return to_snapshot(json.loads(_output_text(reply)))
    except json.JSONDecodeError:
        raise VisionError("이미지 인식 결과를 읽지 못했습니다 (JSON이 아닙니다)") from None


def read_file(path, **options):
    """A picture file -> snapshot; a ``.json`` file is taken as a snapshot (or a list of them) already read."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return recognize(fh.read(), mime_type, **options)
