"""Read a poker table off one frame of a program's window.

:func:`recognize` sends one image to a vision model and returns a snapshot for
:mod:`pokerlab.table`: who sits where, stacks, the chips in front of each player, who still holds
cards, the dealer button, the board, the pot and the hero's cards. Only what is visible is read.
Positions, table size, who did what, bet sizes, all-ins and stack depths are worked out afterwards
from the snapshots, not asked of the model.

Two services can do the reading; ``POKERLAB_VISION_PROVIDER`` picks one:

- ``gemini`` (the default): Google's Gemini over its Interactions API, with the standard library
  only. The key is read from ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
- ``claude``: Anthropic's Claude over the Messages API, with the ``anthropic`` package
  (``pip install "pokerlab[claude]"``). The key is read from ``ANTHROPIC_API_KEY``.

``POKERLAB_VISION_MODEL`` names another model of the chosen service, ``POKERLAB_VISION_EFFORT``
how hard it should think (``low`` by default: reading a table needs little, and thinking is most of
the wait). The picture leaves this machine: it is sent to the chosen service (to Google with
``store`` off). :func:`info` says where, for the page to show.
"""

import base64
import json
import os
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
DEFAULT_MODEL = "gemini-4-hydrogen"
MAX_IMAGE_BYTES = 12_000_000
MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif")
DEFAULT_EFFORT = "low"          # Gemini's thinking_level, Claude's effort

CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
CLAUDE_MAX_TOKENS = 16_000      # thinking counts towards it; the answer itself is a few hundred tokens
CLAUDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"
CLAUDE_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2")    # prefixes
SERVICES = {"gemini": "Google Gemini", "claude": "Anthropic Claude"}

PROMPT = """You are reading one frame of a program's window that shows a Texas Hold'em table. Report only what is visible in the picture. \
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
- acting: true for the one player whose turn it is, if the table shows it (a highlighted plate, a timer bar, a \
countdown); false otherwise.

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
hero_to_act: true if the hero's own action buttons (fold, check, call, bet, raise) are on screen and usable, false \
if they are not, null if you cannot tell.
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
        "hero_to_act": {"type": ["boolean", "null"]},
        "seats": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]}, "x": {"type": "number"}, "y": {"type": "number"},
                "stack": _NUMBER, "bet": _NUMBER, "in_hand": {"type": "boolean"}, "is_hero": {"type": "boolean"},
                "has_button": {"type": "boolean"}, "sitting_out": {"type": "boolean"}, "all_in": {"type": "boolean"},
                "acting": {"type": "boolean"}},
            "required": ["x", "y", "stack", "bet", "in_hand", "is_hero", "has_button", "sitting_out", "all_in", "acting"]}},
    },
    "required": ["is_poker_table", "hero_cards", "board", "pot", "hero_to_act", "seats"],
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
                      "button": bool(seat.get("has_button")), "sitting_out": bool(seat.get("sitting_out")),
                      "acting": bool(seat.get("acting"))})
    includes = answer.get("pot_includes_bets")
    return {"hero": "".join(_card(c) for c in answer.get("hero_cards") or []),
            "board": "".join(_card(c) for c in answer.get("board") or []), "pot": answer.get("pot"),
            "pot_includes_bets": True if includes is None else bool(includes), "big_blind": answer.get("big_blind"),
            "hero_to_act": bool(answer.get("hero_to_act")), "seats": seats}


def _output_text(reply):
    """The text the model wrote, from an Interactions API reply."""
    if isinstance(reply.get("output_text"), str):
        return reply["output_text"]
    texts = [part.get("text", "") for step in reply.get("steps") or [] if step.get("type") == "model_output"
             for part in step.get("content") or [] if part.get("type") == "text"]
    if not texts:
        raise VisionError(f"이미지 인식 서비스가 답을 주지 않았습니다 (상태: {reply.get('status', '알 수 없음')})")
    return "".join(texts)


_replays = {}       # file of snapshots that were read before -> how many of them have been handed out


def _replay(path):
    """The next snapshot from a file of snapshots: stands in for the model when trying a capture out without a key."""
    with open(path, encoding="utf-8") as fh:
        found = json.load(fh)
    found = found if isinstance(found, list) else [found]
    index = _replays.get(path, 0)
    if index >= len(found):
        raise VisionError("다시 보여 줄 스냅샷이 더 없습니다")
    _replays[path] = index + 1
    return found[index]


def provider():
    """Which service reads the pictures: 'gemini' or 'claude', from ``POKERLAB_VISION_PROVIDER``."""
    name = (os.environ.get("POKERLAB_VISION_PROVIDER") or "gemini").strip().lower()
    name = {"anthropic": "claude", "google": "gemini"}.get(name, name)
    if name not in SERVICES:
        raise VisionError(f"POKERLAB_VISION_PROVIDER는 {', '.join(SERVICES)} 중 하나여야 합니다 ({name!r})")
    return name


def info():
    """Where a picture goes when it is read, for the page to say so: {'provider', 'service', 'model'}.

    While ``POKERLAB_VISION_REPLAY`` is set nothing is sent: the service is None.
    """
    if os.environ.get("POKERLAB_VISION_REPLAY"):
        return {"provider": "replay", "service": None, "model": None}
    name = provider()
    return {"provider": name, "service": SERVICES[name],
            "model": os.environ.get("POKERLAB_VISION_MODEL") or (CLAUDE_MODEL if name == "claude" else DEFAULT_MODEL)}


def _effort():
    return (os.environ.get("POKERLAB_VISION_EFFORT") or DEFAULT_EFFORT).strip().lower()


def recognize(image, mime_type="image/png", model=None, key=None, timeout=90, notes=None):
    """One picture of a table (bytes) -> snapshot dict. Raises :class:`VisionError`.

    notes: what is particular about the program the picture comes from ("stacks are in chips,
    the hero sits bottom left"), added to the instructions. With ``POKERLAB_VISION_REPLAY`` set to
    a file of snapshots nothing is sent anywhere: the snapshots are handed out one per call.
    """
    if os.environ.get("POKERLAB_VISION_REPLAY"):
        return _replay(os.environ["POKERLAB_VISION_REPLAY"])
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise VisionError("이미지가 비어 있거나 너무 큽니다 (12MB 이하)")
    prompt = PROMPT + (f"\n\nAbout the program in this picture: {notes}" if notes else "")
    model = model or os.environ.get("POKERLAB_VISION_MODEL")
    reader = _claude if provider() == "claude" else _gemini
    try:
        return to_snapshot(json.loads(reader(image, mime_type, prompt, model, key, timeout)))
    except json.JSONDecodeError:
        raise VisionError("이미지 인식 결과를 읽지 못했습니다 (JSON이 아닙니다)") from None


def _gemini(image, mime_type, prompt, model, key, timeout):
    """The model's answer (JSON text) from Gemini's Interactions API."""
    if mime_type not in MIME_TYPES:
        raise VisionError(f"지원하지 않는 이미지 형식입니다 ({mime_type}). PNG, JPEG, WebP를 쓸 수 있습니다")
    key = key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    body = {
        "model": model or DEFAULT_MODEL,
        "input": [{"type": "text", "text": prompt},
                  {"type": "image", "data": base64.b64encode(image).decode(), "mime_type": mime_type}],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA},
        "generation_config": {"thinking_level": _effort()},
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
    return _output_text(reply)


def _strict(schema):
    """SCHEMA as Claude's structured outputs take it: every object closed, "X or null" spelled out with anyOf."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if "properties" in out:
        out["properties"] = {name: _strict(sub) for name, sub in out["properties"].items()}
    if "items" in out:
        out["items"] = _strict(out["items"])
    if isinstance(out.get("type"), list):
        out["anyOf"] = [{"type": kind} for kind in out.pop("type")]
    elif out.get("type") == "object":
        out["additionalProperties"] = False
    return out


def _claude(image, mime_type, prompt, model, key, timeout):
    """The model's answer (JSON text) from Claude's Messages API, through the ``anthropic`` package."""
    try:
        import anthropic
    except ImportError:
        raise VisionError('Claude로 읽으려면 anthropic 패키지가 필요합니다: pip install "pokerlab[claude]"') from None
    if mime_type not in CLAUDE_MIME_TYPES:
        raise VisionError(f"지원하지 않는 이미지 형식입니다 ({mime_type}). Claude는 PNG, JPEG, WebP, GIF를 받습니다")
    model = model or CLAUDE_MODEL
    request = {
        "model": model, "max_tokens": CLAUDE_MAX_TOKENS,
        "output_config": {"effort": _effort(), "format": {"type": "json_schema", "schema": _strict(SCHEMA)}},
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": base64.standard_b64encode(image).decode()}},
            {"type": "text", "text": prompt}]}],
    }
    try:
        client = anthropic.Anthropic(api_key=key, timeout=timeout, max_retries=1) if key else anthropic.Anthropic(timeout=timeout, max_retries=1)
        if model.startswith(CLAUDE_FALLBACK_MODELS):    # if the model declines for policy reasons, the service's own fallback model answers
            message = client.beta.messages.create(betas=[CLAUDE_FALLBACK_BETA], fallbacks="default", **request)
        else:
            message = client.messages.create(**request)
    except anthropic.AuthenticationError:
        raise VisionError("Claude API 키가 없거나 거부됐습니다. 환경 변수 ANTHROPIC_API_KEY를 확인해 주세요") from None
    except anthropic.PermissionDeniedError:
        raise VisionError("이 Claude API 키로는 이 모델을 쓸 수 없습니다. 키의 사용 권한을 확인해 주세요") from None
    except anthropic.NotFoundError:
        raise VisionError(f"Claude 모델을 찾지 못했습니다 ({model}). POKERLAB_VISION_MODEL을 확인해 주세요") from None
    except anthropic.RateLimitError:
        raise VisionError("Claude API 사용량 한도에 걸렸습니다. 잠시 뒤 다시 시도해 주세요") from None
    except anthropic.APIStatusError as error:
        raise VisionError(f"이미지 인식 서비스 오류 {error.status_code}: {str(error.message)[:300]}") from None
    except anthropic.APIConnectionError as error:
        raise VisionError(f"이미지 인식 서비스에 연결하지 못했습니다: {error}") from None
    except anthropic.AnthropicError as error:             # no key at all, among others
        raise VisionError(f"Claude API를 부르지 못했습니다: {error}") from None
    if message.stop_reason == "refusal":
        raise VisionError("Claude가 이 이미지를 읽지 않겠다고 답했습니다")
    if message.stop_reason == "max_tokens":
        raise VisionError("이미지 인식 결과가 중간에 잘렸습니다")
    text = next((block.text for block in message.content if block.type == "text"), None)
    if text is None:
        raise VisionError("이미지 인식 서비스가 답을 주지 않았습니다")
    return text
