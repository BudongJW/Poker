"""Read a poker table off one frame of a program's window.

:func:`recognize` sends one image to a vision model and returns a snapshot for
:mod:`pokerlab.table`: who sits where, stacks, the chips in front of each player, who still holds
cards, the dealer button, the board, the pot and the hero's cards. Only what is visible is read.
Positions, table size, who did what, bet sizes, all-ins and stack depths are worked out afterwards
from the snapshots, not asked of the model.

Two services can read the picture; :func:`provider` says which one is used.

* Anthropic's Claude, over the Messages API with the official ``anthropic`` package (``pip install
  anthropic``). The key is read from ``ANTHROPIC_API_KEY`` (or ``POKER_ANTHROPIC_API_KEY``). The
  answer is held to :data:`SCHEMA` by the API's structured outputs.
* Google's Gemini, over its Interactions API with the standard library only. The key is read from
  ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).

``POKERLAB_VISION_PROVIDER`` ('claude' or 'gemini') chooses; without it Claude is used when only a
Claude key is there. ``POKERLAB_VISION_MODEL`` names another model of that service. The picture
leaves this machine: it is sent to Anthropic or to Google (with ``store`` off).
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
DEFAULT_MODEL = "gemini-4-hydrogen"
MAX_IMAGE_BYTES = 12_000_000
MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/heic", "image/heif")
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_MIME_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")
CLAUDE_EFFORT = "low"           # reading a picture needs little thinking; POKERLAB_VISION_EFFORT changes it
# models that can hand a declined request on to another model inside the same call (server-side fallbacks)
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5"}
LAST_CALL = {}                  # model, seconds and tokens of the latest call to a service (for measuring)
# dollars per million tokens: input, output, cache read, cache write (5 minutes), as listed when this was written
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5),
          "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}


SAVING_MODEL = "claude-sonnet-5-5"   # the cheaper reader of the 'saving' way (POKERLAB_VISION_SAVING_MODEL changes it)


def saving_model():
    return os.environ.get("POKERLAB_VISION_SAVING_MODEL") or SAVING_MODEL


def cost(call):
    """What a call (as :data:`LAST_CALL` describes it) cost in dollars, about; None for a model without a known price."""
    model = str(call.get("model") or "")
    price = next((value for name, value in PRICES.items() if model.startswith(name)), None)
    if price is None:
        return None
    tokens = (call.get("input_tokens", 0), call.get("output_tokens", 0), call.get("cache_read_tokens", 0),
              call.get("cache_write_tokens", 0))
    return sum(count * each for count, each in zip(tokens, price)) / 1e6

PROMPT = """You are reading one frame of a program's window that shows a Texas Hold'em table. Report only what is visible in the picture. \
Never guess a card or a number you cannot read: use null (or leave the list short) instead.

seats: one entry for every seat that has a player (a name plate or avatar), including players who have folded and \
players who are sitting out. Skip empty seats, and a dealer drawn at the table who deals but does not play.
- x, y: the centre of the player's name plate as fractions of the image width and height (0 to 1, y measured from the top).
- name: the name on the plate, or null.
- stack: the chips the player has left, as shown on the plate; null if the table does not show it as a number. If the \
plate says the player is all-in, stack is 0 and all_in is true.
- bet: the chips lying in front of the player for the current betting round (posted blinds count). 0 if there are none. \
The pot in the middle is not a bet. Some programs show by each player what that player has put in during the whole \
hand instead (the amount stays after a betting round ends, and a player who folded still shows what they paid): then \
report that amount and answer bets_whole_hand true.
- in_hand: true if the player still takes part in the hand (card backs or face-up cards next to the plate); false if \
the player has folded or was not dealt in, even when the folded cards are still shown (greyed, dimmed or marked folded).
- is_hero: true for the one seat the program is played from: normally the seat at the bottom centre, or the one the \
notes about the program name. Keep marking that seat after its player folds. When several players' cards are face up \
(a program that shows every hand, a replay), still mark only that one seat. When the picture is a spectator's view \
with no seat of its own, mark no seat.
- has_button: true for the one player who has the dealer button (a disc marked D) next to them.
- sitting_out: true if the plate says the player is sitting out or away.
- acting: true for the one player whose turn it is, if the table shows it (a highlighted plate, a timer bar, a \
countdown); false otherwise.
- cards: for a player other than the hero whose two hole cards can be seen face up (at a showdown, or in a program \
that shows every hand), those two cards; otherwise an empty list. Never list card backs.

hero_cards: the two hole cards of the hero's seat if they can be seen; an empty list when there is no hero seat or \
its cards are not shown. board: the community cards in the middle, left to right \
(0, 3, 4 or 5 cards). Write each card as rank then suit: ranks 2 3 4 5 6 7 8 9 T J Q K A, suits s h d c \
(for example "As", "Td", "9h"). In four-colour decks blue is diamonds and green is clubs.

pot: the number the table shows for the pot (prefer the one labelled total pot). pot_includes_bets: true if that number \
already includes the chips lying in front of the players, false if it only counts chips already gathered in the middle, \
null if you cannot tell.
bets_whole_hand: true if the amounts by the players count the whole hand (see bet; the notes may say so), false if \
they are for the current betting round, null if you cannot tell.
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
        "bets_whole_hand": {"type": ["boolean", "null"]},
        "big_blind": _NUMBER,
        "hero_to_act": {"type": ["boolean", "null"]},
        "seats": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]}, "x": {"type": "number"}, "y": {"type": "number"},
                "stack": _NUMBER, "bet": _NUMBER, "in_hand": {"type": "boolean"}, "is_hero": {"type": "boolean"},
                "has_button": {"type": "boolean"}, "sitting_out": {"type": "boolean"}, "all_in": {"type": "boolean"},
                "acting": {"type": "boolean"}, "cards": {"type": "array", "items": {"type": "string"}}},
            "required": ["x", "y", "stack", "bet", "in_hand", "is_hero", "has_button", "sitting_out", "all_in", "acting"]}},
    },
    "required": ["is_poker_table", "hero_cards", "board", "pot", "hero_to_act", "seats"],
}


def _strict(node):
    """:data:`SCHEMA` as Claude's structured outputs take it: nullable types as ``anyOf``, and every
    object closed (``additionalProperties: false``) with all of its properties required."""
    if isinstance(node, list):
        return [_strict(item) for item in node]
    if not isinstance(node, dict):
        return node
    node = {key: _strict(value) for key, value in node.items()}
    if isinstance(node.get("type"), list):
        rest = {key: value for key, value in node.items() if key != "type"}
        return {"anyOf": [dict(rest, type=kind) for kind in node["type"]]}
    if node.get("type") == "object":
        node["additionalProperties"] = False
        node["required"] = list(node.get("properties", {}))
    return node


CLAUDE_SCHEMA = _strict(SCHEMA)


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
                      "acting": bool(seat.get("acting")),
                      "cards": "" if seat.get("is_hero") else "".join(_card(c) for c in seat.get("cards") or [])})
    includes = answer.get("pot_includes_bets")
    return {"hero": "".join(_card(c) for c in answer.get("hero_cards") or []),
            "board": "".join(_card(c) for c in answer.get("board") or []), "pot": answer.get("pot"),
            "pot_includes_bets": True if includes is None else bool(includes), "big_blind": answer.get("big_blind"),
            "bets_whole_hand": answer.get("bets_whole_hand") is True, "hero_to_act": bool(answer.get("hero_to_act")),
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


def _claude_key():
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("POKER_ANTHROPIC_API_KEY")


def provider():
    """'claude' or 'gemini': POKERLAB_VISION_PROVIDER if set, else Claude when only a Claude key is there."""
    chosen = (os.environ.get("POKERLAB_VISION_PROVIDER") or "").strip().lower()
    if chosen in ("claude", "anthropic"):
        return "claude"
    if chosen in ("gemini", "google"):
        return "gemini"
    if chosen:
        raise VisionError(f"POKERLAB_VISION_PROVIDER는 claude 또는 gemini여야 합니다 ({chosen})")
    gemini = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return "claude" if _claude_key() and not gemini else "gemini"


def _recognize_claude(image, mime_type, model, key, timeout, notes, call):
    """One picture -> snapshot, read by Claude."""
    try:
        import anthropic
    except ImportError:
        raise VisionError("Claude로 읽으려면 anthropic 패키지가 필요합니다: pip install anthropic") from None
    if mime_type not in CLAUDE_MIME_TYPES:
        raise VisionError(f"Claude가 읽을 수 없는 이미지 형식입니다 ({mime_type}). PNG, JPEG, WebP를 쓸 수 있습니다")
    key = key or _claude_key()
    options = {"api_key": key} if key else {}
    if os.environ.get("POKERLAB_ANTHROPIC_BASE_URL"):
        options["base_url"] = os.environ["POKERLAB_ANTHROPIC_BASE_URL"]
    client = anthropic.Anthropic(timeout=timeout, max_retries=1, **options)
    model = model or os.environ.get("POKERLAB_VISION_MODEL") or CLAUDE_MODEL
    request = {
        "model": model, "max_tokens": 4096,
        # the instructions never change between frames: the same prefix every call, cached where it is long enough
        "system": [{"type": "text", "text": PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": mime_type,
                                         "data": base64.standard_b64encode(image).decode()}},
            {"type": "text", "text": f"About the program in this picture: {notes}" if notes else "Read this table."}]}],
        "output_config": {"format": {"type": "json_schema", "schema": CLAUDE_SCHEMA},
                          "effort": os.environ.get("POKERLAB_VISION_EFFORT") or CLAUDE_EFFORT},
    }
    started = time.time()
    try:
        if model in _FALLBACK_MODELS:       # a declined request is handed on to the model the API picks, in the same call
            reply = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
        else:
            reply = client.messages.create(**request)
    except anthropic.AuthenticationError:
        raise VisionError("Claude API 키가 없거나 거부됐습니다. 환경 변수 ANTHROPIC_API_KEY를 확인해 주세요") from None
    except anthropic.PermissionDeniedError:
        raise VisionError("Claude API 키에 이 모델을 쓸 권한이 없습니다") from None
    except anthropic.RateLimitError:
        raise VisionError("Claude API 사용량 한도에 걸렸습니다. 잠시 뒤 다시 시도해 주세요") from None
    except anthropic.BadRequestError as error:
        raise VisionError(f"Claude API가 요청을 받지 않았습니다: {error.message}"[:300]) from None
    except anthropic.APIStatusError as error:
        raise VisionError(f"Claude API 오류 {error.status_code}") from None
    except anthropic.APIConnectionError as error:
        raise VisionError(f"Claude API에 연결하지 못했습니다: {error}") from None
    usage = reply.usage
    info = dict(service="claude", model=reply.model, seconds=time.time() - started, input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens, cache_read_tokens=usage.cache_read_input_tokens or 0,
                cache_write_tokens=usage.cache_creation_input_tokens or 0)
    info["cost"] = cost(info)
    LAST_CALL.clear()
    LAST_CALL.update(info)
    if call is not None:
        call.update(info)
    if reply.stop_reason == "refusal":
        raise VisionError("Claude가 이 화면을 읽지 않겠다고 했습니다")
    if reply.stop_reason == "max_tokens":
        raise VisionError("인식 결과가 길이 제한에 걸려 잘렸습니다")
    text = next((block.text for block in reply.content if block.type == "text"), "")
    try:
        return to_snapshot(json.loads(text))
    except json.JSONDecodeError:
        raise VisionError("이미지 인식 결과를 읽지 못했습니다 (JSON이 아닙니다)") from None


def recognize(image, mime_type="image/png", model=None, key=None, timeout=90, notes=None, call=None):
    """One picture of a table (bytes) -> snapshot dict. Raises :class:`VisionError`.

    notes: what is particular about the program the picture comes from ("stacks are in chips,
    the hero sits bottom left"), added to the instructions. call: a dict that gets the model,
    seconds, tokens and cost of the call (as :data:`LAST_CALL`, which all threads share). With ``POKERLAB_VISION_REPLAY`` set to
    a file of snapshots nothing is sent anywhere: the snapshots are handed out one per call.
    Which service reads the picture: :func:`provider`.
    """
    if os.environ.get("POKERLAB_VISION_REPLAY"):
        return _replay(os.environ["POKERLAB_VISION_REPLAY"])
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise VisionError("이미지가 비어 있거나 너무 큽니다 (12MB 이하)")
    if provider() == "claude":
        return _recognize_claude(image, mime_type, model, key, timeout, notes, call)
    if mime_type not in MIME_TYPES:
        raise VisionError(f"지원하지 않는 이미지 형식입니다 ({mime_type}). PNG, JPEG, WebP를 쓸 수 있습니다")
    key = key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    body = {
        "model": model or os.environ.get("POKERLAB_VISION_MODEL") or DEFAULT_MODEL,
        "input": [{"type": "text", "text": PROMPT + (f"\n\nAbout the program in this picture: {notes}" if notes else "")},
                  {"type": "image", "data": base64.b64encode(image).decode(), "mime_type": mime_type}],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA},
        "store": False,
    }
    request = urllib.request.Request(URL, json.dumps(body).encode(), method="POST", headers={
        "Content-Type": "application/json", "Api-Revision": API_REVISION,
        # without a key of our own a placeholder goes out: a proxy that holds the real one swaps it in
        "x-goog-api-key": key or "missing"})
    started = time.time()
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
    info = dict(service="gemini", model=body["model"], seconds=time.time() - started, cost=None)   # no price kept here
    LAST_CALL.clear()
    LAST_CALL.update(info)
    if call is not None:
        call.update(info)
    try:
        return to_snapshot(json.loads(_output_text(reply)))
    except json.JSONDecodeError:
        raise VisionError("이미지 인식 결과를 읽지 못했습니다 (JSON이 아닙니다)") from None
