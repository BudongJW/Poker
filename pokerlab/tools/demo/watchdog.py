"""What the automatic player does when a game stops going on (a tip, a dialog, a game that ended).

record.py plays a game in a browser while pokerlab reads it. A game can stop on its own: the Flash
game put a "Tip" window over the table and waited for "Close", while the player kept pressing CALL
for almost two minutes (and pokerlab kept reading the covered table). The watchdog notices and acts:

1. Notice: the game's pictures (driver.js saves one about every 0.7 s) have not changed for
   STALL_SECONDS. "Changed" means a step in which at least 3% of a small copy of the picture moved
   clearly; in the Flash video a tip window's little animations stayed under 1.2%, play never went
   20 seconds without a step of 7% or more.
2. Act, cheapest first, looking again after each step:
   a. the fixes known for that game (close a tip and switch tips off, start a new game ...);
   b. Escape, then Enter;
   c. ask Claude what is in the way and where to click (about $0.02), and do that.
3. Give up after MAX_TRIES rounds that did not get the game going: the caller stops playing and
   reading, so that nothing more is spent on a game that does not go on.

Every incident goes to OUT/incidents.json: when, what was tried, what worked, a picture of it.
"""

import base64
import json
import os
import shutil
import time

STALL_SECONDS = 15.0
CHANGED = 0.03              # share of pixels of a 160x100 copy that moved by more than 24 levels
MAX_TRIES = 2               # rounds of the whole ladder for one incident before giving up

ASK = """This is a picture of a poker game being played in a browser by an automatic player. The game has not \
moved for a while. Say whether something is in the way of the game going on: a window, a tip, a message, a menu, \
a question, a game-over or new-game screen, a button waiting to be pressed. Then say the one action that gets the \
game going on: a click (x, y as fractions of the picture width and height, 0 to 1, y from the top) on the button that \
closes it or goes on (prefer "close", "OK", "continue", "start", "play"; if there is a choice such as "hide all tips", \
pick it first), or a key to press (for example "Escape" or "Enter"), or "wait" if nothing is in the way and the \
other players are only slow. Never pick buttons that buy, pay, share, sign in or leave the game."""

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["blocking", "what", "action", "x", "y", "key"],
          "properties": {"blocking": {"type": "boolean"}, "what": {"type": "string"},
                         "action": {"type": "string", "enum": ["click", "key", "wait"]},
                         "x": {"type": "number"}, "y": {"type": "number"}, "key": {"type": "string"}}}


def _small(path):
    from PIL import Image
    return Image.open(path).convert("L").resize((160, 100))


def changed_share(a, b):
    """The share of pixels of two small pictures that differ by more than 24 levels."""
    from PIL import ImageChops, ImageStat
    return ImageStat.Stat(ImageChops.difference(a, b).point(lambda v: 255 if v > 24 else 0)).mean[0] / 255


def ask_claude(picture_path, model="claude-opus-5-5", timeout=60):
    """What Claude sees in the way, and the action it suggests -> dict as SCHEMA, with 'cost'."""
    import anthropic
    options = {}
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("POKER_ANTHROPIC_API_KEY")
    if key:
        options["api_key"] = key
    if os.environ.get("POKERLAB_ANTHROPIC_BASE_URL"):
        options["base_url"] = os.environ["POKERLAB_ANTHROPIC_BASE_URL"]
    client = anthropic.Anthropic(timeout=timeout, max_retries=1, **options)
    with open(picture_path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode()
    reply = client.messages.create(model=model, max_tokens=1024, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
        {"type": "text", "text": ASK}]}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"})
    text = next((block.text for block in reply.content if block.type == "text"), "{}")
    answer = json.loads(text)
    usage = reply.usage
    answer["cost"] = round((usage.input_tokens * 4 + usage.output_tokens * 20 + (usage.cache_read_input_tokens or 0) * 0.2) / 1e6, 4)
    return answer


class Watchdog:
    """Watch the pictures in ``frames_dir`` and get the game going again when it stops.

    act(kind, **what): does one thing in the game: ('click', x=, y=) in pixels of the game picture,
    ('key', key=), or ('fix', name=) for a fix known for the game (see ``fixes``).
    fixes: names of the known fixes, tried in this order. size: the game picture (width, height).
    ask: the function that asks Claude (ask_claude; None: never ask)."""

    def __init__(self, frames_dir, out_dir, act, fixes=(), size=(1280, 800), ask=ask_claude, stall=STALL_SECONDS, settle=3.0):
        self.frames_dir, self.out_dir, self.act, self.fixes, self.size, self.ask = frames_dir, out_dir, act, list(fixes), size, ask
        self.stall, self.settle = stall, settle      # settle: seconds given to the game after each step before looking
        self.incidents, self.gave_up = [], False
        self._last_move = None          # when the picture last changed clearly (from the first picture on)

    def _pictures(self):
        try:
            names = sorted(name for name in os.listdir(self.frames_dir) if name.endswith(".jpg"))
        except FileNotFoundError:
            return []
        return [os.path.join(self.frames_dir, name) for name in names]

    def _moved_since(self, since):
        """Has the game picture changed clearly since ``since`` (a time)? Looks at the pictures saved since then."""
        paths = [p for p in self._pictures() if os.path.getmtime(p) >= since - 1.0]
        for a, b in zip(paths, paths[1:]):
            if changed_share(_small(a), _small(b)) >= CHANGED:
                return True
        return False

    def stalled(self):
        """Has the picture gone STALL_SECONDS without a clear change?"""
        paths = self._pictures()[-40:]
        now = time.time()
        if not paths:
            return False
        if self._last_move is None:     # watching starts with the first picture, not before the game has come up
            self._last_move = os.path.getmtime(paths[0])
        recent = [p for p in paths if os.path.getmtime(p) >= now - self.stall]
        if len(recent) < 5:             # pictures are not coming in (yet): nothing to judge
            return False
        for a, b in zip(recent, recent[1:]):
            if changed_share(_small(a), _small(b)) >= CHANGED:
                self._last_move = max(self._last_move, os.path.getmtime(b))
        return now - self._last_move >= self.stall

    def _try(self, incident, label, kind, **what):
        before = time.time()
        try:
            self.act(kind, **what)
        except Exception as error:      # a fix that does not apply now (no such button): the next one is tried
            incident["tried"].append({"step": label, "error": str(error)[:200]})
            return False
        time.sleep(self.settle)
        moved = self._moved_since(before)
        incident["tried"].append({"step": label, "moved": moved})
        if moved:
            incident["solved_by"] = label
        return moved

    def check(self):
        """Look once -> 'ok' (going on), 'solved' (it had stopped and is going again) or 'stop' (give up)."""
        if self.gave_up:
            return "stop"
        if not self.stalled():
            return "ok"
        pictures = self._pictures()
        incident = {"time": time.strftime("%H:%M:%S"), "still_seconds": round(time.time() - self._last_move, 1), "tried": [],
                    "picture": None}
        if pictures:
            kept = os.path.join(self.out_dir, f"incident_{len(self.incidents) + 1}.jpg")
            shutil.copyfile(pictures[-1], kept)
            incident["picture"] = kept
        self.incidents.append(incident)
        for _ in range(MAX_TRIES):
            for name in self.fixes:
                if self._try(incident, f"알려진 처리: {name}", "fix", name=name):
                    return self._done("solved")
            for key in ("Escape", "Enter"):
                if self._try(incident, f"키 {key}", "key", key=key):
                    return self._done("solved")
            if self.ask is not None and incident["picture"]:
                try:
                    seen = self.ask(incident["picture"])
                except Exception as error:
                    incident["tried"].append({"step": "Claude에게 묻기", "error": str(error)[:200]})
                    seen = None
                if seen:
                    incident.setdefault("claude", []).append(seen)
                    if seen["action"] == "click" and 0 <= seen["x"] <= 1 and 0 <= seen["y"] <= 1:
                        if self._try(incident, f"Claude가 고른 곳 클릭: {seen['what']}", "click",
                                     x=round(seen["x"] * self.size[0]), y=round(seen["y"] * self.size[1])):
                            return self._done("solved")
                    elif seen["action"] == "key" and seen["key"]:
                        if self._try(incident, f"Claude가 고른 키 {seen['key']}: {seen['what']}", "key", key=seen["key"]):
                            return self._done("solved")
                    elif seen["action"] == "wait" and not seen["blocking"]:
                        incident["tried"].append({"step": f"Claude: 막은 것 없음, 기다림 ({seen['what']})"})
                        self._last_move = time.time()
                        return self._done("ok")
            if pictures:                        # a fresh picture for the next round
                pictures = self._pictures()
                shutil.copyfile(pictures[-1], incident["picture"])
        incident["gave_up"] = True
        self.gave_up = True
        return self._done("stop")

    def _done(self, result):
        if result != "stop":
            self._last_move = time.time()
        with open(os.path.join(self.out_dir, "incidents.json"), "w", encoding="utf-8") as fh:
            json.dump(self.incidents, fh, ensure_ascii=False, indent=1)
        return result
