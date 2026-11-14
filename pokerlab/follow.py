"""Follow a table as it is played, from a stream of snapshots.

A :class:`Follower` takes the snapshots that a running capture of one window produces, in the
order they come, and keeps the hand being played up to date: it notices when a new hand starts,
puts the hand so far back together (:func:`pokerlab.table.reconstruct`) and, when a frame does
not fit the ones before it, carries on from that frame alone rather than stop. Each window that
is captured gets a follower of its own, so several programs can be followed side by side.

:func:`watch` feeds a follower from a folder that a capture tool fills with pictures: the way in
for captures that do not come through the web page.
"""

import datetime
import json
import os
import time

from . import advisor, table, vision
from .cards import card_str

MAX_FRAMES = 80             # snapshots kept per hand
PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".json")


class Follower:
    """The table one source shows, hand after hand."""

    def __init__(self, name="", record=None):
        self.name, self.record = name, record
        self.shots, self.hand = [], None        # the snapshots of the hand being played and what they add up to
        self.note = self.error = None
        self.version = self.hands = 0           # version: counts every change; hands: hands that have ended
        self._last = self._analysis = None

    def push(self, snapshot):
        """Take the next snapshot. Returns True if anything changed."""
        if snapshot == self._last:
            return False
        self._last = snapshot
        self.version += 1
        self._analysis = None
        try:
            shot = table.normalize(snapshot)
        except ValueError as error:             # no hand to read: between hands, or the picture is something else
            self.error = str(error)
            return True
        if self.shots:
            before = table.normalize(self.shots[-1])
            same = (shot["hero"] == before["hero"] and shot["street"] >= before["street"]
                    and shot["board"][:len(before["board"])] == before["board"])
            if not same:
                self.close()
        self.shots = (self.shots + [snapshot])[-MAX_FRAMES:]
        self.error = self.note = None
        try:
            self.hand = table.reconstruct(self.shots)
        except ValueError as error:
            try:                                # this frame does not follow from the earlier ones: start again from it
                self.hand, self.shots = table.reconstruct([snapshot]), [snapshot]
                self.note = f"앞선 화면과 이어지지 않아 지금 화면부터 다시 읽었습니다 ({error})"
            except ValueError as alone:
                self.error, self.shots = str(alone), self.shots[:-1]
        return True

    def close(self):
        """The hand being followed is over: write it to the record, if there is one, and forget it."""
        if self.hand is not None:
            self.hands += 1
            if self.record:
                line = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "source": self.name,
                        "hero": "".join(card_str(c) for c in self.hand["hero"]),
                        "board": "".join(card_str(c) for c in self.hand["board"]),
                        **{key: self.hand[key] for key in ("n_players", "button", "hero_seat", "names", "big_blind", "actions",
                                                           "absent", "notes")},
                        "snapshots": self.shots}
                with open(self.record, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.shots, self.hand, self.note = [], None, None

    def state(self):
        """What the source shows now, JSON-ready."""
        out = {"name": self.name, "version": self.version, "hands": self.hands, "frames": len(self.shots),
               "error": self.error, "note": self.note, "phase": "hand" if self.hand else "waiting"}
        if self.hand:
            hand = self.hand
            out["hero_to_act"] = hand["actor"] == hand["hero_seat"]
            out["table"] = {
                "source": "capture", "seats": table.describe(hand), "notes": hand["notes"] + ([self.note] if self.note else []),
                "pot": hand["pot"], "to_call": hand["to_call"], "big_blind": hand["big_blind"], "n_players": hand["n_players"],
                "live": len(hand["live"]), "hero": [card_str(c) for c in hand["hero"]],
                "board": [card_str(c) for c in hand["board"]], "button": hand["button"], "actor": hand["actor"]}
        return out

    def analysis(self, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None):
        """The calculator's view of the hand as it stands (None while there is no hand); kept until the table changes."""
        if self.hand is None:
            return None
        key = (self.version, model, profile, trials, seed, precision)
        if self._analysis is None or self._analysis[0] != key:
            out = advisor.to_json(table.analyze_hand(self.hand, model, profile, trials, seed, precision))
            out["table"] = self.state()["table"]
            self._analysis = (key, out)
        return self._analysis[1]


def watch(directory, follower, on_change, notes=None, poll=0.5, once=False):
    """Read every picture that turns up in ``directory``, in name order, into ``follower``.

    ``on_change(follower)`` is called after each picture that changed what the table shows.
    Pictures are read with :func:`pokerlab.vision.read_file` (``.json`` files hold snapshots that
    were read before). With ``once`` the pictures already there are read and the call returns;
    otherwise it keeps looking until interrupted.
    """
    seen = set()
    while True:
        names = sorted(name for name in os.listdir(directory) if name.lower().endswith(PICTURES) and name not in seen)
        for name in names:
            seen.add(name)
            path = os.path.join(directory, name)
            try:
                found = vision.read_file(path) if name.lower().endswith(".json") else vision.read_file(path, notes=notes)
            except (ValueError, OSError) as error:
                follower.error, changed = f"{name}: {error}", True
                follower.version += 1
            else:
                changed = False
                for snapshot in found if isinstance(found, list) else [found]:
                    changed = follower.push(snapshot) or changed
            if changed:
                on_change(follower)
        if once:
            return
        time.sleep(poll)
