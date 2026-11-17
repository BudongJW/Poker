"""Follow a table as it is played, from a stream of snapshots.

A :class:`Follower` takes the snapshots that one window being watched produces, in the order
they come, and keeps the hand being played up to date: it notices when a new hand starts, puts
the hand so far back together (:func:`pokerlab.table.reconstruct`) and, when a frame does not
fit the ones before it, carries on from that frame alone rather than stop. Each window gets a
follower of its own, so several programs can be followed side by side.

What different programs show is taken as it comes:

* other players' cards that turn up during a hand are a showdown: they are kept, the last round
  of betting is closed, and the chips are not read any more (the pot is on its way to the winner);
  a program that shows everyone's cards all along is simply read, its cards kept as they come;
* a frame in which chips have gone back to a player (the pot was pushed to the winner) ends the
  hand as well, with or without a showdown;
* a hero whose cards stay on the screen after a fold is followed out to the end of the hand, and so
  is a hero whose cards disappear, if the frame still marks the hero's seat;
* a table with no hero (a spectator's view, a replay) is followed hand by hand by its button.

The record of a hand says whether the follower had to start again from a single frame, so that
learning can leave such hands out.
"""

import datetime
import json

from . import advisor, table
from .cards import card_str, parse_cards

MAX_FRAMES = 80             # snapshots kept per hand


class Follower:
    """The table one source shows, hand after hand."""

    def __init__(self, name="", record=None, named=None):
        """named: whether ``name`` says which program this is (a source the visitor named), which
        decides whether its hands may be learned from; by default whenever there is a name."""
        self.name, self.record = name, record
        self.named = bool(name) if named is None else named
        self.shots, self.hand = [], None        # the snapshots of the hand being played and what they add up to
        self.note = self.error = None
        self.version = self.hands = 0           # version: counts every change; hands: hands that have ended
        self._last = self._analysis = None
        # how well the window is being read, without knowing the right answers (see quality())
        self.counts = {"frames": 0, "unread": 0, "restarts": 0, "hands": 0, "restarted_hands": 0, "checked": 0, "mismatched": 0}
        self._clear()

    def _clear(self):
        self.shown = {}                         # seat -> cards seen in this hand (showdown, or a program that shows them all)
        self.restarted = False                  # had to start again from a single frame in this hand
        self.finished = False                   # the betting of this hand is over: no more chips to read
        self.open = False                       # this program shows other players' cards all along

    def push(self, snapshot):
        """Take the next snapshot. Returns True if anything changed."""
        if snapshot == self._last:
            return False
        self._last = snapshot
        self.version += 1
        self._analysis = None
        self.counts["frames"] += 1
        snapshot = self._with_hero_cards(snapshot)
        try:
            shot = table.normalize(snapshot)
        except ValueError as error:             # no hand to read: between hands, or the picture is something else
            try:                                # a seat marked as the hero's with no cards in it, and no hand to take them from:
                if self.hand is not None and self.hand["hero_seat"] is not None:
                    raise ValueError            # (the hero's own hand goes on through _with_hero_cards)
                snapshot = table.without_hero(snapshot)
                shot = table.normalize(snapshot)          # watched from outside
            except ValueError:
                self.error = str(error)
                self.counts["unread"] += 1
                return True
        if self.shots:
            base = table.normalize(self.shots[0])
            if not self._same(table.normalize(self.shots[-1]), shot):
                self.close()
            else:
                turned = self._keep_cards(base, shot)
                if self.finished:
                    return True
                if turned and not self.open:
                    if self.hand is not None and not self.hand["actions"]:
                        self.open = True        # shown before anyone acted: a program that shows them all
                    else:                       # cards turned up: a showdown
                        self._finish()
                        return True
        if not self.shots:
            self.open = bool(table.shown_cards(shot))
            self._keep_cards(shot, shot)
        self.shots = (self.shots + [snapshot])[-MAX_FRAMES:]
        self.error = self.note = None
        try:
            self.hand = table.reconstruct(self.shots)
            if self._betting_over():
                self._finish()
        except ValueError as error:
            if self.hand is not None and len(self.shots) > 1 and self._paid_out(shot):
                self.shots.pop()                # the pot went to the winner: the hand is over
                self._finish()
                return True
            try:                                # this frame does not follow from the earlier ones: start again from it
                self.hand, self.shots = table.reconstruct([snapshot]), [snapshot]
                self.note = f"앞선 화면과 이어지지 않아 지금 화면부터 다시 읽었습니다 ({error})"
                self.restarted, self.shown = True, {}
                self.counts["restarts"] += 1
                self._keep_cards(shot, shot)
            except ValueError as alone:
                self.error, self.shots = str(alone), self.shots[:-1]
        return True

    def _with_hero_cards(self, snapshot):
        """A frame read the way the hand being followed is read.

        Watched from outside: a frame that takes a seat for the hero's (a reader marks the seat at
        the bottom, above all when its cards turn up at a showdown) stays a spectator's frame, those
        cards being that player's. With a hero: a frame that still marks the hero's seat but shows
        no cards there (gone with a fold, or not read) gets them back from the hand."""
        if self.hand is not None and self.hand["hero_seat"] is None:
            seats = snapshot.get("seats") or []
            if not any(seat.get("hero") for seat in seats):
                return snapshot
            return dict(table.without_hero(snapshot), seats=[
                dict(seat, hero=False, cards=seat.get("cards") or snapshot.get("hero") or "") if seat.get("hero")
                else dict(seat) for seat in seats])
        if self.hand is None or self.hand["hero_seat"] is None or snapshot.get("hero"):
            return snapshot
        seats = snapshot.get("seats") or []
        marked = [i for i, seat in enumerate(seats) if seat.get("hero")]
        if len(marked) != 1 or seats[marked[0]].get("cards"):
            return snapshot
        try:                                    # folded if the frame says so; else the cards were there but not read
            before = table.normalize(self.shots[-1])
            filled = dict(snapshot, hero="".join(card_str(c) for c in self.hand["hero"]))
            shot = table.normalize(filled)
        except ValueError:
            return snapshot
        if len(shot["seats"]) != len(before["seats"]) or shot["button"] != before["button"]:
            return snapshot                     # another hand has begun
        return filled

    @staticmethod
    def _same(before, shot):
        """Is ``shot`` a later moment of the hand ``before`` belongs to?"""
        if (before["hero_seat"] is None) != (shot["hero_seat"] is None):
            return False
        if shot["street"] < before["street"] or shot["board"][:len(before["board"])] != before["board"]:
            return False
        if before["hero_seat"] is not None:
            return shot["hero"] == before["hero"]
        if len(shot["seats"]) != len(before["seats"]) or shot["button"] != before["button"]:
            return False                        # watched from outside: the button tells the hands apart
        mine, now = table.shown_cards(before), table.shown_cards(shot)
        return all(now[seat] == cards for seat, cards in mine.items() if seat in now)

    def _keep_cards(self, base, shot):
        """Keep the other players' cards ``shot`` shows, in the seat order of ``base`` -> were any new?"""
        try:
            seats = table._aligned(base, shot)
        except ValueError:
            return False
        found = table.shown_cards(dict(shot, seats=seats))
        new = {seat for seat in found if seat not in self.shown}
        self.shown.update({seat: "".join(card_str(c) for c in cards) for seat, cards in found.items()})
        return bool(new)

    def _paid_out(self, shot):
        """Have chips gone back to a player (stacks only fall while a hand is played) and the pot gone down?"""
        try:
            seats = table._aligned(table.normalize(self.shots[0]), shot)
        except ValueError:
            return False
        grew = any(seat is not None and seat["stack"] is not None and before is not None and seat["stack"] > before + 1e-6
                   for seat, before in zip(seats, self.hand["stacks"]))
        return grew and (shot["pot"] is None or shot["pot"] < self.hand["pot"] - 1e-6)

    def _betting_over(self):
        """Nobody can bet any more: one player left, or everyone but at most one all-in with nothing owed."""
        hand = self.hand
        live, stacks = hand["live"], hand["stacks"]
        if len(live) <= 1:
            return True
        if any(stacks[seat] is None for seat in live):
            return False
        level = max(hand["bets"][seat] for seat in live)
        owed = any(stacks[seat] > 1e-9 and hand["bets"][seat] < level - 1e-9 for seat in live)
        return not owed and sum(stacks[seat] > 1e-9 for seat in live) <= 1

    def _finish(self):
        """The betting is over: whoever had not acted in the last round checked or called."""
        self.finished = True
        try:
            self.hand = table.reconstruct(self.shots, finished=True)
        except ValueError:
            pass

    def close(self):
        """The hand being followed is over: write it to the record, if there is one, and forget it."""
        if self.hand is not None:
            self.hands += 1
            mismatched, checked = table.mismatched_totals(self.shots)
            self.counts.update(hands=self.counts["hands"] + 1, restarted_hands=self.counts["restarted_hands"] + self.restarted,
                               checked=self.counts["checked"] + checked, mismatched=self.counts["mismatched"] + mismatched)
            if self.record:
                line = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "source": self.name, "named": self.named,
                        "hero": "".join(card_str(c) for c in self.hand["hero"]),
                        "board": "".join(card_str(c) for c in self.hand["board"]),
                        **{key: self.hand[key] for key in ("n_players", "button", "hero_seat", "names", "big_blind", "actions",
                                                           "absent", "notes")},
                        "shown": {str(seat): cards for seat, cards in sorted(self.shown.items())},
                        "restarted": self.restarted, "snapshots": self.shots}
                with open(self.record, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.shots, self.hand, self.note = [], None, None
        self._clear()

    def quality(self):
        """How well the window is being read, from the frames themselves (no right answers needed).

        frames: frames taken in; unread: frames from which no table could be read; restarts: frames
        that did not follow from the hand so far (the hand was started again from them); checked /
        mismatched: frames whose chips could be added up, and those that added up to another total
        than the rest of their hand. Within a hand the chips on the table do not change, so a
        mismatched frame had a stack or the pot misread; a misread bet alone does not show here.
        """
        mismatched, checked = table.mismatched_totals(self.shots)        # the hand going on now as well
        out = dict(self.counts, checked=self.counts["checked"] + checked, mismatched=self.counts["mismatched"] + mismatched)
        out["mismatch_rate"] = out["mismatched"] / out["checked"] if out["checked"] else None
        out["restarted_rate"] = out["restarted_hands"] / out["hands"] if out["hands"] else None
        return out

    def state(self):
        """What the source shows now, JSON-ready."""
        out = {"name": self.name, "version": self.version, "hands": self.hands, "frames": len(self.shots),
               "error": self.error, "note": self.note, "phase": "hand" if self.hand else "waiting", "quality": self.quality()}
        if self.hand:
            hand = self.hand
            hero = hand["hero_seat"]
            out.update(hero_to_act=hero is not None and hand["actor"] == hero, observer=hero is None,
                       hero_folded=hero is not None and hero not in hand["live"], finished=self.finished)
            seats = table.describe(hand)
            for seat in seats:
                seat["cards"] = self.shown.get(seat["seat"]) if not seat["hero"] else None
            out["table"] = {
                "source": "capture", "seats": seats, "notes": hand["notes"] + ([self.note] if self.note else []),
                "pot": hand["pot"], "to_call": hand["to_call"], "big_blind": hand["big_blind"], "n_players": hand["n_players"],
                "live": len(hand["live"]), "hero": [card_str(c) for c in hand["hero"]], "observer": hero is None,
                "board": [card_str(c) for c in hand["board"]], "button": hand["button"], "actor": hand["actor"]}
        return out

    def analysis(self, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None):
        """The calculator's view of the hand as it stands, kept until the table changes.

        With a hero still in the hand: the hero's equity against the others (their own cards where
        they can be seen, else the ranges their actions imply). Watched from outside, or after the
        hero's fold: every player's equity. None while there is no hand or nobody to compare.
        """
        if self.hand is None:
            return None
        key = (self.version, model, profile, trials, seed, precision)
        if self._analysis is None or self._analysis[0] != key:
            hand, known = self.hand, {seat: tuple(parse_cards(cards)) for seat, cards in self.shown.items()}
            if hand["hero_seat"] is not None and hand["hero_seat"] in hand["live"] and len(hand["live"]) > 1:
                out = advisor.to_json(table.analyze_hand(hand, model, profile, trials, seed, precision, known=known))
            elif len(hand["live"]) > 1:
                out = table.analyze_table(hand, model, profile, trials, seed, precision, known=known)
            else:
                out = None
            if out is not None:
                out["table"] = self.state()["table"]
            self._analysis = (key, out)
        return self._analysis[1]
