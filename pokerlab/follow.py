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
* a table with no hero (a spectator's view, a replay) is followed hand by hand by its button; a
  hand that began that way only because the hero's cards were not dealt yet becomes the hero's
  as soon as they are (the frames before them are kept);
* the hero's seat is where the frames that show the hero's cards put it: a frame that marks
  another seat (a reader may take the seat at the bottom, or a dealer drawn there, for the
  hero's) has the mark moved back, and such frames are counted (see :meth:`Follower.quality`);
* a hero's card read wrong in some frames (half hidden under the other card, say) does not start a
  new hand: a new hand needs the button to move or the pot to go down as well, and the hand
  goes by the cards most of its frames show.

The record of a hand says whether the follower had to start again from a single frame, so that
learning can leave such hands out.
"""

import copy
import datetime
import json

from . import advisor, table
from .cards import card_str, parse_cards

MAX_FRAMES = 80             # snapshots kept per hand
SEAT_NEAR = 0.08            # two readings of one seat's place are no further apart than this (fractions of the picture)
HERO_VOTES = 15             # the latest frames that showed the hero's cards decide where the hero sits


def _place(seat):
    x, y = seat.get("x"), seat.get("y")
    return None if x is None or y is None else (float(x), float(y))


def _apart(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


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
        self.counts = {"frames": 0, "unread": 0, "restarts": 0, "hands": 0, "restarted_hands": 0, "checked": 0, "mismatched": 0,
                       "moved": 0, "passing": 0}
        self._kept = 0                          # frames set aside as passing moments that were taken in after all
        self.hero_places = []                   # where the latest frames that showed the hero's cards put the hero's seat
        # what reading the window has cost: frames read by a service, dollars (None once a call had no known
        # price), seconds, and frames read again by the main model after the cheaper one's reading was doubted
        self.spending = {"frames": 0, "dollars": 0.0, "seconds": 0.0, "read_again": 0, "doubts": {}, "card_checks": 0}
        self._clear()

    def _clear(self):
        self.shown = {}                         # seat -> cards seen in this hand (showdown, or a program that shows them all)
        self.restarted = False                  # had to start again from a single frame in this hand
        self.finished = False                   # the betting of this hand is over: no more chips to read
        self.open = False                       # this program shows other players' cards all along
        self.waiting = []                       # frames as read while the hero's seat had no cards yet (see _dealt)
        self.moved = 0                          # frames of this hand whose hero mark was put back on the hero's seat
        self.held = None                        # a frame set aside as perhaps a passing moment (see push)
        self.card_checks = []                   # the hero's cards as read again on their own in this hand (see card_tie)
        self._checked_at = None                 # how many frames the hand had at the last such reading

    def push(self, snapshot):
        """Take the next snapshot. Returns True if anything changed.

        A frame taken while it is not the hero's turn, whose pot is ahead of what the chips put in
        add up to (a passing moment: a program that moves the pot before the bets), or that does not
        follow from the hand at all (one frame read wrong), is set aside: dropped if the next frame
        of the hand fits without it, otherwise taken in after all, before the next one. A source
        whose set-aside frames mostly turn out to be needed is not held back any more."""
        if snapshot == self._last:
            return False
        self._last = snapshot
        self.counts["frames"] += 1
        if self.held is not None:
            held, self.held = self.held, None
            trial = self._trial(snapshot)
            doubts = self._doubts(trial, snapshot)
            passed = (trial.counts["hands"] == self.counts["hands"] and trial.counts["unread"] == self.counts["unread"]
                      and "pot" not in doubts and "restart" not in doubts)
            if passed:
                self.counts["passing"] += 1
            else:
                self._kept += 1
                self._take(held)
        if self.hand is not None and not snapshot.get("hero_to_act") and not (self._kept >= 3 and self._kept > self.counts["passing"]):
            trial = self._trial(snapshot)
            doubts = self._doubts(trial, snapshot)
            ahead = doubts == ["pot"] and self._pot_shown(snapshot) > trial.hand["pot"]   # (a pot paid out goes down instead)
            if ahead or "restart" in doubts:
                self.held = snapshot            # a passing moment, or one frame read wrong: the next frame tells
                return True
        return self._take(snapshot)

    @staticmethod
    def _pot_shown(snapshot):
        """The pot a frame shows with the bets in front, or None."""
        try:
            shot = table.normalize(snapshot)
        except ValueError:
            return None
        if shot["pot"] is None:
            return None
        return shot["pot"] + (0.0 if shot["pot_includes_bets"] else sum(seat["bet"] for seat in shot["seats"]))

    def _take(self, snapshot):
        """Take a frame in (see :meth:`push`)."""
        self.version += 1
        self._analysis = None
        snapshot = self._place_hero(snapshot)
        read, outside = snapshot, False
        if self.waiting and self._dealt(snapshot):  # the hero's cards have come: the hand is the hero's after all
            self.shots = [dict(frame, hero=snapshot["hero"], hero_copied=True) for frame in self.waiting]
            self.hand, self.waiting = None, []
        snapshot = self._with_hero_cards(snapshot)
        try:
            shot = table.normalize(snapshot)
        except ValueError as error:             # no hand to read: between hands, or the picture is something else
            try:                                # a seat marked as the hero's with no cards in it, and no hand to take them from:
                if self.hand is not None and self.hand["hero_seat"] is not None:
                    raise ValueError            # (the hero's own hand goes on through _with_hero_cards)
                snapshot, outside = table.without_hero(snapshot), True
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
                        self._finish(self._stayed_out(shot, out_of_hand=True))
                        return True
        starts = not self.shots
        if starts:
            self.open = bool(table.shown_cards(shot))
            self._keep_cards(shot, shot)
        self.shots = (self.shots + [snapshot])[-MAX_FRAMES:]
        self.error = self.note = None
        try:
            self.hand = table.reconstruct(self._agreed())
            if starts:                          # read from outside before the flop: perhaps the hero's cards are still to come
                self.waiting = [read] if outside and shot["street"] == 0 and not self.open else []
            elif self.waiting:
                self.waiting = self.waiting + [read] if shot["street"] == 0 else []
            if self._betting_over():
                self._finish()
        except ValueError as error:
            self.waiting = []
            if self.hand is not None and len(self.shots) > 1 and self._paid_out(shot):
                self.shots.pop()                # the pot went to the winner: the hand is over
                self._finish(self._stayed_out(shot))
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

    def spend(self, calls, doubts=(), card_check=False):
        """Count what reading one frame cost: ``calls`` as :func:`pokerlab.vision.recognize` describes
        them (more than one when it was read again), ``doubts`` the reasons it was read again.
        card_check: the call read the hero's cards of a frame already counted again (see card_tie)."""
        spending = self.spending
        spending["card_checks" if card_check else "frames"] += 1
        spending["seconds"] += sum(call.get("seconds") or 0.0 for call in calls)
        costs = [call.get("cost") for call in calls]
        spending["dollars"] = None if spending["dollars"] is None or None in costs else spending["dollars"] + sum(costs)
        if doubts:
            spending["read_again"] += 1
            for why in doubts:
                spending["doubts"][why] = spending["doubts"].get(why, 0) + 1

    def doubts(self, snapshot):
        """What would make one doubt a frame, were it taken in now (nothing is changed): a list of
        'restart' (it does not follow from the hand), 'chips' (its chips add up to another total than
        the hand's), 'seat' (it marks another seat as the hero's), 'cards' (other cards for the hero in
        the same hand), 'pot' (the pot it shows is not the pot that the chips put in add up to). Empty
        when it fits. A frame from which no table can be read is not doubted:
        between hands there is none."""
        return self._doubts(self._trial(snapshot), snapshot)

    def _trial(self, snapshot):
        """A copy of this follower that has taken ``snapshot`` in (this one is not changed)."""
        trial = copy.copy(self)
        trial.record, trial._analysis, trial._last, trial.held = None, None, None, None
        trial.shots, trial.counts, trial.shown = list(self.shots), dict(self.counts), dict(self.shown)
        trial.hero_places, trial.waiting, trial.spending = list(self.hero_places), list(self.waiting), dict(self.spending)
        trial.card_checks = list(self.card_checks)
        trial._take(snapshot)
        return trial

    def _doubts(self, trial, snapshot):
        """:meth:`doubts`, with the copy that took ``snapshot`` in."""
        out = []
        if trial.counts["restarts"] > self.counts["restarts"]:
            out.append("restart")
        same_hand = self.hand is not None and trial.counts["hands"] == self.counts["hands"] and not out
        if same_hand and table.mismatched_totals(trial.shots)[0] > table.mismatched_totals(self.shots)[0]:
            out.append("chips")
        if trial.counts["moved"] > self.counts["moved"]:
            out.append("seat")
        cards = set(parse_cards(snapshot.get("hero") or ""))
        if same_hand and self.hand["hero_seat"] is not None and len(cards) == 2 and cards != set(self.hand["hero"]):
            out.append("cards")
        shown = self._pot_shown(snapshot)
        if same_hand and trial.hand is not None and not trial.finished and shown is not None:
            if abs(shown - trial.hand["pot"]) > 0.005 * max(shown, 1.0) + 1e-6:      # the pot it shows against its chips' sum
                out.append("pot")
        return out

    def _hero_place(self, votes=2):
        """Where the hero sits in this window: the place most of the latest frames that showed the hero's
        cards marked (``votes`` of them at least, and two thirds of them), or None."""
        places = self.hero_places
        if len(places) < votes:
            return None
        best = max(places, key=lambda p: sum(_apart(p, q) <= SEAT_NEAR for q in places))
        group = [q for q in places if _apart(best, q) <= SEAT_NEAR]
        if 3 * len(group) < 2 * len(places):
            return None
        return sum(q[0] for q in group) / len(group), sum(q[1] for q in group) / len(group)

    def _place_hero(self, snapshot):
        """The frame with the hero's mark on the seat where the hero sits in this window.

        A frame whose marked seat shows the hero's cards is a vote for where that is. A frame that
        marks another seat, or two, has the mark moved there; so has one that marks none while the
        hero's hand is being followed. A view from outside (no mark, no hero's hand) stays one."""
        seats = snapshot.get("seats") or []
        places = [_place(seat) for seat in seats]
        if not seats or None in places:
            return snapshot
        marked = [i for i, seat in enumerate(seats) if seat.get("hero")]
        cards = len(parse_cards(snapshot.get("hero") or "")) == 2
        if len(marked) == 1 and cards:
            self.hero_places = (self.hero_places + [places[marked[0]]])[-HERO_VOTES:]
        where = self._hero_place(2 if cards else 1)     # a mark with no cards under it gives way to one frame that had them
        if where is None:
            return snapshot
        near = min(range(len(seats)), key=lambda i: _apart(places[i], where))
        if _apart(places[near], where) > SEAT_NEAR or marked == [near]:
            return snapshot
        if not marked and (self.hand is None or self.hand["hero_seat"] is None):
            return snapshot
        self.counts["moved"] += 1
        self.moved += 1
        return dict(snapshot, seats=[dict(seat, hero=i == near) for i, seat in enumerate(seats)])

    def _dealt(self, snapshot):
        """Does a frame of a hand followed from outside show the hero's cards dealt (still before the
        flop, at the same table with the button in the same place, nobody else's cards shown)?"""
        try:
            shot = table.normalize(snapshot)
            first = table.normalize(dict(self.waiting[0], hero=snapshot.get("hero")))
        except ValueError:
            return False
        if shot["hero_seat"] is None or first["hero_seat"] is None or shot["street"] or table.shown_cards(shot):
            return False
        n = len(shot["seats"])
        return len(first["seats"]) == n and (shot["button"] - shot["hero_seat"]) % n == (first["button"] - first["hero_seat"]) % n

    CHECK_VOTES = 3                             # a reading of the hero's cards on their own counts as this many frames
    MAX_CARD_CHECKS = 2                         # such readings in one hand at most

    def _readings(self):
        """The hero's cards as the hand's frames read them, and as they were read again on their own (weighted)."""
        reads = [frozenset(parse_cards(s["hero"])) for s in self.shots if s.get("hero") and not s.get("hero_copied")]
        return reads + [frozenset(parse_cards(cards)) for cards in self.card_checks if cards for _ in range(self.CHECK_VOTES)]

    def card_tie(self):
        """Do the frames of the hand being followed disagree about the hero's cards (the second reading
        in two frames at least, and in half as many as the first)? Then -> where the hero's seat is
        (x, y), to read the cards again on their own; else None. At most :data:`MAX_CARD_CHECKS` times
        a hand, and not again before another frame has been read."""
        if self.hand is None or self.hand["hero_seat"] is None or len(self.card_checks) >= self.MAX_CARD_CHECKS:
            return None
        reads = self._readings()
        counts = sorted((reads.count(cards) for cards in set(reads)), reverse=True)
        if len(counts) < 2 or counts[1] < 2 or 2 * counts[1] < counts[0] or self._checked_at == len(self.shots):
            return None
        self._checked_at = len(self.shots)
        hero = table.normalize(self.shots[-1])["seats"][table.normalize(self.shots[-1])["hero_seat"]]
        return None if hero["x"] is None or hero["y"] is None else (hero["x"], hero["y"])

    def settle_cards(self, cards):
        """The hero's cards as read again on their own ('' if they could not be): weighed in with the frames."""
        read = bool(cards) and len(parse_cards(cards)) == 2
        self.card_checks.append(cards if read else "")     # an unread one still counts towards the most a hand gets
        if read:
            try:
                self.hand = table.reconstruct(self._agreed(), finished=self.finished)
            except ValueError:
                pass
            self.version += 1
            self._analysis = None

    def _agreed(self):
        """The hand's snapshots, each with the hero's cards that most of them show (the earliest on a tie);
        a reading of the cards on their own counts as :data:`CHECK_VOTES` frames."""
        reads = self._readings()
        if len(set(reads)) <= 1:
            return self.shots
        common = max(reads, key=lambda cards: (reads.count(cards), -reads.index(cards)))
        if not any(s.get("hero") and frozenset(parse_cards(s["hero"])) == common for s in self.shots):
            return [dict(s, hero="".join(card_str(c) for c in sorted(common, reverse=True))) if s.get("hero") else s for s in self.shots]
        cards = next(s["hero"] for s in self.shots if s.get("hero") and frozenset(parse_cards(s["hero"])) == common)
        return [dict(s, hero=cards) if s.get("hero") else s for s in self.shots]

    @staticmethod
    def _same(before, shot):
        """Is ``shot`` a later moment of the hand ``before`` belongs to?"""
        if (before["hero_seat"] is None) != (shot["hero_seat"] is None):
            return False
        if shot["street"] < before["street"] or shot["board"][:len(before["board"])] != before["board"]:
            return False
        n = len(before["seats"])
        if len(shot["seats"]) != n:
            return False
        if before["hero_seat"] is not None:
            if set(shot["hero"]) == set(before["hero"]):
                return True                     # other cards: a new deal only if the button moved or the pot went down
            if (shot["button"] - shot["hero_seat"]) % n != (before["button"] - before["hero_seat"]) % n:
                return False
            middle = [None if s["pot"] is None else s["pot"] + (0 if s["pot_includes_bets"] else sum(x["bet"] for x in s["seats"]))
                      for s in (before, shot)]
            return None not in middle and middle[1] >= middle[0] - 1e-6
        if shot["button"] != before["button"]:
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
        if len(live) <= 1:      # a fold read in one frame only may be a misread: once the frame before agrees, or the pot is paid out
            try:
                base, last = table.normalize(self.shots[0]), table.normalize(self.shots[-1])
                ins = [{i for i, seat in enumerate(table._aligned(base, table.normalize(frame))) if seat is not None and seat["in_hand"]}
                       for frame in self.shots[-2:]]
            except ValueError:
                return True
            return len(ins) < 2 or ins[0] == ins[1] or self._paid_out(last)
        if any(stacks[seat] is None for seat in live):
            return False
        level = max(hand["bets"][seat] for seat in live)
        owed = any(stacks[seat] > 1e-9 and hand["bets"][seat] < level - 1e-9 for seat in live)
        return not owed and sum(stacks[seat] > 1e-9 for seat in live) <= 1

    def _stayed_out(self, shot, out_of_hand=False):
        """The players who owed chips when the betting ended and whose stack ``shot`` (taken after it
        did: the pot pushed to the winner, or cards turned up) shows as it was: they put nothing more
        in, so they folded. out_of_hand: only those that ``shot`` also shows out of the hand (at a
        showdown the callers' stacks may not have been brought up to date yet)."""
        try:
            seats = table._aligned(table.normalize(self.shots[0]), shot)
        except ValueError:
            return set()
        hand, out = self.hand, set()
        level = max((hand["bets"][i] for i in hand["live"]), default=0.0)
        for i in hand["live"]:
            before, now = hand["stacks"][i], seats[i]["stack"] if seats[i] is not None else None
            if hand["bets"][i] < level - 1e-9 and before is not None and now is not None and abs(now - before) <= 0.005 * max(before, 1.0) \
                    and not (out_of_hand and seats[i]["in_hand"]):
                out.add(i)
        return out

    def _finish(self, folded=()):
        """The betting is over: whoever had not acted in the last round checked or called (or folded, if known)."""
        self.finished = True
        try:
            self.hand = table.reconstruct(self._agreed(), finished=True, folded=folded)
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
                        "restarted": self.restarted, "moved": self.moved, "snapshots": self.shots}
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
        passing: frames set aside and dropped: a passing moment (the pot ahead of the bets) or a frame
        that did not follow from the hand, when the next frame fitted without it.
        moved: frames that marked another seat as the hero's than the one where the hero sits (the
        mark was put back); many of them mean the reader needs to be told where the hero sits.
        """
        mismatched, checked = table.mismatched_totals(self.shots)        # the hand going on now as well
        out = dict(self.counts, checked=self.counts["checked"] + checked, mismatched=self.counts["mismatched"] + mismatched)
        out["mismatch_rate"] = out["mismatched"] / out["checked"] if out["checked"] else None
        out["restarted_rate"] = out["restarted_hands"] / out["hands"] if out["hands"] else None
        return out

    def state(self):
        """What the source shows now, JSON-ready."""
        out = {"name": self.name, "version": self.version, "hands": self.hands, "frames": len(self.shots),
               "error": self.error, "note": self.note, "phase": "hand" if self.hand else "waiting", "quality": self.quality(),
               "spending": dict(self.spending, doubts=dict(self.spending["doubts"]))}
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
