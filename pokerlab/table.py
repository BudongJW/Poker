"""From what a poker table shows to what the calculator needs.

A *snapshot* is what can be read off one picture of a table: who sits where, each stack, the
chips in front of each player, who still holds cards, where the dealer button is, the board, the
pot and the hero's own cards. :func:`reconstruct` turns one snapshot, or several of the same hand
in the order they were taken, into the hand so far in the engine's own format: seats, button and
the log of actions. Positions, table size, bet sizes, all-ins and stack depths then follow from
it exactly as they do for a hand the engine played, so nothing has to be entered by hand.

Snapshots are plain dicts (see :func:`normalize`), which is what a recognizer returns
(:mod:`pokerlab.vision`) and what :func:`frames` makes from a hand the engine played.

The reconstruction is exact when there is one snapshot for each of the hero's decisions: between
two of them every other player acts at most once. From a single snapshot it covers the street
being played; what happened on earlier streets is then unknown and left out.
"""

import math

from . import advisor
from .cards import card_str, parse_cards
from .engine import CALL, CHECK, FOLD, RAISE, position_name
from .equity import Range, matchup
from .inference import ALL_IN, NOTHING_BEHIND, bet_class, depth_class, infer_ranges, load_model

_STREET_OF = {0: 0, 3: 1, 4: 2, 5: 3}
STREET_KO = ("프리플랍", "플랍", "턴", "리버")
POSITION_KO = {"BTN": "버튼", "SB": "스몰 블라인드", "BB": "빅 블라인드", "CO": "컷오프", "HJ": "하이잭", "LJ": "로잭"}
_INF = float("inf")


# ---- snapshots -------------------------------------------------------------------------

def _number(value, default=None):
    if value is None or value == "":
        return default
    value = float(value)
    return default if math.isnan(value) else value


def normalize(snapshot):
    """One snapshot, checked and put in order.

    Expected keys (anything missing counts as unknown):

    * ``hero``: the hero's two cards ('AsKd'); ``board``: the community cards
    * ``pot``: the pot the table shows; ``pot_includes_bets`` (default True): whether that number
      already contains the chips in front of the players
    * ``big_blind``: if the table shows its stakes
    * ``bets_whole_hand``: the ``bet`` of each seat is what that player has put in during the whole
      hand, as some programs show it, not only on this street (see :func:`reconstruct`)
    * ``seats``: one dict per player at the table, with ``stack`` (chips behind), ``bet`` (chips in
      front on this street), ``in_hand`` (still holds cards), ``hero``, ``button``, ``name``,
      ``sitting_out``, ``acting`` (it is this player's turn) and the place on the picture ``x``,
      ``y`` (0-1, y downwards). With places for everyone the seats are sorted clockwise by them;
      otherwise they must come clockwise.
    * ``hero_to_act``: True when it is the hero's turn, False when it is not or cannot be told.
      Leave it out for pictures taken at the hero's decisions: then it is taken to be the hero's.
    * ``cards`` on a seat: that player's two cards when they can be seen (a showdown, or a program
      that shows every hand).

    A table with no seat marked as the hero is watched from outside (a spectator's view, a replay):
    then 'hero_seat' is None and 'hero' is empty. The hero's ``in_hand`` is read like anyone's, so a
    program that keeps showing the hero's cards after a fold says so with ``in_hand`` False.

    Returns {'hero': cards, 'board': cards, 'street', 'pot', 'pot_includes_bets', 'big_blind',
    'seats': [...], 'hero_seat', 'button', 'actor' (the seat to act, None if unknown)} with only
    the players who were dealt in.
    """
    board = parse_cards(snapshot.get("board") or "")
    if len(board) not in _STREET_OF:
        raise ValueError(f"보드는 0, 3, 4, 5장이어야 합니다 ({len(board)}장을 읽었습니다)")
    seats = []
    for raw in snapshot.get("seats") or []:
        if raw.get("empty"):
            continue
        seats.append({"name": (raw.get("name") or "").strip(), "stack": _number(raw.get("stack")),
                      "bet": max(_number(raw.get("bet"), 0.0), 0.0), "in_hand": bool(raw.get("in_hand", True)),
                      "hero": bool(raw.get("hero")), "button": bool(raw.get("button")),
                      "away": bool(raw.get("sitting_out")) and not raw.get("hero"), "acting": bool(raw.get("acting")),
                      "x": _number(raw.get("x")), "y": _number(raw.get("y")), "cards": raw.get("cards")})
    if all(s["x"] is not None and s["y"] is not None for s in seats) and seats:
        cx, cy = sum(s["x"] for s in seats) / len(seats), sum(s["y"] for s in seats) / len(seats)
        seats.sort(key=lambda s: math.atan2(s["y"] - cy, s["x"] - cx))        # y grows downwards: this is clockwise
    marked = [i for i, s in enumerate(seats) if s["button"]]
    if len(marked) == 1 and seats[marked[0]]["away"]:       # a dead button: it counts for the player before it
        at = marked[0]
        seats[at]["button"] = False
        for step in range(1, len(seats)):
            if not seats[at - step]["away"]:
                seats[at - step]["button"] = True
                break
    seats = [s for s in seats if not s["away"]]
    if not 2 <= len(seats) <= 10:
        raise ValueError(f"테이블에서 플레이어 {len(seats)}명을 읽었습니다. 2~10명이어야 합니다")
    heroes = [i for i, s in enumerate(seats) if s["hero"]]
    buttons = [i for i, s in enumerate(seats) if s["button"]]
    if len(heroes) > 1:
        raise ValueError("내 자리가 여러 곳으로 읽혔습니다. 소스 메모에 내 자리를 적어 주세요 (예: 내 자리는 아래 가운데)")
    if len(buttons) != 1:
        raise ValueError("딜러 버튼의 위치를 하나로 정하지 못했습니다")
    hero_seat = heroes[0] if heroes else None
    hero = [] if hero_seat is None else parse_cards(snapshot.get("hero") or seats[hero_seat].get("cards") or "")
    if hero_seat is not None and len(hero) != 2:
        raise ValueError("내 카드 두 장을 읽지 못했습니다")
    if len(set(hero + board)) != len(hero) + len(board):
        raise ValueError("같은 카드가 두 번 읽혔습니다")
    # whose turn it is: a picture that says nothing about it was taken at the hero's turn (pictures put in by
    # hand are); one from a running capture says so when it is, marks another seat, or leaves it open
    acting = [i for i, s in enumerate(seats) if s["acting"] and s["in_hand"]]
    if hero_seat is not None and seats[hero_seat]["in_hand"] and (
            snapshot.get("hero_to_act") or ("hero_to_act" not in snapshot and not acting)):
        actor = hero_seat
    else:
        actor = acting[0] if len(acting) == 1 else None
    return {"hero": hero, "board": board, "street": _STREET_OF[len(board)], "pot": _number(snapshot.get("pot")),
            "pot_includes_bets": bool(snapshot.get("pot_includes_bets", True)),
            "bets_whole_hand": bool(snapshot.get("bets_whole_hand")), "big_blind": _number(snapshot.get("big_blind")),
            "seats": seats, "hero_seat": hero_seat, "button": buttons[0], "actor": actor,
            "to_act": bool(snapshot.get("hero_to_act"))}


def without_hero(snapshot):
    """The same snapshot read as a spectator's view: no seat is the hero. For a frame that marks a
    seat as the hero's (a reader takes the seat at the bottom for it) but shows no cards there."""
    return dict(snapshot, hero="", hero_to_act=False,
                seats=[dict(seat, hero=False) for seat in snapshot.get("seats") or []])


def chip_total(shot):
    """All the chips a normalized snapshot shows: the stacks plus the pot with the bets in front.
    Within a hand it does not change, so a frame that adds up to another total was misread
    (a stack, a bet or the pot). None if a number is missing."""
    if shot["pot"] is None or any(seat["stack"] is None for seat in shot["seats"]):
        return None
    total = sum(seat["stack"] for seat in shot["seats"]) + shot["pot"]
    return total if shot["pot_includes_bets"] else total + sum(seat["bet"] for seat in shot["seats"])


def mismatched_totals(snapshots, tolerance=0.005):
    """Snapshots of one hand -> (how many add up to another chip total than most of them, how many could be checked).

    ``tolerance`` is a share of the total, so that amounts a program rounds ('1.2K') are not counted."""
    totals = []
    for snapshot in snapshots:
        try:
            total = chip_total(normalize(snapshot))
        except ValueError:
            continue
        if total is not None:
            totals.append(total)
    if len(totals) < 2:
        return 0, len(totals)
    common = max(totals, key=lambda t: sum(abs(t - other) <= tolerance * max(t, 1.0) for other in totals))
    return sum(abs(t - common) > tolerance * max(common, 1.0) for t in totals), len(totals)


def shown_cards(shot):
    """{seat: (card, card)} of the players other than the hero whose cards a normalized snapshot shows."""
    out = {}
    for seat, found in enumerate(shot["seats"]):
        cards = parse_cards(found["cards"]) if found and not found["hero"] and found.get("cards") else []
        if len(cards) == 2 and not set(cards) & set(shot["hero"] + shot["board"]):
            out[seat] = tuple(cards)
    return out


def _matched(first, later, near):
    """The seats of ``later`` matched one to one with those of ``first`` by place, the closest pairs
    first (None where no seat of ``later`` is nearer than ``near``)."""
    pairs = sorted((math.hypot(a["x"] - b["x"], a["y"] - b["y"]), i, j) for i, a in enumerate(first) for j, b in enumerate(later))
    out, used = [None] * len(first), set()
    for apart, i, j in pairs:
        if apart >= near:
            break
        if out[i] is None and j not in used:
            out[i] = later[j]
            used.add(j)
    return out


def _aligned(base, shot):
    """The seats of a later snapshot in the seat order of the first one (None where a player is missing).

    The same table shows the same seats in the same places, so seats are matched by place. Without
    places, or when a seat is not near where it was, the order round the table from the hero is
    taken with as many players as before (from the button when nobody is the hero); with a player
    missing as well, names are compared (they are read less reliably than where a seat is).
    """
    first, later = base["seats"], shot["seats"]
    if (base["hero_seat"] is None) != (shot["hero_seat"] is None):
        raise ValueError("한 핸드 안에서 내 자리가 있는 화면과 없는 화면이 섞였습니다")
    placed = all(s["x"] is not None and s["y"] is not None for s in first + later)
    if len(later) == len(first):
        out = _matched(first, later, 0.12) if placed else []
        if not out or None in out:              # no places, or a seat that is not where it was: the order round the table
            anchor = "button" if base["hero_seat"] is None else "hero_seat"
            shift = shot[anchor] - base[anchor]
            return [later[(i + shift) % len(later)] for i in range(len(first))]
    elif placed:
        out = _matched(first, later, 0.08)
    else:
        by_name = {s["name"]: s for s in later if s["name"]}
        out = [by_name.get(seat["name"]) for seat in first]
    if base["hero_seat"] is not None and (out[base["hero_seat"]] is None or not out[base["hero_seat"]]["hero"]):
        raise ValueError("이미지마다 내 자리가 다릅니다. 같은 핸드의 이미지를 순서대로 넣어 주세요")
    return out


def counts_whole_hand(shot):
    """Does a normalized snapshot show by each player what they put in during the whole hand?

    The program says so (``bets_whole_hand``), or it shows: after the flop the pot holds what was
    put in before this street as well, so a pot (with the bets) that is no more than the amounts by
    the players means those amounts are the whole hand's."""
    if shot["bets_whole_hand"]:
        return True
    shown = sum(seat["bet"] for seat in shot["seats"])
    return (shot["street"] > 0 and shot["pot_includes_bets"] and shot["pot"] is not None and shown > 0
            and abs(shot["pot"] - shown) <= 0.005 * shown + 1e-9)


def _steady(frames, shots, bets=False):
    """Who is in the hand and what is in front of them, read again with what all the frames say (in place).

    Nobody comes back after a fold: a player read in or out of the hand in one frame only, against
    the frames on either side, was misread there (unless that player faced a bet on their turn just
    before and put nothing in: then the fold is real and the frame after it is the misread one);
    and a player still in the hand in a later frame was in it all along. The player whose turn it
    is (the hero, when the hero's buttons are up) is in the hand.

    bets: also the chips in front (once they are amounts of this betting round). They never go down
    during a round: an amount that later frames of the round contradict was misread (see
    :func:`_rising`), and so is one that :func:`_answered` rules out. And a player whose turn has
    passed, who faced a bet on that turn and put nothing in, folded, whatever a frame says."""
    owed = lambda frame, i: frame[i]["bet"] < max((o["bet"] for j, o in enumerate(frame) if j != i and o is not None), default=0.0) - 1e-9
    same = lambda k: shots[k]["street"] == shots[k - 1]["street"]
    if not bets:
        for k, frame in enumerate(frames):
            for i, seat in enumerate(frame):
                if seat is not None and not seat["in_hand"] and _to_act(shots, k, i, seat):
                    seat["in_hand"] = True
        for k in range(1, len(frames) - 1):
            for i, (a, b, c) in enumerate(zip(frames[k - 1], frames[k], frames[k + 1])):
                if a is None or b is None or c is None or a["in_hand"] != c["in_hand"] or a["in_hand"] == b["in_hand"]:
                    continue
                folded = (a["in_hand"] and _to_act(shots, k - 1, i, a) and same(k) and owed(frames[k - 1], i)
                          and b["bet"] <= a["bet"] + 1e-9)
                if folded:                          # out right after facing a bet on its turn: so the next frame is wrong
                    c["in_hand"] = False
                else:
                    b["in_hand"] = a["in_hand"]
    else:
        start = 0
        for k in range(1, len(frames) + 1):
            if k == len(frames) or shots[k]["street"] != shots[start]["street"]:
                run, bet = [], False                # the round's frames, without those after its bets were swept into the pot
                for frame in frames[start:k]:
                    shown = any(seat is not None and seat["bet"] > 0 for seat in frame)
                    if shown or not bet:
                        run.append(frame)
                    bet = bet or shown
                for i in range(len(frames[0])):
                    _answered(shots, frames, start, k, i)
                    _rising([frame[i] for frame in run])
                start = k
        for k in range(1, len(frames)):
            if not same(k):
                continue
            moved_on = any(seat is not None and _to_act(shots, k, j, seat) for j, seat in enumerate(frames[k]))
            for i, (a, b) in enumerate(zip(frames[k - 1], frames[k])):
                if (a is not None and b is not None and b["in_hand"] and moved_on and _to_act(shots, k - 1, i, a)
                        and not _to_act(shots, k, i, b) and owed(frames[k - 1], i) and owed(frames[k], i) and b["bet"] <= a["bet"] + 1e-9
                        and (b["stack"] is None or b["stack"] > 1e-9)):
                    b["in_hand"] = False            # its turn passed while it faced a bet, and it put nothing in
    for i in range(len(frames[0])):
        last = max((k for k, frame in enumerate(frames) if frame[i] is not None and frame[i]["in_hand"]), default=0)
        for frame in frames[:last]:
            if frame[i] is not None:
                frame[i]["in_hand"] = True


def _to_act(shots, k, i, seat):
    """Is it seat ``i``'s turn in frame ``k``, as the frame shows it (a marked turn, or the hero's buttons up)?"""
    if seat.get("acting"):
        return True
    return shots is not None and shots[k]["to_act"] and seat.get("hero", False)


def _answered(shots, frames, start, stop, i):
    """Seat ``i`` in the frames ``start``-``stop`` of one betting round, against whose turn it is (in place).

    The player whose turn it is has not answered the last bet yet, so cannot be the only one with
    that much in front; and with a stack shown in two frames running, more in front needs that
    much less behind (when the rise is too big for a stack shown rounded to hide). An amount that
    breaks either is a misread: the amount before it is kept."""
    before = 0.0
    for k in range(start, stop):
        seat = frames[k][i]
        if seat is None:
            continue
        others = max((other["bet"] for j, other in enumerate(frames[k]) if j != i and other is not None), default=0.0)
        if seat["bet"] > before + 1e-9:
            turn = _to_act(shots, k, i, seat) and seat["bet"] > others + 1e-9
            last = frames[k - 1][i] if k > start else None
            rise = seat["bet"] - before             # too big for a rounded stack ('2.5K') to hide, and not out of the stack
            behind = (last is not None and last["stack"] is not None and seat["stack"] is not None and seat["stack"] > 0
                      and rise > 0.02 * last["stack"] and abs(last["stack"] - seat["stack"]) < 0.5 * rise)
            if turn or behind:
                seat["bet"] = before
        before = seat["bet"]


def _rising(seats):
    """One player's seat in the frames of one betting round: amounts in front that do not fit the longest
    run that never goes down (the latest one kept on a tie) are taken for misreads and replaced by the
    amount before them (in place)."""
    seen = [(k, seat["bet"]) for k, seat in enumerate(seats) if seat is not None]
    if len(seen) < 3:
        return
    best, before = [], [None] * len(seen)       # longest non-decreasing run, the smallest last amount at each length
    tails = []
    for j, (_, bet) in enumerate(seen):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if seen[tails[mid]][1] <= bet + 1e-9:
                lo = mid + 1
            else:
                hi = mid
        before[j] = tails[lo - 1] if lo else None
        if lo == len(tails):
            tails.append(j)
        else:
            tails[lo] = j
    keep, j = set(), tails[-1]
    while j is not None:
        keep.add(j)
        j = before[j]
    last = None
    for j, (k, bet) in enumerate(seen):
        if j in keep:
            last = bet
        else:
            seats[k]["bet"] = last if last is not None else 0.0


def _street_bets(shots, frames):
    """Amounts for the whole hand -> the bets of each street, in place (the seats of ``frames``).

    What each player put in on earlier streets is carried along: when a new street begins, the
    last street closed at the level that the players still in reached. Those who have not acted
    on the new street yet all show it, and nobody with chips left shows less: with the stacks
    known it is the smallest amount among the players still in with chips left; without them,
    when an all-in cannot be told, the amount most of them show (the smaller one on a tie).
    Amounts below what was bet on that street before are left out. When everyone still in is
    all-in, all of it went in before. A hand first seen after the flop is taken the same way."""
    n = len(frames[0])
    before, street, level = [0.0] * n, None, 0.0
    for shot, frame in zip(shots, frames):
        if street is None or shot["street"] > street:
            amounts = [0.0 if s is None else s["bet"] - before[i] for i, s in enumerate(frame)]
            floor = 0.0 if street is None else level
            players = [(round(amounts[i], 6), s["stack"]) for i, s in enumerate(frame) if s is not None and s["in_hand"]
                       and s["stack"] != 0 and amounts[i] >= floor - 1e-9]
            still = [amount for amount, _ in players]
            if shot["street"] > 0:
                known = all(stack is not None for _, stack in players)      # nobody left here is all-in
                if still:
                    closed = min(still) if known else min(still, key=lambda x: (-still.count(x), x))
                else:                   # everyone still in is all-in: nothing can be bet any more, all went in before
                    closed = max([floor] + [amounts[i] for i, s in enumerate(frame) if s is not None and s["in_hand"]])
                before = [before[i] + max(0.0, min(amounts[i], closed)) for i in range(n)]
            street, level = shot["street"], 0.0
        for i, s in enumerate(frame):
            if s is not None:
                s["bet"] = max(0.0, s["bet"] - before[i])
                level = max(level, s["bet"])


# ---- the hand so far -------------------------------------------------------------------

class _Street:
    """One betting round being replayed: chips in front, stacks and who is still in."""

    def __init__(self, hand, street, pot, live, stack, cur=None, level=0.0):
        self.hand, self.street, self.pot_start = hand, street, pot
        self.live, self.stack = set(live), stack
        self.cur = dict.fromkeys(range(hand["n_players"]), 0.0)
        self.cur.update(cur or {})
        self.level = level
        self.acted = set()                              # who has acted since the last raise

    def pot(self):
        return self.pot_start + sum(self.cur.values())

    def _log(self, seat, action, added):
        others = [j for j in self.live if j != seat]
        cover = max((self.cur[j] + self.stack[j] for j in others), default=0.0)
        owed = min(max(0.0, min(self.level, cover) - self.cur[seat]), self.stack[seat])
        behind = max(0.0, min(self.cur[seat] + self.stack[seat], cover) - max(self.level, self.cur[seat]))
        self.hand["actions"].append((self.street, seat, action, added, self.pot(), owed, len(self.live), None, behind))

    def fold(self, seat):
        self._log(seat, FOLD, 0.0)
        self.live.discard(seat)

    def put(self, seat, to):
        """Check, call or raise so that ``seat`` has ``to`` chips in front."""
        added = max(0.0, min(to - self.cur[seat], self.stack[seat]))
        to = self.cur[seat] + added
        if to > self.level + 1e-9:
            self._log(seat, RAISE, added)
            self.level, self.acted = to, set()
        else:
            self._log(seat, CALL if added > 0 else CHECK, added)
        self.cur[seat], self.stack[seat] = to, self.stack[seat] - added
        self.acted.add(seat)

    def play(self, target, gone, start, first_acts=False, stop_at=None, complete=False):
        """Replay the betting from ``start`` so that everybody ends with ``target`` chips in front and
        the players in ``gone`` have folded.

        Each player in turn does the least that their final chips require: fold when they put in
        nothing more and face a bet, call when they end level, raise when they end higher. A
        raise waits a turn when somebody else must first put in a smaller raise (the player who
        was raised did bet).

        Where the replay ends. With ``complete``: when the round is over. With ``stop_at``: at that
        seat once everything on the table is explained; it is that player's turn, so everybody
        before them has acted, with a check where nothing else shows. With neither: as soon as
        everything is explained; whose turn it is is unknown, so no check is assumed.
        ``first_acts``: the player at ``start`` is known to have acted.
        """
        n = self.hand["n_players"]
        need = lambda i: max(0.0, target[i] - self.cur[i])
        for i in self.live:         # more chips in front than the player had: a stack or a bet was read wrong
            if need(i) > self.stack[i] * 1.01 + 1e-6:
                raise ValueError("이미지 사이에 칩이 맞지 않습니다 (가진 것보다 많은 칩을 냈습니다)")
        can = lambda i: need(i) > 1e-9 and self.stack[i] > 1e-9     # left to put in, and chips to do it with
        seat = start % n
        for step in range(8 * n):
            unexplained = [i for i in self.live if can(i) or i in gone]
            forced = first_acts and step == 0
            if seat not in self.live or (self.stack[seat] <= 1e-9 and seat not in gone):
                pass                                    # folded earlier, or all-in: no turn
            elif not (forced or unexplained or complete) and stop_at in (None, seat):
                return                                  # everything on the table is explained
            else:
                owed = self.level - self.cur[seat]
                if need(seat) > 1e-9:
                    # somebody who still has to put in a smaller amount made that bet before this raise,
                    # unless it is all that player has: then it is a call for less
                    lower_raise_first = any(
                        j != seat and self.level + 1e-9 < target[j] < target[seat] - 1e-9
                        and 1e-9 < need(j) < self.stack[j] - 1e-9 for j in self.live)
                    if target[seat] > self.level + 1e-9 and lower_raise_first:
                        self.put(seat, min(self.level, target[seat]))      # check or call for now, raise next time round
                    else:
                        self.put(seat, target[seat])
                elif seat in gone:
                    will_face_a_bet = any(target[j] > self.level + 1e-9 for j in self.live if j != seat)
                    if owed > 1e-9 or not will_face_a_bet:
                        self.fold(seat)
                    else:
                        self.put(seat, self.cur[seat])  # checks now, folds to the bet that is coming
                elif owed > 1e-9:
                    if not forced:
                        return                          # has not answered the bet yet: this is where the picture was taken
                elif seat in self.acted and not unexplained:
                    return                              # the round is complete
                else:
                    self.put(seat, self.cur[seat])      # a check
            if complete and not [i for i in self.live if can(i) or i in gone] and all(
                    i in self.acted or self.stack[i] <= 1e-9 for i in self.live) and all(
                    self.cur[i] >= self.level - 1e-9 or self.stack[i] <= 1e-9 for i in self.live):
                return
            seat = (seat + 1) % n


def _blind_seats(button, n):
    return (button, (button + 1) % n) if n == 2 else ((button + 1) % n, (button + 2) % n)


def _big_blind(stated, seats, sb_seat, bb_seat):
    """The big blind: as the table states it, or as the chips in front before the flop (``seats``) suggest."""
    if stated:
        return stated, None
    bets = [s["bet"] for i, s in enumerate(seats) if s is not None and s["bet"] > 0 and i != sb_seat]
    if bets:
        return min(bets), "빅 블라인드 금액은 앞에 놓인 칩에서 추정했습니다"
    small = seats[sb_seat]["bet"] if seats[sb_seat] is not None else 0.0
    if small > 0:
        return 2 * small, "빅 블라인드 금액은 스몰 블라인드의 두 배로 추정했습니다"
    raise ValueError("블라인드 금액을 알 수 없습니다")


def reconstruct(snapshots, finished=False, folded=()):
    """Snapshots of one hand, in order -> the hand so far.

    finished: the betting is known to be over (the cards were turned up for a showdown), so the
    players who had not acted yet in the last round checked or called, except those in ``folded``
    (seats in the first snapshot's order that are known to have folded: a later picture showed
    them put nothing more in).

    The snapshots may be pictures taken at the hero's decisions, or the frames of a running
    capture taken at any moment (see ``hero_to_act`` in :func:`normalize`). A frame that shows the
    same chips as the one before adds nothing and is skipped.

    Returns a dict: 'n_players', 'button', 'hero_seat', 'hero', 'board', 'big_blind' (None if never
    seen), 'actions' (the engine's action log, hero's own actions included), 'absent' (seats that
    were out before the first picture's street), 'posted' (blinds by seat, if the hand was seen
    before the flop), 'pot' (everything in it now), 'to_call' (what the hero must add to
    continue), 'stacks' and 'bets' per seat now, 'live' (seats still in), 'actor' (the seat to
    act, None if unknown), 'names', and 'notes' (what had to be assumed).
    """
    if not snapshots:
        raise ValueError("이미지가 없습니다")
    shots = [normalize(s) for s in snapshots]
    base = shots[0]
    n, button, hero = len(base["seats"]), base["button"], base["hero_seat"]
    frames = [base["seats"]] + [_aligned(base, s) for s in shots[1:]]
    if hero is not None:        # the button does not move during a hand: where most frames put it (the first on a tie)
        marks = [next((i for i, s in enumerate(frame) if s is not None and s["button"]), None) for frame in frames]
        marks = [mark for mark in marks if mark is not None]
        button = max(marks, key=lambda mark: (marks.count(mark), -marks.index(mark)))
    _steady(frames, shots)
    if any(counts_whole_hand(shot) for shot in shots):
        _street_bets(shots, frames)
    _steady(frames, shots, bets=True)
    for a, b in zip(shots, shots[1:]):
        if b["street"] < a["street"] or b["board"][:len(a["board"])] != a["board"] or b["hero"] != a["hero"]:
            raise ValueError("이미지들이 같은 핸드의 순서대로가 아닙니다 (내 카드나 보드가 맞지 않습니다)")
    hand = {"n_players": n, "button": button, "hero_seat": hero, "hero": base["hero"], "board": shots[-1]["board"],
            "big_blind": next((s["big_blind"] for s in shots if s["big_blind"]), None), "actions": [], "notes": [],
            "absent": [], "posted": {}, "names": [s["name"] for s in base["seats"]]}
    unknown_stacks = any(s is not None and s["stack"] is None for frame in frames for s in frame)
    if unknown_stacks:
        hand["notes"].append("일부 스택을 읽지 못해 스택 깊이와 올인 여부를 쓰지 않았습니다")

    def read(frame, key, default):
        return [default if s is None or s[key] is None else s[key] for s in frame]

    def start_of_street(shot, frame):
        """The pot before this street's bets, as the picture shows it."""
        if shot["pot"] is None:
            return None
        return shot["pot"] - sum(read(frame, "bet", 0.0)) if shot["pot_includes_bets"] else shot["pot"]

    kept = []                                   # [snapshot, its seats in the first one's order, the seat to act]
    for shot, frame in zip(shots, frames):
        actor = None if shot["actor"] is None else next(
            (i for i, s in enumerate(frame) if s is shot["seats"][shot["actor"]]), None)
        chips = (shot["street"], [None if s is None else (s["bet"], s["in_hand"], s["stack"]) for s in frame])
        if kept and kept[-1][3] == chips:       # nothing moved: at most it is now known whose turn it is
            kept[-1][2] = actor if actor is not None else kept[-1][2]
            continue
        kept.append([shot, frame, actor, chips])

    sb_seat, bb_seat = _blind_seats(button, n)
    first_to_act = lambda street: bb_seat + 1 if street == 0 else button + 1

    def resume(table, was_to_act):
        """Where the replay goes on after a picture: (seat, whether that seat is known to have acted)."""
        if was_to_act is not None and was_to_act in table.live:
            return was_to_act, True
        last = next((a[1] for a in reversed(hand["actions"]) if a[0] == table.street), None)
        return (first_to_act(table.street) if last is None else last + 1), False

    first, frame, actor, _ = kept[0]
    bets, live_now = read(frame, "bet", 0.0), {i for i, s in enumerate(frame) if s["in_hand"]}
    stacks = [_INF if unknown_stacks else x for x in read(frame, "stack", 0.0)]
    if first["street"] == 0:
        # from the first frame that shows the big blind posted (a frame taken while the blinds go in may not yet)
        posting = next((f for s, f, _, _ in kept if s["street"] == 0 and f[bb_seat] is not None and f[bb_seat]["bet"] > 0), frame)
        big, note = _big_blind(hand["big_blind"], posting, sb_seat, bb_seat)
        if note:
            hand["notes"].append(note)
        hand["big_blind"] = hand["big_blind"] or big
        small = frame[sb_seat]["bet"] if 0 < frame[sb_seat]["bet"] < big else big / 2
        posted = hand["posted"] = {sb_seat: small, bb_seat: big}
        target = {i: max(bets[i], posted.get(i, 0.0)) for i in range(n)}
        stack = {i: stacks[i] + bets[i] - posted.get(i, 0.0) for i in range(n)}      # behind once the blinds are in (a frame
        # taken while they go in shows a blind still in the stack)
        table = _Street(hand, 0, 0.0, range(n), stack, posted, big)
        table.play(target, set(range(n)) - live_now, first_to_act(0), stop_at=actor)
    else:
        hand["notes"].append(f"{STREET_KO[first['street']]} 이전의 액션은 이미지에 없어 반영하지 않았습니다")
        pot = start_of_street(first, frame)
        if pot is None:
            pot = 0.0
            hand["notes"].append("팟을 읽지 못해 벳 크기를 쓰지 않았습니다")
        dealt = live_now | {i for i in range(n) if bets[i] > 0}     # folded on this street after putting chips in
        hand["absent"] = sorted(set(range(n)) - dealt)              # out of the hand before the first picture
        table = _Street(hand, first["street"], max(pot, 0.0), dealt, {i: stacks[i] + bets[i] for i in range(n)})
        table.play(dict(enumerate(bets)), dealt - live_now, first_to_act(first["street"]), stop_at=actor)

    for shot, frame, now_to_act, _ in kept[1:]:
        was_to_act, actor = actor, now_to_act
        bets, live_now = read(frame, "bet", 0.0), {i for i, s in enumerate(frame) if s is not None and s["in_hand"]}
        stacks = [_INF if unknown_stacks else x for x in read(frame, "stack", 0.0)]
        start, acted = resume(table, was_to_act)
        if shot["street"] == table.street:
            target = {i: max(bets[i], table.cur[i]) for i in range(n)}
            if not unknown_stacks and shot["pot"] is not None:
                # chips gone from a stack that no bet in front shows, and that the pot shows: drawn in already
                hidden = {i: table.stack[i] - stacks[i] - (target[i] - table.cur[i]) for i in table.live & live_now}
                hidden = {i: chips for i, chips in hidden.items() if chips > 1e-6}
                shown = shot["pot"] + (0.0 if shot["pot_includes_bets"] else sum(bets))
                expected = table.pot() + sum(target[i] - table.cur[i] for i in range(n)) + sum(hidden.values())
                if hidden and abs(shown - expected) <= 0.005 * max(shown, 1.0) + 1e-6:
                    for i, chips in hidden.items():
                        target[i] += chips
            table.play(target, table.live - live_now, start, first_acts=acted, stop_at=actor)
            continue
        # the street the last picture showed is over: what everybody put in since then went into it
        if unknown_stacks:
            level = max(table.level, table.cur[hero] if hero is not None else 0.0)
            put = {i: (max(0.0, level - table.cur[i]) if i in live_now else 0.0) for i in table.live}
            hand["notes"].append("스택을 읽지 못해, 이미지 사이의 벳은 콜로만 이어진 것으로 보았습니다")
        else:
            put = {i: max(0.0, table.stack[i] - stacks[i] - bets[i]) for i in table.live}
        # the last street closed where the players still in with chips left all were; more than that was
        # bet on the new street, though the picture shows it in the pot rather than in front of the player
        reached = [table.cur[i] + put[i] for i in table.live if i in live_now and stacks[i] > 1e-9]
        moved = 0.0
        if reached and not unknown_stacks:
            closed = max(table.level, min(reached))
            for i in table.live:
                extra = table.cur[i] + put[i] - closed
                if extra > 1e-9:
                    put[i], bets[i], moved = put[i] - extra, bets[i] + extra, moved + extra
        target = {i: table.cur[i] + put.get(i, 0.0) for i in range(n)}
        final = max(target[i] for i in table.live)
        if hero in table.live and hero in live_now and target[hero] < final - 1e-9 and stacks[hero] + bets[hero] > 1e-9:
            raise ValueError("이미지 사이에 내 액션이 맞지 않습니다. 같은 핸드의 이미지를 순서대로 넣었는지 확인해 주세요")
        folded_here = {i for i in table.live - live_now if target[i] < final - 1e-9}   # did not match the last bet of the street
        table.play(target, folded_here, start, first_acts=acted, complete=True)
        pot = table.pot()
        shown = start_of_street(shot, frame)
        shown = None if shown is None else shown - moved
        if shown is not None and abs(shown - pot) > 0.1 * max(pot, 1e-9) + 1e-9:
            hand["notes"].append(f"{STREET_KO[shot['street']]}의 팟이 앞선 이미지에서 이어 계산한 값과 달라, 화면의 팟을 썼습니다")
            pot = max(shown, 0.0)
        if shot["street"] > table.street + 1:
            hand["notes"].append("이미지가 없는 스트리트의 액션은 반영하지 않았습니다")
        dealt = table.live
        table = _Street(hand, shot["street"], pot, dealt, table.stack)
        table.play(dict(enumerate(bets)), dealt - live_now, first_to_act(shot["street"]), stop_at=actor)

    if finished and len(table.live) > 1:      # with one player left the hand ended on that fold: nobody else acts
        start, acted = resume(table, actor)     # the others still in matched the last bet, or put in all they had
        out = (table.live - live_now) | (set(folded) & table.live)
        table.play({i: table.cur[i] if i in out else min(table.level, table.cur[i] + table.stack[i]) for i in range(n)},
                   out, start, first_acts=acted, complete=True)
    others = [j for j in table.live if j != hero]
    to_call = 0.0
    if hero in table.live and others:
        cover = max(table.cur[j] + table.stack[j] for j in others)
        to_call = min(max(0.0, min(table.level, cover) - table.cur[hero]), table.stack[hero])
    hand.update(pot=table.pot(), live=sorted(table.live), bets=[table.cur[i] for i in range(n)], actor=actor,
                stacks=[None if table.stack[i] == _INF else table.stack[i] for i in range(n)], to_call=to_call)
    return hand


# ---- frames of a hand the engine played (for tests and replays) ------------------------

def chips(actions, stacks, button, small_blind, big_blind):
    """Where the chips are after ``actions`` of a hand that started with ``stacks``:
    {'stack', 'bet' (in front on the street being played), 'live', 'pot', 'street', 'posted'}."""
    n = len(stacks)
    stack, bet, live = list(stacks), [0] * n, set(range(n))
    posted = {}
    for seat, blind in zip(_blind_seats(button, n), (small_blind, big_blind)):
        posted[seat] = bet[seat] = min(blind, stack[seat])
        stack[seat] -= bet[seat]
    pot, street = sum(bet), 0
    for action in actions:
        if action[0] != street:
            street, bet = action[0], [0] * n
        seat = action[1]
        if action[2] == FOLD:
            live.discard(seat)
        else:
            stack[seat] -= action[3]
            bet[seat] += action[3]
            pot += action[3]
    return {"stack": stack, "bet": bet, "live": live, "pot": pot, "street": street, "posted": posted}


def frames(result, stacks, hero_seat, small_blind, every=False, turns=True, to_end=False, cards="hero", keep_hero=False,
           payout=False):
    """A hand the engine played -> the snapshots a camera would have taken at each of the hero's
    decisions (seat places on a circle, so that sorting by place is exercised too).

    every: a snapshot before every action instead, as a running capture sees the table, up to the
    hero's last decision. turns: those say whose turn it is; without, they leave it open.
    to_end (with every): go on for as long as the hero holds cards, the way a window that hides
    folded cards shows a hand: after the last action one more frame, and at a showdown a last one
    with the cards turned up.
    hero_seat None: the table as a spectator sees it, no seat is the hero's, every action to the end.
    cards: 'hero' (other players' cards only at a showdown) or 'all' (everyone still in the hand
    shows their cards all along, as study and replay programs do).
    keep_hero: the hero's cards stay on the screen after a fold, so the frames go on to the end.
    payout: a last frame with the pot pushed to the winners (stacks after the hand, no bets, no pot).
    """
    n, button, big = result.n_players, result.button, result.big_blind
    observer = hero_seat is None
    every, to_end = every or observer, to_end or observer
    mine = [] if observer else [index for index, action in enumerate(result.actions) if action[1] == hero_seat]
    folds = [index for index in mine if result.actions[index][2] == FOLD]
    to_the_end = observer or keep_hero or not folds
    if to_end:
        last = len(result.actions) + 1 if to_the_end else folds[0] + 1
    else:
        last = mine[-1] + 1 if mine else 0
    hero_cards = "" if observer else "".join(card_str(c) for c in result.hole[hero_seat])
    out = []
    for index in range(min(last, len(result.actions) + (1 if to_end else 0))):
        action = result.actions[index] if index < len(result.actions) else None
        if action is not None and action[1] != hero_seat and not every:
            continue
        now = chips(result.actions[:index], stacks, button, small_blind, big)
        street = action[0] if action else now["street"]
        board = result.board[:(0, 3, 4, 5)[street]]
        bet = now["bet"] if now["street"] == street else [0] * n            # the first action of a new street
        shot = {
            "hero": hero_cards, "board": "".join(card_str(c) for c in board), "pot": now["pot"], "big_blind": big,
            "seats": [{"name": f"p{i}", "stack": now["stack"][i], "bet": bet[i], "in_hand": i in now["live"],
                       "hero": i == hero_seat, "button": i == button, "x": 0.5 + 0.4 * math.cos(2 * math.pi * i / n),
                       "y": 0.5 + 0.3 * math.sin(2 * math.pi * i / n)} for i in range(n)]}
        if cards == "all":
            for i in now["live"]:
                if i != hero_seat:
                    shot["seats"][i]["cards"] = "".join(card_str(c) for c in result.hole[i])
        if every:
            shot["hero_to_act"] = bool(turns and action and action[1] == hero_seat)
            if action:
                shot["seats"][action[1]]["acting"] = turns
        out.append(shot)
    if to_end and to_the_end and result.showdown and out:   # the cards turned up, before the pot goes to the winner
        shown = dict(out[-1], board="".join(card_str(c) for c in result.board), hero_to_act=False,     # bets gathered in
                     seats=[dict(seat, acting=False, bet=0) for seat in out[-1]["seats"]])
        for i, seat in enumerate(shown["seats"]):
            if seat["in_hand"] and not seat["hero"]:
                seat["cards"] = "".join(card_str(c) for c in result.hole[i])
        out.append(shown)
    if payout and to_end and to_the_end and out:
        after = [stacks[i] - result.invested[i] + result.payouts[i] for i in range(n)]
        out.append(dict(out[-1], board="".join(card_str(c) for c in result.board), pot=0, hero_to_act=False,
                        seats=[dict(seat, stack=after[i], bet=0, acting=False) for i, seat in enumerate(out[-1]["seats"])]))
    return out


# ---- what the hand says ----------------------------------------------------------------

def _amount(hand, chips):
    big = hand["big_blind"]
    return f"{chips / big:.1f}bb" if big else f"{chips:g}"


def describe(hand):
    """Every seat in words: position, chips, and what the player did street by street."""
    n, button = hand["n_players"], hand["button"]
    did, front = {}, {(0, seat): blind for seat, blind in hand["posted"].items()}
    for street, seat, action, added, pot, owed, _, _, behind in hand["actions"]:
        front[street, seat] = front.get((street, seat), 0.0) + added
        if action == FOLD:
            word = "폴드"
        elif action == CHECK:
            word = "체크"
        elif action == CALL:
            word = ("올인 콜 " if depth_class(behind, pot, owed) == NOTHING_BEHIND else "콜 ") + _amount(hand, added)
        else:
            share = (added - owed) / (pot + owed) if pot + owed > 0 else None
            size = "올인" if bet_class(added, pot, owed, behind) == ALL_IN else f"팟의 {share:.0%}" if share is not None else ""
            word = f"{'레이즈' if owed > 0 or street == 0 else '벳'} {_amount(hand, front[street, seat])}"
            word += f" ({size})" if size else ""
        did.setdefault(seat, {}).setdefault(street, []).append(word)
    out = []
    for seat in range(n):
        position = position_name(seat, button, n)
        out.append({
            "seat": seat, "name": hand["names"][seat], "position": position,
            "position_ko": POSITION_KO.get(position, position), "hero": seat == hand["hero_seat"],
            "in_hand": seat in hand["live"], "stack": hand["stacks"][seat], "bet": hand["bets"][seat],
            "actions": " / ".join(f"{STREET_KO[street]} {', '.join(words)}" for street, words in sorted(did.get(seat, {}).items())),
        })
    return out


def summary(hand):
    """The reconstructed table as lines of text."""
    seats, big = describe(hand), hand["big_blind"]
    street = STREET_KO[_STREET_OF[len(hand["board"])]]
    lines = [f"[인식된 테이블] {hand['n_players']}명 중 {len(hand['live'])}명 남음 | {street} | 팟 {_amount(hand, hand['pot'])}"
             f" | 콜 금액 {_amount(hand, hand['to_call'])}" + (f" | 빅 블라인드 {big:g}" if big else "")]
    for seat in seats:
        who = "나" if seat["hero"] else (seat["name"] or f"자리 {seat['seat'] + 1}")
        stack = "스택 모름" if seat["stack"] is None else f"스택 {_amount(hand, seat['stack'])}"
        state = "" if seat["in_hand"] else " (폴드)"
        lines.append(f"  {seat['position_ko']:<8} {who}{state} | {stack}" + (f" | {seat['actions']}" if seat["actions"] else ""))
    lines += [f"참고: {note}" for note in hand["notes"]]
    return lines


def analyze_hand(hand, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None, known=None):
    """The hand so far (from :func:`reconstruct`) -> analysis as :func:`pokerlab.advisor.analyze` returns it.

    Every opponent still in the hand gets the range that their actions imply; opponents who have
    not acted yet hold random hands. known: {seat: (card, card)} of opponents whose cards can be
    seen: they hold exactly those.
    """
    seats, known = describe(hand), known or {}
    found = infer_ranges(load_model(model), hand["actions"], hand["hero_seat"], hand["hero"], hand["board"],
                         hand["n_players"], hand["button"], profile, absent=hand["absent"])
    rivals = [seat for seat in hand["live"] if seat != hand["hero_seat"]]
    for seat in rivals:
        if seat in known:
            found[seat] = Range([known[seat]])
    ranged = [seat for seat in rivals if seat in found]
    labels = [f"{seats[seat]['position_ko']}: " + (f"공개된 카드 {''.join(card_str(c) for c in known[seat])}" if seat in known
                                                   else seats[seat]["actions"]) for seat in ranged]
    unit = hand["big_blind"] or 1.0             # pot odds in big blinds when the big blind is known
    analysis = advisor.analyze(hand["hero"], hand["board"], len(rivals) + 1, hand["pot"] / unit,
                               hand["to_call"] / unit or None, [found[seat] for seat in ranged], trials, seed, precision,
                               labels=labels)
    analysis["unit"] = "bb" if hand["big_blind"] else ""
    for seat, summary in zip(ranged, analysis["ranges"]):
        summary["seat"] = seat
    return analysis


def analyze_table(hand, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None, known=None):
    """Every player still in the hand, watched from outside (or after the hero's fold) -> JSON-ready dict.

    Each player holds the cards that can be seen (known: {seat: (card, card)}), else the range
    their actions imply, else a random hand; then everyone's equity against everyone else.
    {'mode': 'table', 'board', 'exact', 'trials', 'players': [{'seat', 'name', 'position_ko',
    'cards', 'equity', 'win', 'tie', 'range' (label, width, top, classes) or None}]}.
    """
    seats, known = describe(hand), known or {}
    dead = list(hand["hero"]) if hand["hero_seat"] is not None else []      # a hero who folded: those cards are out
    found = infer_ranges(load_model(model), hand["actions"], None, dead, hand["board"], hand["n_players"], hand["button"],
                         profile, absent=hand["absent"])
    live = [seat for seat in hand["live"] if seat != hand["hero_seat"]]
    players = [known[seat] if seat in known else found.get(seat) for seat in live]
    results = matchup(players, hand["board"], dead, trials, seed=seed, precision=precision)
    taken = [c for cards in known.values() for c in cards] + list(hand["board"]) + dead
    out = []
    for seat, player, result in zip(live, players, results):
        summary = None
        if isinstance(player, Range):
            summary = dict(advisor.range_summary(player.without(taken)), label=f"{seats[seat]['position_ko']}: {seats[seat]['actions']}")
            summary["top"] = [list(item) for item in summary["top"]]
        out.append({"seat": seat, "name": seats[seat]["name"], "position_ko": seats[seat]["position_ko"],
                    "cards": [card_str(c) for c in known[seat]] if seat in known else None,
                    "equity": result.equity, "win": result.win, "tie": result.tie, "range": summary})
    return {"mode": "table", "board": [card_str(c) for c in hand["board"]], "exact": results[0].exact,
            "trials": results[0].trials, "ci95": results[0].ci95, "players": out}


def analyze(snapshots, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None):
    """Snapshots of a hand -> (analysis, the hand as :func:`reconstruct` returns it)."""
    hand = reconstruct(snapshots)
    return analyze_hand(hand, model, profile, trials, seed, precision), hand
