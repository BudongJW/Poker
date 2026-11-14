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
    * ``seats``: one dict per player at the table, with ``stack`` (chips behind), ``bet`` (chips in
      front on this street), ``in_hand`` (still holds cards), ``hero``, ``button``, ``name``,
      ``sitting_out``, ``acting`` (it is this player's turn) and the place on the picture ``x``,
      ``y`` (0-1, y downwards). With places for everyone the seats are sorted clockwise by them;
      otherwise they must come clockwise.
    * ``hero_to_act``: True when it is the hero's turn, False when it is not or cannot be told.
      Leave it out for pictures taken at the hero's decisions: then it is taken to be the hero's.

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
    if len(heroes) != 1:
        raise ValueError("내 자리를 하나로 정하지 못했습니다 (앞면이 보이는 카드를 가진 자리가 한 곳이어야 합니다)")
    if len(buttons) != 1:
        raise ValueError("딜러 버튼의 위치를 하나로 정하지 못했습니다")
    hero = parse_cards(snapshot.get("hero") or seats[heroes[0]].get("cards") or "")
    if len(hero) != 2:
        raise ValueError("내 카드 두 장을 읽지 못했습니다")
    if len(set(hero + board)) != len(hero) + len(board):
        raise ValueError("같은 카드가 두 번 읽혔습니다")
    seats[heroes[0]]["in_hand"] = True
    # whose turn it is: a picture that says nothing about it was taken at the hero's turn (pictures put in by
    # hand are); one from a running capture says so when it is, marks another seat, or leaves it open
    acting = [i for i, s in enumerate(seats) if s["acting"] and s["in_hand"]]
    if snapshot.get("hero_to_act") or ("hero_to_act" not in snapshot and not acting):
        actor = heroes[0]
    else:
        actor = acting[0] if len(acting) == 1 else None
    return {"hero": hero, "board": board, "street": _STREET_OF[len(board)], "pot": _number(snapshot.get("pot")),
            "pot_includes_bets": bool(snapshot.get("pot_includes_bets", True)),
            "big_blind": _number(snapshot.get("big_blind")), "seats": seats, "hero_seat": heroes[0],
            "button": buttons[0], "actor": actor}


def _aligned(base, shot):
    """The seats of a later snapshot in the seat order of the first one (None where a player is missing).

    The same table shows the same seats in the same places, so with as many players as before
    the order round the table from the hero is taken. Only when a player is missing are places
    (or, failing those, names) compared: names are read less reliably than where a seat is.
    """
    first, later = base["seats"], shot["seats"]
    if len(later) == len(first):
        shift = shot["hero_seat"] - base["hero_seat"]
        return [later[(i + shift) % len(later)] for i in range(len(first))]
    if all(s["x"] is not None for s in first + later):
        out = []
        for seat in first:
            nearest = min(later, key=lambda other: math.hypot(other["x"] - seat["x"], other["y"] - seat["y"]))
            close = math.hypot(nearest["x"] - seat["x"], nearest["y"] - seat["y"]) < 0.08
            out.append(nearest if close else None)
    else:
        by_name = {s["name"]: s for s in later if s["name"]}
        out = [by_name.get(seat["name"]) for seat in first]
    if out[base["hero_seat"]] is None or not out[base["hero_seat"]]["hero"]:
        raise ValueError("이미지마다 내 자리가 다릅니다. 같은 핸드의 이미지를 순서대로 넣어 주세요")
    return out


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
        seat = start % n
        for step in range(8 * n):
            unexplained = [i for i in self.live if need(i) > 1e-9 or i in gone]
            forced = first_acts and step == 0
            if seat not in self.live or (self.stack[seat] <= 1e-9 and need(seat) <= 1e-9):
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
            if complete and not [i for i in self.live if need(i) > 1e-9 or i in gone] and all(
                    i in self.acted or self.stack[i] <= 1e-9 for i in self.live) and all(
                    self.cur[i] >= self.level - 1e-9 or self.stack[i] <= 1e-9 for i in self.live):
                return
            seat = (seat + 1) % n


def _blind_seats(button, n):
    return (button, (button + 1) % n) if n == 2 else ((button + 1) % n, (button + 2) % n)


def _big_blind(shot, sb_seat, bb_seat):
    """The big blind: as the table states it, or as the chips in front before the flop suggest."""
    if shot["big_blind"]:
        return shot["big_blind"], None
    bets = [s["bet"] for i, s in enumerate(shot["seats"]) if s["bet"] > 0 and i != sb_seat]
    if bets:
        return min(bets), "빅 블라인드 금액은 앞에 놓인 칩에서 추정했습니다"
    small = shot["seats"][sb_seat]["bet"]
    if small > 0:
        return 2 * small, "빅 블라인드 금액은 스몰 블라인드의 두 배로 추정했습니다"
    raise ValueError("블라인드 금액을 알 수 없습니다")


def reconstruct(snapshots):
    """Snapshots of one hand, in order -> the hand so far.

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
        big, note = _big_blind(first, sb_seat, bb_seat)
        if note:
            hand["notes"].append(note)
        hand["big_blind"] = hand["big_blind"] or big
        small = frame[sb_seat]["bet"] if 0 < frame[sb_seat]["bet"] < big else big / 2
        posted = hand["posted"] = {sb_seat: small, bb_seat: big}
        target = {i: max(bets[i], posted.get(i, 0.0)) for i in range(n)}
        stack = {i: stacks[i] + target[i] - posted.get(i, 0.0) for i in range(n)}
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
        if hero not in table.live:
            raise ValueError("내가 폴드한 뒤의 이미지가 들어 있습니다")
        start, acted = resume(table, was_to_act)
        if shot["street"] == table.street:
            target = {i: max(bets[i], table.cur[i]) for i in range(n)}
            table.play(target, table.live - live_now, start, first_acts=acted, stop_at=actor)
            continue
        # the street the last picture showed is over: what everybody put in since then went into it
        if unknown_stacks:
            level = max(table.level, table.cur[hero])
            put = {i: (max(0.0, level - table.cur[i]) if i in live_now else 0.0) for i in table.live}
            hand["notes"].append("스택을 읽지 못해, 이미지 사이의 벳은 콜로만 이어진 것으로 보았습니다")
        else:
            put = {i: max(0.0, table.stack[i] - stacks[i] - bets[i]) for i in table.live}
        target = {i: table.cur[i] + put.get(i, 0.0) for i in range(n)}
        final = max(target[i] for i in table.live)
        if target[hero] < final - 1e-9 and stacks[hero] + bets[hero] > 1e-9:
            raise ValueError("이미지 사이에 내 액션이 맞지 않습니다. 같은 핸드의 이미지를 순서대로 넣었는지 확인해 주세요")
        folded = table.live - live_now
        folded_here = {i for i in folded if target[i] < final - 1e-9}       # did not match the last bet of the street
        table.play(target, folded_here, start, first_acts=acted, complete=True)
        pot = table.pot()
        shown = start_of_street(shot, frame)
        if shown is not None and abs(shown - pot) > 0.1 * max(pot, 1e-9) + 1e-9:
            hand["notes"].append(f"{STREET_KO[shot['street']]}의 팟이 앞선 이미지에서 이어 계산한 값과 달라, 화면의 팟을 썼습니다")
            pot = max(shown, 0.0)
        if shot["street"] > table.street + 1:
            hand["notes"].append("이미지가 없는 스트리트의 액션은 반영하지 않았습니다")
        dealt = table.live
        table = _Street(hand, shot["street"], pot, dealt, table.stack)
        table.play(dict(enumerate(bets)), dealt - live_now, first_to_act(shot["street"]), stop_at=actor)

    others = [j for j in table.live if j != hero]
    if not others:
        raise ValueError("남은 상대가 없습니다")
    cover = max(table.cur[j] + table.stack[j] for j in others)
    hand.update(pot=table.pot(), live=sorted(table.live), bets=[table.cur[i] for i in range(n)], actor=actor,
                stacks=[None if table.stack[i] == _INF else table.stack[i] for i in range(n)],
                to_call=min(max(0.0, min(table.level, cover) - table.cur[hero]), table.stack[hero]))
    return hand


# ---- pictures of a hand the engine played (for tests and demonstrations) ---------------

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


def frames(result, stacks, hero_seat, small_blind, every=False, turns=True):
    """A hand the engine played -> the snapshots a camera would have taken at each of the hero's
    decisions (seat places on a circle, so that sorting by place is exercised too).

    every: a snapshot before every action instead, as a running capture sees the table, up to the
    hero's last decision. turns: those say whose turn it is; without, they leave it open.
    """
    n, button, big = result.n_players, result.button, result.big_blind
    mine = [index for index, action in enumerate(result.actions) if action[1] == hero_seat]
    out = []
    for index, action in enumerate(result.actions[:mine[-1] + 1] if mine else []):
        if action[1] != hero_seat and not every:
            continue
        now = chips(result.actions[:index], stacks, button, small_blind, big)
        cards = result.board[:(0, 3, 4, 5)[action[0]]]
        bet = now["bet"] if now["street"] == action[0] else [0] * n         # the first action of a new street
        shot = {
            "hero": "".join(card_str(c) for c in result.hole[hero_seat]), "board": "".join(card_str(c) for c in cards),
            "pot": now["pot"], "big_blind": big,
            "seats": [{"name": f"p{i}", "stack": now["stack"][i], "bet": bet[i], "in_hand": i in now["live"],
                       "hero": i == hero_seat, "button": i == button, "x": 0.5 + 0.4 * math.cos(2 * math.pi * i / n),
                       "y": 0.5 + 0.3 * math.sin(2 * math.pi * i / n)} for i in range(n)]}
        if every:
            shot["hero_to_act"] = turns and action[1] == hero_seat
            shot["seats"][action[1]]["acting"] = turns
        out.append(shot)
    return out


def from_engine(actions, stacks, button, small_blind, big_blind, hero_seat, hole, board, to_call, names=None):
    """A hand the engine is playing, at one of the hero's decisions, in the form :func:`reconstruct`
    returns: nothing has to be worked out, the log is the engine's own."""
    n, now = len(stacks), chips(actions, stacks, button, small_blind, big_blind)
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    return {"n_players": n, "button": button, "hero_seat": hero_seat, "hero": list(hole), "board": list(board),
            "big_blind": big_blind, "actions": list(actions), "notes": [], "absent": [], "posted": now["posted"],
            "names": list(names or [f"p{i}" for i in range(n)]), "pot": now["pot"], "live": sorted(now["live"]),
            "bets": now["bet"] if now["street"] == street else [0] * n, "stacks": now["stack"], "to_call": to_call}


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


def analyze_hand(hand, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None):
    """The hand so far (from :func:`reconstruct` or :func:`from_engine`) -> analysis as
    :func:`pokerlab.advisor.analyze` returns it.

    Every opponent still in the hand gets the range that their actions imply; opponents who have
    not acted yet hold random hands.
    """
    seats = describe(hand)
    found = infer_ranges(load_model(model), hand["actions"], hand["hero_seat"], hand["hero"], hand["board"],
                         hand["n_players"], hand["button"], profile, absent=hand["absent"])
    rivals = [seat for seat in hand["live"] if seat != hand["hero_seat"]]
    ranged = [seat for seat in rivals if seat in found]
    labels = [f"{seats[seat]['position_ko']}: {seats[seat]['actions']}" for seat in ranged]
    unit = hand["big_blind"] or 1.0             # pot odds in big blinds when the big blind is known
    analysis = advisor.analyze(hand["hero"], hand["board"], len(rivals) + 1, hand["pot"] / unit,
                               hand["to_call"] / unit or None, [found[seat] for seat in ranged], trials, seed, precision,
                               labels=labels)
    analysis["unit"] = "bb" if hand["big_blind"] else ""
    for seat, summary in zip(ranged, analysis["ranges"]):
        summary["seat"] = seat
    return analysis


def analyze(snapshots, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None):
    """Snapshots of a hand -> (analysis, the hand as :func:`reconstruct` returns it)."""
    hand = reconstruct(snapshots)
    return analyze_hand(hand, model, profile, trials, seed, precision), hand
