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
      ``sitting_out`` and the place on the picture ``x``, ``y`` (0-1, y downwards). With places for
      everyone the seats are sorted clockwise by them; otherwise they must come clockwise.

    Returns {'hero': cards, 'board': cards, 'street', 'pot', 'pot_includes_bets', 'big_blind',
    'seats': [...], 'hero_seat', 'button'} with only the players who were dealt in.
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
                      "away": bool(raw.get("sitting_out")) and not raw.get("hero"),
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
    return {"hero": hero, "board": board, "street": _STREET_OF[len(board)], "pot": _number(snapshot.get("pot")),
            "pot_includes_bets": bool(snapshot.get("pot_includes_bets", True)),
            "big_blind": _number(snapshot.get("big_blind")), "seats": seats, "hero_seat": heroes[0],
            "button": buttons[0]}


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

    def play(self, target, gone, start, hero_first=False, until_hero=True):
        """Replay the betting from ``start`` so that everybody ends with ``target`` chips in front and
        the players in ``gone`` have folded.

        Each player in turn does the least that their final chips require: fold when they put in
        nothing more and face a bet, call when they end level, raise when they end higher. A
        raise waits a turn when somebody else must first put in a smaller raise (the player who
        was raised did bet). With ``until_hero`` the replay stops where the hero is next to act;
        otherwise it runs until the round is over.
        """
        n, hero = self.hand["n_players"], self.hand["hero_seat"]
        need = lambda i: max(0.0, target[i] - self.cur[i])
        seat, first = start % n, True
        for _ in range(8 * n):
            unexplained = [i for i in self.live if need(i) > 1e-9 or i in gone]
            if seat not in self.live or (self.stack[seat] <= 1e-9 and need(seat) <= 1e-9):
                pass                                    # folded earlier, or all-in: no turn
            elif seat == hero and until_hero and not (first and hero_first) and need(seat) <= 1e-9 and not unexplained:
                return                                  # everything on the table is explained: the hero is to act
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
                    return                              # has not answered the bet yet: this is where the picture was taken
                elif seat in self.acted and not unexplained:
                    return                              # the round is complete
                else:
                    self.put(seat, self.cur[seat])      # a check
                first = False
            if not until_hero and not unexplained and all(
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

    Returns a dict: 'n_players', 'button', 'hero_seat', 'hero', 'board', 'big_blind' (None if never
    seen), 'actions' (the engine's action log, hero's own actions included), 'absent' (seats that
    were out before the first picture's street), 'posted' (blinds by seat, if the hand was seen
    before the flop), 'pot' (everything in it now), 'to_call' (what the
    hero must add to continue), 'stacks' and 'bets' per seat now, 'live' (seats still in),
    'names', and 'notes' (what had to be assumed).
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

    first, frame = base, frames[0]
    bets, live_now = read(frame, "bet", 0.0), {i for i, s in enumerate(frame) if s["in_hand"]}
    stacks = [_INF if unknown_stacks else x for x in read(frame, "stack", 0.0)]
    after_button = (button + 1) % n
    if first["street"] == 0:
        sb_seat, bb_seat = _blind_seats(button, n)
        big, note = _big_blind(first, sb_seat, bb_seat)
        if note:
            hand["notes"].append(note)
        hand["big_blind"] = hand["big_blind"] or big
        small = frame[sb_seat]["bet"] if 0 < frame[sb_seat]["bet"] < big else big / 2
        posted = hand["posted"] = {sb_seat: small, bb_seat: big}
        target = {i: max(bets[i], posted.get(i, 0.0)) for i in range(n)}
        stack = {i: stacks[i] + target[i] - posted.get(i, 0.0) for i in range(n)}
        table = _Street(hand, 0, 0.0, range(n), stack, posted, big)
        table.play(target, set(range(n)) - live_now, bb_seat + 1)
    else:
        hand["notes"].append(f"{STREET_KO[first['street']]} 이전의 액션은 이미지에 없어 반영하지 않았습니다")
        pot = start_of_street(first, frame)
        if pot is None:
            pot = 0.0
            hand["notes"].append("팟을 읽지 못해 벳 크기를 쓰지 않았습니다")
        dealt = live_now | {i for i in range(n) if bets[i] > 0}     # folded on this street after putting chips in
        hand["absent"] = sorted(set(range(n)) - dealt)              # out of the hand before the first picture
        table = _Street(hand, first["street"], max(pot, 0.0), dealt, {i: stacks[i] + bets[i] for i in range(n)})
        table.play(dict(enumerate(bets)), dealt - live_now, after_button)

    for shot, frame in zip(shots[1:], frames[1:]):
        bets, live_now = read(frame, "bet", 0.0), {i for i, s in enumerate(frame) if s is not None and s["in_hand"]}
        stacks = [_INF if unknown_stacks else x for x in read(frame, "stack", 0.0)]
        if hero not in table.live:
            raise ValueError("내가 폴드한 뒤의 이미지가 들어 있습니다")
        if shot["street"] == table.street:
            target = {i: max(bets[i], table.cur[i]) for i in range(n)}
            if target[hero] <= table.cur[hero] + 1e-9 and table.level > table.cur[hero] + 1e-9:
                raise ValueError("연속된 두 이미지 사이에 내 액션이 보이지 않습니다. 같은 이미지를 두 번 넣지 않았는지 확인해 주세요")
            table.play(target, table.live - live_now, hero, hero_first=True)
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
        table.play(target, folded_here, hero, hero_first=True, until_hero=False)
        pot = table.pot()
        shown = start_of_street(shot, frame)
        if shown is not None and abs(shown - pot) > 0.1 * max(pot, 1e-9) + 1e-9:
            hand["notes"].append(f"{STREET_KO[shot['street']]}의 팟이 앞선 이미지에서 이어 계산한 값과 달라, 화면의 팟을 썼습니다")
            pot = max(shown, 0.0)
        if shot["street"] > table.street + 1:
            hand["notes"].append("이미지가 없는 스트리트의 액션은 반영하지 않았습니다")
        dealt = table.live
        table = _Street(hand, shot["street"], pot, dealt, table.stack)
        table.play(dict(enumerate(bets)), dealt - live_now, after_button)

    others = [j for j in table.live if j != hero]
    if not others:
        raise ValueError("남은 상대가 없습니다")
    cover = max(table.cur[j] + table.stack[j] for j in others)
    hand.update(pot=table.pot(), live=sorted(table.live), bets=[table.cur[i] for i in range(n)],
                stacks=[None if table.stack[i] == _INF else table.stack[i] for i in range(n)],
                to_call=min(max(0.0, min(table.level, cover) - table.cur[hero]), table.stack[hero]))
    return hand


# ---- pictures of a hand the engine played (for tests and demonstrations) ---------------

def frames(result, stacks, hero_seat, small_blind):
    """A hand the engine played -> the snapshots a camera would have taken at each of the hero's
    decisions (seat places on a circle, so that sorting by place is exercised too)."""
    n, button, big = result.n_players, result.button, result.big_blind
    stack, bet, live = list(stacks), [0] * n, set(range(n))
    sb_seat, bb_seat = _blind_seats(button, n)
    for seat, blind in ((sb_seat, small_blind), (bb_seat, big)):
        bet[seat] = min(blind, stack[seat])
        stack[seat] -= bet[seat]
    pot, street, out = bet[sb_seat] + bet[bb_seat], 0, []
    for action in result.actions:
        if action[0] != street:
            street, bet = action[0], [0] * n
        seat = action[1]
        if seat == hero_seat:
            cards = result.board[:(0, 3, 4, 5)[street]]
            out.append({
                "hero": "".join(card_str(c) for c in result.hole[hero_seat]), "board": "".join(card_str(c) for c in cards),
                "pot": pot, "big_blind": big,
                "seats": [{"name": f"p{i}", "stack": stack[i], "bet": bet[i], "in_hand": i in live, "hero": i == hero_seat,
                           "button": i == button, "x": 0.5 + 0.4 * math.cos(2 * math.pi * i / n),
                           "y": 0.5 + 0.3 * math.sin(2 * math.pi * i / n)} for i in range(n)]})
        if action[2] == FOLD:
            live.discard(seat)
        else:
            stack[seat] -= action[3]
            bet[seat] += action[3]
            pot += action[3]
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


def analyze(snapshots, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None):
    """Snapshots of a hand -> (analysis as :func:`pokerlab.advisor.analyze` returns it, the hand).

    Every opponent still in the hand gets the range that their reconstructed actions imply;
    opponents who have not acted yet hold random hands.
    """
    hand = reconstruct(snapshots)
    seats = describe(hand)
    found = infer_ranges(load_model(model), hand["actions"], hand["hero_seat"], hand["hero"], hand["board"],
                         hand["n_players"], hand["button"], profile, absent=hand["absent"])
    rivals = [seat for seat in hand["live"] if seat != hand["hero_seat"]]
    ranged = [seat for seat in rivals if seat in found]
    labels = [f"{seats[seat]['position_ko']}: {seats[seat]['actions']}" for seat in ranged]
    analysis = advisor.analyze(hand["hero"], hand["board"], len(rivals) + 1, hand["pot"], hand["to_call"] or None,
                               [found[seat] for seat in ranged], trials, seed, precision, labels=labels)
    for seat, summary in zip(ranged, analysis["ranges"]):
        summary["seat"] = seat
    return analysis, hand
