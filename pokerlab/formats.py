"""Tournament formats as data, and tables played the way a format says.

``data/formats/kr_formats.json`` holds the formats: table size, starting stack, blinds and how
fast they rise, how long late registration stays open, how the event ends. Every value says
where it comes from: *observed* in a published source, *estimated* from observed values,
*assumed* so that a table can be played at all, or *unknown*. The tables the values were taken
from are next to it in ``data/formats/source``.

:func:`table` plays one table of a format with the engine, hand after hand, so that self-play
meets what such an event produces: stacks that get shallow as the blinds rise, fresh stacks that
sit down late, short-handed endings. One table stands in for the whole field (see :func:`table`
for how); bots play for chips and know nothing of prize money.
"""

import csv
import json
import math
import os

from .engine import play_hand

DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "formats")
PATH = os.path.join(DIRECTORY, "kr_formats.json")
BASES = ("observed", "estimated", "assumed", "unknown")
_NEEDED = ("seats", "start_chips", "big_blind", "blind_growth", "level_min", "hand_sec")


def _even(chips):
    """A blind as tables post it: two significant digits, and even, so that the small blind is half of it."""
    if chips < 100:
        return max(2, 2 * round(chips / 2))
    step = 10 ** (int(math.floor(math.log10(chips))) - 1)
    return int(round(chips / step) * step)


class Format:
    """One format. Values are attributes; ``basis`` tells how each one is known."""

    def __init__(self, name, spec):
        self.name, self.label, self.kind = name, spec.get("label", name), spec.get("kind", "")
        self.example = spec.get("example", "")
        self.values = {key: entry["value"] for key, entry in spec.items() if isinstance(entry, dict) and "value" in entry}
        self.basis = {key: entry.get("basis", "unknown") for key, entry in spec.items()
                      if isinstance(entry, dict) and "value" in entry}
        self.notes = {key: entry for key, entry in spec.items() if isinstance(entry, dict) and "value" in entry}
        get = self.values.get
        self.seats, self.start_chips, self.big_blind = get("seats"), get("start_chips"), get("big_blind")
        self.ante_bb, self.growth = get("ante_bb") or 0.0, get("blind_growth")
        self.level_min, self.hand_sec = get("level_min"), get("hand_sec")
        self.late_reg_levels, self.entries = get("late_reg_levels") or 0, get("entries")
        self.finish, self.finish_after_level = get("finish") or 1, get("finish_after_level") or 0
        self.action_sec = get("action_sec")

    @property
    def playable(self):
        """Can a table be played? Only when nothing it needs is missing."""
        return all(isinstance(self.values.get(key), (int, float)) for key in _NEEDED)

    @property
    def hands_per_level(self):
        """Hands one table gets through before the blinds rise."""
        return max(1, round(self.level_min * 60 / self.hand_sec))

    @property
    def start_bb(self):
        return self.start_chips / self.big_blind

    def level(self, hand):
        """The blind level during the table's ``hand``-th hand (both counted from 1 and 0)."""
        return 1 + hand // self.hands_per_level

    def blinds(self, level):
        """(small blind, big blind, ante) at a level."""
        big = _even(self.big_blind * self.growth ** (level - 1))
        return big // 2, big, int(round(self.ante_bb * big))


def _read(path=PATH):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load(path=PATH):
    """{name: Format} of every format that can be played."""
    return {name: Format(name, spec) for name, spec in _read(path)["formats"].items()}


def get(name, path=PATH):
    found = load(path)
    if name not in found:
        raise ValueError(f"unknown format {name!r}; choose from {', '.join(sorted(found) + sorted(mixes(path)))}")
    return found[name]


def mixes(path=PATH):
    """{name: {format: weight}}: how often formats occur next to each other."""
    return {name: dict(mix["weights"]) for name, mix in _read(path).get("mixes", {}).items()}


def resolve(names, path=PATH):
    """'mtt_turbo', 'mtt_turbo,satellite' or the name of a mix -> [(Format, weight)]."""
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]
    known, out = mixes(path), []
    for name in names:
        for member, weight in (known[name].items() if name in known else ((name, 1),)):
            fmt = get(member, path)
            if not fmt.playable:
                raise ValueError(f"format {member!r} cannot be played: a value it needs is unknown")
            out.append((fmt, weight))
    if not out:
        raise ValueError("no format named")
    return out


def source(name):
    """Rows of a table in ``data/formats/source`` ('league_parameters' or 'league_wpl_schedule')."""
    with open(os.path.join(DIRECTORY, "source", name + ".csv"), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def schedule_speeds():
    """How many of the tournaments that run every day are turbos, from the published schedule."""
    daily = [row for row in source("league_wpl_schedule") if row["recurrence"] == "daily"]
    turbo = sum(row["speed"] == "turbo" for row in daily)
    return {"turbo": turbo, "regular": len(daily) - turbo}


def table(fmt, new_bot, rng, max_hands=None):
    """Play one table of ``fmt`` to its end, yielding a dict for every hand:
    {'result', 'bots', 'stacks', 'button', 'small_blind', 'big_blind', 'ante', 'level', 'hand'}.

    ``new_bot()`` returns the bot for a player who sits down. The table starts full with starting
    stacks, and the blinds rise every :attr:`Format.hands_per_level` hands. A player without chips
    leaves. What happens to the empty seat is where one table has to stand in for a field:

    * while late registration is open, a new player sits down with a starting stack;
    * afterwards the field shrinks. Every table is taken to lose players at the rate this one
      does, so a bust here takes one player from each table still running. While other tables
      remain, one of their players is moved into the seat, with a stack drawn from the stacks at
      this table (the field is taken to look like this table);
    * once the field fits at one table nobody is moved in any more: the final table plays down.

    The table ends when ``fmt.finish`` players are left in the field (one, or a satellite's nine).
    """
    hands_per_level = fmt.hands_per_level
    players = [[new_bot(), fmt.start_chips] for _ in range(fmt.seats)]
    button, hand, late_entries, field = rng.randrange(fmt.seats), 0, 0, None
    while len(players) > 1 and (max_hands is None or hand < max_hands):
        level = fmt.level(hand)
        small, big, ante = fmt.blinds(level)
        button %= len(players)
        stacks = [chips for _, chips in players]
        bots = [bot for bot, _ in players]
        result = play_hand(bots, stacks, button, small, big, rng, ante)
        yield {"result": result, "bots": bots, "stacks": stacks, "button": button, "small_blind": small,
               "big_blind": big, "ante": ante, "level": level, "hand": hand}
        for player, won in zip(players, result.net):
            player[1] += won
        hand += 1
        closed = fmt.level(hand) > fmt.late_reg_levels
        if closed and field is None:        # every table has seen as many late entries for each seat as this one
            field = max(len(players), round((fmt.entries or fmt.seats) * fmt.seats / (fmt.seats + late_entries)))
        busted = [index for index, (_, chips) in enumerate(players) if chips <= 0]
        for index in reversed(busted):
            del players[index]
            if index < button:              # the seats to the right of the button moved up by one
                button -= 1
        for _ in busted:
            if not closed:
                players.append([new_bot(), fmt.start_chips])
                late_entries += 1
                continue
            field -= math.ceil(field / fmt.seats)
            if field > len(players) and len(players) < fmt.seats and players:
                players.append([new_bot(), rng.choice([chips for _, chips in players])])
        button += 1
        if field is not None:
            field = max(field, len(players))
            if field <= fmt.finish and fmt.level(hand) > fmt.finish_after_level:
                break


def depths(fmt, new_bot, rng, hands=2000):
    """What a format's tables look like, over ``hands`` hands: how deep the stacks are and how many play.

    -> {'hands', 'tables', 'players': mean dealt in, 'median_bb', 'share': {'<10bb', '10-20bb',
    '20-40bb', '40bb+'} of the stacks dealt in, 'max_level'}.
    """
    seen, dealt, tables, top = [], 0, 0, 0
    played = 0
    while played < hands:
        tables += 1
        for row in table(fmt, new_bot, rng, max_hands=hands - played):
            played += 1
            dealt += len(row["stacks"])
            top = max(top, row["level"])
            seen += [chips / row["big_blind"] for chips in row["stacks"]]
    seen.sort()
    edges = (("<10bb", 0, 10), ("10-20bb", 10, 20), ("20-40bb", 20, 40), ("40bb+", 40, float("inf")))
    return {"hands": played, "tables": tables, "players": dealt / played, "median_bb": seen[len(seen) // 2],
            "share": {name: sum(low <= x < high for x in seen) / len(seen) for name, low, high in edges}, "max_level": top}


def describe(path=PATH):
    """The formats as lines of text: every value with how it is known."""
    data, lines = _read(path), []
    marks = {"observed": "관찰", "estimated": "추정", "assumed": "가정", "unknown": "미확인"}
    for name, spec in data["formats"].items():
        fmt = Format(name, spec)
        lines.append(f"{name}: {fmt.label}  ({fmt.example})")
        if fmt.playable:
            small, big, ante = fmt.blinds(1)
            lines.append(f"  시작 {fmt.start_bb:g}BB · {fmt.seats}인 · 블라인드 {small}/{big}"
                         f"{f' 앤티 {ante}' if ante else ''} · 레벨마다 ×{fmt.growth:g}, {fmt.hands_per_level}핸드마다 상승"
                         f" · 추가 등록 레벨 {fmt.late_reg_levels}까지 · {fmt.finish}명 남으면 종료")
        for key, entry in fmt.notes.items():
            note = f"  — {entry['note']}" if entry.get("note") else ""
            origin = f" {entry['source']}" if entry.get("source") else ""
            lines.append(f"    {key:<20} {entry['value']!s:<10} [{marks[entry.get('basis', 'unknown')]}{origin}]{note}")
    for name, mix in data.get("mixes", {}).items():
        lines.append(f"{name}: {mix.get('label', '')}  " + ", ".join(f"{k} {w}" for k, w in mix["weights"].items())
                     + f"  [{marks[mix.get('basis', 'unknown')]} {mix.get('source', '')}]")
    for name, spec in data.get("observations", {}).items():
        lines.append(f"({name}: {spec.get('label', '')} — 돌릴 수 없음: {spec.get('why_not_played', '')})")
    return lines
