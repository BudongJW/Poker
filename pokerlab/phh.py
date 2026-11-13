"""Read public hand histories in the PHH format (https://phh.readthedocs.io) into the hand
records the range model learns from.

Each hand becomes ``{'board', 'n_players', 'button', 'holes', 'bots', 'actions'}`` like a hand loaded from a
self-play database, with two differences: ``holes`` only has the seats whose cards are known
(dealt face up in the log, or shown at showdown), and ``bots`` holds a label per seat that the
caller chooses (a player type, a stake level, ...), which becomes a model profile.

Only no-limit hold'em hands are read. Bet sizes are not kept: the model looks at who folded,
called or raised, on which street, facing what.
"""

import os
import random
import re

from .cards import parse_cards
from .engine import CALL, CHECK, FOLD, RAISE

_HAND_SPLIT = re.compile(r"^\[\d+\]\s*$", re.MULTILINE)
_LIST = {name: re.compile(rf"^{name} = \[(.*)\]\s*$", re.MULTILINE)
         for name in ("actions", "blinds_or_straddles", "players", "starting_stacks")}
_VARIANT = re.compile(r"^variant = '(\w+)'", re.MULTILINE)
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def _field(text, name):
    match = _LIST[name].search(text)
    return match.group(1) if match else None


def _two_cards(text):
    """'AsKd' -> (card, card); None when hidden ('????') or not two cards."""
    try:
        cards = parse_cards(text)
    except ValueError:
        return None
    return tuple(cards) if len(cards) == 2 else None


def parse_hand(text, label=None):
    """One PHH hand -> hand record, or None if it is not a no-limit hold'em hand this reader understands.

    label: profile name for every seat, or a function (player name, big blind) -> profile name.
    Seats are 0-based in the order of the file (p1 is seat 0).
    """
    variant, raw_actions, raw_blinds = _VARIANT.search(text), _field(text, "actions"), _field(text, "blinds_or_straddles")
    if not variant or variant.group(1) != "NT" or raw_actions is None or raw_blinds is None:
        return None
    try:
        bets = [float(x) for x in raw_blinds.split(",")]
    except ValueError:
        return None
    n = len(bets)
    if not 2 <= n <= 10:
        return None
    if n == 2:
        bets.reverse()      # heads-up the last player is the button and posts the small blind
    names = [a or b for a, b in _QUOTED.findall(_field(text, "players") or "")]
    big_blind = max(bets)

    current, street, live = big_blind, 0, set(range(n))
    board, holes, actions = [], {}, []
    for a, b in _QUOTED.findall(raw_actions):
        words = (a or b).split("#")[0].split()
        if len(words) < 2:
            return None
        if words[0] == "d":
            if words[1] == "dh" and len(words) == 4:
                cards = _two_cards(words[3])
                if cards:
                    holes[int(words[2][1:]) - 1] = cards
            elif words[1] == "db" and len(words) == 3:
                try:
                    board += parse_cards(words[2])
                except ValueError:
                    return None
                street += 1
                bets, current = [0.0] * n, 0.0
                if street > 3 or len(board) != (0, 3, 4, 5)[street]:
                    return None
            else:
                return None
            continue
        if not (words[0][0] == "p" and words[0][1:].isdigit()):
            return None
        seat = int(words[0][1:]) - 1
        if not 0 <= seat < n:
            return None
        if words[1] == "sm":                             # show or muck at the end of the hand
            cards = _two_cards(words[2]) if len(words) > 2 else None
            if cards:
                holes[seat] = cards
            continue
        if seat not in live:
            return None
        to_call = current - bets[seat]
        if words[1] == "f":
            actions.append((street, seat, FOLD, 0, 0, to_call, len(live), None))
            live.discard(seat)
        elif words[1] == "cc":
            actions.append((street, seat, CALL if to_call > 0 else CHECK, 0, 0, to_call, len(live), None))
            bets[seat] = max(bets[seat], current)
        elif words[1] == "cbr" and len(words) > 2:
            try:
                amount = float(words[2])
            except ValueError:
                return None
            actions.append((street, seat, RAISE, amount, 0, to_call, len(live), None))
            bets[seat] = current = max(amount, current)
        else:
            return None
    if not actions or len(set(board)) != len(board):
        return None
    used = set(board)
    for seat, cards in list(holes.items()):             # drop impossible reveals rather than the whole hand
        if used & set(cards):
            del holes[seat]
        else:
            used |= set(cards)
    if callable(label):
        bots = {seat: label(names[seat] if seat < len(names) else "", big_blind) for seat in range(n)}
    else:
        bots = {seat: label or "all" for seat in range(n)}
    # seats are listed blinds first, so the last one is the button (heads-up too: it posts the small blind)
    return {"board": board, "n_players": n, "button": n - 1, "holes": holes, "bots": bots, "actions": actions}


def read_file(path, label=None):
    """Every readable hand in a ``.phh`` (one hand) or ``.phhs`` (many hands) file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    parts = _HAND_SPLIT.split(text) if path.endswith(".phhs") else [text]
    return [hand for hand in (parse_hand(part, label) for part in parts if part.strip()) if hand]


def find_files(paths):
    """Files and directories -> sorted list of the ``.phh`` / ``.phhs`` files in them."""
    out = []
    for path in [paths] if isinstance(paths, str) else paths:
        if os.path.isdir(path):
            for folder, _, names in os.walk(path):
                out += [os.path.join(folder, name) for name in names if name.endswith((".phh", ".phhs"))]
        else:
            out.append(path)
    return sorted(out)


def stake_label(name, big_blind):
    """Profile per stake band of a cash game, from the big blind in currency units (0.25 -> 'nl25-50')."""
    cents = round(big_blind * 100)
    return "nl25-50" if cents <= 50 else "nl100-200" if cents <= 200 else "nl400-600" if cents <= 600 else "nl1000"


def pluribus_label(name, big_blind):
    """'pluribus' for the bot, 'pro' for the human professionals it played."""
    return "pluribus" if name == "Pluribus" else "pro"


LABELS = {"stake": stake_label, "pluribus": pluribus_label}


def load(paths, label=None, limit=None, seed=0):
    """Hands from files and directories; with ``limit``, files are read in random order until it is reached."""
    files = find_files(paths)
    if limit:
        random.Random(seed).shuffle(files)
    hands = []
    for path in files:
        hands += read_file(path, label)
        if limit and len(hands) >= limit:
            return hands[:limit]
    return hands
