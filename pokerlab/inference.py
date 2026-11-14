"""Estimate what an opponent holds from the way they have acted.

An :class:`ActionModel` stores P(action | situation, hand strength), counted from a self-play
database where every player's hole cards are known. Given the actions an opponent has taken in
a hand, Bayes' rule turns a uniform prior over the 1,326 hole-card combos into a weighted
:class:`~pokerlab.equity.Range`, which the equity calculator accepts in place of "a random hand".

Hand strength is measured as a percentile: where a combo ranks, among all combos possible on
the board, by its chance of beating one random hand at showdown. Ten equal buckets of that
percentile mean "top 10% of possible hands", whatever the board texture.

An action is a fold, a check or call, a bet or raise in one of three sizes relative to the
pot, or an all-in. The situation says, among other things, how expensive the bet being faced
is and how many chips are still left to play for.
"""

import json
import os
import random
import re
import sqlite3
from functools import lru_cache

import numpy as np

from .cards import hand_class, parse_cards
from .engine import FOLD, RAISE
from .equity import ALL_COMBOS, COMBO_INDEX, Range, _boards, equity, matchup
from .evaluator import evaluate_pairs, rank_counts
from .preflop import preflop_equity

_DATA = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(_DATA, "action_model.json")
MODELS = {      # bundled models by name
    "selfplay": MODEL_PATH,                                     # this package's bots; updated by the daily cycle
    "online": os.path.join(_DATA, "action_model_online.json"),  # real-money online play, July 2009 (HandHQ)
    "pro": os.path.join(_DATA, "action_model_pro.json"),        # Pluribus and the professionals it played
}
N_BUCKETS = 10
# What a player can do. A bet or raise comes in three sizes, measured by the chips put in beyond the
# call as a share of the pot after calling: small (under 45%), medium, large (90% and more). A bet that
# puts in everything an opponent can match is an all-in, whatever its size.
GAVE_UP, PASSIVE, BET_SMALL, BET_MEDIUM, BET_LARGE, ALL_IN = range(6)      # fold / check or call / bet or raise
N_ACTIONS = 6
AGGRESSIVE = 6      # "a bet or raise, size unknown": the four kinds together. For asking a model only
SIZE_EDGES = (0.45, 0.9)
# What it costs to stay in: nothing, or the amount to call as a share of the pot, cut at the prices that
# bets of those two sizes create (a bet of b pots asks for b / (1 + b) of the pot).
NO_BET, CHEAP, NORMAL, EXPENSIVE = range(4)
ANY_PRICE = -1      # facing a bet whose size is unknown
PRICE_EDGES = tuple(b / (1 + b) for b in SIZE_EDGES)
# How deep the stacks are: the chips that can still go in after calling (the player's own, as far as an
# opponent can match them) as a share of the pot after calling. Nothing behind means the call ends the
# betting: the player faces an all-in, or calling is one.
NOTHING_BEHIND, SHALLOW, MIDDLING, DEEP, VERY_DEEP = range(5)
N_DEPTHS = 5
DEPTH_EDGES = (1.0, 4.0, 20.0)
ANY_DEPTH = -1      # stacks unknown: counted in a slot of its own (the last), asked as the average over all
# street, raises faced, price to call, players left, table size, position, stack depth, strength bucket, action
_SHAPE = (4, 3, 4, 4, 3, 6, N_DEPTHS + 1, N_BUCKETS, N_ACTIONS)
_AXES = {"price": 0, "live": 1, "table": 2, "position": 3, "depth": 4}      # within one street and raise count
ORDER = ("price", "depth", "live", "table", "position")     # how a situation is narrowed down for fold / call / bet
SIZE_ORDER = ("depth", "live", "table", "position")         # ... and for the size of a bet
POSITIONS = ("ep", "mp", "co", "btn", "sb", "bb")        # pre-flop seat classes, in model index order
_POSITION_ALIASES = {"utg": "ep", "lj": "mp", "hj": "mp", "button": "btn"}
_ACTS_AFTER_FLOP = {"sb": 0, "bb": 1, "ep": 2, "mp": 3, "co": 4, "btn": 5}      # order of action after the flop
SHRINK = 20.0       # pseudo-counts with which a detailed situation leans on the coarser one above it
_BOARD_CARDS = (0, 3, 4, 5)


# ---- hand strength of every combo ------------------------------------------------------

def _showdown_percentiles(boards):
    """Boards (r, 5) -> (r, 1326): share of the other possible hands each combo beats (ties half); NaN if blocked."""
    values = evaluate_pairs(boards, ALL_COMBOS)                 # -1 for blocked combos: they sort below every hand
    order = np.argsort(values, axis=1)
    ordered = np.take_along_axis(values, order, axis=1)
    starts = np.ones(ordered.shape, dtype=bool)                 # first / last entry of each run of equal values
    starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    ends = np.ones(ordered.shape, dtype=bool)
    ends[:, :-1] = starts[:, 1:]
    position = np.arange(1326, dtype=np.int16)
    first = np.maximum.accumulate(np.where(starts, position, 0), axis=1)
    last = np.minimum.accumulate(np.where(ends, position, 1325)[:, ::-1], axis=1)[:, ::-1]
    n_blocked = 1326 - 1081                                     # combos using one of the five board cards
    out = np.empty(values.shape, dtype=np.float32)              # hands below + half of the hands tied with
    np.put_along_axis(out, order, ((first + last) * 0.5 - n_blocked) / (1081 - 1), axis=1)
    out[values < 0] = np.nan
    return out


def _to_percentiles(strength):
    """Strengths (NaN = impossible) -> rank among the possible combos, scaled to 0..1."""
    possible = ~np.isnan(strength)
    keyed = np.round(strength[possible] * 1e9).astype(np.int64)[None, :]
    below, equal, _ = rank_counts(keyed)
    out = np.full(1326, np.nan)
    out[possible] = (below[0] + 0.5 * equal[0]) / (possible.sum() - 1)
    return out


@lru_cache(maxsize=1)
def _preflop_percentiles():
    return _to_percentiles(np.array([preflop_equity(hand_class(a, b), 1) for a, b in ALL_COMBOS.tolist()]))


def _mean_percentiles(boards, board):
    """Average showdown percentile over the given completions of ``board``, re-ranked to 0..1."""
    pct = _showdown_percentiles(boards)
    seen = ~np.isnan(pct)
    mean = np.nansum(pct, axis=0) / np.maximum(seen.sum(axis=0), 1)
    mean[~seen.any(axis=0)] = 0.5               # never possible in a small sample of runouts
    mean[np.isin(ALL_COMBOS, board).any(axis=1)] = np.nan
    return _to_percentiles(mean)


@lru_cache(maxsize=256)
def _exact_percentiles(board):
    boards, _ = _boards([c for c in range(52) if c not in board], list(board))
    return _mean_percentiles(boards, board)


def hand_percentiles(board, flop_runouts=None, rng=None):
    """Strength percentile (0 = weakest, 1 = strongest) of each row of ALL_COMBOS on ``board``.

    Combos that use a board card are NaN. The turn and river are enumerated; so is the flop
    (1,176 runouts, about 0.3 s) unless ``flop_runouts`` asks for that many sampled runouts instead.
    """
    board = tuple(sorted(board))
    if not board:
        return _preflop_percentiles()
    if len(board) == 3 and flop_runouts:
        deck = np.array([c for c in range(52) if c not in board], dtype=np.int8)
        rng = rng or np.random.default_rng()
        picks = np.argsort(rng.random((flop_runouts, 49)), axis=1)[:, :2]
        flop = np.tile(np.array(board, dtype=np.int8), (flop_runouts, 1))
        return _mean_percentiles(np.concatenate([flop, deck[picks]], axis=1), board)
    return _exact_percentiles(board)


def _buckets(percentiles):
    """Percentiles -> bucket index 0..N_BUCKETS-1, or -1 for impossible combos."""
    out = np.full(1326, -1, dtype=np.int64)
    possible = ~np.isnan(percentiles)
    out[possible] = np.minimum((percentiles[possible] * N_BUCKETS).astype(np.int64), N_BUCKETS - 1)
    return out


# ---- the situations a player acts in ---------------------------------------------------

def seat_class(seat, button, n_players):
    """Index into POSITIONS of a seat: early, middle, cutoff, button, small blind, big blind."""
    offset = (seat - button) % n_players
    if n_players == 2:
        return 3 if offset == 0 else 5              # heads-up the button posts the small blind
    if offset < 3:
        return (3, 4, 5)[offset]
    before_button = n_players - offset
    return 2 if before_button == 1 else 1 if before_button <= 3 else 0


def position_now(seat, button, n_players, folded, street):
    """A player's position as the model sees it: pre-flop the seat class; after the flop where
    the player acts among those still in the hand (0 first, 1 in between, 2 last)."""
    if street == 0:
        return seat_class(seat, button, n_players)
    live = [s for s in ((button + 1 + i) % n_players for i in range(n_players)) if s not in folded]
    return 0 if live and live[0] == seat else 2 if live and live[-1] == seat else 1


def table_class(n_players):
    """0 heads-up, 1 short-handed (3-6 dealt in), 2 full ring (7+); -1 when unknown."""
    return -1 if n_players is None else 0 if n_players == 2 else 1 if n_players <= 6 else 2


def price_class(to_call, pot):
    """NO_BET, or how expensive the call is: CHEAP, NORMAL or EXPENSIVE (ANY_PRICE if the pot is unknown)."""
    if to_call <= 0:
        return NO_BET
    if pot <= 0:
        return ANY_PRICE
    price = to_call / pot
    return CHEAP if price < PRICE_EDGES[0] else NORMAL if price < PRICE_EDGES[1] else EXPENSIVE


def _known(chips):
    return chips is not None and chips < float("inf")       # hand histories write an unknown stack as inf


def bet_class(added, pot, to_call, behind=None):
    """Kind of a bet or raise that put ``added`` chips in: ALL_IN if that was everything that could
    still go in (``behind``, when known), else BET_SMALL, BET_MEDIUM or BET_LARGE by its size
    (AGGRESSIVE, i.e. unknown, without a pot to measure against)."""
    call = max(to_call, 0)
    if _known(behind) and added - call >= behind - 1e-6:
        return ALL_IN
    if pot + call <= 0:
        return AGGRESSIVE
    size = (added - call) / (pot + call)
    return BET_SMALL if size < SIZE_EDGES[0] else BET_MEDIUM if size < SIZE_EDGES[1] else BET_LARGE


def depth_class(behind, pot, to_call):
    """How much is left to play for after calling: NOTHING_BEHIND, or SHALLOW .. VERY_DEEP by
    ``behind`` as a share of the pot after the call (ANY_DEPTH if the stacks are not known)."""
    after = pot + max(to_call, 0)
    if not _known(behind) or after <= 0:
        return ANY_DEPTH
    if behind <= 0:
        return NOTHING_BEHIND
    return SHALLOW + sum(behind / after >= edge for edge in DEPTH_EDGES)


def context(street, raises, price, n_live, n_players=None, position=-1, depth=ANY_DEPTH):
    """The model's notion of a situation: (street, raises already made this street capped at 2,
    price of calling, players still in the hand capped at 5, table class, position, stack depth).

    price is what :func:`price_class` returns; True stands for "there is a bet, size unknown".
    position is what :func:`position_now` returns, depth what :func:`depth_class` returns. A price
    of ANY_PRICE, a depth of ANY_DEPTH and a table class or position of -1 mean "unknown": the
    model then answers for the average over it.
    """
    price = ANY_PRICE if price is True else int(price)
    return street, min(raises, 2), price, min(max(n_live, 2), 5) - 2, table_class(n_players), position, depth


def observed(actions, n_players, button, absent=()):
    """Engine action log -> [(seat, street, context, action code)] in order.

    absent: seats that had already folded before the log starts (a log that begins after the flop)."""
    out, folded = [], set(absent)
    street, raises = -1, 0
    for a in actions:
        if a[0] != street:
            street, raises = a[0], 0
        behind = a[8] if len(a) > 8 else None           # logs from before stacks were kept have none
        code = bet_class(a[3], a[4], a[5], behind) if a[2] == RAISE else GAVE_UP if a[2] == FOLD else PASSIVE
        position = position_now(a[1], button, n_players, folded, street)
        ctx = context(street, raises, price_class(a[5], a[4]), a[6], n_players, position, depth_class(behind, a[4], a[5]))
        out.append((a[1], street, ctx, code))
        raises += a[2] == RAISE
        if code == GAVE_UP:
            folded.add(a[1])
    return out


_LINE_WORDS = {      # word -> (bets or raises?, raises already made, facing a bet?), pre-flop and post-flop
    0: {"limp": (False, 0, True), "open": (True, 0, True), "raise": (True, 0, True), "call": (False, 1, True),
        "3bet": (True, 1, True), "call3bet": (False, 2, True), "4bet": (True, 2, True), "check": (False, 0, False)},
    1: {"check": (False, 0, False), "bet": (True, 0, False), "call": (False, 1, True), "raise": (True, 1, True),
        "callraise": (False, 2, True), "reraise": (True, 2, True)},
}
_SHOVES = {"shove": ("openallin", "betallin"), "jam": ("openallin", "betallin")}     # pre-flop, post-flop
_SIZED_WORD = re.compile(r"([a-z0-9]*[a-z])(\d+(?:\.\d+)?)(x?)")
_STACK = re.compile(r"(\d+(?:\.\d+)?)bb")


def _line_word(word, street):
    """One word of an action line -> (bets or raises?, raises already made, facing a bet?, size, the open
    in big blinds, all-in?).

    A number after the word is a bet size as a share of the pot, in percent: for a bet or raise
    its own size ('bet66'), for a call the size of the bet that was called ('call66'). Pre-flop,
    'open3x' and 'call3x' give the open raise as a multiple of the big blind instead. 'allin'
    in place of a number says the bet (or the bet that was called) was all-in.
    """
    words = _LINE_WORDS[min(street, 1)]
    word = _SHOVES.get(word, (word, word))[min(street, 1)]
    size = opened_to = None
    all_in = word.endswith("allin") and word[:-5] in words
    match = _SIZED_WORD.fullmatch(word)
    if all_in:
        word = word[:-5]
    elif word not in words and match and match.group(1) in words:
        word, size = match.group(1), float(match.group(2)) / 100
        if match.group(3):
            if street or word not in ("open", "raise", "call"):
                raise ValueError(f"a size in big blinds is only understood for a pre-flop open or call, not {word!r}")
            opened_to, size = size * 100, None
        elif street == 0 and word == "call":
            opened_to, size = 1 + 2.5 * size, None      # the open that was called, as its raiser sized it
    if word not in words:
        raise ValueError(f"unknown action {word!r} on street {street + 1}; use one of {', '.join(words)}, optionally "
                         f"followed by a size in percent of the pot or by allin, e.g. bet66, raiseallin")
    return words[word] + (size, opened_to, all_in)


class _Chips:
    """The pot and one opponent's stack along an action line, in big blinds, as far as a line can
    tell: a size that was not given is taken as a typical one, every bet is called by exactly one
    player, and a line that starts after the flop comes from a pot that was raised once."""
    TYPICAL = 0.66      # a bet or raise of unknown size, as a share of the pot
    OPEN = 3.0          # an open raise of unknown size, in big blinds

    def __init__(self, stack, seat, heads_up):
        self.stack_known = stack is not None
        self.seat, self.heads_up = seat, heads_up
        self.pot, self.level, self.mine, self.left = 1.5, 1.0, None, float("inf") if stack is None else float(stack)
        self.street = self.raises = 0           # raises made on this street so far, as the line has shown them
        self.all_in = False

    def _sit(self, raises):
        """Post the blind, once it is clear which seat this is."""
        if self.mine is None:
            posted = {"sb": 0.5, "bb": 1.0}.get(self.seat, 0.0)
            if self.heads_up and self.seat != "bb":         # the button is the small blind, the other seat the big
                posted = 0.5 if raises == 0 else 1.0
            self.mine, self.left = posted, self.left - posted

    def _to(self, street):
        if street > self.street:
            if self.mine is None:               # no pre-flop action in the line: somebody opened, one player called
                self._sit(1)
                self.left -= self.OPEN - self.mine
                self.pot = 2 * self.OPEN + 1.5 - self.mine
            self.street, self.level, self.mine, self.raises = street, 0.0, 0.0, 0

    def _somebody_raises(self, share, opened_to, all_in):
        """A bet or raise by another player that the opponent then has to answer."""
        if all_in:                                          # ... which leaves the opponent nothing behind
            more = max(0.0, self.left - (self.level - self.mine))
        elif self.street == 0 and self.raises == 0:         # an open raise: its first big blind is new to the pot too
            more = (opened_to or self.OPEN) - self.level
            self.pot += self.level - (0.5 if self.heads_up else 0.0)
        else:
            more = share * self.pot
        self.pot, self.level, self.raises = self.pot + more, self.level + more, self.raises + 1

    def act(self, street, aggressive, raises, facing, size, opened_to, all_in):
        """Move the chips for one action -> (action code, price class, depth class) as the model wants them."""
        if self.all_in:
            raise ValueError("an action line cannot go on after an all-in: the opponent has no chips left to act with")
        self._to(street)
        self._sit(raises)
        if raises == 2 and self.raises == 0:    # a raise over the opponent's own bet, which the line does not show
            self.act(street, True, 0, street == 0, None, None, False)
        while self.raises < raises:             # the bet being faced: its size is only known if a call names it
            known = not aggressive
            self._somebody_raises(size if known and size is not None else self.TYPICAL, opened_to if known else None,
                                  all_in and known)
        sized = size is not None or opened_to is not None or (all_in and self.stack_known)
        call = max(0.0, min(self.level - self.mine, self.left))
        behind = self.left - call
        if not facing:
            price = NO_BET
        elif street == 0 and raises == 0 or sized and not aggressive:
            price = price_class(call, self.pot)
        else:
            price = ANY_PRICE                   # the size of the bet being faced was not given
        depth = depth_class(behind, self.pot, call) if self.stack_known and self.left > 0 else ANY_DEPTH
        if all_in and not aggressive:
            depth = NOTHING_BEHIND              # called an all-in: nothing left to play for
        code, chips = PASSIVE, call
        if aggressive:
            share = self.TYPICAL if size is None else size
            more = behind if all_in else min(behind, opened_to - self.level if opened_to else share * (self.pot + call))
            chips = call + more
            code = ALL_IN if all_in else bet_class(chips, self.pot, call, behind) if sized else AGGRESSIVE
            self.pot += more                    # the one player who calls
            self.raises += 1
        self.pot, self.left, self.mine = self.pot + chips, self.left - chips, self.mine + chips
        self.level, self.all_in = max(self.level, self.mine), all_in
        return code, price, depth


def _position_name(text, table):
    """'UTG' / 'hj' / 'button' -> a name from POSITIONS that exists at a table of that size; None for ''."""
    name = (text or "").strip().lower()
    if not name:
        return None
    name = _POSITION_ALIASES.get(name, name)
    if name not in POSITIONS:
        raise ValueError(f"unknown position {text!r}; use one of {', '.join(POSITIONS)} (or utg, hj, lj)")
    exists = {POSITIONS[seat_class(seat, 0, table)] for seat in range(table)}
    while name not in exists:                       # a short table has no early seats: use the next one round
        name = POSITIONS[POSITIONS.index(name) + 1] if name in ("ep", "mp", "co") else "btn"
    return name


def default_table(n_players):
    """Table size to assume when only the number of players in the hand is known."""
    return 6 if n_players <= 6 else 9


def parse_line(text, n_players, hero_position=None, table=None, stack=None):
    """'btn: open/bet/check' -> [(street, context, action code)]: one opponent's actions, streets split by '/'.

    Pre-flop words: limp, open, call, 3bet, call3bet, 4bet, check. Post-flop: check, bet, call,
    raise, callraise, reraise. Several words on one street are separated by spaces; an empty
    street ('open//bet') means the opponent did not act there. A word may carry a size in percent
    of the pot ('bet66', 'raise100', 'call50' = called a half-pot bet) or, pre-flop, in big blinds
    ('open3x', 'call3x'); without one the model answers for the average size. 'allin' in place of
    the size is an all-in ('betallin', 'raiseallin', 'callallin' = called one; 'shove' for short).

    n_players: players in the hand now. table: players dealt in (6, or 9 for more than six players,
    when not given). The optional prefix before a colon holds the opponent's seat (ep, mp, co,
    btn, sb, bb) and the stack ('btn 40bb: open/bet'); without a seat the model answers for the
    average seat. stack: big blinds at the start of the hand, the smaller of the opponent's and
    the biggest stack against them (the prefix overrides it); without it the model answers for
    the average stack depth. How deep the stacks are at each action is worked out from it along
    the line (see :class:`_Chips`).

    A line does not say who had folded when the opponent acted, so this is assumed: pre-flop
    everybody in front of a known seat had folded, and after the flop ``n_players`` were left.
    After the flop the model wants to know who acts last: that follows from the two seats when
    ``hero_position`` is given too (as if only the two of you were left), and otherwise only for
    the button and the blinds.
    """
    table = table or default_table(n_players)
    if not 2 <= table <= 10 or n_players > table:
        raise ValueError("the table has 2-10 seats and at least as many as there are players in the hand")
    seat = None
    if ":" in text:
        prefix, text = text.split(":", 1)
        for part in prefix.lower().split():
            if _STACK.fullmatch(part):
                stack = float(part[:-2])
            else:
                seat = _position_name(part, table)
    if stack is not None and not 1 <= stack <= 100_000:
        raise ValueError("a stack is given in big blinds, at least 1")
    hero_seat = _position_name(hero_position, table)
    if seat is None:
        after_flop, left_before_flop = -1, table
    else:
        left_before_flop = min(table, {"ep": table, "mp": 5, "co": 4, "btn": 3, "sb": 2, "bb": 2}[seat])
        if table == 2 or seat == "btn":
            after_flop = 2 if seat == "btn" else 0
        elif hero_seat and hero_seat != seat:
            after_flop = 2 if _ACTS_AFTER_FLOP[seat] > _ACTS_AFTER_FLOP[hero_seat] else 0
        else:
            after_flop = 0 if seat in ("sb", "bb") else -1
    out, chips = [], _Chips(stack, seat, table == 2)
    for street, part in enumerate(text.lower().split("/")):
        if street > 3:
            raise ValueError("an action line has at most four streets")
        for word in part.replace(",", " ").split():
            aggressive, raises, facing, size, opened_to, all_in = _line_word(word, street)
            code, price, depth = chips.act(street, aggressive, raises, facing, size, opened_to, all_in)
            if street:
                out.append((street, context(street, raises, price, n_players, table, after_flop, depth), code))
            else:
                position = -1 if seat is None else POSITIONS.index(seat)
                out.append((street, context(street, raises, price, left_before_flop, table, position, depth), code))
    return out


# ---- the model -------------------------------------------------------------------------

def _narrowed(block, known, order):
    """Counts per (bucket, action) over a block of situations, then again each time one more thing
    that is known about the situation (in ``order``) narrows the block down."""
    levels = [block.sum(axis=(0, 1, 2, 3, 4))]
    for name in order:
        if known[name] is not None:
            block = block.take([known[name]], axis=_AXES[name])
            levels.append(block.sum(axis=(0, 1, 2, 3, 4)))
    return levels


class ActionModel:
    """Counts of (situation, strength bucket) -> action, per player profile ('all' pools everyone)."""

    def __init__(self, counts, source=""):
        self.counts = {name: np.asarray(c, dtype=np.float64).reshape(_SHAPE) for name, c in counts.items()}
        self.source = source
        self._probabilities = {}

    def likelihood(self, ctx, action, profile="all"):
        """P(action | context, bucket) for every strength bucket -> array (N_BUCKETS,).

        The probability has two factors. Fold, check/call or bet: situations are nested, (street,
        raises, bet or no bet) > price of calling > stack depth > players left > table size >
        position, and each level is estimated from its own counts plus SHRINK pseudo-counts of the
        level above, so a seat that was rarely seen leans on the average seat instead of on a
        handful of hands. The kind of bet (three sizes or all-in): the same nesting without the
        price. There a narrower situation first shifts every hand strength alike, the way it
        shifts all hands taken together (with little behind, most bets are all-ins, whatever the
        hand), and only then do the counts of each strength speak, so that a size says something
        about the hand only where the counts show it.

        ``action`` may be AGGRESSIVE for "a bet of any kind". A profile the model has never seen
        falls back to the pooled 'all'.
        """
        if profile not in self.counts:
            profile = "all"
        key = (profile, ctx)
        if key not in self._probabilities:
            street, raises, price, live, table, position, depth = ctx
            block = self.counts[profile][street, raises]        # (price, players left, table, position, depth, bucket, action)
            block = block[:1] if price == NO_BET else block[1:]
            known = {"price": price - 1 if price > NO_BET else None, "live": live, "table": table if table >= 0 else None,
                     "position": position if min(table, position) >= 0 else None, "depth": depth if depth >= 0 else None}
            # one pseudo-count of a flat prior keeps every action possible: real players do, rarely,
            # fold with nothing to call, and an impossible observation would wipe out a whole range
            p, strength = np.array([1 / 3, 1 / 3, 1 / 3] if price != NO_BET else [0.02, 0.49, 0.49]), 1.0
            for counts in _narrowed(block, known, ORDER):
                counts = np.column_stack([counts[:, :BET_SMALL], counts[:, BET_SMALL:].sum(axis=1)])
                p = (counts + strength * p) / (counts.sum(axis=1, keepdims=True) + strength)
                strength = SHRINK
            if depth == NOTHING_BEHIND:         # nothing to raise with: fold or call, whatever thin counts suggest
                p[:, BET_SMALL] = 0.001
                p /= p.sum(axis=1, keepdims=True)
            split = together = None
            for counts in _narrowed(block[:, :, :, :, :N_DEPTHS], known, SIZE_ORDER):      # hands whose stacks are known
                kinds = counts[:, BET_SMALL:]
                total = kinds.sum(axis=0)
                if together is None:
                    now = (total + 1 / len(total)) / (total.sum() + 1)
                    split = np.tile(now, (N_BUCKETS, 1))
                else:
                    now = (total + SHRINK * together) / (total.sum() + SHRINK)
                    split = split * (now / together)
                    split /= split.sum(axis=1, keepdims=True)
                together = now
                split = (kinds + SHRINK * split) / (kinds.sum(axis=1, keepdims=True) + SHRINK)
            self._probabilities[key] = np.column_stack([p[:, :BET_SMALL], p[:, BET_SMALL:] * split])
        p = self._probabilities[key]
        return p[:, BET_SMALL:].sum(axis=1) if action == AGGRESSIVE else p[:, action]

    def merged(self, newer, decay=1.0):
        """This model's counts scaled by ``decay`` plus the counts of ``newer``: learning across sessions."""
        names = set(self.counts) | set(newer.counts)
        zero = np.zeros(_SHAPE)
        return ActionModel({name: decay * self.counts.get(name, zero) + newer.counts.get(name, zero) for name in names},
                           newer.source)

    def save(self, path=MODEL_PATH):
        """Write the model as JSON; only the situations that were seen are stored."""
        def seen(counts):       # hidden hands are counted in fractions: three decimals keep the rare cells
            rows = np.round(counts, 3).reshape(-1, N_BUCKETS * N_ACTIONS)
            return {",".join(str(i) for i in np.unravel_index(index, _SHAPE[:-2])):
                    [int(x) if x == int(x) else x for x in row.tolist()]
                    for index, row in enumerate(rows) if row.any()}

        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"buckets": N_BUCKETS, "shape": _SHAPE, "source": self.source,
                       "counts": {name: seen(c) for name, c in sorted(self.counts.items())}},
                      fh, separators=(",", ":"))

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data["buckets"] != N_BUCKETS or tuple(data["shape"]) != _SHAPE:
            raise ValueError(f"{path} was fitted with a different model layout; fit it again")
        counts = {}
        for name, cells in data["counts"].items():
            counts[name] = np.zeros(_SHAPE)
            for key, row in cells.items():
                counts[name][tuple(int(i) for i in key.split(","))] = np.reshape(row, (N_BUCKETS, N_ACTIONS))
        return cls(counts, data.get("source", ""))


@lru_cache(maxsize=8)
def load_model(name=MODEL_PATH):
    """A bundled model by name (see :data:`MODELS`) or the model stored in a file, read once per process."""
    return ActionModel.load(MODELS.get(name, name))


def default_model():
    return load_model(MODEL_PATH)


# ---- inference -------------------------------------------------------------------------

def strength_buckets(board, flop_runouts=None, rng=None, cache=None):
    """Strength bucket (0 weakest .. N_BUCKETS-1) of each row of ALL_COMBOS on ``board``; -1 if impossible.

    ``cache`` (a dict) is filled and reused, so several callers looking at one board compute it once.
    """
    key = tuple(board)
    if cache is None:
        return _buckets(hand_percentiles(key, flop_runouts, rng))
    if key not in cache:
        cache[key] = _buckets(hand_percentiles(key, flop_runouts, rng))
    return cache[key]


def _range_from(model, actions, hero, board, profile, flop_runouts, rng, cache):
    """Bayes update over one opponent's [(street, context, code)] -> Range, or None without information."""
    cache = {} if cache is None else cache
    weights = np.ones(1326)
    for street, ctx, code in actions:
        buckets = strength_buckets(board[:_BOARD_CARDS[street]], flop_runouts, rng, cache)
        weights *= np.where(buckets >= 0, model.likelihood(ctx, code, profile)[buckets], 0.0)
    weights[np.isin(ALL_COMBOS, list(hero) + list(board)).any(axis=1)] = 0.0
    return Range.from_vector(weights) if weights.sum() > 0 else None


def range_from_line(line, hero=(), board=(), n_players=2, model=None, profile="all", hero_position=None, table=None,
                    stack=None):
    """Range of an opponent described by an action line such as 'btn: open/bet/bet' (see :func:`parse_line`)."""
    hero, board = parse_cards(hero), parse_cards(board)
    actions = parse_line(line, n_players, hero_position, table, stack)
    if any(_BOARD_CARDS[street] > len(board) for street, _, _ in actions):
        raise ValueError("the action line goes past the streets that are on the board")
    return _range_from(model or default_model(), actions, hero, board, profile, None, None, None)


def infer_ranges(model, actions, hero_seat, hero, board, n_players, button, profile="all", flop_runouts=None,
                 rng=None, cache=None, absent=()):
    """Ranges of the opponents still in the hand, from an engine action log -> {seat: Range}.

    profile: one profile name for everybody, or {seat: name}. Opponents who have not acted yet
    are left out (treat them as random hands). absent: see :func:`observed`.
    """
    by_seat, folded = {}, set(absent)
    for seat, street, ctx, code in observed(actions, n_players, button, absent):
        if code == GAVE_UP:
            folded.add(seat)
        by_seat.setdefault(seat, []).append((street, ctx, code))
    out = {}
    for seat, own in by_seat.items():
        if seat == hero_seat or seat in folded:
            continue
        name = profile.get(seat, "all") if isinstance(profile, dict) else profile
        found = _range_from(model, own, hero, board, name, flop_runouts, rng, cache)
        if found is not None:
            out[seat] = found
    return out


# ---- fitting and scoring on a self-play database ---------------------------------------

def _load_hands(db_path, n_hands, seed, held_out=None):
    """A random sample of hands -> [{'board', 'n_players', 'button', 'holes': {seat: cards},
    'bots': {seat: name}, 'actions'}].

    held_out: None for any hand; True for the fifth of the hands set aside for scoring
    (hand_id divisible by 5), False for the other four fifths.
    """
    with sqlite3.connect(db_path) as db:
        ids = [row[0] for row in db.execute("SELECT DISTINCT hand_id FROM actions")]
        if not ids:
            raise ValueError(f"{db_path} has no action log (it was collected with --no-actions)")
        if held_out is not None:
            ids = [i for i in ids if (i % 5 == 0) == held_out]
        picked = random.Random(seed).sample(ids, min(n_hands, len(ids)))
        db.execute("CREATE TEMP TABLE picked (hand_id INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO picked VALUES (?)", [(i,) for i in picked])
        hands = {hand_id: {"board": parse_cards(board), "n_players": n, "button": button,
                           "holes": {}, "bots": {}, "actions": []}
                 for hand_id, board, n, button in db.execute(
                     "SELECT hand_id, board, n_players, button FROM hands JOIN picked USING (hand_id)")}
        for hand_id, seat, bot, hole in db.execute(
                "SELECT hand_id, seat, bot, hole FROM players JOIN picked USING (hand_id)"):
            hands[hand_id]["holes"][seat] = tuple(parse_cards(hole))
            hands[hand_id]["bots"][seat] = bot
        stacks = "behind" if "behind" in [row[1] for row in db.execute("PRAGMA table_info(actions)")] else "NULL"
        for hand_id, street, seat, action, amount, pot, to_call, n_live, behind in db.execute(
                f"SELECT hand_id, street, seat, action, amount, pot, to_call, n_live, {stacks} FROM actions "
                "JOIN picked USING (hand_id) ORDER BY hand_id, seq"):
            hands[hand_id]["actions"].append((street, seat, action, amount, pot, to_call, n_live, None, behind))
    return [hands[i] for i in sorted(hands)]


def _in_chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _prepare_chunk(job):
    hands, flop_runouts, seed = job
    rng = np.random.default_rng(seed)
    out = []
    for hand in hands:
        by_seat = {}
        for seat, street, ctx, code in observed(hand["actions"], hand["n_players"], hand["button"]):
            if code == AGGRESSIVE or ctx[2] == ANY_PRICE:
                raise ValueError("a hand record has a bet without a pot to measure it against")
            by_seat.setdefault(seat, []).append((street, ctx, code))
        buckets = {street: _buckets(hand_percentiles(hand["board"][:_BOARD_CARDS[street]], flop_runouts, rng)).astype(np.int8)
                   for street in {street for acts in by_seat.values() for street, _, _ in acts if street}}
        players = []
        for seat, acts in by_seat.items():
            hole = hand["holes"].get(seat)
            players.append((hand["bots"][seat], -1 if hole is None else int(COMBO_INDEX[hole[0], hole[1]]), tuple(acts)))
        out.append((players, buckets))
    return out


def prepare(hands, flop_runouts=150, seed=0, processes=None):
    """Hand records -> what learning needs, computed once: per hand, each player's
    (profile label, combo index or -1 if the cards are hidden, actions as (street, context, code))
    and the strength bucket of every combo on each street that saw action."""
    from multiprocessing import Pool

    jobs = [(chunk, flop_runouts, [seed, i]) for i, chunk in enumerate(_in_chunks(hands, 250))]
    with Pool(processes) as pool:
        return [item for part in pool.imap(_prepare_chunk, jobs) for item in part]


def _unsized(ctx, code):
    """An action without its sizes: a bet to call or none, stacks unknown, and a bet or raise of any kind."""
    return ctx[:2] + (ANY_PRICE if ctx[2] else NO_BET,) + ctx[3:6] + (ANY_DEPTH,), AGGRESSIVE if code > PASSIVE else code


def _coarse(ctx, code):
    """The same without table size and position either."""
    ctx, code = _unsized(ctx, code)
    return ctx[:4] + (-1, -1, ANY_DEPTH), code


def _priced(ctx, code):
    """An action with the price of calling but a bet or raise of any size."""
    return ctx, AGGRESSIVE if code > PASSIVE else code


def _as_it_is(ctx, code):
    return ctx, code


VIEWS = {"coarse": _coarse, "unsized": _unsized, "priced": _priced, "full": _as_it_is}


def _posterior(model, acts, buckets, preflop, view=_as_it_is):
    """Weights over the 1,326 combos for a player with hidden cards, the likelihood of their actions
    averaged over the possible combos, and the number of possible combos."""
    postflop = [street for street, _, _ in acts if street]
    weights = (buckets[max(postflop)] >= 0).astype(np.float64) if postflop else np.ones(1326)
    possible = weights.sum()
    for street, ctx, code in acts:      # a blocked combo (bucket -1) already has weight 0, whatever it indexes
        weights *= model.likelihood(*view(ctx, code))[buckets[street] if street else preflop]
    return weights, weights.sum() / possible, possible


def expected_counts(prepared, model, view="full", tally=True):
    """One E-step -> ({label: counts}, log-likelihood of all observed actions, number of actions).

    A player whose cards are known adds one count per action. A player whose cards are hidden adds
    fractional counts, spread over the strength buckets in proportion to how likely each combo is
    given everything that player did in the hand under ``model``.

    view: how much of each action the model is shown while it judges the hidden hands (a name from
    VIEWS): 'full', 'priced' (bet sizes merged), 'unsized' (the price of calling merged too) or
    'coarse' (table size and position left out as well). The counts are filed under the full
    situation and action whatever the view. ``tally=False`` only computes the likelihood.
    """
    view = VIEWS[view]
    preflop = _buckets(_preflop_percentiles())
    preflop_share = np.bincount(preflop, minlength=N_BUCKETS) / 1326
    counts, folded_early = {}, {}
    log_likelihood, n_actions = 0.0, 0
    for players, buckets in prepared:
        for label, combo, acts in players:
            n_actions += len(acts)
            if combo < 0 and not any(street for street, _, _ in acts):
                key = (label, acts)         # hidden and only acted pre-flop: identical for everyone alike
                folded_early[key] = folded_early.get(key, 0) + 1
                continue
            if tally and label not in counts:
                counts[label] = np.zeros(_SHAPE)
            target = counts.get(label)
            if combo >= 0:
                for street, ctx, code in acts:
                    bucket = int(buckets[street][combo] if street else preflop[combo])
                    if tally:
                        target[ctx + (bucket, code)] += 1
                    log_likelihood += np.log(model.likelihood(*view(ctx, code))[bucket])
            else:
                weights, likelihood, _ = _posterior(model, acts, buckets, preflop, view)
                if likelihood <= 0:         # cannot happen with a smoothed model; never let one hand poison the fit
                    n_actions -= len(acts)
                    continue
                log_likelihood += np.log(likelihood)
                if not tally:
                    continue
                weights /= weights.sum()
                for street, ctx, code in acts:
                    where = buckets[street].clip(0) if street else preflop
                    target[ctx][:, code] += np.bincount(where, weights=weights, minlength=N_BUCKETS)
    for (label, acts), n in folded_early.items():
        share = preflop_share.copy()
        for _, ctx, code in acts:
            share *= model.likelihood(*view(ctx, code))
        log_likelihood += n * np.log(share.sum())
        if tally:
            share *= n / share.sum()
            if label not in counts:
                counts[label] = np.zeros(_SHAPE)
            target = counts[label]
            for _, ctx, code in acts:
                target[ctx][:, code] += share
    if tally:
        counts["all"] = sum(counts.values())
    return counts, float(log_likelihood), n_actions


STAGES = ("coarse", "unsized")      # see fit_prepared


def fit_prepared(prepared, initial=None, iterations=30, tolerance=5e-5, source="", progress=None, stages=None):
    """Learn an :class:`ActionModel` from prepared hands.

    With every hole card known this is plain counting. With hidden cards it is EM: infer what the
    hidden hands probably were, recount, and repeat until the log-likelihood per action improves by
    less than ``tolerance`` (at most ``iterations`` passes per stage).

    stages: the views (see :func:`expected_counts`) under which the hidden hands are judged, one
    EM run after the other. By default first 'coarse', then 'unsized'; with a starting model only
    the last of them. Started cold on the detailed situations, EM spreads the hidden hands thinly
    and takes very long to sharpen them, hence the coarse stage.

    Bet sizes and call prices are left out of the judging by default. Bots that act on hand
    strength alone are recovered a little better with them (stages ending in 'full'). On
    real-money hands that fit read shown hands worse than one that ignored sizes: a bet that
    takes the pot leaves no cards to check against, and whatever habit makes a player choose a
    size gets blamed on the hand.
    """
    model = initial or ActionModel({"all": np.zeros(_SHAPE)})
    if not any(combo < 0 for players, _ in prepared for _, combo, _ in players):
        return ActionModel(expected_counts(prepared, model)[0], source)
    stages = stages or (STAGES[-1:] if initial else STAGES)
    for view in stages:
        previous = None
        for step in range(iterations):
            counts, log_likelihood, n_actions = expected_counts(prepared, model, view)
            model = ActionModel(counts, source)
            per_action = log_likelihood / max(n_actions, 1)
            if progress:
                progress(f"EM {view} {step + 1}: log-likelihood per action {per_action:.5f} "
                         f"(of the model before this step)")
            if previous is not None and per_action - previous < tolerance:
                break
            previous = per_action
    return model


def action_fit(prepared, model):
    """How well ``model`` explains play it was not fitted on.

    log_likelihood: average log-probability of each observed action (hidden cards averaged out).
    shown_hands / shown_lift: for hands shown at showdown, how many times more probability the
    range inferred from that player's actions put on the hand they really held than a uniform
    guess over the possible combos (geometric mean).
    """
    preflop = _buckets(_preflop_percentiles())
    _, log_likelihood, n_actions = expected_counts(prepared, model, tally=False)
    lifts = []
    for players, buckets in prepared:
        for _, combo, acts in players:
            if combo >= 0:
                weights, _, possible = _posterior(model, acts, buckets, preflop)
                lifts.append(np.log(weights[combo] / weights.sum() * possible))
    return {"actions": n_actions, "log_likelihood": log_likelihood / max(n_actions, 1),
            "shown_hands": len(lifts), "shown_lift": float(np.exp(np.mean(lifts))) if lifts else float("nan")}


def fit(db_path, hands=15_000, flop_runouts=300, seed=0, processes=None, held_out=None):
    """Count how the players in a self-play database acted with each hand strength -> :class:`ActionModel`.

    One profile per bot name found in the database, plus 'all'. ``held_out=False`` leaves the
    scoring fifth of the hands untouched (see :func:`score`).
    """
    sample = _load_hands(db_path, hands, seed, held_out)
    return fit_prepared(prepare(sample, flop_runouts, seed, processes),
                        source=f"{len(sample)} hands from {os.path.basename(db_path)}")


def _score_chunk(job):
    hands, model, trials, seed = job
    rng = random.Random(seed)
    rows = []
    for hand in hands:
        if not hand["actions"]:
            continue
        cut = rng.randrange(len(hand["actions"]))               # one decision point per hand
        street, hero_seat = hand["actions"][cut][0], hand["actions"][cut][1]
        before = hand["actions"][:cut]
        board = hand["board"][:_BOARD_CARDS[street]]
        hero = hand["holes"][hero_seat]
        folded = {a[1] for a in before if a[2] == FOLD}
        rivals = [seat for seat in hand["holes"] if seat != hero_seat and seat not in folded]
        np_seed = rng.getrandbits(32)

        truth = matchup([hero] + [hand["holes"][seat] for seat in rivals], board, trials=trials,
                        exact="auto" if board else False, seed=np_seed)[0].equity
        row = {"street": street, "opponents": len(rivals), "truth": truth,
               "random": equity(hero, board, len(rivals), trials=trials, seed=np_seed).equity}
        for label, profile in (("model", "all"), ("model_profiled", hand["bots"])):
            ranges = infer_ranges(model, before, hero_seat, hero, board, hand["n_players"], hand["button"], profile,
                                  100, np.random.default_rng(np_seed))
            row[label] = equity(hero, board, len(rivals), list(ranges.values()), trials=trials, seed=np_seed).equity
        rows.append(row)
    return rows


METHODS = ("random", "model", "model_profiled")


def score_hands(hands, model=None, trials=4000, seed=1, processes=None):
    """Equity estimates next to the truth, on one decision point from each hand (all hole cards must be known).

    Returns a DataFrame with a row per decision: street, opponents, ``truth`` (the acting player's
    equity against the hole cards the opponents really held) and the estimate of each method:
    ``random`` (opponents hold random hands), ``model`` (ranges inferred with the pooled profile)
    and ``model_profiled`` (ranges inferred with each opponent's own profile).
    See :func:`summarize` and :func:`reliability`.
    """
    from multiprocessing import Pool

    import pandas as pd

    model = model or default_model()
    jobs = [(chunk, model, trials, seed * 1000 + i) for i, chunk in enumerate(_in_chunks(hands, 100))]
    with Pool(processes) as pool:
        return pd.DataFrame([row for part in pool.imap(_score_chunk, jobs) for row in part])     # in hand order


def score(db_path, model=None, hands=4000, trials=4000, seed=1, processes=None, held_out=None):
    """:func:`score_hands` on hands sampled from a self-play database."""
    return score_hands(_load_hands(db_path, hands, seed, held_out), model, trials, seed, processes)


def summarize(rows):
    """Per street and overall: decisions, then mean absolute error and bias (estimate - truth) per method."""
    import pandas as pd

    def one(part):
        out = {"decisions": len(part)}
        for method in METHODS:
            out[f"mae_{method}"] = (part[method] - part["truth"]).abs().mean()
            out[f"bias_{method}"] = (part[method] - part["truth"]).mean()
        return out

    names = ("preflop", "flop", "turn", "river")
    table = {names[street]: one(part) for street, part in rows.groupby("street")}
    table["all"] = one(rows)
    return pd.DataFrame(table).T


def reliability(rows, bins=10):
    """When a method says x%, what was the truth on average? One row per estimate bin and method."""
    import pandas as pd

    out = {}
    for method in METHODS:
        bucket = np.minimum((rows[method] * bins).astype(int), bins - 1)
        grouped = rows.groupby(bucket)
        out[f"n_{method}"] = grouped.size()
        out[f"said_{method}"] = grouped[method].mean()
        out[f"truth_{method}"] = grouped["truth"].mean()
    table = pd.DataFrame(out)
    table.index = [f"{i * 100 // bins}-{(i + 1) * 100 // bins}%" for i in table.index]
    return table
