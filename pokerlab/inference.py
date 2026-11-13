"""Estimate what an opponent holds from the way they have acted.

An :class:`ActionModel` stores P(action | situation, hand strength), counted from a self-play
database where every player's hole cards are known. Given the actions an opponent has taken in
a hand, Bayes' rule turns a uniform prior over the 1,326 hole-card combos into a weighted
:class:`~pokerlab.equity.Range`, which the equity calculator accepts in place of "a random hand".

Hand strength is measured as a percentile: where a combo ranks, among all combos possible on
the board, by its chance of beating one random hand at showdown. Ten equal buckets of that
percentile mean "top 10% of possible hands", whatever the board texture.
"""

import json
import os
import random
import sqlite3
from functools import lru_cache

import numpy as np

from .cards import hand_class, parse_cards
from .engine import FOLD, RAISE
from .equity import ALL_COMBOS, COMBO_INDEX, Range, _boards, equity, matchup
from .evaluator import evaluate_pairs, rank_counts
from .preflop import preflop_equity

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "action_model.json")
N_BUCKETS = 10
GAVE_UP, PASSIVE, AGGRESSIVE = 0, 1, 2          # fold / check or call / bet or raise
_SHAPE = (4, 3, 2, 4, N_BUCKETS, 3)             # street, raises faced, facing a bet, players left, bucket, action
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

def context(street, raises, facing_bet, n_live):
    """The model's notion of a situation: (street, raises already made this street capped at 2,
    whether there is a bet to call, players still in the hand capped at 5)."""
    return street, min(raises, 2), int(bool(facing_bet)), min(max(n_live, 2), 5) - 2


def observed(actions):
    """Engine action log -> [(seat, street, context, action code)] in order."""
    out = []
    street, raises = -1, 0
    for a in actions:
        if a[0] != street:
            street, raises = a[0], 0
        code = AGGRESSIVE if a[2] == RAISE else GAVE_UP if a[2] == FOLD else PASSIVE
        out.append((a[1], street, context(street, raises, a[5] > 0, a[6]), code))
        raises += a[2] == RAISE
    return out


_LINE_WORDS = {      # word -> (action code, raises already made, facing a bet), pre-flop and post-flop
    0: {"limp": (PASSIVE, 0, 1), "open": (AGGRESSIVE, 0, 1), "raise": (AGGRESSIVE, 0, 1), "call": (PASSIVE, 1, 1),
        "3bet": (AGGRESSIVE, 1, 1), "call3bet": (PASSIVE, 2, 1), "4bet": (AGGRESSIVE, 2, 1), "check": (PASSIVE, 0, 0)},
    1: {"check": (PASSIVE, 0, 0), "bet": (AGGRESSIVE, 0, 0), "call": (PASSIVE, 1, 1), "raise": (AGGRESSIVE, 1, 1),
        "callraise": (PASSIVE, 2, 1), "reraise": (AGGRESSIVE, 2, 1)},
}


def parse_line(text, n_players):
    """'open/bet/check' -> [(street, context, action code)]: one opponent's actions, streets split by '/'.

    Pre-flop words: limp, open, call, 3bet, call3bet, 4bet, check. Post-flop: check, bet, call,
    raise, callraise, reraise. Several words on one street are separated by spaces; an empty
    street ('open//bet') means the opponent did not act there.
    """
    out = []
    for street, part in enumerate(text.lower().split("/")):
        if street > 3:
            raise ValueError("an action line has at most four streets")
        words = _LINE_WORDS[min(street, 1)]
        for word in part.replace(",", " ").split():
            if word not in words:
                raise ValueError(f"unknown action {word!r} on street {street + 1}; use one of {', '.join(words)}")
            code, raises, facing = words[word]
            out.append((street, context(street, raises, facing, n_players), code))
    return out


# ---- the model -------------------------------------------------------------------------

class ActionModel:
    """Counts of (situation, strength bucket) -> action, per player profile ('all' pools everyone)."""

    def __init__(self, counts, source=""):
        self.counts = {name: np.asarray(c, dtype=np.float64).reshape(_SHAPE) for name, c in counts.items()}
        self.source = source
        self._likelihoods = {}

    def likelihood(self, ctx, action, profile="all"):
        """P(action | context, bucket) for every strength bucket -> array (N_BUCKETS,).

        A profile the model has never seen falls back to the pooled 'all'.
        """
        if profile not in self.counts:
            profile = "all"
        key = (profile, ctx, action)
        if key not in self._likelihoods:
            counts = self.counts[profile][ctx]
            if counts.sum() < 200:                      # rarely seen: pool the table sizes
                counts = self.counts[profile][ctx[0], ctx[1], ctx[2]].sum(axis=0)
            prior = np.array([1 / 3] * 3 if ctx[2] else [0.0, 0.5, 0.5])     # cannot fold without a bet
            self._likelihoods[key] = ((counts + prior) / (counts.sum(axis=1, keepdims=True) + 1))[:, action]
        return self._likelihoods[key]

    def merged(self, newer, decay=1.0):
        """This model's counts scaled by ``decay`` plus the counts of ``newer``: learning across sessions."""
        names = set(self.counts) | set(newer.counts)
        zero = np.zeros(_SHAPE)
        return ActionModel({name: decay * self.counts.get(name, zero) + newer.counts.get(name, zero) for name in names},
                           newer.source)

    def save(self, path=MODEL_PATH):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"buckets": N_BUCKETS, "shape": _SHAPE, "source": self.source,
                       "counts": {name: np.round(c, 1).ravel().tolist() for name, c in sorted(self.counts.items())}},
                      fh, separators=(",", ":"))

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data["buckets"] != N_BUCKETS or tuple(data["shape"]) != _SHAPE:
            raise ValueError(f"{path} was fitted with a different model layout")
        return cls(data["counts"], data.get("source", ""))


@lru_cache(maxsize=4)
def load_model(path=MODEL_PATH):
    """The action model stored at ``path`` (the bundled one by default), read once per process."""
    return ActionModel.load(path)


def default_model():
    return load_model(MODEL_PATH)


# ---- inference -------------------------------------------------------------------------

def _range_from(model, actions, hero, board, profile, flop_runouts, rng, cache):
    """Bayes update over one opponent's [(street, context, code)] -> Range, or None without information."""
    cache = {} if cache is None else cache
    weights = np.ones(1326)
    for street, ctx, code in actions:
        key = tuple(board[:_BOARD_CARDS[street]])
        if key not in cache:
            cache[key] = _buckets(hand_percentiles(key, flop_runouts, rng))
        buckets = cache[key]
        weights *= np.where(buckets >= 0, model.likelihood(ctx, code, profile)[buckets], 0.0)
    weights[np.isin(ALL_COMBOS, list(hero) + list(board)).any(axis=1)] = 0.0
    return Range.from_vector(weights) if weights.sum() > 0 else None


def range_from_line(line, hero=(), board=(), n_players=2, model=None, profile="all"):
    """Range of an opponent described by an action line such as 'open/bet/bet' (see :func:`parse_line`)."""
    hero, board = parse_cards(hero), parse_cards(board)
    actions = parse_line(line, n_players)
    if any(_BOARD_CARDS[street] > len(board) for street, _, _ in actions):
        raise ValueError("the action line goes past the streets that are on the board")
    return _range_from(model or default_model(), actions, hero, board, profile, None, None, None)


def infer_ranges(model, actions, hero_seat, hero, board, profile="all", flop_runouts=None, rng=None, cache=None):
    """Ranges of the opponents still in the hand, from an engine action log -> {seat: Range}.

    profile: one profile name for everybody, or {seat: name}. Opponents who have not acted yet
    are left out (treat them as random hands).
    """
    by_seat, folded = {}, set()
    for seat, street, ctx, code in observed(actions):
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
    """A random sample of hands -> [{'board', 'holes': {seat: cards}, 'bots': {seat: name}, 'actions'}].

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
        hands = {hand_id: {"board": parse_cards(board), "holes": {}, "bots": {}, "actions": []}
                 for hand_id, board in db.execute("SELECT hand_id, board FROM hands JOIN picked USING (hand_id)")}
        for hand_id, seat, bot, hole in db.execute(
                "SELECT hand_id, seat, bot, hole FROM players JOIN picked USING (hand_id)"):
            hands[hand_id]["holes"][seat] = tuple(parse_cards(hole))
            hands[hand_id]["bots"][seat] = bot
        for hand_id, street, seat, action, amount, pot, to_call, n_live in db.execute(
                "SELECT hand_id, street, seat, action, amount, pot, to_call, n_live FROM actions "
                "JOIN picked USING (hand_id) ORDER BY hand_id, seq"):
            hands[hand_id]["actions"].append((street, seat, action, amount, pot, to_call, n_live, None))
    return [hands[i] for i in sorted(hands)]


def _fit_chunk(job):
    hands, flop_runouts, seed = job
    rng = np.random.default_rng(seed)
    counts = {}
    for hand in hands:
        buckets = {}
        for seat, street, ctx, code in observed(hand["actions"]):
            if street not in buckets:
                buckets[street] = _buckets(hand_percentiles(hand["board"][:_BOARD_CARDS[street]], flop_runouts, rng))
            hole = hand["holes"][seat]
            cell = ctx + (buckets[street][COMBO_INDEX[hole[0], hole[1]]], code)
            for name in ("all", hand["bots"][seat]):
                counts.setdefault(name, np.zeros(_SHAPE))[cell] += 1
    return counts


def _in_chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def fit(db_path, hands=15_000, flop_runouts=300, seed=0, processes=None, held_out=None):
    """Count how the players in a self-play database acted with each hand strength -> :class:`ActionModel`.

    One profile per bot name found in the database, plus 'all'. ``held_out=False`` leaves the
    scoring fifth of the hands untouched (see :func:`score`).
    """
    from multiprocessing import Pool

    sample = _load_hands(db_path, hands, seed, held_out)
    jobs = [(chunk, flop_runouts, [seed, i]) for i, chunk in enumerate(_in_chunks(sample, 250))]
    total = {}
    with Pool(processes) as pool:
        for part in pool.imap_unordered(_fit_chunk, jobs):
            for name, counts in part.items():
                total[name] = total.get(name, 0) + counts
    return ActionModel(total, f"{len(sample)} hands from {os.path.basename(db_path)}")


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
            ranges = infer_ranges(model, before, hero_seat, hero, board, profile, 100, np.random.default_rng(np_seed))
            row[label] = equity(hero, board, len(rivals), list(ranges.values()), trials=trials, seed=np_seed).equity
        rows.append(row)
    return rows


METHODS = ("random", "model", "model_profiled")


def score(db_path, model=None, hands=4000, trials=4000, seed=1, processes=None, held_out=None):
    """Equity estimates next to the truth, on one decision point from each of ``hands`` sampled hands.

    Returns a DataFrame with a row per decision: street, opponents, ``truth`` (the acting player's
    equity against the hole cards the opponents really held) and the estimate of each method:
    ``random`` (opponents hold random hands), ``model`` (ranges inferred with the pooled profile)
    and ``model_profiled`` (ranges inferred with each opponent's own profile).
    See :func:`summarize` and :func:`reliability`.
    """
    from multiprocessing import Pool

    import pandas as pd

    model = model or default_model()
    sample = _load_hands(db_path, hands, seed, held_out)
    jobs = [(chunk, model, trials, seed * 1000 + i) for i, chunk in enumerate(_in_chunks(sample, 100))]
    with Pool(processes) as pool:
        return pd.DataFrame([row for part in pool.imap_unordered(_score_chunk, jobs) for row in part])


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
