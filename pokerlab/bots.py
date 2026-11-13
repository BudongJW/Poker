"""Bot players for self-play.

Three bots need no hand reading at all (random, calling station, maniac). The others are one
parameterised :class:`EquityBot` that decides from its estimated win probability: a pre-flop
table lookup, a small Monte Carlo simulation after the flop.
"""

import random

import numpy as np

from .cards import hand_class
from .engine import CALL, CHECK, FOLD, RAISE
from .equity import simulate
from .preflop import preflop_equity, top_range


class Bot:
    name = "bot"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, view):
        raise NotImplementedError

    def _raise_to(self, view, pot_fraction):
        """Raise sized as a fraction of the pot after calling."""
        return view.current_bet + max(view.big_blind, round(pot_fraction * (view.pot + view.to_call)))


class CallingStation(Bot):
    """Never folds, never raises: every hand goes to showdown."""
    name = "station"

    def act(self, view):
        return CALL, 0


class RandomBot(Bot):
    name = "random"

    def act(self, view):
        r = self.rng.random()
        if r < 0.15 and view.to_call > 0:
            return FOLD, 0
        if r < 0.75 or not view.can_raise:
            return CALL, 0
        return RAISE, self._raise_to(view, self.rng.uniform(0.3, 1.0))


class Maniac(Bot):
    """Raises most of the time regardless of cards."""
    name = "maniac"

    def act(self, view):
        r = self.rng.random()
        if r < 0.55 and view.can_raise and view.raises < 3:
            return RAISE, self._raise_to(view, self.rng.uniform(0.5, 1.0))
        if r < 0.95 or view.to_call == 0:
            return CALL, 0
        return FOLD, 0


class EquityBot(Bot):
    """Bets by estimated equity against the opponents still in the hand.

    value:  how far above the fair share (1/players) equity must be to bet or raise, as a
            fraction of the gap between fair share and 1. Lower = looser and more aggressive.
    margin: extra equity demanded on top of pot odds before calling. Higher = tighter.
    bluff:  probability of betting a hand that does not qualify when checked to.
    villain_range: if set (e.g. 0.2), an opponent who has raised in this hand is assumed to hold
            a top-x% starting hand instead of a random one.
    """

    def __init__(self, name, value, margin, bluff, trials=300, villain_range=None, seed=None):
        super().__init__(seed)
        self.name = name
        self.value, self.margin, self.bluff = value, margin, bluff
        self.trials, self.villain_range = trials, villain_range
        self.np_rng = np.random.default_rng(seed)
        self._cache_key = self._cache_value = None

    def estimate(self, view):
        n_opp = view.n_live - 1
        ranged = bool(self.villain_range) and _live_raiser(view)
        key = (view.hole, view.street, n_opp, ranged)
        if key != self._cache_key:
            if view.street == 0 and not ranged:
                value = preflop_equity(hand_class(*view.hole), n_opp)
            else:
                ranges = [top_range(self.villain_range)] if ranged else ()
                share = simulate(list(view.hole), list(view.board), n_opp, self.trials, self.np_rng, ranges)[1]
                value = float(share.mean())
            self._cache_key, self._cache_value = key, value
        return self._cache_value

    def act(self, view):
        eq = self.estimate(view)
        fair = 1.0 / view.n_live
        strong = fair + (1.0 - fair) * (self.value + 0.2 * view.raises)
        if view.can_raise and eq > strong:
            target = self._raise_to(view, 0.5 + 0.5 * self.rng.random())
            if target > 0.6 * view.max_raise_to:         # pot-committed: move all-in
                target = view.max_raise_to
            return RAISE, target, eq
        if view.to_call == 0:
            if view.can_raise and self.rng.random() < self.bluff:
                return RAISE, self._raise_to(view, 0.6), eq
            return CHECK, 0, eq
        pot_odds = view.to_call / (view.pot + view.to_call)
        if eq >= pot_odds + self.margin:
            return CALL, 0, eq
        return FOLD, 0, eq


def _live_raiser(view):
    """Has an opponent who is still in the hand raised at any point?"""
    folded = {a[1] for a in view.actions if a[2] == FOLD}
    return any(a[2] == RAISE and a[1] != view.seat and a[1] not in folded for a in view.actions)


BOT_TYPES = {
    "station": lambda seed: CallingStation(seed),
    "random": lambda seed: RandomBot(seed),
    "maniac": lambda seed: Maniac(seed),
    "nit": lambda seed: EquityBot("nit", value=0.30, margin=0.12, bluff=0.0, seed=seed),
    "tag": lambda seed: EquityBot("tag", value=0.15, margin=0.06, bluff=0.05, seed=seed),
    "lag": lambda seed: EquityBot("lag", value=0.05, margin=0.0, bluff=0.20, seed=seed),
    "tag_range": lambda seed: EquityBot("tag_range", value=0.15, margin=0.06, bluff=0.05,
                                        villain_range=0.2, seed=seed),
}


def make_bot(kind, seed=None):
    if kind not in BOT_TYPES:
        raise ValueError(f"unknown bot {kind!r}; choose from {sorted(BOT_TYPES)}")
    return BOT_TYPES[kind](seed)
