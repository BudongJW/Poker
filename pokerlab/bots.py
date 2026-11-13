"""Bot players for self-play.

Three bots need no hand reading at all (random, calling station, maniac). The others are one
parameterised :class:`EquityBot` that decides from its estimated win probability: a pre-flop
table lookup, a small Monte Carlo simulation after the flop, optionally against opponent ranges
inferred from the betting.
"""

import random

import numpy as np

from .cards import hand_class
from .engine import CALL, CHECK, FOLD, RAISE
from .equity import COMBO_INDEX, Range, quick_equity
from .inference import context, default_model, infer_ranges, load_model, position_now, strength_buckets
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

    value:   how far above the fair share (1/players) equity must be to bet or raise, as a
             fraction of the gap between fair share and 1. Lower = looser and more aggressive.
    margin:  extra equity demanded on top of pot odds before calling. Higher = tighter.
    bluff:   probability of betting a hand that does not qualify when checked to.
    respect: how much each raise already made on the street lifts the bar for raising again.
    villain_range: if set (e.g. 0.2), an opponent who has raised in this hand is assumed to hold
             a top-x% starting hand instead of a random one.
    model:   an :class:`~pokerlab.inference.ActionModel` (or True for the bundled one): every
             opponent's range is inferred from their actions so far. Replaces ``villain_range``.
    """

    def __init__(self, name, value, margin, bluff, respect=0.2, trials=300, villain_range=None, model=None,
                 seed=None):
        super().__init__(seed)
        self.name = name
        self.value, self.margin, self.bluff, self.respect = value, margin, bluff, respect
        self.trials, self.villain_range = trials, villain_range
        self.model = default_model() if model is True else model
        self.np_rng = np.random.default_rng(seed)
        self._cache_key = self._cache_value = None

    def estimate(self, view):
        n_opp = view.n_live - 1
        if self.model:
            if len(_BUCKETS) > 64:
                _BUCKETS.clear()
            ranges = list(infer_ranges(self.model, view.actions, view.seat, view.hole, view.board, view.n_players,
                                       view.button, flop_runouts=60, rng=self.np_rng, cache=_BUCKETS).values())
            if not ranges and view.street == 0:
                return preflop_equity(hand_class(*view.hole), n_opp)
            return quick_equity(view.hole, view.board, n_opp, self.trials, self.np_rng, ranges)
        ranged = bool(self.villain_range) and _live_raiser(view)
        key = (view.hole, tuple(view.board), n_opp, ranged)
        if key != self._cache_key:
            if view.street == 0 and not ranged:
                value = preflop_equity(hand_class(*view.hole), n_opp)
            else:
                ranges = [Range(top_range(self.villain_range))] if ranged else ()
                value = quick_equity(view.hole, view.board, n_opp, self.trials, self.np_rng, ranges)
            self._cache_key, self._cache_value = key, value
        return self._cache_value

    def act(self, view):
        eq = self.estimate(view)
        fair = 1.0 / view.n_live
        strong = fair + (1.0 - fair) * (self.value + self.respect * view.raises)
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


class PopulationBot(Bot):
    """Plays like the players an action model was fitted on.

    In each situation it looks up how often those players folded, called and raised with a hand as
    strong as its own (same strength bucket) and picks one of the three with those frequencies.
    Bet sizes are not part of the model: raises are half to full pot.
    """

    def __init__(self, name, model, profile="all", seed=None):
        super().__init__(seed)
        self.name, self.model, self.profile = name, model, profile
        self.np_rng = np.random.default_rng(seed)

    def act(self, view):
        if len(_BUCKETS) > 64:
            _BUCKETS.clear()
        bucket = strength_buckets(view.board, 60, self.np_rng, _BUCKETS)[COMBO_INDEX[view.hole[0], view.hole[1]]]
        folded = {a[1] for a in view.actions if a[2] == FOLD}
        position = position_now(view.seat, view.button, view.n_players, folded, view.street)
        ctx = context(view.street, view.raises, view.to_call > 0, view.n_live, view.n_players, position)
        fold, passive, _ = (self.model.likelihood(ctx, code, self.profile)[bucket] for code in range(3))
        r = self.rng.random()
        if r < fold:
            return FOLD, 0
        if r < fold + passive or not view.can_raise:
            return CALL, 0
        return RAISE, self._raise_to(view, 0.5 + 0.5 * self.rng.random())


_BUCKETS = {}       # board -> strength bucket of every combo; shared by all model bots of a process


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
    "tag_model": lambda seed: EquityBot("tag_model", value=0.15, margin=0.06, bluff=0.05, model=True, seed=seed),
    # players reconstructed from public hand histories: 2009 online cash games, and Pluribus with its opponents
    "human": lambda seed: PopulationBot("human", load_model("online"), seed=seed),
    "pro": lambda seed: PopulationBot("pro", load_model("pro"), seed=seed),
}


def _champion(seed):
    from .daily import load_champion
    return EquityBot("champion", model=True, seed=seed, **load_champion()["params"])


BOT_TYPES["champion"] = _champion       # the betting parameters the daily cycle has settled on so far


def make_bot(kind, seed=None):
    """A bot by type name, or a custom equity bot written as
    'equity:name=mine;value=0.15;margin=0.06;bluff=0.05;respect=0.2;model=1'.

    ``model=1`` uses the bundled action model, ``model=<path>`` the one stored in that file.
    """
    if kind.startswith("equity:"):
        options = dict(item.split("=", 1) for item in kind[len("equity:"):].split(";") if item)
        name = options.pop("name", "custom")
        model = options.pop("model", "0")
        model = None if model in ("0", "false", "") else True if model in ("1", "true") else load_model(model)
        trials = int(options.pop("trials", 300))
        return EquityBot(name, model=model, trials=trials, seed=seed,
                         **{key: float(value) for key, value in options.items()})
    if kind not in BOT_TYPES:
        raise ValueError(f"unknown bot {kind!r}; choose from {sorted(BOT_TYPES)}")
    return BOT_TYPES[kind](seed)
