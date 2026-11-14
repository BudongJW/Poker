"""Equity (expected share of the pot at showdown) in Hold'em.

* :func:`equity`     one hand against N opponents who hold random hands or ranges
* :func:`matchup`    every player's equity at once (known hands, ranges or random hands)
* :func:`next_card`  the hero's equity after each card that can come next

Results are exact whenever the deals can be enumerated cheaply (all hole cards known; one
unknown opponent after the flop; two on the turn or river) and Monte Carlo otherwise, either
for a fixed number of trials or until a requested precision is reached.

A tie counts as a share of the pot (1 / number of tied players), not as a win.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import chain, combinations
from math import comb

import numpy as np

from .cards import parse_cards
from .evaluator import CATEGORIES, evaluate_many, evaluate_pairs

MC_BATCH = 100_000

# Every two-card combo, higher card first. A 1,326-long weight vector is aligned with these rows.
ALL_COMBOS = np.array([(a, b) for a in range(51, 0, -1) for b in range(a - 1, -1, -1)], dtype=np.int8)
COMBO_INDEX = np.zeros((52, 52), dtype=np.int16)
COMBO_INDEX[ALL_COMBOS[:, 0], ALL_COMBOS[:, 1]] = COMBO_INDEX[ALL_COMBOS[:, 1], ALL_COMBOS[:, 0]] = np.arange(1326)


class Range:
    """The hole-card combos a player may hold, each with a relative weight."""

    def __init__(self, combos, weights=None):
        self.combos = np.asarray(combos, dtype=np.int8).reshape(-1, 2)
        self.weights = np.ones(len(self.combos)) if weights is None else np.asarray(weights, dtype=np.float64)
        if len(self.weights) != len(self.combos) or (self.weights < 0).any():
            raise ValueError("a range needs one non-negative weight per combo")

    @classmethod
    def from_vector(cls, weights):
        """From one weight per row of :data:`ALL_COMBOS`."""
        weights = np.asarray(weights, dtype=np.float64)
        keep = weights > 0
        return cls(ALL_COMBOS[keep], weights[keep])

    def vector(self):
        out = np.zeros(1326)
        np.add.at(out, COMBO_INDEX[self.combos[:, 0], self.combos[:, 1]], self.weights)
        return out

    def without(self, cards):
        """The part of the range that uses none of ``cards`` (and has positive weight)."""
        keep = ~np.isin(self.combos, list(cards)).any(axis=1) & (self.weights > 0)
        return Range(self.combos[keep], self.weights[keep])

    def __len__(self):
        return len(self.combos)


def as_range(spec):
    """Range, {combo: weight} or an iterable of combos -> :class:`Range`."""
    if isinstance(spec, Range):
        return spec
    if isinstance(spec, dict):
        return Range(list(spec), list(spec.values()))
    return Range(list(spec))


@dataclass
class EquityResult:
    win: float                  # P(this player beats everyone)
    tie: float                  # P(this player ties for best)
    equity: float               # expected pot share: win + tie shares
    trials: int                 # simulated or enumerated deals
    exact: bool
    stderr: float               # standard error of ``equity`` (0 when exact)
    n_opponents: int
    hand_dist: dict = field(default_factory=dict)   # final hand category -> probability

    @property
    def lose(self):
        return 1.0 - self.win - self.tie

    @property
    def ci95(self):
        """Half-width of the 95% confidence interval of ``equity``."""
        return 1.96 * self.stderr


# ---- dealing ---------------------------------------------------------------------------

def _deal(deck, k, trials, rng):
    """``trials`` independent uniform draws of ``k`` distinct cards from ``deck`` (partial Fisher-Yates)."""
    n = len(deck)
    cols = np.tile(np.asarray(deck, dtype=np.int8)[:, None], (1, trials))   # cols[i] is contiguous
    rows = np.arange(trials)
    for i in range(k):
        j = rng.integers(i, n, size=trials)
        picked = cols[j, rows]
        cols[j, rows] = cols[i]
        cols[i] = picked
    return cols[:k].T


def _sample_ranges(ranges, trials, rng):
    """One combo per ranged player and trial, drawn by weight, conditioned on no card being used twice."""
    cumulative = [np.cumsum(r.weights) for r in ranges]
    out = np.empty((0, 2 * len(ranges)), dtype=np.int8)
    for _ in range(200):
        n = max(1024, 2 * (trials - len(out)))
        picks = [np.minimum(np.searchsorted(c, rng.random(n) * c[-1], side="right"), len(c) - 1) for c in cumulative]
        cand = np.concatenate([r.combos[i] for r, i in zip(ranges, picks)], axis=1)
        if len(ranges) > 1:
            srt = np.sort(cand, axis=1)
            cand = cand[(srt[:, 1:] != srt[:, :-1]).all(axis=1)]
        out = np.concatenate([out, cand])
        if len(out) >= trials:
            return out[:trials]
    raise ValueError("the ranges cannot all be held at once")


def simulate(players, board, trials, rng, dead=()):
    """One Monte Carlo batch -> hand values of shape (len(players), trials).

    A player is a pair of known cards, a :class:`Range`, or None for a random hand.
    """
    known = set(board) | set(dead)
    fixed, ranged, unknown = [], [], []
    for i, player in enumerate(players):
        if player is None:
            unknown.append(i)
        elif isinstance(player, Range):
            ranged.append(i)
        else:
            fixed.append(i)
            known.update(player)
    deck = [c for c in range(52) if c not in known]
    need_board = 5 - len(board)
    k = need_board + 2 * len(unknown)

    hands = np.empty((len(players), trials, 7), dtype=np.int8)
    for i in fixed:
        hands[i, :, 0:2] = players[i]
    sampled = None
    if ranged:
        usable = [players[i].without(known) for i in ranged]
        if any(len(r) == 0 for r in usable):
            raise ValueError("a range has no combos left after removing known cards")
        sampled = _sample_ranges(usable, trials, rng)
        hands[ranged, :, 0:2] = sampled.reshape(trials, len(ranged), 2).transpose(1, 0, 2)
    draw = _deal(deck, k, trials, rng)
    if ranged and k:                        # redraw rows whose random cards hit a sampled range hand
        bad = np.flatnonzero((draw[:, :, None] == sampled[:, None, :]).any(axis=(1, 2)))
        while bad.size:
            draw[bad] = _deal(deck, k, bad.size, rng)
            bad = bad[(draw[bad][:, :, None] == sampled[bad][:, None, :]).any(axis=(1, 2))]
    if board:
        hands[:, :, 2:2 + len(board)] = board
    hands[:, :, 2 + len(board):] = draw[:, :need_board]
    if unknown:
        hands[unknown, :, 0:2] = draw[:, need_board:].reshape(trials, len(unknown), 2).transpose(1, 0, 2)
    return evaluate_many(hands.reshape(-1, 7)).reshape(len(players), trials)


def quick_equity(hero, board, n_opponents, trials, rng, ranges=()):
    """Hero's average pot share from one small batch. For bots: no validation, no statistics."""
    values = simulate([hero] + list(ranges) + [None] * (n_opponents - len(ranges)), board, trials, rng)
    winners = values == values.max(axis=0)
    return float((winners[0] / winners.sum(axis=0)).mean())


# ---- turning hand values into results --------------------------------------------------

def _tally(values):
    """values (players, deals) -> per player [sole wins, ties, share sum, share^2 sum] and category counts."""
    winners = values == values.max(axis=0)
    n_winners = winners.sum(axis=0)
    share = winners / n_winners
    sole = (winners & (n_winners == 1)).sum(axis=1)
    sums = np.array([sole, winners.sum(axis=1) - sole, share.sum(axis=1), (share * share).sum(axis=1)])
    return sums, np.stack([np.bincount(v >> 20, minlength=9) for v in values])


def _results(sums, cats, deals, exact):
    out = []
    for i in range(sums.shape[1]):
        mean = sums[2, i] / deals
        stderr = 0.0 if exact else (max(sums[3, i] / deals - mean * mean, 0.0) / deals) ** 0.5
        out.append(EquityResult(float(sums[0, i] / deals), float(sums[1, i] / deals), float(mean), int(deals),
                                exact, float(stderr), sums.shape[1] - 1,
                                {CATEGORIES[c]: float(cats[i, c] / deals) for c in range(9) if cats[i, c]}))
    return out


def _monte_carlo(players, board, dead, trials, rng, precision, max_trials, watch):
    """Fixed ``trials``, or batches until the first ``watch`` players' 95% interval is within ``precision``."""
    sums = np.zeros((4, len(players)))
    cats = np.zeros((len(players), 9), dtype=np.int64)
    target = max_trials if precision else trials
    done = 0
    while done < target:
        n = min(MC_BATCH, target - done)
        batch_sums, batch_cats = _tally(simulate(players, board, n, rng, dead))
        sums += batch_sums
        cats += batch_cats
        done += n
        if precision:
            mean = sums[2, :watch] / done
            stderr = (np.maximum(sums[3, :watch] / done - mean * mean, 0.0) / done) ** 0.5
            if 1.96 * stderr.max() <= precision:
                break
    return _results(sums, cats, done, False)


# ---- exact enumeration -----------------------------------------------------------------

@lru_cache(maxsize=4)
def _index_combos(n, k):
    """All k-subsets of range(n) as a (C(n, k), k) int8 array."""
    if k == 0:
        return np.empty((1, 0), dtype=np.int8)
    flat = np.fromiter(chain.from_iterable(combinations(range(n), k)), dtype=np.int8, count=comb(n, k) * k)
    return flat.reshape(-1, k)


def _boards(deck, board):
    """Every completion of ``board`` to five cards from ``deck`` -> (boards (r, 5), added cards (r, k))."""
    runouts = np.asarray(deck, dtype=np.int8)[_index_combos(len(deck), 5 - len(board))]
    boards = np.empty((len(runouts), 5), dtype=np.int8)
    boards[:, :len(board)] = board
    boards[:, len(board):] = runouts
    return boards, runouts


def _exact_known(hands, board, dead):
    """Everyone's hole cards are known: enumerate the rest of the board."""
    known = set(board) | set(dead) | {c for hand in hands for c in hand}
    boards, _ = _boards([c for c in range(52) if c not in known], board)
    full = np.empty((len(boards), 7), dtype=np.int8)
    full[:, 2:] = boards
    values = np.empty((len(hands), len(boards)), dtype=np.int32)
    for i, hand in enumerate(hands):
        full[:, 0:2] = hand
        values[i] = evaluate_many(full)
    return _results(*_tally(values), len(boards), True)


def _exact_unknown(hero, board, n_opponents, dead, opp_range=None):
    """Hero against one or two unknown opponents: enumerate runouts x opponent hands."""
    known = set(hero) | set(board) | set(dead)
    deck = [c for c in range(52) if c not in known]
    if opp_range is None:
        pairs = np.asarray(deck, dtype=np.int8)[_index_combos(len(deck), 2)]
        weights = np.ones(len(pairs))
    else:
        pairs, weights = opp_range.combos, opp_range.weights
    boards, _ = _boards(deck, board)
    n_run = len(boards)

    hero_v = evaluate_many(np.concatenate([np.tile(np.array(hero, dtype=np.int8), (n_run, 1)), boards], axis=1))
    opp_v = evaluate_pairs(boards, pairs)
    valid = opp_v >= 0          # an opponent hand is impossible on runouts that contain one of its cards

    if n_opponents == 1:
        hv = hero_v[:, None]
        weight = valid * weights                    # (runouts, opponent hands)
        total = weight.sum()
        wins = (weight * (opp_v < hv)).sum()
        ties = (weight * (opp_v == hv)).sum()
        share = wins + ties / 2
        n_deals = int(valid.sum())
        per_runout = weight.sum(axis=1)             # a runout weighs as much as the hands it leaves possible
    else:
        disjoint = ~(pairs[:, None, :, None] == pairs[None, :, None, :]).any(axis=(2, 3))
        n_deals = wins = ties = 0
        share = 0.0
        per_runout = np.zeros(n_run)
        for r in range(n_run):
            ok = disjoint & valid[r][:, None] & valid[r][None, :]
            v1, v2 = opp_v[r][:, None], opp_v[r][None, :]
            best = np.maximum(v1, v2)
            n_tied = (v1 == hero_v[r]).astype(np.int8) + (v2 == hero_v[r])
            win = ok & (best < hero_v[r])
            tie = ok & (best == hero_v[r])
            per_runout[r] = ok.sum()
            n_deals += int(per_runout[r])
            wins += int(win.sum())
            ties += int(tie.sum())
            share += float(win.sum() + (tie / (1 + n_tied)).sum())
        total = n_deals
    if total <= 0:
        raise ValueError("no possible deals: the opponent range conflicts with the known cards")
    cats = np.bincount(hero_v >> 20, weights=per_runout, minlength=9) / per_runout.sum()
    return EquityResult(float(wins / total), float(ties / total), float(share / total), n_deals, True, 0.0,
                        n_opponents, {CATEGORIES[i]: float(cats[i]) for i in range(9) if cats[i]})


# ---- public API ------------------------------------------------------------------------

def equity(hero, board=(), n_opponents=1, ranges=(), dead=(), trials=100_000, exact="auto", seed=None,
           precision=None, max_trials=20_000_000):
    """Hero's equity with ``board`` (0/3/4/5 cards) against ``n_opponents``.

    ranges:    one entry per opponent whose range is known (:class:`Range`, list of combos or
               {combo: weight}); the remaining opponents hold random hands. A one-combo range is
               an opponent whose cards are known.
    exact:     'auto' enumerates when the deal space is small, True forces it, False forces Monte Carlo.
    precision: if set (e.g. 0.001 for +-0.1 percentage points), Monte Carlo keeps going until the 95%
               confidence interval is that narrow, up to ``max_trials``; ``trials`` is then ignored.
    """
    hero, board, dead = parse_cards(hero), parse_cards(board), parse_cards(dead)
    if len(hero) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("need 2 hole cards and 0, 3, 4 or 5 board cards")
    known = set(hero + board + dead)
    if len(known) != len(hero + board + dead):
        raise ValueError("the same card appears twice")
    if not 1 <= n_opponents <= 22 or len(ranges) > n_opponents:
        raise ValueError("bad number of opponents or ranges")
    ranges = [as_range(r).without(known) for r in ranges]
    if any(len(r) == 0 for r in ranges):
        raise ValueError("an opponent range has no combos left after removing known cards")

    need_board = 5 - len(board)
    if exact is not False:
        pinned = [tuple(r.combos[0].tolist()) for r in ranges if len(r) == 1]
        if len(pinned) == n_opponents and len({c for hand in pinned for c in hand}) == 2 * n_opponents:
            return _exact_known([tuple(hero)] + pinned, board, dead)[0]
        if (n_opponents == 1 and need_board <= 2) or (n_opponents == 2 and not ranges and need_board <= 1):
            return _exact_unknown(hero, board, n_opponents, dead, ranges[0] if ranges else None)
        if exact is True:
            raise ValueError("exact enumeration needs known hands, 1 opponent post-flop or 2 on the turn/river")
    players = [tuple(hero)] + ranges + [None] * (n_opponents - len(ranges))
    return _monte_carlo(players, board, dead, trials, np.random.default_rng(seed), precision, max_trials, 1)[0]


def matchup(players, board=(), dead=(), trials=200_000, exact="auto", seed=None, precision=None,
            max_trials=20_000_000):
    """Equity of every player -> list of :class:`EquityResult`, in the order given.

    A player is two known cards ('AsKs' or two card ints), a range (:class:`Range`, list of combos,
    {combo: weight}) or None for a random hand. Exact when every hand is known.
    """
    board, dead = parse_cards(board), parse_cards(dead)
    if len(board) not in (0, 3, 4, 5) or not 2 <= len(players) <= 23:
        raise ValueError("need 2-23 players and 0, 3, 4 or 5 board cards")
    specs = []
    for player in players:
        if player is None or isinstance(player, (Range, dict)):
            specs.append(player if player is None else as_range(player))
        elif isinstance(player, str) or (len(player) == 2 and all(isinstance(c, (int, np.integer)) for c in player)):
            hand = parse_cards(player)
            if len(hand) != 2:
                raise ValueError(f"not a two-card hand: {player!r}")
            specs.append(tuple(hand))
        else:
            specs.append(as_range(player))
    fixed = [c for s in specs if isinstance(s, tuple) for c in s]
    known = set(fixed + board + dead)
    if len(known) != len(fixed + board + dead):
        raise ValueError("the same card appears twice")
    specs = [s.without(known) if isinstance(s, Range) else s for s in specs]
    if any(isinstance(s, Range) and len(s) == 0 for s in specs):
        raise ValueError("a range has no combos left after removing known cards")
    if exact is not False and all(isinstance(s, tuple) for s in specs):
        return _exact_known(specs, board, dead)
    if exact is True:
        raise ValueError("exact enumeration needs every hand to be known")
    return _monte_carlo(specs, board, dead, trials, np.random.default_rng(seed), precision, max_trials, len(specs))


def next_card(hero, board, n_opponents=1, ranges=(), dead=(), trials=20_000, seed=None):
    """Hero's equity after each card that can come next (on the flop or turn) -> {card: EquityResult}."""
    hero, board, dead = parse_cards(hero), parse_cards(board), parse_cards(dead)
    if len(board) not in (3, 4):
        raise ValueError("next_card needs a flop or turn board")
    ranges = [as_range(r) for r in ranges]
    out = {}
    for card in range(52):
        if card in hero or card in board or card in dead:
            continue
        if any(len(r.without(hero + board + dead + [card])) == 0 for r in ranges):
            continue                        # an opponent is known to hold this card
        out[card] = equity(hero, board + [card], n_opponents, ranges, dead, trials=trials, seed=seed)
    return out
