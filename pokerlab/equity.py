"""Equity (win probability) of a Hold'em hand against N opponents.

``equity()`` enumerates every possibility exactly when that is cheap (one opponent on the
flop/turn/river, two on the turn/river) and falls back to vectorised Monte Carlo otherwise.
Opponents hold uniformly random hands unless a range (a list of allowed hole-card combos)
is given for them.

Equity counts a tie as a share of the pot (1/number of tied players), not as a win.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .cards import parse_cards
from .evaluator import CATEGORIES, evaluate_many

MC_BATCH = 100_000


@dataclass
class EquityResult:
    win: float                  # P(hero beats everyone)
    tie: float                  # P(hero ties for best)
    equity: float               # expected pot share: win + tie shares
    trials: int                 # simulated or enumerated deals
    exact: bool
    stderr: float               # standard error of ``equity`` (0 when exact)
    n_opponents: int
    hand_dist: dict = field(default_factory=dict)   # hero's final hand category -> probability

    @property
    def lose(self):
        return 1.0 - self.win - self.tie


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


def _sample_ranges(ranges, known, trials, rng):
    """Jointly sample one combo per ranged opponent, conditioned on no shared cards -> (trials, 2*len(ranges))."""
    usable = []
    for combos in ranges:
        combos = np.array([c for c in combos if c[0] not in known and c[1] not in known], dtype=np.int8)
        if len(combos) == 0:
            raise ValueError("an opponent range has no combos left after removing known cards")
        usable.append(combos)
    out = np.empty((0, 2 * len(ranges)), dtype=np.int8)
    attempts = 0
    while len(out) < trials:
        n = max(1024, 2 * (trials - len(out)))
        cand = np.concatenate([c[rng.integers(0, len(c), size=n)] for c in usable], axis=1)
        if len(ranges) > 1:
            srt = np.sort(cand, axis=1)
            cand = cand[(srt[:, 1:] != srt[:, :-1]).all(axis=1)]
        out = np.concatenate([out, cand])
        attempts += 1
        if attempts > 200:
            raise ValueError("opponent ranges are mutually incompatible")
    return out[:trials]


def simulate(hero, board, n_opponents, trials, rng, ranges=(), dead=()):
    """One Monte Carlo batch -> (hero values, pot share per trial, beaten flags, tie flags)."""
    known = set(hero) | set(board) | set(dead)
    deck = [c for c in range(52) if c not in known]
    n_ranged = len(ranges)
    n_random = n_opponents - n_ranged
    need_board = 5 - len(board)
    k = need_board + 2 * n_random

    ranged = _sample_ranges(ranges, known, trials, rng) if n_ranged else None
    draw = _deal(deck, k, trials, rng)
    if n_ranged and k:                      # redraw rows whose random cards hit a ranged hand
        bad = np.flatnonzero((draw[:, :, None] == ranged[:, None, :]).any(axis=(1, 2)))
        while bad.size:
            draw[bad] = _deal(deck, k, bad.size, rng)
            still = (draw[bad][:, :, None] == ranged[bad][:, None, :]).any(axis=(1, 2))
            bad = bad[still]

    hands = np.empty((n_opponents + 1, trials, 7), dtype=np.int8)
    hands[0, :, 0:2] = hero
    if board:
        hands[:, :, 2:2 + len(board)] = board
    hands[:, :, 2 + len(board):] = draw[:, :need_board]
    if n_ranged:
        hands[1:1 + n_ranged, :, 0:2] = ranged.reshape(trials, n_ranged, 2).transpose(1, 0, 2)
    if n_random:
        hands[1 + n_ranged:, :, 0:2] = draw[:, need_board:].reshape(trials, n_random, 2).transpose(1, 0, 2)

    values = evaluate_many(hands.reshape(-1, 7)).reshape(n_opponents + 1, trials)
    hero_v, opp_v = values[0], values[1:]
    beaten = (opp_v > hero_v).any(axis=0)
    n_tied = (opp_v == hero_v).sum(axis=0)
    share = np.where(beaten, 0.0, 1.0 / (1 + n_tied))
    return hero_v, share, beaten, (n_tied > 0) & ~beaten


def _monte_carlo(hero, board, n_opponents, trials, rng, ranges, dead):
    cats = np.zeros(9, dtype=np.int64)
    wins = ties = 0
    total = sq = 0.0
    done = 0
    while done < trials:
        n = min(MC_BATCH, trials - done)
        hero_v, share, beaten, tied = simulate(hero, board, n_opponents, n, rng, ranges, dead)
        cats += np.bincount(hero_v >> 20, minlength=9)
        ties += int(tied.sum())
        wins += int(n - beaten.sum() - tied.sum())
        total += float(share.sum())
        sq += float((share * share).sum())
        done += n
    mean = total / done
    var = max(sq / done - mean * mean, 0.0)
    return EquityResult(wins / done, ties / done, mean, done, False, (var / done) ** 0.5, n_opponents,
                        {CATEGORIES[i]: float(cats[i] / done) for i in range(9) if cats[i]})


def _exact(hero, board, n_opponents, dead, opp_range=None):
    """Full enumeration of runouts x opponent hands (1 or 2 opponents)."""
    known = set(hero) | set(board) | set(dead)
    deck = [c for c in range(52) if c not in known]
    if opp_range is None:
        pairs = np.array(list(combinations(deck, 2)), dtype=np.int8)
    else:
        pairs = np.array([c for c in opp_range if c[0] not in known and c[1] not in known], dtype=np.int8)
    need_board = 5 - len(board)
    runouts = list(combinations(deck, need_board))          # one empty runout on the river
    runouts = np.array(runouts, dtype=np.int8).reshape(len(runouts), need_board)
    n_run, n_pairs = len(runouts), len(pairs)

    boards = np.empty((n_run, 5), dtype=np.int8)
    boards[:, :len(board)] = board
    boards[:, len(board):] = runouts
    hero_hands = np.concatenate([np.tile(np.array(hero, dtype=np.int8), (n_run, 1)), boards], axis=1)
    hero_v = evaluate_many(hero_hands)

    opp_hands = np.empty((n_run, n_pairs, 7), dtype=np.int8)
    opp_hands[:, :, 0:2] = pairs
    opp_hands[:, :, 2:] = boards[:, None, :]
    # an opponent hand is impossible on runouts that contain one of its cards
    valid = ~(pairs[None, :, :, None] == runouts[:, None, None, :]).any(axis=(2, 3))
    opp_v = np.zeros((n_run, n_pairs), dtype=np.int32)
    opp_v[valid] = evaluate_many(opp_hands[valid])

    if n_opponents == 1:
        hv = hero_v[:, None]
        n_deals = int(valid.sum())
        wins = int(((opp_v < hv) & valid).sum())
        ties = int(((opp_v == hv) & valid).sum())
        share = wins + ties / 2
        weight = valid.sum(axis=1)            # a runout weighs as many deals as it leaves opponent hands
    else:
        disjoint = ~(pairs[:, None, :, None] == pairs[None, :, None, :]).any(axis=(2, 3))
        n_deals = wins = ties = 0
        share = 0.0
        weight = np.zeros(n_run, dtype=np.int64)
        for r in range(n_run):
            ok = disjoint & valid[r][:, None] & valid[r][None, :]
            v1, v2 = opp_v[r][:, None], opp_v[r][None, :]
            best = np.maximum(v1, v2)
            n_tied = (v1 == hero_v[r]).astype(np.int8) + (v2 == hero_v[r])
            win = ok & (best < hero_v[r])
            tie = ok & (best == hero_v[r])
            weight[r] = ok.sum()
            n_deals += int(weight[r])
            wins += int(win.sum())
            ties += int(tie.sum())
            share += float(win.sum() + (tie / (1 + n_tied)).sum())
    if n_deals == 0:
        raise ValueError("no possible deals: the opponent range conflicts with the known cards")
    cats = np.bincount(hero_v >> 20, weights=weight, minlength=9) / weight.sum()
    return EquityResult(wins / n_deals, ties / n_deals, share / n_deals, n_deals, True, 0.0, n_opponents,
                        {CATEGORIES[i]: float(cats[i]) for i in range(9) if cats[i]})


def equity(hero, board=(), n_opponents=1, ranges=(), dead=(), trials=100_000, exact="auto", seed=None):
    """Hero's equity with ``board`` (0/3/4/5 cards) against ``n_opponents``.

    ranges: per-opponent lists of (card, card) combos for the opponents whose range is known;
            the remaining opponents hold random hands.
    exact:  'auto' enumerates when the deal space is small, True forces it, False forces Monte Carlo.
    """
    hero, board, dead = parse_cards(hero), parse_cards(board), parse_cards(dead)
    if len(hero) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("need 2 hole cards and 0, 3, 4 or 5 board cards")
    if len(set(hero + board + dead)) != len(hero + board + dead):
        raise ValueError("the same card appears twice")
    if not 1 <= n_opponents <= 22 or len(ranges) > n_opponents:
        raise ValueError("bad number of opponents or ranges")

    need_board = 5 - len(board)
    can_enumerate = (not ranges or (n_opponents == 1 and len(ranges) == 1)) and (
        (n_opponents == 1 and need_board <= 2) or (n_opponents == 2 and need_board <= 1))
    if exact is True and not can_enumerate:
        raise ValueError("exact enumeration supports 1 opponent post-flop or 2 opponents on turn/river")
    if can_enumerate and exact in (True, "auto"):
        return _exact(hero, board, n_opponents, dead, ranges[0] if ranges else None)
    return _monte_carlo(hero, board, n_opponents, trials, np.random.default_rng(seed), list(ranges), dead)
