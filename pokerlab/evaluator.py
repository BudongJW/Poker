"""Hand evaluator for 5-7 cards, scalar (pure Python) and vectorised (NumPy).

A hand value is an int: ``category << 20 | five 4-bit tiebreak ranks``. Higher wins,
equal values split the pot.

How it works: suits only matter for flushes, so a hand is looked up twice.
 * Rank part: the multiset of ranks is encoded as a base-5 number (sum of ``5**rank``;
   at most four cards per rank, so the encoding is unique) and looked up in a table
   holding every 5/6/7-card rank multiset (73,775 entries).
 * Flush part: if some suit has five or more cards, the 13-bit mask of that suit's
   ranks indexes an 8,192-entry table. With at most seven cards a flush excludes a
   full house or quads, so the flush table value is final whenever it applies.

Building the rank table takes about 0.6 s, so it is cached under ``~/.cache/pokerlab``.
"""

import os

import numpy as np

CATEGORIES = ("high card", "pair", "two pair", "three of a kind", "straight",
              "flush", "full house", "four of a kind", "straight flush")
CATEGORIES_KO = ("하이카드", "원페어", "투페어", "트리플", "스트레이트",
                 "플러시", "풀하우스", "포카드", "스트레이트 플러시")

_POW5 = [5 ** r for r in range(13)]


def _value(category, *ranks):
    v = category
    for i in range(5):
        v = (v << 4) | (ranks[i] if i < len(ranks) else 0)
    return v


def _top_straight(present):
    """Highest straight in a set of ranks -> rank of its top card, or -1. The wheel tops out at the five."""
    for hi in range(12, 3, -1):
        if all(r in present for r in range(hi - 4, hi + 1)):
            return hi
    if all(r in present for r in (12, 0, 1, 2, 3)):
        return 3
    return -1


def _rank_value(counts):
    """Best five-card value ignoring suits, from per-rank counts of 5-7 cards."""
    present = [r for r in range(12, -1, -1) if counts[r]]
    quads = [r for r in present if counts[r] == 4]
    trips = [r for r in present if counts[r] == 3]
    pairs = [r for r in present if counts[r] == 2]
    if quads:
        return _value(7, quads[0], max(r for r in present if r != quads[0]))
    if trips and (len(trips) > 1 or pairs):
        return _value(6, trips[0], max(trips[1:2] + pairs[:1]))
    straight = _top_straight(set(present))
    if straight >= 0:
        return _value(4, straight)
    if trips:
        return _value(3, trips[0], *[r for r in present if r != trips[0]][:2])
    if len(pairs) >= 2:
        return _value(2, pairs[0], pairs[1], max(r for r in present if r not in pairs[:2]))
    if pairs:
        return _value(1, pairs[0], *[r for r in present if r != pairs[0]][:3])
    return _value(0, *present[:5])


def _build_rank_table():
    table = {}
    counts = [0] * 13

    def rec(rank, left, key):
        if rank == 13:
            if left <= 2:                      # 5, 6 or 7 cards placed
                table[key] = _rank_value(counts)
            return
        for c in range(min(4, left) + 1):
            counts[rank] = c
            rec(rank + 1, left - c, key + c * _POW5[rank])
        counts[rank] = 0

    rec(0, 7, 0)
    return table


def _build_flush_table():
    table = [0] * 8192
    for mask in range(8192):
        ranks = [r for r in range(12, -1, -1) if mask >> r & 1]
        if len(ranks) >= 5:
            straight = _top_straight(set(ranks))
            table[mask] = _value(8, straight) if straight >= 0 else _value(5, *ranks[:5])
    return table


_RANK_ENTRIES = 73_775


def _load_tables():
    """(sorted rank keys, their values, flush table) as arrays, from the disk cache when it is usable."""
    path = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                        "pokerlab", "evaluator-v1.npz")
    try:
        with np.load(path) as cached:
            keys, values, flush = cached["keys"], cached["values"], cached["flush"]
        if len(keys) == len(values) == _RANK_ENTRIES and len(flush) == 8192:
            return keys, values, flush
    except Exception:           # missing, truncated or foreign file: rebuild
        pass
    table = _build_rank_table()
    keys = np.array(sorted(table), dtype=np.int64)
    values = np.array([table[k] for k in keys.tolist()], dtype=np.int32)
    flush = np.array(_build_flush_table(), dtype=np.int32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = f"{path}.{os.getpid()}.npz"
        np.savez(partial, keys=keys, values=values, flush=flush)
        os.replace(partial, path)           # atomic, so a concurrent reader never sees half a file
    except OSError:
        pass                                # read-only home: just rebuild next time
    return keys, values, flush


_NP_RANK_KEYS, _NP_RANK_VALS, _NP_FLUSH = _load_tables()
RANK_TABLE = dict(zip(_NP_RANK_KEYS.tolist(), _NP_RANK_VALS.tolist()))
FLUSH_TABLE = _NP_FLUSH.tolist()

_CARD_KEY = [_POW5[c >> 2] for c in range(52)]
_CARD_BIT = [1 << (c >> 2) for c in range(52)]


def evaluate(cards):
    """Value of the best five-card hand among 5-7 card ints."""
    key = 0
    suits = [0, 0, 0, 0]
    for c in cards:
        key += _CARD_KEY[c]
        suits[c & 3] |= _CARD_BIT[c]
    for mask in suits:
        if FLUSH_TABLE[mask]:
            return FLUSH_TABLE[mask]
    return RANK_TABLE[key]


# ---- vectorised ------------------------------------------------------------------------
# One gather+sum yields both lookups: the low 32 bits carry the base-5 rank key, the bits
# above carry per-suit card counts in base 8 (seven cards never overflow a base-8 digit).
_NP_CARD_KEY = np.array([_POW5[c >> 2] + (8 ** (c & 3) << 32) for c in range(52)], dtype=np.int64)
_NP_FLUSH_SUIT = np.full(8 ** 4, -1, dtype=np.int8)       # suit-count key -> flush suit or -1
for _key in range(8 ** 4):
    for _suit in range(4):
        if (_key >> (3 * _suit)) & 7 >= 5:
            _NP_FLUSH_SUIT[_key] = _suit


def evaluate_many(hands):
    """Vectorised :func:`evaluate`: int array of shape (m, 5..7) -> int32 values of shape (m,)."""
    hands = np.asarray(hands)
    keys = _NP_CARD_KEY[hands].sum(axis=1)
    values = _NP_RANK_VALS[np.searchsorted(_NP_RANK_KEYS, keys & 0xFFFFFFFF)]
    flush_suit = _NP_FLUSH_SUIT[keys >> 32]
    rows = np.flatnonzero(flush_suit >= 0)
    if rows.size:
        sub = hands[rows].astype(np.int64)
        in_suit = (sub & 3) == flush_suit[rows, None]
        values[rows] = _NP_FLUSH[((1 << (sub >> 2)) * in_suit).sum(axis=1)]
    return values


def evaluate_pairs(boards, pairs):
    """Value of every hole-card pair on every five-card board -> int32 (boards, pairs); -1 where a
    pair uses a board card.

    Much faster than :func:`evaluate_many` on the expanded hands: the board's part of both lookup
    keys is computed once per board and each pair only adds its own two cards.
    """
    boards, pairs = np.asarray(boards, dtype=np.intp), np.asarray(pairs, dtype=np.intp)
    keys = _NP_CARD_KEY[boards].sum(axis=1)[:, None] + (_NP_CARD_KEY[pairs[:, 0]] + _NP_CARD_KEY[pairs[:, 1]])[None, :]
    index = np.minimum(np.searchsorted(_NP_RANK_KEYS, keys & 0xFFFFFFFF), len(_NP_RANK_KEYS) - 1)
    values = _NP_RANK_VALS[index]
    flush_suit = _NP_FLUSH_SUIT[np.minimum(keys >> 32, len(_NP_FLUSH_SUIT) - 1)]
    row, col = np.nonzero(flush_suit >= 0)
    if row.size:
        suit = flush_suit[row, col].astype(np.intp)
        suits = np.arange(4)[:, None]
        board_bits = ((1 << (boards >> 2))[:, None, :] * ((boards & 3)[:, None, :] == suits[None, :, :])).sum(axis=2)
        pair_bits = ((1 << (pairs >> 2)).T[None, :, :] * ((pairs & 3).T[None, :, :] == suits[:, None, :])).sum(axis=1)
        values[row, col] = _NP_FLUSH[board_bits[row, suit] | pair_bits[suit, col]]
    on_board = np.zeros((len(boards), 52), dtype=bool)
    on_board[np.arange(len(boards))[:, None], boards] = True
    values[on_board[:, pairs[:, 0]] | on_board[:, pairs[:, 1]]] = -1
    return values


def rank_counts(values):
    """Row-wise ranking of an int array (rows, n) -> (below, equal, rank), each of shape (rows, n).

    ``below`` counts the entries of the row that are strictly smaller, ``equal`` the *other* entries
    with the same value, and ``rank`` is a dense rank (equal values share a rank).
    """
    n = values.shape[1]
    order = np.argsort(values, axis=1)
    ordered = np.take_along_axis(values, order, axis=1)
    starts = np.ones(ordered.shape, dtype=bool)
    starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    ends = np.ones(ordered.shape, dtype=bool)
    ends[:, :-1] = starts[:, 1:]
    position = np.arange(n)
    group_start = np.maximum.accumulate(np.where(starts, position, 0), axis=1)
    group_end = np.minimum.accumulate(np.where(ends, position, n)[:, ::-1], axis=1)[:, ::-1]
    below, equal, rank = (np.empty(values.shape, dtype=np.int32) for _ in range(3))
    np.put_along_axis(below, order, group_start, axis=1)
    np.put_along_axis(equal, order, group_end - group_start, axis=1)
    np.put_along_axis(rank, order, np.cumsum(starts, axis=1) - 1, axis=1)
    return below, equal, rank
