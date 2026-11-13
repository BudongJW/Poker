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
"""

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


RANK_TABLE = _build_rank_table()
FLUSH_TABLE = _build_flush_table()

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
_NP_RANK_KEYS = np.array(sorted(RANK_TABLE), dtype=np.int64)
_NP_RANK_VALS = np.array([RANK_TABLE[k] for k in sorted(RANK_TABLE)], dtype=np.int32)
_NP_FLUSH = np.array(FLUSH_TABLE, dtype=np.int32)
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
