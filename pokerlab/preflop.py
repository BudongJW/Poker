"""Pre-flop lookup table (169 starting hands x 1..9 random opponents) and hand ranges."""

import json
import os
import re
from functools import lru_cache

from .cards import ALL_CLASSES, RANKS, class_combos, parse_cards

TABLE_PATH = os.path.join(os.path.dirname(__file__), "data", "preflop_equity.json")
MAX_OPPONENTS = 9
DEALS_PER_COMBO = 1225 * 1_712_304      # opposing hands x boards for one specific pair of hole cards


@lru_cache(maxsize=1)
def _data():
    with open(TABLE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def table():
    """{'AKs': [equity vs 1 opponent, vs 2, ..., vs 9]} for all 169 classes."""
    return _data()["equity"]


def preflop_equity(cls, n_opponents):
    return table()[cls][min(n_opponents, MAX_OPPONENTS) - 1]


def preflop_stderr(cls, n_opponents):
    """Standard error of :func:`preflop_equity`: 0 heads-up (enumerated), Monte Carlo error otherwise."""
    return _data()["stderr"][cls][min(n_opponents, MAX_OPPONENTS) - 1]


def heads_up_exact(cls):
    """Exact (win, tie, lose) probabilities of a starting hand against one random hand."""
    wins, ties = _data()["heads_up_counts"][cls]
    total = len(class_combos(cls)) * DEALS_PER_COMBO
    return wins / total, ties / total, (total - wins - ties) / total


@lru_cache(maxsize=None)
def ranking(n_opponents=1):
    """The 169 classes, strongest first, by all-in equity against ``n_opponents`` random hands."""
    return sorted(ALL_CLASSES, key=lambda c: -preflop_equity(c, n_opponents))


@lru_cache(maxsize=None)
def top_range(fraction, n_opponents=1):
    """Combos of the strongest classes covering at least ``fraction`` of the 1,326 starting hands."""
    combos = []
    for cls in ranking(n_opponents):
        if len(combos) >= fraction * 1326:
            break
        combos.extend(class_combos(cls))
    return tuple(combos)


def _expand(token):
    """One range token -> list of classes. Supports 'QQ', 'AK', 'AKs', 'TT+', 'A9s+', '22-77', 'A2s-A5s'."""
    m = re.fullmatch(r"([2-9TJQKA])([2-9TJQKA])([so]?)(\+?)", token)
    if m:
        hi, lo = sorted((RANKS.index(m.group(1)), RANKS.index(m.group(2))), reverse=True)
        kinds = [""] if hi == lo else ([m.group(3)] if m.group(3) else ["s", "o"])
        if not m.group(4):
            lows = [lo]
        elif hi == lo:
            return [RANKS[r] * 2 for r in range(hi, 13)]
        else:
            lows = range(lo, hi)
        return [RANKS[hi] + RANKS[l] + k for l in lows for k in kinds]
    m = re.fullmatch(r"([2-9TJQKA])([2-9TJQKA])([so]?)-([2-9TJQKA])([2-9TJQKA])([so]?)", token)
    if m and m.group(3) == m.group(6):
        a, b = (RANKS.index(m.group(1)), RANKS.index(m.group(2))), (RANKS.index(m.group(4)), RANKS.index(m.group(5)))
        if a[0] == a[1] and b[0] == b[1]:
            return [RANKS[r] * 2 for r in range(min(a[0], b[0]), max(a[0], b[0]) + 1)]
        if a[0] == b[0] and a[0] > max(a[1], b[1]):
            kinds = [m.group(3)] if m.group(3) else ["s", "o"]
            return [RANKS[a[0]] + RANKS[l] + k for l in range(min(a[1], b[1]), max(a[1], b[1]) + 1) for k in kinds]
    raise ValueError(f"cannot parse range token {token!r}")


def parse_range(text):
    """'TT+, AQs+, KQo, 15%' or explicit combos like 'AsKs' -> list of (card, card) combos.

    A percentage means the top x% of starting hands by heads-up equity.
    """
    combos = set()
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        pct = re.fullmatch(r"(?:top)?(\d+(?:\.\d+)?)%", token, re.IGNORECASE)
        if pct:
            combos.update(top_range(float(pct.group(1)) / 100))
        elif re.fullmatch(r"(?:(?:10|[2-9TJQKAtjqka])[cdhsCDHS]){2}", token):
            combos.add(tuple(sorted(parse_cards(token), reverse=True)))
        else:
            for cls in _expand(token.upper().replace("S", "s").replace("O", "o")):
                combos.update(class_combos(cls))
    if not combos:
        raise ValueError(f"empty range: {text!r}")
    return sorted(combos)


def build_table(trials=4_000_000, processes=None, seed=2024, progress=None):
    """Recompute the table and write it to ``TABLE_PATH``.

    One opponent: exact, by enumerating every board, hand and opposing hand.
    Two to nine opponents: Monte Carlo with ``trials`` deals per cell, standard errors kept.
    """
    from multiprocessing import Pool

    n_boards, chunk = 2_598_960, 200
    chunks = [(start, min(start + chunk, n_boards)) for start in range(0, n_boards, chunk)]
    jobs = [(cls, n, trials, seed) for cls in ALL_CLASSES for n in range(2, MAX_OPPONENTS + 1)]
    counts = 0
    with Pool(processes) as pool:
        for done, part in enumerate(pool.imap_unordered(_heads_up_chunk, chunks, chunksize=16), 1):
            counts = counts + part
            if progress and done % 1000 == 0:
                progress(f"exact heads-up: {done}/{len(chunks)} board chunks")
        cells = pool.map(_table_job, jobs, chunksize=4)
    wins, ties = counts
    if int(2 * wins.sum() + ties.sum()) != 1326 * DEALS_PER_COMBO:
        raise RuntimeError("heads-up enumeration did not cover every deal exactly once")

    equity, stderr, heads_up = {}, {}, {}
    for i, cls in enumerate(ALL_CLASSES):
        total = len(class_combos(cls)) * DEALS_PER_COMBO
        heads_up[cls] = [int(wins[i]), int(ties[i])]
        equity[cls] = [round((int(wins[i]) + int(ties[i]) / 2) / total, 6)]
        stderr[cls] = [0.0]
    for (cls, _, _, _), (value, error) in zip(jobs, cells):
        equity[cls].append(round(value, 5))
        stderr[cls].append(round(error, 6))
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"description": "all-in equity of each starting hand vs 1..9 random opponents (index = opponents - 1)",
                   "method": f"1 opponent: exact enumeration; 2-9 opponents: Monte Carlo, {trials} trials per cell",
                   "equity": equity, "stderr": stderr, "heads_up_counts": heads_up}, fh, separators=(",", ":"))
    _data.cache_clear()
    ranking.cache_clear()
    top_range.cache_clear()
    return equity


def _table_job(job):
    from .equity import equity

    cls, n, trials, seed = job
    result = equity(list(class_combos(cls)[0]), n_opponents=n, trials=trials, seed=[seed, ALL_CLASSES.index(cls), n])
    return result.equity, result.stderr


def _heads_up_chunk(bounds):
    """Exact heads-up win and tie counts per class over boards[start:end] -> int64 array (2, 169).

    For each board, every pair of the other 47 cards is ranked once. A hand beats the hands ranked
    below it, minus those that share one of its cards and so cannot be held at the same time.
    """
    import numpy as np

    from .equity import _index_combos
    from .evaluator import evaluate_many, rank_counts

    boards = _index_combos(52, 5)[bounds[0]:bounds[1]]
    n, (first, second) = len(boards), _index_combos(47, 2).T.astype(np.intp)
    n_pairs = len(first)                                        # 1,081

    free = np.ones((n, 52), dtype=bool)
    free[np.arange(n)[:, None], boards] = False
    deck = np.broadcast_to(np.arange(52, dtype=np.int8), (n, 52))[free].reshape(n, 47)
    hands = np.empty((n, n_pairs, 7), dtype=np.int8)
    hands[:, :, 0], hands[:, :, 1] = deck[:, first], deck[:, second]
    hands[:, :, 2:] = boards[:, None, :]
    values = evaluate_many(hands.reshape(-1, 7)).reshape(n, n_pairs)

    below, equal, rank = rank_counts(values)

    # ranks of the 46 other hands containing each card, to take out the hands that share a card
    by_card = np.full((n, 47, 47), 32767, dtype=np.int16)
    by_card[:, first, second] = by_card[:, second, first] = rank
    sharing = np.concatenate([by_card[:, first, :], by_card[:, second, :]], axis=2)       # (n, 1081, 94)
    mine = rank.astype(np.int16)[:, :, None]
    wins = below - (sharing < mine).sum(axis=2)
    ties = equal - ((sharing == mine).sum(axis=2) - 2)          # the hand itself sits in both of its rows

    cls = _CLASS_OF()[hands[:, :, 0], hands[:, :, 1]].ravel()
    return np.array([np.bincount(cls, weights=wins.ravel(), minlength=169),
                     np.bincount(cls, weights=ties.ravel(), minlength=169)]).astype(np.int64)


@lru_cache(maxsize=1)
def _CLASS_OF():
    """52 x 52 array: index into ALL_CLASSES of the starting hand made by two cards."""
    import numpy as np

    from .cards import hand_class

    out = np.zeros((52, 52), dtype=np.int16)
    for a in range(52):
        for b in range(52):
            if a != b:
                out[a, b] = ALL_CLASSES.index(hand_class(a, b))
    return out
