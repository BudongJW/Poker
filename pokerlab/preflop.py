"""Pre-flop lookup table (169 starting hands x 1..8 random opponents) and hand ranges."""

import json
import os
import re
from functools import lru_cache

from .cards import ALL_CLASSES, RANKS, class_combos, parse_cards

TABLE_PATH = os.path.join(os.path.dirname(__file__), "data", "preflop_equity.json")
MAX_OPPONENTS = 8


@lru_cache(maxsize=1)
def table():
    """{'AKs': [equity vs 1 opponent, vs 2, ..., vs 8]} for all 169 classes."""
    with open(TABLE_PATH, encoding="utf-8") as fh:
        return json.load(fh)["equity"]


def preflop_equity(cls, n_opponents):
    return table()[cls][min(n_opponents, MAX_OPPONENTS) - 1]


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


def build_table(trials=1_000_000, processes=None, seed=2024):
    """Recompute the table by Monte Carlo and write it to ``TABLE_PATH``."""
    from multiprocessing import Pool

    jobs = [(cls, n, trials, seed) for cls in ALL_CLASSES for n in range(1, MAX_OPPONENTS + 1)]
    with Pool(processes) as pool:
        results = pool.map(_table_job, jobs, chunksize=8)
    out = {cls: [None] * MAX_OPPONENTS for cls in ALL_CLASSES}
    for (cls, n, _, _), value in zip(jobs, results):
        out[cls][n - 1] = round(value, 4)
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"description": "all-in equity of each starting hand vs 1..8 random opponents (index = opponents - 1)",
                   "method": f"Monte Carlo, {trials} trials per cell", "equity": out}, fh, indent=0)
    table.cache_clear()
    return out


def _table_job(job):
    from .equity import equity

    cls, n, trials, seed = job
    return equity(list(class_combos(cls)[0]), n_opponents=n, trials=trials,
                  seed=[seed, ALL_CLASSES.index(cls), n]).equity
