"""Bot-vs-bot self-play that appends every hand to a SQLite database.

Work is split into batches (one table, one line-up, ``batch_size`` hands). Batches run in
worker processes and are fully determined by ``(seed, batch number)``, so a run can be stopped
and resumed, and any batch can be replayed. Stacks are reset every hand and seats are
reshuffled every hand, which makes hands independent samples.
"""

import os
import random
import sqlite3
import time
from multiprocessing import Pool

from .bots import make_bot
from .cards import card_str, hand_class
from .engine import CALL, RAISE, play_hand, position_name

SMALL_BLIND, BIG_BLIND = 5, 10
HANDS_PER_BATCH_ID = 10_000_000          # hand_id = batch * this + index within the batch

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch INTEGER PRIMARY KEY, seed INTEGER, n_players INTEGER, lineup TEXT, hands INTEGER,
    seconds REAL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS hands (
    hand_id INTEGER PRIMARY KEY, batch INTEGER, n_players INTEGER, button INTEGER, board TEXT,
    pot INTEGER, final_street INTEGER, showdown INTEGER);
CREATE TABLE IF NOT EXISTS players (
    hand_id INTEGER, seat INTEGER, bot TEXT, position TEXT, n_players INTEGER, hole TEXT,
    hand_class TEXT, invested INTEGER, net INTEGER, net_bb REAL, pot_share REAL, vpip INTEGER,
    pfr INTEGER, saw_flop INTEGER, showdown INTEGER, category INTEGER,
    PRIMARY KEY (hand_id, seat)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS actions (
    hand_id INTEGER, seq INTEGER, street INTEGER, seat INTEGER, action TEXT, amount INTEGER,
    pot INTEGER, to_call INTEGER, n_live INTEGER, equity REAL, behind INTEGER,
    PRIMARY KEY (hand_id, seq)) WITHOUT ROWID;
"""


def run_batch(job):
    """Play one batch -> (job, hand rows, player rows, action rows, seconds)."""
    batch, seed, n_hands, lineup, stack_bb, log_actions = job
    started = time.time()
    rng = random.Random(seed)
    bots = [make_bot(kind, seed=rng.getrandbits(32)) for kind in lineup]
    n = len(bots)
    stacks = [stack_bb * BIG_BLIND] * n
    hand_rows, player_rows, action_rows = [], [], []

    for index in range(n_hands):
        rng.shuffle(bots)                                   # new seating every hand
        button = index % n
        result = play_hand(bots, stacks, button, SMALL_BLIND, BIG_BLIND, rng)
        hand_id = batch * HANDS_PER_BATCH_ID + index
        pot = sum(result.invested)
        hand_rows.append((hand_id, batch, n, button, "".join(card_str(c) for c in result.board),
                          pot, result.final_street, int(result.showdown)))

        vpip, pfr = [0] * n, [0] * n
        for street, seat, action, *_ in result.actions:
            if street == 0 and action in (CALL, RAISE):
                vpip[seat] = 1
                pfr[seat] |= action == RAISE
        for seat in range(n):
            hole = result.hole[seat]
            net = result.payouts[seat] - result.invested[seat]
            in_at_end = result.folded_street[seat] < 0
            value = result.values[seat]
            player_rows.append((
                hand_id, seat, bots[seat].name, position_name(seat, button, n), n,
                card_str(hole[0]) + card_str(hole[1]), hand_class(*hole), result.invested[seat], net,
                net / BIG_BLIND, result.payouts[seat] / pot, vpip[seat], int(pfr[seat]),
                int(result.final_street > 0 and result.folded_street[seat] != 0),
                int(result.showdown and in_at_end), None if value is None else value >> 20))
        if log_actions:
            for seq, (street, seat, action, amount, pot_before, to_call, n_live, info, behind) in enumerate(result.actions):
                action_rows.append((hand_id, seq, street, seat, action, amount, pot_before, to_call, n_live, info, behind))
    return job, hand_rows, player_rows, action_rows, time.time() - started


def connect(db_path):
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=OFF")
    db.executescript(SCHEMA)
    if "behind" not in [row[1] for row in db.execute("PRAGMA table_info(actions)")]:
        db.execute("ALTER TABLE actions ADD COLUMN behind INTEGER")     # a database from before stacks were logged
    return db


def _lineup(pool, n_players, rng):
    """Every bot type in the pool sits if there is room; otherwise a random subset."""
    if n_players <= len(pool):
        return rng.sample(pool, n_players)
    return [pool[i % len(pool)] for i in range(n_players)]


def run(db_path, hands=None, seconds=None, players=(6,), bots=("tag", "lag", "nit", "station", "maniac", "random"),
        batch_size=2000, processes=None, seed=0, stack_bb=100, log_actions=True, progress=print):
    """Collect hands until ``hands`` or ``seconds`` is reached (both None = until interrupted).

    ``players`` may list several table sizes; batches cycle through them.
    Returns (hands played, wall seconds).
    """
    db = connect(db_path)
    next_batch = (db.execute("SELECT MAX(batch) FROM batches").fetchone()[0] or 0) + 1
    started = time.time()
    played = 0
    last_report = started

    def jobs():
        nonlocal next_batch
        out = []
        for _ in range(wave):
            remaining = None if hands is None else hands - played - sum(j[2] for j in out)
            if remaining is not None and remaining <= 0:
                break
            batch = next_batch
            next_batch += 1
            rng = random.Random(seed * 1_000_003 + batch)
            n_players = players[batch % len(players)]
            size = batch_size if remaining is None else min(batch_size, remaining)
            out.append((batch, rng.getrandbits(48), size, _lineup(list(bots), n_players, rng), stack_bb, log_actions))
        return out

    processes = processes or os.cpu_count()
    wave = 2 * processes                 # batches submitted at a time, so stopping is prompt
    with Pool(processes) as pool:
        try:
            while (hands is None or played < hands) and (seconds is None or time.time() - started < seconds):
                for job, hand_rows, player_rows, action_rows, secs in pool.imap_unordered(run_batch, jobs()):
                    with db:
                        db.execute("INSERT INTO batches (batch, seed, n_players, lineup, hands, seconds) VALUES (?,?,?,?,?,?)",
                                   (job[0], job[1], len(job[3]), ",".join(job[3]), job[2], secs))
                        db.executemany("INSERT INTO hands VALUES (?,?,?,?,?,?,?,?)", hand_rows)
                        db.executemany("INSERT INTO players VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", player_rows)
                        db.executemany("INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?)", action_rows)
                    played += len(hand_rows)
                    if progress and time.time() - last_report > 5:
                        last_report = time.time()
                        progress(f"{played:,} hands, {played / (last_report - started):,.0f} hands/s")
        except KeyboardInterrupt:
            pool.terminate()
    elapsed = time.time() - started
    db.close()
    if progress:
        progress(f"done: {played:,} hands in {elapsed:.1f}s ({played / max(elapsed, 1e-9):,.0f} hands/s) -> {db_path}")
    return played, elapsed
