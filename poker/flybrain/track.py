"""Local record of live play: what the fly chose, what the baseline chose, what happened.

Before this module a live hand produced one `log.info` line and nothing queryable, and
`FlyDecision.reinforce()` was never called at all - so the fly made a judgement on every
real hand, was never told whether it was right, and left no evidence behind. A thousand
hands of live play produced exactly as much track record as zero.

What a row is
-------------
One row per *decision*, carrying the table state it was made on and both sides' choices.
The hand's chip outcome is not known when the decision is made - it arrives when the next
hand starts - so rows are written unsettled and updated in place by `settle_hand()`.
Several decisions can share a game_id; they all settle to the same hand outcome.

What the numbers can and cannot say
-----------------------------------
This is the part that decides whether a track record means anything.

In **shadow** mode the baseline drives the mouse, so the hand outcome is a measurement of
*the baseline's* play. The fly's choices changed nothing about it, and averaging outcomes
over shadow rows produces the baseline's win rate no matter what the fly did. Shadow rows
support exactly two honest claims: how often the fly agrees with the baseline, and what
the fly does on real scraped states. They do not support "the fly wins X bb/100".

In **active** mode the fly drives, so the outcome is attributable to it and a win rate is
meaningful. `summary()` therefore splits every outcome figure by mode and refuses to pool
them, because pooling is how a shadow-mode baseline result gets quoted as the fly's.

Failure policy: a live hand must never be dropped because logging failed. Every public
method swallows its own exceptions and reports the failure through the return value.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
import datetime
import logging
import os
import sqlite3

log = logging.getLogger(__name__)

DEFAULT_PATH = os.path.join('poker', 'data', 'flybrain', 'live_play.sqlite')

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT    NOT NULL,
    game_id           TEXT    NOT NULL,
    mode              TEXT    NOT NULL,
    stage             TEXT,
    round_number      INTEGER,
    equity            REAL,
    pot               REAL,
    to_call           REAL,
    my_funds          REAL,
    big_blind         REAL,
    fly_decision      TEXT,
    baseline_decision TEXT,
    fly_active        INTEGER NOT NULL DEFAULT 0,
    agreed            INTEGER,
    fly_error         TEXT,
    synthetic         INTEGER NOT NULL DEFAULT 0,
    outcome_chips     REAL,
    outcome_bb        REAL,
    settled           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_decisions_game ON decisions(game_id);
CREATE INDEX IF NOT EXISTS idx_decisions_settled ON decisions(settled);
"""


def _connect(path):
    """Open the database, creating the file and schema if this is the first call."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0)
    connection.executescript(SCHEMA)
    return connection


_RECORDER_CACHE = {}


def recorder(path=None):
    """Return the process-wide recorder for this path, opening it at most once.

    Same shape as decision.get_brain(): one per process, so the caller does not have to
    carry it around and the hand-boundary state has a single home.
    """
    key = path or DEFAULT_PATH
    if key not in _RECORDER_CACHE:
        _RECORDER_CACHE[key] = LiveRecorder(key)
    return _RECORDER_CACHE[key]


def reset_recorder_cache():
    """Drop cached recorders. For tests."""
    _RECORDER_CACHE.clear()


class LiveRecorder:
    """Append-and-settle record of live decisions, backed by a local SQLite file.

    Deliberately local: the existing game log posts to a third-party host, and a track
    record that lives on someone else's server is not a track record you own.
    """

    def __init__(self, path=None):
        self.path = path or DEFAULT_PATH
        self.enabled = True
        # Which hand is in progress, so a hand is settled exactly once however many
        # decisions it contains. Lives here rather than in the caller because the
        # recorder is the thing that knows what it has already written.
        self.current_game_id = None
        try:
            _connect(self.path).close()
            log.info("Live play recorder writing to %s", os.path.abspath(self.path))
        except (sqlite3.Error, OSError) as exc:
            self.enabled = False
            log.warning("Live play recorder disabled (%s); play continues unrecorded", exc)

    def note_hand(self, game_id):
        """Register the hand now in progress; return the id of the one that just ended.

        Returns None while still inside the same hand, and None for the first hand of a
        session, when nothing has finished yet. Callers use the returned id to settle.
        """
        game_id = str(game_id or '')
        if not game_id or game_id == self.current_game_id:
            return None
        previous, self.current_game_id = self.current_game_id, game_id
        return previous

    def record_decision(self, game_id, mode, fly_decision, baseline_decision,
                        fly_active=False, fly_error=None, synthetic=False,
                        stage=None, round_number=None, equity=None, pot=None,
                        to_call=None, my_funds=None, big_blind=None):
        """Write one unsettled decision row. Returns its id, or None if it could not."""
        if not self.enabled:
            return None
        agreed = None
        if fly_decision is not None and baseline_decision is not None:
            agreed = int(str(fly_decision) == str(baseline_decision))
        try:
            with _connect(self.path) as connection:
                cursor = connection.execute(
                    """INSERT INTO decisions
                       (ts, game_id, mode, stage, round_number, equity, pot, to_call,
                        my_funds, big_blind, fly_decision, baseline_decision, fly_active,
                        agreed, fly_error, synthetic)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (datetime.datetime.now().isoformat(timespec='seconds'), str(game_id),
                     str(mode), stage, round_number, equity, pot, to_call, my_funds,
                     big_blind,
                     None if fly_decision is None else str(fly_decision),
                     None if baseline_decision is None else str(baseline_decision),
                     int(bool(fly_active)), agreed,
                     None if fly_error is None else str(fly_error),
                     int(bool(synthetic))))
                return cursor.lastrowid
        except (sqlite3.Error, OSError) as exc:
            log.warning("Could not record decision (%s); play continues", exc)
            return None

    def settle_hand(self, game_id, chips, big_blind=None):
        """Attach a finished hand's chip outcome to every decision made in it.

        Returns the number of rows updated, or None on failure. Already-settled rows are
        left alone so a repeated call cannot double-count a hand.
        """
        if not self.enabled or game_id is None or chips is None:
            return None
        try:
            chips = float(chips)
        except (TypeError, ValueError):
            return None
        outcome_bb = None
        try:
            if big_blind and float(big_blind) > 0:
                outcome_bb = chips / float(big_blind)
        except (TypeError, ValueError):
            outcome_bb = None
        try:
            with _connect(self.path) as connection:
                cursor = connection.execute(
                    """UPDATE decisions SET outcome_chips = ?, outcome_bb = ?, settled = 1
                       WHERE game_id = ? AND settled = 0""",
                    (chips, outcome_bb, str(game_id)))
                return cursor.rowcount
        except (sqlite3.Error, OSError) as exc:
            log.warning("Could not settle hand %s (%s)", game_id, exc)
            return None

    def summary(self):
        """Aggregate the record. Outcome figures are split by mode and never pooled."""
        if not self.enabled:
            return {}
        try:
            with _connect(self.path) as connection:
                connection.row_factory = sqlite3.Row
                return {
                    'path': os.path.abspath(self.path),
                    'totals': self._totals(connection),
                    'by_mode': self._by_mode(connection),
                    'disagreements': self._disagreements(connection),
                    'errors': self._errors(connection),
                }
        except (sqlite3.Error, OSError) as exc:
            log.warning("Could not summarise the live record (%s)", exc)
            return {}

    @staticmethod
    def _totals(connection):
        row = connection.execute(
            """SELECT COUNT(*) AS decisions,
                      COUNT(DISTINCT game_id) AS hands,
                      SUM(settled) AS settled_decisions,
                      COUNT(DISTINCT CASE WHEN settled = 1 THEN game_id END) AS settled_hands,
                      SUM(CASE WHEN synthetic = 1 THEN 1 ELSE 0 END) AS synthetic_decisions
               FROM decisions""").fetchone()
        return dict(row) if row else {}

    @staticmethod
    def _by_mode(connection):
        """Per-mode aggregates. bb/100 is over settled *hands*, not decisions."""
        rows = connection.execute(
            """SELECT mode,
                      COUNT(*) AS decisions,
                      COUNT(DISTINCT game_id) AS hands,
                      AVG(agreed) AS agreement_rate,
                      COUNT(DISTINCT CASE WHEN settled = 1 THEN game_id END) AS settled_hands
               FROM decisions GROUP BY mode ORDER BY mode""").fetchall()
        out = []
        for row in rows:
            entry = dict(row)
            # One outcome per hand: averaging over decisions would weight multi-decision
            # hands more heavily than single-decision ones.
            hand = connection.execute(
                """SELECT SUM(outcome_chips) AS chips, SUM(outcome_bb) AS bb,
                          COUNT(*) AS hands
                   FROM (SELECT game_id, MAX(outcome_chips) AS outcome_chips,
                                MAX(outcome_bb) AS outcome_bb
                         FROM decisions WHERE settled = 1 AND mode = ?
                         GROUP BY game_id)""", (row['mode'],)).fetchone()
            entry['total_chips'] = hand['chips'] if hand else None
            entry['total_bb'] = hand['bb'] if hand else None
            entry['bb_per_100'] = (
                100.0 * hand['bb'] / hand['hands']
                if hand and hand['hands'] and hand['bb'] is not None else None)
            # Only active-mode outcomes are attributable to the fly; in shadow the
            # baseline drove every one of them.
            entry['attributable_to_fly'] = (row['mode'] == 'active')
            out.append(entry)
        return out

    @staticmethod
    def _disagreements(connection):
        """Where the fly and the baseline parted company, and what each chose."""
        rows = connection.execute(
            """SELECT baseline_decision, fly_decision, COUNT(*) AS n
               FROM decisions WHERE agreed = 0
               GROUP BY baseline_decision, fly_decision
               ORDER BY n DESC LIMIT 20""").fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _errors(connection):
        rows = connection.execute(
            """SELECT fly_error, COUNT(*) AS n FROM decisions
               WHERE fly_error IS NOT NULL
               GROUP BY fly_error ORDER BY n DESC LIMIT 10""").fetchall()
        return [dict(row) for row in rows]


def format_summary(summary):
    """Render summary() as text. Kept separate so the CLI and the GUI can share it."""
    if not summary:
        return "No live play recorded yet."

    totals = summary.get('totals') or {}
    lines = [f"Live play record: {summary.get('path', '?')}", '',
             f"  decisions      {totals.get('decisions') or 0:,}",
             f"  hands          {totals.get('hands') or 0:,}",
             f"  settled hands  {totals.get('settled_hands') or 0:,}"]
    if totals.get('synthetic_decisions'):
        lines.append(f"  ** {totals['synthetic_decisions']:,} decisions were made on a "
                     f"SYNTHETIC connectome and say nothing about the fly **")

    lines += ['', f"  {'mode':8s} {'hands':>7s} {'decisions':>10s} {'agree%':>8s} "
                  f"{'settled':>8s} {'chips':>10s} {'bb/100':>9s}"]
    for entry in summary.get('by_mode') or []:
        agreement = entry.get('agreement_rate')
        chips = entry.get('total_chips')
        per_100 = entry.get('bb_per_100')
        lines.append(
            f"  {entry['mode']:8s} {entry['hands']:>7,} {entry['decisions']:>10,} "
            f"{'' if agreement is None else f'{100 * agreement:.1f}':>8s} "
            f"{entry['settled_hands']:>8,} "
            f"{'' if chips is None else f'{chips:+.2f}':>10s} "
            f"{'' if per_100 is None else f'{per_100:+.2f}':>9s}")

    shadow = [e for e in (summary.get('by_mode') or []) if not e['attributable_to_fly']]
    if shadow:
        lines += ['', "  Outcomes on shadow rows are the BASELINE's results - the fly did "
                      "not drive those hands.",
                  "  Only 'active' rows carry a win rate attributable to the fly."]

    disagreements = summary.get('disagreements') or []
    if disagreements:
        lines += ['', "  where they disagreed (baseline -> fly):"]
        for row in disagreements:
            lines.append(f"    {str(row['baseline_decision']):18s} -> "
                         f"{str(row['fly_decision']):18s} {row['n']:>6,}")

    errors = summary.get('errors') or []
    if errors:
        lines += ['', "  fly errors:"]
        for row in errors:
            lines.append(f"    {row['n']:>6,}  {row['fly_error']}")

    return '\n'.join(lines)
