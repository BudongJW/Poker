"""Statistics over a self-play database (see :mod:`pokerlab.selfplay` for the schema)."""

import sqlite3

import pandas as pd

from .preflop import preflop_equity


def _query(db_path, sql, params=()):
    with sqlite3.connect(db_path) as db:
        return pd.read_sql_query(sql, db, params=params)


def _add_bb100(df):
    """bb/100 with a 95% confidence interval from per-hand mean and mean of squares."""
    sd = (df.pop("m2") - df["mean_bb"] ** 2).clip(lower=0) ** 0.5
    df["bb100"] = df.pop("mean_bb") * 100
    df["ci95"] = 1.96 * sd / df["hands"] ** 0.5 * 100
    return df


def bot_summary(db_path):
    """Per bot: hands, win rate in bb/100 +- 95% CI, VPIP, PFR, went-to-showdown and won-at-showdown."""
    df = _query(db_path, """
        SELECT bot, COUNT(*) AS hands, AVG(net_bb) AS mean_bb, AVG(net_bb * net_bb) AS m2,
               100.0 * AVG(vpip) AS vpip, 100.0 * AVG(pfr) AS pfr,
               100.0 * SUM(showdown) / MAX(SUM(saw_flop), 1) AS wtsd,
               100.0 * SUM(showdown = 1 AND pot_share > 0) / MAX(SUM(showdown), 1) AS wsd
        FROM players GROUP BY bot""")
    return _add_bb100(df).sort_values("bb100", ascending=False).reset_index(drop=True)


def by_table_size(db_path):
    """bb/100 per bot (rows) and table size (columns)."""
    df = _query(db_path, "SELECT bot, n_players, 100.0 * AVG(net_bb) AS bb100 FROM players GROUP BY bot, n_players")
    return df.pivot(index="bot", columns="n_players", values="bb100")


def by_position(db_path, n_players):
    """bb/100 per bot (rows) and position (columns) at one table size."""
    df = _query(db_path, """SELECT bot, position, 100.0 * AVG(net_bb) AS bb100 FROM players
                            WHERE n_players = ? GROUP BY bot, position""", (int(n_players),))
    return df.pivot(index="bot", columns="position", values="bb100")


def by_hand_class(db_path, bot=None):
    """Per starting hand: times dealt, average result, and how often it reached and won a showdown."""
    df = _query(db_path, """
        SELECT hand_class, COUNT(*) AS hands, AVG(net_bb) AS mean_bb, AVG(net_bb * net_bb) AS m2,
               100.0 * AVG(vpip) AS vpip, 100.0 * AVG(showdown) AS showdown,
               100.0 * SUM(showdown = 1 AND pot_share > 0) / MAX(SUM(showdown), 1) AS wsd
        FROM players WHERE ?1 IS NULL OR bot = ?1 GROUP BY hand_class""", (bot,))
    return _add_bb100(df).sort_values("bb100", ascending=False).reset_index(drop=True)


def equity_validation(db_path):
    """Observed pot share vs the theoretical all-in equity, per starting hand and table size.

    Only meaningful for runs where every hand goes to showdown with equal stacks
    (e.g. a table of calling stations): there the average pot share must converge to the equity.
    ``z`` is the deviation in standard errors.
    """
    df = _query(db_path, """
        SELECT n_players, hand_class, COUNT(*) AS hands, AVG(pot_share) AS observed,
               AVG(pot_share * pot_share) AS m2
        FROM players GROUP BY n_players, hand_class""")
    df["expected"] = [preflop_equity(c, n - 1) for c, n in zip(df["hand_class"], df["n_players"])]
    se = ((df.pop("m2") - df["observed"] ** 2).clip(lower=0) / df["hands"]) ** 0.5
    df["z"] = (df["observed"] - df["expected"]) / se
    return df


def calibration(db_path, bot=None):
    """Equity a bot estimated when it acted vs what then happened, in 10%-wide equity buckets per street.

    ``share_at_showdown`` is the average pot share over the decisions whose hand reached a showdown.
    """
    return _query(db_path, """
        SELECT a.street, MIN(CAST(a.equity * 10 AS INTEGER), 9) AS bucket, COUNT(*) AS decisions,
               AVG(a.equity) AS predicted, AVG(p.showdown) AS reached_showdown,
               AVG(CASE WHEN p.showdown = 1 THEN p.pot_share END) AS share_at_showdown,
               AVG(p.net_bb) AS mean_net_bb
        FROM actions a JOIN players p ON p.hand_id = a.hand_id AND p.seat = a.seat
        WHERE a.equity IS NOT NULL AND (?1 IS NULL OR p.bot = ?1)
        GROUP BY a.street, bucket ORDER BY a.street, bucket""", (bot,))


def storage(db_path):
    """Row counts per table."""
    with sqlite3.connect(db_path) as db:
        return {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("batches", "hands", "players", "actions")}


def print_report(db_path):
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("rows:", storage(db_path))
    print("\n== bots (bb/100 = big blinds won per 100 hands) ==")
    print(bot_summary(db_path).to_string(index=False))
    sizes = by_table_size(db_path)
    if sizes.shape[1] > 1:
        print("\n== bb/100 by table size ==")
        print(sizes.to_string())
    for n in sizes.columns:
        print(f"\n== bb/100 by position, {n} players ==")
        print(by_position(db_path, n).to_string())
    hands = by_hand_class(db_path)
    print("\n== best / worst starting hands (all bots) ==")
    print(pd.concat([hands.head(8), hands.tail(5)]).to_string(index=False))
    cal = calibration(db_path)
    if len(cal):
        print("\n== equity estimated at decision time vs outcome ==")
        print(cal.to_string(index=False))
