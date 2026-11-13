import itertools
import random
import sqlite3

import numpy as np
import pytest

from pokerlab import advisor, report, selfplay
from pokerlab.bots import BOT_TYPES, Bot, make_bot
from pokerlab.cards import ALL_CLASSES, card, class_combos, hand_class, parse_cards
from pokerlab.engine import CALL, FOLD, RAISE, play_hand, position_name
from pokerlab.equity import equity
from pokerlab.evaluator import evaluate, evaluate_many
from pokerlab.preflop import parse_range, preflop_equity, ranking, top_range


def value(text):
    return evaluate(parse_cards(text))


# ---- cards ---------------------------------------------------------------------------------

def test_card_parsing():
    assert parse_cards("AsKd") == parse_cards(["AS", "KD"]) == parse_cards("as, kd") == [card("As"), card("Kd")]
    assert parse_cards("10h") == [card("Th")]
    with pytest.raises(ValueError):
        parse_cards("AsAs")
    with pytest.raises(ValueError):
        parse_cards("Ax")


def test_hand_classes():
    assert len(ALL_CLASSES) == 169
    assert sum(len(class_combos(c)) for c in ALL_CLASSES) == 1326
    assert hand_class(card("Kd"), card("As")) == "AKo"
    assert hand_class(card("7h"), card("7c")) == "77"
    assert all(hand_class(*combo) == "T9s" for combo in class_combos("T9s"))


# ---- evaluator -----------------------------------------------------------------------------

def test_five_card_hands_exhaustive():
    hands = np.array(list(itertools.combinations(range(52), 5)), dtype=np.int8)
    values = evaluate_many(hands)
    assert np.bincount(values >> 20).tolist() == [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
    assert len(np.unique(values)) == 7462           # number of distinct poker hand ranks


def test_seven_card_ordering():
    ordered = ["2c3d5h7s8c9dJh",        # high card
               "2c2d5h7s8c9dJh",        # pair
               "2c2d5h5s8c9dJh",        # two pair
               "2c2d2h5s8c9dJh",        # trips
               "As2d3h4s5c9dJh",        # wheel
               "2c3d4h5s6c9dJh",        # six-high straight
               "2c7c4c5c9cAdAh",        # flush
               "2c2d2h5s5c9dJh",        # full house
               "2c2d2h2s5c9dJh",        # quads
               "As2s3s4s5s9dJh",        # steel wheel
               "TsJsQsKsAs9dJh"]        # royal flush
    values = [value(h) for h in ordered]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert [v >> 20 for v in values] == [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8]


def test_kickers_and_ties():
    assert value("AsKd7h7c2s3d9h") > value("AsQd7h7c2s3d9h")            # pair, better kicker
    assert value("3h3s4h4s8s8cQh") < value("Kh6c4h4s8s8cQh")            # three pair: best two + kicker
    assert value("AsAdKhKcQs2d3h") == value("AcAhKsKdQc2h3s")            # identical two pair
    assert value("2c3c4h5s6dAsKs") == value("2c3c4h5s6dKhKd")            # both play the board straight
    assert value("7c7s7hAdAsKdKs") == value("7c7s7hAdAs2d3s")            # full house ignores the rest


def test_scalar_matches_vectorised():
    rng = np.random.default_rng(0)
    for n_cards in (5, 6, 7):
        hands = np.array([rng.permutation(52)[:n_cards] for _ in range(3000)], dtype=np.int8)
        assert [evaluate(h.tolist()) for h in hands] == evaluate_many(hands).tolist()


# ---- equity --------------------------------------------------------------------------------

@pytest.mark.parametrize("hand, expected", [("AsAh", 0.852), ("KsKh", 0.824), ("AsKs", 0.670), ("7d2c", 0.346)])
def test_preflop_heads_up_reference_values(hand, expected):
    assert equity(hand, trials=300_000, seed=1).equity == pytest.approx(expected, abs=0.004)


def test_monte_carlo_agrees_with_enumeration():
    for board, n_opp in (("QsJs2d", 1), ("QsJs2d7h", 2)):
        exact = equity("AsKs", board, n_opp)
        mc = equity("AsKs", board, n_opp, exact=False, trials=300_000, seed=3)
        assert exact.exact and not mc.exact
        assert abs(exact.equity - mc.equity) < 4 * mc.stderr
        assert abs(exact.win - mc.win) < 0.004 and abs(exact.tie - mc.tie) < 0.002


def test_river_enumeration_counts_and_ties():
    r = equity("2c3d", "AsKsQsJsTs")                 # royal flush on board: always a split
    assert (r.win, r.tie, r.equity, r.trials) == (0.0, 1.0, 0.5, 990)
    r = equity("2c3d", "AsKsQsJsTs", n_opponents=2)
    assert r.equity == pytest.approx(1 / 3) and r.trials == 990 * 903
    assert equity("AhAd", "AsAc7d2h9s").win == 1.0   # quad aces, no straight flush possible


def test_more_opponents_means_less_equity():
    values = [equity("AsAh", n_opponents=n, trials=50_000, seed=1).equity for n in (1, 3, 6, 8)]
    assert values == sorted(values, reverse=True)


def test_ranges():
    qq, kk = class_combos("QQ"), class_combos("KK")
    assert equity("AhKd", ranges=[qq], trials=200_000, seed=5).equity == pytest.approx(0.43, abs=0.01)
    assert equity("AsAh", ranges=[kk], trials=200_000, seed=5).equity == pytest.approx(0.82, abs=0.01)
    exact = equity("AsKs", "QsJs2d", ranges=[qq + class_combos("JJ")])
    mc = equity("AsKs", "QsJs2d", ranges=[qq + class_combos("JJ")], exact=False, trials=200_000, seed=5)
    assert exact.exact and abs(exact.equity - mc.equity) < 4 * mc.stderr
    # hero holds two of the kings: an opponent on exactly KK has one combo left, never zero
    assert equity("KsKh", ranges=[kk], trials=20_000, seed=1).tie > 0.9
    with pytest.raises(ValueError):
        equity("AsAh", "AdAc2s", ranges=[class_combos("AA")])


def test_hand_distribution_sums_to_one():
    for kwargs in ({"board": "QsJs2d"}, {"n_opponents": 4, "trials": 20_000, "seed": 1}):
        assert sum(equity("AsKs", **kwargs).hand_dist.values()) == pytest.approx(1.0)


# ---- pre-flop table and ranges -------------------------------------------------------------

def test_preflop_table():
    assert preflop_equity("AA", 1) == pytest.approx(0.852, abs=0.003)
    assert preflop_equity("72o", 1) == pytest.approx(0.346, abs=0.003)
    assert ranking(1)[0] == "AA" and ranking(8)[0] == "AA"
    for cls in ALL_CLASSES:
        row = [preflop_equity(cls, n) for n in range(1, 9)]
        assert row == sorted(row, reverse=True)
        assert preflop_equity(cls, 30) == row[-1]            # more than 8 opponents: clamp


def test_range_parsing():
    assert len(parse_range("AA")) == 6
    assert len(parse_range("AK")) == 16 and len(parse_range("AKs")) == 4 and len(parse_range("AKo")) == 12
    assert len(parse_range("TT+")) == 30
    assert len(parse_range("ATs+")) == 16
    assert len(parse_range("22-55, A2s-A5s")) == 24 + 16
    assert len(parse_range("QQ, qq, AsAh")) == 7
    assert parse_range("KsAs") == [(card("As"), card("Ks"))]
    assert len(parse_range("AKs, KsAs, AsKs")) == 4              # the explicit combo is already in AKs
    assert 0.10 * 1326 <= len(parse_range("10%")) <= 0.10 * 1326 + 12
    assert set(class_combos("AA")) <= set(top_range(0.05))
    with pytest.raises(ValueError):
        parse_range("XY+")


# ---- engine --------------------------------------------------------------------------------

class Script(Bot):
    """Plays a fixed list of answers, then check/calls."""

    def __init__(self, *answers):
        super().__init__()
        self.answers = list(answers)
        self.views = []

    def act(self, view):
        self.views.append(view)
        return self.answers.pop(0) if self.answers else (CALL, 0)


class Deck:
    """Stands in for the rng: deals the given hole cards and board."""

    def __init__(self, holes, board):
        self.cards = parse_cards("".join(holes) + board)

    def sample(self, population, k):
        assert k == len(self.cards)
        return list(self.cards)


def test_positions():
    assert [position_name(s, 0, 6) for s in range(6)] == ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
    assert [position_name(s, 1, 2) for s in range(2)] == ["BB", "BTN"]
    assert [position_name(s, 0, 9) for s in range(9)][3:] == ["UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO"]


def test_everyone_folds_to_the_big_blind():
    bots = [Script((FOLD, 0)), Script((FOLD, 0)), Script()]
    r = play_hand(bots, [1000] * 3, 0, 5, 10, random.Random(1))
    assert r.net == [0, -5, 5] and not r.showdown and r.board == [] and r.final_street == 0
    assert bots[2].views == []                                   # the big blind never has to act


def test_heads_up_button_posts_small_blind_and_acts_first():
    bots = [Script(), Script()]
    r = play_hand(bots, [1000, 1000], 0, 5, 10, Deck(["AsAh", "KsKh"], "2c7d9hJcQd"))
    assert [a[1] for a in r.actions[:2]] == [0, 1]               # button first pre-flop
    assert [a[1] for a in r.actions[2:4]] == [1, 0]              # big blind first post-flop
    assert r.net == [10, -10] and r.showdown


def test_side_pots():
    # seat 0 has 100, seat 1 has 300, seat 2 has 1000; button = seat 0, blinds 5/10.
    # seat 0 shoves, both call, then seat 1 shoves the flop and seat 2 calls.
    bots = [Script((RAISE, 100)), Script((CALL, 0), (RAISE, 10_000)), Script()]
    deck = Deck(["AsAh", "7c2d", "KsKh"], "3c8dJhQc4s")
    r = play_hand(bots, [100, 300, 1000], 0, 5, 10, deck)
    assert r.invested == [100, 300, 300]
    assert r.payouts == [300, 0, 400]        # aces win the 300 main pot, kings the 400 side pot
    assert r.showdown and r.final_street == 3


def test_split_pot_and_uncalled_bet():
    bots = [Script((RAISE, 500)), Script()]
    r = play_hand(bots, [1000, 200], 0, 5, 10, Deck(["Ac2d", "Ad3h"], "KsKhQcQd9s"))
    assert r.invested == [500, 200] and r.payouts == [500, 200]  # board plays; 300 comes back


def test_short_all_in_does_not_reopen_betting():
    # blinds 5/10. Seat 0 (button) raises to 30, seat 1 shoves 45 (a raise of 15 < the minimum 20),
    # seat 2 folds, so seat 0 may only call or fold: its attempted re-raise becomes a call.
    bots = [Script((RAISE, 30), (RAISE, 500)), Script((RAISE, 45)), Script((FOLD, 0))]
    r = play_hand(bots, [1000, 45, 1000], 0, 5, 10, Deck(["AsAh", "KsKh", "7c2d"], "3c8dJhQc4s"))
    assert bots[0].views[1].can_raise is False and bots[0].views[1].to_call == 15
    assert r.invested == [45, 45, 10]


def test_short_big_blind_caps_what_others_owe():
    r = play_hand([Script((FOLD, 0)), Script()], [1000, 3], 0, 5, 10, Deck(["7c2d", "AsAh"], "3c8dJhQc4s"))
    assert r.showdown and r.net == [-3, 3]                       # nothing to call, so no fold


def test_illegal_answers_are_coerced():
    bots = [Script((RAISE, -50), (FOLD, 0)), Script((RAISE, 10 ** 9))]
    r = play_hand(bots, [1000, 1000], 0, 5, 10, random.Random(2))
    assert r.actions[0][3] == 15                                 # clamped up to the minimum raise (to 20)
    assert r.invested == [20, 1000] and r.net == [-20, 20]


def test_chips_are_conserved_with_uneven_stacks():
    rng = random.Random(7)
    for n in range(2, 10):
        bots = [make_bot(k, seed=n) for k in (["random", "maniac", "station"] * 3)[:n]]
        for hand in range(400):
            stacks = [rng.choice([3, 15, 40, 100, 250, 1000, 3000]) for _ in range(n)]
            r = play_hand(bots, stacks, hand % n, 5, 10, rng)
            assert sum(r.payouts) == sum(r.invested)
            assert all(0 <= r.invested[i] <= stacks[i] for i in range(n))
            for i in range(n):
                covered = sum(min(r.invested[j], r.invested[i]) for j in range(n))
                assert 0 <= r.payouts[i] <= covered
                assert r.folded_street[i] < 0 or r.payouts[i] == 0


# ---- bots, self-play, report ---------------------------------------------------------------

def test_every_bot_type_plays():
    rng = random.Random(3)
    bots = [make_bot(kind, seed=i) for i, kind in enumerate(sorted(BOT_TYPES))]
    for hand in range(60):
        r = play_hand(bots, [1000] * len(bots), hand % len(bots), 5, 10, rng)
        assert sum(r.net) == 0


def test_selfplay_is_reproducible_and_resumable(tmp_path):
    def totals(path):
        with sqlite3.connect(path) as db:
            return db.execute("SELECT bot, SUM(net), COUNT(*) FROM players GROUP BY bot ORDER BY bot").fetchall()

    a, b = str(tmp_path / "a.sqlite"), str(tmp_path / "b.sqlite")
    kwargs = dict(players=(3, 6), bots=("tag", "station", "maniac"), batch_size=100, processes=2, seed=11, progress=None)
    assert selfplay.run(a, hands=400, **kwargs)[0] == 400
    selfplay.run(b, hands=200, **kwargs)
    selfplay.run(b, hands=200, **kwargs)                         # second call continues with new batches
    assert totals(a) == totals(b)
    with sqlite3.connect(a) as db:
        assert db.execute("SELECT SUM(net) FROM players").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM hands").fetchone()[0] == 400
        assert db.execute("SELECT COUNT(*) FROM actions WHERE equity IS NOT NULL").fetchone()[0] > 0
    assert set(report.bot_summary(a)["bot"]) == {"tag", "station", "maniac"}
    assert len(report.calibration(a)) > 0


def test_showdown_frequencies_converge_to_equity(tmp_path):
    """Calling stations always reach showdown, so observed pot shares must match the equity table."""
    path = str(tmp_path / "stations.sqlite")
    selfplay.run(path, hands=30_000, players=(2, 6), bots=("station",), batch_size=5000, processes=2,
                 log_actions=False, progress=None)
    v = report.equity_validation(path)
    assert len(v) == 2 * 169
    assert v["z"].abs().max() < 5 and (v["z"].abs() < 2).mean() > 0.9


def test_advisor_report():
    a = advisor.analyze("AsKs", "QsJs2d", n_players=4, pot=100, to_call=50, trials=50_000, seed=1)
    assert a["made_hand"] == 0 and 0.5 < a["result"].equity < 0.65
    assert a["pot_odds"]["required_equity"] == pytest.approx(1 / 3) and a["pot_odds"]["profitable"]
    assert sum(a["ahead_now"]) == pytest.approx(1.0)
    assert advisor.analyze("AsAh", n_players=6)["preflop_rank"] == 1
    text = advisor.explain("7d2c", n_players=9, pot=30, to_call=20, trials=20_000, seed=1)
    assert "약함" in text and "손실 콜" in text
    ranged = advisor.explain("AhKd", n_players=2, ranges=[parse_range("QQ+")], trials=20_000, seed=1)
    assert "레인지 지정" in ranged
