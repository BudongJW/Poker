import itertools
import json
import random
import sqlite3
import threading
import urllib.request

import numpy as np
import pytest

from pokerlab import advisor, daily, inference, phh, preflop, report, selfplay, web
from pokerlab.bots import BOT_TYPES, Bot, make_bot
from pokerlab.cards import ALL_CLASSES, card, class_combos, hand_class, parse_cards
from pokerlab.engine import CALL, FOLD, RAISE, play_hand, position_name
from pokerlab.equity import ALL_COMBOS, COMBO_INDEX, Range, _index_combos, equity, matchup, next_card
from pokerlab.evaluator import evaluate, evaluate_many, rank_counts
from pokerlab.preflop import heads_up_exact, parse_range, preflop_equity, preflop_stderr, ranking, top_range


def value(text):
    return evaluate(parse_cards(text))


# ---- cards ---------------------------------------------------------------------------------

def test_card_parsing():
    assert parse_cards("AsKd") == parse_cards(["AS", "KD"]) == parse_cards("as, kd") == [card("As"), card("Kd")]
    assert parse_cards("10h") == [card("Th")]
    assert parse_cards("A♠K♦ 10♥") == parse_cards("AsKdTh")
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


def test_known_hands_are_enumerated():
    ak, qq = matchup(["AhKd", "QsQc"])
    assert ak.exact and ak.trials == 1_712_304                   # C(48, 5) boards
    assert ak.equity == pytest.approx(0.42835, abs=1e-5) and ak.equity + qq.equity == pytest.approx(1.0)
    assert equity("AhKd", ranges=[[tuple(parse_cards("QsQc"))]]).equity == pytest.approx(ak.equity)
    three = matchup(["AsAh", "KsKh", "7c2d"], "Kd 7h 2s")
    assert sum(r.equity for r in three) == pytest.approx(1.0) and three[1].equity > 0.9
    with pytest.raises(ValueError):
        matchup(["AsAh", "AsKh"])


def test_matchup_with_ranges_and_random_hands():
    results = matchup(["AsKs", parse_range("TT+"), None], trials=60_000, seed=1)
    assert not results[0].exact and sum(r.equity for r in results) == pytest.approx(1.0)
    assert results[1].equity > results[0].equity > results[2].equity


def test_precision_target_is_met():
    r = equity("AsKs", n_opponents=3, precision=0.002, seed=1)
    assert r.ci95 <= 0.002 and r.trials % 100_000 == 0 and r.trials < 2_000_000
    loose = equity("AsKs", n_opponents=3, precision=0.01, seed=1)
    assert loose.trials < r.trials


def test_weighted_range():
    aa, kk = class_combos("AA"), class_combos("KK")
    weighted = Range(aa + kk, [3.0] * 6 + [1.0] * 6)
    parts = [equity("QsQh", "Jd 7c 2s", ranges=[r]).equity for r in (aa, kk)]
    exact = equity("QsQh", "Jd 7c 2s", ranges=[weighted])
    assert exact.exact and exact.equity == pytest.approx((3 * parts[0] + parts[1]) / 4)
    mc = equity("QsQh", "Jd 7c 2s", ranges=[weighted], exact=False, trials=200_000, seed=2)
    assert abs(mc.equity - exact.equity) < 4 * mc.stderr
    assert equity("QsQh", ranges=[{aa[0]: 1.0, kk[0]: 0.0}], trials=20_000, seed=1).equity < 0.25   # only aces left
    vector = weighted.vector()
    assert vector.sum() == 24 and len(Range.from_vector(vector)) == 12
    assert len(weighted.without(parse_cards("As"))) == 9
    assert (ALL_COMBOS[COMBO_INDEX[5, 40]] == [40, 5]).all()


def test_next_card_averages_back_to_current_equity():
    after = next_card("AsKs", "Qs Js 2d")
    assert len(after) == 47 and after[card("Ts")].equity == 1.0             # royal flush
    assert np.mean([r.equity for r in after.values()]) == pytest.approx(equity("AsKs", "Qs Js 2d").equity)
    pinned = next_card("AsKs", "Qs Js 2d 7h", ranges=[[tuple(parse_cards("QhQd"))]])
    assert len(pinned) == 44 and card("Qh") not in pinned                    # the opponent's cards cannot come


def test_hand_distribution_sums_to_one():
    for kwargs in ({"board": "QsJs2d"}, {"n_opponents": 4, "trials": 20_000, "seed": 1}):
        assert sum(equity("AsKs", **kwargs).hand_dist.values()) == pytest.approx(1.0)


# ---- pre-flop table and ranges -------------------------------------------------------------

def test_preflop_table():
    assert ranking(1)[0] == "AA" and ranking(9)[0] == "AA"
    for cls in ALL_CLASSES:
        row = [preflop_equity(cls, n) for n in range(1, 10)]
        assert row == sorted(row, reverse=True)
        assert preflop_equity(cls, 30) == row[-1]            # more than 9 opponents: clamp
        assert preflop_stderr(cls, 1) == 0 and 0 < preflop_stderr(cls, 5) < 0.0005


def test_heads_up_column_is_exact():
    # 72o against one random hand, counted independently: 665,146,081 wins and 120,541,920 ties
    # out of 2,097,572,400 deals for each of its 12 combos
    assert preflop._data()["heads_up_counts"]["72o"] == [12 * 665_146_081, 12 * 120_541_920]
    win, tie, lose = heads_up_exact("AA")
    assert (round(win, 4), round(tie, 4), round(lose, 4)) == (0.8493, 0.0054, 0.1452)
    assert preflop_equity("AA", 1) == pytest.approx(win + tie / 2, abs=1e-6)
    counts = preflop._data()["heads_up_counts"]
    assert sum(2 * w + t for w, t in counts.values()) == 1326 * preflop.DEALS_PER_COMBO


def test_heads_up_enumeration_against_brute_force():
    start = 777_777
    board = _index_combos(52, 5)[start].tolist()
    pairs = list(itertools.combinations([c for c in range(52) if c not in board], 2))
    values = {p: evaluate(list(p) + board) for p in pairs}
    wins, ties = np.zeros(169, dtype=np.int64), np.zeros(169, dtype=np.int64)
    for h in pairs:
        cls = ALL_CLASSES.index(hand_class(*h))
        for g in pairs:
            if g[0] not in h and g[1] not in h:
                wins[cls] += values[g] < values[h]
                ties[cls] += values[g] == values[h]
    fast = preflop._heads_up_chunk((start, start + 1))
    assert (fast[0] == wins).all() and (fast[1] == ties).all()


def test_rank_counts():
    below, equal, rank = rank_counts(np.array([[5, 1, 5, 3, 1, 9]]))
    assert below.tolist() == [[3, 0, 3, 2, 0, 5]] and equal.tolist() == [[1, 1, 1, 0, 1, 0]]
    assert rank.tolist() == [[2, 0, 2, 1, 0, 3]]


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


# ---- range inference -----------------------------------------------------------------------

def test_hand_percentiles():
    preflop_pct = inference.hand_percentiles([])
    aces = COMBO_INDEX[card("As"), card("Ah")]
    assert preflop_pct[aces] == np.nanmax(preflop_pct) > 0.99
    board = parse_cards("Qs Js 2d 7h 7c")
    river = inference.hand_percentiles(board)
    assert np.isnan(river).sum() == 1326 - 1081                  # combos that use a board card
    assert river[COMBO_INDEX[card("7s"), card("7d")]] == 1.0     # quads: the nuts
    assert river[COMBO_INDEX[card("3c"), card("4d")]] < 0.1
    flop = inference.hand_percentiles(parse_cards("Qs Js 2d"))
    draw, air = COMBO_INDEX[card("As"), card("Ks")], COMBO_INDEX[card("8c"), card("3h")]
    assert flop[draw] > 0.8 > 0.3 > flop[air]                    # a big draw counts, although it is ace-high now
    sampled = inference.hand_percentiles(parse_cards("Qs Js 2d"), 200, np.random.default_rng(1))
    ok = ~np.isnan(flop)
    assert np.abs(sampled[ok] - flop[ok]).mean() < 0.04


def test_action_lines():
    line = inference.parse_line("open/bet/check call", 6)
    assert [(street, code) for street, _, code in line] == [(0, 2), (1, 2), (2, 1), (2, 1)]
    assert line[0][1] == (0, 0, 1, 3, 1, -1) and line[3][1] == (2, 1, 1, 3, 1, -1)      # seat unknown: pooled
    # two players left in the hand at a six-handed table (the default): the button opened after three folds
    seated = inference.parse_line("BTN: open/bet", 2)
    assert seated[0][1] == (0, 0, 1, 1, 1, 3)               # pre-flop: three players left, short-handed table, button
    assert seated[1][1] == (1, 0, 0, 0, 1, 2)               # flop: two left, the button acts last
    assert inference.parse_line("utg: open/bet", 3, table=9)[0][1] == (0, 0, 1, 3, 2, 0)   # full ring, early seat
    assert inference.parse_line("utg: open", 2)[0][1][3:] == (3, 1, 1)                     # six-handed UTG is a middle seat
    assert inference.parse_line("sb: open/bet", 2, table=2)[1][1][4:] == (0, 2)            # heads-up table: the button
    assert inference.parse_line("co: open/bet", 2)[1][1][5] == -1                          # last to act? unknown ...
    assert inference.parse_line("co: open/bet", 2, "bb")[1][1][5] == 2                     # ... unless my seat is known
    assert inference.parse_line("co: open/bet", 2, "btn")[1][1][5] == 0
    assert inference.parse_line("open", 8)[0][1][4] == 2                                   # eight in the hand: a full table
    with pytest.raises(ValueError):
        inference.parse_line("open", 6, table=4)
    with pytest.raises(ValueError):
        inference.parse_line("dealer: open", 6)
    assert inference.parse_line("open//bet", 2)[1][0] == 2
    with pytest.raises(ValueError):
        inference.parse_line("open/jam", 2)
    with pytest.raises(ValueError):
        inference.range_from_line("open/bet/bet", "AsKs", "Qs Js 2d", 2)     # no turn on the board yet


def test_seats_and_table_sizes():
    names = lambda n, button=0: [inference.POSITIONS[inference.seat_class(seat, button, n)] for seat in range(n)]
    assert names(6) == ["btn", "sb", "bb", "mp", "mp", "co"]
    assert names(9) == ["btn", "sb", "bb", "ep", "ep", "ep", "mp", "mp", "co"]
    assert names(4) == ["btn", "sb", "bb", "co"] and names(3) == ["btn", "sb", "bb"]
    assert names(2, button=1) == ["bb", "btn"]
    assert [inference.table_class(n) for n in (2, 3, 6, 7, 10, None)] == [0, 1, 1, 2, 2, -1]
    # after the flop: first, in between, last among the players left (button = seat 0, seats 2 and 4 folded)
    assert [inference.position_now(seat, 0, 6, {2, 4}, 1) for seat in (1, 3, 5, 0)] == [0, 1, 1, 2]
    assert inference.position_now(3, 0, 6, {2, 4}, 0) == inference.POSITIONS.index("mp")

    rng = random.Random(8)
    bots = [make_bot("random", seed=i) for i in range(6)]
    result = play_hand(bots, [1000] * 6, 2, 5, 10, rng)
    seen = inference.observed(result.actions, 6, 2)
    assert seen[0][0] == 5 and seen[0][2][4:] == (1, inference.POSITIONS.index("mp"))    # first to act six-handed
    assert all(ctx[4] == 1 and 0 <= ctx[5] <= 5 for _, _, ctx, _ in seen)


def test_model_shrinks_rare_seats_and_stores_only_what_it_saw(tmp_path):
    counts = np.zeros(inference._SHAPE)
    common = inference.context(0, 0, True, 6, 6, inference.POSITIONS.index("btn"))
    rare = inference.context(0, 0, True, 6, 6, inference.POSITIONS.index("sb"))
    counts[common][:, inference.AGGRESSIVE] = 900        # the button raised 90% of the time ...
    counts[common][:, inference.GAVE_UP] = 100
    counts[rare][:, inference.GAVE_UP] = 2               # ... the small blind was seen twice, folding
    model = inference.ActionModel({"all": counts})
    assert model.likelihood(common, inference.AGGRESSIVE)[0] == pytest.approx(0.9, abs=0.01)
    assert 0.5 < model.likelihood(rare, inference.AGGRESSIVE)[0] < 0.9           # leans on the table average
    unknown_seat = inference.context(0, 0, True, 6, 6)
    unknown_table = inference.context(0, 0, True, 6)
    assert model.likelihood(unknown_seat, inference.AGGRESSIVE)[0] == pytest.approx(900 / 1002, abs=0.01)
    assert model.likelihood(unknown_table, inference.AGGRESSIVE)[0] == pytest.approx(900 / 1002, abs=0.01)

    path = str(tmp_path / "model.json")
    model.save(path)
    assert len(json.load(open(path))["counts"]["all"]) == 2                      # two situations, not 1,728
    assert (inference.ActionModel.load(path).counts["all"] == counts).all()
    old_layout = json.load(open(path)); old_layout["shape"] = [4, 3, 2, 4, 10, 3]
    json.dump(old_layout, open(path, "w"))
    with pytest.raises(ValueError):
        inference.ActionModel.load(path)


def test_bundled_action_model_reads_strength_from_aggression():
    model = inference.default_model()
    assert {"all", "tag", "lag", "nit"} <= set(model.counts)
    for ctx in (inference.context(0, 0, True, 2), inference.context(1, 0, False, 2, 6, 2),
                inference.context(3, 1, True, 2, 9, 0)):
        total = sum(model.likelihood(ctx, action) for action in range(3))
        assert total == pytest.approx(np.ones(inference.N_BUCKETS))
    flop_bet = inference.context(1, 0, False, 2)                                 # betting the flop heads-up
    raise_by_strength = model.likelihood(flop_bet, inference.AGGRESSIVE)
    assert raise_by_strength[-1] > 3 * raise_by_strength[0]
    assert (model.likelihood(flop_bet, inference.AGGRESSIVE, "someone new") == raise_by_strength).all()

    hero, board = parse_cards("AsKs"), parse_cards("Qs Js 2d")
    pct = inference.hand_percentiles(board)
    strength = {}
    for line in ("open/bet", "limp/check"):
        vector = inference.range_from_line(line, hero, board, 2).vector()
        assert vector[COMBO_INDEX[card("As"), card("Qh")]] == 0             # uses one of hero's cards
        strength[line] = np.nansum(vector * pct) / vector.sum()
    assert strength["open/bet"] > strength["limp/check"] + 0.1
    tight = equity(hero, board, ranges=[inference.range_from_line("open/bet", hero, board, 2)]).equity
    assert tight < equity(hero, board).equity


def test_models_fitted_on_public_hand_histories():
    online, pro = inference.load_model("online"), inference.load_model("pro")
    assert {"all", "nl25-50", "nl100-200", "nl400-600", "nl1000"} == set(online.counts)
    assert {"all", "pro", "pluribus"} == set(pro.counts)
    assert online.counts["all"].sum() > 1_000_000 and pro.counts["all"].sum() > 80_000
    river_bet = inference.context(3, 1, True, 2)                 # heads-up on the river, facing a bet
    for model in (online, pro):
        fold = model.likelihood(river_bet, inference.GAVE_UP)
        assert fold[0] > 0.8 > 0.2 > fold[-1]                    # weak hands fold, strong hands do not
        assert model.likelihood(river_bet, inference.AGGRESSIVE)[0] > 0.01      # and people do bluff-raise
    # real players bet weak hands when checked to far more often than the equity bots do
    checked_to = inference.context(1, 0, False, 2)
    assert online.likelihood(checked_to, inference.AGGRESSIVE)[-1] < inference.default_model().likelihood(
        checked_to, inference.AGGRESSIVE)[-1]
    # position: at a full table the button, folded to, opens far more hands than the first seat does
    early = inference.context(0, 0, True, 9, 9, inference.POSITIONS.index("ep"))
    button = inference.context(0, 0, True, 3, 9, inference.POSITIONS.index("btn"))
    assert online.likelihood(button, inference.AGGRESSIVE).sum() > 1.5 * online.likelihood(early, inference.AGGRESSIVE).sum()
    # table size: heads-up the button plays most hands, at a six-handed table the same seat count does not arise
    duel = inference.context(0, 0, True, 2, 2, inference.POSITIONS.index("btn"))
    assert online.likelihood(duel, inference.GAVE_UP).mean() < 0.5
    seated = advisor.analyze("AsKs", "Qs Js 2d", lines=["ep: open/bet"], position="bb", table=9)
    loose = advisor.analyze("AsKs", "Qs Js 2d", lines=["btn: open/bet"], position="bb", table=9)
    assert seated["ranges"][0]["width"] < loose["ranges"][0]["width"]             # an early open is a tighter range

    a = advisor.analyze("AsKs", "Qs Js 2d", lines=["open/bet"])                  # the default reads with 'online'
    b = advisor.analyze("AsKs", "Qs Js 2d", lines=["open/bet"], model="selfplay")
    assert advisor.DEFAULT_MODEL == "online" and a["ranges"][0]["width"] > b["ranges"][0]["width"]
    assert web.analyze_request({"hand": "AsKs", "board": "QsJs2d", "model": "pro", "profile": "pluribus",
                                "opponents": [{"type": "line", "value": "open/bet"}]})["ranged"]
    with pytest.raises(ValueError):
        web.analyze_request({"hand": "AsKs", "model": "/etc/passwd"})


def test_population_bot_follows_the_model():
    model = inference.default_model()
    bot = make_bot("human", seed=1)
    assert bot.name == "human" and bot.model is inference.load_model("online")
    rng = random.Random(4)
    table = [make_bot("pro", seed=i) for i in range(4)]
    seen = {"fold": 0, "call": 0, "check": 0, "raise": 0}
    for hand in range(300):
        result = play_hand(table, [1000] * 4, hand % 4, 5, 10, rng)
        assert sum(result.net) == 0
        for action in result.actions:
            seen[action[2]] += 1
    assert min(seen.values()) > 20 and seen["fold"] > seen["raise"]             # it folds, calls, checks and raises
    assert model is not bot.model


def test_fit_and_score_on_self_play(tmp_path):
    path = str(tmp_path / "pool.sqlite")
    selfplay.run(path, hands=600, players=(2, 3), bots=("tag", "lag", "nit"), batch_size=150, processes=2,
                 seed=5, progress=None)
    model = inference.fit(path, hands=120, flop_runouts=20, processes=2)
    assert {"all", "tag", "lag", "nit"} == set(model.counts)
    assert model.counts["all"].sum() == sum(model.counts[name].sum() for name in ("tag", "lag", "nit")) > 500
    assert model.counts["all"][:, :, 0, ..., inference.GAVE_UP].sum() == 0    # nobody folds without a bet
    model.save(str(tmp_path / "model.json"))
    again = inference.ActionModel.load(str(tmp_path / "model.json"))
    assert (again.counts["all"] == model.counts["all"]).all()
    rows = inference.score(path, again, hands=60, trials=1000, processes=2)
    table = inference.summarize(rows)
    assert table.loc["all", "decisions"] == 60 and 0 < table.loc["all", "mae_model"] < 0.5
    assert inference.reliability(rows)["n_model"].sum() == 60

    hands = inference._load_hands(path, 20, seed=0)
    hand = max(hands, key=lambda h: len(h["actions"]))
    hero_seat = hand["actions"][-1][1]
    ranges = inference.infer_ranges(again, hand["actions"][:-1], hero_seat, hand["holes"][hero_seat], hand["board"],
                                    hand["n_players"], hand["button"])
    assert hero_seat not in ranges and all(len(r) > 0 for r in ranges.values())


# ---- advisor extras and the web page -------------------------------------------------------

def test_advisor_new_sections():
    a = advisor.analyze("AsKs", "Qs Js 2d", n_players=2, pot=100, bet=50, lines=["open/bet"])
    assert a["ranged"] and a["ranges"][0]["label"] == "open/bet" and 0 < a["ranges"][0]["width"] < 1
    assert sum(a["ranges"][0]["classes"].values()) == pytest.approx(1.0)
    assert len(a["next_cards"]["cards"]) == 47 and a["next_cards"]["groups"][0]["equity"] > a["result"].equity
    assert a["bet"]["bluff_breakeven_folds"] == pytest.approx(1 / 3) and a["bet"]["called_breakeven_equity"] == 0.25
    data = json.loads(json.dumps(advisor.to_json(a)))
    assert data["hero"] == ["As", "Ks"] and data["made_hand"] == "high card" and "Ts" in data["next_cards"]["cards"]
    exact = advisor.analyze("AsAh")
    assert exact["result"].exact and exact["result"].equity == pytest.approx(0.85204, abs=1e-5)
    assert "전수 계산" in advisor.explain("AsAh") and "다음 카드별" in advisor.explain("AsKs", "Qs Js 2d", seed=1)


def test_web_request_handling():
    out = web.analyze_request({"hand": "AsKs", "board": "QsJs2d", "players": 3, "trials": 20000, "pot": "100",
                               "to_call": "50", "bet": "", "position": "bb", "table": 6,
                               "opponents": [{"type": "line", "value": "open/bet", "position": "btn"},
                                             {"type": "cards", "value": "Qh Qd"}]})
    assert out["n_players"] == 3 and len(out["ranges"]) == 2 and out["pot_odds"]["required_equity"] == pytest.approx(1 / 3)
    assert {r["label"] for r in out["ranges"]} == {"btn: open/bet", "지정 레인지"}
    json.dumps(out)
    for bad in ({"hand": "AsKs", "players": 2, "opponents": [{"type": "range", "value": "QQ"}] * 2},
                {"hand": "AsAs"}, {"hand": "AsKs", "players": 99}, {"hand": "AsKs", "opponents": [{"type": "cards", "value": "Qh"}]}):
        with pytest.raises(ValueError):
            web.analyze_request(bad)


def test_web_server_round_trip():
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "승률 계산기" in urllib.request.urlopen(base + "/").read().decode()

        def post(payload):
            request = urllib.request.Request(base + "/api/analyze", json.dumps(payload).encode(),
                                             {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        status, data = post({"hand": "9h9d", "board": "9sKd4c2h7s", "players": 2})
        assert status == 200 and data["exact"] and data["equity"] > 0.99
        status, data = post({"hand": "9h9h"})
        assert status == 400 and "error" in data
    finally:
        server.shutdown()
        server.server_close()


# ---- the daily learning cycle --------------------------------------------------------------

def test_custom_bot_specs_and_mutation():
    bot = make_bot("equity:name=mine;value=0.2;margin=0.01;bluff=0.1;respect=0;model=1;trials=50", seed=1)
    assert (bot.name, bot.value, bot.margin, bot.bluff, bot.respect, bot.trials) == ("mine", 0.2, 0.01, 0.1, 0.0, 50)
    assert bot.model is inference.default_model()
    assert make_bot(daily._spec("challenger", daily.START)).name == "challenger"
    rng = random.Random(3)
    for _ in range(200):
        child = daily.mutate(daily.START, rng)
        assert all(daily.BOUNDS[k][0] <= v <= daily.BOUNDS[k][1] for k, v in child.items())
        assert sum(child[k] != daily.START[k] for k in child) <= 2


def test_models_merge_with_decay():
    model = inference.default_model()
    merged = model.merged(model, decay=0.5)
    assert merged.counts["all"].sum() == pytest.approx(1.5 * model.counts["all"].sum())
    assert set(merged.counts) == set(model.counts)


def test_learning_cycle_keeps_state(tmp_path, capsys):
    state = str(tmp_path / "state")
    options = dict(state_dir=state, hands=300, challengers=1, fit_hands=120, score_hands=40, processes=2,
                   work_dir=str(tmp_path), progress=None)
    first = daily.run_cycle(**options)
    assert first["cycle"] == 1 and first["hands"] == 300 and first["model"]["mae_before"] is None and first["model"]["kept"]

    # starting from the bundled model, which has never seen a 'champion' or 'challenger' profile
    seeded = tmp_path / "seeded"
    seeded.mkdir()
    inference.default_model().save(str(seeded / "action_model.json"))
    entry = daily.run_cycle(**dict(options, state_dir=str(seeded)))
    assert entry["model"]["mae_before"] is not None and entry["cycle"] == 1
    assert {"champion", "challenger", "human", "nit", "tag_range"} == set(first["standings"])     # two seats play 'human'
    assert first["opponents"] == ["human", "human", "nit", "tag_range"]
    with pytest.raises(ValueError):
        daily.run_cycle(**dict(options, references=("human",)))
    assert not list(tmp_path.glob("*.sqlite"))                                  # the hand database is cleaned up
    second = daily.run_cycle(**options)
    assert second["cycle"] == 2 and second["model"]["mae_before"] is not None
    assert [e["cycle"] for e in daily.load_history(state)] == [1, 2]
    assert daily.load_champion(state) == second["champion"]
    model = inference.ActionModel.load(str(tmp_path / "state" / "action_model.json"))
    assert {"champion", "challenger"} <= set(model.counts)
    daily.print_history(state)
    assert "현재 챔피언" in capsys.readouterr().out


# ---- public hand histories -----------------------------------------------------------------

PHH_HAND = """variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [1, 2, 0]
min_bet = 2
starting_stacks = [200, 200, 200]
actions = ['d dh p1 ????', 'd dh p2 ????', 'd dh p3 AsKs', 'p3 cbr 6', 'p1 f', 'p2 cc', 'd db Qs7h2d', 'p2 cc', 'p3 cbr 8', 'p2 cc', 'd db 5c', 'p2 cc', 'p3 cc', 'd db 9d', 'p2 cbr 20 # a comment', 'p3 cc', 'p2 sm QhJh', 'p3 sm']
players = ['ann', 'bob', 'Pluribus']
"""


def test_phh_hand_is_replayed():
    hand = phh.parse_hand(PHH_HAND, phh.pluribus_label)
    assert hand["board"] == parse_cards("Qs7h2d5c9d")
    assert hand["holes"] == {2: tuple(parse_cards("AsKs")), 1: tuple(parse_cards("QhJh"))}   # dealt face up / shown
    assert hand["bots"] == {0: "pro", 1: "pro", 2: "pluribus"}
    assert [(a[0], a[1], a[2], a[5], a[6]) for a in hand["actions"]] == [
        (0, 2, "raise", 2, 3), (0, 0, "fold", 5, 3), (0, 1, "call", 4, 2),
        (1, 1, "check", 0, 2), (1, 2, "raise", 0, 2), (1, 1, "call", 8, 2),
        (2, 1, "check", 0, 2), (2, 2, "check", 0, 2), (3, 1, "raise", 0, 2), (3, 2, "call", 20, 2)]
    assert (hand["n_players"], hand["button"]) == (3, 2)                              # the last seat listed is the button
    seen = inference.observed(hand["actions"], hand["n_players"], hand["button"])
    # (street, raises, facing, players left, table class, position): the button opens, the small blind
    # folds to the raise, and on the flop the big blind (first to act of the two left) calls a bet
    assert seen[0][2] == (0, 0, 1, 1, 1, 3) and seen[1][2] == (0, 1, 1, 1, 1, 4) and seen[5][2] == (1, 1, 1, 0, 1, 0)
    assert seen[4][2][5] == 2                                                         # the button bets last to act

    assert phh.parse_hand(PHH_HAND.replace("'NT'", "'FT'")) is None                   # not no-limit hold'em
    assert phh.parse_hand(PHH_HAND.replace("p1 f", "p1 xyz")) is None                 # unknown action
    clash = phh.parse_hand(PHH_HAND.replace("p2 sm QhJh", "p2 sm QsJh"))              # shows a card that is on the board
    assert 1 not in clash["holes"] and 2 in clash["holes"]
    assert phh.stake_label("", 0.25) == "nl25-50" and phh.stake_label("", 10) == "nl1000"

    # heads-up the blinds are listed small, big but the last player (the button) posts the small one
    duel = phh.parse_hand("variant = 'NT'\nblinds_or_straddles = [5, 10]\n"
                          "actions = ['d dh p1 ????', 'd dh p2 ????', 'p2 cc', 'p1 cc', 'd db 2c3d4h', 'p1 cc', 'p2 f']\n")
    assert [(a[1], a[2], a[5]) for a in duel["actions"]] == [(1, "call", 5), (0, "check", 0), (0, "check", 0), (1, "fold", 0)]
    free_fold = inference.observed(duel["actions"], duel["n_players"], duel["button"])[-1]
    assert free_fold[2][4:] == (0, 2)                                                 # heads-up table, the button acts last
    assert free_fold[3] == inference.GAVE_UP and free_fold[2][2] == 0           # a fold with nothing to call ...
    assert (inference.default_model().likelihood(free_fold[2], inference.GAVE_UP) > 0).all()   # ... is never impossible


def test_phh_files(tmp_path):
    (tmp_path / "one.phh").write_text(PHH_HAND)
    (tmp_path / "many.phhs").write_text("[1]\n" + PHH_HAND + "\n[2]\n" + PHH_HAND.replace("'NT'", "'FT'") + "\n[3]\n" + PHH_HAND)
    (tmp_path / "notes.txt").write_text("ignored")
    assert len(phh.read_file(str(tmp_path / "many.phhs"))) == 2
    hands = phh.load(str(tmp_path), "table")
    assert len(hands) == 3 and hands[0]["bots"][0] == "table"
    assert len(phh.load(str(tmp_path), limit=2)) == 2

    model = inference.fit_prepared(inference.prepare(hands, flop_runouts=20, processes=1), inference.default_model(), iterations=3)
    assert {"all", "table"} == set(model.counts)
    assert model.counts["all"].sum() == pytest.approx(3 * 10)            # every action counted once, hidden hands as fractions
    fit = inference.action_fit(inference.prepare(hands, flop_runouts=20, processes=1), inference.default_model())
    assert fit["actions"] == 30 and fit["shown_hands"] == 6 and fit["log_likelihood"] < 0


def test_em_recovers_the_model_when_cards_are_hidden(tmp_path):
    """Hide every hand that did not reach a showdown, as a real hand history does, and learn anyway."""
    path = str(tmp_path / "pool.sqlite")
    selfplay.run(path, hands=3000, players=(2, 3), bots=("tag", "lag", "nit"), batch_size=750, processes=2,
                 seed=9, progress=None)
    full = inference.prepare(inference._load_hands(path, 3000, seed=0), flop_runouts=20, processes=2)
    # player tuples are in the order the players first acted, so pair them up with the hands to hide cards
    hands = inference._load_hands(path, 3000, seed=0)
    hidden = []
    for (players, buckets), hand in zip(full, hands):
        folded = {a[1] for a in hand["actions"] if a[2] == FOLD}
        seats = list(dict.fromkeys(a[1] for a in hand["actions"]))
        hidden.append(([(label, combo if seat not in folded and len(hand["holes"]) - len(folded) > 1 else -1, acts)
                        for (label, combo, acts), seat in zip(players, seats)], buckets))
    shown_only = [([p for p in players if p[1] >= 0], buckets) for players, buckets in hidden]
    assert 0.1 < sum(len(p) for p, _ in shown_only) / sum(len(p) for p, _ in full) < 0.6

    truth = inference.fit_prepared(full)
    naive = inference.fit_prepared(shown_only)
    steps = []
    learned = inference.fit_prepared(hidden, progress=steps.append)

    def distance(model):                 # average gap in action probabilities, weighted by how often a cell occurs
        weight = truth.counts["all"].sum(axis=-1, keepdims=True)
        share = lambda m: m.counts["all"] / np.maximum(m.counts["all"].sum(axis=-1, keepdims=True), 1e-9)
        return float((np.abs(share(model) - share(truth)) * weight).sum() / weight.sum() / 2)

    assert len(steps) >= 3 and distance(learned) < 0.5 * distance(naive)
    assert learned.counts["all"].sum() == pytest.approx(truth.counts["all"].sum())
    fits = {name: inference.action_fit(full, model)["log_likelihood"] for name, model in
            (("truth", truth), ("learned", learned), ("naive", naive))}
    assert fits["truth"] >= fits["learned"] > fits["naive"]
