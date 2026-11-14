import itertools
import json
import random
import sqlite3
import threading
import urllib.request

import numpy as np
import pytest

from pokerlab import advisor, daily, follow, inference, phh, preflop, report, selfplay, table, vision, web
from pokerlab.bots import BOT_TYPES, Bot, make_bot
from pokerlab.cards import ALL_CLASSES, card, class_combos, hand_class, parse_cards
from pokerlab.engine import CALL, FOLD, RAISE, play_hand, position_name
from pokerlab.equity import ALL_COMBOS, COMBO_INDEX, Range, _index_combos, equity, matchup, next_card
from pokerlab.evaluator import evaluate, evaluate_many, rank_counts
from pokerlab.preflop import heads_up_exact, parse_range, preflop_equity, preflop_stderr, ranking, top_range


def value(text):
    return evaluate(parse_cards(text))


@pytest.fixture(autouse=True)
def _no_real_keys(monkeypatch):
    """No test may reach a paid service: the keys of the machine the tests run on are hidden from them."""
    for name in ("POKER_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


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
    any_bet, any_price, expensive = inference.AGGRESSIVE, inference.ANY_PRICE, inference.EXPENSIVE
    line = inference.parse_line("open/bet/check call", 6)
    assert [(street, code) for street, _, code in line] == [(0, any_bet), (1, any_bet), (2, 1), (2, 1)]
    # seat unknown: pooled. Opening means paying the big blind into 1.5 big blinds: an expensive call
    assert line[0][1] == (0, 0, expensive, 3, 1, -1, -1) and line[3][1] == (2, 1, any_price, 3, 1, -1, -1)
    # two players left in the hand at a six-handed table (the default): the button opened after three folds
    seated = inference.parse_line("BTN: open/bet", 2)
    assert seated[0][1] == (0, 0, expensive, 1, 1, 3, -1)   # pre-flop: three players left, short-handed table, button
    assert seated[1][1] == (1, 0, 0, 0, 1, 2, -1)           # flop: two left, the button acts last; stacks unknown
    assert inference.parse_line("utg: open/bet", 3, table=9)[0][1] == (0, 0, expensive, 3, 2, 0, -1)   # full ring, early
    # sizes: percent of the pot, or big blinds for a pre-flop open or call
    sized = inference.parse_line("btn: open3x/bet30/bet66 /raise120", 2)
    assert [code for _, _, code in sized] == [inference.BET_MEDIUM, inference.BET_SMALL, inference.BET_MEDIUM,
                                              inference.BET_LARGE]
    assert inference.parse_line("open2x", 2)[0][2] == inference.BET_SMALL                  # a min-raise
    called = inference.parse_line("bb: call3x/call33/call66/call150", 2)
    assert [ctx[2] for _, ctx, _ in called] == [inference.NORMAL, inference.CHEAP, inference.NORMAL, expensive]
    assert inference.parse_line("sb: limp", 2)[0][1][2] == inference.NORMAL                # half a blind into 1.5
    # calling an open is cheaper for the big blind than for anybody else, however the size is written
    price = lambda line, **table: inference.parse_line(line, 2, **table)[0][1][2]
    assert [price(f"{seat}: call3x") for seat in ("bb", "sb", "btn")] == [inference.NORMAL, expensive, expensive]
    assert [price(f"{seat}: call80") for seat in ("bb", "sb", "btn")] == [inference.NORMAL, expensive, expensive]
    assert price("bb: call2x") == inference.CHEAP
    # at a heads-up table the button is the small blind, known seat or not
    assert price("open", table=2) == price("btn: open", table=2) == inference.NORMAL
    assert price("call3x", table=2) == expensive            # 2 more into the 4 that the two blinds and the raise make
    assert price("call2.5x", table=2) == inference.NORMAL and price("bb: check", table=2) == inference.NO_BET
    assert inference.parse_line("btn: open3x", 2, table=2)[0][2] == inference.BET_LARGE    # 2 more into a pot of 2
    assert inference.parse_line("3bet90", 2)[0][2] == inference.BET_LARGE
    assert inference.parse_line("call3bet50", 2)[0][1][2] == inference.NORMAL
    with pytest.raises(ValueError):
        inference.parse_line("open/bet3x", 2)                                              # big blinds only pre-flop
    assert inference.parse_line("utg: open", 2)[0][1][3:6] == (3, 1, 1)                    # six-handed UTG is a middle seat
    assert inference.parse_line("sb: open/bet", 2, table=2)[1][1][4:6] == (0, 2)           # heads-up table: the button
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
        inference.parse_line("open/donk", 2)
    with pytest.raises(ValueError):
        inference.range_from_line("open/bet/bet", "AsKs", "Qs Js 2d", 2)     # no turn on the board yet


def test_action_lines_with_stacks_and_all_ins():
    depths = lambda line, **more: [ctx[6] for _, ctx, _ in inference.parse_line(line, 2, **more)]
    codes = lambda line, **more: [code for _, _, code in inference.parse_line(line, 2, **more)]
    very_deep, deep, middling, shallow = inference.VERY_DEEP, inference.DEEP, inference.MIDDLING, inference.SHALLOW
    # a hundred big blinds: lots behind before the flop, less with every bet that goes in
    assert depths("btn: open/bet/bet/bet", stack=100) == [very_deep, deep, deep, middling]
    assert depths("btn 20bb: open/bet/bet/bet") == [deep, middling, middling, shallow]     # the stack in the prefix
    assert depths("btn 20bb: open/bet/bet/bet", stack=100) == depths("btn: open/bet/bet/bet", stack=20)
    assert depths("btn: open/bet") == [inference.ANY_DEPTH] * 2                            # no stack given: the average
    # all-ins: as a bet, and as the bet that was called
    assert codes("btn: openallin", stack=12) == codes("shove", stack=12) == codes("btn: openallin") == [inference.ALL_IN]
    assert codes("bb: call/check raiseallin", stack=40)[-1] == inference.ALL_IN
    called = inference.parse_line("bb: call/call/call/callallin", 2)
    assert called[-1][2] == inference.PASSIVE and called[-1][1][6] == inference.NOTHING_BEHIND
    assert called[-1][1][2] == inference.ANY_PRICE                         # how big the shove was depends on the stack
    assert inference.parse_line("bb: call/call/call/callallin", 2, stack=100)[-1][1][2] == inference.EXPENSIVE
    assert inference.parse_line("bb: callallin", 2, stack=15)[0][1][2:] == (inference.EXPENSIVE, 0, 1, 5, 0)
    # a bet of a size that the stack cannot pay is an all-in whether the line says so or not
    assert codes("btn: open/bet100/bet100/bet100", stack=30)[-1] == inference.ALL_IN
    assert codes("btn: open/bet100/bet100/bet100", stack=100)[-1] == inference.BET_LARGE
    with pytest.raises(ValueError):
        inference.parse_line("btn 0bb: open", 2)
    with pytest.raises(ValueError):
        inference.parse_line("open/bet3xallin", 2)
    with pytest.raises(ValueError):
        inference.parse_line("open/betallin/bet", 2)                       # nothing left to bet the turn with
    share = lambda found: found.vector() / found.vector().sum()
    short = share(inference.range_from_line("btn: openallin", "AsKs", (), 2, inference.load_model("pro"), stack=10))
    deep_shove = share(inference.range_from_line("btn: openallin", "AsKs", (), 2, inference.load_model("pro"), stack=100))
    assert np.abs(short - deep_shove).sum() > 0.01          # the same shove reads differently with ten big blinds


def test_bet_sizes_and_prices():
    # a bet of 45% / 90% of the pot is where small ends and large begins
    assert [inference.bet_class(x, 100, 0) for x in (20, 44, 45, 89, 90, 300)] == [2, 2, 3, 3, 4, 4]
    # a raise is measured after the call: raising to 250 over a 100 bet into 150 adds 150 beyond the call to a pot of 250
    assert inference.bet_class(250, 150, 100) == inference.BET_MEDIUM
    assert inference.bet_class(350, 150, 100) == inference.BET_LARGE
    # the price of calling those same bets: 45% of the pot costs 0.45 / 1.45 of the pot it makes
    assert [inference.price_class(x, 100 + x) for x in (0, 30, 44, 46, 89, 91, 500)] == [0, 1, 1, 2, 2, 3, 3]
    assert inference.price_class(10, 0) == inference.ANY_PRICE and inference.bet_class(10, 0, 0) == inference.AGGRESSIVE

    result = play_hand([make_bot("maniac", seed=i) for i in range(3)], [1000] * 3, 0, 5, 10, random.Random(5))
    seen = inference.observed(result.actions, 3, 0)
    assert seen[0][2][2] == inference.EXPENSIVE                      # first in: 10 to call into the 15 of blinds
    raises = [(a, code) for a, (_, _, _, code) in zip(result.actions, seen) if a[2] == "raise"]
    assert raises and all(code == inference.bet_class(a[3], a[4], a[5], a[8]) for a, code in raises)


def test_stack_depth_and_all_ins():
    # what is left after calling, measured against the pot after the call
    depths = [inference.depth_class(behind, 100, 0) for behind in (0, 50, 99, 100, 399, 400, 1999, 2000)]
    assert depths == [inference.NOTHING_BEHIND, 1, 1, 2, 2, 3, 3, inference.VERY_DEEP]
    assert inference.depth_class(100, 50, 50) == inference.MIDDLING
    assert inference.depth_class(None, 100, 0) == inference.depth_class(float("inf"), 100, 0) == inference.ANY_DEPTH
    # a bet that takes everything left is an all-in whatever its size; in a raise the call does not count
    assert inference.bet_class(30, 100, 0, behind=30) == inference.ALL_IN
    assert inference.bet_class(30, 100, 0, behind=31) == inference.BET_SMALL
    assert inference.bet_class(250, 150, 100, behind=150) == inference.ALL_IN
    assert inference.bet_class(250, 150, 100, behind=float("inf")) == inference.BET_MEDIUM       # stack unknown

    class Shover:
        def act(self, view):
            return "raise", view.max_raise_to

    for stacks in ([200, 1000], [1000, 200]):       # the short stack shoves, or the big stack puts it all-in
        result = play_hand([Shover(), make_bot("station")], stacks, 0, 5, 10, random.Random(1))
        shove, call = result.actions[:2]
        assert (shove[2], shove[8]) == ("raise", 190)       # 200 is all that can go in; 10 of it is the big blind
        assert (call[2], call[5], call[8]) == ("call", 190, 0)
        seen = inference.observed(result.actions, 2, 0)
        assert seen[0][3] == inference.ALL_IN and seen[0][2][6] == inference.DEEP          # 190 behind a pot of 20
        assert seen[1][3] == inference.PASSIVE and seen[1][2][6] == inference.NOTHING_BEHIND
        assert len(result.actions) == 2 and result.showdown and sum(result.net) == 0
    # a log without stacks (a database from before they were kept) still reads: depth unknown, no all-ins
    old = inference.observed([a[:8] for a in result.actions], 2, 0)
    assert old[0][3] == inference.BET_LARGE and old[0][2][6] == old[1][2][6] == inference.ANY_DEPTH


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
    assert seen[0][0] == 5 and seen[0][2][4:6] == (1, inference.POSITIONS.index("mp"))   # first to act six-handed
    assert all(ctx[4] == 1 and 0 <= ctx[5] <= 5 for _, _, ctx, _ in seen)


def test_model_shrinks_rare_seats_and_stores_only_what_it_saw(tmp_path):
    counts = np.zeros(inference._SHAPE)
    price = inference.EXPENSIVE
    common = inference.context(0, 0, price, 6, 6, inference.POSITIONS.index("btn"), inference.VERY_DEEP)
    rare = inference.context(0, 0, price, 6, 6, inference.POSITIONS.index("sb"), inference.VERY_DEEP)
    counts[common][:, inference.BET_MEDIUM] = 600        # the button raised 90% of the time ...
    counts[common][:, inference.BET_LARGE] = 300
    counts[common][:, inference.GAVE_UP] = 100
    counts[rare][:, inference.GAVE_UP] = 2               # ... the small blind was seen twice, folding
    model = inference.ActionModel({"all": counts})
    assert model.likelihood(common, inference.AGGRESSIVE)[0] == pytest.approx(0.9, abs=0.01)
    assert model.likelihood(common, inference.BET_MEDIUM)[0] == pytest.approx(0.6, abs=0.01)    # sizes add up to "a bet"
    assert model.likelihood(common, inference.BET_SMALL)[0] < 0.01
    assert 0.5 < model.likelihood(rare, inference.AGGRESSIVE)[0] < 0.9           # leans on the table average
    unknown_seat = inference.context(0, 0, price, 6, 6)
    unknown_table = inference.context(0, 0, price, 6)
    any_price = inference.context(0, 0, True, 6, 6, inference.POSITIONS.index("btn"))         # a bet, size unknown
    assert any_price[2] == inference.ANY_PRICE
    assert model.likelihood(any_price, inference.AGGRESSIVE)[0] == pytest.approx(0.9, abs=0.01)
    assert model.likelihood(inference.context(0, 0, inference.CHEAP, 6, 6), inference.AGGRESSIVE)[0] > 0.8   # unseen price
    assert model.likelihood(unknown_seat, inference.AGGRESSIVE)[0] == pytest.approx(900 / 1002, abs=0.01)
    assert model.likelihood(unknown_table, inference.AGGRESSIVE)[0] == pytest.approx(900 / 1002, abs=0.01)

    path = str(tmp_path / "model.json")
    model.save(path)
    assert len(json.load(open(path))["counts"]["all"]) == 2                      # two situations, not 20,736
    assert (inference.ActionModel.load(path).counts["all"] == counts).all()
    old_layout = json.load(open(path)); old_layout["shape"] = [4, 3, 4, 4, 3, 6, 10, 5]
    json.dump(old_layout, open(path, "w"))
    with pytest.raises(ValueError):
        inference.ActionModel.load(path)


def test_the_size_of_a_bet_is_a_separate_cautious_factor():
    counts = np.zeros(inference._SHAPE)
    often = inference.context(1, 0, inference.NO_BET, 2, 6, 2, inference.DEEP)       # flop, checked to, last to act
    rare = inference.context(1, 0, inference.NO_BET, 2, 6, 0, inference.DEEP)        # the same first to act: hardly ever seen
    small, large = inference.BET_SMALL, inference.BET_LARGE
    counts[often][:, inference.PASSIVE] = 500
    counts[often][1:5, small], counts[often][1:5, large] = 400, 100          # weak hands bet small ...
    counts[often][5:, small], counts[often][5:, large] = 100, 400            # ... strong hands large
    counts[rare][9, small] = 3                                               # three small bets with the best hands
    model = inference.ActionModel({"all": counts})

    def large_share(ctx, bucket):
        return model.likelihood(ctx, large)[bucket] / model.likelihood(ctx, inference.AGGRESSIVE)[bucket]

    assert large_share(often, 1) == pytest.approx(0.2, abs=0.02) and large_share(often, 9) == pytest.approx(0.8, abs=0.02)
    assert large_share(rare, 9) > 0.6                           # three hands do not overturn what thousands said
    # the weakest hands never bet here: should one, its size says what a bet of that size says of any hand
    assert model.likelihood(often, inference.AGGRESSIVE)[0] < 0.01
    assert large_share(often, 0) == pytest.approx(2400 / 4503, abs=0.01)
    for ctx in (often, rare, inference.context(1, 0, inference.NO_BET, 2)):
        assert sum(model.likelihood(ctx, a) for a in range(inference.N_ACTIONS)) == pytest.approx(np.ones(10))


def test_stack_depth_in_the_model():
    counts = np.zeros(inference._SHAPE)
    spot = lambda depth: inference.context(2, 0, inference.NO_BET, 2, 6, 2, depth)       # the turn, checked to
    faced = lambda depth: inference.context(2, 1, inference.NORMAL, 2, 6, 2, depth)      # the turn, facing a bet
    counts[spot(inference.DEEP)][:, inference.PASSIVE] = 700
    counts[spot(inference.DEEP)][:, inference.BET_MEDIUM] = 300          # deep: a third bet, never all-in
    counts[spot(inference.SHALLOW)][:, inference.PASSIVE] = 50
    counts[spot(inference.SHALLOW)][5:, inference.ALL_IN] = 150          # shallow: strong hands shove
    counts[spot(inference.ANY_DEPTH)][:, inference.PASSIVE] = 400        # hands whose stacks were not logged ...
    counts[spot(inference.ANY_DEPTH)][:, inference.BET_LARGE] = 400      # ... where a shove looks like a large bet
    counts[faced(inference.DEEP)][:, inference.BET_MEDIUM] = 200         # deep, people raise ...
    counts[faced(inference.DEEP)][:, inference.PASSIVE] = 800
    counts[faced(inference.NOTHING_BEHIND)][:, inference.PASSIVE] = 3    # ... facing an all-in nobody can
    model = inference.ActionModel({"all": counts})
    all_in = lambda ctx: model.likelihood(ctx, inference.ALL_IN)
    assert all_in(spot(inference.DEEP)).max() < 0.01
    assert all_in(spot(inference.SHALLOW))[9] > 0.7 > 0.05 > all_in(spot(inference.SHALLOW))[0]
    # stacks unknown: the average over every depth for betting at all, hands with known stacks for the kind of bet
    unknown = spot(inference.ANY_DEPTH)
    assert model.likelihood(unknown, inference.AGGRESSIVE)[9] == pytest.approx((300 + 150 + 400) / 2000, abs=0.01)
    assert model.likelihood(unknown, inference.BET_LARGE)[9] < 0.01 < all_in(unknown)[9]
    # nothing behind: fold or call, even where three hands are all the model has seen
    assert model.likelihood(faced(inference.DEEP), inference.AGGRESSIVE)[0] == pytest.approx(0.2, abs=0.01)
    assert model.likelihood(faced(inference.NOTHING_BEHIND), inference.AGGRESSIVE).max() < 0.002
    assert model.likelihood(faced(inference.NOTHING_BEHIND), inference.PASSIVE).min() > 0.7


def test_hidden_hands_are_judged_without_bet_sizes():
    priced = inference.context(1, 1, inference.NORMAL, 2, 6, 2, inference.SHALLOW)
    any_price = priced[:2] + (inference.ANY_PRICE,) + priced[3:6] + (inference.ANY_DEPTH,)
    assert inference._unsized(priced, inference.BET_LARGE) == (any_price, inference.AGGRESSIVE)
    assert inference._unsized(priced, inference.ALL_IN) == (any_price, inference.AGGRESSIVE)
    assert inference._unsized(priced, inference.GAVE_UP) == (any_price, inference.GAVE_UP)
    assert inference._priced(priced, inference.BET_SMALL) == (priced, inference.AGGRESSIVE)
    assert inference._coarse(priced, inference.PASSIVE) == ((1, 1, inference.ANY_PRICE, 0, -1, -1, -1), inference.PASSIVE)
    free = inference.context(1, 0, inference.NO_BET, 2, 6, 2, inference.DEEP)
    assert inference._unsized(free, inference.PASSIVE) == (free[:6] + (inference.ANY_DEPTH,), inference.PASSIVE)
    assert inference.STAGES == ("coarse", "unsized")

    # everybody bets this flop, weak hands small and strong hands large; somebody whose cards stay hidden bets large
    counts = np.zeros(inference._SHAPE)
    counts[free][:5, inference.BET_SMALL] = 100
    counts[free][5:, inference.BET_LARGE] = 100
    model = inference.ActionModel({"all": counts})
    buckets = {1: inference.strength_buckets(parse_cards("Qs 7h 2d"))}
    prepared = [([("someone", -1, ((1, free, inference.BET_LARGE),))], buckets)]
    told = inference.expected_counts(prepared, model, "full")[0]["all"][free][:, inference.BET_LARGE]
    not_told = inference.expected_counts(prepared, model, "unsized")[0]["all"][free][:, inference.BET_LARGE]
    assert told.sum() == pytest.approx(1) and not_told.sum() == pytest.approx(1)      # one bet, counted once
    assert told[:5].sum() < 0.1                                  # the size gives the hand away ...
    assert not_told[:5].sum() == pytest.approx(0.5, abs=0.05)    # ... unless the fit is not allowed to look at it


def test_bundled_action_model_reads_strength_from_aggression():
    model = inference.default_model()
    assert {"all", "tag", "lag", "nit"} <= set(model.counts)
    for ctx in (inference.context(0, 0, True, 2), inference.context(1, 0, False, 2, 6, 2),
                inference.context(3, 1, inference.NORMAL, 2, 9, 0)):
        total = sum(model.likelihood(ctx, action) for action in range(inference.N_ACTIONS))
        assert total == pytest.approx(np.ones(inference.N_BUCKETS))
        sizes = sum(model.likelihood(ctx, action) for action in range(inference.BET_SMALL, inference.N_ACTIONS))
        assert model.likelihood(ctx, inference.AGGRESSIVE) == pytest.approx(sizes)
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
        assert "실시간 인식" in urllib.request.urlopen(base + "/").read().decode()
        info = json.loads(urllib.request.urlopen(base + "/api/info").read())["vision"]      # where the pictures go, for the page to say
        assert info == vision.info() and set(info) == {"provider", "service", "model"}

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
    # (street, raises, price, players left, table class, position): the button opens (paying 2 into 3 is
    # an expensive call), the small blind folds to the raise, and on the flop the big blind (first to
    # act of the two left) calls 8 into 21, a normal price
    # stacks of 100 big blinds: 198 behind a pot of 5 before the flop, 186 behind a pot of 29 when the bet is called
    assert seen[0][2] == (0, 0, 3, 1, 1, 3, 4) and seen[1][2] == (0, 1, 3, 1, 1, 4, 3) and seen[5][2] == (1, 1, 2, 0, 1, 0, 3)
    assert [a[8] for a in hand["actions"][:3]] == [198, 194, 194]
    assert seen[4][2][5] == 2                                                         # the button bets last to act
    # chips: the open adds 6 to a pot of 3, the flop bet 8 to 13, the river bet 20 to 29: all medium-sized
    assert [(a[3], a[4]) for a in hand["actions"] if a[2] == "raise"] == [(6, 3), (8, 13), (20, 29)]
    assert [code for _, _, _, code in seen if code > inference.PASSIVE] == [inference.BET_MEDIUM] * 3
    assert hand["actions"][2][3:5] == (4, 9)                                          # the big blind calls 4 more

    assert phh.parse_hand(PHH_HAND.replace("'NT'", "'FT'")) is None                   # not no-limit hold'em
    assert phh.parse_hand(PHH_HAND.replace("p1 f", "p1 xyz")) is None                 # unknown action
    assert phh.parse_hand(PHH_HAND.replace("[1, 2, 0]", "[1, 2, -1]")) is None        # a negative blind: pot unknown
    clash = phh.parse_hand(PHH_HAND.replace("p2 sm QhJh", "p2 sm QsJh"))              # shows a card that is on the board
    assert 1 not in clash["holes"] and 2 in clash["holes"]
    assert phh.stake_label("", 0.25) == "nl25-50" and phh.stake_label("", 10) == "nl1000"

    # heads-up the blinds are listed small, big but the last player (the button) posts the small one
    assert phh.parse_hand(PHH_HAND.replace("[200, 200, 200]", "[200, 200]")) is None  # a stack is missing
    unknown = phh.parse_hand(PHH_HAND.replace("[200, 200, 200]", "[inf, inf, inf]"))  # some sites do not log stacks
    assert unknown["actions"][0][8] == float("inf")
    assert {ctx[6] for _, _, ctx, _ in inference.observed(unknown["actions"], 3, 2)} == {inference.ANY_DEPTH}
    short = phh.parse_hand(PHH_HAND.replace("[200, 200, 200]", "[200, 200, 14]"))     # the button has 14: the flop bet of 8 is all of it
    seen_short = inference.observed(short["actions"], 3, 2)
    assert seen_short[4][3] == inference.ALL_IN and seen_short[5][2][6] == inference.NOTHING_BEHIND
    duel = phh.parse_hand("variant = 'NT'\nblinds_or_straddles = [5, 10]\nstarting_stacks = [1000, 1000]\n"
                          "actions = ['d dh p1 ????', 'd dh p2 ????', 'p2 cc', 'p1 cc', 'd db 2c3d4h', 'p1 cc', 'p2 f']\n")
    assert [(a[1], a[2], a[5]) for a in duel["actions"]] == [(1, "call", 5), (0, "check", 0), (0, "check", 0), (1, "fold", 0)]
    free_fold = inference.observed(duel["actions"], duel["n_players"], duel["button"])[-1]
    assert free_fold[2][4:6] == (0, 2)                                                 # heads-up table, the button acts last
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
    assert "coarse" in steps[0] and "unsized" in steps[-1] and not any("full" in step for step in steps)
    assert learned.counts["all"].sum() == pytest.approx(truth.counts["all"].sum())
    fits = {name: inference.action_fit(full, model)["log_likelihood"] for name, model in
            (("truth", truth), ("learned", learned), ("naive", naive))}
    assert fits["truth"] >= fits["learned"] > fits["naive"]


# ---- reading a hand off pictures of the table ------------------------------------------

def _played(kinds, n, stacks, seed, hands):
    """Hands the engine played -> (result, stacks, button) for each."""
    rng = random.Random(seed)
    bots = [make_bot(kinds[i % len(kinds)], seed=seed * 100 + i) for i in range(n)]
    for index in range(hands):
        chips = [rng.choice(stacks) for _ in range(n)]
        yield play_hand(bots, chips, index % n, 5, 10, rng), chips, index % n


@pytest.mark.parametrize("kinds, n, stacks", [
    (("random", "maniac", "station"), 6, (1000,)),              # raises and re-raises, everybody deep
    (("maniac", "random"), 2, (1000,)),                         # heads-up: the button posts the small blind
    (("maniac", "random", "lag"), 6, (60, 150, 300, 1000, 2500)),   # short stacks: all-ins, calls for less, side pots
    (("tag", "lag", "nit", "random"), 9, (400, 1000)),
])
def test_a_picture_per_decision_gives_the_hand_back_exactly(kinds, n, stacks):
    checked = 0
    for result, chips, button in _played(kinds, n, stacks, seed=n, hands=40):
        for hero in range(n):
            shots = table.frames(result, chips, hero, 5)
            decisions = [i for i, a in enumerate(result.actions) if a[1] == hero]
            assert len(shots) == len(decisions)
            for k, upto in enumerate(decisions):
                hand = table.reconstruct(shots[:k + 1])
                seat_of = lambda seat: int(hand["names"][seat][1:])        # frames() names the players p0, p1, ...
                got = [(seat_of(seat), street, ctx, code) for seat, street, ctx, code in
                       inference.observed(hand["actions"], n, hand["button"], hand["absent"])]
                # positions, players left, prices, bet sizes, all-ins and stack depths: all as the engine logged them
                assert got == inference.observed(result.actions[:upto], n, button)
                assert seat_of(hand["button"]) == button and seat_of(hand["hero_seat"]) == hero
                assert hand["pot"] == pytest.approx(result.actions[upto][4])
                assert hand["to_call"] == pytest.approx(result.actions[upto][5])
                checked += 1
    assert checked > 200


def test_a_single_picture_reads_the_street_being_played():
    exact = total = 0
    for result, chips, button in _played(("tag", "lag", "nit"), 6, (1000,), seed=3, hands=60):
        for hero in range(6):
            shots = table.frames(result, chips, hero, 5)
            for k, upto in enumerate(i for i, a in enumerate(result.actions) if a[1] == hero):
                street = result.actions[upto][0]
                truth = [x for x in inference.observed(result.actions[:upto], 6, button) if x[1] == street]
                if any(seat == hero for seat, _, _, _ in truth):
                    continue                            # the hero has acted on this street before: one picture cannot show how
                hand = table.reconstruct(shots[k:k + 1])
                got = [(int(hand["names"][seat][1:]), code) for seat, st, _, code in
                       inference.observed(hand["actions"], 6, hand["button"], hand["absent"]) if st == street]
                total += 1
                exact += got == [(seat, code) for seat, _, _, code in truth]
                assert bool(hand["notes"]) == (street > 0)      # it says so when earlier streets are missing
                assert all(a[0] == street for a in hand["actions"])
    assert total > 150 and exact / total > 0.97


def _shot(**changes):
    """Six-handed, blinds 1/2, seats clockwise from the button: the cutoff has opened to 6, everybody else has
    folded, and the hero in the big blind is to act."""
    seats = [{"name": name, "stack": stack, "bet": bet, "in_hand": live} for name, stack, bet, live in (
        ("btn", 200, 0, False), ("sb", 199, 1, False), ("me", 198, 2, True), ("utg", 200, 0, False), ("hj", 200, 0, False),
        ("co", 194, 6, True))]
    seats[0]["button"], seats[2]["hero"] = True, True
    return dict({"hero": "AsKd", "board": "", "pot": 9, "big_blind": 2, "seats": seats}, **changes)


def test_positions_and_actions_come_from_what_the_table_shows():
    hand = table.reconstruct([_shot()])
    assert (hand["n_players"], hand["button"], hand["hero_seat"], hand["big_blind"]) == (6, 0, 2, 2)
    assert [(a[1], a[2]) for a in hand["actions"]] == [(3, "fold"), (4, "fold"), (5, "raise"), (0, "fold"), (1, "fold")]
    assert hand["live"] == [2, 5] and hand["pot"] == 9 and hand["to_call"] == 4
    seats = table.describe(hand)
    assert [seat["position"] for seat in seats] == ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
    assert seats[5]["actions"] == "프리플랍 레이즈 3.0bb (팟의 80%)" and seats[2]["hero"] and not seats[0]["in_hand"]
    opened = inference.observed(hand["actions"], 6, 0)[2]
    assert opened[2][5] == inference.POSITIONS.index("co") and opened[3] == inference.BET_MEDIUM
    assert opened[2][6] == inference.VERY_DEEP                               # 194 behind a pot of 5

    # the next picture: I called, and on the flop the cutoff has bet 10 into 13
    flop = _shot(board="Qs7h2d", pot=23)
    for seat, (stack, bet, live) in zip(flop["seats"], ((200, 0, False), (199, 0, False), (194, 0, True),
                                                         (200, 0, False), (200, 0, False), (184, 10, True))):
        seat.update(stack=stack, bet=bet, in_hand=live)
    hand = table.reconstruct([_shot(), flop])
    assert [(a[0], a[1], a[2]) for a in hand["actions"]][5:] == [
        (0, 2, "call"), (1, 2, "check"), (1, 5, "raise")]                     # I act first on the flop, so I checked
    assert hand["pot"] == 23 and hand["to_call"] == 10 and not hand["notes"]
    analysis, hand = table.analyze([_shot(), flop], trials=5000, seed=1)
    assert analysis["n_players"] == 2 and analysis["pot_odds"]["required_equity"] == pytest.approx(10 / 33)
    assert analysis["ranges"][0]["seat"] == 5 and analysis["ranges"][0]["label"].startswith("컷오프: 프리플랍 레이즈")
    assert "플랍 벳 5.0bb (팟의 77%)" in analysis["ranges"][0]["label"]
    assert any("컷오프" in line for line in table.summary(hand)) and "승률" in advisor.report(analysis)

    # the flop picture alone: what happened before the flop is not there to be read
    alone = table.reconstruct([flop])
    assert [(a[1], a[2]) for a in alone["actions"]] == [(2, "check"), (5, "raise")]
    assert alone["absent"] == [0, 1, 3, 4] and "플랍 이전" in alone["notes"][0]
    assert inference.observed(alone["actions"], 6, 0, alone["absent"])[1][2][3] == 0     # two players left


def test_snapshots_are_checked_and_put_in_order():
    # seats given in any order with their places on the picture are sorted clockwise
    places = {"btn": (0.5, 0.9), "sb": (0.15, 0.7), "me": (0.15, 0.3), "utg": (0.5, 0.1), "hj": (0.85, 0.3), "co": (0.85, 0.7)}
    shuffled = _shot()
    for seat in shuffled["seats"]:
        seat["x"], seat["y"] = places[seat["name"]]
    shuffled["seats"] = shuffled["seats"][::-1][2:] + shuffled["seats"][::-1][:2]
    hand = table.reconstruct([shuffled])
    order = hand["names"][hand["button"]:] + hand["names"][:hand["button"]]
    assert order == ["btn", "sb", "me", "utg", "hj", "co"]
    assert [(hand["names"][a[1]], a[2]) for a in hand["actions"]] == [
        ("utg", "fold"), ("hj", "fold"), ("co", "raise"), ("btn", "fold"), ("sb", "fold")]
    # a player who sits out is not dealt in; a button left at such a seat counts for the player before it
    away = _shot()
    away["seats"][0].update(sitting_out=True)
    away["seats"].insert(0, dict(name="gone", stack=50, bet=0, in_hand=False, sitting_out=True))
    assert table.normalize(away)["button"] == 4 and len(table.normalize(away)["seats"]) == 5
    assert table.normalize(_shot(big_blind=2, pot=None))["pot"] is None
    for bad, message in ((_shot(hero="As"), "내 카드"), (_shot(board="QsJs"), "보드"), (_shot(hero="AsAs"), "duplicate"),
                         (_shot(board="AsQd2c"), "같은 카드"), (_shot(seats=_shot()["seats"][:1]), "플레이어")):
        with pytest.raises(ValueError, match=message):
            table.normalize(bad)
    two_buttons = _shot()
    two_buttons["seats"][3]["button"] = True
    with pytest.raises(ValueError, match="딜러 버튼"):
        table.normalize(two_buttons)
    with pytest.raises(ValueError, match="순서"):
        table.reconstruct([_shot(board="Qs7h2d"), _shot()])                 # a later picture shows an earlier street
    assert table.reconstruct([_shot(), _shot()])["actions"] == table.reconstruct([_shot()])["actions"]    # the same picture twice
    # stacks that could not be read: actions are still found, depth and all-ins are left out, and it says so
    blind = _shot()
    for seat in blind["seats"]:
        seat["stack"] = None
    hand = table.reconstruct([blind])
    assert [a[2] for a in hand["actions"]] == ["fold", "fold", "raise", "fold", "fold"] and "스택" in hand["notes"][0]
    assert {ctx[6] for _, _, ctx, _ in inference.observed(hand["actions"], 6, 0)} == {inference.ANY_DEPTH}
    # the stakes are not on the picture: the big blind is taken from the chips in front, and it says so
    guessed = table.reconstruct([_shot(big_blind=None)])
    assert guessed["big_blind"] == 2 and "빅 블라인드" in guessed["notes"][0]
    # a picture taken while somebody before me still has to act: the replay stops there and makes nothing up
    early = _shot()
    early["seats"][1]["in_hand"] = True
    hand = table.reconstruct([early])
    assert [a[1] for a in hand["actions"]] == [3, 4, 5, 0] and hand["live"] == [1, 2, 5]


def test_vision_answers_become_snapshots(monkeypatch):
    answer = {"is_poker_table": True, "hero_cards": ["as", "10d"], "board": ["Q♠", "7h", "2D"], "pot": 23, "pot_includes_bets": None,
              "big_blind": 2, "seats": [
                  {"name": "me", "x": 0.5, "y": 0.9, "stack": 194, "bet": 0, "in_hand": True, "is_hero": True, "has_button": False,
                   "sitting_out": False, "all_in": False},
                  {"name": "co", "x": 0.5, "y": 0.1, "stack": None, "bet": 10, "in_hand": True, "is_hero": False, "has_button": True,
                   "sitting_out": False, "all_in": True}]}
    snapshot = vision.to_snapshot(answer)
    assert (snapshot["hero"], snapshot["board"], snapshot["pot_includes_bets"]) == ("AsTd", "Qs7h2d", True)
    assert snapshot["seats"][1]["stack"] == 0 and snapshot["seats"][1]["button"] and snapshot["seats"][0]["hero"]
    assert table.normalize(snapshot)["street"] == 1
    with pytest.raises(vision.VisionError, match="홀덤 테이블"):
        vision.to_snapshot({"is_poker_table": False, "seats": []})
    with pytest.raises(vision.VisionError):
        vision.to_snapshot(["not", "an", "object"])

    sent = {}

    class Reply:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, *_):
            return json.dumps(self.payload).encode()

    def answer_with(payload):
        def fake(request, timeout=None):
            sent.update(url=request.full_url, headers=dict(request.header_items()), body=json.loads(request.data))
            return Reply(payload)
        monkeypatch.setattr(vision.urllib.request, "urlopen", fake)

    answer_with({"status": "completed", "steps": [{"type": "thought"}, {"type": "model_output", "content": [
        {"type": "text", "text": json.dumps(answer)}]}]})
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("POKERLAB_VISION_PROVIDER", "gemini")                # the Gemini call is what is under test here
    assert vision.recognize(b"\x89PNG...", "image/png") == snapshot
    assert sent["url"] == vision.URL and sent["headers"]["X-goog-api-key"] == "test-key"
    assert sent["body"]["model"] == vision.DEFAULT_MODEL and sent["body"]["store"] is False
    assert [part["type"] for part in sent["body"]["input"]] == ["text", "image"]
    assert sent["body"]["input"][1]["mime_type"] == "image/png" and sent["body"]["response_format"]["schema"] == vision.SCHEMA
    answer_with({"status": "failed", "steps": []})
    with pytest.raises(vision.VisionError, match="답을 주지"):
        vision.recognize(b"x", "image/png")
    answer_with({"output_text": "not json"})
    with pytest.raises(vision.VisionError, match="JSON"):
        vision.recognize(b"x", "image/png")

    def refuse(code):
        def fake(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, code, "no", {}, __import__("io").BytesIO(b"{}"))
        monkeypatch.setattr(vision.urllib.request, "urlopen", fake)

    refuse(403)
    with pytest.raises(vision.VisionError, match="거부"):
        vision.recognize(b"x", "image/png")
    refuse(429)
    with pytest.raises(vision.VisionError, match="한도"):
        vision.recognize(b"x", "image/png")
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    refuse(400)
    with pytest.raises(vision.VisionError, match="GEMINI_API_KEY"):
        vision.recognize(b"x", "image/png")
    with pytest.raises(vision.VisionError, match="형식"):
        vision.recognize(b"x", "image/gif")
    with pytest.raises(vision.VisionError, match="비어"):
        vision.recognize(b"", "image/png")


def test_frames_of_one_window_through_the_web_api():
    flop = _shot(board="Qs7h2d", pot=23)
    for seat, (stack, bet, live) in zip(flop["seats"], ((200, 0, False), (199, 0, False), (194, 0, True),
                                                         (200, 0, False), (200, 0, False), (184, 10, True))):
        seat.update(stack=stack, bet=bet, in_hand=live)
    web._SOURCES.clear()
    web.capture_request({"source": "T", "snapshot": _shot(), "trials": 5000, "model": "online"})
    out = web.capture_request({"source": "T", "snapshot": flop, "trials": 5000, "model": "online"})["analysis"]
    json.dumps(out)
    assert out["table"]["hero"] == ["As", "Kd"] and out["table"]["board"] == ["Qs", "7h", "2d"]
    assert [seat["position"] for seat in out["table"]["seats"]] == ["BTN", "SB", "BB", "UTG", "HJ", "CO"]
    assert out["ranges"][0]["seat"] == 5 and out["table"]["live"] == 2
    assert (out["pot_odds"]["to_call"], out["pot_odds"]["pot"], out["unit"]) == (5, 11.5, "bb")      # in big blinds of 2
    with pytest.raises(ValueError):
        web.capture_request({"source": "T", "do": "state", "model": "/etc/passwd"})
    with pytest.raises(ValueError, match="이미지 데이터"):
        web.capture_request({"source": "U", "image": "not base64!"})
    assert set(web.ROUTES) == {"/api/analyze", "/api/capture"}
    web._SOURCES.clear()


# ---- following a window as it is played ------------------------------------------------

@pytest.mark.parametrize("kinds, n, stacks", [
    (("random", "maniac", "station"), 6, (1000,)),
    (("maniac", "random"), 2, (1000,)),
    (("maniac", "random", "lag"), 6, (60, 150, 300, 1000, 2500)),
])
def test_a_running_capture_gives_the_hand_back(kinds, n, stacks):
    """Frames at every moment of a hand, as a capture of the window sees them."""
    checked = 0
    for result, chips, button in _played(kinds, n, stacks, seed=n + 1, hands=30):
        for hero in range(n):
            mine = [i for i, a in enumerate(result.actions) if a[1] == hero]
            if not mine:
                continue
            upto, street = mine[-1], result.actions[mine[-1]][0]
            truth = inference.observed(result.actions[:upto], n, button)

            def read(shots):
                hand = table.reconstruct(shots)
                seat_of = lambda seat: int(hand["names"][seat][1:])
                return hand, [(seat_of(seat), st, ctx, code) for seat, st, ctx, code in
                              inference.observed(hand["actions"], n, hand["button"], hand["absent"])]

            # the program shows whose turn it is: exact, also when every other frame is missed
            marked = table.frames(result, chips, hero, 5, every=True)
            hand, got = read(marked)
            assert got == truth and hand["actor"] == hand["hero_seat"] and hand["pot"] == pytest.approx(result.actions[upto][4])
            assert read(marked[::2] + marked[-1:])[1] == truth
            # it does not: everything that shows is still found, only checks nobody has acted after are left open
            last = upto
            while last and result.actions[last - 1][2] == "check" and result.actions[last - 1][0] == street:
                last -= 1
            hand, got = read(table.frames(result, chips, hero, 5, every=True, turns=False))
            assert got == inference.observed(result.actions[:last], n, button) and hand["actor"] is None
            checked += 1
    assert checked > 40


def test_follower_keeps_up_with_a_table(tmp_path):
    record = tmp_path / "hands.jsonl"
    follower = follow.Follower("lab A", str(record))
    assert follower.state()["phase"] == "waiting" and follower.analysis() is None
    played = list(_played(("human", "pro", "tag"), 6, (300, 1000, 1500), seed=5, hands=12))
    hero, followed = 2, 0
    for result, chips, button in played:
        shots = table.frames(result, chips, hero, 5, every=True)
        for shot in shots:
            assert follower.push(shot) and not follower.push(shot)       # a capture sends the same picture again and again
            assert follower.error is None and follower.note is None
        if not shots:
            continue
        followed += 1
        assert follower.hands == followed - 1                             # a new deal ended the hand before
        mine = [i for i, a in enumerate(result.actions) if a[1] == hero]
        hand = follower.hand
        got = [(int(hand["names"][seat][1:]), st, ctx, code) for seat, st, ctx, code in
               inference.observed(hand["actions"], 6, hand["button"], hand["absent"])]
        assert got == inference.observed(result.actions[:mine[-1]], 6, button)
    state = follower.state()
    assert state["phase"] == "hand" and state["hero_to_act"] and state["table"]["source"] == "capture"
    analysis = follower.analysis(trials=3000, seed=1)
    assert 0 < analysis["equity"] < 1 and analysis is follower.analysis(trials=3000, seed=1)     # kept until the table changes
    json.dumps(state), json.dumps(analysis)
    # between hands the window shows no cards: that is said, and the hand that was followed stays
    version = follower.version
    assert follower.push({"hero": "", "board": "", "seats": []}) and follower.error and follower.hand is not None
    assert follower.version == version + 1
    # a frame that cannot follow from the ones before (here: stacks that went up) restarts the hand from that frame
    broken = json.loads(json.dumps(follower.shots[-1]))
    for seat in broken["seats"]:
        seat["stack"] += 500
        seat["bet"] = 0
    broken["board"], broken["pot"] = "2c3c4c5d9h" if not broken["board"] else broken["board"], 9999
    follower.push(broken)
    assert follower.hand is not None and follower.error is None
    follower.close()
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(rows) == follower.hands >= followed and rows[0]["source"] == "lab A"
    assert all(row["snapshots"] and len(row["hero"]) == 4 for row in rows) and any(row["actions"] for row in rows)


def test_capture_sources_through_the_web_api(tmp_path, monkeypatch):
    web._SOURCES.clear()
    monkeypatch.setattr(web, "RECORD", str(tmp_path / "followed.jsonl"))
    result, chips, button = next(x for x in _played(("human", "pro", "tag"), 6, (1000,), seed=8, hands=30)
                                 if sum(a[1] == 0 for a in x[0].actions) >= 2)
    shots = table.frames(result, chips, 0, 5, every=True)
    assert web.capture_request({"do": "state", "source": "A"})["phase"] == "waiting"
    for shot in shots:
        state = web.capture_request({"source": "A", "label": "lab A", "snapshot": shot, "trials": 2000})
    assert state["phase"] == "hand" and state["name"] == "lab A" and state["hero_to_act"] and "equity" in state["analysis"]
    assert state["analysis"]["table"]["source"] == "capture"
    json.dumps(state)
    # a second window is a second source, followed on its own
    other = web.capture_request({"source": "B", "snapshot": shots[0], "trials": 2000})
    assert other["frames"] == 1 and web.capture_request({"do": "state", "source": "A"})["frames"] == state["frames"]
    assert set(web._SOURCES) == {"A", "B"}
    # a picture is read first: here by a stand-in that says what it was given
    seen = {}
    monkeypatch.setattr(vision, "recognize", lambda image, mime_type, notes=None: seen.update(n=len(image), m=mime_type, notes=notes) or shots[0])
    web.capture_request({"source": "C", "image": "data:image/png;base64,QUJD", "notes": "stacks are in chips"})
    assert seen == {"n": 3, "m": "image/png", "notes": "stacks are in chips"}
    assert web.capture_request({"do": "reset", "source": "A"})["phase"] == "waiting"
    assert web.capture_request({"do": "remove", "source": "B"}) == {"removed": "B"} and "B" not in web._SOURCES
    assert (tmp_path / "followed.jsonl").read_text().count("\n") == 2            # the hands of A and B went into the record
    for _ in range(web.MAX_SOURCES):
        try:
            web.capture_request({"do": "state", "source": f"extra {len(web._SOURCES)}"})
        except ValueError as error:
            assert "소스는" in str(error)
            break
    else:
        raise AssertionError("no limit on the number of sources")
    with pytest.raises(ValueError):
        web.capture_request({"do": "jump", "source": "A"})
    web._SOURCES.clear()


def test_a_recording_keeps_where_each_hand_began(tmp_path, monkeypatch):
    """Frames read off a video file say where they are in it; the record keeps where each hand began."""
    web._SOURCES.clear()
    record = tmp_path / "followed.jsonl"
    monkeypatch.setattr(web, "RECORD", str(record))
    ask = {"source": "V", "label": "session.mp4", "trials": 2000}
    web.capture_request({**ask, "snapshot": _shot(), "at": "0:05"})
    web.capture_request({**ask, "snapshot": _shot(pot=10), "at": "0:09"})            # the same hand, later on
    assert web._SOURCES["V"].started == "0:05"
    web.capture_request({**ask, "snapshot": _shot(hero="QhQd"), "at": "1:02"})       # new hole cards: a new hand
    assert web._SOURCES["V"].started == "1:02"
    web.capture_request({"source": "V", "do": "reset"})
    web.capture_request({"source": "W", "snapshot": _shot(), "trials": 2000})        # a window: no place in a recording
    web.capture_request({"source": "W", "do": "remove"})
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert [(row["source"], row.get("at"), row["hero"]) for row in rows] == [
        ("session.mp4", "0:05", "AsKd"), ("session.mp4", "1:02", "QhQd"), ("W", None, "AsKd")]
    assert web._SOURCES["V"].started is None
    web._SOURCES.clear()


def test_claude_as_the_reader(monkeypatch):
    """Claude, the default reader: the prompt and schema go to the Messages API through the anthropic package
    (here a stand-in for it that keeps what it was asked)."""
    import sys
    import types
    answer = {"is_poker_table": True, "hero_cards": ["As", "Kd"], "board": [], "pot": 3, "big_blind": 2, "hero_to_act": True, "seats": [
        {"x": 0.5, "y": 0.9, "stack": 199, "bet": 1, "in_hand": True, "is_hero": True, "has_button": True, "sitting_out": False,
         "all_in": False, "acting": True},
        {"x": 0.5, "y": 0.1, "stack": 198, "bet": 2, "in_hand": True, "is_hero": False, "has_button": False, "sitting_out": False,
         "all_in": False, "acting": False}]}
    asked, state = {}, {"reply": None, "error": None}
    fake = types.ModuleType("anthropic")

    class AnthropicError(Exception):
        pass

    class APIConnectionError(AnthropicError):
        pass

    class APIStatusError(AnthropicError):
        status_code, message = 500, "boom"

    for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "RateLimitError"):
        setattr(fake, name, type(name, (APIStatusError,), {}))
    fake.AnthropicError, fake.APIConnectionError, fake.APIStatusError = AnthropicError, APIConnectionError, APIStatusError

    def create(route):
        def call(**request):
            asked.update(route=route, request=request)
            if state["error"]:
                raise state["error"]
            return state["reply"]
        return call

    class Anthropic:
        def __init__(self, **options):
            asked["client"] = options
            self.messages = types.SimpleNamespace(create=create("messages"))
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=create("beta")))

    fake.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    for name in ("POKERLAB_VISION_REPLAY", "POKERLAB_VISION_MODEL", "POKERLAB_VISION_EFFORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("POKERLAB_VISION_PROVIDER", raising=False)              # nothing chosen: Claude
    assert vision.provider() == "claude"
    said = lambda reason, text: types.SimpleNamespace(stop_reason=reason, content=[
        types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=text)])
    state["reply"] = said("end_turn", json.dumps(answer))

    assert vision.info() == {"provider": "claude", "service": "Anthropic Claude", "model": "claude-opus-5-5"}
    snapshot = vision.recognize(b"x", "image/jpeg", notes="stacks are in chips")
    assert snapshot == vision.to_snapshot(answer) and table.normalize(snapshot)["actor"] is not None
    request = asked["request"]
    assert asked["route"] == "beta" and request["betas"] == [vision.CLAUDE_FALLBACK_BETA] and request["fallbacks"] == "default"
    assert request["model"] == "claude-opus-5-5" and request["max_tokens"] == vision.CLAUDE_MAX_TOKENS
    assert asked["client"] == {"timeout": 90, "max_retries": 1}                 # no key passed: the package finds ANTHROPIC_API_KEY itself
    monkeypatch.setenv("POKER_ANTHROPIC_API_KEY", "project-key")                # a key kept for this project comes first
    vision.recognize(b"x", "image/png")
    assert asked["client"]["api_key"] == "project-key"
    vision.recognize(b"x", "image/png", key="handed-in")
    assert asked["client"]["api_key"] == "handed-in"
    monkeypatch.delenv("POKER_ANTHROPIC_API_KEY")
    picture, words = request["messages"][0]["content"]
    assert picture == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "eA=="}}
    assert words["text"].startswith(vision.PROMPT) and words["text"].endswith("About the program in this picture: stacks are in chips")
    config = request["output_config"]
    assert config["effort"] == "low" and config["format"]["type"] == "json_schema"
    schema = config["format"]["schema"]                                         # every object closed, "or null" as anyOf
    seat = schema["properties"]["seats"]["items"]
    assert schema["additionalProperties"] is False and seat["additionalProperties"] is False
    assert schema["properties"]["pot"] == {"anyOf": [{"type": "number"}, {"type": "null"}]} and "type" not in seat["properties"]["name"]
    assert set(seat["required"]) == set(vision.SCHEMA["properties"]["seats"]["items"]["required"])
    assert "additionalProperties" not in vision.SCHEMA                          # the schema Gemini gets is left as it was

    # another model and effort; a key handed in; a model without the service's fallback goes the plain way
    monkeypatch.setenv("POKERLAB_VISION_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("POKERLAB_VISION_EFFORT", "medium")
    vision.recognize(b"x", "image/png", key="k")
    assert asked["route"] == "messages" and "fallbacks" not in asked["request"] and asked["client"]["api_key"] == "k"
    assert asked["request"]["model"] == "claude-haiku-5-5" and asked["request"]["output_config"]["effort"] == "medium"
    assert vision.info()["model"] == "claude-haiku-5-5"

    # what can go wrong is said in words, not raised as something else
    state["reply"] = said("refusal", "")
    with pytest.raises(vision.VisionError, match="읽지 않겠다고"):
        vision.recognize(b"x", "image/png")
    state["reply"] = said("max_tokens", "{")
    with pytest.raises(vision.VisionError, match="잘렸습니다"):
        vision.recognize(b"x", "image/png")
    state["reply"] = said("end_turn", "not json")
    with pytest.raises(vision.VisionError, match="JSON이 아닙니다"):
        vision.recognize(b"x", "image/png")
    state["reply"] = said("end_turn", json.dumps(answer))
    for error, words in ((fake.AuthenticationError(), "POKER_ANTHROPIC_API_KEY 또는 ANTHROPIC_API_KEY"), (fake.RateLimitError(), "한도"), (fake.NotFoundError(), "모델을 찾지"),
                         (APIStatusError(), "오류 500"), (APIConnectionError("down"), "연결하지 못했습니다"), (AnthropicError("no key"), "부르지 못했습니다")):
        state["error"] = error
        with pytest.raises(vision.VisionError, match=words):
            vision.recognize(b"x", "image/png")
    state["error"] = TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token, ..."')
    with pytest.raises(vision.VisionError, match="키가 없습니다"):               # the package found no key anywhere
        vision.recognize(b"x", "image/png")
    state["error"] = TypeError("something else")
    with pytest.raises(TypeError, match="something else"):
        vision.recognize(b"x", "image/png")
    state["error"] = None
    with pytest.raises(vision.VisionError, match="Claude는 PNG"):
        vision.recognize(b"x", "image/heic")
    monkeypatch.setitem(sys.modules, "anthropic", None)                         # the package is not installed
    with pytest.raises(vision.VisionError, match="anthropic 패키지"):
        vision.recognize(b"x", "image/png")
    monkeypatch.setenv("POKERLAB_VISION_PROVIDER", "somebody")
    with pytest.raises(vision.VisionError, match="POKERLAB_VISION_PROVIDER"):
        vision.recognize(b"x", "image/png")
    monkeypatch.setenv("POKERLAB_VISION_REPLAY", "anything.json")
    assert vision.info() == {"provider": "replay", "service": None, "model": None}


def test_recognizer_extras(tmp_path, monkeypatch):
    result, chips, button = next(x for x in _played(("human", "pro", "tag"), 6, (1000,), seed=9, hands=30)
                                 if sum(a[1] == 1 for a in x[0].actions) >= 2)
    shots = table.frames(result, chips, 1, 5, every=True)

    # the recognizer: whose turn it is, notes about the program, and replaying snapshots instead of asking the model
    answer = {"is_poker_table": True, "hero_cards": ["As", "Kd"], "board": [], "pot": 3, "hero_to_act": None, "seats": [
        {"x": 0.5, "y": 0.9, "stack": 199, "bet": 1, "in_hand": True, "is_hero": True, "has_button": True, "sitting_out": False,
         "all_in": False, "acting": False},
        {"x": 0.5, "y": 0.1, "stack": 198, "bet": 2, "in_hand": True, "is_hero": False, "has_button": False, "sitting_out": False,
         "all_in": False, "acting": True}]}
    snapshot = vision.to_snapshot(answer)
    assert snapshot["hero_to_act"] is False and [seat["acting"] for seat in snapshot["seats"]] == [False, True]
    read = table.normalize(snapshot)
    assert not read["seats"][read["actor"]]["hero"]                         # the seat that is marked, not the hero
    read = table.normalize(dict(snapshot, hero_to_act=True))
    assert read["actor"] == read["hero_seat"]
    snapshot["seats"][1]["acting"] = False
    assert table.normalize(snapshot)["actor"] is None                       # nothing on the screen says whose turn it is
    sent = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, *_):
            return json.dumps({"output_text": json.dumps(answer)}).encode()

    monkeypatch.setattr(vision.urllib.request, "urlopen", lambda request, timeout=None: sent.update(body=json.loads(request.data)) or Reply())
    monkeypatch.setenv("POKERLAB_VISION_PROVIDER", "gemini")                # Claude is the default reader: this is the other one
    monkeypatch.delenv("POKERLAB_VISION_EFFORT", raising=False)
    monkeypatch.delenv("POKERLAB_VISION_MODEL", raising=False)
    assert vision.info() == {"provider": "gemini", "service": "Google Gemini", "model": "gemini-4-hydrogen"}
    vision.recognize(b"x", "image/png", key="k", notes="stacks are in chips")
    assert sent["body"]["input"][0]["text"].endswith("About the program in this picture: stacks are in chips")
    assert sent["body"]["generation_config"] == {"thinking_level": "low"} and sent["body"]["store"] is False
    monkeypatch.setenv("POKERLAB_VISION_EFFORT", "medium")
    vision.recognize(b"x", "image/png", key="k")
    assert sent["body"]["generation_config"] == {"thinking_level": "medium"}
    monkeypatch.delenv("POKERLAB_VISION_EFFORT")
    vision.recognize(b"x", "image/png", key="k")
    assert sent["body"]["input"][0]["text"] == vision.PROMPT
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(shots[:2]))
    monkeypatch.setenv("POKERLAB_VISION_REPLAY", str(replay))
    assert [vision.recognize(b"x"), vision.recognize(b"y")] == shots[:2]
    with pytest.raises(vision.VisionError, match="더 없습니다"):
        vision.recognize(b"z")
