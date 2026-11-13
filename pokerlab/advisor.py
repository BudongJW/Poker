"""Turn a spot (my hand, the board, how many players, optionally pot and bet) into a readable assessment."""

from itertools import combinations

import numpy as np

from .cards import cards_str, hand_class, parse_cards
from .equity import equity
from .evaluator import CATEGORIES, CATEGORIES_KO, evaluate, evaluate_many
from .preflop import preflop_equity, ranking

STREET_KO = {0: "프리플랍", 3: "플랍", 4: "턴", 5: "리버"}
# (minimum edge, label); edge = how far equity sits between the fair share 1/N (0) and certainty (1)
VERDICTS = ((0.50, "매우 강함"), (0.25, "강함"), (0.08, "약간 우세"), (-0.05, "평균 수준"), (-1.0, "약함"))


def ahead_now(hero, board):
    """Share of random opponent hands hero beats / ties / loses to if the hand ended right now."""
    deck = [c for c in range(52) if c not in hero and c not in board]
    pairs = np.array(list(combinations(deck, 2)), dtype=np.int8)
    hands = np.concatenate([pairs, np.tile(np.array(board, dtype=np.int8), (len(pairs), 1))], axis=1)
    values = evaluate_many(hands)
    mine = evaluate(hero + board)
    return float((values < mine).mean()), float((values == mine).mean()), float((values > mine).mean())


def analyze(hero, board=(), n_players=2, pot=None, to_call=None, ranges=(), trials=200_000, seed=None):
    """All the numbers behind :func:`explain`, as a dict."""
    hero, board = parse_cards(hero), parse_cards(board)
    n_opp = n_players - 1
    result = equity(hero, board, n_opp, ranges=ranges, trials=trials, seed=seed)
    fair = 1.0 / n_players
    edge = (result.equity - fair) / (1.0 - fair)
    out = {
        "hero": hero, "board": board, "n_players": n_players, "hand_class": hand_class(*hero),
        "result": result, "fair_share": fair, "edge": edge,
        "verdict": next(label for floor, label in VERDICTS if edge >= floor),
        "ranged": bool(ranges),
    }
    if board:
        out["made_hand"] = evaluate(hero + board) >> 20
        out["ahead_now"] = ahead_now(hero, board)
    else:
        order = ranking(min(n_opp, 8))
        out["preflop_rank"] = order.index(out["hand_class"]) + 1
    if not ranges:      # how the same hand fares as the table fills up
        if board:
            out["by_opponents"] = {n: (result.equity if n == n_opp else
                                       equity(hero, board, n, trials=20_000, seed=seed).equity) for n in range(1, 9)}
        else:
            out["by_opponents"] = {n: preflop_equity(out["hand_class"], n) for n in range(1, 9)}
    if pot is not None and to_call:
        need = to_call / (pot + to_call)
        out["pot_odds"] = {"pot": pot, "to_call": to_call, "required_equity": need,
                           "ev_call": result.equity * (pot + to_call) - to_call,
                           "profitable": result.equity > need}
    return out


def explain(hero, board=(), n_players=2, pot=None, to_call=None, ranges=(), trials=200_000, seed=None):
    """Korean-language report for one spot."""
    a = analyze(hero, board, n_players, pot, to_call, ranges, trials, seed)
    r = a["result"]
    n_opp = n_players - 1
    lines = [f"[{STREET_KO[len(a['board'])]}] 내 핸드 {cards_str(a['hero'])} ({a['hand_class']})"
             + (f" | 보드 {cards_str(a['board'])}" if a["board"] else "")
             + f" | {n_players}명 (상대 {n_opp}명{', 레인지 지정' if a['ranged'] else ', 랜덤 핸드 가정'})"]
    method = "전수 계산" if r.exact else f"몬테카를로 {r.trials:,}회, 오차 ±{1.96 * r.stderr:.2%}"
    lines.append(f"승률(에퀴티) {r.equity:.1%}  = 승 {r.win:.1%} + 무 {r.tie:.1%} 분배 / 패 {r.lose:.1%}   ({method})")
    lines.append(f"판정: {a['verdict']}  (인원 기준 평균 승률 {a['fair_share']:.1%}의 {r.equity / a['fair_share']:.2f}배)")
    if "made_hand" in a:
        ahead, tied, behind = a["ahead_now"]
        lines.append(f"현재 족보: {CATEGORIES_KO[a['made_hand']]}  | 지금 당장 쇼다운이면 랜덤 핸드 1개 상대로 "
                     f"앞섬 {ahead:.1%} / 동률 {tied:.1%} / 뒤짐 {behind:.1%}")
    else:
        lines.append(f"프리플랍 순위: 169개 스타팅 핸드 중 {a['preflop_rank']}위 (상대 {min(n_opp, 8)}명 기준)")
    if len(a["board"]) < 5:
        dist = sorted(r.hand_dist.items(), key=lambda kv: -kv[1])
        lines.append("리버까지 갔을 때 내 최종 족보: " + ", ".join(
            f"{CATEGORIES_KO[CATEGORIES.index(name)]} {p:.1%}" for name, p in dist if p >= 0.005))
    if "by_opponents" in a:
        lines.append("상대 수별 승률: " + "  ".join(f"{n}명 {e:.0%}" for n, e in a["by_opponents"].items()))
    if "pot_odds" in a:
        o = a["pot_odds"]
        lines.append(f"팟 오즈: 팟 {o['pot']:g}에 {o['to_call']:g} 콜 -> 필요 승률 {o['required_equity']:.1%}, "
                     f"내 승률 {r.equity:.1%} -> 콜 기대값 {o['ev_call']:+.1f} ({'수익' if o['profitable'] else '손실'} 콜)")
    if not a["ranged"]:
        lines.append("주의: 상대가 랜덤 핸드라는 가정입니다. 베팅/레이즈한 상대는 더 강한 범위를 가지므로 실제 승률은 이보다 낮습니다.")
    return "\n".join(lines)
