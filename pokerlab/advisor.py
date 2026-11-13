"""Turn a spot (my hand, the board, how many players, what the opponents did, pot and bet sizes)
into numbers (:func:`analyze`), a Korean-language report (:func:`explain`) or JSON (:func:`to_json`)."""

from itertools import combinations

import numpy as np

from .cards import ALL_CLASSES, card_str, cards_str, class_combos, hand_class, parse_cards
from .equity import COMBO_INDEX, EquityResult, as_range, equity, next_card
from .evaluator import CATEGORIES, CATEGORIES_KO, evaluate, evaluate_many
from .inference import range_from_line
from .preflop import DEALS_PER_COMBO, MAX_OPPONENTS, heads_up_exact, preflop_equity, preflop_stderr, ranking

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


def range_summary(rng, top=8):
    """A range in words and numbers: how wide it is and how its weight spreads over starting hands."""
    vector = rng.vector()
    total = vector.sum()
    classes = {cls: float(vector[[COMBO_INDEX[a, b] for a, b in class_combos(cls)]].sum() / total)
               for cls in ALL_CLASSES}
    # "effective" combos: 1,326 for a uniform range, fewer the more the weight concentrates
    return {"width": float(total * total / (vector * vector).sum() / 1326),
            "top": sorted(classes.items(), key=lambda item: -item[1])[:top], "classes": classes}


def _next_cards(hero, board, n_opp, ranges, current, seed):
    """Equity after each possible next card, grouped by the hand it gives hero."""
    cards = next_card(hero, board, n_opp, ranges, seed=seed)
    groups = {}
    for card, result in cards.items():
        groups.setdefault(evaluate(hero + board + [card]) >> 20, []).append((card, result.equity))
    return {
        "cards": {card: result.equity for card, result in cards.items()},
        "groups": sorted(({"made_hand": cat, "cards": [c for c, _ in members],
                           "equity": float(np.mean([e for _, e in members]))} for cat, members in groups.items()),
                         key=lambda g: -g["equity"]),
        "better": sum(1 for r in cards.values() if r.equity >= current + 0.10),
        "worse": sum(1 for r in cards.values() if r.equity <= current - 0.10),
    }


def analyze(hero, board=(), n_players=2, pot=None, to_call=None, ranges=(), trials=200_000, seed=None,
            precision=None, bet=None, lines=(), profile="all"):
    """All the numbers behind :func:`explain`, as a dict.

    ranges: known ranges of opponents; lines: action lines of opponents (see
    :func:`pokerlab.inference.parse_line`) from which a range is inferred. Opponents with neither
    hold random hands. ``precision`` (e.g. 0.001) makes Monte Carlo run until that 95% interval.
    """
    hero, board = parse_cards(hero), parse_cards(board)
    n_opp = n_players - 1
    labels = ["지정 레인지"] * len(ranges) + list(lines)
    ranges = [as_range(r) for r in ranges] + [range_from_line(line, hero, board, n_players, profile=profile)
                                              for line in lines]
    cls = hand_class(*hero)
    if not board and not ranges and n_opp == 1:                 # enumerated once and stored in the table
        win, tie, _ = heads_up_exact(cls)
        dist = equity(hero, trials=100_000, seed=seed).hand_dist
        result = EquityResult(win, tie, win + tie / 2, DEALS_PER_COMBO, True, 0.0, 1, dist)
    else:
        result = equity(hero, board, n_opp, ranges=ranges, trials=trials, seed=seed, precision=precision)
    fair = 1.0 / n_players
    edge = (result.equity - fair) / (1.0 - fair)
    out = {
        "hero": hero, "board": board, "n_players": n_players, "hand_class": cls,
        "result": result, "fair_share": fair, "edge": edge,
        "verdict": next(label for floor, label in VERDICTS if edge >= floor),
        "ranged": bool(ranges),
        "ranges": [dict(range_summary(r.without(hero + board)), label=label) for r, label in zip(ranges, labels)],
    }
    if board:
        out["made_hand"] = evaluate(hero + board) >> 20
        out["ahead_now"] = ahead_now(hero, board)
    else:
        out["preflop_rank"] = ranking(min(n_opp, MAX_OPPONENTS)).index(cls) + 1
    if len(board) in (3, 4):
        out["next_cards"] = _next_cards(hero, board, n_opp, ranges, result.equity, seed)
    if not ranges:      # how the same hand fares as the table fills up
        if board:
            out["by_opponents"] = {n: (result.equity if n == n_opp else
                                       equity(hero, board, n, trials=20_000, seed=seed).equity) for n in range(1, 10)}
        else:
            out["by_opponents"] = {n: preflop_equity(cls, n) for n in range(1, 10)}
            out["table_stderr"] = preflop_stderr(cls, n_opp)
    if pot is not None and to_call:
        need = to_call / (pot + to_call)
        out["pot_odds"] = {"pot": pot, "to_call": to_call, "required_equity": need,
                           "ev_call": result.equity * (pot + to_call) - to_call,
                           "profitable": result.equity > need}
    if pot is not None and bet:
        out["bet"] = {"pot": pot, "bet": bet,
                      "bluff_breakeven_folds": bet / (pot + bet),          # folds needed if hero never wins when called
                      "called_breakeven_equity": bet / (pot + 2 * bet),    # equity needed when the bet is called
                      "ev_if_called": result.equity * (pot + 2 * bet) - bet,
                      "ev_check": result.equity * pot}
    return out


def to_json(analysis):
    """The dict from :func:`analyze` with cards as text and plain numbers, ready for ``json.dumps``."""
    r = analysis["result"]
    out = {k: v for k, v in analysis.items() if k not in ("hero", "board", "result", "next_cards", "ranges")}
    out.update(hero=[card_str(c) for c in analysis["hero"]], board=[card_str(c) for c in analysis["board"]],
               win=r.win, tie=r.tie, lose=r.lose, equity=r.equity, exact=r.exact, trials=r.trials,
               ci95=r.ci95, hand_dist=r.hand_dist)
    if "made_hand" in analysis:
        out["made_hand"] = CATEGORIES[analysis["made_hand"]]
    out["ranges"] = [{"label": s["label"], "width": s["width"], "top": [list(item) for item in s["top"]],
                      "classes": s["classes"]} for s in analysis["ranges"]]
    if "next_cards" in analysis:
        nc = analysis["next_cards"]
        out["next_cards"] = {
            "cards": {card_str(c): e for c, e in nc["cards"].items()}, "better": nc["better"], "worse": nc["worse"],
            "groups": [{"made_hand": CATEGORIES[g["made_hand"]], "equity": g["equity"],
                        "cards": [card_str(c) for c in g["cards"]]} for g in nc["groups"]]}
    return out


def explain(hero, board=(), n_players=2, pot=None, to_call=None, ranges=(), trials=200_000, seed=None,
            precision=None, bet=None, lines=(), profile="all"):
    """Korean-language report for one spot."""
    a = analyze(hero, board, n_players, pot, to_call, ranges, trials, seed, precision, bet, lines, profile)
    r = a["result"]
    n_opp = n_players - 1
    assumed = "레인지 지정" if a["ranged"] else "랜덤 핸드 가정"
    lines_out = [f"[{STREET_KO[len(a['board'])]}] 내 핸드 {cards_str(a['hero'])} ({a['hand_class']})"
                 + (f" | 보드 {cards_str(a['board'])}" if a["board"] else "")
                 + f" | {n_players}명 (상대 {n_opp}명, {assumed})"]
    method = "전수 계산, 오차 없음" if r.exact else f"몬테카를로 {r.trials:,}회, 오차 ±{r.ci95:.2%}p"
    lines_out.append(f"승률(에퀴티) {r.equity:.2%}  = 승 {r.win:.2%} + 무 {r.tie:.2%} 분배 / 패 {r.lose:.2%}   ({method})")
    lines_out.append(f"판정: {a['verdict']}  (인원 기준 평균 승률 {a['fair_share']:.1%}의 {r.equity / a['fair_share']:.2f}배)")
    for i, s in enumerate(a["ranges"], 1):
        top = ", ".join(f"{cls} {share:.0%}" for cls, share in s["top"][:6])
        lines_out.append(f"상대{i} 레인지 ({s['label']}): 전체 핸드의 약 {s['width']:.0%} 폭 | 비중 상위 {top}")
    if "made_hand" in a:
        ahead, tied, behind = a["ahead_now"]
        lines_out.append(f"현재 족보: {CATEGORIES_KO[a['made_hand']]}  | 지금 당장 쇼다운이면 랜덤 핸드 1개 상대로 "
                         f"앞섬 {ahead:.1%} / 동률 {tied:.1%} / 뒤짐 {behind:.1%}")
    else:
        lines_out.append(f"프리플랍 순위: 169개 스타팅 핸드 중 {a['preflop_rank']}위 (상대 {min(n_opp, MAX_OPPONENTS)}명 기준)")
    if len(a["board"]) < 5:
        dist = sorted(r.hand_dist.items(), key=lambda kv: -kv[1])
        lines_out.append("리버까지 갔을 때 내 최종 족보: " + ", ".join(
            f"{CATEGORIES_KO[CATEGORIES.index(name)]} {p:.1%}" for name, p in dist if p >= 0.005))
    if "next_cards" in a:
        nc = a["next_cards"]
        groups = ", ".join(f"{CATEGORIES_KO[g['made_hand']]} {len(g['cards'])}장 → {g['equity']:.0%}" for g in nc["groups"])
        lines_out.append(f"다음 카드별 승률: {groups}")
        lines_out.append(f"  승률이 10%p 이상 오르는 카드 {nc['better']}장, 10%p 이상 떨어지는 카드 {nc['worse']}장 "
                         f"(전체 {len(nc['cards'])}장)")
    if "by_opponents" in a:
        lines_out.append("상대 수별 승률: " + "  ".join(f"{n}명 {e:.0%}" for n, e in a["by_opponents"].items()))
    if "pot_odds" in a:
        o = a["pot_odds"]
        lines_out.append(f"팟 오즈: 팟 {o['pot']:g}에 {o['to_call']:g} 콜 -> 필요 승률 {o['required_equity']:.1%}, "
                         f"내 승률 {r.equity:.1%} -> 콜 기대값 {o['ev_call']:+.1f} ({'수익' if o['profitable'] else '손실'} 콜)")
    if "bet" in a:
        b = a["bet"]
        lines_out.append(f"베팅: 팟 {b['pot']:g}에 {b['bet']:g} 베팅 -> 블러프 손익분기 폴드율 {b['bluff_breakeven_folds']:.1%}, "
                         f"콜 받을 때 필요 승률 {b['called_breakeven_equity']:.1%}")
        lines_out.append(f"  현재 승률 그대로 콜 받으면 기대값 {b['ev_if_called']:+.1f}, 체크로 쇼다운이면 {b['ev_check']:+.1f}")
    if not a["ranged"]:
        lines_out.append("주의: 상대가 랜덤 핸드라는 가정입니다. 상대의 액션을 --villain 으로 주면 레인지를 추정해 다시 계산합니다.")
    return "\n".join(lines_out)
