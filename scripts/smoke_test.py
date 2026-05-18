"""봇 코어 알고리즘 스모크 테스트 (PokerStars 불필요).

이 PC처럼 PokerStars 미설치 환경에서 봇의 핵심 로직(몬테카를로 equity,
hand evaluator, outs calculator)이 정상 작동하는지 빠르게 검증한다.

`poker.tests.__init__` 이 PyQt6/tensorflow 등 풀 스택을 끌어와서
일반 pytest 수집이 막히므로, 핵심 모듈만 직접 import해서 호출한다.

실행: ``.venv/Scripts/python.exe scripts/smoke_test.py``
의존성: numpy, pandas, requests (montecarlo_python -> helper -> pandas/requests)
"""
from __future__ import annotations

import os
import sys
import time
import traceback

# repo 루트를 sys.path에 추가
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _result(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}  {detail}")
    return (name, ok, detail)


def test_hand_evaluator():
    """페어 비교: KH 6C vs 3H 3S 4H 4S 8S 8C QH 보드"""
    from poker.decisionmaker import montecarlo_python as mc
    sim = mc.MonteCarlo()
    sim.player_final_cards = [
        ['3H', '3S', '4H', '4S', '8S', '8C', 'QH'],
        ['KH', '6C', '4H', '4S', '8S', '8C', 'QH'],
    ]
    winner_idx = sim.player_final_cards.index(
        sim.eval_best_hand(sim.player_final_cards)[0])
    return _result(
        "hand evaluator (two pair vs higher kicker)",
        winner_idx == 1,
        f"winner_idx={winner_idx} (expected 1)",
    )


def test_outs_inside_straight():
    """Gutshot straight draw → 4 outs"""
    from poker.decisionmaker.outs_calculator import Outs_Calculator
    oc = Outs_Calculator()
    my_cards = ['JH', '9C']
    board = ['QS', '8D', '4C']
    outs = oc.evaluate_hands(my_cards, board, oc)
    return _result(
        "outs calculator (inside straight)",
        outs == 4,
        f"outs={outs} (expected 4)",
    )


def test_outs_made_hand_zero():
    """이미 풀하우스 → outs 0"""
    from poker.decisionmaker.outs_calculator import Outs_Calculator
    oc = Outs_Calculator()
    my_cards = ['6S', '6D']
    board = ['JS', 'JH', 'JD']
    outs = oc.evaluate_hands(my_cards, board, oc)
    return _result(
        "outs calculator (full house, no outs)",
        outs == 0,
        f"outs={outs} (expected 0)",
    )


def test_montecarlo_equity():
    """AA vs 무작위 1명 — equity 약 0.85 근처 기대.

    montecarlo_python.MonteCarlo.run_montecarlo 는 시간 제한 기반이라
    짧게 돌려도 추정치는 충분히 안정.
    """
    from poker.decisionmaker import montecarlo_python as mc
    sim = mc.MonteCarlo()

    # API: run_montecarlo(my_cards, cards_on_table, player_amount, ui, mc_iterations,
    #                     timeout, ghost_cards, opponent_range)
    my_cards = [['AS', 'AH']]
    board = []
    t0 = time.perf_counter()
    sim.run_montecarlo(
        logger=None,
        original_player_card_list=my_cards,
        original_table_card_list=board,
        player_amount=2,
        ui=None,
        max_runs=1000,
        timeout=4.0,
        ghost_cards='',
        opponent_range=1.0,
    )
    dt = time.perf_counter() - t0
    eq = sim.equity
    ok = 0.78 <= eq <= 0.92  # 이론값 ~0.851, 분산 고려 여유
    return _result(
        "montecarlo equity (AA vs random)",
        ok,
        f"equity={eq:.3f}, time={dt:.2f}s (expected 0.78-0.92)",
    )


def test_montecarlo_heads_up_kk_vs_aa_low_equity():
    """KK 가 보드 A 깔리면 equity 급락 — sanity 체크"""
    from poker.decisionmaker import montecarlo_python as mc
    sim = mc.MonteCarlo()
    my_cards = [['KS', 'KH']]
    board = ['AC', '7D', '2H']
    sim.run_montecarlo(
        logger=None,
        original_player_card_list=my_cards,
        original_table_card_list=board,
        player_amount=2,
        ui=None,
        max_runs=1500,
        timeout=4.0,
        ghost_cards='',
        opponent_range=1.0,
    )
    eq = sim.equity
    # A 보드에서 KK vs 무작위 핸드 이론값 약 0.78~0.82
    ok = 0.72 <= eq <= 0.88
    return _result(
        "montecarlo equity (KK on A-high board)",
        ok,
        f"equity={eq:.3f} (expected 0.72-0.88)",
    )


TESTS = [
    test_hand_evaluator,
    test_outs_inside_straight,
    test_outs_made_hand_zero,
    test_montecarlo_equity,
    test_montecarlo_heads_up_kk_vs_aa_low_equity,
]


def main() -> int:
    print("== DeeperMind Poker 코어 스모크 테스트 ==")
    print(f"Repo: {REPO_ROOT}")
    print()

    results = []
    for fn in TESTS:
        try:
            results.append(fn())
        except Exception as e:
            traceback.print_exc()
            results.append((fn.__name__, False, f"{type(e).__name__}: {e}"))

    print()
    fails = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    if fails == 0:
        print(f"전부 통과 ({total}/{total}). 코어 로직 정상. 집 PC 통합 단계 진행 가능.")
        return 0
    print(f"{fails}/{total} 실패. 코어 로직 점검 필요.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
