"""확장 스모크 테스트 — 마우스/이미지 파이프라인/GUI 헤드리스 검증.

기본 ``smoke_test.py`` (수학 코어)에 더해, PokerStars 없이 검증 가능한
경계 레이어 3종을 추가로 확인한다:

- **A. 마우스 좌표 시퀀스** — ``MouseMoverTableBased.mouse_mover`` 가 생성하는
  좌표 경로의 길이·jitter·시작/끝점 보존. ``poker.pymouse`` 와 ``vbox_manager``
  를 ``sys.modules`` 주입으로 stub 처리해 pywin32·VirtualBox 없이 실행.
- **B. 이미지 파이프라인** — 번들 PokerStars/PartyPoker 스크린샷 + 카드 이미지를
  PIL 로 열고 OpenCV 로 변환해 dimension·channel 검증. 템플릿 매칭이
  수치적으로 동작하는지 작은 합성 템플릿으로 확인.
- **F. PyQt6 헤드리스 init** — ``QT_QPA_PLATFORM=offscreen`` 으로 QApplication
  생성 가능 여부 확인. GUI 의존 코드가 import 단계에서 깨지지 않는지 검증.

실행: ``.venv/Scripts/python.exe scripts/smoke_test_extended.py``
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _result(name: str, ok: bool, detail: str = "") -> tuple[str, bool, str]:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}  {detail}")
    return (name, ok, detail)


# ---------- A. 마우스 좌표 시퀀스 ----------
# pymouse / vbox_manager 를 stub 으로 교체 (모듈 import 캐싱 회피 위해 1회만 설치).

_CAPTURED: list = []


class _FakeMouse:
    def __init__(self):
        self._pos = (0, 0)
    def move(self, x, y):
        self._pos = (x, y)
        _CAPTURED.append(("move", x, y))
    def click(self, x, y):
        _CAPTURED.append(("click", x, y))
    def position(self):
        return self._pos


def _install_stubs_once() -> None:
    if "poker.pymouse" in sys.modules and getattr(sys.modules["poker.pymouse"], "_is_stub", False):
        return
    fake_pymouse = types.ModuleType("poker.pymouse")
    fake_pymouse.PyMouse = _FakeMouse
    fake_pymouse._is_stub = True
    sys.modules["poker.pymouse"] = fake_pymouse

    fake_vbox = types.ModuleType("poker.tools.vbox_manager")
    class _FakeVBox:
        def __init__(self, *a, **kw): pass
    fake_vbox.VirtualBoxController = _FakeVBox
    sys.modules["poker.tools.vbox_manager"] = fake_vbox


def _reset_capture() -> None:
    _CAPTURED.clear()


def test_mouse_path_endpoints():
    """mouse_mover(100,100 → 500,500) — 마지막 move 가 (500,500) 이어야"""
    _install_stubs_once()
    _reset_capture()
    from poker.tools.mouse_mover import MouseMover
    mm = MouseMover(vbox_mode=False)
    mm.mouse_mover(100, 100, 500, 500)
    moves = [c for c in _CAPTURED if c[0] == "move"]
    last_move = moves[-1] if moves else None
    ok = last_move == ("move", 500, 500)
    return _result(
        "mouse path endpoint (100,100 → 500,500)",
        ok,
        f"last move = {last_move}, total moves = {len(moves)}",
    )


def test_mouse_path_step_count():
    """대각 이동 (0,0)→(400,400) — 스텝 약 15~80 사이 (stepMin=7, stepMax=20)"""
    _install_stubs_once()
    _reset_capture()
    from poker.tools.mouse_mover import MouseMover
    mm = MouseMover(vbox_mode=False)
    mm.mouse_mover(0, 0, 400, 400)
    moves = [c for c in _CAPTURED if c[0] == "move"]
    ok = 15 <= len(moves) <= 80
    return _result(
        "mouse path step count (diagonal 400px)",
        ok,
        f"moves={len(moves)} (expected 15-80)",
    )


def test_mouse_path_bounding_box():
    """모든 좌표가 베지에 곡선 + curvature(15%) + jitter 허용 영역 안에 있는지.

    새 구현은 베지에 곡선이라 직선 P0→P3 에서 최대 ~15% * dist 만큼
    수직으로 휘어질 수 있음. dist=566 (대각 400px) 의 15% ≈ 85 + jitter 여유.
    """
    _install_stubs_once()
    _reset_capture()
    from poker.tools.mouse_mover import MouseMover
    mm = MouseMover(vbox_mode=False)
    mm.mouse_mover(0, 0, 400, 400)
    moves = [c for c in _CAPTURED if c[0] == "move"]
    PAD = 120  # curvature(15% × 566) + jitter + 여유
    out_of_box = [
        (x, y) for _, x, y in moves
        if not (-PAD <= x <= 400 + PAD and -PAD <= y <= 400 + PAD)
    ]
    ok = not out_of_box
    return _result(
        "mouse path bounding box (Bezier + curvature ≤ 120)",
        ok,
        f"moves={len(moves)}, out-of-box={len(out_of_box)}",
    )


def test_mouse_velocity_profile():
    """smoothstep 가감속: 중간 스텝의 좌표 간격이 시작/끝 스텝보다 커야.

    인접 좌표 간 거리를 segment[i] 라 하면, 정상 사람 마우스는
    중앙부 segment 가 양 끝부 segment 보다 큼 (Fitts/사인 곡선 속도).
    """
    _install_stubs_once()
    _reset_capture()
    from poker.tools.mouse_mover import MouseMover
    import math as _math
    mm = MouseMover(vbox_mode=False)
    mm.mouse_mover(0, 0, 600, 0)  # 긴 수평 이동 — 속도 차이 명확
    moves = [c for c in _CAPTURED if c[0] == "move"]
    coords = [(x, y) for _, x, y in moves]
    segs = [
        _math.hypot(coords[i+1][0] - coords[i][0],
                    coords[i+1][1] - coords[i][1])
        for i in range(len(coords) - 1)
    ]
    if len(segs) < 6:
        return _result("mouse velocity profile (smoothstep)", False,
                       f"not enough segments: {len(segs)}")
    n = len(segs)
    head = sum(segs[:n // 4]) / (n // 4)
    mid = sum(segs[n // 3:2 * n // 3]) / (2 * n // 3 - n // 3)
    tail = sum(segs[3 * n // 4:]) / (n - 3 * n // 4)
    # 중앙 평균 segment 가 시작·끝 평균보다 명확히 커야 (≥ 1.3 배)
    ok = mid > head * 1.3 and mid > tail * 1.3
    return _result(
        "mouse velocity profile (smoothstep)",
        ok,
        f"segments avg head={head:.1f} mid={mid:.1f} tail={tail:.1f} (mid > head·1.3, mid > tail·1.3)",
    )


def test_mouse_sleep_lognormal():
    """sleep 시간 분포가 로그정규 특성을 보이는지 확인.

    중요: time.sleep 호출 자체는 mock 하지 않고 실제 호출되므로,
    여기서는 ``_human_sleep`` 함수만 직접 호출해 분포를 평가한다.
    """
    from poker.tools.mouse_mover import _human_sleep
    samples = [_human_sleep() for _ in range(1000)]
    pos = sum(1 for s in samples if s > 0)
    mean = sum(samples) / len(samples)
    # 평균 ≈ exp(-3.91 + 0.40²/2) ≈ exp(-3.83) ≈ 0.0217s
    ok = pos == 1000 and 0.012 <= mean <= 0.035
    return _result(
        "mouse sleep distribution (log-normal, mean ~22ms)",
        ok,
        f"positive={pos}/1000, mean={mean*1000:.1f}ms (expected 12-35ms)",
    )


def test_mouse_path_is_curved():
    """베지에 곡선 — 다수 경로 평균에서 직선 P0→P3 대비 명확한 수직 편차.

    개별 경로는 control point 부호 우연히 반대로 나오면 S-커브 → 편차 작을 수 있음.
    20번 샘플의 평균 max|y| 로 통계적 안정성 확보.

    구버전(zigzag) 비교: 구버전 평균 max|y| ≈ jitter 한계(20) 이내.
    신버전은 curvature 5~15% × 직선거리 → 평균 max|y| > 30 기대.
    """
    _install_stubs_once()
    from poker.tools.mouse_mover import MouseMover

    devs: list[int] = []
    for _ in range(20):
        _reset_capture()
        mm = MouseMover(vbox_mode=False)
        mm.mouse_mover(0, 0, 500, 0)
        moves = [c for c in _CAPTURED if c[0] == "move"]
        if moves:
            devs.append(max(abs(y) for _, _, y in moves))
    avg = sum(devs) / len(devs) if devs else 0
    threshold = 500 * 0.05  # 5% — 직선 jitter 한계(20) 초과해야 의미 있음
    ok = avg >= threshold
    return _result(
        "mouse path is curved (avg perpendicular deviation across 20 paths)",
        ok,
        f"avg max|y|={avg:.1f} on 500px horizontal (expected >= {threshold:.0f})",
    )


# ---------- B. 이미지 파이프라인 ----------

def test_screenshots_loadable():
    """번들 스크린샷 7장 PIL 로드 & cv2 변환"""
    import cv2
    import numpy as np
    from PIL import Image

    ss_dir = REPO_ROOT / "poker" / "tests" / "screenshots"
    pngs = sorted(ss_dir.glob("*.png"))
    bad: list[str] = []
    sizes: list[tuple[int, int]] = []
    for p in pngs:
        try:
            img = Image.open(p)
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            h, w = arr.shape[:2]
            if h < 50 or w < 50:
                bad.append(f"{p.name}: too small ({w}x{h})")
            else:
                sizes.append((w, h))
        except Exception as e:
            bad.append(f"{p.name}: {e}")
    ok = not bad and len(pngs) > 0
    detail = f"{len(pngs)} screenshots, sizes: {sizes[:3]}{'...' if len(sizes)>3 else ''}"
    if bad:
        detail += f", FAILED: {bad}"
    return _result("bundled screenshots loadable", ok, detail)


def test_card_images_loadable():
    """번들 카드 이미지 7장 (2H, 5C, 6C, 8S, AH, JC, QS)"""
    import cv2
    import numpy as np
    from PIL import Image

    cd_dir = REPO_ROOT / "poker" / "tests" / "test_cards"
    pngs = sorted(cd_dir.glob("*.png"))
    bad: list[str] = []
    for p in pngs:
        try:
            img = Image.open(p)
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            if arr.shape[0] < 5 or arr.shape[1] < 5:
                bad.append(f"{p.name}: too small")
        except Exception as e:
            bad.append(f"{p.name}: {e}")
    ok = not bad and len(pngs) == 7
    return _result(
        "bundled card images loadable (2H..QS)",
        ok,
        f"{len(pngs)}/7 cards, failures: {len(bad)}",
    )


def test_template_matching():
    """cv2.matchTemplate 수치적 동작 확인 — 같은 이미지를 자기 자신에 매칭하면 best=(0,0)"""
    import cv2
    import numpy as np
    from PIL import Image

    ss_dir = REPO_ROOT / "poker" / "tests" / "screenshots"
    target = list(ss_dir.glob("*.png"))[0]
    full_img = cv2.cvtColor(np.array(Image.open(target)), cv2.COLOR_BGR2RGB)
    # 좌상단 60x60 패치 잘라서 그것을 템플릿으로 매칭
    h, w = full_img.shape[:2]
    patch = full_img[:60, :60]
    res = cv2.matchTemplate(full_img, patch, cv2.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
    ok = min_loc == (0, 0) and min_val < 0.01
    return _result(
        "cv2.matchTemplate sanity (self-match)",
        ok,
        f"best_loc={min_loc}, min_val={min_val:.4f} (expected (0,0), <0.01)",
    )


# ---------- C'. OCR 정규화 (tesserocr 없이도 검증 가능) ----------

def test_chip_amount_small_cash():
    """소액 캐시 ($0.08) — 기존 동작 보존"""
    from poker.tools.text_normalize import normalize_chip_amount
    cases = [
        ('$0.08', 0.08),
        ('$0.05', 0.05),
        ('1.25', 1.25),
    ]
    bad = [(s, v, normalize_chip_amount(s)) for s, v in cases
           if abs(normalize_chip_amount(s) - v) > 1e-6]
    return _result(
        "OCR normalize (small cash $0.08)",
        not bad,
        f"failures: {bad}" if bad else "all 3 ok",
    )


def test_chip_amount_european_decimal():
    """유럽식 쉼표 소수점 (5,00€) — 기존 동작 보존"""
    from poker.tools.text_normalize import normalize_chip_amount
    cases = [
        ('5,00€', 5.0),
        ('5,50', 5.5),
        ('0,25', 0.25),
    ]
    bad = [(s, v, normalize_chip_amount(s)) for s, v in cases
           if abs(normalize_chip_amount(s) - v) > 1e-6]
    return _result(
        "OCR normalize (EU decimal '5,00€')",
        not bad,
        f"failures: {bad}" if bad else "all 3 ok",
    )


def test_chip_amount_play_money_thousands():
    """플레이머니 천 단위 (1,250,000) — 새로운 케이스 (구버전은 깨짐)"""
    from poker.tools.text_normalize import normalize_chip_amount
    cases = [
        ('1,250,000', 1_250_000.0),
        ('5,000', 5_000.0),
        ('100,000', 100_000.0),
    ]
    bad = [(s, v, normalize_chip_amount(s)) for s, v in cases
           if abs(normalize_chip_amount(s) - v) > 1e-6]
    return _result(
        "OCR normalize (play money '1,250,000')",
        not bad,
        f"failures: {bad}" if bad else "all 3 ok",
    )


def test_chip_amount_suffixes():
    """K/M 접미사 — 새로운 케이스"""
    from poker.tools.text_normalize import normalize_chip_amount
    cases = [
        ('1.5K', 1_500.0),
        ('10M', 10_000_000.0),
        ('25k', 25_000.0),
        ('2.5m', 2_500_000.0),
    ]
    bad = [(s, v, normalize_chip_amount(s)) for s, v in cases
           if abs(normalize_chip_amount(s) - v) > 1e-6]
    return _result(
        "OCR normalize (K/M suffix '1.5K' '10M')",
        not bad,
        f"failures: {bad}" if bad else "all 4 ok",
    )


def test_chip_amount_invalid():
    """OCR 실패 → -1.0 반환"""
    from poker.tools.text_normalize import normalize_chip_amount
    cases = [
        ('abc', -1.0),
        ('', -1.0),
        ('$$$', -1.0),
        (None, -1.0),
    ]
    bad = [(s, v, normalize_chip_amount(s)) for s, v in cases
           if normalize_chip_amount(s) != v]
    return _result(
        "OCR normalize (invalid input → -1.0)",
        not bad,
        f"failures: {bad}" if bad else "all 4 ok",
    )


# ---------- F. PyQt6 헤드리스 ----------

def test_pyqt6_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6 import QtWidgets, QtCore
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        widget = QtWidgets.QLabel("smoke")
        widget.resize(100, 30)
        # 실제 화면 안 띄움. 위젯 인스턴스화 자체가 Qt 의 plugin 로드를 검증
        return _result(
            "PyQt6 offscreen QApplication + widget",
            True,
            f"Qt {QtCore.QT_VERSION_STR}, platform={app.platformName()}",
        )
    except Exception as e:
        return _result("PyQt6 offscreen QApplication + widget", False, f"{type(e).__name__}: {e}")


TESTS = [
    test_mouse_path_endpoints,
    test_mouse_path_step_count,
    test_mouse_path_bounding_box,
    test_mouse_velocity_profile,
    test_mouse_sleep_lognormal,
    test_mouse_path_is_curved,
    test_screenshots_loadable,
    test_card_images_loadable,
    test_template_matching,
    test_chip_amount_small_cash,
    test_chip_amount_european_decimal,
    test_chip_amount_play_money_thousands,
    test_chip_amount_suffixes,
    test_chip_amount_invalid,
    test_pyqt6_offscreen,
]


def main() -> int:
    print("== 확장 스모크 테스트 (마우스 / 이미지 / GUI) ==")
    print(f"Repo: {REPO_ROOT}\n")
    results = []
    for fn in TESTS:
        try:
            results.append(fn())
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append((fn.__name__, False, f"{type(e).__name__}: {e}"))
    print()
    fails = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    if fails == 0:
        print(f"전부 통과 ({total}/{total}). 외부 PC 검증 가능 범위 완료.")
        return 0
    print(f"{fails}/{total} 실패.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
