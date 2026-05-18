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
    """모든 좌표가 출발-도착 경계 박스 + jitter(=20) 안에 있는지.

    참고: 봇의 마우스 경로는 x/y 가 독립 step 으로 움직여 직선이 아닌
    지그재그 형태가 됨. 이건 사람 패턴과 통계적으로 구분 가능한
    탐지 시그널이기도 함 — 향후 베지어 곡선 등으로 개선 여지.
    """
    _install_stubs_once()
    _reset_capture()
    from poker.tools.mouse_mover import MouseMover
    mm = MouseMover(vbox_mode=False)
    mm.mouse_mover(0, 0, 400, 400)
    moves = [c for c in _CAPTURED if c[0] == "move"]
    JITTER = 25  # xTremble=yTremble=20 + 여유
    out_of_box = [
        (x, y) for _, x, y in moves
        if not (-JITTER <= x <= 400 + JITTER and -JITTER <= y <= 400 + JITTER)
    ]
    ok = not out_of_box
    return _result(
        "mouse path bounding box (-25 ≤ x,y ≤ 425)",
        ok,
        f"moves={len(moves)}, out-of-box={len(out_of_box)}",
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
    test_screenshots_loadable,
    test_card_images_loadable,
    test_template_matching,
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
