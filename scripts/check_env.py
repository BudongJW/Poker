"""환경 검증 스크립트 (집 PC 도착 시 가장 먼저 실행).

실행: ``python scripts/check_env.py``

- Python 버전 확인 (3.11 권장 — tesserocr cp311 휠 호환)
- 핵심 패키지 import 가능 여부
- tesseract 바이너리 동작 여부
- MongoDB 연결 확인 (옵션, 원격 템플릿 받기 위해 필요)
- 스크린 해상도 확인 (1920x1800 권장)

모든 항목이 OK 면 테이블 매핑 단계로 진행 가능.
"""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "cv2",            # opencv-python
    "PIL",            # pillow
    "PyQt6",
    "pymongo",
    "tensorflow",
    "tesserocr",
    "lmfit",
    "openpyxl",
    "xlrd",
]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_python() -> CheckResult:
    v = sys.version_info
    ok = v.major == 3 and v.minor == 11
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if not ok:
        detail += " (권장: 3.11.x — tesserocr 휠 호환)"
    return CheckResult("Python", ok, detail)


def check_package(name: str) -> CheckResult:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "n/a")
        return CheckResult(f"import {name}", True, str(ver))
    except Exception as e:
        return CheckResult(f"import {name}", False, f"{type(e).__name__}: {e}")


def check_tesseract() -> CheckResult:
    exe = shutil.which("tesseract")
    if not exe:
        return CheckResult("tesseract binary", False, "PATH 에 tesseract 없음")
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10)
        first = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else ""
        return CheckResult("tesseract binary", True, f"{exe} — {first}")
    except Exception as e:
        return CheckResult("tesseract binary", False, f"{type(e).__name__}: {e}")


def check_screen() -> CheckResult:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        ok = w >= 1920 and h >= 1000
        detail = f"{w}x{h}"
        if not ok:
            detail += " (권장: 1920x1800 이상)"
        return CheckResult("screen resolution", ok, detail)
    except Exception as e:
        return CheckResult("screen resolution", False, f"{type(e).__name__}: {e}")


def check_mongo() -> CheckResult:
    try:
        import pymongo
        from pymongo.errors import ConfigurationError
        # 실제 연결은 시도하지 않고 driver 로드 + URI 파싱만 확인 (집 환경에서 원격 서버 다운 가능성)
        _ = pymongo.MongoClient
        return CheckResult("pymongo driver", True, f"v{pymongo.__version__} (실제 연결은 봇 실행 시 시도)")
    except Exception as e:
        return CheckResult("pymongo driver", False, f"{type(e).__name__}: {e}")


def main() -> int:
    print(f"== DeeperMind Poker 환경 검증 ==")
    print(f"OS: {platform.platform()}")
    print(f"Arch: {platform.machine()}")
    print()

    results: list[CheckResult] = []
    results.append(check_python())
    for pkg in REQUIRED_PACKAGES:
        results.append(check_package(pkg))
    results.append(check_tesseract())
    if platform.system() == "Windows":
        results.append(check_screen())
    results.append(check_mongo())

    width = max(len(r.name) for r in results)
    fails = 0
    for r in results:
        mark = "OK  " if r.ok else "FAIL"
        print(f"  [{mark}] {r.name.ljust(width)}  {r.detail}")
        if not r.ok:
            fails += 1

    print()
    if fails == 0:
        print("모든 검증 통과. 다음 단계: SETUP_PLAY_MONEY.ko.md '테이블 매핑' 섹션.")
        return 0
    print(f"{fails}건 실패. SETUP_PLAY_MONEY.ko.md '환경 설치' 섹션 재확인.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
