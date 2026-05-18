"""OCR/문자열 정규화 보조 함수.

순수 Python — tesserocr·OpenCV·PyQt 등 무거운 의존성 없음.
별도 모듈로 분리해서 단위 테스트와 외부 PC 환경에서도 import 가능하게 함.
"""
from __future__ import annotations

_CURRENCY_SYMBOLS = ('$', '£', '€', '₩', '¥', 'B', 'P')
_CHIP_SUFFIX_MULTIPLIERS = {
    'K': 1_000.0, 'k': 1_000.0,
    'M': 1_000_000.0, 'm': 1_000_000.0,
}


def normalize_chip_amount(s: str, default: float = -1.0) -> float:
    """OCR 결과 문자열을 float 금액으로 정규화.

    PokerStars 실머니 (``$0.05``, ``5,00€``) 와 플레이머니
    (``1,250,000``, ``1.5K``, ``10M``) 를 모두 처리한다.
    잘못된 입력은 ``default`` (기본 -1.0) 반환.

    콤마 처리 규칙:
        - 콤마 2개 이상  → 천 단위 구분자 (전부 제거)
        - 콤마 1개 + 뒤 1~2자리 숫자만 → 소수점 (``,`` → ``.``)
        - 콤마 1개 + 뒤 3자리 숫자 → 천 단위 구분자 (제거)
        - 그 외 → 그냥 제거 (보수적)

    K/M 접미사는 1,000 / 1,000,000 배수로 변환.

    Examples:
        >>> normalize_chip_amount('$0.08')
        0.08
        >>> normalize_chip_amount('5,00€')
        5.0
        >>> normalize_chip_amount('1,250,000')
        1250000.0
        >>> normalize_chip_amount('1.5K')
        1500.0
        >>> normalize_chip_amount('10M')
        10000000.0
        >>> normalize_chip_amount('abc')
        -1.0
    """
    if not isinstance(s, str):
        return default
    s = s.strip().replace('\n', '').replace(':', '').replace(' ', '')
    for sym in _CURRENCY_SYMBOLS:
        s = s.replace(sym, '')
    if not s:
        return default

    multiplier = 1.0
    if s[-1] in _CHIP_SUFFIX_MULTIPLIERS:
        multiplier = _CHIP_SUFFIX_MULTIPLIERS[s[-1]]
        s = s[:-1]

    n_commas = s.count(',')
    if n_commas >= 2:
        s = s.replace(',', '')
    elif n_commas == 1:
        idx = s.find(',')
        after = s[idx + 1:]
        if after.isdigit() and len(after) in (1, 2):
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')

    try:
        return float(s) * multiplier
    except ValueError:
        return default
