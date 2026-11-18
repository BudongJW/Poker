"""``python -m pokerlab check``: is this computer ready to read windows?

Says which recognition service and model would be used and where the key comes from (never the
key itself), then reads one bundled picture of a table whose values are known, and tells how long
it took, what it cost and whether what was read is right. ``--no-read`` stops before the picture,
so that nothing is sent and nothing is spent.
"""

import json
import os
import platform
import sys

from . import vision

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample_table.jpg")
ANSWERS = os.path.join(os.path.dirname(__file__), "data", "sample_table.json")


def _key_name():
    for name in ("ANTHROPIC_API_KEY", "POKER_ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return name
    return None


def compare(read, answers):
    """What was read from the sample picture against its known values -> list of what is wrong (empty: all right)."""
    wrong = []
    from .cards import parse_cards
    try:
        same_hero = sorted(parse_cards(read.get("hero") or "")) == sorted(parse_cards(answers["hero"]))
    except ValueError:
        same_hero = False
    if not same_hero:
        wrong.append(f"내 카드: {read.get('hero') or '없음'} (정답 {answers['hero']})")
    if read.get("board") != answers["board"]:
        wrong.append(f"보드: {read.get('board') or '없음'} (정답 {answers['board']})")
    if read.get("pot") != answers["pot"]:
        wrong.append(f"팟: {read.get('pot')} (정답 {answers['pot']})")
    seats = list(read.get("seats") or [])
    for expected in answers["seats"]:          # the stacks of the sample all differ: a seat is found by its stack
        found = next((seat for seat in seats if seat.get("stack") == expected["stack"]), None)
        if found is None:
            wrong.append(f"스택 {expected['stack']}인 자리를 찾지 못함")
            continue
        seats.remove(found)
        for key, label in (("bet", "앞에 놓인 칩"), ("in_hand", "카드 보유"), ("hero", "내 자리"), ("button", "딜러 버튼")):
            got = found.get(key) or (0 if key == "bet" else False)
            if got != expected[key]:
                wrong.append(f"스택 {expected['stack']}인 자리의 {label}: {got} (정답 {expected[key]})")
    if seats:
        wrong.append(f"없는 자리 {len(seats)}개를 더 읽음")
    return wrong


def run(read=True, model=None, out=print):
    """Check and report (through ``out``) -> 0 when ready (and, with ``read``, the sample was read right), else 1."""
    out(f"Python {platform.python_version()} · {sys.platform}")
    key = _key_name()
    if key is None:
        out("API 키: 없음. 창을 읽으려면 Claude API 키를 환경 변수 ANTHROPIC_API_KEY에 넣어 주세요 "
            "(Gemini를 쓰려면 GEMINI_API_KEY). 넣는 방법은 README의 '설치와 실행'에 있습니다.")
        return 1
    try:
        service = vision.provider()
    except vision.VisionError as error:
        out(f"인식 서비스: {error}")
        return 1
    key = _key_name()
    if service == "claude":
        try:
            import anthropic
            package = f"anthropic {anthropic.__version__}"
        except ImportError:
            out("인식 서비스: Claude · anthropic 패키지가 없습니다. 설치: pip install -e \".[claude]\"")
            return 1
        main = model or os.environ.get("POKERLAB_VISION_MODEL") or vision.CLAUDE_MODEL
        out(f"인식 서비스: Claude ({package}) · 기본 모델 {main} · 아껴 읽기 모델 {vision.saving_model()}")
        if os.environ.get("POKERLAB_ANTHROPIC_BASE_URL"):
            out(f"API 주소: {os.environ['POKERLAB_ANTHROPIC_BASE_URL']} (POKERLAB_ANTHROPIC_BASE_URL)")
    else:
        main = model or os.environ.get("POKERLAB_VISION_MODEL") or vision.DEFAULT_MODEL
        out(f"인식 서비스: Gemini · 모델 {main}")
    out(f"API 키: 환경 변수 {key}에 있음 (키 값은 표시하지 않습니다)")
    if not read:
        out("그림은 읽지 않았습니다 (--no-read). 아무것도 보내지 않았고 비용도 없습니다.")
        return 0
    with open(SAMPLE, "rb") as fh:
        image = fh.read()
    with open(ANSWERS, encoding="utf-8") as fh:
        answers = json.load(fh)
    out("값을 아는 테스트용 테이블 그림 한 장을 읽습니다 (6인, 턴)...")
    call = {}
    try:
        got = vision.recognize(image, "image/jpeg", model=main, call=call)
    except vision.VisionError as error:
        out(f"읽지 못했습니다: {error}")
        return 1
    spent = call.get("cost")
    out(f"읽기: {call.get('seconds', 0):.1f}초" + (f", 약 ${spent:.3f}" if spent is not None else "")
        + (f" · 응답한 모델 {call['model']}" if call.get("model") else ""))
    wrong = compare(got, answers)
    if wrong:
        out("틀리게 읽은 값:")
        for line in wrong:
            out(f"  - {line}")
        return 1
    out("모두 맞게 읽었습니다 (내 카드, 보드, 팟, 6자리의 스택·앞에 놓인 칩·카드 보유·내 자리·딜러 버튼).")
    out("준비됐습니다: python -m pokerlab serve 를 실행하고 브라우저에서 http://127.0.0.1:8765 를 여세요.")
    return 0
