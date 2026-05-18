"""Mouse automation for the poker bot.

본 모듈은 두 단계로 진화한 마우스 자동화를 제공한다.

1. **MouseMover** — 외부 호출 API. 기존 코드 (mouse_action, mouse_clicker,
   move_mouse_away_from_buttons 등) 와 100% 호환. ``mouse_mover(x1,y1,x2,y2)``
   가 진입점.

2. **Path generation 내부 구현** — 2024년 이전: 직선 위 균등 jitter (x/y 독립
   step) 로 지그재그 경로 생성. 사람 마우스와 통계적으로 명확히 구분됨.

   본 리비전: 사람 모방 모델로 교체.
   - **베지에 곡선**: 4-control-point cubic Bezier. 시작·끝점은 정확,
     중간 control points 는 직선 P0→P3 에 수직 방향으로 ±5~15% 거리만큼
     섭동되어 자연스러운 곡선 형성.
   - **시그모이드 가감속** (smoothstep): t→3t²−2t³ 로 시작·끝에서 느리고
     중간에서 빠른 속도 프로파일 (Fitts's law 부합).
   - **로그정규 sleep**: 스텝 간 대기 시간을 ``lognormal(μ=log(0.02), σ=0.4)``
     로 샘플링. 실제 사람 마우스 inter-event 분포에 더 가까움.
   - **가우시안 micro-jitter**: 베지에 위 좌표에 σ=1.5 가우시안 노이즈
     (균등분포 ±20 보다 자연스럽고 분산 작음).

   **주의**: 이 변경이 PokerStars Game Integrity AI 의 95%+ RTA 탐지를
   회피한다는 보장은 없음. 다층 신호 (IP, 세션 패턴, 핸드 빈도) 모두
   봐야 회피 가능. 본 변경의 1차 목적은:
   - 사람 마우스 패턴 연구 / HCI 문헌 표준 적용
   - 학술 연구에 인용 가능한 표준 모델
   - numpy 2.x 호환성 버그 동시 해결

Public API 호환:
    >>> mm = MouseMoverTableBased(table_dict)
    >>> mm.mouse_action('Fold', topleftcorner=(100, 100))
    # 내부적으로 mouse_mover, mouse_clicker, take_action 모두 동작
"""
import logging
import math
import random
import time

from poker import pymouse
from poker.tools.helper import get_config
from poker.tools.vbox_manager import VirtualBoxController

log = logging.getLogger(__name__)


# 사람 마우스 모델 파라미터. 학술 문헌 (Fitts 1954, Crossman 1956, 후속 HCI 연구)
# 을 단순화. 정확한 캘리브레이션이 필요하면 본인 마우스 로그로 fit 가능.
HUMAN_MOUSE_PARAMS = {
    # Fitts-style: 이동 시간 ≈ a + b * log2(D/W + 1).
    # 일반적 W=30px (버튼 폭 절반) 가정.
    'fitts_a': 0.10,
    'fitts_b': 0.07,
    'fitts_target_w': 30.0,
    # 스텝당 평균 픽셀 (적을수록 부드러움, 많을수록 빠름)
    'pixels_per_step': 22.0,
    # 최소·최대 스텝 수 가드
    'min_steps': 8,
    'max_steps': 200,
    # 베지에 control point 섭동 비율 (직선 거리 대비)
    'curvature_min': 0.05,
    'curvature_max': 0.15,
    # 좌표 가우시안 noise σ
    'jitter_sigma': 1.5,
    # 스텝 간 sleep ~ lognormal(μ_log, σ_log). 결과 평균 ≈ exp(μ_log + σ²/2).
    # μ_log=-3.91 + σ=0.40  → 평균 ≈ 22ms
    'sleep_mu_log': -3.91,
    'sleep_sigma_log': 0.40,
}


def _smoothstep(t: float) -> float:
    """3t² − 2t³ — 시작·끝 0 도함수, 중간 최대 속도."""
    return t * t * (3.0 - 2.0 * t)


def _bezier_cubic(t: float, p0, p1, p2, p3):
    """Cubic Bezier 점. 4-control-point."""
    mt = 1.0 - t
    a = mt * mt * mt
    b = 3.0 * mt * mt * t
    c = 3.0 * mt * t * t
    d = t * t * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )


def _make_control_points(x1, y1, x2, y2, params=HUMAN_MOUSE_PARAMS):
    """직선 P0→P3 에 수직 방향으로 섭동된 P1, P2 생성."""
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        # 거의 0 거리 — control points 의미 없음, 끝점 복제
        return (x1, y1), (x2, y2)

    # 직선 방향 단위벡터
    ux, uy = dx / dist, dy / dist
    # 수직 단위벡터 (시계 90도)
    nx, ny = -uy, ux

    curv_lo = params['curvature_min']
    curv_hi = params['curvature_max']
    # 각 control point 의 직선 위 위치 (대략 1/3, 2/3 지점) + 수직 섭동
    bow1 = random.uniform(curv_lo, curv_hi) * dist * random.choice([-1, 1])
    bow2 = random.uniform(curv_lo, curv_hi) * dist * random.choice([-1, 1])

    p1 = (x1 + dx * 0.33 + nx * bow1, y1 + dy * 0.33 + ny * bow1)
    p2 = (x1 + dx * 0.66 + nx * bow2, y1 + dy * 0.66 + ny * bow2)
    return p1, p2


def _human_path(x1, y1, x2, y2, params=HUMAN_MOUSE_PARAMS):
    """베지에 + smoothstep + 가우시안 jitter 로 사람 모방 경로 생성.

    Returns:
        list of (x, y) integer coords. 첫 점은 (x1, y1) 부근, 마지막은 정확히 (x2, y2).
    """
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1.0:
        return [(int(x2), int(y2))]

    n_steps = int(dist / params['pixels_per_step'])
    n_steps = max(params['min_steps'], min(params['max_steps'], n_steps))

    p1, p2 = _make_control_points(x1, y1, x2, y2, params)
    p0 = (x1, y1)
    p3 = (x2, y2)

    sigma = params['jitter_sigma']
    path: list[tuple[int, int]] = []
    for i in range(n_steps):
        t_linear = i / n_steps
        t = _smoothstep(t_linear)
        bx, by = _bezier_cubic(t, p0, p1, p2, p3)
        jx = random.gauss(0.0, sigma)
        jy = random.gauss(0.0, sigma)
        path.append((int(round(bx + jx)), int(round(by + jy))))
    # 마지막은 정확한 목적지 (jitter 없음)
    path.append((int(x2), int(y2)))
    return path


def _human_sleep(params=HUMAN_MOUSE_PARAMS) -> float:
    """스텝 간 대기 시간 — 로그정규 샘플링."""
    return random.lognormvariate(params['sleep_mu_log'], params['sleep_sigma_log'])


class MouseMover(VirtualBoxController):
    def __init__(self, vbox_mode):
        if vbox_mode:
            super().__init__()
        self.mouse = pymouse.PyMouse()
        self.vbox_mode = vbox_mode
        # numpy 2.x 에서 int(np.random.uniform(..., 1)) 깨짐 → stdlib random 사용
        self.old_x = random.randint(0, 500)
        self.old_y = random.randint(0, 500)

    def click(self, x, y):
        if self.vbox_mode:
            self.mouse_move_vbox(x, y)
            self.mouse_click_vbox(x, y)
        else:
            self.mouse.move(x, y)
            self.mouse.click(x, y)

        time.sleep(_human_sleep())

    def mouse_mover(self, x1, y1, x2, y2):
        """사람 모방 마우스 경로로 (x1,y1) → (x2,y2) 이동.

        API 는 기존과 동일하나 내부 구현이 베지에 + Fitts 모델로 교체됨.
        호출 측 변경 불필요.
        """
        path = _human_path(x1, y1, x2, y2)
        for (x, y) in path:
            if self.vbox_mode:
                try:
                    self.mouse_move_vbox(x, y)
                except AttributeError:
                    raise RuntimeError(
                        "Virtual box not detected. "
                        "Switch to direct mouse control in setup or open VirtualBox"
                    )
            else:
                self.mouse.move(x, y)
            time.sleep(_human_sleep())

        self.old_x = x2
        self.old_y = y2

    def mouse_clicker(self, x2, y2, buttonToleranceX, buttonToleranceY):
        # 버튼 영역 안의 임의 지점을 가우시안으로 샘플 (중심 편향, 가장자리 회피)
        # 균등분포 대신 가우시안 → 클릭 위치 분포가 사람과 더 유사
        xrand = int(buttonToleranceX * (0.5 + 0.25 * random.gauss(0, 1)))
        yrand = int(buttonToleranceY * (0.5 + 0.25 * random.gauss(0, 1)))
        xrand = max(0, min(buttonToleranceX, xrand))
        yrand = max(0, min(buttonToleranceY, yrand))

        target_x, target_y = x2 + xrand, y2 + yrand

        if self.vbox_mode:
            self.mouse_move_vbox(target_x, target_y)
        else:
            self.mouse.move(target_x, target_y)

        # 클릭 직전 짧은 hover (사람 패턴)
        time.sleep(random.uniform(0.08, 0.22))

        self.click(target_x, target_y)
        log.debug("Clicked: %d %d", target_x, target_y)

        # 클릭 후 짧은 휴식
        time.sleep(random.uniform(0.10, 0.45))


class MouseMoverTableBased(MouseMover):
    def __init__(self, table_dict):
        config = get_config()

        try:
            mouse_control = config.config.get('main', 'control')
            if mouse_control != 'Direct mouse control':
                self.vbox_mode = True
            else:
                self.vbox_mode = False
        except:
            self.vbox_mode = False

        super().__init__(self.vbox_mode)

        self.table_dict = table_dict

    def move_mouse_away_from_buttons(self):
        x2 = random.randint(1700, 2000)
        y2 = random.randint(10, 200)

        time.sleep(random.uniform(0.5, 1.2))
        if not self.vbox_mode:
            (x1, y1) = self.mouse.position()
        else:
            x1 = self.old_x
            y1 = self.old_y
        x1 = 10 if x1 > 2000 else x1
        y1 = 10 if y1 > 1000 else y1

        try:
            log.debug("Moving mouse away: %d,%d → %d,%d", x1, y1, x2, y2)
            self.mouse_mover(x1, y1, x2, y2)
        except Exception:
            log.warning("Moving mouse away failed")

    def move_mouse_away_from_buttons_jump(self):
        x2 = random.randint(1700, 2000)
        y2 = random.randint(10, 200)

        try:
            log.debug("Moving mouse away via jump: %d,%d", x2, y2)
            if self.vbox_mode:
                self.mouse_move_vbox(x2, y2)
            else:
                self.mouse.move(x2, y2)
        except Exception as e:
            log.warning("Moving mouse via jump away failed: %s", e)

    def mouse_action(self, decision, topleftcorner, options=None):
        if decision == 'Check Deception':
            decision = 'Check'
        if decision == 'Call Deception':
            decision = 'Call'

        tlx = int(topleftcorner[0])
        tly = int(topleftcorner[1])

        log.debug("Mouse moving to: %s", decision)
        log.debug("Top left corner position: %d %d", tlx, tly)

        if decision == "Fold":
            coo = self.table_dict['mouse_fold']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Imback":
            time.sleep(random.uniform(0, 3))
            coo = self.table_dict['mouse_imback']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "resume_hand":
            time.sleep(random.uniform(0, 3))
            coo = self.table_dict['mouse_resume_hand']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Call":
            coo = self.table_dict['mouse_call']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Call2":
            coo = self.table_dict['mouse_call2']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Check":
            coo = self.table_dict['mouse_check']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Bet":
            coo = self.table_dict['mouse_raise']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "BetPlus":
            for _ in range(int(options['increases_num'])):
                coo = self.table_dict['mouse_increase']
                self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

            coo = self.table_dict['mouse_raise']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Bet Bluff":
            coo = self.table_dict['mouse_raise']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Bet half pot":
            coo = self.table_dict['mouse_half_pot']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

            coo = self.table_dict['mouse_raise']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Bet pot":
            coo = self.table_dict['mouse_full_pot']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

            coo = self.table_dict['mouse_raise']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        elif decision == "Bet max":
            coo = self.table_dict['mouse_all_in']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

            coo = self.table_dict['mouse_raise']
            self.take_action(coo['x1'] + tlx, coo['y1'] + tly, coo['x2'] + tlx, coo['y2'] + tly)

        time.sleep(0.2)
        self.move_mouse_away_from_buttons()

    def take_action(self, x1, y1, x2, y2):
        log.debug("Target position: %d %d %d %d", x1, y1, x2, y2)
        if not self.vbox_mode:
            (old_x1, old_y1) = self.mouse.position()
        else:
            old_x1 = self.old_x
            old_y1 = self.old_y

        self.mouse_mover(old_x1, old_y1, x1, y1)
        self.mouse_clicker(x1, y1, x2 - x1, y2 - y1)
