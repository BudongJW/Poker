"""layout_measurements.csv 에서 09_layout_geometry.md 의 수치를 다시 계산한다.

    python3 layout_stats.py [layout_measurements.csv]

좌표는 게임 화면의 왼쪽 위를 (0, 0), 오른쪽 아래를 (100, 100)으로 본 백분율이다.
가로·세로 축척이 다르므로 거리와 종횡비는 화면 높이를 1로 둔 물리 단위로 바꿔 계산한다
(x 방향 길이에 화면 종횡비 W/H 를 곱한다).
"""
import csv
import math
import sys
from collections import defaultdict
from statistics import mean, pstdev

PATH = sys.argv[1] if len(sys.argv) > 1 else 'layout_measurements.csv'

screens = {}                      # id -> meta
boxes = defaultdict(dict)         # id -> element -> (x0, y0, x1, y1)
points = defaultdict(dict)        # id -> element -> (x, y)
labels = defaultdict(dict)        # id -> element -> note
with open(PATH, encoding='utf-8') as f:
    for r in csv.DictReader(f):
        s = r['screen']
        screens[s] = dict(app=r['app'], mode=r['mode'], platform=r['platform'], orientation=r['orientation'],
                          aspect=float(r['aspect_w_over_h']), conf=r['confidence'])
        labels[s][r['element']] = r['note']
        if r['kind'] == 'box':
            boxes[s][r['element']] = tuple(float(r[k]) for k in ('x0', 'y0', 'x1', 'y1'))
        else:
            points[s][r['element']] = (float(r['x0']), float(r['y0']))


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def size(b):
    return (b[2] - b[0], b[3] - b[1])


def phys_dist(p, q, aspect):
    """두 점 사이 거리. 화면 높이를 100으로 둔 단위."""
    return math.hypot((p[0] - q[0]) * aspect, p[1] - q[1])


def stats(vals, nd=1):
    vals = list(vals)
    if not vals:
        return '-'
    if len(vals) == 1:
        return f'{vals[0]:.{nd}f} (n=1)'
    return f'{mean(vals):.{nd}f} ± {pstdev(vals):.{nd}f} [{min(vals):.{nd}f}–{max(vals):.{nd}f}] (n={len(vals)})'


def fit_ellipse(pts, aspect):
    """축에 나란한 타원 ((x-cx)/a)^2 + ((y-cy)/b)^2 = 1 의 최소제곱 맞춤.
    A x^2 + B y^2 + C x + D y = 1 을 정규방정식으로 푼다. 좌표는 물리 단위(높이=100)."""
    P = [(x * aspect, y) for x, y in pts]
    rows = [(x * x, y * y, x, y) for x, y in P]
    n = 4
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] for r in rows) for i in range(n)]
    m = [ata[i] + [atb[i]] for i in range(n)]            # Gauss-Jordan
    for i in range(n):
        piv = max(range(i, n), key=lambda k: abs(m[k][i]))
        m[i], m[piv] = m[piv], m[i]
        d = m[i][i]
        m[i] = [v / d for v in m[i]]
        for k in range(n):
            if k != i:
                fct = m[k][i]
                m[k] = [a - fct * b for a, b in zip(m[k], m[i])]
    A, B, C, D = (m[i][n] for i in range(n))
    cx, cy = -C / (2 * A), -D / (2 * B)
    k = 1 + A * cx * cx + B * cy * cy
    a, b = math.sqrt(k / A), math.sqrt(k / B)
    res = [abs(math.hypot((x - cx) / a, (y - cy) / b) - 1) for x, y in P]
    # 좌석 각도: 타원 중심에서 본 방향. 0° = 아래(내 자리), 90° = 왼쪽, 180° = 위, 270° = 오른쪽
    ang = [round(math.degrees(math.atan2(-(x - cx) / a, (y - cy) / b)) % 360) for x, y in P]
    return dict(cx=cx / aspect, cy=cy, a=a / aspect, b=b, a_phys=a, rms=math.sqrt(mean(v * v for v in res)), angles=ang)


def section(title):
    print('\n' + '=' * 8, title)


# ---------------------------------------------------------------- per screen
section('1. 화면별 파생 지표')
for s, m in screens.items():
    A = m['aspect']
    B, P = boxes[s], points[s]
    print(f"\n[{s}] {m['app']} · {m['mode']} · {m['platform']} · 종횡비 {A:.3f} · 신뢰도 {m['conf']}")
    if 'action_bar' in B:
        b = B['action_bar']
        w, h = size(b)
        btns = sorted(k for k in B if k.startswith('action_btn_'))
        bw = [size(B[k])[0] for k in btns]
        bh = [size(B[k])[1] for k in btns]
        print(f"  액션 영역: x {b[0]:.1f}–{b[2]:.1f}, y {b[1]:.1f}–{b[3]:.1f} | 면적 {w * h / 100:.1f}% | 버튼 {len(btns)}개,"
              f" 버튼 폭 {mean(bw):.1f}%W, 높이 {mean(bh):.1f}%H, 버튼 종횡비(물리) {mean(bw) * A / mean(bh):.2f}")
    if 'board' in B:
        b = B['board']
        c = center(b)
        cw, ch = size(B['board_card'])
        pitch = (size(b)[0] - cw) / 4
        print(f"  공용 카드: 중심 ({c[0]:.1f}, {c[1]:.1f}), 폭 {size(b)[0]:.1f}%W | 카드 {cw:.1f}%W × {ch:.1f}%H,"
              f" 종횡비(물리) {cw * A / ch:.2f}, 간격(pitch) {pitch:.1f}%W, 틈 {pitch - cw:.1f}%W")
    if 'hero_cards' in B:
        b = B['hero_cards']
        c = center(b)
        line = f"  내 카드: 중심 ({c[0]:.1f}, {c[1]:.1f}), 크기 {size(b)[0]:.1f}%W × {size(b)[1]:.1f}%H"
        if 'hero_card' in B:
            cw, ch = size(B['hero_card'])
            line += f" | 한 장 {cw:.1f}%W × {ch:.1f}%H, 종횡비(물리) {cw * A / ch:.2f}"
        if 'board_card' in B and 'hero_card' in B:
            line += f", 공용 카드 대비 높이 {size(B['hero_card'])[1] / size(B['board_card'])[1]:.2f}배"
        print(line)
        if 'action_bar' in B:
            btns = sorted(k for k in B if k.startswith('action_btn_'))
            ds = [phys_dist(c, center(B[k]), A) for k in btns]
            print(f"  내 카드 중심 → 버튼 중심 거리(높이=100): 최소 {min(ds):.1f}, 최대 {max(ds):.1f}, 평균 {mean(ds):.1f}")
    if 'board' in B and 'hero_cards' in B:
        print(f"  공용 카드 중심 → 내 카드 중심 세로 간격: {center(B['hero_cards'])[1] - center(B['board'])[1]:.1f}%H")
    if 'pot' in B and 'board' in B:
        print(f"  팟 중심 → 공용 카드 중심 세로 간격: {center(B['board'])[1] - center(B['pot'])[1]:.1f}%H (양수 = 팟이 위)")
    seats = [P[k] for k in sorted(P) if k.startswith('seat_')]
    if seats and 'board' in B or (seats and m['mode'].startswith('홀덤') and 'table' in B):
        pts = seats + ([P['hero_avatar']] if 'hero_avatar' in P else [])
        if len(pts) >= 5:
            e = fit_ellipse(pts, A)
            print(f"  좌석 타원(내 자리 포함 {len(pts)}점): 중심 ({e['cx']:.1f}, {e['cy']:.1f}), 반축 a {e['a']:.1f}%W · b {e['b']:.1f}%H,"
                  f" 물리 비 b/a {e['b'] / e['a_phys']:.2f}, 잔차 RMS {e['rms']:.3f}")
            print(f"    좌석 각도(0°=아래, 시계 방향으로 왼쪽 90°·위 180°·오른쪽 270°): {sorted(e['angles'])}")
    trays = [k for k in ('tray_tl', 'tray_ml', 'tray_tr', 'tray_mr') if k in B]
    if len(trays) == 4:
        l = center(B['tray_tl'])[0]
        r = center(B['tray_tr'])[0]
        t = center(B['tray_tl'])[1]
        mrow = center(B['tray_ml'])[1]
        print(f"  트레이: 좌열 중심 x {l:.1f}, 우열 중심 x {r:.1f} (좌우 대칭 오차 {abs((l + r) / 2 - 50):.1f}), 윗줄 y {t:.1f}, 아랫줄 y {mrow:.1f},"
              f" 줄 간격 {mrow - t:.1f}%H, 트레이 크기 {size(B['tray_tl'])[0]:.1f}%W × {size(B['tray_tl'])[1]:.1f}%H")

# ---------------------------------------------------------------- family aggregates
fam = {
    '세로형 홀덤 (PS16, PS195, WPL, HR)': ['PS16', 'PS195', 'WPL', 'HR'],
    '가로형 7포커 (PP7, WJ7, HGPC7)': ['PP7', 'WJ7', 'HGPC7'],
    '가로형 홀덤 (PPH, HGPCH)': ['PPH', 'HGPCH'],
}
section('2. 계열별 집계 (평균 ± 표준편차 [최소–최대])')
for name, ids in fam.items():
    print(f"\n[{name}]")

    def col(el, fn):
        return [fn(boxes[s][el]) for s in ids if el in boxes[s]]

    def pcol(el, i):
        return [points[s][el][i] for s in ids if el in points[s]]

    for el, lab in (('table', '테이블'), ('pot', '팟'), ('total_box', 'Total/Call 박스'), ('board', '공용 카드 줄'),
                    ('hero_cards', '내 카드'), ('action_bar', '액션 영역'), ('menu_tr', '우상단 메뉴'), ('tabs', '테이블 탭'),
                    ('rank_panel', '족보/결과 패널'), ('tray_tl', '트레이 좌상'), ('tray_ml', '트레이 좌하'),
                    ('tray_tr', '트레이 우상'), ('tray_mr', '트레이 우하')):
        if not col(el, lambda b: b[0]):
            continue
        print(f"  {lab}: x0 {stats(col(el, lambda b: b[0]))} | y0 {stats(col(el, lambda b: b[1]))}"
              f" | x1 {stats(col(el, lambda b: b[2]))} | y1 {stats(col(el, lambda b: b[3]))}")
    for el, lab in (('hero_avatar', '내 아바타'), ('emoji', '이모티콘 버튼')):
        if pcol(el, 0):
            print(f"  {lab}: x {stats(pcol(el, 0))} | y {stats(pcol(el, 1))}")
    cw = [size(boxes[s]['board_card'])[0] for s in ids if 'board_card' in boxes[s]]
    ca = [size(boxes[s]['board_card'])[0] * screens[s]['aspect'] / size(boxes[s]['board_card'])[1] for s in ids if 'board_card' in boxes[s]]
    if cw:
        print(f"  공용 카드 한 장 폭(%W): {stats(cw)} | 물리 종횡비: {stats(ca, 2)}")

# ---------------------------------------------------------------- width-unit normalisation for portrait
section('3. 세로형: 세로 위치를 화면 폭 단위로 환산 (y% ÷ 종횡비 ÷ 100)')
for el in ('table', 'pot', 'board', 'hero_cards'):
    out = []
    for s in fam['세로형 홀덤 (PS16, PS195, WPL, HR)']:
        if el in boxes[s]:
            c = center(boxes[s][el])[1] / screens[s]['aspect'] / 100
            out.append((s, c))
    print(f"  {el} 중심 y (폭 단위): " + ', '.join(f'{s} {v:.2f}' for s, v in out) + f"  → {stats([v * 100 for _, v in out])} (×0.01W)")

# ---------------------------------------------------------------- Fitts-style reach
section('4. 버튼 도달 지수: ID = log2(D / W + 1)')
print('  기준점: 세로형은 화면 아래 중앙 (50, 100), 가로형 모바일은 가까운 아래 모서리, PC는 내 카드 중심(마우스 이동 가정)')
for s, m in screens.items():
    B = boxes[s]
    btns = sorted(k for k in B if k.startswith('action_btn_'))
    if not btns:
        continue
    A = m['aspect']
    out = []
    for k in btns:
        c = center(B[k])
        w = min(size(B[k])[0] * A, size(B[k])[1])
        if m['platform'] == 'pc':
            ref = center(B['hero_cards'])
        elif m['orientation'] == 'portrait':
            ref = (50, 100)
        else:
            ref = (0, 100) if c[0] < 50 else (100, 100)
        d = phys_dist(ref, c, A)
        out.append((labels[s][k], d, math.log2(d / w + 1)))
    print(f"  [{s}] " + ' | '.join(f'{lab}: D {d:.0f}, ID {i:.2f}' for lab, d, i in out))
