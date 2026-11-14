#!/usr/bin/env python3
"""리그·대회 통계 (10_leagues_tournaments.md의 수치를 다시 계산한다).

사용법:  python3 league_stats.py            # 같은 폴더의 CSV 두 개를 읽는다
표준 라이브러리만 쓴다.

입력
  league_wpl_schedule.csv   WPL 홈페이지 토너먼트 일정표(2026-11-13 수집) 154행
  league_parameters.csv     게임별 대회·리그 변수(문서에서 인용한 값만 모은 표). source 열은 문서의 근거 ID
"""
import csv
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DAY = 24 * 60


# ---------------------------------------------------------------- 공통 함수
def minutes(hhmm):
    h, m = hhmm.split(':')
    return int(h) * 60 + int(m)


def gaps(starts):
    """하루를 원으로 보고 인접한 시작 시각 사이 간격(분)을 돌려준다."""
    s = sorted(set(starts))
    if not s:
        return []
    return [(s[(i + 1) % len(s)] - s[i]) % DAY or DAY for i in range(len(s))]


def expected_wait(starts):
    """임의 시각에 접속한 사람이 다음 시작까지 기다리는 시간의 기댓값.

    간격이 g_i 이면 E[W] = sum(g_i^2) / (2 * 1440). 간격이 고르면 g/2 가 된다.
    """
    g = gaps(starts)
    return sum(x * x for x in g) / (2 * DAY) if g else float('nan')


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float('nan')
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def linfit(xs, ys):
    """최소제곱 직선 y = a + b x 와 결정계수."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1 - ss_res / ss_tot


def won(n):
    """코인 금액을 조·억·만 단위로 적는다."""
    n = int(round(n))
    out = []
    for unit, name in ((10 ** 12, '조'), (10 ** 8, '억'), (10 ** 4, '만')):
        if n >= unit:
            out.append(f'{n // unit:,}{name}')
            n %= unit
    if n or not out:
        out.append(f'{n:,}')
    return ' '.join(out[:2])


def head(title):
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


# ---------------------------------------------------------------- 1부: WPL 일정표
def wpl(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    for r in rows:
        r['t'] = minutes(r['start'])
        r['b'] = int(r['buyin']) if r['buyin'] else None
        r['g'] = int(r['gtd']) if r['gtd'] else None

    head('1. WPL 일정표 구성')
    print('행 수:', len(rows), dict(Counter(r['recurrence'] for r in rows)))
    daily = [r for r in rows if r['recurrence'] == 'daily']
    wd = daily + [r for r in rows if r['recurrence'] == 'weekday']
    we = daily + [r for r in rows if r['recurrence'] == 'weekend']
    print(f'평일 하루 대회 수: {len(wd)}   주말 하루 대회 수: {len(we)}   (시리즈 날짜 지정 대회 제외)')

    print('\n[등급별 대회 수(평일·주말 전용 포함) · 바이인 범위 · 개런티 범위]')
    by = defaultdict(list)
    for r in wd + [r for r in rows if r['recurrence'] == 'weekend']:
        by[r['tier']].append(r)
    for tier, rs in sorted(by.items(), key=lambda kv: min((x['b'] or 0) for x in kv[1])):
        bs = [x['b'] for x in rs if x['b']]
        gs = [x['g'] for x in rs if x['g']]
        btxt = f'{won(min(bs))} ~ {won(max(bs))}' if bs else '무료'
        print(f'  {tier:14s} n={len(rs):3d}  바이인 {btxt:28s} 개런티 {won(min(gs))} ~ {won(max(gs))}')

    print('\n[형식 비중 — 매일 대회 117개]')
    n = len(daily)
    for k, v in Counter(r['format'] for r in daily).most_common():
        print(f'  {k:10s} {v:3d}  {v / n:6.1%}')
    turbo = sum(r['speed'] == 'turbo' for r in daily)
    print(f'  turbo      {turbo:3d}  {turbo / n:6.1%}   (이름에 "터보"가 붙은 대회)')
    ser = sum(r['series'] == '1' for r in daily)
    print(f'  시리즈 딱지([단풍]) {ser}개 {ser / n:.1%}')

    head('2. 시작 간격과 기다리는 시간 (WPL, 평일 기준)')
    t_all = [r['t'] for r in wd]
    g = gaps(t_all)
    print(f'시작 시각 {len(set(t_all))}개, 간격 분포 {dict(sorted(Counter(g).items()))}')
    print(f'간격 평균 {st.mean(g):.1f}분, 중앙값 {st.median(g):.0f}분, 최대 {max(g)}분')
    print(f'한 시작 시각에 열리는 대회 수 평균 {len(wd) / len(set(t_all)):.2f}개')
    print(f'임의 시각 접속자의 다음 대회 대기 기댓값 E[W] = {expected_wait(t_all):.1f}분 (최대 {max(g)}분)')

    print('\n[조건별 다음 대회 대기 — 시작 수, 평균 간격, E[W], 최대 대기]')

    def show(label, rs):
        ts = [r['t'] for r in rs]
        gg = gaps(ts)
        if not gg:
            print(f'  {label:26s} 없음')
            return
        print(f'  {label:26s} 하루 {len(set(ts)):3d}회  평균 간격 {st.mean(gg):6.1f}분  '
              f'E[W] {expected_wait(ts):6.1f}분  최대 {max(gg):4d}분')

    show('전체', wd)
    show('프리롤', [r for r in wd if r['format'] == 'freeroll'])
    for cap, lab in ((10 ** 6, '바이인 100만 이하'), (10 ** 7, '바이인 1,000만 이하'),
                     (10 ** 8, '바이인 1억 이하'), (3 * 10 ** 8, '바이인 3억 이하')):
        show(lab + '(유료)', [r for r in wd if r['b'] and r['b'] <= cap])
    show('바이인 3억 초과', [r for r in wd if r['b'] and r['b'] > 3 * 10 ** 8])
    show('터보', [r for r in wd if r['speed'] == 'turbo'])
    show('바운티', [r for r in wd if r['format'] == 'bounty'])
    show('프리즈아웃', [r for r in wd if r['format'] == 'freezeout'])
    for tier in ('beginners', 'rookies', 'stepup', 'warmup', 'classic', 'daily_special'):
        show('등급 ' + tier, [r for r in wd if r['tier'] == tier])

    print('\n[시간대별 대회 수 — 평일]')
    hours = Counter(r['t'] // 60 for r in wd)
    for h in range(24):
        print(f'  {h:02d}시 {hours[h]:2d} ' + '#' * hours[h])
    vals = [hours[h] for h in range(24)]
    print(f'  시간당 평균 {st.mean(vals):.2f}개, 표준편차 {st.pstdev(vals):.2f}, '
          f'최소 {min(vals)}개({vals.index(min(vals)):02d}시대), 최대 {max(vals)}개')
    for lab, hs in (('새벽 02~07시', range(2, 8)), ('오전 08~11시', range(8, 12)),
                    ('낮 12~17시', range(12, 18)), ('저녁 18~23시', range(18, 24)), ('자정 00~01시', range(0, 2))):
        v = [hours[h] for h in hs]
        print(f'  {lab}: 시간당 {st.mean(v):.2f}개')

    head('3. 바이인 분포 (WPL 매일 유료 대회)')
    paid = [r for r in wd if r['b']]
    bs = [r['b'] for r in paid]
    print(f'n={len(bs)}  최소 {won(min(bs))}  Q1 {won(quantile(bs, .25))}  중앙값 {won(st.median(bs))}  '
          f'Q3 {won(quantile(bs, .75))}  최대 {won(max(bs))}')
    print(f'산술평균 {won(st.mean(bs))}  기하평균 {won(gmean(bs))}  최대/최소 = {max(bs) / min(bs):,.0f}배 '
          f'(log10 폭 {math.log10(max(bs) / min(bs)):.1f}자리)')
    lg = [math.log10(b) for b in bs]
    print(f'log10(바이인) 평균 {st.mean(lg):.2f}, 표준편차 {st.pstdev(lg):.2f}')
    print('[자릿수별 대회 수]')
    for lo in range(5, 10):
        c = sum(10 ** lo <= b < 10 ** (lo + 1) for b in bs)
        print(f'  {won(10 ** lo):>6s} 이상 {won(10 ** (lo + 1)):>6s} 미만: {c:3d}  {c / len(bs):6.1%}')

    head('4. 개런티와 바이인의 관계 (WPL 매일 유료 대회)')
    gs = [r['g'] for r in paid]
    print(f'하루 개런티 합계(평일, 프리롤 포함): {won(sum(r["g"] for r in wd if r["g"]))}')
    print(f'하루 개런티 합계(주말, 프리롤 포함): {won(sum(r["g"] for r in we if r["g"]))}')
    print(f'개런티 중앙값 {won(st.median(gs))}, 기하평균 {won(gmean(gs))}')
    ratio = [r['g'] / r['b'] for r in paid]
    print(f'개런티 ÷ 바이인(참가비만으로 개런티를 채우는 데 필요한 엔트리 수의 하한): 중앙값 {st.median(ratio):.0f}, '
          f'Q1 {quantile(ratio, .25):.0f}, Q3 {quantile(ratio, .75):.0f}, 최소 {min(ratio):.0f}, 최대 {max(ratio):.0f}, '
          f'기하평균 {gmean(ratio):.0f}')
    a, b, r2 = linfit([math.log10(r['b']) for r in paid], [math.log10(r['g']) for r in paid])
    print(f'log10(개런티) = {a:.2f} + {b:.3f}·log10(바이인),  R² = {r2:.3f}')
    print(f'  → 개런티 ≈ {10 ** a:,.0f} × 바이인^{b:.3f}  (바이인 10배 → 개런티 {10 ** b:.1f}배)')
    print('[등급별 개런티 ÷ 바이인 중앙값]')
    by = defaultdict(list)
    for r in paid:
        by[r['tier']].append(r['g'] / r['b'])
    for tier, v in sorted(by.items(), key=lambda kv: -st.median(kv[1])):
        print(f'  {tier:14s} n={len(v):3d}  중앙값 {st.median(v):6.0f}  범위 {min(v):.0f} ~ {max(v):.0f}')
    print('[형식별 개런티 ÷ 바이인 중앙값]')
    for key, lab in (('turbo', '터보'), ('regular', '일반')):
        v = [r['g'] / r['b'] for r in paid if r['speed'] == key]
        print(f'  {lab}: n={len(v)} 중앙값 {st.median(v):.0f}')
    v1 = [r['g'] / r['b'] for r in paid if r['series'] == '1']
    v0 = [r['g'] / r['b'] for r in paid if r['series'] == '0']
    print(f'  시리즈 딱지 있음: n={len(v1)} 중앙값 {st.median(v1):.0f} / 없음: n={len(v0)} 중앙값 {st.median(v0):.0f}')

    head('5. 멀티데이·시리즈 대회 (WPL 단풍 시리즈)')

    def days(sched):
        """'10.19(월) ~ 10.25(일)' 같은 표기에서 날짜 수를 센다."""
        import datetime
        import re
        d = [datetime.date(2026, int(m), int(dd)) for m, dd in re.findall(r'(\d+)\.(\d+)\(', sched)]
        return (d[-1] - d[0]).days + 1 if d else 0

    d1 = [r for r in rows if r['tier'] == 'multiday_day1']
    fin = {r['name'].replace('FINAL', 'X'): r for r in rows if r['tier'] == 'multiday_final'}
    ev = defaultdict(list)
    for r in d1:
        ev[r['name']].append(r)
    rat, per_flight, gap_h = [], [], []
    for name, rs in ev.items():
        f = fin[name.replace('DAY1', 'X')]
        slots = sorted(r['start'] for r in rs)
        b = rs[0]['b']
        flights = sum(days(r['schedule']) for r in rs)   # LAST CHANCE(마지막 날 낮 회차) 포함
        m = f['g'] / b
        last = min(minutes(s) for s in slots)            # LAST CHANCE는 가장 이른 시각에 열린다
        gap = (minutes(f['start']) - last) / 60
        rat.append(m); per_flight.append(m / flights); gap_h.append(gap)
        print(f'  {name:34s} DAY1 {"/".join(slots)}  플라이트 {flights:2d}개  바이인 {won(b):12s} '
              f'FINAL {f["schedule"]} {f["start"]}  배수 {m:5.0f}  플라이트당 {m / flights:4.0f}  LAST CHANCE→FINAL {gap:.0f}시간')
    print(f'  멀티데이 개런티÷바이인: 평균 {st.mean(rat):.0f}, 중앙값 {st.median(rat):.0f}, 범위 {min(rat):.0f} ~ {max(rat):.0f}')
    print(f'  플라이트 하나당 배수: 평균 {st.mean(per_flight):.0f}, 중앙값 {st.median(per_flight):.0f}, 범위 {min(per_flight):.0f} ~ {max(per_flight):.0f}')
    print(f'  LAST CHANCE 시작에서 FINAL 시작까지: 평균 {st.mean(gap_h):.1f}시간, 범위 {min(gap_h):.0f} ~ {max(gap_h):.0f}시간')
    for r in rows:
        if r['tier'] in ('trophy', 'series_side'):
            print(f'  {r["name"]:42s} {r["schedule"]} {r["start"]}  바이인 {won(r["b"])}  개런티÷바이인 {r["g"] / r["b"]:.0f}')


# ---------------------------------------------------------------- 2부: 게임 간 비교
def params(path):
    if not os.path.exists(path):
        print('\n(league_parameters.csv 없음 — 2부 생략)')
        return
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    head('6. 게임 간 비교 — 변수별 평균 (league_parameters.csv)')
    print(f'행 수 {len(rows)}, 게임 {len(set(r["game"] for r in rows))}종')
    print('\n[종류별 행 수]')
    for k, v in Counter(r['kind'] for r in rows).most_common():
        games = sorted(set(r['game'] for r in rows if r['kind'] == k))
        print(f'  {k:16s} {v:2d}  ({", ".join(games)})')

    def stat(label, vals):
        """vals: (값, 행) 목록. 평균·중앙값·범위를 찍고 구성 행을 나열한다."""
        if not vals:
            return
        xs = [v for v, _ in vals]
        line = f'  {label:34s} n={len(xs):2d}  평균 {st.mean(xs):7.1f}  중앙값 {st.median(xs):7.1f}  범위 {min(xs):g} ~ {max(xs):g}'
        if len(xs) > 1:
            line += f'  표준편차 {st.pstdev(xs):.1f}'
        print(line)
        for v, r in sorted(vals, key=lambda x: x[0]):
            print(f'      {v:8g}  {r["game"]} / {r["competition"]}')

    def pick(col, kinds=None):
        return [(float(r[col]), r) for r in rows
                if r.get(col) not in (None, '') and (kinds is None or r['kind'] in kinds)]

    print('\n[변수별 통계 — 값이 있는 행만, 성격이 같은 종류끼리 묶는다]')
    stat('예선: 하루 시작 횟수', pick('starts_per_day', {'satellite'}))
    stat('예선: 시작 간격(분)', pick('interval_min', {'satellite'}))
    stat('예선: 티켓 획득 비율(%)', pick('advance_pct', {'satellite'}))
    stat('본선·멀티데이: DAY1 통과 비율(%)', pick('advance_pct', {'offline_main', 'multiday'}))
    stat('리그: 승급 비율(%)', pick('advance_pct', {'league'}))
    stat('리그·단체전·미션: 한 회차 길이(일)', pick('season_days', {'league', 'team', 'mission'}))
    stat('시리즈: 기간(일)', pick('season_days', {'series'}))
    stat('온라인 대회: 레벨 길이(분)', pick('level_min', {'satellite', 'mtt', 'sng', 'championship', 'multiday'}))
    stat('오프라인 본선: 레벨 길이(분)', pick('level_min', {'offline_main'}))
    stat('한 번의 베팅 시간(초)', pick('action_sec'))
    stat('온라인: 추가 등록 마감 레벨', pick('late_reg_levels', {'satellite', 'mtt'}))
    stat('온라인: 시작 뒤 추가 등록 시간(분)', pick('late_reg_min', {'satellite', 'mtt', 'championship', 'multiday'}))
    stat('오프라인: 시작 뒤 추가 등록 시간(분)', pick('late_reg_min', {'offline_main'}))
    stat('리엔트리·리바이인 횟수', pick('reentry_max'))
    stat('정원제 대회 인원(명)', pick('seats', {'sng'}))
    stat('리그 한 조 인원(명)', pick('seats', {'league'}))
    stat('단발 대회 진행 시간(분)', pick('duration_min'))
    stat('자리 비움 뒤 기권까지(분)', pick('away_forfeit_min'))
    print('\n[시작 시각이 정해진 대회의 간격과 대기]')
    print('  창 = 첫 시작부터 마지막 시작까지. E[W]창 = 창 안에서 접속했을 때, E[W]하루 = 하루 중 아무 때나 접속했을 때')
    sat_hours = []
    for r in rows:
        ts = [minutes(x) for x in r['start_times'].split(';') if x]
        if len(ts) < 2:
            continue
        ts = sorted(ts)
        inner = [b - a for a, b in zip(ts, ts[1:])]
        span = ts[-1] - ts[0]
        ew_in = sum(x * x for x in inner) / (2 * span)
        day = gaps([t % DAY for t in ts])
        print(f'  {r["game"]:10s} {r["competition"]:32s} {len(ts):2d}회  창 {ts[0] // 60:02d}:{ts[0] % 60:02d}~{ts[-1] // 60:02d}:{ts[-1] % 60:02d}'
              f'  평균 간격 {st.mean(inner):5.0f}분  E[W]창 {ew_in:5.1f}분  E[W]하루 {expected_wait([t % DAY for t in ts]):5.0f}분  최대 공백 {max(day):4d}분')
        if r['kind'] == 'satellite':
            sat_hours += [t / 60 for t in ts]
    if sat_hours:
        eve = sum(17 <= h <= 24 for h in sat_hours) / len(sat_hours)
        print(f'  예선(새틀라이트) 시작 {len(sat_hours)}건: 평균 시각 {st.mean(sat_hours):.1f}시, 중앙값 {st.median(sat_hours):.1f}시, '
              f'17~24시 비중 {eve:.0%}')
    sats = [r for r in rows if r['kind'] == 'satellite' and r['starts_per_day']]
    if sats:
        xs = [float(r['starts_per_day']) for r in sats]
        print(f'  예선 하루 횟수: 평균 {st.mean(xs):.1f}, 중앙값 {st.median(xs):.1f}, 범위 {min(xs):g} ~ {max(xs):g} (n={len(xs)})')
    print('\n[추가 등록 창 = 레벨 길이 × 마감 레벨 수]')
    for r in rows:
        if r.get('late_reg_levels') and r.get('level_min'):
            lv, m = float(r['late_reg_levels']), float(r['level_min'])
            print(f'  {r["game"]:16s} {r["competition"]:34s} {lv:g}레벨 × {m:g}분 = {lv * m:g}분')


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else HERE
    wpl(os.path.join(base, 'league_wpl_schedule.csv'))
    params(os.path.join(base, 'league_parameters.csv'))
