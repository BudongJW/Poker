# 로컬에서 이어가기

클라우드 세션에서 하던 작업을 사용자 컴퓨터에서 이어가기 위한 안내입니다. 기능 설명은 `README.md`, 개발 도구는 `tools/README.md`에 있습니다.

## 지금 상태

- 브랜치 `claude/confident-maxwell-j8xzfh`에 모든 작업이 올라가 있습니다. 테스트 122개가 통과합니다.
- 핵심 기능: 다른 프로그램의 창을 소스로 골라 Claude로 실시간으로 읽고(`vision.py`), 핸드를 따라가며 액션을 복원하고(`follow.py`, `table.py`), 상대 레인지와 내 승률을 계산해 웹 화면에 보여 줍니다(`web.py`, `static/index.html`).
- 실제로 확인한 것: 외부 브라우저 게임 두 개(Flash, HTML5)를 플레이하며 읽었고, 기록을 화면과 게임 내부 상태에 대조했습니다(README의 '외부 게임으로 확인', '실제 플레이 영상').
- 아직 확인하지 못한 것: 사용자 컴퓨터의 실제 창 선택 대화상자, 실제로 쓰는 연구용, 기록용 프로그램 화면.

## 받아 오기

```bash
git clone https://github.com/BudongJW/Poker.git
```

```bash
cd Poker
```

```bash
git checkout claude/confident-maxwell-j8xzfh
```

이미 받아 둔 저장소라면 `git fetch origin` 후 같은 브랜치로 바꾸고 `git pull`합니다.

## 설치와 점검

Python 3.9 이상이 필요합니다. Windows PowerShell 기준이며, macOS와 Linux는 `py` 대신 `python`, 키는 `export ANTHROPIC_API_KEY=...`로 넣습니다.

```powershell
py -m pip install -e ".[claude,dev]"
```

```powershell
setx ANTHROPIC_API_KEY "..."
```

```powershell
$env:ANTHROPIC_API_KEY = "..."
```

```powershell
py -m pokerlab check
```

`check`는 키와 모델을 확인하고 테스트용 그림 한 장을 읽습니다(약 0.03달러). `--no-read`를 붙이면 아무것도 보내지 않습니다.

## 실행

```powershell
py -m pokerlab serve --record hands.jsonl
```

브라우저(Chrome, Edge, Firefox)에서 `http://127.0.0.1:8765`를 열고 "창 추가"로 읽을 프로그램 창을 고릅니다. 소스마다 이름(프로그램 이름), 읽는 방식, 소스 메모를 적습니다. 상태 줄에 읽은 화면 수, 비용, 칩 합계 일치, 다시 읽은 횟수, 내 자리를 다른 곳으로 읽은 화면 수가 나옵니다.

소스 메모에는 화면에서 확실한 것만 적습니다. Flash 게임에서 "회색 태그는 폴드한 사람"이라고 잘못 적었더니 내 카드가 있는데도 폴드로 읽었습니다. `tools/fixtures/*_notes.txt`가 고친 예입니다.

## 테스트

```powershell
py -m pytest pokerlab/tests -q
```

약 2분 반 걸립니다. 추적기를 고쳤다면 `tools/misread_audit.py`와 `tools/replay_reads.py`도 돌려 봅니다(비용 없음).

## 클라우드에서 돌고 있는 것

- 매일 학습 예약 작업이 클라우드 세션에서 하루 두 번(07:38, 19:38 UTC) 불리고, 사이클은 하루 한 번 돕니다. 사이클이 돌면 `pokerlab/data/action_model.json`, `champion.json`, `history.jsonl`을 이 브랜치에 커밋합니다. 로컬에서 작업을 시작하기 전에 `git pull`로 받아 주세요.
- 사이클 11에서 처음으로 챔피언이 바뀌었습니다. 계속 돌릴지, 주 1회로 줄일지, 멈출지는 아직 정하지 않았습니다. 로컬에서 직접 돌리려면 `py -m pokerlab daily --formats kr_mtt`입니다(4코어에서 8~10분).

## 비용 기준

| 항목 | 비용 |
|---|---|
| Claude Opus 5.5로 한 장 읽기 (기본) | 약 0.018달러, 4~6초 |
| Claude Sonnet 5.5 (읽는 방식 "아껴서") | 약 0.009달러, 3.5~4초. 스택이 숫자로 보이는 프로그램에서만 권함 |
| 내 카드만 다시 읽기 (읽기가 갈릴 때 자동) | 약 0.007달러, 한 핸드 2번까지 |
| 보드만 크게 다시 읽기 (새 스트리트마다 자동) | 약 0.01달러, 한 핸드 3번까지 (플랍을 본 핸드 평균 1.75번) |
| 화면이 계속 움직이는 프로그램 | 창 하나에 시간당 약 11달러 (다시 읽는 간격을 늘리면 줄어듦) |

## 남은 일 (권하는 순서)

1. 사용자 컴퓨터에서 `check`, `serve`, 창 추가까지 실제로 확인하기.
2. 실제로 쓰는 프로그램 화면으로 시험하고 소스 메모 다듬기. 상태 줄의 품질 표시와 `py -m pokerlab quality hands.jsonl`로 확인합니다.
3. 순서를 추정한 레이즈("(추정)" 표시)가 들어간 핸드를 학습에서 뺄지 정하기. 지금은 기록에 표시만 하고 학습에는 그대로 씁니다.
4. 매일 학습을 계속할지 정하기.
5. 읽기 지연(4~6초) 줄이기. 짧은 답 형식(`POKERLAB_VISION_COMPACT=1`)은 1~2초 빠르지만 정확도가 떨어져 기본으로 두지 않았습니다.
