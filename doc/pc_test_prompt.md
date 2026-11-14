# 실제 PC 테스트용 프롬프트

캡처 영역 선택 기능과 시간 측정 도구를 실제 Windows PC에서 검증할 때 쓰는 프롬프트다. PC에서 실행하는 Claude Code 같은 코딩 에이전트에게 아래 `---` 사이의 내용을 그대로 붙여 넣는다.

- 걸리는 시간: 설치를 빼고 약 20분이다. 이 중 사람이 직접 하는 GUI 점검이 5~10분이다.
- 결과: 에이전트가 `pc_test_results/report.md`를 만들고, 마지막 답변에 보고서 전문을 출력한다. 그 내용을 이 저장소 작업 세션에 다시 붙여 넣으면 된다.

---

# 작업: Poker 저장소의 캡처 영역 기능을 이 PC에서 검증하고 보고서 작성

## 배경

- 저장소: https://github.com/BudongJW/Poker, 브랜치 `claude/wizardly-archimedes-abp3ex`
- 이 브랜치에 새로 들어간 기능은 두 가지다.
  - **캡처 영역 선택**: 봇이 전체 화면, 지정한 좌표 영역, 특정 게임창 중 어디를 캡처할지 고른다.
  - **시간 측정 도구**: 화면 인식과 계산에 걸리는 시간을 잰다.
- 지금까지는 리눅스 가상 디스플레이에서만 확인했다. 이번 목표는 실제 PC(주로 Windows)에서 같은 결과가 나오는지 확인하는 것이다.
- 먼저 `doc/capture_area.md`와 `doc/timing_benchmark.md`를 읽고 기능과 기대값을 파악한다.
- 봇 본체(`poker/main.py`)는 시작할 때 원격 서버(`dickreuter.com:7778`)가 필요한데, 이 서버가 응답하지 않는다. 그래서 봇 본체는 실행하지 않는다. 대신 아래의 단독 실행 도구만 쓴다.

## 규칙

1. 실제 게임에서 봇을 실행하거나 자동 클릭을 하지 않는다. 화면 인식 점검에는 저장소의 스크린샷을 띄운 **가짜 게임창**만 쓴다. 점검 도구는 마우스를 움직이지 않는다.
2. 저장소 코드는 수정하지 않는다. 문제가 생기면 명령, 전체 출력, 스크린샷 경로를 기록하고 다음 단계로 넘어간다.
   - 설치 문제를 풀기 위한 우회(다른 버전 설치 등)는 해도 된다. 무엇을 했는지 기록한다.
3. 커밋이나 푸시를 하지 않는다. 결과물은 모두 `pc_test_results/` 폴더에 둔다.
4. 사람이 해야 하는 단계(4단계)는 사용자에게 단계별로 안내하고, 사용자가 답한 결과를 그대로 기록한다. 직접 했다고 가정하지 않는다.
5. 측정(2단계, 5단계)이 도는 동안에는 PC를 쓰지 말아 달라고 사용자에게 미리 알린다.

## 0단계: 준비

1. git과 Python 3.11(64비트)이 있는지 확인한다. 없으면 설치한다(예: `winget install Git.Git`, `winget install Python.Python.3.11`).
   - Python 3.11이어야 하는 이유: 배포 빌드가 3.11이고, Windows용 tesserocr 휠이 3.11용이다.
2. 저장소를 받고 브랜치로 이동한다.
   - `git clone https://github.com/BudongJW/Poker.git`
   - `cd Poker`
   - `git checkout claude/wizardly-archimedes-abp3ex`
   - `git log -1 --oneline` 출력을 기록한다.
3. 가상환경을 만든다. 아래 명령은 PowerShell과 Git Bash 둘 다에서 쓸 수 있게 `./.venv-pctest/Scripts/python.exe`로 적었다.
   - `py -3.11 -m venv .venv-pctest`
4. 패키지를 설치한다.
   - `./.venv-pctest/Scripts/python.exe -m pip install "numpy<2" "pandas==2.0.3" "matplotlib==3.7.2" "numexpr==2.8.4" opencv-python pillow PyQt6 scipy "lmfit==1.3.3" openpyxl "fastapi==0.101.1" requests tqdm virtualbox pyyaml seaborn "uvicorn==0.23.2" pywin32 pytest`
   - `./.venv-pctest/Scripts/python.exe -m pip install https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.6.0-tesseract-5.3.1/tesserocr-2.6.0-cp311-cp311-win_amd64.whl`
   - lmfit은 반드시 1.3.3이다. 1.3.4에서는 의사결정 단계가 오류로 멈춘다.
   - macOS라면 tesserocr는 `brew install tesseract` 후 pip로 설치하고, `pyobjc-framework-Quartz`를 추가로 설치한다. pywin32는 빼고, 이후 명령의 python 경로는 `./.venv-pctest/bin/python`으로 바꾼다.
5. `mkdir pc_test_results`
6. `poker/config.ini`를 `pc_test_results/config.ini.bak`으로 복사한다. 4단계에서 설정을 저장하므로 마지막에 되돌린다.
7. 환경 정보를 기록한다.
   - OS 버전, CPU 모델, RAM
   - 주 모니터 해상도, 디스플레이 배율(%), 모니터 개수
   - `./.venv-pctest/Scripts/python.exe -m pip freeze`에서 numpy, pandas, opencv, pillow, PyQt6, lmfit, tesserocr 버전

## 1단계: 자동 테스트

- `./.venv-pctest/Scripts/python.exe -m pytest poker/tests/test_capture_target.py poker/tests/test_outs.py "poker/tests/test_table_and_ocr.py::test_orc_problems1" "poker/tests/test_table_and_ocr.py::test_orc_problems2" -q -p no:cacheprovider -o log_cli=false`
- 기대값: `39 passed`
- 다른 테스트 파일은 원격 서버가 필요해서 실패하는 게 정상이므로 돌리지 않는다.

## 2단계: 실제 화면 자동 점검

1. 사용자에게 1~2분 동안 PC를 쓰지 말고, 화면 왼쪽 위 (300,150)~(1370,930) 부근을 다른 창이 가리지 않게 해 달라고 알린다.
2. 실행한다.
   - `./.venv-pctest/Scripts/python.exe -m poker.tools.capture_check --repeat 10 --out pc_test_results/capture_check`
3. 이 도구는 별도 프로세스로 가짜 게임창을 (300,150)에 띄우고 다음을 확인한다.
   - 창 제목: `Capture Check - Fake Hold'em Table $0.01/$0.02`
   - 크기: 950×700
   - 제목에 실제 테이블처럼 `$`가 들어 있다.

| 항목 | 기대 결과 |
|---|---|
| window listing | 창을 (300, 150, 1250, 850)에서 찾음 |
| window capture | 창 캡처가 원본 이미지와 픽셀 단위로 같음(차이 0.00%) |
| recognition window mode | 카드 JD JS / 6C 8H 9H JC, pot 0.29, call 0.04, raise 0.08, mouse reference (300, 150) |
| recognition full screen mode | 위와 같음 |
| moved window | 창이 (420,230)으로 다시 열려도 mouse reference가 (420, 230) |
| missing window | 창이 없으면 전체 화면을 캡처함 |
| [TIME] | 게임창 모드의 테이블 위치 찾기와 화면 인식이 전체 화면 모드보다 짧음 |

4. 마지막 줄 `Result: PASS/FAIL`과 `pc_test_results/capture_check/result.json`을 기록한다.
   - FAIL이 있으면 `pc_test_results/capture_check/`에 저장된 이미지를 열어 보고 무엇이 보이는지 적는다.
   - window listing의 크기가 950×700이 아니면 디스플레이 배율 문제일 수 있으니 배율과 함께 적는다.
5. (선택) 디스플레이 배율이 100%라면, 사용자에게 125% 또는 150%로 바꿔 달라고 요청한 뒤 2단계를 한 번 더 실행한다. 출력 폴더는 `pc_test_results/capture_check_scaled`로 한다. 끝나면 배율을 원래대로 돌려 달라고 한다.

## 3단계: 명령줄 도구

1. 다른 터미널에서 가짜 게임창을 10분 동안 띄운다.
   - `./.venv-pctest/Scripts/python.exe -m poker.tools.capture_check --window-only 600`
   - 백그라운드로 실행하고 4단계가 끝날 때까지 둔다.
2. 아래 명령을 차례로 실행하고, 출력에 나온 크기와 위치를 기록한다.
   - `./.venv-pctest/Scripts/python.exe -m poker.tools.capture_target --list`
   - `./.venv-pctest/Scripts/python.exe -m poker.tools.capture_target --window "Fake Hold'em" --grab pc_test_results/grab_window.png`
   - `./.venv-pctest/Scripts/python.exe -m poker.tools.capture_target --region 300,150,1250,850 --grab pc_test_results/grab_region.png`
   - `./.venv-pctest/Scripts/python.exe -m poker.tools.capture_target --full-screen --grab pc_test_results/grab_full.png`
3. 기대 결과
   - `--list`에 가짜 게임창이 `300, 150 950x700`으로 나온다.
   - grab_window와 grab_region은 950×700이고 화면 위치는 (300, 150)이다. 이미지를 열어 보면 포커 테이블 전체가 잘림 없이 보인다.
   - grab_full은 주 모니터 해상도와 크기가 같다.
4. 각 명령이 출력한 캡처 시간(ms)도 기록한다.

## 4단계: 사람이 하는 GUI 점검

준비: 3단계의 가짜 게임창이 떠 있어야 한다. 아래 명령으로 설정 창만 따로 띄운다.

- `./.venv-pctest/Scripts/python.exe -m poker.gui.capture_target_dialog`

사용자에게 항목을 하나씩 안내하고, 각 항목마다 "예상대로 됨 / 다르게 됨(무엇이 보였는지)"을 받아 기록한다. 사용자가 스크린샷을 주면 경로를 기록한다.

| 번호 | 사용자가 할 일 | 예상 결과 |
|---|---|---|
| M1 | Game window 목록을 연다 | `Capture Check - Fake Hold'em Table $0.01/$0.02 (950 x 700)`이 있다. 봇 설정 창 자신은 목록에 없다. |
| M2 | **Pick on screen...** 을 누르고 가짜 게임창 위에 마우스를 올린 뒤 클릭한다 | 설정 창이 잠시 사라지고 멈춘 화면이 어둡게 보인다. 마우스를 올린 창에 파란 테두리와 제목이 표시된다. 클릭하면 설정 창이 돌아오고, Title contains에 창 제목이 들어간다. 미리보기에 테이블 전체가 보이고, 상태 줄에 `950 x 700 at 300,150`이 나온다. |
| M3 | Screen region에서 **Select on screen...** 을 누르고 가짜 게임창 테두리에 맞춰 드래그한다 | 드래그하는 동안 좌표와 크기가 표시된다. 놓으면 x1, y1, x2, y2가 약 300, 150, 1250, 850(±3)이 된다. 디스플레이 배율이 100%가 아니어도 이 값이어야 한다. |
| M4 | **Select on screen...** 을 다시 누르고 Esc를 누른다. 한 번 더 열어 오른쪽 클릭한다 | 두 경우 모두 선택이 취소되고, 설정 창이 그대로 돌아오며 값이 바뀌지 않는다. 프로그램이 종료되지 않는다. |
| M5 | Game window를 고른다(M2에서 고른 상태여도 된다). Title contains는 `$`가 들어간 전체 제목 그대로 두고 **Test capture**, 이어서 **Save**를 누른다 | 미리보기가 테이블 전체를 보여준다. Save를 누르면 오류 없이 창이 닫히고, 터미널에 `Saved to config.ini: Window with title containing 'Capture Check - Fake Hold'em Table $0.01/$0.02'`가 출력된다. |
| M6 | (에이전트가 확인) `poker/config.ini`를 읽고, `./.venv-pctest/Scripts/python.exe -m poker.tools.capture_target --grab pc_test_results/grab_saved.png`를 실행한다 | config.ini에 `capture_mode = window`와 `capture_window_title = Capture Check - Fake Hold'em Table $$0.01/$$0.02`가 있다. config.ini에서는 `$`를 `$$`로 적는 게 정상이다. 캡처는 950×700, 위치 (300, 150)이다. |
| M7 | (에이전트가 확인) 가짜 게임창 프로세스를 끝낸 뒤 M6의 `--grab`을 다시 실행한다 | 창을 찾지 못했다는 경고가 나오고, 전체 화면 크기로 캡처된다. |

선택 항목(가능할 때만):

- **M8 모니터 2개**: 메모장 같은 평범한 창을 두 번째 모니터로 옮긴다.
  - `--window "메모장"`(영문 Windows는 `"Notepad"`)으로 `--grab`한다.
  - 이미지에 그 창이 제대로 담기는지 확인한다.
- **M9 실제 게임 클라이언트**: 사용자가 게임 클라이언트를 이미 열어 둔 경우에만 한다.
  - `--list`에서 창 제목을 확인하고, 그 제목 일부로 `--grab`해서 화면이 정상으로 캡처되는지(검은 화면이나 잘림이 없는지) 확인한다.
  - 캡처만 하고 봇은 실행하지 않는다.

끝나면 `pc_test_results/config.ini.bak`을 `poker/config.ini`로 되돌린다.

## 5단계: 시간 측정

사용자에게 약 6분 동안 PC를 쓰지 말아 달라고 알린 뒤 차례로 실행한다.

- `./.venv-pctest/Scripts/python.exe -m poker.tools.timing_benchmark --repeat 30 --warmup 3 --json pc_test_results/bench_full.json`
- `./.venv-pctest/Scripts/python.exe -m poker.tools.timing_benchmark --repeat 30 --warmup 3 --capture window --no-stages --json pc_test_results/bench_window.json`

두 실행의 콘솔 출력을 각각 `pc_test_results/bench_full.txt`, `pc_test_results/bench_window.txt`로도 저장한다. `mismatches against expected values: none`이 아니면 FAIL로 기록한다.

참고값은 클라우드 서버(Xeon 2.1GHz 4 vCPU, Linux)에서 잰 평균이다. 단위는 ms다.

| 항목 | PokerStars | PartyPoker |
|---|---:|---:|
| 화면 인식 (전체 화면) | 448~466 | 459~497 |
| 화면 인식 (게임창만) | 379 | 356 |
| 테이블 위치 찾기 (전체 화면 → 게임창만) | 146 → 51 | 147 → 48 |
| 계산 (몬테카를로 + 의사결정) | 277 | 254 |
| 판단 1회 합계 (전체 화면) | 743 | 751 |

## 6단계: 보고서

`pc_test_results/report.md`를 아래 구성으로 작성하고, 마지막 답변에 보고서 전문을 그대로 출력한다.

1. **요약**: 전체 결과(통과/실패 개수)와 가장 중요한 문제 3개 이내
2. **환경**: 0단계에서 기록한 정보, 테스트한 커밋(`git log -1 --oneline`)
3. **설치**: 설치 중 생긴 문제와 해결 방법. 문제가 없었으면 "없음".
4. **단계별 결과 표**: 항목 | 결과(PASS/FAIL/WARN/SKIP) | 측정값 또는 메모
   - 1단계, 2단계 6개 항목, 3단계, M1~M9를 모두 넣는다.
5. **시간 측정**
   - capture_check의 [TIME] 두 줄
   - 벤치마크 결과를 참고값 표와 같은 형식으로 정리하고, 참고값과 나란히 비교한다.
   - 게임 단계·상대 수별 계산 시간 표(bench_full의 마지막 표)도 넣는다.
6. **실패와 이상 내역**: 항목마다 실행한 명령, 전체 오류 출력, 관련 이미지 경로, 사용자가 본 내용
7. **config.ini 복원 여부**

---
