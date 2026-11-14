# 캡처 영역 선택 (전체 화면 / 화면 좌표 / 게임창)

봇이 화면의 어느 부분을 캡처할지 고르는 기능이다. 지금까지는 항상 전체 화면을 캡처한 뒤 그 안에서 테이블을 찾았다. 이제는 아래 세 가지 중에서 고를 수 있다.

| 모드 | 캡처 범위 | 쓰임 |
|---|---|---|
| 전체 화면 (`full_screen`) | 주 모니터 전체 | 기존 동작과 같다. 설정이 없으면 이 모드로 동작한다. |
| 화면 영역 (`region`) | 지정한 좌표 `x1,y1,x2,y2` | 게임창 위치가 고정되어 있을 때 |
| 게임창 (`window`) | 제목에 지정한 글자가 들어간 창 | 게임창을 옮겨도 따라간다. 캡처할 때마다 창 위치를 다시 찾는다. |

캡처한 범위 안에서만 테이블을 찾으므로 범위가 작을수록 빠르다. 마우스 클릭 좌표는 캡처 범위의 화면 위치만큼 자동으로 보정된다.

## 사용 방법

### GUI

1. 메인 창의 **Settings**를 열고 **Select capture area...** 를 누른다. Table Setup 창의 **Capture area...** 버튼으로도 같은 창이 열린다.
2. 모드를 고른다.
   - **Screen region (coordinates)**: 좌표를 직접 입력하거나 **Select on screen...** 을 누른다. 화면이 멈춘 상태로 표시되면 원하는 영역을 마우스로 드래그한다. 드래그하는 동안 좌표와 크기가 표시된다.
   - **Game window**: 목록에서 창을 고르거나 **Pick on screen...** 을 누른 뒤 게임창을 클릭한다. 마우스를 올린 창이 강조 표시된다.
     - **Title contains** 에는 창 제목 전체가 들어간다. 제목 중 테이블 번호처럼 바뀌는 부분이 있으면 바뀌지 않는 부분만 남기면 된다.
3. **Test capture** 로 실제로 캡처될 화면과 캡처 시간을 확인한다.
4. **Save** 를 누르면 `config.ini`에 저장된다. 봇은 다음 캡처부터 새 설정을 사용한다.

화면에서 고르는 동안에는 봇의 창들이 잠시 숨겨졌다가 다시 나타난다. Esc를 누르거나 오른쪽 클릭하면 취소된다.

### 명령줄

저장소 루트에서 실행한다.

```bash
python -m poker.tools.capture_target --list
```

```bash
python -m poker.tools.capture_target --window "Hold'em" --grab capture.png
```

```bash
python -m poker.tools.capture_target --window "Hold'em" --save
```

```bash
python -m poker.tools.capture_target --region 0,0,1280,900 --save
```

```bash
python -m poker.tools.capture_target --full-screen --save
```

- `--list`: 보이는 창의 위치, 크기, 제목 목록
- `--grab FILE`: 한 번 캡처해 파일로 저장하고 크기, 화면 위치, 걸린 시간을 출력
- `--save`: 고른 설정을 `config.ini`에 저장. 아무 모드도 주지 않으면 현재 설정을 사용한다.

### config.ini

```ini
[main]
capture_mode = window
capture_region = 0,0,1280,900
capture_window_title = Hold'em
```

창 제목의 `$`는 config.ini에 `$$`로 저장된다(config.ini 문법상 `$`가 특수 문자이기 때문). 직접 편집할 때 `$`를 하나만 적어도 그대로 읽는다.

설정이 비어 있거나 잘못되면 경고를 남기고 전체 화면을 캡처한다. 지정한 창을 찾지 못할 때(창이 닫혔거나 최소화된 경우)도 전체 화면을 캡처하고 경고를 한 번 남긴다.

### 봇 없이 실행하기

봇 본체는 시작할 때 원격 서버가 필요하다. 서버 없이 기능만 확인할 때는 아래 명령을 쓴다.

```bash
python -m poker.gui.capture_target_dialog
```

```bash
python -m poker.tools.capture_check
```

- `capture_target_dialog`: 캡처 영역 설정 창만 띄운다.
- `capture_check`: 가짜 게임창(저장소의 테이블 스크린샷)을 띄우고 아래 항목을 자동으로 확인한다. 마우스는 움직이지 않는다.
  - 창 찾기
  - 픽셀 단위로 정확한 창 캡처
  - 실제 화면 캡처를 거친 인식과 마우스 기준 좌표
  - 창 이동
  - 창이 없을 때의 동작
  - 두 모드의 시간
- `capture_check --window-only 600`: 수동 점검용으로 가짜 게임창만 600초 동안 띄운다.

실제 PC에서 점검하는 절차는 [pc_test_prompt.md](pc_test_prompt.md)에 정리했다.

## 효과

캡처 범위가 작아지면 테이블 위치를 찾는 템플릿 매칭(`get_top_left_corner`)이 빨라진다. [측정 스크립트](../poker/tools/timing_benchmark.py)로 30회씩 측정했다. 측정 방법은 [timing_benchmark.md](timing_benchmark.md)와 같다.

| 단위 ms, 평균 | 전체 화면 (1920×1080) | 게임창만 (950×700) |
|---|---:|---:|
| 테이블 위치 찾기, PokerStars | 146 | 51 |
| 테이블 위치 찾기, PartyPoker | 147 | 48 |
| 화면 인식 전체, PokerStars | 448 | 379 |
| 화면 인식 전체, PartyPoker | 459 | 356 |

## 함께 수정한 문제

새 핸드를 감지하면 `get_new_hand`가 화면을 다시 캡처하는데, 다시 찍은 화면을 테이블 위치에 맞춰 잘라내지 않았다. 그래서 테이블이 캡처 이미지의 (0,0)에서 시작하지 않으면 이후 단계가 엉뚱한 영역을 읽었다. 그 결과 새 핸드 첫 사이클의 인식이 실패하고 한 번 더 돌았다.

화면 영역이나 게임창을 캡처하면 테이블이 캡처 이미지의 (0,0)에서 시작하지 않는 경우가 많다. 그래서 다시 찍은 화면을 이미 찾아 둔 테이블 위치로 잘라내도록 고쳤다(`crop_at_known_top_left_corner`).

## 바뀐 코드

| 파일 | 내용 |
|---|---|
| `poker/tools/capture_target.py` | 캡처 대상 설정 읽기·저장, 창 목록(Windows·macOS·Linux X11), 캡처, 명령줄 도구 |
| `poker/gui/capture_target_dialog.py` | 설정 창과 화면 위 선택 오버레이, 단독 실행 |
| `poker/tools/capture_check.py` | 실제 화면에서 가짜 게임창으로 하는 자동 점검 |
| `poker/tools/screen_operations.py` | `take_screenshot_with_offset`: 캡처 대상을 캡처하고 화면 위치를 함께 반환 |
| `poker/scraper/table_scraper.py` | 캡처 위치(`capture_offset`)를 저장하고, 마우스용 화면 좌표(`screen_tlc`)를 제공 |
| `poker/main.py`, `poker/scraper/table_screen_based.py` | 마우스 클릭에 `screen_tlc` 사용, 새 핸드에서 다시 찍은 화면을 잘라냄 |
| `poker/gui/ui/setup_form.ui`, `poker/gui/ui/table_setup_form.ui` | 캡처 영역 버튼 추가 |
| `poker/config_default.ini` | 기본값 `capture_mode = full_screen` |
| `poker/tests/test_capture_target.py` | 설정, 창 찾기, 캡처 위치, 캡처를 거친 전체 인식 테스트 |

## 확인한 내용

- **오프라인 테스트** (`poker/tests/test_capture_target.py`, 22개): 설정 읽기·저장(창 제목의 `$` 포함), 창 제목 일치 규칙, 캡처 위치, 창이 없을 때의 동작을 확인했다.
  - 테이블을 화면 (200,100)에 둔 상태에서 전체 화면 캡처와 게임창 캡처 모두 카드, 금액 등 인식값이 기대값과 같았다.
  - 마우스 기준 좌표가 (200,100)으로 나오는 것도 확인했다.
- **Linux 가상 디스플레이(Xvfb)에서 실제로 확인한 동작**
  - 다른 프로세스로 띄운 게임창을 목록에서 찾았고, 창 위치 (300,150)이 정확했다.
  - 창 캡처 결과가 원본 이미지와 픽셀 단위로 같았다(950×700, 약 26 ms).
  - 실제 화면 캡처로 인식을 돌렸을 때 인식값과 마우스 기준 좌표 (300,150)이 맞았다.
  - 설정 창에서 창 클릭 선택, 영역 드래그 선택, 미리보기, 저장이 동작했다.
- **확인하지 못한 환경**: Windows와 macOS에서는 실행해 보지 못했다.
  - Windows는 `EnumWindows`/`DwmGetWindowAttribute`로, macOS는 Quartz(`pyobjc-framework-Quartz` 필요)로 창 목록을 가져오도록 구현했다.

## 제한

- 화면 위에서 고르는 기능(**Select on screen**, **Pick on screen**)은 주 모니터만 보여준다. 다른 모니터에 있는 창은 목록에서 고르면 캡처된다(Windows).
- VirtualBox 모드(Settings의 가상 머신 선택)에는 적용되지 않는다. 가상 머신 화면은 지금처럼 전체를 받는다.
- macOS에서 다른 앱의 창 제목을 읽으려면 화면 기록 권한이 필요하다. 권한이 없으면 앱 이름만 보인다.
- 테이블 매핑(카드·버튼 템플릿과 좌표)은 여전히 Table Setup에서 만들고, 저장은 원격 서버를 거친다. 이 서버가 응답하지 않으면 새 게임의 매핑을 저장할 수 없다.
