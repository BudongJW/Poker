# 캡처 소스: OBS 분석과 적용

"실시간 인식" 화면의 소스를 OBS Studio의 캡처 소스에 맞춰 다시 만들면서, OBS가 소스를 어떻게 다루는지 소스 코드에서 확인한 내용과 그중 무엇을 가져왔는지를 정리합니다.

- 기준: `obsproject/obs-studio` master 브랜치 (2026-11-14에 받은 소스)
- 본 파일: `plugins/win-capture/window-capture.c`, `duplicator-monitor-capture.c`, `game-capture.c`, `plugins/linux-pipewire/screencast-portal.c`, `plugins/mac-capture/mac-sck-video-capture.m`, `plugins/obs-ffmpeg/obs-ffmpeg-source.c`, `plugins/obs-filters/crop-filter.c`, `scale-filter.c`, `plugins/image-source/image-source.c`, `obs-slideshow-mk2.c`

## 1. OBS의 소스: 종류 + 설정 + 필터

OBS의 소스는 종류(`id`), 설정(`obs_data`, JSON), 그 위에 걸리는 필터 목록으로 이루어집니다. 장면 모음 파일에는 이 설정만 저장되고, 캡처 중인 창 핸들이나 스트림은 저장되지 않습니다. 다음에 OBS를 열면 저장된 설정으로 대상을 다시 찾습니다. 그래서 대상이 사라져도 소스는 그대로 남고, 대상이 돌아오면 다시 붙습니다.

## 2. 소스 종류별로 본 것

| OBS 소스 | 무엇으로 대상을 기억하나 | 대상을 잃었을 때 | 그 밖의 설정 |
|---|---|---|---|
| 창 캡처 (Windows, `window-capture.c`) | `window` = "제목:클래스:실행 파일" 문자열, `priority`(제목·클래스·실행 파일 중 무엇을 먼저 맞출지) | 창 핸들이 무효가 되면 `unhooked` 신호를 내고, 1초(`WC_CHECK_TIMER`)마다 같은 조건으로 창을 다시 찾아 찾으면 `hooked` | `method`(자동·BitBlt·WGC), `cursor`(기본 켬), `client_area`(기본 켬: 제목 표시줄과 테두리 제외). 0.2초(`RESIZE_CHECK_TIME`)마다 창 크기를 보고 바뀌면 캡처를 새로 만듦. 최소화된 창은 건너뜀 |
| 디스플레이 캡처 (`duplicator-monitor-capture.c`) | `monitor_id` (모니터 장치 ID) | 3초(`RESET_INTERVAL_SEC`)마다 다시 시도 | `capture_cursor`, `method`(DXGI·WGC) |
| 화면 캡처 (Linux PipeWire, `screencast-portal.c`) | 창 목록을 OBS가 아니라 시스템(포털)이 보여 주고 사람이 고름. `persist_mode` 2로 받은 `RestoreToken`을 설정에 저장 | 다음 실행 때 토큰으로 묻지 않고 같은 대상에 다시 연결. "Reload" 버튼은 토큰을 지우고 다시 고르게 함 | `ShowCursor` → 포털의 `cursor_mode`(숨김·영상에 포함·메타데이터) |
| 화면 캡처 (macOS ScreenCaptureKit, `mac-sck-video-capture.m`) | `type`(디스플레이·창·앱)과 그에 맞는 ID | 목록을 다시 만들어 같은 ID를 찾음 | `show_cursor`, `hide_obs`(OBS 자신을 캡처에서 제외: `excludingApplications`) |
| 게임 캡처 (`game-capture.c`) | 대상 프로세스 | 다시 훅 | 대상 프로세스에 훅 DLL을 넣어 그래픽 버퍼를 직접 가져옴 |
| 미디어 소스 (`obs-ffmpeg-source.c`) | `local_file` 경로 | 파일이 없으면 빈 화면 | `looping`(기본 끔), `restart_on_activate`(기본 켬: 보이면 처음부터 재생), `speed_percent`(1~200), `clear_on_media_end`, `seekable`, 재생·일시정지·탐색 컨트롤 |
| 이미지, 슬라이드쇼 (`image-source.c`, `obs-slideshow-mk2.c`) | 파일 경로(들) | — | 넘김 간격, 반복, 수동·자동 넘김 |

필터 두 가지도 봤습니다.

| 필터 | 설정 | 특징 |
|---|---|---|
| 자르기/패딩 (`crop-filter.c`) | `relative`(기본 켬)일 때 `left`·`top`·`right`·`bottom` = 가장자리에서 잘라 낼 픽셀, 끄면 `x`·`y`·`cx`·`cy` 픽셀 | 픽셀 단위라서, 원본(창) 크기가 바뀌면 같은 숫자가 다른 부분을 가리킴. 장면에서 Alt+드래그로 자르는 것도 픽셀 |
| 크기 조정 (`scale-filter.c`) | `resolution`, `sampling`(point·bilinear·bicubic·lanczos·area) | 출력 해상도를 정해 두고 원본을 맞춤 |

## 3. 브라우저에서 같은 일을 할 때 다른 점

pokerlab의 화면은 웹 페이지이므로 `getDisplayMedia`로 화면을 받습니다. 이것은 Windows·macOS의 OBS보다 Linux PipeWire 쪽과 구조가 같습니다.

- **페이지는 창 목록을 볼 수 없습니다.** 브라우저가 선택 창을 보여 주고 사람이 고릅니다. 페이지가 할 수 있는 것은 선택 창이 어느 종류(창·전체 화면·탭)를 먼저 보여 줄지 정하는 것(`displaySurface`)까지입니다.
- **다시 고르기를 건너뛸 수 없습니다.** 브라우저에는 PipeWire의 `RestoreToken` 같은 것이 없어, 창이 닫히거나 페이지를 다시 열면 사람이 다시 골라야 합니다. 그래서 OBS처럼 설정만 저장하고, 소스는 남겨 둔 채 "다시 연결"을 기다립니다.
- **창 크기를 따로 물을 필요는 없습니다.** 받은 영상의 크기(`videoWidth`, `videoHeight`)가 곧 창 크기입니다.

## 4. 적용한 것

| OBS에서 | pokerlab에서 |
|---|---|
| 소스 종류: 창 캡처, 디스플레이 캡처, 미디어 소스 | 소스 종류: 창, 전체 화면, 브라우저 탭, 동영상 파일. 선택 창은 고른 종류의 목록부터 보여 주고, 사람이 다른 종류를 고르면 실제로 고른 종류(`getSettings().displaySurface`)로 바꿔 표시 |
| 장면 모음에 설정만 저장 | 소스 설정(종류, 이름, 메모, 간격, 영역, 일시정지, 재생 속도, 파일 이름)을 `localStorage`의 `pokerlab.capture.sources`에 저장. 스트림은 저장하지 않음 |
| `unhooked` → 1초마다 다시 찾기 → `hooked` | 스트림이 끝나면(0.4초마다 확인, `ended` 이벤트도 받음) 소스를 지우지 않고 "연결되어 있지 않습니다"로 표시. "다시 연결"로 고르면 같은 소스 ID로 이어 써서, 서버가 따라가던 핸드와 끝난 핸드 수가 그대로 이어짐. 페이지를 다시 열었을 때도 같음 |
| PipeWire의 "Reload" | "창 바꾸기": 연결된 소스의 대상을 다시 고름. Chrome에서는 공유 중에 대상을 바꾸는 것도 허용(`surfaceSwitching: "include"`) |
| 0.2초마다 크기 확인 | 0.4초마다 영상 크기를 보고 바뀌면 미리보기와 영역 표시를 다시 맞춤 |
| 자르기 필터, Alt+드래그 | "영역": 미리보기에서 끌어 고름. 변화 감지와 보내는 그림 모두 그 영역만 씀. 픽셀이 아니라 **비율**(0~1)로 저장: 포커 프로그램은 창 크기에 맞춰 테이블을 늘리고 줄이므로, 비율이어야 창 크기가 바뀌어도 같은 부분을 가리킴. 영역을 바꾸면 좌석 좌표가 달라지므로 따라가던 핸드를 새로 시작 |
| 크기 조정 필터 | 영역을 자른 뒤 긴 변을 1,600px 이하로 줄여 JPEG(품질 0.92)로 보냄. 자른 만큼 테이블이 같은 해상도 안에서 크게 보임 |
| `hide_obs` (자기 자신 제외) | 이 페이지의 탭은 선택 목록에서 뺌(`selfBrowserSurface: "exclude"`) |
| `cursor`, `ShowCursor` | 커서는 찍지 않음(`cursor: "never"`, 지원하는 브라우저에서만). OBS는 방송용이라 기본이 켬이지만, 여기서는 커서가 카드나 숫자를 가릴 뿐 |
| 소스 보이기/숨기기(눈 아이콘) | "일시정지": 소스와 미리보기는 두고 읽기만 멈춤 |
| 미디어 소스 `restart_on_activate`, `speed_percent`, `seekable` | 동영상 파일을 열면 바로 재생, 재생 속도 0.5~4배, 브라우저의 재생 컨트롤로 탐색 |

OBS에 없는 것도 두 가지 더했습니다. OBS는 모든 프레임을 그리면 되지만, 여기서는 한 프레임을 읽는 데 몇 초가 걸리고 호출마다 비용이 들기 때문입니다.

- **읽는 동안 멈춤** (동영상): 한 프레임을 읽는 동안 영상을 멈췄다가 끝나면 다시 재생합니다. 녹화를 복기할 때 그사이의 화면 변화를 놓치지 않습니다. 기록에는 핸드가 영상의 어디서 시작했는지(`at`)가 남습니다.
- **변화 감지를 칸 단위로**: 원래는 축소 화면(48×27) 전체의 평균 차이로 바뀜을 판단해서, 카드 한 장이 놓이는 것처럼 화면의 0.5%쯤 되는 변화는 문턱을 넘지 못했습니다. 이제는 마지막으로 읽은 화면과 비교해 눈에 띄게(밝기 14 넘게) 바뀐 칸이 하나라도 있으면 바뀐 것으로 보고, 0.4초 간격 사이에 바뀐 칸이 두 개 이하일 때 멈춘 것으로 봅니다. 차례 타이머처럼 한두 칸만 계속 움직이는 것은 읽기를 막지 않습니다.

## 5. 적용하지 않은 것

| OBS에서 | 이유 |
|---|---|
| 창을 제목·클래스·실행 파일로 자동으로 다시 찾기 | 웹 페이지는 창 목록을 볼 수 없어 불가능합니다. 설정을 남겨 두고 사람이 다시 고르는 것으로 대신했습니다 |
| 게임 캡처 (훅 주입) | 브라우저에서 할 수 없고, 다른 프로그램 안에 코드를 넣는 방식이라 이 프로젝트에서 다루지 않습니다 |
| 비디오 캡처 장치 (캡처 카드, 가상 카메라) | 프로그램 창은 화면 공유로, 녹화는 파일로 받을 수 있어서 이 경로가 더하는 것은 화면 공유를 막아 둔 프로그램을 읽는 것뿐입니다. 그런 프로그램은 읽히지 않기를 원하는 것이므로 넣지 않았습니다 |
| 이미지, 이미지 슬라이드쇼 | 스크린샷을 올려 읽는 입력은 바로 앞 변경에서 뺐으므로 다시 넣지 않았습니다 |
| 장면, 여러 소스 합성, 오디오 | 소스마다 다른 테이블을 따로 읽으므로 합칠 일이 없고, 소리는 읽지 않습니다 |

## 6. 확인한 방법

헤드리스 Chromium에서 선택 창 대신 캔버스 영상(1280×720)을, Gemini 대신 저장된 스냅샷 재생(`POKERLAB_VISION_REPLAY`)을 써서 다음을 확인했습니다.

- 창 소스 추가: 선택 창 옵션(`displaySurface: "window"`, `selfBrowserSurface: "exclude"`, `surfaceSwitching: "include"`), 첫 프레임이 1280×720 JPEG로 전송, 분석 표시
- 영역: 미리보기에서 끈 상자가 비율 `{x: 0.15, y: 0.2, w: 0.7, h: 0.7}`로 저장되고, 그 뒤로 보내는 그림이 896×504
- 일시정지 중 7초 동안 전송 없음
- 스트림이 끝나면 "연결되어 있지 않습니다"와 "다시 연결", 다시 연결한 뒤에도 영역 유지
- 페이지를 다시 연 뒤 이름·메모·영역이 그대로 복원되고, 서버가 따라가던 핸드가 표시되며, "다시 연결"로 이어짐
- 동영상 파일(VP9 WebM, 2초마다 바뀌는 16초 영상): 파일 이름이 소스 이름이 되고, 0:00·0:02·0:04·0:05에서 읽었으며, 기록에 `at`이 남음
- 소스 삭제가 저장된 설정에서도 빠짐, 페이지 오류 없음

`at` 기록은 `tests/test_pokerlab.py`의 `test_a_recording_keeps_where_each_hand_began`로 시험합니다. 실제 선택 창, 실제 Gemini 인식, Firefox·Edge는 확인하지 못했습니다.
