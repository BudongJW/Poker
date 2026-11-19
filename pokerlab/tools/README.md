# pokerlab 시험 도구

개발하면서 쓰는 도구입니다. 패키지에는 들어가지 않고, 저장소 루트에서 실행합니다. Windows에서는 `python` 대신 `py`를 씁니다.

## 오독 측정 (비용 없음)

엔진이 친 핸드를 창 화면처럼 바꾸고 오독을 섞어 넣은 뒤, 추적기가 만든 기록이 오독 없는 기록과 같은지 셉니다. 추적기(`follow.py`, `table.py`)를 고친 뒤에는 이것으로 나빠지지 않았는지 확인합니다.

```bash
python pokerlab/tools/misread_audit.py
```

```bash
python pokerlab/tools/misread_audit.py --stacks 60,150,300,1000
```

```bash
python pokerlab/tools/misread_audit.py --mixed --rate 0.02
```

기록의 액션과 보드가 오독 없는 기록과 같은 핸드의 비율을 셉니다. 마지막으로 잰 값(100핸드, 오독 10%): 앞에 놓인 칩 99%, 폴드 여부 96~97%, 스택 98%, 보드 카드 100%, 나머지 99~100%. 모든 종류를 2%씩 섞으면 97%입니다.

## 저장된 읽기 다시 돌리기 (비용 없음)

`fixtures/`에는 외부 게임 두 개의 화면을 Claude Opus 5.5로 읽은 결과가 들어 있습니다. 추적기를 고친 뒤 실제 읽기로 결과가 어떻게 바뀌는지 봅니다.

```bash
python pokerlab/tools/replay_reads.py pokerlab/tools/fixtures/flash_reads.json
```

```bash
python pokerlab/tools/replay_reads.py pokerlab/tools/fixtures/html5_reads.json
```

| 파일 | 내용 |
|---|---|
| `flash_reads.json` | Flash 게임(Governor of Poker) 45장, 소스 메모와 함께 읽음. 2핸드, 화면과 대조해 맞는 것을 확인함 |
| `html5_reads.json` | HTML5 게임(js-css-poker) 24장, 메모 없이 읽음 (지시문이 바뀌기 전) |
| `html5_states.json` | 같은 HTML5 화면들을 찍을 때 게임 내부 상태 (정답 대조용) |
| `flash_notes.txt`, `html5_notes.txt` | 두 게임에 맞게 고친 소스 메모 |

## 화면 읽기 (비용 있음)

저장된 화면 폴더(`f000.png`, `f001.png` ...)를 지금 설정된 인식 서비스로 읽어 저장합니다. Claude Opus 5.5로 한 장에 약 0.018달러입니다.

```bash
python pokerlab/tools/read_frames.py FRAMES_DIR out.json --notes pokerlab/tools/fixtures/flash_notes.txt --jpeg
```

## 플레이 영상 찍기 (비용 있음)

게임을 플레이하면서 pokerlab이 그 창을 읽는 모습을 휴대전화 세로 영상(1080x1920)으로 만듭니다. 읽기는 실제 호출이라 1분에 약 0.2달러가 듭니다. `POKERLAB_VISION_REPLAY=저장된값.json`을 주면 호출 없이 연결만 시험할 수 있습니다.

준비물:
- Node.js와 Playwright: `pokerlab/tools/demo` 폴더에서 `npm install` 후 `npx playwright install chromium`
- ffmpeg, Pillow(`pip install pillow`)
- 게임 파일 (저장소에 넣지 않음)
  - HTML5: js-css-poker (sourceforge의 js-css-poker 프로젝트). 압축을 풀어 `poker.html`이 있는 폴더를 GAME_DIR로 줍니다.
  - Flash: Governor of Poker 데모 `.swf`와 Ruffle 웹 버전(ruffle.rs). `index.html`이 Ruffle로 그 `.swf`를 1280x800으로 띄우는 폴더를 GAME_DIR로 줍니다.

```bash
python pokerlab/tools/demo/record.py html5 GAME_DIR out_html5 --minutes 4
```

```bash
python pokerlab/tools/demo/record.py flash GAME_DIR out_flash --minutes 4
```

결과: `OUT/demo.mp4`(영상), `OUT/hands.jsonl`(따라간 핸드 기록), HTML5는 `OUT/truth.json`(게임 내부 상태, 기록과 대조용).

플레이하는 동안 감시 장치(`demo/watchdog.py`)가 게임이 15초 넘게 멈추면 알려진 처리, Esc와 Enter, Claude에게 묻기 순서로 다시 움직이게 하고, 안 되면 플레이와 읽기를 멈춥니다. 있었던 일은 `OUT/incidents.json`과 `incident_N.jpg`에 남습니다(자세히는 `../INCIDENTS.md`). 옵션: `--budget 달러`(읽기 비용이 닿으면 멈춤), `--no-read`(pokerlab 없이 플레이와 감시만, 비용 없음), `--no-ask`(Claude에게 묻지 않음).

메뉴 클릭 좌표는 1280x800 화면 기준입니다. 게임 버전이 다르면 `record.py`의 좌표를 고쳐야 합니다. Flash 게임은 그래픽 가속 없이 돌리면 느려서, 화면을 0.7초보다 자주 찍으면 멈출 수 있습니다.

## 실제 창 캡처로 시험하기 (비용 있음)

`record.py`는 게임 화면을 캔버스로 넣어 주지만, 이 도구는 페이지의 "창 추가"가 브라우저의 실제 창 캡처(getDisplayMedia)를 부르고 Chromium이 게임 탭을 제목으로 고르게 합니다. 사람이 창을 고르는 것과 같은 경로로 화면이 들어가고, 처음에는 "정확하게", 나중에는 "빠르게"로 읽습니다. 모든 요청의 시간과 답, 게임 내부 상태, 상태 줄을 저장합니다.

```bash
python -m pokerlab serve --port 8790 --record out_capture/hands.jsonl
```

```bash
python -I -m http.server 8781 --directory GAME_DIR
```

```bash
cd pokerlab/tools/demo && xvfb-run -a node capture_check.js ../../../out_capture ../fixtures/html5_notes.txt 2 1.5
```

화면이 있는 Windows, macOS에서는 `xvfb-run -a` 없이 실행합니다. Chromium의 화면 없는 모드(headless)는 창 캡처를 지원하지 않습니다. 2026-11-19에 3.5분 돌린 결과: 48장, 정확하게 중앙값 5.24초, 빠르게 2.29초, 약 1.64달러. 기록된 3핸드 모두 보드와 각자 낸 칩이 게임과 같았습니다.

