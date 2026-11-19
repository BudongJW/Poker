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

메뉴 클릭 좌표는 1280x800 화면 기준입니다. 게임 버전이 다르면 `record.py`의 좌표를 고쳐야 합니다. Flash 게임은 그래픽 가속 없이 돌리면 느려서, 화면을 0.7초보다 자주 찍으면 멈출 수 있습니다.
