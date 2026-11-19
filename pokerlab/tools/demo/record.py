"""Film pokerlab reading a poker game as it is played, as a video for a phone held upright.

It serves the game's files, starts `pokerlab serve`, opens the game and the pokerlab page in a
browser (driver.js), feeds the game's picture to the page as the window it reads, plays (checks and
calls; in the HTML5 game a fold now and then), and makes OUT/demo.mp4 (compose.py, ffmpeg).

Reading is real: it costs about $0.2 a minute with Claude Opus 5.5 (a read every 5 seconds or so).

    python pokerlab/tools/demo/record.py html5 GAME_DIR OUT_DIR --minutes 4
    python pokerlab/tools/demo/record.py flash GAME_DIR OUT_DIR --minutes 4

GAME_DIR: the HTML5 game (js-css-poker: poker.html), or a folder with the Flash game as index.html
and its .swf run by Ruffle (see the tools README). Needs Node.js with Playwright in this folder
(npm install, npx playwright install chromium), ffmpeg, Pillow, and a reading key
(python -m pokerlab check).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIXTURES = os.path.join(HERE, "..", "fixtures")
GAME_PORT, POKERLAB_PORT, DRIVER_PORT = 8781, 8790, 8795


def ask(path, **query):
    url = f"http://127.0.0.1:{DRIVER_PORT}{path}" + ("?" + urllib.parse.urlencode(query) if query else "")
    with urllib.request.urlopen(url, timeout=60) as reply:
        return reply.read().decode("utf-8")


def wait(ms):
    ask("/wait", ms=ms)


def flash(minutes, notes):
    """Governor of Poker (a Flash game) in Ruffle: through the menus to a five-player table, then call on every turn."""
    ask("/goto", url=f"http://127.0.0.1:{GAME_PORT}/index.html")
    wait(8000)
    ask("/click", x=1223, y=22)                 # Ruffle's notice about hardware acceleration
    wait(6000)
    ask("/click", x=643, y=285)                 # name
    ask("/type", text="Tester")
    ask("/click", x=738, y=465)                 # character
    wait(500)
    ask("/click", x=636, y=576)                 # OK
    wait(5000)
    ask("/click", x=527, y=542)                 # Quick Poker
    wait(3000)
    ask("/ui_open", label="Governor of Poker", notes=notes)
    ask("/click", x=438, y=316)                 # five players
    wait(400)
    ask("/start")
    ask("/click", x=522, y=649)                 # Play
    end = time.time() + minutes * 60
    while time.time() < end:
        ask("/click", x=725, y=693)             # CALL (CHECK when nobody bet)
        time.sleep(3)


def html5(minutes, notes):
    """js-css-poker: five opponents at the slowest speed; call on our turn, fold every fourth decision that faces a bet."""
    ask("/goto", url=f"http://127.0.0.1:{GAME_PORT}/poker.html")
    wait(1500)
    ask("/ui_open", label="JS Poker", notes=notes)
    ask("/start")
    ask("/click", x=628, y=86)                  # five opponents
    wait(1000)
    ask("/select", sel="#speed-options", value="0")
    turn = ("getComputedStyle(document.getElementById('call-button')).visibility==='visible'"
            "?document.getElementById('call-button').innerText:''")
    decisions, end = 0, time.time() + minutes * 60
    while time.time() < end:
        label = json.loads(ask("/eval", expr=turn) or '""')
        if label:
            decisions += 1
            time.sleep(0.8)
            fold = decisions % 4 == 0 and "call" in label.lower()
            try:
                ask("/clicksel", sel="#fold-button" if fold else "#call-button")
            except OSError:
                pass
        time.sleep(1.5)


GAMES = {"flash": (flash, "flash_notes.txt", "Flash 포커 게임 창을 Claude Opus 5.5로 읽는 중", None),
         "html5": (html5, "html5_notes.txt", "HTML5 포커 게임 창을 Claude Opus 5.5로 읽는 중", "0,0,860,520")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", choices=sorted(GAMES))
    parser.add_argument("game_dir")
    parser.add_argument("out")
    parser.add_argument("--minutes", type=float, default=4)
    args = parser.parse_args()
    play, notes_file, subtitle, crop = GAMES[args.game]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    notes = open(os.path.join(FIXTURES, notes_file), encoding="utf-8").read().strip()
    started = []
    try:
        started.append(subprocess.Popen([sys.executable, "-I", "-m", "http.server", str(GAME_PORT), "--directory",
                                         os.path.abspath(args.game_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        started.append(subprocess.Popen([sys.executable, "-m", "pokerlab", "serve", "--port", str(POKERLAB_PORT),
                                         "--record", os.path.join(out, "hands.jsonl")], cwd=ROOT))
        started.append(subprocess.Popen(["node", os.path.join(HERE, "driver.js"), str(DRIVER_PORT), out,
                                         f"http://127.0.0.1:{POKERLAB_PORT}/"], cwd=HERE))
        time.sleep(4)
        play(args.minutes, notes)
        print("pokerlab:", ask("/state"))
        ask("/quit")
    finally:
        for process in started:
            process.terminate()
    command = [sys.executable, os.path.join(HERE, "compose.py"), out, subtitle] + ([crop] if crop else [])
    subprocess.run(command, check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(out, "compose.txt"),
                    "-vf", "fps=10,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-movflags", "+faststart", os.path.join(out, "demo.mp4")], check=True)
    print("영상:", os.path.join(out, "demo.mp4"), "기록:", os.path.join(out, "hands.jsonl"))


if __name__ == "__main__":
    main()
