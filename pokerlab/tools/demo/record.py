"""Film pokerlab reading a poker game as it is played, as a video for a phone held upright.

It serves the game's files, starts `pokerlab serve`, opens the game and the pokerlab page in a
browser (driver.js), feeds the game's picture to the page as the window it reads, plays (checks and
calls; in the HTML5 game a fold now and then), and makes OUT/demo.mp4 (compose.py, ffmpeg).

While it plays, a watchdog (watchdog.py) looks for a game that has stopped going on (a tip window,
a browser alert, a game that ended) and gets it going again; what happened is in OUT/incidents.json.
If it cannot, playing and reading stop there, and the video is made of what was filmed.

Reading is real: it costs about $0.2 a minute with Claude Opus 5.5 (a read every 5 seconds or so).

    python pokerlab/tools/demo/record.py html5 GAME_DIR OUT_DIR --minutes 4
    python pokerlab/tools/demo/record.py flash GAME_DIR OUT_DIR --minutes 4 --budget 1.5
    python pokerlab/tools/demo/record.py flash GAME_DIR OUT_DIR --minutes 6 --no-read     (the player alone, no cost)

GAME_DIR: the HTML5 game (js-css-poker: poker.html), or a folder with the Flash game as index.html
and its .swf run by Ruffle (see the tools README). Needs Node.js with Playwright in this folder
(npm install, npx playwright install chromium), ffmpeg, Pillow, and a reading key
(python -m pokerlab check).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIXTURES = os.path.join(HERE, "..", "fixtures")
sys.path.insert(0, HERE)
from watchdog import Watchdog, ask_claude  # noqa: E402

GAME_PORT, POKERLAB_PORT, DRIVER_PORT = 8781, 8790, 8795


def ask(path, **query):
    url = f"http://127.0.0.1:{DRIVER_PORT}{path}" + ("?" + urllib.parse.urlencode(query) if query else "")
    with urllib.request.urlopen(url, timeout=60) as reply:
        return reply.read().decode("utf-8")


def wait(ms):
    ask("/wait", ms=ms)


def ready(process, seconds=60):
    """Wait until the driver answers; stop with what went wrong if it ends or does not answer (a browser that does
    not start, a port in use)."""
    end = time.time() + seconds
    while time.time() < end:
        if process.poll() is not None:
            sys.exit(f"브라우저 드라이버가 시작하자마자 끝났습니다 (종료 코드 {process.returncode}). 위의 오류를 보세요. "
                     "Playwright의 Chromium이 설치돼 있는지(npx playwright install chromium), 포트가 비어 있는지 확인하세요")
        try:
            ask("/wait", ms=0)
            return
        except OSError:
            time.sleep(0.5)
    sys.exit(f"브라우저 드라이버가 {seconds}초 동안 답하지 않았습니다 (포트 {DRIVER_PORT})")


class Run:
    """What the player loop needs besides the game: the watchdog, the time and the money left."""

    def __init__(self, minutes, watchdog, budget=None, reading=True):
        self.end, self.watchdog, self.budget, self.reading = time.time() + minutes * 60, watchdog, budget, reading
        self.spent, self.stopped, self._checked = 0.0, None, 0.0

    def going(self):
        """Go on playing? False when the time is up, the watchdog gave up, or the reading has cost the budget."""
        if self.stopped:
            return False
        if time.time() >= self.end:
            return False
        for dialog in json.loads(ask("/dialogs") or "[]"):          # an alert box: closed already, noted here
            self.watchdog.incidents.append({"time": time.strftime("%H:%M:%S"), "tried": [],
                                            "solved_by": "브라우저 알림 창 닫음", "what": f"{dialog['type']}: {dialog['message'][:200]}"})
        if self.watchdog.check() == "stop":
            self.stopped = "게임이 멈춰 다시 움직이지 못함 (incidents.json)"
            return False
        if self.reading and self.budget is not None and time.time() - self._checked > 10:
            self._checked = time.time()
            found = re.search(r"약 \$([0-9.]+)", ask("/state"))
            self.spent = float(found.group(1)) if found else self.spent
            if self.spent >= self.budget:
                self.stopped = f"읽기 비용이 예산 {self.budget}달러에 닿음"
                self.watchdog.incidents.append({"time": time.strftime("%H:%M:%S"), "tried": [], "what": self.stopped})
                return False
        return True


def flash(run, notes):
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
    if run.reading:
        ask("/ui_open", label="Governor of Poker", notes=notes)
    ask("/click", x=438, y=316)                 # five players
    wait(400)
    ask("/start")
    ask("/click", x=522, y=649)                 # Play
    while run.going():
        ask("/click", x=725, y=693)             # CALL (CHECK when nobody bet)
        time.sleep(3)


def html5(run, notes):
    """js-css-poker: five opponents at the slowest speed; call on our turn, fold every fourth decision that faces a bet."""
    ask("/goto", url=f"http://127.0.0.1:{GAME_PORT}/poker.html")
    wait(1500)
    if run.reading:
        ask("/ui_open", label="JS Poker", notes=notes)
    ask("/start")
    ask("/click", x=628, y=86)                  # five opponents
    wait(1000)
    ask("/select", sel="#speed-options", value="0")
    turn = ("getComputedStyle(document.getElementById('call-button')).visibility==='visible'"
            "?document.getElementById('call-button').innerText:''")
    decisions = 0
    while run.going():
        label = json.loads(ask("/eval", expr=turn) or '""')
        if label and "new game" not in label.lower():   # (a game that ended is the watchdog's: see FIXES)
            decisions += 1
            time.sleep(0.8)
            fold = decisions % 4 == 0 and "call" in label.lower()
            try:
                ask("/clicksel", sel="#fold-button" if fold else "#call-button")
            except OSError:
                pass
        time.sleep(1.5)


def _flash_tips():
    """Governor of Poker's tip window: tick "Hide all tips", then "Close"."""
    ask("/click", x=943, y=553)
    wait(500)
    ask("/click", x=603, y=555)


def _html5_new_game():
    """js-css-poker after a game ended: close its message box (OK), press "Start a new game" (it is the fold
    button then), choose five opponents, the slowest speed."""
    ask("/eval", expr="(typeof leave_pseudo_alert === 'function' && leave_pseudo_alert(), 1)")   # the OK of its message box
    wait(500)
    for button in ("#fold-button", "#call-button"):
        label = json.loads(ask("/eval", expr=f"document.querySelector('{button}').innerText") or '""')
        if "new game" in label.lower():
            ask("/clicksel", sel=button)
            wait(1500)
            break
    ask("/click", x=628, y=86)                  # the number of opponents, if that box is up
    wait(1000)
    ask("/select", sel="#speed-options", value="0")


# the game, its notes, the video's subtitle, the part of the picture shown, and the fixes the watchdog knows for it
GAMES = {"flash": (flash, "flash_notes.txt", "Flash 포커 게임 창을 Claude Opus 5.5로 읽는 중", None,
                   {"팁 창 닫고 팁 끄기": _flash_tips}),
         "html5": (html5, "html5_notes.txt", "HTML5 포커 게임 창을 Claude Opus 5.5로 읽는 중", "0,0,860,520",
                   {"끝난 게임 다시 시작": _html5_new_game})}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", choices=sorted(GAMES))
    parser.add_argument("game_dir")
    parser.add_argument("out")
    parser.add_argument("--minutes", type=float, default=4)
    parser.add_argument("--budget", type=float, help="stop playing and reading once reading has cost this many dollars")
    parser.add_argument("--no-read", action="store_true", help="play only (no pokerlab, no cost, no video): to try the player and the watchdog")
    parser.add_argument("--no-ask", action="store_true", help="the watchdog does not ask Claude what is in the way")
    args = parser.parse_args()
    play, notes_file, subtitle, crop, fixes = GAMES[args.game]
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    notes = open(os.path.join(FIXTURES, notes_file), encoding="utf-8").read().strip()
    reading = not args.no_read
    started = []

    def act(kind, **what):
        if kind == "click":
            ask("/click", x=what["x"], y=what["y"])
        elif kind == "key":
            ask("/key", key=what["key"])
        elif kind == "fix":
            fixes[what["name"]]()

    watchdog = Watchdog(os.path.join(out, "frames"), out, act, fixes=list(fixes), ask=None if args.no_ask else ask_claude)
    run = Run(args.minutes, watchdog, budget=args.budget, reading=reading)
    try:
        started.append(subprocess.Popen([sys.executable, "-I", "-m", "http.server", str(GAME_PORT), "--directory",
                                         os.path.abspath(args.game_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if reading:
            started.append(subprocess.Popen([sys.executable, "-m", "pokerlab", "serve", "--port", str(POKERLAB_PORT),
                                             "--record", os.path.join(out, "hands.jsonl")], cwd=ROOT))
        driver = subprocess.Popen(["node", os.path.join(HERE, "driver.js"), str(DRIVER_PORT), out,
                                   f"http://127.0.0.1:{POKERLAB_PORT}/"], cwd=HERE)
        started.append(driver)
        ready(driver)
        play(run, notes)
        if reading:
            print("pokerlab:", ask("/state"))
        ask("/quit")
    finally:
        for process in started:
            process.terminate()
    print(f"돌발상황 {len(watchdog.incidents)}건" + (f", 멈춘 이유: {run.stopped}" if run.stopped else "") +
          (f" (자세히: {os.path.join(out, 'incidents.json')})" if watchdog.incidents else ""))
    for incident in watchdog.incidents:
        print(f"  {incident['time']}: " + (incident.get("solved_by") or ("해결 못함" if incident.get("gave_up") else incident.get("what", ""))))
    if not reading:
        return
    command = [sys.executable, os.path.join(HERE, "compose.py"), out, subtitle] + ([crop] if crop else [])
    subprocess.run(command, check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(out, "compose.txt"),
                    "-vf", "fps=10,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                    "-movflags", "+faststart", os.path.join(out, "demo.mp4")], check=True)
    print("영상:", os.path.join(out, "demo.mp4"), "기록:", os.path.join(out, "hands.jsonl"))


if __name__ == "__main__":
    main()
