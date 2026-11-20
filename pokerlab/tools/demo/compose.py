"""Game pictures and pokerlab page pictures (from driver.js) -> a 1080x1920 video for a phone held upright.

    python compose.py OUT_DIR "subtitle" [x0,y0,x1,y1 crop of the game picture]
Writes OUT_DIR/compose/*.jpg and OUT_DIR/compose.txt; record.py turns them into OUT_DIR/demo.mp4 with ffmpeg.
Incidents the watchdog met (OUT_DIR/incidents.json) are shown over the game: while the game stood still, and
for a while after, with what got it going again.
"""
import bisect, datetime, json, os, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watchdog import outcome  # noqa: E402
S = sys.argv[1]
SUB = sys.argv[2] if len(sys.argv) > 2 else "Flash 포커 게임 창을 Claude Opus 5.5로 읽는 중"
CROP = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
W, H = 1080, 1920
FONTS = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
         "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
FONT = next((f for f in FONTS if os.path.exists(f)), None)
if FONT is None:
    sys.exit("한글 글꼴을 찾지 못했습니다. compose.py의 FONTS에 글꼴 파일 경로를 넣어 주세요")
title_font, sub_font, small_font = ImageFont.truetype(FONT, 46), ImageFont.truetype(FONT, 28), ImageFont.truetype(FONT, 26)
def stamped(d):
    files = sorted(f for f in os.listdir(f"{S}/{d}") if "_" in f)
    return [float(f.split("_")[1].rsplit(".", 1)[0]) for f in files], [f"{S}/{d}/{f}" for f in files]
gt, gf = stamped("frames"); st, sf = stamped("state"); rt, rf = stamped("result")
start = gt[0]
os.makedirs(f"{S}/compose", exist_ok=True)


def clock(seconds_path):            # seconds since midnight, by the clock that wrote the file
    m = datetime.datetime.fromtimestamp(os.path.getmtime(seconds_path))
    return m.hour * 3600 + m.minute * 60 + m.second + m.microsecond / 1e6


incidents = []                      # (from, noticed, until, incident) in seconds since midnight
if os.path.exists(f"{S}/incidents.json"):
    for incident in json.load(open(f"{S}/incidents.json", encoding="utf-8")):
        hh, mm, ss = (int(x) for x in incident["time"].split(":"))
        noticed = hh * 3600 + mm * 60 + ss
        incidents.append((noticed - float(incident.get("still_seconds") or 0), noticed, noticed + 12, incident))


def banner(at):
    """The words over the game for an incident at this moment, or None."""
    for since, noticed, until, incident in incidents:
        if since <= at <= until:
            if at < noticed:
                return f"돌발상황: 게임이 멈춤 ({int(at - since)}초째), 감시 장치가 보고 있음"
            return f"돌발상황 처리: {outcome(incident)}"
    return None


def fit(draw, text, font, width):
    while draw.textlength(text, font=font) > width and len(text) > 4:
        text = text[:-2]
    return text
lines = []
cache = {}
def load(path):
    if path not in cache:
        cache.clear(); cache[path] = Image.open(path).convert("RGB")
    return cache[path]
for k, (t, path) in enumerate(zip(gt, gf)):
    frame = Image.new("RGB", (W, H), (245, 245, 242))
    draw = ImageDraw.Draw(frame)
    draw.rectangle([0, 0, W, 140], fill=(20, 24, 31))
    draw.text((40, 22), "pokerlab 실시간 인식", font=title_font, fill=(255, 255, 255))
    draw.text((40, 88), SUB, font=sub_font, fill=(170, 178, 190))
    elapsed = int(t - start)
    draw.text((W - 150, 92), f"{elapsed // 60}:{elapsed % 60:02d}", font=sub_font, fill=(170, 178, 190))
    game = Image.open(path).convert("RGB")
    if CROP:
        game = game.crop(CROP)
    game = game.resize((W, int(game.height * W / game.width)), Image.LANCZOS)
    frame.paste(game, (0, 140))
    words = banner(clock(path)) if incidents else None
    if words:
        draw.rectangle([0, 140, W, 200], fill=(196, 72, 32))
        draw.text((24, 154), fit(draw, words, sub_font, W - 48), font=sub_font, fill=(255, 255, 255))
    gy = 140 + game.height
    draw.text((30, gy + 13), "게임 화면을 읽은 pokerlab", font=small_font, fill=(90, 90, 90))
    i = max(0, bisect.bisect_right(st, t) - 1)
    state = Image.open(sf[i]).convert("RGB")
    scale = min(1020 / state.width, 2.0)
    state = state.resize((int(state.width * scale), int(state.height * scale)), Image.LANCZOS)
    frame.paste(state, (30, gy + 53))
    j = max(0, bisect.bisect_right(rt, t) - 1)
    result = Image.open(rf[j]).convert("RGB")
    scale = 1050 / result.width
    result = result.resize((1050, int(result.height * scale)), Image.LANCZOS)
    top = gy + 53 + state.height + 16
    frame.paste(result.crop((0, 0, 1050, min(result.height, H - top))), (15, top))
    out = f"{S}/compose/{k:05d}.jpg"
    frame.save(out, quality=90)
    lines.append((out, t))
with open(f"{S}/compose.txt", "w") as fh:
    for (out, t), nxt in zip(lines, [t for _, t in lines[1:]] + [lines[-1][1] + 1.0]):
        fh.write(f"file '{out}'\nduration {nxt - t:.3f}\n")
    fh.write(f"file '{lines[-1][0]}'\n")
print(len(lines), "frames,", round(gt[-1] - gt[0]), "seconds")
