"""Read a folder of saved frames (fNNN.png) with the configured reading service and save what was read.

This calls the service for every frame and costs money: about $0.018 a frame with Claude Opus 5.5.

    python pokerlab/tools/read_frames.py FRAMES_DIR OUT.json [--notes NOTES.txt] [--only 5,6,7] [--jpeg]
"""

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pokerlab import vision                             # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames")
    parser.add_argument("out")
    parser.add_argument("--notes", help="a text file with the source notes for this program")
    parser.add_argument("--only", help="frame numbers to read, e.g. 5,6,7")
    parser.add_argument("--jpeg", action="store_true", help="send frames as JPEG, as the page does")
    args = parser.parse_args()
    notes = open(args.notes, encoding="utf-8").read().strip() if args.notes else None
    files = sorted(f for f in os.listdir(args.frames) if f.startswith("f") and f.endswith(".png"))
    if args.only:
        keep = {int(k) for k in args.only.split(",")}
        files = [f for f in files if int(f[1:-4]) in keep]
    rows = []
    for name in files:
        k = int(name[1:-4])
        data, mime = open(os.path.join(args.frames, name), "rb").read(), "image/png"
        if args.jpeg:
            from PIL import Image
            buffer = io.BytesIO()
            Image.open(io.BytesIO(data)).convert("RGB").save(buffer, "JPEG", quality=92)
            data, mime = buffer.getvalue(), "image/jpeg"
        started, call = time.time(), {}
        try:
            got = vision.recognize(data, mime, notes=notes, call=call)
            rows.append({"k": k, "read": got, "call": call})
            print(k, f"{time.time() - started:.1f}초", "내 카드", got["hero"], "보드", got["board"], "팟", got["pot"], flush=True)
        except vision.VisionError as error:
            rows.append({"k": k, "error": str(error)})
            print(k, "오류", error, flush=True)
        json.dump({"rows": rows}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
