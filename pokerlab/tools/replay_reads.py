"""Push saved reads (a reads file from read_frames.py, or a fixture) through the follower, frame by frame. No cost.

    python pokerlab/tools/replay_reads.py pokerlab/tools/fixtures/flash_reads.json
    python pokerlab/tools/replay_reads.py pokerlab/tools/fixtures/html5_reads.json --record out.jsonl
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pokerlab import follow                             # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reads", help="a JSON file {'rows': [{'k': frame number, 'read': snapshot}, ...]}")
    parser.add_argument("--record", help="where to write the followed hands (default: a temporary file)")
    parser.add_argument("--name", default="game")
    args = parser.parse_args()
    record = args.record or os.path.join(tempfile.mkdtemp(), "hands.jsonl")
    if os.path.exists(record):
        os.remove(record)
    follower = follow.Follower(args.name, record)
    for row in json.load(open(args.reads, encoding="utf-8"))["rows"]:
        if "read" not in row:
            print(row.get("k"), "읽지 못한 화면")
            continue
        follower.push(row["read"])
        state = "다시 읽음: " + follower.note if follower.note else follower.error or ""
        print(row["k"], f"끝난 핸드 {follower.counts['hands']}", state[:80])
    follower.close()
    print("품질:", follower.quality())
    for line in open(record, encoding="utf-8"):
        hand = json.loads(line)
        names = hand["names"]
        print(hand["hero"], hand["board"], "(처음부터 다시 읽음)" if hand.get("restarted") else "",
              "추정:", hand.get("guessed"), "공개된 카드:", hand.get("shown"))
        print("   ", [(a[0], "나" if a[1] == hand["hero_seat"] else (names[a[1]] or f"자리 {a[1] + 1}")[:8], a[2], a[3])
                      for a in hand["actions"]])


if __name__ == "__main__":
    main()
