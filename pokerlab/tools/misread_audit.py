"""How well the follower keeps a hand when frames are misread. No reading service, no cost.

Hands the engine plays are turned into the frames a window would show, misreads are mixed in, and
the hand the follower puts together is compared with the one it puts together from the clean
frames.

    python pokerlab/tools/misread_audit.py                       # one kind of misread at a time, in 10% of frames
    python pokerlab/tools/misread_audit.py --stacks 60,150,300,1000
    python pokerlab/tools/misread_audit.py --mixed --rate 0.02   # every kind together, each in 2% of frames
"""

import argparse
import copy
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pokerlab import follow, table                      # noqa: E402
from pokerlab.bots import make_bot                      # noqa: E402
from pokerlab.cards import card_str                     # noqa: E402
from pokerlab.engine import play_hand                   # noqa: E402

KINDS = ["in_hand flicker", "bet misread", "stack misread", "pot misread", "hero card", "board card", "button moved",
         "hero mark moved", "frame missing"]


def hands(count, seed, stacks):
    rng = random.Random(seed)
    bots = [make_bot(kind, seed=i) for i, kind in enumerate(("tag", "maniac", "human", "pro", "lag", "nit"))]
    for i in range(count):
        chips = [rng.choice(stacks) for _ in range(6)]
        yield play_hand(bots, chips, i % 6, 5, 10, rng), chips


def actions(hand):
    """What a record must get right: the actions in order, and the board."""
    names = [int(name[1:]) for name in hand["names"]]
    return [(a[0], names[a[1]], a[2], round(a[3])) for a in hand["actions"]] + [tuple(hand["board"])]


def misread(shot, kind, rng):
    """One frame with one kind of misread (None: the frame is missed)."""
    shot = copy.deepcopy(shot)
    seats = shot["seats"]
    i = rng.randrange(len(seats))
    if kind == "in_hand flicker":
        seats[i]["in_hand"] = not seats[i]["in_hand"]
    elif kind == "bet misread":
        seats[i]["bet"] = (seats[i]["bet"] or 0) * 10 + 5
    elif kind == "stack misread":
        if seats[i]["stack"] is not None:
            seats[i]["stack"] = seats[i]["stack"] * 10 + 5
    elif kind == "pot misread":
        shot["pot"] = (shot["pot"] or 0) * 2 + 30
    elif kind == "hero card":
        used = {shot["hero"][j:j + 2] for j in (0, 2)} | {shot["board"][j:j + 2] for j in range(0, len(shot["board"]), 2)}
        shot["hero"] = shot["hero"][:2] + next(card_str(c) for c in rng.sample(range(52), 52) if card_str(c) not in used)
    elif kind == "board card":
        if shot["board"]:
            used = {shot["hero"][j:j + 2] for j in (0, 2)} | {shot["board"][j:j + 2] for j in range(0, len(shot["board"]), 2)}
            j = 2 * rng.randrange(len(shot["board"]) // 2)
            other = next(card_str(c) for c in rng.sample(range(52), 52) if card_str(c) not in used)
            shot["board"] = shot["board"][:j] + other + shot["board"][j + 2:]
    elif kind == "button moved":
        button = next(j for j, s in enumerate(seats) if s["button"])
        for j, s in enumerate(seats):
            s["button"] = j == (button + 1) % len(seats)
    elif kind == "hero mark moved":
        for j, s in enumerate(seats):
            s["hero"] = j == (i if i else 1)
        shot["hero"] = ""
    elif kind == "frame missing":
        return None
    return shot


def run(kinds, rate, stacks, count, turns=True, stacks_shown=True):
    rng = random.Random(1)
    exact = restarted = total = 0
    for result, chips in hands(count, 3, stacks):
        shots = table.frames(result, chips, 0, 5, every=True, to_end=True, turns=turns)
        if not stacks_shown:
            shots = [dict(s, seats=[dict(x, stack=None) for x in s["seats"]]) for s in shots]
        if len(shots) < 3:
            continue
        clean, noisy = follow.Follower("clean"), follow.Follower("noisy")
        for shot in shots:
            clean.push(shot)
        for k, shot in enumerate(shots):
            if 0 < k < len(shots) - 1:
                for kind in kinds:
                    if shot is not None and rng.random() < rate:
                        shot = misread(shot, kind, rng)
            if shot is not None:
                noisy.push(shot)
        total += 1
        exact += actions(noisy.hand) == actions(clean.hand)
        restarted += noisy.restarted
    return exact, total, restarted


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stacks", default="1000", help="starting stacks to draw from, in chips (big blind 10)")
    parser.add_argument("--hands", type=int, default=150)
    parser.add_argument("--rate", type=float, default=0.10, help="share of frames with a misread (of each kind)")
    parser.add_argument("--mixed", action="store_true", help="every kind of misread together")
    args = parser.parse_args()
    stacks = tuple(int(x) for x in args.stacks.split(","))
    groups = [("모든 종류", KINDS)] if args.mixed else [("없음", [])] + [(kind, [kind]) for kind in KINDS]
    for label, kinds in groups:
        exact, total, restarted = run(kinds, args.rate, stacks, args.hands)
        print(f"{label:<16} 오독 없는 기록과 같음 {exact}/{total} ({exact / total:.0%}), 처음부터 다시 읽음 {restarted}")


if __name__ == "__main__":
    main()
