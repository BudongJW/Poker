"""Learning from the hands followed in a window.

``pokerlab serve --record hands.jsonl`` appends one line per hand that a window showed: seats,
button, the action log worked out from the chips, the hero's cards and any cards turned up at a
showdown (:mod:`pokerlab.follow`). :func:`learn` turns those lines into an action model that the
live tab can read opponents with: the counts of a base model (by default ``online``, real-money
players), scaled down to the weight of ``prior`` hands, plus what the players of each program did.

Each program (the source's name in the record) gets a profile of its own, so the live tab reads
the players of a program with that program's habits. A program with few hands stays close to the
base model; the more hands it has, the more its own play counts. A program with no hands of its
own is read exactly as the base model reads it: the pooled profile of the live model is the base
model's, untouched by any program's records.

Only hands that were followed from the start are used: the follower did not have to start again
from a single frame (a frame that did not fit, perhaps misread), and the action log begins
before the flop. The hero's own actions are left out: the model is for reading opponents (a table
watched from outside has no hero, and everyone at it is learned from). Cards that a program shows
all along are used like cards shown at a showdown, so such programs teach the most. Hands
of a source that was never given a name are left out too: an unnamed source could be any
program, and its hands would blur the profile of whichever program it was taken for.
"""

import json

import numpy as np

from . import inference
from .cards import parse_cards

BASE = "online"
PRIOR_HANDS = 500           # the weight of the base model, in hands of the program's own play


def read(path):
    """The lines of a record file (lines that cannot be read are skipped)."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                found = json.loads(line)
            except ValueError:
                continue
            if isinstance(found, dict) and found.get("actions") is not None:
                out.append(found)
    return out


def to_hand(record, label=None):
    """A record line -> a hand as :func:`pokerlab.inference.prepare` takes it, or None if it cannot be learned from."""
    actions = [tuple(action) for action in record.get("actions") or []]
    if record.get("named") is False or record.get("restarted") or record.get("absent") or not actions or actions[0][0] != 0:
        return None
    try:
        n, button = int(record["n_players"]), int(record["button"])
        hero_seat = None if record.get("hero_seat") is None else int(record["hero_seat"])       # None: watched from outside
        board, hero = parse_cards(record.get("board") or ""), parse_cards(record.get("hero") or "")
        holes = {hero_seat: tuple(hero)} if len(hero) == 2 else {}
        for seat, cards in (record.get("shown") or {}).items():
            cards = parse_cards(cards)
            if len(cards) == 2:
                holes[int(seat)] = tuple(cards)
        observed = list(inference.observed(actions, n, button))
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    if any(code == inference.AGGRESSIVE or ctx[2] == inference.ANY_PRICE for _, _, ctx, code in observed):
        return None                         # a bet with no pot to measure it against: the pot was not read
    label = label or record.get("source") or "window"
    return {"board": board, "n_players": n, "button": button, "holes": holes, "actions": actions,
            "bots": {seat: None if seat == hero_seat else label for seat in range(n)}}


def adapt(base, prepared, prior_hands=PRIOR_HANDS, iterations=4, view="unsized"):
    """A model for the players in ``prepared``: a profile per label (program), each the base's pooled
    counts scaled to ``prior_hands`` hands plus that program's own counts.

    The hidden cards are filled in by EM as in :func:`pokerlab.inference.fit_prepared`, but each
    step starts from the base counts, so a little data moves the model only a little, and each
    program is fitted on its own, so one program's habits never colour how another's hidden hands
    are judged. 'all' is the base's pooled profile as it is.
    """
    groups = {}
    for players, buckets in prepared:
        for label in {label for label, _, _ in players}:
            groups.setdefault(label, []).append(([p for p in players if p[0] == label], buckets))
    pooled = base.counts["all"]
    profiles = {}
    for label, hands in sorted(groups.items()):
        acted = sum(len(acts) for players, _ in hands for _, _, acts in players) / len(hands)
        prior = pooled * (prior_hands * acted / max(pooled.sum(), 1e-9))
        model = inference.ActionModel({"all": prior}, base.source)
        for _ in range(iterations):
            counts, _, _ = inference.expected_counts(hands, model, view)
            model = inference.ActionModel({"all": prior + counts.get(label, np.zeros_like(prior))}, base.source)
        profiles[label] = model.counts["all"]
    return inference.ActionModel({"all": pooled.copy(), **profiles}, base.source)


def learn(records, out=None, base=BASE, prior_hands=PRIOR_HANDS, flop_runouts=150, processes=None):
    """Record lines (or a record file) -> the live model, saved to ``out`` (by default where the
    model named 'live' is read from) -> a summary dict."""
    out = out or inference.MODELS["live"]
    records = read(records) if isinstance(records, str) else list(records)
    hands = [hand for hand in (to_hand(record) for record in records) if hand is not None]
    summary = {"records": len(records), "hands": len(hands), "out": out,
               "programs": sorted({label for hand in hands for label in hand["bots"].values() if label}),
               "shown": sum(sum(1 for seat in hand["holes"] if hand["bots"].get(seat)) for hand in hands)}   # others' cards
    if not hands:
        return summary
    model = adapt(inference.load_model(base), inference.prepare(hands, flop_runouts, processes=processes), prior_hands)
    model.source = f"{base} with {len(hands)} hands followed in windows ({', '.join(summary['programs'])})"[:200]
    model.save(out)
    inference.load_model.cache_clear()      # whoever reads 'live' from now on gets this one
    return summary
