"""Card encoding shared by the whole package.

A card is an int in 0..51: ``rank * 4 + suit`` with rank 0 = deuce .. 12 = ace
and suit 0..3 = clubs, diamonds, hearts, spades.
"""

import re

RANKS = "23456789TJQKA"
SUITS = "cdhs"

_CARD_RE = re.compile(r"(10|[2-9TJQKA])([CDHS])", re.IGNORECASE)
_SUIT_SYMBOLS = str.maketrans("♣♧♦♢♥♡♠♤", "ccddhhss")


def card(text):
    """'As' / 'AS' / '10h' / 'A♠' -> card int."""
    m = _CARD_RE.fullmatch(text.strip().translate(_SUIT_SYMBOLS))
    if not m:
        raise ValueError(f"not a card: {text!r}")
    rank = "T" if m.group(1) == "10" else m.group(1).upper()
    return RANKS.index(rank) * 4 + SUITS.index(m.group(2).lower())


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def parse_cards(cards):
    """Accept 'AsKd', 'As Kd', 'As,Kd', 'A♠K♦', ['AS', 'KD'] or card ints; reject duplicates."""
    if isinstance(cards, str):
        cards = cards.translate(_SUIT_SYMBOLS)
        found = _CARD_RE.findall(cards)
        leftover = _CARD_RE.sub("", cards).strip(" ,/|-")
        if leftover:
            raise ValueError(f"cannot parse cards: {cards!r}")
        out = [card(r + s) for r, s in found]
    else:
        out = [c if isinstance(c, int) else card(c) for c in cards]
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate card in {cards!r}")
    return out


def cards_str(cards):
    return " ".join(card_str(c) for c in cards)


def hand_class(c1, c2):
    """Two hole cards -> one of the 169 starting-hand classes ('AA', 'AKs', 'T9o')."""
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if (c1 & 3) == (c2 & 3) else "o")


def class_combos(cls):
    """'AKs' -> its 4 concrete combos, 'AKo' -> 12, 'AA' -> 6. Every combo is (higher card int, lower)."""
    hi, lo = RANKS.index(cls[0].upper()), RANKS.index(cls[1].upper())
    if hi == lo:
        return [(hi * 4 + a, hi * 4 + b) for a in range(4) for b in range(a)]
    kind = cls[2].lower() if len(cls) > 2 else ""
    out = []
    for a in range(4):
        for b in range(4):
            if (kind == "s" and a == b) or (kind == "o" and a != b) or kind == "":
                out.append((hi * 4 + a, lo * 4 + b))
    return out


# The 169 classes, strongest rank first: AA, AKs, AKo, AQs, ... 22
ALL_CLASSES = []
for _hi in range(12, -1, -1):
    for _lo in range(_hi, -1, -1):
        if _hi == _lo:
            ALL_CLASSES.append(RANKS[_hi] * 2)
        else:
            ALL_CLASSES.append(RANKS[_hi] + RANKS[_lo] + "s")
            ALL_CLASSES.append(RANKS[_hi] + RANKS[_lo] + "o")
