"""No-limit Texas Hold'em engine: one call to :func:`play_hand` plays one complete hand.

Chips are ints. Seats are 0..n-1 clockwise. Bots implement ``act(view)`` and return
``(action, amount)`` or ``(action, amount, info)``; ``amount`` is only read for a raise and
means "raise to this total bet on the current street". Illegal answers are coerced to the
nearest legal one (folding with nothing to call becomes a check, a raise that is not allowed
becomes a call, raise sizes are clamped), so a buggy bot cannot corrupt a hand.
"""

from .evaluator import evaluate

FOLD, CHECK, CALL, RAISE = "fold", "check", "call", "raise"     # RAISE also covers an opening bet
STREETS = ("preflop", "flop", "turn", "river")
_BOARD_CARDS = (0, 3, 4, 5)


class View:
    """What a bot may know when it has to act."""

    __slots__ = ("seat", "hole", "board", "street", "pot", "to_call", "current_bet", "min_raise_to",
                 "max_raise_to", "can_raise", "stack", "behind", "bet", "big_blind", "n_players", "n_live",
                 "button", "position", "raises", "actions")


class HandResult:
    __slots__ = ("n_players", "button", "hole", "board", "invested", "payouts", "folded_street",
                 "final_street", "showdown", "values", "actions", "big_blind")

    @property
    def net(self):
        return [p - i for p, i in zip(self.payouts, self.invested)]


def position_name(seat, button, n):
    """BTN / SB / BB / UTG(+k) / LJ / HJ / CO for a seat at an n-handed table."""
    offset = (seat - button) % n
    if n == 2:
        return "BTN" if offset == 0 else "BB"
    if offset < 3:
        return ("BTN", "SB", "BB")[offset]
    before_button = n - offset
    if offset > 3 and before_button <= 3:
        return ("CO", "HJ", "LJ")[before_button - 1]
    return "UTG" if offset == 3 else f"UTG+{offset - 3}"


def play_hand(bots, stacks, button, small_blind, big_blind, rng):
    """Deal and play one hand. ``stacks`` is not modified; chip movements are in the result."""
    n = len(bots)
    deck = rng.sample(range(52), 2 * n + 5)
    hole = [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]
    runout = deck[2 * n:]

    stack = list(stacks)
    invested = [0] * n          # chips put in over the whole hand
    bet = [0] * n               # chips put in on the current street
    folded_street = [-1] * n
    # (street, seat, action, chips added, pot before, to call, players in, info, behind): "behind" is what
    # could still go in after calling, i.e. the player's remaining chips as far as an opponent can match them
    actions = []
    live = n

    def put(seat, amount):
        amount = min(amount, stack[seat])
        stack[seat] -= amount
        bet[seat] += amount
        invested[seat] += amount
        return amount

    sb_seat, bb_seat = (button, 1 - button) if n == 2 else ((button + 1) % n, (button + 2) % n)
    put(sb_seat, small_blind)
    put(bb_seat, big_blind)

    final_street = 0
    for street in range(4):
        final_street = street
        if street == 0:
            current, seat = big_blind, (bb_seat + 1) % n
        else:
            bet = [0] * n
            current, seat = 0, (button + 1) % n
        min_raise = big_blind
        raises = 0
        board = runout[:_BOARD_CARDS[street]]
        may_raise = [True] * n      # False once a player has acted since the last full raise
        to_act = sum(1 for i in range(n) if folded_street[i] < 0 and stack[i] > 0)

        while to_act > 0 and live > 1:
            if folded_street[seat] >= 0 or stack[seat] == 0:
                seat = (seat + 1) % n
                continue
            others = [i for i in range(n) if i != seat and folded_street[i] < 0]
            others_with_chips = sum(1 for i in others if stack[i] > 0)
            # never owe more than an opponent can cover (matters when the big blind is posted short)
            cover = max(bet[i] + stack[i] for i in others)
            to_call = min(current, cover) - bet[seat]
            if to_call <= 0 and others_with_chips == 0:
                break                                   # everyone else is all-in: nothing to decide
            max_to = bet[seat] + stack[seat]
            can_raise = may_raise[seat] and max_to > current and others_with_chips > 0
            owed = min(to_call, stack[seat])
            behind = max(0, min(max_to, cover) - max(current, bet[seat]))

            view = View()
            view.seat, view.hole, view.board, view.street = seat, hole[seat], board, street
            view.pot = pot = sum(invested)
            view.to_call = owed
            view.current_bet = current
            view.min_raise_to = min(current + min_raise, max_to)
            view.max_raise_to = max_to
            view.can_raise = can_raise
            view.stack, view.behind, view.bet, view.big_blind = stack[seat], behind, bet[seat], big_blind
            view.n_players, view.n_live, view.button = n, live, button
            view.position = position_name(seat, button, n)
            view.raises = raises
            view.actions = actions

            answer = bots[seat].act(view)
            action, amount = answer[0], answer[1]
            info = answer[2] if len(answer) > 2 else None
            to_act -= 1

            if action == RAISE and can_raise:
                target = max(view.min_raise_to, min(int(amount), max_to))
                added = put(seat, target - bet[seat])
                if target - current >= min_raise:       # a full raise reopens the betting for everyone
                    min_raise = target - current
                    may_raise = [True] * n
                current = target
                raises += 1
                to_act = others_with_chips
                actions.append((street, seat, RAISE, added, pot, owed, live, info, behind))
            elif action == FOLD and to_call > 0:
                folded_street[seat] = street
                live -= 1
                actions.append((street, seat, FOLD, 0, pot, owed, live + 1, info, behind))
            else:
                added = put(seat, to_call) if to_call > 0 else 0
                actions.append((street, seat, CALL if to_call > 0 else CHECK, added, pot, owed, live, info, behind))
            may_raise[seat] = False
            seat = (seat + 1) % n

        if live == 1:
            break

    result = HandResult()
    result.n_players, result.button, result.hole, result.big_blind = n, button, hole, big_blind
    result.invested, result.folded_street, result.actions = invested, folded_street, actions
    result.final_street = final_street
    result.showdown = live > 1
    result.board = runout if live > 1 else runout[:_BOARD_CARDS[final_street]]
    result.values = [None] * n
    result.payouts = payouts = [0] * n

    contenders = [i for i in range(n) if folded_street[i] < 0]
    if live == 1:
        payouts[contenders[0]] = sum(invested)
        return result

    for i in contenders:
        result.values[i] = evaluate(hole[i] + tuple(runout))
    previous = 0
    for level in sorted({invested[i] for i in contenders}):      # main pot first, then side pots
        pot = sum(min(x, level) - min(x, previous) for x in invested)
        eligible = [i for i in contenders if invested[i] >= level]
        best = max(result.values[i] for i in eligible)
        winners = [i for i in eligible if result.values[i] == best]
        share, odd = divmod(pot, len(winners))
        for w in winners:
            payouts[w] += share
        for w in sorted(winners, key=lambda i: (i - button - 1) % n)[:odd]:   # odd chips: left of the button
            payouts[w] += 1
        previous = level
    return result
