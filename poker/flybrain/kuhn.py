"""Kuhn poker: the measuring instrument.

Kuhn poker (Harold W. Kuhn, 1950) is the minimal two-player poker game. It exists here
for one reason: its game-theoretic optimum is known in closed form, so a policy's
**exploitability** - how much an optimal opponent would win against it - can be computed
exactly. That makes it a yardstick a degenerate policy cannot game.

The equity game this replaces could be gamed. A bet-everything policy scored well on mean
regret while being wrong about which bet, and a random policy scored well on
optimal-action-rate by folding often; the two metrics disagreed completely. No amount of
tuning on top of a broken measurement means anything.

Rules, in full:

    Three cards, J < Q < K. Both players ante 1, so the pot starts at 2. Each is dealt
    one card; the third is unused. One betting round, bet size 1.

    Player 1 checks or bets.
      check -> Player 2 checks (showdown) or bets.
                 bet -> Player 1 folds or calls.
      bet   -> Player 2 folds or calls.

    Higher card wins at showdown.

Every decision is binary - pass (check/fold) or aggress (bet/call) - so a policy is a
probability of aggression at each of the 12 information sets.

Key facts used as self-tests:

    * 12 information sets, 6 per player.
    * Player 1's equilibria form a one-parameter family in alpha from 0 to 1/3:
      bluff with J at rate alpha, always check Q, bet K at rate 3*alpha, and call a bet
      with Q at rate alpha + 1/3.
    * Player 2's equilibrium is unique: facing a bet, fold J, call K, call Q one third of
      the time; facing a check, bet K, check Q, bet J one third of the time.
    * The game value is -1/18 per hand to Player 1.

A pure (deterministic) policy can never reach equilibrium here, because equilibrium
requires mixing. `best_pure_exploitability()` computes that floor by enumeration, which
is the right bar for a fly reading off an argmax.
"""

# pylint: disable=too-many-locals

import itertools
import logging

from poker.flybrain.encoding import TableView

log = logging.getLogger(__name__)

JACK, QUEEN, KING = 0, 1, 2
CARDS = (JACK, QUEEN, KING)
CARD_NAMES = {JACK: 'J', QUEEN: 'Q', KING: 'K'}

PASS, AGGRESS = 'p', 'b'

# Non-terminal histories, and which seat acts at each. Seat is fully determined by the
# history, which is why one 12-entry policy dict covers both players.
P1_HISTORIES = ('', 'pb')
P2_HISTORIES = ('p', 'b')
HISTORIES = P1_HISTORIES + P2_HISTORIES

# Every information set, in a fixed order.
INFO_SETS = tuple((card, history) for history in HISTORIES for card in CARDS)

GAME_VALUE_TO_P1 = -1.0 / 18.0
DEAL_PROBABILITY = 1.0 / 6.0
DEALS = tuple((a, b) for a, b in itertools.permutations(CARDS, 2))


def acting_player(history):
    """Which seat acts at `history`: 1 or 2."""
    if history in P1_HISTORIES:
        return 1
    if history in P2_HISTORIES:
        return 2
    raise ValueError(f"{history!r} is terminal or not a Kuhn history")


def is_terminal(history):
    """True once the hand is over."""
    return history in ('pp', 'bp', 'bb', 'pbp', 'pbb')


def payoff_to_p1(history, card_p1, card_p2):
    """Chips won by player 1 at a terminal history."""
    p1_wins = card_p1 > card_p2
    if history == 'pp':           # both checked, showdown for the 2-chip pot
        return 1 if p1_wins else -1
    if history == 'bp':           # player 2 folded to a bet
        return 1
    if history == 'pbp':          # player 1 folded to a bet
        return -1
    if history in ('bb', 'pbb'):  # showdown for the 4-chip pot
        return 2 if p1_wins else -2
    raise ValueError(f"{history!r} is not terminal")


def pot_and_to_call(history):
    """(pot, amount to call) facing the player about to act at `history`."""
    if history == '':
        return 2.0, 0.0
    if history == 'p':
        return 2.0, 0.0
    if history == 'b':
        return 3.0, 1.0
    if history == 'pb':
        return 3.0, 1.0
    raise ValueError(f"{history!r} is terminal or not a Kuhn history")


def legal_fly_actions(history):
    """The action names the fly may choose from, in its own vocabulary."""
    _, to_call = pot_and_to_call(history)
    if to_call > 0:
        return ('Fold', 'Call')
    return ('Check', 'Bet')


def action_is_aggressive(action):
    """Map the fly's action vocabulary onto Kuhn's pass/aggress."""
    return action in ('Bet', 'Call', 'Bet half pot', 'Bet pot')


# --- policies --------------------------------------------------------------------

def nash_policy(alpha=1.0 / 6.0):
    """An equilibrium policy. alpha selects a member of player 1's family, in [0, 1/3].

    Returns {(card, history): probability of aggression}.
    """
    if not 0.0 <= alpha <= 1.0 / 3.0 + 1e-12:
        raise ValueError(f"alpha must lie in [0, 1/3], got {alpha}")
    return {
        # Player 1, opening.
        (JACK, ''): alpha,
        (QUEEN, ''): 0.0,
        (KING, ''): 3.0 * alpha,
        # Player 1, facing a bet after checking.
        (JACK, 'pb'): 0.0,
        (QUEEN, 'pb'): alpha + 1.0 / 3.0,
        (KING, 'pb'): 1.0,
        # Player 2, after a check.
        (JACK, 'p'): 1.0 / 3.0,
        (QUEEN, 'p'): 0.0,
        (KING, 'p'): 1.0,
        # Player 2, facing a bet.
        (JACK, 'b'): 0.0,
        (QUEEN, 'b'): 1.0 / 3.0,
        (KING, 'b'): 1.0,
    }


def uniform_policy():
    """Aggress half the time everywhere. The random baseline."""
    return {info_set: 0.5 for info_set in INFO_SETS}


def always_pass_policy():
    """Never bet, never call. A degenerate reference."""
    return {info_set: 0.0 for info_set in INFO_SETS}


def merge_policies(p1_source, p2_source):
    """Take player 1's information sets from one policy and player 2's from another."""
    merged = {}
    for card, history in INFO_SETS:
        source = p1_source if history in P1_HISTORIES else p2_source
        merged[(card, history)] = source[(card, history)]
    return merged


# --- exact evaluation ------------------------------------------------------------

def _walk(history, card_p1, card_p2, policy, reach):
    """Expected payoff to player 1 from `history`, weighted by `reach`."""
    if is_terminal(history):
        return reach * payoff_to_p1(history, card_p1, card_p2)

    card = card_p1 if acting_player(history) == 1 else card_p2
    p_aggress = policy[(card, history)]

    total = 0.0
    if p_aggress > 0.0:
        total += _walk(history + AGGRESS, card_p1, card_p2, policy, reach * p_aggress)
    if p_aggress < 1.0:
        total += _walk(history + PASS, card_p1, card_p2, policy, reach * (1.0 - p_aggress))
    return total


def expected_value(policy):
    """Exact expected chips per hand to player 1 under `policy` for both seats."""
    return sum(
        DEAL_PROBABILITY * _walk('', card_p1, card_p2, policy, 1.0)
        for card_p1, card_p2 in DEALS
    )


def _pure_strategies(histories):
    """Every deterministic assignment over one seat's 6 information sets (64 of them)."""
    info_sets = [(card, history) for history in histories for card in CARDS]
    for bits in itertools.product((0.0, 1.0), repeat=len(info_sets)):
        yield dict(zip(info_sets, bits))


def best_response_value(policy, br_seat):
    """Best expected value seat `br_seat` can obtain against `policy`.

    Computed by enumerating all 64 pure strategies for that seat. A best response to a
    fixed opponent always has a pure maximiser, so enumeration is exact - and it avoids
    the belief arithmetic an efficient best-response walk would need to get right.
    """
    if br_seat == 1:
        best = None
        for candidate in _pure_strategies(P1_HISTORIES):
            value = expected_value(merge_policies(candidate, policy))
            if best is None or value > best:
                best = value
        return best

    best = None
    for candidate in _pure_strategies(P2_HISTORIES):
        # Player 2's utility is the negation of player 1's.
        value = -expected_value(merge_policies(policy, candidate))
        if best is None or value > best:
            best = value
    return best


def exploitability(policy):
    """How far `policy` is from equilibrium, in chips per hand.

    The standard two-player zero-sum measure: the average of what a best response wins
    in each seat. Zero exactly at equilibrium, and never negative.
    """
    return 0.5 * (best_response_value(policy, 1) + best_response_value(policy, 2))


def best_pure_exploitability():
    """Lowest exploitability any deterministic policy can reach, by enumeration.

    Equilibrium in Kuhn poker requires mixing, so a policy that reads off an argmax - as
    the fly does at temperature 0 - cannot reach 0. This is the bar it should be judged
    against. 4096 pure policies, each needing two 64-way best-response searches.
    """
    best, best_policy = None, None
    for bits in itertools.product((0.0, 1.0), repeat=len(INFO_SETS)):
        policy = dict(zip(INFO_SETS, bits))
        value = exploitability(policy)
        if best is None or value < best:
            best, best_policy = value, policy
    return best, best_policy


def describe_policy(policy):
    """A readable table of the policy, for logs and run reports."""
    labels = {'': 'open', 'p': 'vs check', 'b': 'vs bet', 'pb': 'vs bet (after check)'}
    lines = [f"{'info set':28s} {'P(aggress)':>11s}"]
    for history in HISTORIES:
        for card in CARDS:
            name = f"{CARD_NAMES[card]} {labels[history]}"
            lines.append(f"{name:28s} {policy[(card, history)]:>11.3f}")
    return "\n".join(lines)


def self_test():
    """Check the implementation against the published facts. Raises on mismatch."""
    assert len(INFO_SETS) == 12, f"expected 12 information sets, got {len(INFO_SETS)}"

    for alpha in (0.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 3.0):
        value = expected_value(nash_policy(alpha))
        assert abs(value - GAME_VALUE_TO_P1) < 1e-9, (
            f"alpha={alpha}: game value {value:.6f}, expected {GAME_VALUE_TO_P1:.6f}")
        exploit = exploitability(nash_policy(alpha))
        assert abs(exploit) < 1e-9, f"alpha={alpha}: Nash exploitability {exploit:.6f}"

    log.info("Kuhn self-test passed: 12 info sets, game value %.6f, "
             "Nash exploitability 0 across the alpha family", GAME_VALUE_TO_P1)
    return True


# --- presenting Kuhn to the fly --------------------------------------------------

class KuhnTable(TableView):
    """Kuhn state wearing the same attribute surface as a scraped poker table.

    So the fly sees Kuhn poker through exactly the encoding path it uses in a live hand -
    no separate code path that could quietly behave differently.

    Two mappings worth stating plainly:

    * The card becomes `abs_equity`: J -> 0.0, Q -> 0.5, K -> 1.0. This is not a fudge.
      Against one unknown card out of the remaining two, J wins 0 of 2, Q wins 1, K wins
      2 - so these are the true equities.
    * The history occupies the four street slots as a one-hot code: '' -> PreFlop,
      'p' -> Flop, 'b' -> Turn, 'pb' -> River. Kuhn has one betting round and no streets;
      this reuses four existing discrete feature channels to tell the fly which of the
      four decision points it is at. The names are a carrier, not a claim.
    """

    HISTORY_AS_STAGE = {'': 'PreFlop', 'p': 'Flop', 'b': 'Turn', 'pb': 'River'}

    def __init__(self, card, history, stack=100.0):
        pot, to_call = pot_and_to_call(history)
        super().__init__(card / 2.0, pot, to_call, stack,
                         self.HISTORY_AS_STAGE[history], opponents=1, big_blind=1.0,
                         min_bet=1.0)
        # Kuhn's bet is a fixed 1 chip, so betting is only available when nothing is due.
        self.betButton = to_call == 0
        self.card = card
        self.history = history


def extract_policy(brain, temperature=0.0, stack=100.0):
    """Read the fly's policy off all 12 information sets.

    Returns {(card, history): probability of aggression}. At temperature 0 this is a pure
    policy, whose exploitability can be no better than best_pure_exploitability().
    """
    policy = {}
    for card, history in INFO_SETS:
        table = KuhnTable(card, history, stack=stack)
        allowed = legal_fly_actions(history)
        observation = brain.decide(table, allowed=allowed)
        probabilities = brain.decoder.action_probabilities(
            observation.mbon_rates, allowed, temperature)
        policy[(card, history)] = float(
            sum(p for action, p in probabilities.items() if action_is_aggressive(action))
        )
    brain.abandon_hand()
    return policy


def play_hand(brain, opponent_policy, fly_seat, rng, stack=100.0, temperature=0.0):
    """Play one hand with the fly in `fly_seat` against a fixed opponent policy.

    The fly plays one seat only. In self-play it would be acting for both sides of a
    zero-sum hand, and a single reward applied to every pending decision would then be
    reinforcing the winning and losing decisions identically.

    Returns (chips_to_fly, n_fly_decisions).
    """
    deal = list(CARDS)
    rng.shuffle(deal)
    card_p1, card_p2 = deal[0], deal[1]

    history = ''
    fly_decisions = 0
    while not is_terminal(history):
        seat = acting_player(history)
        card = card_p1 if seat == 1 else card_p2
        if seat == fly_seat:
            table = KuhnTable(card, history, stack=stack)
            allowed = legal_fly_actions(history)
            observation = brain.decide(table, allowed=allowed, temperature=temperature)
            aggressive = action_is_aggressive(observation.action)
            fly_decisions += 1
        else:
            aggressive = rng.random() < opponent_policy[(card, history)]
        history += AGGRESS if aggressive else PASS

    chips_to_p1 = payoff_to_p1(history, card_p1, card_p2)
    chips_to_fly = chips_to_p1 if fly_seat == 1 else -chips_to_p1
    return chips_to_fly, fly_decisions
