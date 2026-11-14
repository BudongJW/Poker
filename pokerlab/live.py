"""Sit down and play: one human seat at a table of bots, a hand at a time.

A :class:`Session` deals hands with the engine. The bots act at once; when it is the human's turn
the hand waits. Because the game itself knows every seat, stack and action, the calculator gets
its input from the play as it happens: nothing is picked by hand and nothing has to be read off
a picture. :meth:`Session.analysis` is the same analysis the rest of the package gives, for the
decision the human is facing right now.

This is the package's own game. It does not watch other programs.
"""

import queue
import random
import threading

from . import advisor, table
from .bots import BOT_TYPES, make_bot
from .cards import card_str
from .engine import CALL, FOLD, RAISE, play_hand, position_name

SMALL_BLIND, BIG_BLIND = 5, 10
OPPONENTS = ("human", "pro", "champion", "tag_model", "tag", "lag", "nit", "station", "maniac", "random")
WAIT_SECONDS = 60


class _Leave(Exception):
    """Raised inside a hand that is abandoned."""


class _Seat:
    """The human's seat: a bot whose answer comes from outside."""
    name = "you"

    def __init__(self, session):
        self.session = session

    def act(self, view):
        return self.session._ask(view)


class Session:
    """A table with the human in seat 0 and bots in the other seats. Stacks are reset every hand."""

    def __init__(self, opponents=("human",) * 5, stack_bb=100, seed=None):
        opponents = list(opponents)
        if not 1 <= len(opponents) <= 8:
            raise ValueError("상대는 1~8명이어야 합니다")
        unknown = [kind for kind in opponents if kind not in OPPONENTS or kind not in BOT_TYPES]
        if unknown:
            raise ValueError(f"고를 수 없는 상대입니다: {', '.join(unknown)} ({', '.join(OPPONENTS)} 중에서 고릅니다)")
        if not 5 <= stack_bb <= 1000:
            raise ValueError("스택은 5~1000bb여야 합니다")
        self.rng = random.Random(seed)
        self.kinds = opponents
        self.bots = [_Seat(self)] + [make_bot(kind, seed=self.rng.getrandbits(32)) for kind in opponents]
        self.names = ["나"] + [f"{kind} {i}" for i, kind in enumerate(opponents, 1)]
        self.n, self.stack = len(self.bots), stack_bb * BIG_BLIND
        self.button = self.rng.randrange(self.n)
        self.hands = self.net = 0                       # hands finished and the human's chips won over them
        self._answers, self._ready = queue.Queue(), threading.Event()
        self._view = self._result = self._error = self._thread = None

    # ---- the hand in progress ---------------------------------------------------------

    def _ask(self, view):
        self._view = view
        self._ready.set()
        answer = self._answers.get()
        if answer is None:
            raise _Leave
        return answer

    def _run(self):
        try:
            self._result = play_hand(self.bots, [self.stack] * self.n, self.button, SMALL_BLIND, BIG_BLIND, self.rng)
            self.hands += 1
            self.net += self._result.net[0]
        except _Leave:
            pass
        except Exception as error:                      # a bot failed: report it instead of hanging the table
            self._error = error
        finally:
            self._view = None
            self._ready.set()

    def _wait(self):
        if not self._ready.wait(WAIT_SECONDS):
            raise RuntimeError("봇의 응답을 기다리다 시간이 초과됐습니다")
        if self._error:
            error, self._error = self._error, None
            raise RuntimeError(f"핸드를 진행하지 못했습니다: {error}")

    def deal(self):
        """Start the next hand and play it up to the human's first decision (or to its end)."""
        if self._view is not None:
            raise ValueError("진행 중인 핸드가 있습니다")
        self.button = (self.button + 1) % self.n
        self._result = None
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._wait()
        return self.state()

    def act(self, action, amount=None):
        """The human's answer: 'fold', 'call' (also a check) or 'raise' to ``amount`` chips in front."""
        view = self._view
        if view is None:
            raise ValueError("지금은 액션할 차례가 아닙니다")
        if action == "fold":
            answer = (FOLD, 0) if view.to_call > 0 else (CALL, 0)      # nothing to call: a check
        elif action in ("call", "check"):
            answer = (CALL, 0)
        elif action == "raise":
            if not view.can_raise:
                raise ValueError("지금은 레이즈할 수 없습니다")
            if amount is None:
                raise ValueError("레이즈 금액이 필요합니다")
            answer = (RAISE, max(view.min_raise_to, min(int(round(float(amount))), view.max_raise_to)))
        else:
            raise ValueError(f"알 수 없는 액션입니다: {action!r}")
        self._ready.clear()
        self._answers.put(answer)
        self._wait()
        return self.state()

    def close(self):
        """Abandon the hand in progress, if any."""
        if self._view is not None:
            self._ready.clear()
            self._answers.put(None)
            self._ready.wait(WAIT_SECONDS)

    # ---- what the table looks like ----------------------------------------------------

    def _hand(self):
        """The hand at the human's decision, in the form the analysis takes."""
        view = self._view
        return table.from_engine(view.actions, [self.stack] * self.n, self.button, SMALL_BLIND, BIG_BLIND, 0,
                                 view.hole, view.board, view.to_call, self.names)

    def state(self):
        """Everything a screen needs, JSON-ready."""
        out = {"hands": self.hands, "net_bb": self.net / BIG_BLIND, "big_blind": BIG_BLIND, "stack_bb": self.stack / BIG_BLIND,
               "bb100": 100 * self.net / BIG_BLIND / self.hands if self.hands else None, "opponents": self.kinds}
        view, result = self._view, self._result
        if view is not None:
            hand = self._hand()
            level = view.current_bet
            out.update(
                phase="decision", hero=[card_str(c) for c in view.hole], board=[card_str(c) for c in view.board],
                pot=view.pot, to_call=view.to_call, can_raise=view.can_raise, min_raise_to=view.min_raise_to,
                max_raise_to=view.max_raise_to, current_bet=level, my_bet=view.bet, seats=table.describe(hand),
                button=self.button)
        elif result is not None:
            folded = {a[1] for a in result.actions if a[2] == FOLD}
            hand = table.from_engine(result.actions, [self.stack] * self.n, self.button, SMALL_BLIND, BIG_BLIND, 0,
                                     result.hole[0], result.board, 0, self.names)
            seats = table.describe(hand)
            for seat in seats:
                i = seat["seat"]
                seat.update(net=result.net[i], stack=self.stack + result.net[i], bet=0, in_hand=i not in folded,
                            won=result.payouts[i] > 0,
                            cards=[card_str(c) for c in result.hole[i]] if result.showdown and i not in folded else None)
            out.update(
                phase="over", hero=[card_str(c) for c in result.hole[0]], board=[card_str(c) for c in result.board],
                pot=sum(result.invested), seats=seats, button=self.button, showdown=result.showdown,
                result_bb=result.net[0] / BIG_BLIND)
        else:
            out.update(phase="idle")
        if "seats" in out:
            for seat in out["seats"]:
                seat["position"] = position_name(seat["seat"], self.button, self.n)
        return out

    def analysis(self, model=advisor.DEFAULT_MODEL, profile="all", trials=200_000, seed=None, precision=None):
        """The calculator's view of the decision the human faces now (None between decisions)."""
        if self._view is None:
            return None
        hand = self._hand()
        out = advisor.to_json(table.analyze_hand(hand, model, profile, trials, seed, precision))
        out["table"] = {"source": "game", "seats": table.describe(hand), "notes": [], "pot": hand["pot"], "to_call": hand["to_call"],
                        "big_blind": BIG_BLIND, "n_players": self.n, "live": len(hand["live"]),
                        "hero": [card_str(c) for c in hand["hero"]], "board": [card_str(c) for c in hand["board"]]}
        return out
