"""The play-money declaration checkpoint.

Kept in its own module, free of any poker-stack import, so it can be reasoned about and
tested without pulling in matplotlib, PyQt6 or a database config. A safety-relevant check
should not be reachable only through a heavy import chain.
"""

import logging

from poker.flybrain.config import RunMode

log = logging.getLogger(__name__)


class PlayMoneyDeclarationMissing(RuntimeError):
    """Raised when active mode is requested without a play-money declaration."""


class PlayMoneyGuard:
    """Refuses to let the fly drive the mouse unless play money is declared.

    PokerStars' terms prohibit automated play, and that prohibition covers play-money
    tables too - the exposure there is the account, not a balance. This repository's
    operator plays points/play-money games only, so this guard makes that constraint
    explicit in code rather than leaving it as an intention: 'active' mode requires
    flybrain.confirmed_play_money to be set by hand in config.ini.

    The guard cannot detect a cash table on its own. It is a declaration checkpoint, not
    a safety net, and it is deliberately impossible to satisfy by accident.
    """

    def __init__(self, config):
        self.config = config

    def check(self):
        """Return None if active play is permitted, or a reason string if not."""
        if self.config.mode is not RunMode.active:
            return None
        if not self.config.play_money_only:
            return None
        if not self.config.confirmed_play_money:
            return ("flybrain.mode=active requires flybrain.confirmed_play_money=true "
                    "in config.ini. Set it only for a play-money/points table.")
        return None

    def enforce(self):
        """Raise if active play is not permitted."""
        reason = self.check()
        if reason:
            raise PlayMoneyDeclarationMissing(reason)
