"""FlyDecision: a drop-in replacement for Decision, driven by the fly brain.

main.py's contract is narrow:

    d = Decision(table, history, strategy, game_logger)
    d.make_decision(table, history, strategy, game_logger)
    mouse_target = d.decision          # a DecisionTypes string

plus the attributes the GUI reads (finalCallLimit, finalBetLimit, outs, maxCallEV,
pot_multiple). FlyDecision subclasses Decision and lets the original run first, so all
of that stays populated and the existing behaviour is always available as a baseline.
Only the final action choice is replaced, and only in 'active' mode.

Run modes
---------
shadow  (default) The fly decides, the choice is logged next to the baseline's, and the
        baseline still drives the mouse. Nothing about the account's behaviour changes,
        and the paired (state, fly action, baseline action, outcome) records this
        produces are exactly the training data the fly needs. Start here.
active  The fly drives the mouse. Refused unless the operator has explicitly declared
        the table is play money - see PlayMoneyGuard.
offline No scraper; used by cli.py for self-play and replay.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
import logging

from poker.decisionmaker.decisionmaker import Decision
from poker.flybrain import decoding
from poker.flybrain.brain import FlyBrain
from poker.flybrain.config import FlyBrainConfig, RunMode
from poker.flybrain.guard import PlayMoneyDeclarationMissing, PlayMoneyGuard

log = logging.getLogger(__name__)

_BRAIN_CACHE = {}

# Re-exported so `from poker.flybrain.decision import PlayMoneyGuard` keeps working.
__all__ = ['FlyDecision', 'PlayMoneyGuard', 'PlayMoneyDeclarationMissing',
           'get_brain', 'reset_brain_cache']


def get_brain(config=None):
    """Return a process-wide FlyBrain for this config, building it at most once.

    Rebuilding per decision would re-read the cached connectome every hand and, worse,
    throw away everything the fly had learned.
    """
    config = config or FlyBrainConfig()
    key = (config.scope.value, config.weight_threshold, config.use_synthetic,
           config.cache_dir, config.seed)
    if key not in _BRAIN_CACHE:
        _BRAIN_CACHE[key] = FlyBrain(config)
    return _BRAIN_CACHE[key]


def reset_brain_cache():
    """Drop cached brains. For tests, and for reloading after a config change."""
    _BRAIN_CACHE.clear()


class FlyDecision(Decision):
    """Decision whose final action can come from the connectome instead of the rules."""

    def __init__(self, t, h, p, l, config=None, brain=None):
        super().__init__(t, h, p, l)
        self.config = config or FlyBrainConfig()
        self.guard = PlayMoneyGuard(self.config)
        self.brain = brain or get_brain(self.config)

        # Populated by make_decision so the logger and GUI can report both sides.
        self.baseline_decision = None
        self.fly_decision = None
        self.fly_observation = None
        self.fly_active = False
        self.fly_error = None

    def make_decision(self, t, h, p, l):
        """Run the rule-based decision, then let the fly weigh in."""
        super().make_decision(t, h, p, l)
        self.baseline_decision = self.decision

        try:
            allowed = decoding.legal_actions(t)
            observation = self.brain.decide(t, h, allowed=allowed)
            self.fly_observation = observation
            self.fly_decision = observation.action
        except (ValueError, AttributeError, RuntimeError) as exc:
            # A live hand must never be dropped because the fly failed. Fall back to
            # the rule-based decision and record why.
            self.fly_error = str(exc)
            log.warning("Fly brain failed on this decision (%s); keeping baseline %s",
                        exc, self.baseline_decision)
            return

        reason = self.guard.check()
        if self.config.mode is RunMode.active and reason is None:
            self.decision = self.fly_decision
            self.fly_active = True
            log.info("Fly brain drives: %s (baseline would have been %s)",
                     self.fly_decision, self.baseline_decision)
        else:
            if reason:
                log.warning("Fly brain held in shadow: %s", reason)
            log.info("Fly brain (shadow): %s | baseline drives: %s",
                     self.fly_decision, self.baseline_decision)

    def reinforce(self, reward_bb):
        """Hand the hand's result to the fly. Call once a hand's outcome is known."""
        return self.brain.reinforce(reward_bb)

    def log_dict(self):
        """Fly-specific fields to record alongside the normal game log row."""
        row = {
            'fly_mode': self.config.mode.value,
            'fly_active': self.fly_active,
            'fly_decision': self.fly_decision,
            'baseline_decision': self.baseline_decision,
            'fly_synthetic_connectome': self.brain.is_synthetic,
        }
        if self.fly_error:
            row['fly_error'] = self.fly_error
        if self.fly_observation is not None:
            row.update(self.fly_observation.as_log_dict())
        return row
