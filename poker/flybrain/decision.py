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
import os

from poker.decisionmaker.decisionmaker import Decision
from poker.flybrain import decoding
from poker.flybrain.brain import FlyBrain
from poker.flybrain.config import FlyBrainConfig, RunMode
from poker.flybrain.guard import PlayMoneyDeclarationMissing, PlayMoneyGuard

log = logging.getLogger(__name__)

_BRAIN_CACHE = {}

# Re-exported so `from poker.flybrain.decision import PlayMoneyGuard` keeps working.
__all__ = ['FlyDecision', 'PlayMoneyGuard', 'PlayMoneyDeclarationMissing',
           'get_brain', 'reset_brain_cache', 'persist_brain']


def get_brain(config=None):
    """Return a process-wide FlyBrain for this config, building it at most once.

    Rebuilding per decision would re-read the cached connectome every hand and, worse,
    throw away everything the fly had learned.

    On first build the saved state at `config.brain_path` is restored if it exists, so
    learning carries across bot restarts. Without that, every session would be the fly's
    first and a live track record would measure a permanently naive fly.
    """
    config = config or FlyBrainConfig()
    key = (config.scope.value, config.weight_threshold, config.use_synthetic,
           config.cache_dir, config.seed)
    if key not in _BRAIN_CACHE:
        brain = FlyBrain(config)
        _restore(brain, config.brain_path)
        _BRAIN_CACHE[key] = brain
    return _BRAIN_CACHE[key]


def _restore(brain, path):
    """Load saved state if there is any. A missing or broken file is not fatal."""
    if not path or not os.path.exists(path):
        if path:
            log.info("No saved fly brain at %s; starting from an untrained readout", path)
        return False
    try:
        brain.load(path)
        return True
    except (OSError, ValueError, KeyError) as exc:
        # A corrupt save must not stop the bot playing; it just starts untrained.
        log.warning("Could not load the fly brain from %s (%s); starting untrained",
                    path, exc)
        return False


def persist_brain(brain, config, force=False):
    """Save the brain every `save_every_hands` reinforced hands.

    A poker session almost always ends by the process being killed rather than by a clean
    shutdown, so saving on exit would lose the session. Returns True if it wrote.
    """
    path = getattr(config, 'brain_path', '')
    every = getattr(config, 'save_every_hands', 0)
    if not path or (not force and (not every or brain.hands_reinforced % every)):
        return False
    try:
        brain.save(path)
        return True
    except (OSError, ValueError) as exc:
        log.warning("Could not save the fly brain to %s (%s)", path, exc)
        return False


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
        """Hand the hand's result to the fly. Call once a hand's outcome is known.

        Persists periodically so a killed session keeps what it learned.
        """
        result = self.brain.reinforce(reward_bb)
        if persist_brain(self.brain, self.config):
            result = dict(result, saved_to=self.config.brain_path)
        return result

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
