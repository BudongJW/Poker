"""MBON population activity -> a poker action the table actually offers.

In the fly, mushroom body output neurons bias behaviour along an approach/avoid axis;
which compartment an MBON reports from determines the sign of that bias. Rather than
hard-coding a valence for each of the 97 MBONs from the literature - the published
assignments do not cover every type in MaleCNS v1.0, and guessing would quietly bake
in the answer - the MBON -> action readout here is *learned*, and which MBONs end up
driving which action is reported as a result.

That keeps this a connectome-constrained model in the sense the field uses the term:
the wiring from ORN through KC to MBON is fixed by the published connectome, and only
the small final readout is fitted.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# The action vocabulary, matching DecisionTypes in poker/decisionmaker/decisionmaker.py
# so a chosen action can be handed straight to the existing mouse mover.
ACTIONS = ('Fold', 'Check', 'Call', 'Bet', 'Bet half pot', 'Bet pot')

def _get(table, name, default=None):
    """Read an attribute the scraper may not have populated."""
    return getattr(table, name, default)


def legal_actions(table):
    """Return the subset of ACTIONS the table currently allows.

    Mirrors the button logic in decisionmaker.admin(): when the check button is up
    there is nothing to fold to and nothing to call, and an all-in call collapses the
    choice to calling or folding.
    """
    if _get(table, 'allInCallButton'):
        return ('Fold', 'Call')

    allowed = []
    if _get(table, 'checkButton'):
        allowed.append('Check')
    else:
        allowed.append('Fold')
        if _get(table, 'callButton', True):
            allowed.append('Call')

    if _get(table, 'betButton', True):
        allowed.extend(['Bet', 'Bet half pot', 'Bet pot'])
    return tuple(allowed)


class ActionDecoder:
    """Linear readout from MBON firing rates to action scores.

    Attributes:
        weights: float32 array (n_mbon, n_actions).
        bias: float32 array (n_actions,).
    """

    BIAS_RATE_FACTOR = 0.05
    MAX_ABS_WEIGHT = 10.0

    def __init__(self, n_mbon, seed=0, actions=ACTIONS):
        self.actions = tuple(actions)
        rng = np.random.default_rng(seed)
        # Small random init: no action is favoured before any learning happens.
        self.weights = rng.normal(0.0, 0.05, size=(n_mbon, len(self.actions))).astype(np.float32)
        self.bias = np.zeros(len(self.actions), dtype=np.float32)
        self._last_rates = None

    def scores(self, mbon_rates):
        """Raw score per action, before legality masking.

        The rates are centred and L2-normalised. Normalising matters beyond tidiness:
        the readout's output doubles as the value prediction the reward prediction
        error is computed against, and raw Hz over 97 MBONs puts that prediction two
        orders of magnitude away from the tanh-squashed reward term, which inverts the
        sign of the error and turns every win into a punishment.
        """
        rates = np.asarray(mbon_rates, dtype=np.float32)
        centred = rates - rates.mean()
        # Standardise rather than L2-normalise. With 97 MBONs, unit L2 norm leaves each
        # component at ~0.1, so the cue-dependent part of a score is ~0.05 while the
        # cue-independent bias grows to ~0.5 - the policy then stops depending on the
        # cue at all and collapses onto whichever action it happened to favour early.
        spread = float(centred.std())
        if spread > 0:
            centred = centred / spread
        self._last_rates = centred
        return centred @ self.weights + self.bias

    def choose(self, mbon_rates, allowed, temperature=0.0, rng=None):
        """Pick an action from `allowed`.

        Args:
            mbon_rates: MBON firing rates in Hz.
            allowed: iterable of action names that are currently legal.
            temperature: 0 picks the argmax; >0 samples from a softmax, which is how
                the fly's matching behaviour looks - choices distributed in proportion
                to value rather than always taking the best option. Expressed in units
                of the current score spread, not raw score units: the readout's absolute
                scale changes as it learns, and a fixed absolute temperature silently
                turns into uniform noise early on and into a hard argmax later.
            rng: numpy Generator, required when temperature > 0.

        Returns:
            (action_name, scores_dict)
        """
        allowed = [a for a in allowed if a in self.actions]
        if not allowed:
            raise ValueError("No legal action is in the decoder's vocabulary")

        raw = self.scores(mbon_rates)
        index = {a: self.actions.index(a) for a in allowed}
        masked = np.asarray([raw[index[a]] for a in allowed], dtype=np.float32)
        # A non-finite score must not become a NaN probability and kill the run.
        if not np.all(np.isfinite(masked)):
            log.warning("Non-finite action scores %s; treating them as neutral", masked)
            masked = np.nan_to_num(masked, nan=0.0, posinf=0.0, neginf=0.0)

        if temperature > 0:
            rng = rng or np.random.default_rng()
            spread = float(masked.std())
            scale = max(temperature * spread, 1e-6) if spread > 0 else 1.0
            shifted = np.clip((masked - masked.max()) / scale, -60.0, 0.0)
            probabilities = np.exp(shifted)
            total = probabilities.sum()
            if not np.isfinite(total) or total <= 0:
                probabilities = np.full(len(allowed), 1.0 / len(allowed))
            else:
                probabilities = probabilities / total
            pick = allowed[int(rng.choice(len(allowed), p=probabilities))]
        else:
            pick = allowed[int(np.argmax(masked))]

        return pick, {a: float(raw[index[a]]) for a in allowed}

    def action_probabilities(self, mbon_rates, allowed, temperature):
        """Return {action: probability} using the same softmax `choose` uses.

        Exposed because a mixed policy has to be read out of the fly, not inferred: Kuhn
        poker's equilibrium requires mixing, so the probabilities are the object of
        interest rather than the single action sampled from them.
        """
        allowed = [a for a in allowed if a in self.actions]
        if not allowed:
            raise ValueError("No legal action is in the decoder's vocabulary")

        raw = self.scores(mbon_rates)
        index = {a: self.actions.index(a) for a in allowed}
        masked = np.asarray([raw[index[a]] for a in allowed], dtype=np.float32)
        if not np.all(np.isfinite(masked)):
            masked = np.nan_to_num(masked, nan=0.0, posinf=0.0, neginf=0.0)

        if temperature <= 0:
            probabilities = np.zeros(len(allowed), dtype=np.float64)
            probabilities[int(np.argmax(masked))] = 1.0
            return dict(zip(allowed, probabilities))

        spread = float(masked.std())
        scale = max(temperature * spread, 1e-6) if spread > 0 else 1.0
        shifted = np.clip((masked - masked.max()) / scale, -60.0, 0.0)
        probabilities = np.exp(shifted)
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            probabilities = np.full(len(allowed), 1.0 / len(allowed))
        else:
            probabilities = probabilities / total
        return dict(zip(allowed, probabilities))

    def value_of(self, action):
        """The decoder's current estimate for one action, from the last scores() call.

        Used as the prediction the reward prediction error is computed against.
        """
        if self._last_rates is None or action not in self.actions:
            return 0.0
        column = self.actions.index(action)
        return float(self._last_rates @ self.weights[:, column] + self.bias[column])

    def update(self, action, error, learning_rate):
        """Nudge the readout for one action in the direction of the error.

        This is a delta rule on the readout only; the connectome-constrained weights
        upstream are handled by plasticity.KCtoMBONPlasticity.
        """
        if self._last_rates is None or action not in self.actions:
            return
        column = self.actions.index(action)
        if not np.isfinite(error):
            return
        self.weights[:, column] += learning_rate * error * self._last_rates
        np.clip(self.weights, -self.MAX_ABS_WEIGHT, self.MAX_ABS_WEIGHT,
                out=self.weights)
        # The bias cannot represent cue dependence, so it is deliberately slow: left at
        # the same rate as the weights it outgrows them and drives the collapse above.
        self.bias[column] += learning_rate * self.BIAS_RATE_FACTOR * error

    def mbon_contributions(self, action):
        """Per-MBON contribution to one action, for reporting which cells drive what."""
        if action not in self.actions:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return self.weights[:, self.actions.index(action)].copy()

    def state_dict(self):
        """Serialisable readout state."""
        return {'weights': self.weights, 'bias': self.bias, 'actions': np.asarray(self.actions)}

    def load_state_dict(self, state):
        """Restore a readout saved by state_dict()."""
        self.weights = np.asarray(state['weights'], dtype=np.float32)
        self.bias = np.asarray(state['bias'], dtype=np.float32)
