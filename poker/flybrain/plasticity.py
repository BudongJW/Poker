"""KC->MBON plasticity gated by dopaminergic reward prediction error.

Follows the reduced mushroom body model of Bennett, Philippides & Nowotny
(Nat Commun 12:2569, 2021): dopaminergic neurons do not report absolute reward, they
report a reinforcement prediction error computed against reward predictions fed back
from MBONs, and they depress the KC->MBON synapses that were active when the
reinforcement arrived. Rajagopalan et al. (PNAS 120:e2221415120, 2023) showed the
plasticity rule in the living fly must incorporate reward expectation specifically -
bypassing the expectation abolishes operant matching.

Two opposing DAN channels carry the signal, matching the anatomy:
    PAM clusters (316 cells in MaleCNS v1.0)  - appetitive, driven by chips won
    PPL1 clusters (16 cells, incl. PPL101)    - aversive, driven by chips lost

Depression, not potentiation, is the primary direction in the fly; a decaying
eligibility trace over the KCs active for a decision is what the DAN signal acts on,
because the payoff for a poker decision arrives at the end of the hand rather than
immediately.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


class KCtoMBONPlasticity:
    """Owns the plastic KC->MBON block and applies DAN-gated updates.

    Attributes:
        weights: float32 array (n_kc, n_mbon), in engine units. Mutated in place so
            the LIF engine reads the current values on every step.
        trace: float32 array (n_kc,), the eligibility trace of recently active KCs.
    """

    def __init__(self, weights, params, n_mbon=None):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.params = params
        self.trace = np.zeros(self.weights.shape[0], dtype=np.float32)
        self.n_mbon = n_mbon if n_mbon is not None else self.weights.shape[1]
        self._scale = float(np.abs(self.weights).max()) or 1.0
        self.updates_applied = 0

    def observe(self, kc_counts):
        """Record which KCs were active for this decision, decaying what came before.

        Args:
            kc_counts: spike counts for the KC population over the decision window.
        """
        counts = np.asarray(kc_counts, dtype=np.float32)
        peak = counts.max()
        normalised = counts / peak if peak > 0 else counts
        self.trace *= self.params.trace_decay
        self.trace += normalised
        np.clip(self.trace, 0.0, 1.0, out=self.trace)

    # The reward term is bounded by tanh, but the prediction it is compared against is
    # not. Left unbounded, a large prediction makes a large error, which grows the
    # weights, which grows the prediction: the run diverges to inf and then NaN, and the
    # first symptom is a softmax full of NaN several hundred hands later. Clipping the
    # error is what keeps the update contractive.
    MAX_ABS_ERROR = 2.0

    def reward_prediction_error(self, reward_bb, predicted):
        """Normalised, bounded RPE. reward_bb is the realised payoff in big blinds."""
        scaled = float(np.tanh(reward_bb / max(self.params.reward_scale_bb, 1e-6)))
        predicted = float(predicted)
        if not np.isfinite(predicted):
            predicted = 0.0
        error = scaled - predicted
        return float(np.clip(error, -self.MAX_ABS_ERROR, self.MAX_ABS_ERROR))

    def dan_drive(self, error):
        """Split a signed RPE across the appetitive and aversive DAN channels.

        Returns (appetitive_drive, aversive_drive), both non-negative. Only one is
        non-zero for a given error, which is what the anatomy implies: PAM cells
        respond to reward, PPL1 cells to punishment.
        """
        if error >= 0:
            return float(error), 0.0
        return 0.0, float(-error)

    def apply(self, error, mbon_rates=None):
        """Depress the KC->MBON synapses on the eligibility trace, scaled by the RPE.

        A positive error (better than predicted) weakens the synapses driving the
        MBONs that oppose the chosen action; in the fly this is implemented as
        compartment-specific depression. Here the depression is applied across the
        traced KCs and modulated per MBON by that MBON's own activity, which is the
        feedback term Bennett et al. use to turn absolute reward into a prediction
        error.
        """
        if not self.params.enabled or abs(error) < 1e-9:
            return 0.0

        if mbon_rates is None:
            mbon_modulation = np.ones(self.weights.shape[1], dtype=np.float32)
        else:
            rates = np.asarray(mbon_rates, dtype=np.float32)
            peak = rates.max()
            mbon_modulation = rates / peak if peak > 0 else np.ones_like(rates)

        delta = (
            -self.params.learning_rate
            * float(error)
            * self._scale
            * np.outer(self.trace, mbon_modulation)
        ).astype(np.float32)

        self.weights += delta
        np.clip(self.weights,
                self.params.weight_min * self._scale,
                self.params.weight_max * self._scale,
                out=self.weights)
        self.updates_applied += 1
        return float(np.abs(delta).sum())

    def reset_trace(self):
        """Clear the eligibility trace. Call at the start of a new hand."""
        self.trace[:] = 0.0

    def state_dict(self):
        """Serialisable plasticity state."""
        return {'kc_mbon_weights': self.weights, 'trace': self.trace,
                'updates_applied': np.asarray([self.updates_applied])}

    def load_state_dict(self, state):
        """Restore weights saved by state_dict()."""
        self.weights[:] = np.asarray(state['kc_mbon_weights'], dtype=np.float32)
        if 'trace' in state:
            self.trace[:] = np.asarray(state['trace'], dtype=np.float32)
