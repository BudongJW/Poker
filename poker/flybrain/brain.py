"""FlyBrain: one object that turns a poker table into an action and learns from chips.

Pipeline for a single decision:

    table state -> features -> ORN firing rates        (encoding.py)
                -> 100 ms LIF simulation of the
                   ORN -> ALPN/ALLN -> KC -> MBON
                   pathway from MaleCNS v1.0           (lif.py, circuit.py)
                -> MBON rates -> action scores
                -> legality mask -> chosen action       (decoding.py)

Then when the hand settles:

    chips won/lost -> reward prediction error -> DAN drive
                   -> depression of the KC->MBON synapses on the eligibility trace
                                                        (plasticity.py)
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes,too-many-locals,too-few-public-methods
import logging
import os

import numpy as np

from poker.flybrain import circuit as circuit_mod
from poker.flybrain import connectome as connectome_mod
from poker.flybrain import decoding, encoding, lif, plasticity
from poker.flybrain.config import FlyBrainConfig

log = logging.getLogger(__name__)


class Observation:
    """What the brain saw and did for one decision, kept so learning can be deferred.

    A poker payoff arrives at the end of a hand, several decisions later, so each
    decision is recorded and reinforced retrospectively.
    """

    __slots__ = ('features', 'action', 'scores', 'mbon_rates', 'kc_counts',
                 'allowed', 'predicted_value')

    def __init__(self, features, action, scores, mbon_rates, kc_counts, allowed,
                 predicted_value):
        self.features = features
        self.action = action
        self.scores = scores
        self.mbon_rates = mbon_rates
        self.kc_counts = kc_counts
        self.allowed = allowed
        self.predicted_value = predicted_value

    def as_log_dict(self):
        """Flat dict for the game logger."""
        return {
            'fly_action': self.action,
            'fly_scores': self.scores,
            'fly_allowed': list(self.allowed),
            'fly_predicted_value': self.predicted_value,
            'fly_mbon_mean_hz': float(np.mean(self.mbon_rates)),
            'fly_kc_active_frac': float((self.kc_counts > 0).mean()),
        }


class FlyBrain:
    """A connectome-constrained mushroom body that picks poker actions."""

    def __init__(self, config=None, connectome=None):
        self.config = config or FlyBrainConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.connectome = connectome or connectome_mod.load(
            scope=self.config.scope,
            weight_threshold=self.config.weight_threshold,
            cache_dir=self.config.cache_dir or None,
            use_synthetic=self.config.use_synthetic,
            seed=self.config.seed,
        )
        self.circuit = circuit_mod.build(self.connectome)

        competitive, normalizing = [], []
        if self.config.lif.enforce_kc_sparsity:
            if self.config.lif.kc_hard_winner_take_all:
                competitive.append((self.circuit.kc, self.config.lif.kc_target_sparsity))
            else:
                normalizing.append((self.circuit.kc, self.config.lif.kc_inhibition_beta))
        self.engine = lif.LIFEngine(self.connectome, self.config.lif,
                                    seed=self.config.seed,
                                    competitive_groups=competitive,
                                    normalizing_groups=normalizing)

        # Hand KC->MBON to the plasticity rule: it is the synapse the fly actually
        # modifies when learning which stimulus predicts reward.
        block = self.engine.detach_block(self.circuit.kc, self.circuit.mbon)
        self.plasticity = plasticity.KCtoMBONPlasticity(block, self.config.plasticity)
        self.engine.add_plastic_block(self.circuit.kc, self.circuit.mbon,
                                      self.plasticity.weights)

        self.encoder = encoding.OdourEncoder(self.circuit, self.config.encoding)
        self.decoder = decoding.ActionDecoder(len(self.circuit.mbon), seed=self.config.seed)

        self.pending = []
        self.decisions_made = 0
        self.hands_reinforced = 0
        log.info("FlyBrain ready (%s, mode=%s)",
                 self.connectome.describe(), self.config.mode.value)

    @property
    def is_synthetic(self):
        """True when running on the stand-in connectome rather than real wiring."""
        return self.connectome.synthetic

    def decide(self, table, history=None, temperature=0.0, allowed=None):
        """Run one decision. Returns an Observation.

        The simulation is reset before each decision, so decisions are independent
        given the state: any sequential dependence comes from the plastic weights, not
        from leftover membrane voltages.
        """
        features, orn_rates = self.encoder.encode_table(table, history)
        counts, n_steps = self._simulate(orn_rates)
        rates = self.engine.rates_hz(counts, n_steps)

        mbon_rates = rates[self.circuit.mbon]
        kc_counts = counts[self.circuit.kc]

        if allowed is None:
            allowed = decoding.legal_actions(table)
        action, scores = self.decoder.choose(mbon_rates, allowed,
                                             temperature=temperature, rng=self.rng)
        predicted = self.decoder.value_of(action)

        observation = Observation(features, action, scores, mbon_rates, kc_counts,
                                  allowed, predicted)
        self.pending.append(observation)
        self.decisions_made += 1
        return observation

    def _simulate(self, orn_rates):
        """Run the decision window with the configured sensory drive.

        Returns (spike_counts, n_steps).
        """
        params = self.config.encoding
        self.engine.reset()
        n_steps = self.engine.steps_for_ms(params.window_ms)

        if params.sensory_mode == 'poisson':
            counts, _ = self.engine.run(n_steps, sensory_idx=self.circuit.sensory,
                                        sensory_rate_hz=orn_rates)
            return counts, n_steps

        # Graded current injection: rate -> per-step voltage increment.
        lif_params = self.config.lif
        tonic = np.zeros(self.connectome.n_neurons, dtype=np.float32)
        per_step = np.asarray(orn_rates, dtype=np.float32) / 1000.0 * lif_params.dt_ms
        tonic[self.circuit.sensory] = per_step * lif_params.v_threshold * 1.05
        counts, _ = self.engine.run(n_steps, tonic=tonic)
        return counts, n_steps

    def reinforce(self, reward_bb):
        """Apply the hand's payoff to every decision that led to it.

        Args:
            reward_bb: chips won (positive) or lost (negative) in big blinds.

        Returns:
            dict summarising the update, for logging.
        """
        if not self.pending:
            return {'applied': 0, 'reward_bb': reward_bb}

        total_delta = 0.0
        errors = []
        # Most recent decision first: it is the one most responsible for the outcome.
        for observation in reversed(self.pending):
            error = self.plasticity.reward_prediction_error(
                reward_bb, observation.predicted_value)
            errors.append(error)

            self.plasticity.observe(observation.kc_counts)
            # Re-establish the readout's view of this decision before updating it.
            self.decoder.scores(observation.mbon_rates)
            self.decoder.update(observation.action, error,
                                self.config.plasticity.learning_rate)
            total_delta += self.plasticity.apply(error, observation.mbon_rates)

        appetitive, aversive = self.plasticity.dan_drive(float(np.mean(errors)))
        summary = {
            'applied': len(self.pending),
            'reward_bb': float(reward_bb),
            'mean_rpe': float(np.mean(errors)),
            'dan_appetitive_drive': appetitive,
            'dan_aversive_drive': aversive,
            'n_dan_appetitive': int(len(self.circuit.dan_appetitive)),
            'n_dan_aversive': int(len(self.circuit.dan_aversive)),
            'weight_delta_l1': total_delta,
        }
        self.pending.clear()
        self.plasticity.reset_trace()
        self.hands_reinforced += 1
        log.debug("Reinforced hand: %s", summary)
        return summary

    def abandon_hand(self):
        """Drop pending decisions without reinforcement (e.g. a misread hand)."""
        dropped = len(self.pending)
        self.pending.clear()
        self.plasticity.reset_trace()
        return dropped

    def save(self, path):
        """Persist the learned state: plastic KC->MBON weights and the readout."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {}
        state.update(self.plasticity.state_dict())
        readout = self.decoder.state_dict()
        state['readout_weights'] = readout['weights']
        state['readout_bias'] = readout['bias']
        state['decisions_made'] = np.asarray([self.decisions_made])
        state['hands_reinforced'] = np.asarray([self.hands_reinforced])
        state['synthetic'] = np.asarray([self.is_synthetic])
        np.savez_compressed(path, **state)
        log.info("Saved fly brain state to %s (%d decisions, %d hands)",
                 path, self.decisions_made, self.hands_reinforced)

    def load(self, path):
        """Restore learned state saved by save()."""
        data = np.load(path, allow_pickle=False)
        if bool(data['synthetic'][0]) != self.is_synthetic:
            log.warning("Loading state trained on a %s connectome into a %s one",
                        'synthetic' if data['synthetic'][0] else 'real',
                        'synthetic' if self.is_synthetic else 'real')
        self.plasticity.load_state_dict({'kc_mbon_weights': data['kc_mbon_weights'],
                                         'trace': data['trace']})
        self.decoder.load_state_dict({'weights': data['readout_weights'],
                                      'bias': data['readout_bias']})
        self.decisions_made = int(data['decisions_made'][0])
        self.hands_reinforced = int(data['hands_reinforced'][0])
        log.info("Loaded fly brain state from %s", path)

    def manifest(self):
        """Everything needed to reproduce or audit a run."""
        return {
            'connectome': self.connectome.describe(),
            'synthetic': self.is_synthetic,
            'attribution': connectome_mod.ATTRIBUTION,
            'populations': {name: int(len(getattr(self.circuit, name)))
                            for name in ('orn', 'alpn', 'alln', 'kc', 'mbon', 'dan')},
            'dan_appetitive': int(len(self.circuit.dan_appetitive)),
            'dan_aversive': int(len(self.circuit.dan_aversive)),
            'config': self.config.as_dict(),
            'decisions_made': self.decisions_made,
            'hands_reinforced': self.hands_reinforced,
        }
