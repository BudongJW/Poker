"""Vectorised leaky integrate-and-fire engine over a sparse connectome.

The connectome gives wiring, not dynamics. Membrane time constants, thresholds and
the scale that turns a synapse count into a voltage are all calibration choices, not
measurements - see LIFParams in config.py. Treat the absolute numbers as arbitrary and
the comparisons between conditions (real vs shuffled wiring, plastic vs frozen) as the
thing that carries meaning.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals
import logging

import numpy as np

log = logging.getLogger(__name__)


class LIFEngine:
    """Current-based LIF network driven by a sparse signed weight matrix.

    The weight matrix from Connectome is indexed [pre, post]. Postsynaptic drive is
    therefore W.T @ spikes, so the transpose is built once at construction and kept in
    CSR form where the matrix-vector product is fast.
    """

    def __init__(self, connectome, params, seed=0, competitive_groups=None,
                 normalizing_groups=None):
        """Args:
            normalizing_groups: iterable of (indices, beta). Models the APL neuron's
                graded feedback inhibition onto the Kenyon cells: each member has
                beta * mean(drive to the group) subtracted from its input. This is the
                mechanism to prefer. It holds the population sparse while keeping its
                response a continuous function of the input, so two similar poker states
                still produce similar codes.
            competitive_groups: iterable of (indices, target_sparsity). A hard
                top-k-per-step alternative. It hits a sparsity target exactly but makes
                the response a step function of the input: a small change in drive flips
                which cells win, nearby states map to unrelated codes, and a readout can
                memorise training states without generalising to new ones (measured
                out-of-sample R^2 on decoding equity: -6.9 with top-k). Kept for
                ablations only.
        """
        self.params = params
        self.n = connectome.n_neurons
        self.rng = np.random.default_rng(seed)
        self.plastic_blocks = []
        self.normalizing_groups = []
        self.competitive_groups = []
        for indices, sparsity in (competitive_groups or []):
            indices = np.asarray(indices, dtype=np.int64)
            k = max(1, int(np.ceil(sparsity * len(indices))))
            self.competitive_groups.append((indices, k))
        for indices, beta in (normalizing_groups or []):
            self.normalizing_groups.append((np.asarray(indices, dtype=np.int64),
                                            np.float32(beta)))

        weights = connectome.weights.astype(np.float32) * np.float32(params.synaptic_gain)
        self._w_in = weights.T.tocsr()

        self.steps_per_refractory = max(1, int(round(params.refractory_ms / params.dt_ms)))
        self._decay = np.float32(np.exp(-params.dt_ms / params.tau_m_ms))

        self.v = None
        self.refractory = None
        self._pending = None
        self.reset()

    def detach_block(self, pre_idx, post_idx):
        """Remove a pre->post block from the static matrix and return it dense.

        Used to pull KC->MBON out of the fixed wiring so the plasticity rule can own
        it. The returned block is already scaled by synaptic_gain, so it is in the
        same units the engine integrates. Register it back with add_plastic_block.
        """
        pre_idx = np.asarray(pre_idx, dtype=np.int64)
        post_idx = np.asarray(post_idx, dtype=np.int64)
        block = np.asarray(self._w_in[post_idx][:, pre_idx].todense(), dtype=np.float32).T

        # Zero those entries in the static matrix so they are not counted twice.
        matrix = self._w_in.tolil()
        for row in post_idx:
            matrix[row, pre_idx] = 0.0
        self._w_in = matrix.tocsr()
        self._w_in.eliminate_zeros()
        return block

    def add_plastic_block(self, pre_idx, post_idx, weights):
        """Register a dense pre->post block that is integrated alongside the matrix.

        weights has shape (len(pre_idx), len(post_idx)) and may be mutated in place by
        the caller between steps; the engine reads it fresh on every step.
        """
        self.plastic_blocks.append((np.asarray(pre_idx, dtype=np.int64),
                                    np.asarray(post_idx, dtype=np.int64),
                                    weights))

    def reset(self):
        """Return every neuron to rest. Call between independent decisions."""
        self.v = np.full(self.n, self.params.v_rest, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self._pending = None

    def step(self, external=None):
        """Advance one dt. Returns a bool array of which neurons spiked.

        Args:
            external: optional float32 array of length n, extra drive in voltage units
                per step (used to inject sensory input and DAN stimulation).
        """
        params = self.params
        # Leak toward rest.
        self.v = params.v_rest + (self.v - params.v_rest) * self._decay

        if external is not None:
            self.v += external

        # Refractory neurons are clamped and cannot fire.
        busy = self.refractory > 0
        if busy.any():
            self.v[busy] = params.v_reset
            self.refractory[busy] -= 1

        spikes = (self.v >= params.v_threshold) & ~busy
        spikes = self._apply_competition(spikes)
        if spikes.any():
            self.v[spikes] = params.v_reset
            self.refractory[spikes] = self.steps_per_refractory
            # Recurrent drive lands on the next step, which is the causally correct
            # order and avoids a spike propagating arbitrarily far within one dt.
            fired = spikes.astype(np.float32)
            pending = self._w_in @ fired
            for pre_idx, post_idx, block in self.plastic_blocks:
                active = fired[pre_idx]
                if active.any():
                    pending[post_idx] += block.T @ active
            for indices, beta in self.normalizing_groups:
                drive = pending[indices]
                positive = np.maximum(drive, 0.0)
                pending[indices] = drive - beta * positive.mean()
            self._pending = pending
        else:
            self._pending = None

        return spikes

    def _apply_competition(self, spikes):
        """Suppress all but the most strongly driven spikes within each group."""
        for indices, k in self.competitive_groups:
            candidates = indices[spikes[indices]]
            if len(candidates) <= k:
                continue
            # Keep the k highest membrane potentials; silence the rest.
            potentials = self.v[candidates]
            winners = candidates[np.argpartition(-potentials, k - 1)[:k]]
            spikes[candidates] = False
            spikes[winners] = True
        return spikes

    def run(self, n_steps, sensory_idx=None, sensory_rate_hz=None, tonic=None,
            record=None):
        """Simulate n_steps, returning spike counts per neuron.

        Args:
            n_steps: number of dt steps.
            sensory_idx: indices receiving Poisson input.
            sensory_rate_hz: per-index firing rates in Hz, same length as sensory_idx.
            tonic: optional float32 array of length n, constant extra drive per step
                (used for DAN stimulation during reinforcement).
            record: optional iterable of indices to also return a per-step spike
                raster for, shape (n_steps, len(record)).

        Returns:
            (counts, raster) where counts is an int32 array of length n and raster is
            None unless record was given.
        """
        params = self.params
        counts = np.zeros(self.n, dtype=np.int32)
        raster = None
        record_idx = None
        if record is not None:
            record_idx = np.asarray(record, dtype=np.int64)
            raster = np.zeros((n_steps, len(record_idx)), dtype=bool)

        self._pending = None
        p_spike = None
        if sensory_idx is not None and sensory_rate_hz is not None:
            # Poisson rate -> per-step probability.
            p_spike = np.clip(
                np.asarray(sensory_rate_hz, dtype=np.float32) * params.dt_ms / 1000.0,
                0.0, 1.0,
            )
            sensory_idx = np.asarray(sensory_idx, dtype=np.int64)

        for t in range(n_steps):
            external = np.zeros(self.n, dtype=np.float32)
            if self._pending is not None:
                external += self._pending
            if tonic is not None:
                external += tonic
            if p_spike is not None:
                fired = self.rng.random(len(sensory_idx)) < p_spike
                # A sensory spike drives the cell straight past threshold.
                external[sensory_idx[fired]] += params.v_threshold * 1.05

            spikes = self.step(external)
            counts += spikes
            if raster is not None:
                raster[t] = spikes[record_idx]

        return counts, raster

    def rates_hz(self, counts, n_steps):
        """Convert spike counts from a run of n_steps into firing rates in Hz."""
        duration_s = n_steps * self.params.dt_ms / 1000.0
        if duration_s <= 0:
            return np.zeros_like(counts, dtype=np.float32)
        return counts.astype(np.float32) / np.float32(duration_s)

    def steps_for_ms(self, window_ms):
        """How many dt steps a simulated window of window_ms corresponds to."""
        return max(1, int(round(window_ms / self.params.dt_ms)))
