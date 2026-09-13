"""Controls that decide whether the connectome is doing any of the work.

The flychess authors put the bar plainly: "we commit to reward-conditioned experiments
and frozen/shuffled controls before making any claim about learning or chess skill."
Almost none of the viral fly-brain demos clear it. A connectome-shaped network can
perform a task for reasons that have nothing to do with the fly - the readout alone may
be carrying it - and the only way to find out is to break the wiring in a way that
preserves everything else and see whether performance follows.

Three controls, in increasing strength:

shuffled  Degree-preserving edge rewiring. Every neuron keeps its in- and out-degree
          and the weight distribution is untouched; only *who connects to whom* is
          destroyed. If performance survives this, the specific wiring was irrelevant
          and only the network's coarse statistics mattered.
frozen    Plasticity off. Separates what the wiring computes from what learning adds.
random    Actions drawn uniformly from the legal set. The floor.

A caveat that matters for interpretation: the operating point is gain-sensitive (see
doc/flybrain.md), and a rewired network does not sit at the same point as the real one.
calibrate_gain() exists so each condition can be brought to the same Kenyon cell
sparsity before they are compared - otherwise the comparison is between activity levels,
not between wiring diagrams.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
import logging

import numpy as np

from poker.flybrain.connectome import Connectome

log = logging.getLogger(__name__)


def shuffle_preserving_degree(connectome, seed=0, n_swap_rounds=10):
    """Return a copy whose edges are rewired but whose degree sequence is preserved.

    Uses double-edge swaps: pick two edges (a->b) and (c->d), replace them with
    (a->d) and (c->b). Every node's in- and out-degree is invariant under this, and the
    multiset of weights is carried along unchanged.
    """
    from scipy import sparse  # pylint: disable=import-outside-toplevel

    matrix = connectome.weights.tocoo()
    rows = matrix.row.copy()
    cols = matrix.col.copy()
    vals = matrix.data.copy()
    n_edges = len(vals)
    if n_edges < 2:
        log.warning("Too few edges (%d) to shuffle", n_edges)
        return connectome

    rng = np.random.default_rng(seed)
    for _ in range(max(1, n_swap_rounds)):
        order = rng.permutation(n_edges)
        half = n_edges // 2
        left, right = order[:half], order[half:2 * half]
        # Swap the targets of paired edges.
        cols[left], cols[right] = cols[right].copy(), cols[left].copy()

    # Drop self-loops the swap may have created rather than leaving a neuron driving
    # itself with an edge it never had.
    keep = rows != cols
    dropped = int((~keep).sum())
    if dropped:
        log.info("Shuffle created %d self-loops; dropped them", dropped)
    rows, cols, vals = rows[keep], cols[keep], vals[keep]

    shuffled = sparse.coo_matrix(
        (vals, (rows, cols)), shape=connectome.weights.shape, dtype=np.float32
    ).tocsr()

    log.info("Shuffled connectome: %d edges (was %d), degrees preserved",
             shuffled.nnz, connectome.n_edges)
    return Connectome(connectome.body_ids, connectome.classes, connectome.types,
                      connectome.signs, shuffled, is_synthetic=connectome.synthetic)


def measure_kc_sparsity(brain, tables, repeats=1):
    """Mean fraction of Kenyon cells active across a set of table states."""
    fractions = []
    for table in tables:
        for _ in range(repeats):
            observation = brain.decide(table)
            fractions.append(float((observation.kc_counts > 0).mean()))
    brain.abandon_hand()
    return float(np.mean(fractions)) if fractions else 0.0


def calibrate_gain(brain_factory, tables, target_sparsity=0.09, bounds=(0.0005, 0.0040),
                   tolerance=0.015, max_iterations=14):
    """Bisect on synaptic_gain until Kenyon cell sparsity hits target_sparsity.

    Args:
        brain_factory: callable taking a gain and returning a FlyBrain built with it.
        tables: representative table states to measure on.
        target_sparsity: fraction of KCs that should be active. ~0.09 matches the real
            connectome at the calibrated default and sits in the range reported for
            odour responses in the fly.
        bounds: (low, high) gains to search between.
        tolerance: stop once |measured - target| is within this.
        max_iterations: bisection steps.

    Returns:
        (gain, measured_sparsity)
    """
    low, high = bounds
    best = (None, None)
    for iteration in range(max_iterations):
        gain = 0.5 * (low + high)
        measured = measure_kc_sparsity(brain_factory(gain), tables)
        log.info("calibrate_gain [%02d] gain=%.5f -> KC sparsity %.3f (target %.3f)",
                 iteration, gain, measured, target_sparsity)
        best = (gain, measured)
        if abs(measured - target_sparsity) <= tolerance:
            break
        if measured > target_sparsity:
            high = gain
        else:
            low = gain
    return best


class RandomAgent:
    """Uniform choice over the legal actions. The performance floor."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.decisions_made = 0

    def decide(self, table, history=None, temperature=0.0, allowed=None):
        """Mirror FlyBrain.decide()'s signature closely enough to swap in."""
        from poker.flybrain import decoding  # pylint: disable=import-outside-toplevel
        from poker.flybrain.brain import Observation  # pylint: disable=import-outside-toplevel

        del history, temperature
        if allowed is None:
            allowed = decoding.legal_actions(table)
        allowed = list(allowed)
        action = allowed[int(self.rng.integers(len(allowed)))]
        self.decisions_made += 1
        return Observation(
            features=np.zeros(0, dtype=np.float32),
            action=action,
            scores={a: 0.0 for a in allowed},
            mbon_rates=np.zeros(1, dtype=np.float32),
            kc_counts=np.zeros(1, dtype=np.int32),
            allowed=tuple(allowed),
            predicted_value=0.0,
        )

    def reinforce(self, reward_bb):
        """No-op: there is nothing to learn."""
        return {'applied': 0, 'reward_bb': float(reward_bb)}

    def abandon_hand(self):
        """No-op."""
        return 0
