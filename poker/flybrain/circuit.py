"""The mushroom body, wired up as a decision organ.

Why this circuit and not the whole brain
----------------------------------------
The mushroom body is where the living fly solves the problem poker poses. Rajagopalan
et al. (PNAS 120:e2221415120, 2023) put individual flies through a dynamic foraging
task with probabilistic rewards and found they follow Herrnstein's operant matching
law, tracking changing reward probabilities over time; the mechanism is synaptic
plasticity in the mushroom body that incorporates an expectation of reward. Bennett
et al. (Nat Commun 12:2569, 2021) took a reduced mushroom body model through a
multi-armed bandit and showed DANs can compute a reinforcement prediction error from
MBON feedback.

So the pathway below is not an arbitrary slice. It is the part of the fly that does
value-based choice under uncertainty, which is the part of poker a fly could plausibly
do at all.

    ORN (2,639)  olfactory receptor neurons - the input channels
      |
    ALPN (686) / ALLN (420)  antennal lobe projection + local neurons; the local
      |                      neurons provide inhibitory gain control
    KC (4,064)   Kenyon cells - sparse, high-dimensional recoding
      |
    MBON (97)    mushroom body output neurons - the action readout
      ^
    DAN (340)    dopaminergic neurons - the teaching signal.
                 PAM clusters (316 cells) carry appetitive signals,
                 PPL1 clusters (16 cells, including the PPL101 pair) aversive ones.

Measured edge counts in MaleCNS v1.0 at weight>=3: ORN->ALPN 20,742, ALPN->KC 20,777,
KC->MBON 44,042.
"""

# pylint: disable=too-many-instance-attributes
import logging

import numpy as np

from poker.flybrain.config import Scope

log = logging.getLogger(__name__)

# Appetitive DANs. PAM clusters innervate the compartments whose MBONs bias approach.
APPETITIVE_DAN_PREFIXES = ('PAM',)
# Aversive DANs. PPL1 includes PPL101, the pair DOOMFLY drove with in-game damage.
AVERSIVE_DAN_PREFIXES = ('PPL1',)


class MushroomBodyCircuit:
    """Population index maps over a loaded Connectome, plus the reward channels.

    Attributes:
        connectome: the underlying Connectome.
        orn, alpn, alln, kc, mbon, dan: int arrays of neuron indices per population.
        dan_appetitive, dan_aversive: DAN subsets used as the reward/punishment inputs.
        sensory: the neurons that external input is injected into (the ORNs).
        readout: the neurons the action decoder reads (the MBONs).
    """

    def __init__(self, connectome):
        self.connectome = connectome
        self.orn = connectome.index_of_class('olfactory')
        self.alpn = connectome.index_of_class('ALPN')
        self.alln = connectome.index_of_class('ALLN')
        self.kc = connectome.index_of_class('Kenyon_Cell')
        self.mbon = connectome.index_of_class('MBON')
        self.dan = connectome.index_of_class('DAN')

        dan_set = set(self.dan.tolist())
        app = connectome.index_of_type_prefix(*APPETITIVE_DAN_PREFIXES)
        avr = connectome.index_of_type_prefix(*AVERSIVE_DAN_PREFIXES)
        self.dan_appetitive = np.asarray([i for i in app if i in dan_set], dtype=np.int64)
        self.dan_aversive = np.asarray([i for i in avr if i in dan_set], dtype=np.int64)

        if len(self.dan_appetitive) == 0 or len(self.dan_aversive) == 0:
            # Synthetic connectomes, or a scope that excluded the DAN annotations.
            # Split the DANs so the plasticity rule still has two opposing channels.
            half = len(self.dan) // 2
            self.dan_appetitive = self.dan[:half]
            self.dan_aversive = self.dan[half:]
            log.warning("DAN clusters not resolvable by type; splitting the %d DANs "
                        "arbitrarily into %d appetitive / %d aversive.",
                        len(self.dan), len(self.dan_appetitive), len(self.dan_aversive))

        self._validate()

    @property
    def sensory(self):
        """Indices that external stimulation is injected into."""
        return self.orn

    @property
    def readout(self):
        """Indices the action decoder reads spike counts from."""
        return self.mbon

    def _validate(self):
        """Fail loudly rather than silently simulating an empty circuit."""
        for name in ('orn', 'kc', 'mbon', 'dan'):
            if len(getattr(self, name)) == 0:
                raise ValueError(
                    f"Population '{name}' is empty. The loaded connectome has no "
                    f"neurons of that class - check the scope and weight threshold."
                )

    def kc_to_mbon_block(self):
        """Return the KC->MBON submatrix, the synapses plasticity acts on.

        Returned dense because it is only 4,064 x 97 (~1.6 MB in float32) and the
        plasticity rule touches all of it on every update.
        """
        block = self.connectome.weights[self.kc][:, self.mbon]
        return np.asarray(block.todense(), dtype=np.float32)

    def describe(self):
        """Multi-line summary for logs and run manifests."""
        lines = [self.connectome.describe()]
        for name in ('orn', 'alpn', 'alln', 'kc', 'mbon', 'dan'):
            lines.append(f"  {name:4s} {len(getattr(self, name)):6,d}")
        lines.append(f"  DAN appetitive (PAM)  {len(self.dan_appetitive):6,d}")
        lines.append(f"  DAN aversive   (PPL1) {len(self.dan_aversive):6,d}")
        return "\n".join(lines)


def build(connectome):
    """Construct the circuit from a Connectome, checking the scope makes sense."""
    circuit = MushroomBodyCircuit(connectome)
    log.info("Mushroom body circuit:\n%s", circuit.describe())
    return circuit


def load_and_build(scope=Scope.mushroom_body, **kwargs):
    """Convenience: load a connectome then build the circuit over it."""
    from poker.flybrain import connectome as connectome_mod  # pylint: disable=import-outside-toplevel
    return build(connectome_mod.load(scope=scope, **kwargs))
