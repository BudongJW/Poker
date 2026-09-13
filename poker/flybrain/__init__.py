"""Connectome-constrained decision module based on the MaleCNS v1.0 fruit fly connectome.

The fly's mushroom body is the circuit that, in the living animal, tracks probabilistic
reward and drives value-based choice (Rajagopalan et al., PNAS 2023; Bennett et al.,
Nat Commun 2021). This package instantiates that circuit from the published wiring
diagram and uses it to pick a poker action among the options the table offers.

Connectome data: MaleCNS v1.0, HHMI Janelia + Google Research, licensed CC-BY.
See doc/flybrain.md for the circuit rationale, citations and the limits of the approach.
"""

from poker.flybrain.config import FlyBrainConfig, RunMode

__all__ = ['FlyBrainConfig', 'RunMode']
