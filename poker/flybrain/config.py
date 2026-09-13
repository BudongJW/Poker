"""Configuration for the fly-brain decision module."""

# pylint: disable=too-many-instance-attributes
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class RunMode(Enum):
    """How much authority the fly brain has over the mouse.

    shadow: the fly decides and everything is logged, but the existing Decision drives
            the mouse. This is the default: it collects paired training data at zero
            risk to the account.
    active: the fly drives the mouse. Requires an explicit opt-in in config.ini.
    offline: no scraper involved at all - self-play or replay only.
    """

    shadow, active, offline = ['shadow', 'active', 'offline']


class Scope(Enum):
    """Which part of the connectome to instantiate."""

    # ORN -> ALPN/ALLN -> KC -> MBON with DAN modulation: 8,246 neurons. The decision
    # organ, and small enough to run in real time on a CPU.
    mushroom_body = 'mushroom_body'
    # Every Traced neuron: 165,122. Feasible but much slower; kept for ablations.
    whole_brain = 'whole_brain'


# Neurotransmitter -> postsynaptic sign. acetylcholine is the fly's main excitatory
# transmitter, GABA the main inhibitory one. Glutamate is inhibitory at many central
# fly synapses (via GluCl-alpha), which is why it is negative here rather than positive
# as a vertebrate-trained intuition would suggest.
NT_SIGN = {
    'acetylcholine': +1.0,
    'glutamate': -1.0,
    'gaba': -1.0,
    'histamine': -1.0,
    'dopamine': 0.0,     # modulatory, routed through the plasticity rule instead
    'octopamine': 0.0,
    'serotonin': 0.0,
    'unclear': +1.0,     # fall back to the majority class
}

# Cell classes that make up the mushroom body pathway, as annotated in MaleCNS v1.0.
MB_CLASSES = ('olfactory', 'ALPN', 'ALLN', 'Kenyon_Cell', 'MBON', 'DAN')


@dataclass
class LIFParams:
    """Leaky integrate-and-fire parameters.

    Defaults are in the range normally used for fly central neurons. They are not
    measured values for individual cell types - the connectome gives wiring, not
    dynamics, so these are a calibration choice and are documented as such.
    """

    dt_ms: float = 0.5
    tau_m_ms: float = 20.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory_ms: float = 2.0
    # Scales raw synapse counts into membrane-voltage units. Tuned so population
    # firing rates land in a physiological range rather than saturating - see
    # doc/flybrain.md for the calibration sweep.
    synaptic_gain: float = 0.0026
    # Clamp on |weight| so a handful of very heavy edges cannot dominate.
    max_abs_weight: float = 400.0

    # Kenyon cells code odours sparsely: only a small fraction fire for any given
    # stimulus, enforced in the fly by APL-mediated global feedback inhibition onto
    # the KC population. Without this the network saturates, every poker state looks
    # identical to the MBONs, and the KC->MBON plasticity has nothing to discriminate.
    # Implemented as threshold-gated competition: of the KCs that reach threshold on a
    # step, only the most strongly driven fraction are allowed to spike.
    kc_target_sparsity: float = 0.05
    # Graded APL-style feedback inhibition onto the Kenyon cells. This is the default
    # sparsity mechanism because it keeps the KC response continuous in the input.
    kc_inhibition_beta: float = 1.0
    enforce_kc_sparsity: bool = True
    # Use the hard top-k mechanism instead of graded inhibition. Ablation only: it
    # destroys generalisation (see LIFEngine docstring).
    kc_hard_winner_take_all: bool = False


@dataclass
class EncodingParams:
    """Poker state -> olfactory receptor neuron drive."""

    # Simulated milliseconds per decision. 100ms is roughly one fly behavioural tick.
    window_ms: float = 100.0
    # Peak ORN firing rate in Hz at full feature activation.
    max_rate_hz: float = 180.0
    baseline_rate_hz: float = 2.0
    # The rate vector is rescaled so its mean equals this. It decouples *how much*
    # the antennal lobe drives the mushroom body - which sets KC sparsity and is fixed
    # once by calibration - from *which pattern* it sends, which is what the table
    # state controls. Without it, adding a feature changes total drive and silently
    # moves the operating point.
    target_mean_rate_hz: float = 25.0
    # How sensory drive enters the network.
    #
    # 'current' injects a graded current proportional to each ORN's rate. Responses are
    # bit-for-bit reproducible and strongly stimulus-dependent.
    #
    # 'poisson' draws spikes from the rate, which is more faithful at the single-spike
    # level but measured d' ~= 1.0 here at every gain and window tried: with 451k
    # recurrent edges the network sits in a chaotic regime, so two runs of the *same*
    # state diverge as much as two different states, and lengthening the window does
    # not average it out. Kept for ablations; not usable as the default.
    sensory_mode: str = 'current'
    # Glomeruli per feature. Each gets a Gaussian tuning curve over [0, 1], so
    # different feature values recruit different channels instead of the same one
    # louder - that is what keeps two table states distinguishable downstream.
    bins_per_feature: int = 5
    tuning_width: float = 0.40
    # Snap each feature to its nearest bin centre before encoding.
    #
    # This is the single most consequential choice in the module. The mushroom body is a
    # pattern separator: the Kenyon cell expansion exists to make similar inputs
    # *distinguishable* so they can be learned about independently. It is not a function
    # approximator, and it does not interpolate - measured out-of-sample R^2 for decoding
    # a continuous equity from MBON activity stayed negative across every gain,
    # inhibition and tuning setting tried, while in-sample R^2 was 0.67. The circuit
    # memorises and separates; it does not generalise between nearby values.
    #
    # So the fly is given what the literature actually tested it on: a modest set of
    # discrete cues, each of which it can learn a value for. Rajagopalan et al. (2023)
    # used discrete odour cues with probabilistic reward; Bennett et al. (2021) used a
    # discrete-cue multi-armed bandit. Quantising turns continuous equity into hand-
    # strength classes - which is, incidentally, closer to how human players think than
    # a continuous number is.
    quantize_features: bool = True
    seed: int = 1729


@dataclass
class PlasticityParams:
    """KC->MBON plasticity gated by dopaminergic reward prediction error.

    Follows the reduced mushroom body model of Bennett et al. 2021 (Nat Commun
    12:2569): DANs signal a reinforcement prediction error computed against MBON
    feedback, and depress the KC->MBON synapses that were active when reward
    arrived. PAM clusters carry appetitive signals, PPL1 clusters aversive ones.
    """

    enabled: bool = True
    learning_rate: float = 0.10
    # Eligibility trace decay over decisions (a hand spans several decisions).
    trace_decay: float = 0.6
    # Keeps weights non-negative and bounded, as synaptic weights must be.
    weight_min: float = 0.0
    weight_max: float = 4.0
    # Reward is normalised by this many big blinds before entering the DAN signal.
    reward_scale_bb: float = 10.0


@dataclass
class FlyBrainConfig:
    """Top-level configuration."""

    mode: RunMode = RunMode.shadow
    scope: Scope = Scope.mushroom_body

    # Connectome selection
    weight_threshold: int = 3
    use_synthetic: bool = False
    cache_dir: str = ''

    # Hard guard: refuse to drive the mouse unless the operator has declared the
    # table is play money. See decision.PlayMoneyGuard.
    play_money_only: bool = True
    confirmed_play_money: bool = False

    lif: LIFParams = field(default_factory=LIFParams)
    encoding: EncodingParams = field(default_factory=EncodingParams)
    plasticity: PlasticityParams = field(default_factory=PlasticityParams)

    # Reproducibility
    seed: int = 20260903  # MaleCNS v1.0 release date

    def as_dict(self):
        """Return a JSON-serialisable view, for logging alongside each decision."""
        out = asdict(self)
        out['mode'] = self.mode.value
        out['scope'] = self.scope.value
        return out

    @classmethod
    def from_config_parser(cls, config):
        """Build from the bot's config.ini. Missing keys fall back to the defaults."""
        cfg = cls()
        if config is None or not config.has_section('flybrain'):
            log.info("No [flybrain] section in config.ini - using defaults (mode=shadow)")
            return cfg

        def _get(key, cast, current):
            if not config.has_option('flybrain', key):
                return current
            raw = config.get('flybrain', key)
            try:
                if cast is bool:
                    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')
                return cast(raw)
            except (TypeError, ValueError):
                log.warning("Bad value for flybrain.%s=%r - keeping %r", key, raw, current)
                return current

        cfg.mode = RunMode(_get('mode', str, cfg.mode.value))
        cfg.scope = Scope(_get('scope', str, cfg.scope.value))
        cfg.weight_threshold = _get('weight_threshold', int, cfg.weight_threshold)
        cfg.use_synthetic = _get('use_synthetic', bool, cfg.use_synthetic)
        cfg.cache_dir = _get('cache_dir', str, cfg.cache_dir)
        cfg.play_money_only = _get('play_money_only', bool, cfg.play_money_only)
        cfg.confirmed_play_money = _get('confirmed_play_money', bool, cfg.confirmed_play_money)
        cfg.plasticity.enabled = _get('plasticity', bool, cfg.plasticity.enabled)
        cfg.plasticity.learning_rate = _get('learning_rate', float, cfg.plasticity.learning_rate)
        cfg.seed = _get('seed', int, cfg.seed)
        return cfg
