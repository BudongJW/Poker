"""Poker table state -> olfactory receptor neuron drive.

The fly has no circuit for reading cards, sizing a pot, or reasoning about position.
What it has is an olfactory front end feeding a mushroom body that learns which
stimulus predicts reward. So the table state is presented to it as an odour: each
poker feature drives a distinct glomerulus, at a firing rate proportional to the
feature's value.

This means the fly is NOT computing equity. The existing Monte Carlo does that, and
the result is handed to the fly as a sensory channel. The fly's job is the part the
literature says a fly can actually do: value-based choice given that signal. Stating
this plainly matters, because the viral fly-brain demos mostly blur it - an external
learner does the work while the connectome supplies a fixed scaffold.

Each feature is given a small bank of glomeruli with overlapping Gaussian tuning
curves across [0, 1], so a different feature value lights up a different subset rather
than the same channel louder. Dense coding - every feature driving one channel in
proportion to its value - leaves two different table states with nearly identical total
drive, which saturates the Kenyon cells and collapses discriminability. Tuning curves
are also what olfactory receptor neurons actually do.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# Feature channels, in a fixed order so an encoding is reproducible across runs.
FEATURES = (
    'equity',             # Monte Carlo win probability
    'relative_equity',    # equity against the assumed opponent range
    'pot_odds',           # minCall / (pot + minCall)
    'stack_to_pot',       # my stack relative to the pot, squashed
    'round_pot_bb',       # this round's pot in big blinds, squashed
    'opponents',          # active opponents, squashed
    'players_ahead',      # opponents still to act, squashed
    'initiative',         # an opponent has the betting initiative
    'street_preflop',
    'street_flop',
    'street_turn',
    'street_river',
    'can_check',          # the check button is available
    'facing_allin',       # an all-in call is on offer
)


def _get(table, name, default=None):
    """Read an attribute the scraper may not have populated."""
    return getattr(table, name, default)


def _squash(value, scale):
    """Map a non-negative unbounded quantity into [0, 1] smoothly."""
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value) or value < 0:
        return 0.0
    return float(value / (value + scale))


def _unit(value):
    """Clamp a probability-like value into [0, 1]."""
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def features_from_table(table, history=None):
    """Extract the feature vector from the bot's scraped table object.

    Every field is read defensively: the scraper does not always populate everything,
    and a missing feature should read as zero rather than crash a live hand.
    """

    min_call = float(_get(table, 'minCall') or 0.0)
    total_pot = float(_get(table, 'totalPotValue') or 0.0)
    big_blind = float(_get(table, 'bigBlind') or 0.0) or 1.0
    stage = str(_get(table, 'gameStage') or '')

    equity = _get(table, 'abs_equity')
    if equity is None:
        equity = _get(table, 'equity')

    rel = _get(table, 'relative_equity')
    if rel in (None, 0, 0.0) or stage == 'PreFlop':
        rel = equity

    pot_odds = min_call / (total_pot + min_call) if (total_pot + min_call) > 0 else 0.0

    values = {
        'equity': _unit(equity),
        'relative_equity': _unit(rel),
        'pot_odds': _unit(pot_odds),
        'stack_to_pot': _squash(_get(table, 'myFunds'), scale=max(total_pot, big_blind)),
        'round_pot_bb': _squash(float(_get(table, 'round_pot_value') or 0.0) / big_blind, scale=10.0),
        'opponents': _squash(_get(table, 'assumedPlayers'), scale=3.0),
        'players_ahead': _squash(_get(table, 'playersAhead'), scale=3.0),
        'initiative': 1.0 if _get(table, 'other_player_has_initiative') else 0.0,
        'street_preflop': 1.0 if stage == 'PreFlop' else 0.0,
        'street_flop': 1.0 if stage == 'Flop' else 0.0,
        'street_turn': 1.0 if stage == 'Turn' else 0.0,
        'street_river': 1.0 if stage == 'River' else 0.0,
        'can_check': 1.0 if _get(table, 'checkButton') else 0.0,
        'facing_allin': 1.0 if _get(table, 'allInCallButton') else 0.0,
    }
    del history
    return np.asarray([values[name] for name in FEATURES], dtype=np.float32)


class TableView:
    """The attribute surface features_from_table() and legal_actions() read.

    Offline tasks present their state by filling this in, so a synthetic hand and a
    scraped hand reach the encoder through one code path rather than two that could drift
    apart. Named for what it is - a view of a table - not a fake one, because the live
    scraper object satisfies the same surface without inheriting from this.
    """

    # pylint: disable=too-many-instance-attributes,invalid-name,too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def __init__(self, equity, pot, to_call, stack, stage, opponents=1,
                 big_blind=1.0, min_bet=None):
        self.abs_equity = equity
        self.equity = equity
        self.relative_equity = equity
        self.totalPotValue = pot
        self.round_pot_value = pot
        self.minCall = to_call
        self.minBet = min_bet if min_bet is not None else max(big_blind, pot * 0.25)
        self.myFunds = stack
        self.bigBlind = big_blind
        self.smallBlind = big_blind / 2.0
        self.gameStage = stage
        self.assumedPlayers = opponents + 1
        self.playersAhead = opponents
        self.other_player_has_initiative = to_call > 0
        self.checkButton = to_call == 0
        self.callButton = to_call > 0
        self.betButton = True
        self.allInCallButton = False


class OdourEncoder:
    """Assigns poker features to real glomeruli and produces ORN firing rates.

    Sensory neurons are grouped by their MaleCNS cell type (ORN_DA1, ORN_VA1d, ...),
    which is the anatomical glomerulus they project to. Features are dealt out across
    those groups so that each feature owns its own channel.
    """

    def __init__(self, circuit, params):
        self.params = params
        self.sensory = circuit.sensory
        types = circuit.connectome.types[self.sensory]
        self.groups = self._group_by_type(types)
        self.channel_map = self._assign_channels()
        log.info("Odour encoder: %d features over %d glomerulus groups (%d ORNs)",
                 len(FEATURES), len(self.groups), len(self.sensory))

    def _group_by_type(self, types):
        """Return a list of index arrays (into self.sensory), one per cell type."""
        buckets = {}
        for position, cell_type in enumerate(types):
            key = cell_type if isinstance(cell_type, str) and cell_type else 'unknown'
            buckets.setdefault(key, []).append(position)
        # Sorted so the mapping is deterministic run to run.
        return [np.asarray(buckets[key], dtype=np.int64) for key in sorted(buckets)]

    def _assign_channels(self):
        """Give each feature its own disjoint bank of glomerulus groups.

        Returns {feature_name: (group_indices, centres)} where centres[i] is the
        feature value that maximally drives group_indices[i].

        The banks must not overlap. Handing every feature a fixed bins_per_feature and
        wrapping with modulo silently aliases features onto each other once
        len(FEATURES) * bins exceeds the number of groups - with 14 features, 5 bins and
        the 54 ORN types in MaleCNS v1.0 the last three features overwrote the first
        three, and two unrelated equities produced identical codes. So the available
        groups are partitioned instead, and bins_per_feature is treated as a maximum.
        """
        n_groups = len(self.groups)
        n_features = len(FEATURES)
        if n_groups < 2 * n_features:
            raise ValueError(
                f"Need at least {2 * n_features} glomerulus groups to give each of "
                f"{n_features} features its own bank, found {n_groups}. The "
                f"connectome's sensory neurons have no usable type annotations."
            )

        per_feature = min(self.params.bins_per_feature, n_groups // n_features)
        mapping = {}
        cursor = 0
        for name in FEATURES:
            groups = list(range(cursor, cursor + per_feature))
            cursor += per_feature
            centres = np.linspace(0.0, 1.0, per_feature, dtype=np.float32)
            mapping[name] = (np.asarray(groups, dtype=np.int64), centres)

        assigned = [g for groups, _ in mapping.values() for g in groups.tolist()]
        assert len(assigned) == len(set(assigned)), "glomerulus banks overlap"
        log.info("Encoder: %d features x %d bins over %d disjoint glomerulus groups",
                 n_features, per_feature, len(assigned))
        return mapping

    def rates(self, feature_vector):
        """Return an ORN firing-rate array aligned with circuit.sensory."""
        params = self.params
        rates = np.full(len(self.sensory), params.baseline_rate_hz, dtype=np.float32)
        span = params.max_rate_hz - params.baseline_rate_hz
        width = params.tuning_width

        for value, name in zip(feature_vector, FEATURES):
            groups, centres = self.channel_map[name]
            value = float(np.clip(value, 0.0, 1.0))
            if params.quantize_features:
                # Snap to the nearest tuning centre so every value inside a bin
                # produces an identical code the circuit can attach a value to.
                value = float(centres[int(np.argmin(np.abs(centres - value)))])
            # Gaussian tuning: the closer a glomerulus' preferred value, the harder it
            # is driven. Normalised so every feature contributes the same total drive
            # regardless of where its value falls.
            response = np.exp(-0.5 * ((value - centres) / width) ** 2)
            total = response.sum()
            if total > 0:
                response = response / total
            for group_index, strength in zip(groups, response):
                rates[self.groups[group_index]] += span * float(strength)

        # Hold total drive constant so only the pattern carries the state.
        mean = float(rates.mean())
        if mean > 0 and params.target_mean_rate_hz > 0:
            rates *= params.target_mean_rate_hz / mean
        np.clip(rates, 0.0, params.max_rate_hz, out=rates)
        return rates

    def encode_table(self, table, history=None):
        """Convenience: scraped table -> (feature_vector, orn_rates)."""
        features = features_from_table(table, history)
        return features, self.rates(features)
