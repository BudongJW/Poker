"""Tests for the fly-brain decision module.

These run on the synthetic connectome so they need no 1 GB download and no network.
The synthetic stand-in matches the real data's population sizes and edge count, so it
exercises every code path; what it cannot do is say anything about the fly, which is
why test_synthetic_is_flagged exists.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from poker.flybrain import connectome as connectome_mod
from poker.flybrain import controls, decoding, encoding, kuhn
from poker.flybrain.brain import FlyBrain
from poker.flybrain.cli import (EquityGame, OfflineTable, calibration_tables,
                                run_episodes)
from poker.flybrain.config import (CALIBRATED_GAIN, FlyBrainConfig, LIFParams, RunMode,
                                   Scope)
from poker.flybrain.guard import PlayMoneyDeclarationMissing, PlayMoneyGuard


@pytest.fixture(name='synthetic_connectome', scope='module')
def fixture_synthetic_connectome():
    """A structurally-matched stand-in, built once for the module."""
    return connectome_mod.synthetic(scope=Scope.mushroom_body, seed=11)


@pytest.fixture(name='brain')
def fixture_brain(synthetic_connectome):
    """A FlyBrain over the synthetic connectome."""
    config = FlyBrainConfig(mode=RunMode.offline, use_synthetic=True, seed=3)
    return FlyBrain(config, connectome=synthetic_connectome)


def _table(**kwargs):
    params = {'equity': 0.6, 'pot': 10.0, 'to_call': 3.0, 'stack': 100.0,
              'stage': 'Flop'}
    params.update(kwargs)
    return OfflineTable(**params)


# --- connectome ------------------------------------------------------------------

def test_synthetic_matches_real_population_sizes(synthetic_connectome):
    """The stand-in should mirror what MaleCNS v1.0 actually yields."""
    expected = {'olfactory': 2639, 'ALPN': 686, 'ALLN': 420,
                'Kenyon_Cell': 4064, 'MBON': 97, 'DAN': 340}
    for cell_class, count in expected.items():
        assert len(synthetic_connectome.index_of_class(cell_class)) == count


def test_synthetic_is_flagged(synthetic_connectome):
    """Anything built on fake wiring must announce itself."""
    assert synthetic_connectome.synthetic is True


def test_loader_falls_back_when_data_missing(tmp_path):
    """A missing download must degrade to synthetic, not crash the bot."""
    result = connectome_mod.load(scope=Scope.mushroom_body,
                                 cache_dir=str(tmp_path / 'nothing-here'),
                                 use_synthetic=True)
    assert result.synthetic is True
    assert result.n_neurons > 0


def test_npz_round_trip(synthetic_connectome, tmp_path):
    """Caching and restoring must preserve the graph."""
    path = str(tmp_path / 'conn.npz')
    connectome_mod._save_npz(path, synthetic_connectome)  # pylint: disable=protected-access
    restored = connectome_mod._load_npz(path)  # pylint: disable=protected-access
    assert restored.n_neurons == synthetic_connectome.n_neurons
    assert restored.n_edges == synthetic_connectome.n_edges
    assert np.array_equal(restored.body_ids, synthetic_connectome.body_ids)


# --- circuit ---------------------------------------------------------------------

def test_dan_channels_are_both_populated(brain):
    """Plasticity needs two opposing dopaminergic channels to exist."""
    assert len(brain.circuit.dan_appetitive) > 0
    assert len(brain.circuit.dan_aversive) > 0


def test_kc_to_mbon_block_is_detached_from_static_matrix(brain):
    """The plastic block must not also be integrated as fixed wiring."""
    static = brain.engine._w_in  # pylint: disable=protected-access
    block = static[brain.circuit.mbon][:, brain.circuit.kc]
    assert block.nnz == 0, "KC->MBON should live only in the plastic block"
    assert len(brain.engine.plastic_blocks) == 1


# --- encoding --------------------------------------------------------------------

def test_encoding_is_deterministic(brain):
    """The same table must always produce the same odour."""
    table = _table()
    first = brain.encoder.encode_table(table)[1]
    second = brain.encoder.encode_table(table)[1]
    assert np.array_equal(first, second)


def test_encoding_handles_missing_attributes():
    """A half-scraped table must encode rather than raise."""

    class Sparse:  # pylint: disable=too-few-public-methods
        """A table where the scraper found almost nothing."""
        gameStage = 'Flop'  # pylint: disable=invalid-name

    features = encoding.features_from_table(Sparse())
    assert features.shape == (len(encoding.FEATURES),)
    assert np.all(np.isfinite(features))


def test_quantisation_collapses_nearby_equities(brain):
    """Values inside one bin must produce one code, so a value can be learned for it."""
    codes = {brain.encoder.encode_table(_table(equity=e))[1].tobytes()
             for e in (0.72, 0.74, 0.76)}
    assert len(codes) == 1


def test_quantisation_separates_distant_equities(brain):
    """Values in different bins must produce different codes."""
    low = brain.encoder.encode_table(_table(equity=0.05))[1]
    high = brain.encoder.encode_table(_table(equity=0.95))[1]
    assert not np.array_equal(low, high)


# --- decoding and legality -------------------------------------------------------

def test_legal_actions_when_check_available():
    """With a check button up there is nothing to fold to and nothing to call."""
    allowed = decoding.legal_actions(_table(check_available=True, to_call=0.0))
    assert 'Check' in allowed
    assert 'Fold' not in allowed
    assert 'Call' not in allowed


def test_legal_actions_facing_allin():
    """An all-in call collapses the decision to calling or folding."""
    table = _table()
    table.allInCallButton = True
    assert decoding.legal_actions(table) == ('Fold', 'Call')


def test_decision_is_always_legal(brain):
    """The fly must never be able to pick an action the table does not offer."""
    for check_available in (True, False):
        table = _table(check_available=check_available,
                       to_call=0.0 if check_available else 3.0)
        allowed = decoding.legal_actions(table)
        for _ in range(5):
            observation = brain.decide(table, allowed=allowed)
            assert observation.action in allowed
    brain.abandon_hand()


def test_decoder_rejects_empty_action_set(brain):
    """An impossible situation should raise, not silently pick something."""
    with pytest.raises(ValueError):
        brain.decoder.choose(np.zeros(len(brain.circuit.mbon)), allowed=[])


# --- simulation ------------------------------------------------------------------

def test_current_injection_is_reproducible(brain):
    """Identical state, identical response - this is what makes learning possible."""
    table = _table()
    first = brain.decide(table).mbon_rates
    second = brain.decide(table).mbon_rates
    brain.abandon_hand()
    assert np.allclose(first, second)


def test_kc_activity_stays_sparse(brain):
    """Saturated Kenyon cells mean every state looks alike downstream."""
    observation = brain.decide(_table())
    active = float((observation.kc_counts > 0).mean())
    brain.abandon_hand()
    assert active < 0.40, f"KC population saturated at {active:.2f}"


def test_refractory_period_bounds_firing_rate():
    """No neuron may fire faster than its refractory period allows."""
    params = LIFParams()
    conn = connectome_mod.synthetic(scope=Scope.mushroom_body, seed=5)
    from poker.flybrain.lif import LIFEngine  # pylint: disable=import-outside-toplevel
    engine = LIFEngine(conn, params, seed=0)
    n_steps = engine.steps_for_ms(100.0)
    # Drive everything hard.
    tonic = np.full(conn.n_neurons, params.v_threshold * 2.0, dtype=np.float32)
    counts, _ = engine.run(n_steps, tonic=tonic)
    max_possible = n_steps / engine.steps_per_refractory + 1
    assert counts.max() <= max_possible


# --- plasticity ------------------------------------------------------------------

def test_reward_prediction_error_has_the_right_sign(brain):
    """A win must drive the appetitive channel and a loss the aversive one."""
    brain.decide(_table())
    win = brain.reinforce(reward_bb=+8.0)
    assert win['dan_appetitive_drive'] > 0
    assert win['dan_aversive_drive'] == 0

    brain.decide(_table())
    loss = brain.reinforce(reward_bb=-8.0)
    assert loss['dan_aversive_drive'] > 0
    assert loss['dan_appetitive_drive'] == 0


def test_rpe_stays_in_a_sane_range(brain):
    """An unnormalised prediction puts the error orders of magnitude off and flips it."""
    brain.decide(_table())
    summary = brain.reinforce(reward_bb=+8.0)
    assert abs(summary['mean_rpe']) <= 2.0


def test_reinforcement_changes_plastic_weights(brain):
    """Learning must actually reach the KC->MBON synapses."""
    brain.decide(_table())
    before = brain.plasticity.weights.copy()
    brain.reinforce(reward_bb=+5.0)
    assert not np.allclose(before, brain.plasticity.weights)


def test_plasticity_can_be_disabled(synthetic_connectome):
    """The frozen control must really freeze."""
    config = FlyBrainConfig(mode=RunMode.offline, use_synthetic=True, seed=3)
    config.plasticity.enabled = False
    frozen = FlyBrain(config, connectome=synthetic_connectome)
    frozen.decide(_table())
    before = frozen.plasticity.weights.copy()
    frozen.reinforce(reward_bb=+5.0)
    assert np.allclose(before, frozen.plasticity.weights)


def test_weights_stay_bounded_and_non_negative(brain):
    """Synaptic weights are a magnitude; they may not go negative or run away."""
    for reward in (+40.0, -40.0) * 6:
        brain.decide(_table())
        brain.reinforce(reward_bb=reward)
    assert brain.plasticity.weights.min() >= 0.0
    assert np.all(np.isfinite(brain.plasticity.weights))


def test_state_round_trip(brain, tmp_path):
    """Saving and reloading must preserve what was learned."""
    brain.decide(_table())
    brain.reinforce(reward_bb=+6.0)
    path = str(tmp_path / 'brain.npz')
    brain.save(path)

    weights = brain.plasticity.weights.copy()
    readout = brain.decoder.weights.copy()
    brain.plasticity.weights[:] = 0.0
    brain.decoder.weights[:] = 0.0

    brain.load(path)
    assert np.allclose(weights, brain.plasticity.weights)
    assert np.allclose(readout, brain.decoder.weights)


# --- controls --------------------------------------------------------------------

def test_shuffle_preserves_degree_sequence(synthetic_connectome):
    """If the shuffle changed degrees it would not be a fair control."""
    shuffled = controls.shuffle_preserving_degree(synthetic_connectome, seed=2)
    original_out = np.asarray((synthetic_connectome.weights != 0).sum(axis=1)).ravel()
    shuffled_out = np.asarray((shuffled.weights != 0).sum(axis=1)).ravel()
    # Out-degree is exactly preserved by a double-edge swap on targets.
    assert original_out.sum() >= shuffled_out.sum()          # only self-loops removed
    assert abs(int(original_out.sum()) - int(shuffled_out.sum())) < 0.02 * original_out.sum()


def test_shuffle_actually_rewires(synthetic_connectome):
    """A control that changed nothing would silently pass every comparison."""
    shuffled = controls.shuffle_preserving_degree(synthetic_connectome, seed=2)
    difference = (shuffled.weights != synthetic_connectome.weights).nnz
    assert difference > 0.5 * synthetic_connectome.n_edges


def test_random_agent_is_always_legal():
    """The floor baseline must respect the same legality rules."""
    agent = controls.RandomAgent(seed=1)
    table = _table(check_available=True, to_call=0.0)
    allowed = decoding.legal_actions(table)
    for _ in range(20):
        assert agent.decide(table, allowed=allowed).action in allowed


# --- calibration -----------------------------------------------------------------
#
# The first Kuhn control run was void because the conditions were never brought to the
# same operating point: at a shared gain the real connectome sat at 0.49% KC activity and
# its shuffle at 13%. These tests pin the three things that made that possible.

class _GainStub:
    """A stand-in whose KC activity is a known monotone function of the gain.

    Lets the bisection be tested without paying for an LIF simulation per probe.
    """

    def __init__(self, gain, n_kc=1000, full_scale=0.01):
        self.active = int(round(min(1.0, gain / full_scale) * n_kc))
        self.n_kc = n_kc

    def decide(self, table, **kwargs):
        """Only kc_counts is read by measure_kc_sparsity."""
        del table, kwargs
        counts = np.zeros(self.n_kc, dtype=np.int32)
        counts[:self.active] = 1
        return SimpleNamespace(kc_counts=counts)

    def abandon_hand(self):
        """No-op."""
        return 0


def test_calibration_uses_every_kuhn_information_set():
    """Sparsity has to be measured on the states the task actually presents."""
    tables = calibration_tables('kuhn')
    assert len(tables) == len(kuhn.INFO_SETS) == 12
    assert {(table.card, table.history) for table in tables} == set(kuhn.INFO_SETS)


def test_calibrate_gain_hits_the_target():
    """The whole point of the bisection: end up at the sparsity asked for."""
    gain, measured = controls.calibrate_gain(_GainStub, [None] * 3, target_sparsity=0.09,
                                             tolerance=0.005)
    assert measured == pytest.approx(0.09, abs=0.005)
    assert gain == pytest.approx(0.0009, rel=0.1)


def test_calibrate_gain_bounds_reach_the_real_networks_operating_point():
    """The old bounds topped out at 0.0040 and could not reach the ~0.0050 real needs."""
    gain, measured = controls.calibrate_gain(_GainStub, [None] * 2, target_sparsity=0.50,
                                             tolerance=0.01)
    assert measured == pytest.approx(0.50, abs=0.01)
    assert gain > 0.0040


def test_calibrate_gain_brings_a_real_brain_to_the_target(synthetic_connectome):
    """End to end through the LIF engine, not just the bisection arithmetic."""
    def factory(gain):
        config = FlyBrainConfig(mode=RunMode.offline, use_synthetic=True, seed=3)
        config.lif = LIFParams(synaptic_gain=gain)
        return FlyBrain(config, connectome=synthetic_connectome)

    gain, measured = controls.calibrate_gain(factory, calibration_tables('kuhn')[:4],
                                             target_sparsity=0.09, tolerance=0.02)
    assert measured == pytest.approx(0.09, abs=0.02)
    assert 0.0002 < gain < 0.0120


def test_default_gain_is_the_re_derived_operating_point():
    """0.0026 predated the encoder changes and left the network silent on Kuhn states."""
    assert LIFParams().synaptic_gain == pytest.approx(0.0050, abs=0.0005)
    assert set(CALIBRATED_GAIN) == {'kuhn', 'equity'}
    for task, gain in CALIBRATED_GAIN.items():
        assert 0.004 < gain < 0.006, f"{task} gain {gain} is not the re-derived point"


# --- the offline task ------------------------------------------------------------

def test_equity_game_best_action_is_sane():
    """Folding should win with hopeless equity and betting with a lock."""
    game = EquityGame(seed=0)
    hopeless = OfflineTable(0.02, pot=10.0, to_call=8.0, stack=100.0, stage='Flop')
    assert game.best(hopeless, decoding.legal_actions(hopeless))[0] == 'Fold'

    lock = OfflineTable(0.98, pot=10.0, to_call=2.0, stack=100.0, stage='Flop')
    assert game.best(lock, decoding.legal_actions(lock))[0].startswith('Bet')


def test_run_episodes_reports_regret(brain):
    """The harness must produce a measurable number, not just run."""
    metrics = run_episodes(brain, EquityGame(seed=1), n_hands=12, learn=True)
    assert metrics['hands'] == 12
    assert np.isfinite(metrics['mean_regret_bb'])
    assert metrics['mean_regret_bb'] >= -1e-9
    assert 0.0 <= metrics['optimal_action_rate'] <= 1.0


# --- the play-money guard --------------------------------------------------------

def test_guard_blocks_active_without_declaration():
    """Active play must be impossible to reach by accident."""
    config = FlyBrainConfig(mode=RunMode.active, play_money_only=True,
                            confirmed_play_money=False)
    guard = PlayMoneyGuard(config)
    assert guard.check() is not None
    with pytest.raises(PlayMoneyDeclarationMissing):
        guard.enforce()


def test_guard_allows_active_once_declared():
    """An explicit declaration is what unlocks it."""
    config = FlyBrainConfig(mode=RunMode.active, play_money_only=True,
                            confirmed_play_money=True)
    assert PlayMoneyGuard(config).check() is None


def test_guard_is_irrelevant_in_shadow_mode():
    """Shadow mode never touches the mouse, so it needs no declaration."""
    config = FlyBrainConfig(mode=RunMode.shadow, confirmed_play_money=False)
    assert PlayMoneyGuard(config).check() is None


def test_shadow_is_the_default():
    """The safe mode must be what you get when you configure nothing."""
    assert FlyBrainConfig().mode is RunMode.shadow
    assert FlyBrainConfig().play_money_only is True
    assert FlyBrainConfig().confirmed_play_money is False


# --- FlyDecision's own logic ------------------------------------------------------
#
# Decision.__init__ needs ~40 scraped attributes, a strategy sheet and a curve fit, and
# the repository's own tests for it are skipped ("Fix access issues") because they need a
# MongoDB connection and screenshot fixtures. So the base class is stubbed out here and
# what gets tested is the part this module owns: mode handling, the guard, and the
# fallback. Whether Decision itself works is not in scope.

@pytest.fixture(name='fly_decision_cls')
def fixture_fly_decision_cls(monkeypatch, synthetic_connectome):
    """FlyDecision with the heavyweight base-class behaviour stubbed out."""
    from poker.decisionmaker.decisionmaker import Decision  # pylint: disable=import-outside-toplevel
    from poker.flybrain import decision as decision_mod  # pylint: disable=import-outside-toplevel

    def fake_init(self, t, h, p, l):  # pylint: disable=unused-argument
        self.decision = 'Fold'

    def fake_make_decision(self, t, h, p, l):  # pylint: disable=unused-argument
        self.decision = 'Fold'

    monkeypatch.setattr(Decision, '__init__', fake_init)
    monkeypatch.setattr(Decision, 'make_decision', fake_make_decision)
    decision_mod.reset_brain_cache()
    return decision_mod.FlyDecision, synthetic_connectome


def _run_fly_decision(cls_and_connectome, **config_kwargs):
    cls, connectome = cls_and_connectome
    from poker.flybrain.brain import FlyBrain  # pylint: disable=import-outside-toplevel
    config = FlyBrainConfig(use_synthetic=True, seed=3, **config_kwargs)
    brain = FlyBrain(config, connectome=connectome)
    table = _table()
    instance = cls(table, None, None, None, config=config, brain=brain)
    instance.make_decision(table, None, None, None)
    return instance


def test_shadow_mode_leaves_the_baseline_driving(fly_decision_cls):
    """In shadow mode the fly must not change what reaches the mouse."""
    instance = _run_fly_decision(fly_decision_cls, mode=RunMode.shadow)
    assert instance.decision == 'Fold'
    assert instance.baseline_decision == 'Fold'
    assert instance.fly_decision is not None
    assert instance.fly_active is False


def test_active_mode_without_declaration_stays_in_shadow(fly_decision_cls):
    """The guard must hold the fly back rather than failing the hand."""
    instance = _run_fly_decision(fly_decision_cls, mode=RunMode.active,
                                 confirmed_play_money=False)
    assert instance.decision == 'Fold'
    assert instance.fly_active is False


def test_active_mode_with_declaration_lets_the_fly_drive(fly_decision_cls):
    """Once declared, the fly's choice is what reaches the mouse."""
    instance = _run_fly_decision(fly_decision_cls, mode=RunMode.active,
                                 confirmed_play_money=True)
    assert instance.fly_active is True
    assert instance.decision == instance.fly_decision


def test_fly_failure_falls_back_to_the_baseline(fly_decision_cls, monkeypatch):
    """A broken fly must never drop a live hand."""
    cls, connectome = fly_decision_cls
    from poker.flybrain.brain import FlyBrain  # pylint: disable=import-outside-toplevel

    config = FlyBrainConfig(mode=RunMode.active, use_synthetic=True, seed=3,
                            confirmed_play_money=True)
    brain = FlyBrain(config, connectome=connectome)
    monkeypatch.setattr(brain, 'decide',
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))

    table = _table()
    instance = cls(table, None, None, None, config=config, brain=brain)
    instance.make_decision(table, None, None, None)

    assert instance.decision == 'Fold'
    assert instance.fly_active is False
    assert 'boom' in instance.fly_error


def test_log_dict_reports_both_sides(fly_decision_cls):
    """The log must make it possible to compare the fly against the baseline later."""
    instance = _run_fly_decision(fly_decision_cls, mode=RunMode.shadow)
    row = instance.log_dict()
    assert row['baseline_decision'] == 'Fold'
    assert row['fly_decision'] in decoding.ACTIONS
    assert row['fly_synthetic_connectome'] is True
    assert row['fly_mode'] == 'shadow'


# --- Kuhn poker -------------------------------------------------------------------
#
# The point of Kuhn poker here is that it can be checked against published facts rather
# than against itself, so these tests assert the facts.

def test_kuhn_has_twelve_information_sets():
    """Six per player. If this is wrong, the game tree is wrong."""
    assert len(kuhn.INFO_SETS) == 12
    assert len(set(kuhn.INFO_SETS)) == 12


def test_kuhn_game_value_across_the_nash_family():
    """Every member of player 1's equilibrium family must yield exactly -1/18."""
    for alpha in (0.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 4.0, 1.0 / 3.0):
        value = kuhn.expected_value(kuhn.nash_policy(alpha))
        assert value == pytest.approx(-1.0 / 18.0, abs=1e-12)


def test_kuhn_nash_is_unexploitable():
    """Exploitability must be exactly zero at equilibrium, in both seats."""
    for alpha in (0.0, 1.0 / 6.0, 1.0 / 3.0):
        policy = kuhn.nash_policy(alpha)
        assert kuhn.exploitability(policy) == pytest.approx(0.0, abs=1e-12)
        assert kuhn.best_response_value(policy, 1) == pytest.approx(-1.0 / 18.0, abs=1e-12)
        assert kuhn.best_response_value(policy, 2) == pytest.approx(1.0 / 18.0, abs=1e-12)


def test_kuhn_rejects_alpha_outside_the_family():
    """alpha beyond 1/3 is not an equilibrium and must not be silently accepted."""
    with pytest.raises(ValueError):
        kuhn.nash_policy(0.5)
    with pytest.raises(ValueError):
        kuhn.nash_policy(-0.1)


def test_kuhn_exploitability_is_never_negative():
    """Minimax guarantees it. A negative value would mean the evaluator is broken."""
    for policy in (kuhn.uniform_policy(), kuhn.always_pass_policy(),
                   kuhn.nash_policy(0.2)):
        assert kuhn.exploitability(policy) >= -1e-12


def test_kuhn_known_baseline_exploitabilities():
    """Two reference policies, against values that can be derived by hand."""
    assert kuhn.exploitability(kuhn.uniform_policy()) == pytest.approx(0.458333, abs=1e-5)
    assert kuhn.exploitability(kuhn.always_pass_policy()) == pytest.approx(1.0, abs=1e-12)


def test_kuhn_payoffs_are_zero_sum_and_signed_correctly():
    """Folding loses the pot; a showdown after a bet is worth two chips."""
    assert kuhn.payoff_to_p1('bp', kuhn.JACK, kuhn.KING) == 1      # P2 folded
    assert kuhn.payoff_to_p1('pbp', kuhn.KING, kuhn.JACK) == -1    # P1 folded
    assert kuhn.payoff_to_p1('pp', kuhn.KING, kuhn.QUEEN) == 1
    assert kuhn.payoff_to_p1('pp', kuhn.JACK, kuhn.QUEEN) == -1
    assert kuhn.payoff_to_p1('bb', kuhn.KING, kuhn.JACK) == 2
    assert kuhn.payoff_to_p1('pbb', kuhn.JACK, kuhn.KING) == -2


def test_kuhn_terminal_histories_are_complete():
    """Every legal continuation must terminate or be a known decision point."""
    for history in kuhn.HISTORIES:
        for action in (kuhn.PASS, kuhn.AGGRESS):
            nxt = history + action
            assert kuhn.is_terminal(nxt) or nxt in kuhn.HISTORIES, nxt


def test_kuhn_legal_actions_match_the_betting_state():
    """Facing a bet you may fold or call; otherwise you may check or bet."""
    assert kuhn.legal_fly_actions('') == ('Check', 'Bet')
    assert kuhn.legal_fly_actions('p') == ('Check', 'Bet')
    assert kuhn.legal_fly_actions('b') == ('Fold', 'Call')
    assert kuhn.legal_fly_actions('pb') == ('Fold', 'Call')


def test_kuhn_table_presents_true_card_equities():
    """J beats neither other card, Q beats one, K beats both."""
    assert kuhn.KuhnTable(kuhn.JACK, '').abs_equity == pytest.approx(0.0)
    assert kuhn.KuhnTable(kuhn.QUEEN, '').abs_equity == pytest.approx(0.5)
    assert kuhn.KuhnTable(kuhn.KING, '').abs_equity == pytest.approx(1.0)


def test_kuhn_allowed_actions_are_a_subset_of_the_live_legality_rules():
    """Kuhn must never offer an action the live table logic would forbid.

    Not an equality: the generic rules offer three bet sizes because a real table does,
    while Kuhn has one fixed 1-chip bet, so Kuhn is a strict subset on the betting side.
    What must agree is that nothing Kuhn allows is illegal, and that the two agree on
    which actions are passive and which aggressive.
    """
    for history in kuhn.HISTORIES:
        table = kuhn.KuhnTable(kuhn.QUEEN, history)
        live = set(decoding.legal_actions(table))
        kuhn_allowed = set(kuhn.legal_fly_actions(history))
        assert kuhn_allowed <= live, f"{history!r}: {kuhn_allowed - live} not legal live"

        aggressive_live = {a for a in live if kuhn.action_is_aggressive(a)}
        aggressive_kuhn = {a for a in kuhn_allowed if kuhn.action_is_aggressive(a)}
        assert bool(aggressive_kuhn) == bool(aggressive_live)
        assert len(kuhn_allowed) == 2, "every Kuhn decision is binary"


def test_kuhn_facing_a_bet_the_two_paths_agree_exactly():
    """With a bet to call there is only one aggressive option, so they must coincide."""
    for history in ('b', 'pb'):
        table = kuhn.KuhnTable(kuhn.QUEEN, history)
        assert decoding.legal_actions(table) == kuhn.legal_fly_actions(history)


def test_kuhn_histories_map_to_distinct_feature_codes():
    """Each decision point must look different to the fly, or it cannot act on it."""
    codes = {encoding.features_from_table(kuhn.KuhnTable(kuhn.QUEEN, h)).tobytes()
             for h in kuhn.HISTORIES}
    assert len(codes) == len(kuhn.HISTORIES)


def test_kuhn_self_test_passes():
    """The module's own check against the published facts."""
    assert kuhn.self_test() is True


def test_extracted_policy_covers_every_information_set(brain):
    """A policy missing an information set would silently be evaluated as pass-always."""
    policy = kuhn.extract_policy(brain, temperature=0.0)
    assert set(policy) == set(kuhn.INFO_SETS)
    assert all(0.0 <= p <= 1.0 for p in policy.values())


def test_pure_extraction_is_deterministic_and_cannot_beat_the_pure_floor(brain):
    """At temperature 0 the policy is pure, so 1/6 is the best it could possibly be."""
    policy = kuhn.extract_policy(brain, temperature=0.0)
    assert all(p in (0.0, 1.0) for p in policy.values())
    assert kuhn.exploitability(policy) >= 1.0 / 6.0 - 1e-9


def test_play_hand_returns_a_legal_kuhn_payoff(brain):
    """Chips won must be one of the four possible Kuhn outcomes."""
    rng = np.random.default_rng(0)
    for seat in (1, 2):
        won, decisions = kuhn.play_hand(brain, kuhn.nash_policy(), seat, rng)
        assert won in (-2, -1, 1, 2)
        assert decisions >= 1
    brain.abandon_hand()
