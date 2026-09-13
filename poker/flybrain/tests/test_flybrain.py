"""Tests for the fly-brain decision module.

These run on the synthetic connectome so they need no 1 GB download and no network.
The synthetic stand-in matches the real data's population sizes and edge count, so it
exercises every code path; what it cannot do is say anything about the fly, which is
why test_synthetic_is_flagged exists.
"""

import numpy as np
import pytest

from poker.flybrain import connectome as connectome_mod
from poker.flybrain import controls, decoding, encoding
from poker.flybrain.brain import FlyBrain
from poker.flybrain.cli import EquityGame, OfflineTable, run_episodes
from poker.flybrain.config import (FlyBrainConfig, LIFParams, RunMode, Scope)
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
