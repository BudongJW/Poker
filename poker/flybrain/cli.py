"""Offline harness: train, evaluate and run controls without touching a poker client.

Everything here runs on synthetic hands, so the fly can be developed and trained with
no account involved at all. Only once it beats its controls here is there any reason to
put it in front of a real table, and then in shadow mode first.

The default task is **Kuhn poker**, because its optimum is known in closed form and a
policy's exploitability can therefore be computed exactly. Results are reported against
two fixed reference points: 0 for equilibrium, and 1/6 for the best any deterministic
policy can do (equilibrium in Kuhn requires mixing, so a fly reading off an argmax cannot
beat that).

The older `equity` task is still reachable with --task equity. It is a contextual bandit
over one betting decision, and it turned out to be a poor instrument: a bet-everything
policy scored well on mean regret while being wrong about which bet, a random policy
scored well on optimal-action-rate by folding often, and the two metrics disagreed. It is
kept for comparison, not for conclusions.

Usage:
    python -m poker.flybrain.cli info
    python -m poker.flybrain.cli calibrate
    python -m poker.flybrain.cli reference
    python -m poker.flybrain.cli train --hands 4000 --save poker/data/flybrain/brain.npz
    python -m poker.flybrain.cli controls --hands 4000
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
import argparse
import logging
import sys

import numpy as np

from poker.flybrain import controls as controls_mod
from poker.flybrain import kuhn
from poker.flybrain.brain import FlyBrain
from poker.flybrain.encoding import TableView
from poker.flybrain.config import (CALIBRATED_GAIN, FlyBrainConfig, LIFParams, RunMode,
                                   Scope)

log = logging.getLogger(__name__)

# Bet sizes as a multiple of the pot, matching the decoder's action vocabulary.
BET_FRACTION = {'Bet': 0.25, 'Bet half pot': 0.5, 'Bet pot': 1.0}


class OfflineTable(TableView):
    """A hand of the equity game, presented through the shared table surface."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, equity, pot, to_call, stack, stage, opponents=1,
                 check_available=False, big_blind=1.0):
        super().__init__(equity, pot, to_call if not check_available else 0.0, stack,
                         stage, opponents=opponents, big_blind=big_blind)
        self.betButton = True


class EquityGame:
    """One-decision contextual bandit with an analytically known best action.

    A hand deals an equity, a pot and a price. Folding is worth nothing. Calling and
    betting have expected values that follow directly from the equity, so the optimal
    action - and therefore the regret of whatever the fly picked - is exact.
    """

    STAGES = ('PreFlop', 'Flop', 'Turn', 'River')

    def __init__(self, seed=0, big_blind=1.0):
        self.rng = np.random.default_rng(seed)
        self.big_blind = big_blind

    def deal(self):
        """Return a fresh OfflineTable."""
        equity = float(self.rng.uniform(0.02, 0.98))
        pot = float(self.rng.uniform(2.0, 30.0)) * self.big_blind
        check_available = bool(self.rng.random() < 0.3)
        to_call = 0.0 if check_available else float(self.rng.uniform(0.25, 1.0)) * pot
        stack = float(self.rng.uniform(20.0, 200.0)) * self.big_blind
        stage = self.STAGES[int(self.rng.integers(len(self.STAGES)))]
        return OfflineTable(equity, pot, to_call, stack, stage,
                            check_available=check_available, big_blind=self.big_blind)

    @staticmethod
    def call_probability(size, pot):
        """How often the opponent calls a bet of `size` into `pot`.

        A bigger bet gets called less, but the opponent still calls most of the time.
        The constant matters: make folding too likely and betting with hopeless equity
        becomes the best action at every size, because the pot won on a fold outweighs
        what is lost on a call. At this setting bluffing with 2% equity is correctly
        unprofitable.

        Larger value bets do still tend to beat smaller ones here, as they do in real
        poker against a calling opponent - the sizing gradient is mostly monotone and
        this task does not claim an interior optimum.
        """
        if pot <= 0:
            return 1.0
        return 1.0 / (1.0 + 0.5 * size / pot)

    def expected_value(self, table, action):
        """EV of one action in big blinds.

        Calling risks `to_call` to win the pot. Betting wins the pot outright when the
        opponent folds and goes to showdown for pot+size when they call, so there is an
        interior best size that depends on equity - the gradient a value-based decision
        circuit could plausibly pick up.
        """
        equity = table.abs_equity
        pot = table.totalPotValue
        to_call = table.minCall
        bb = table.bigBlind

        if action == 'Fold':
            return 0.0
        if action == 'Check':
            return equity * pot / bb * 0.5
        if action == 'Call':
            return (equity * (pot + to_call) - (1.0 - equity) * to_call) / bb
        if action in BET_FRACTION:
            size = min(BET_FRACTION[action] * pot, table.myFunds)
            p_call = self.call_probability(size, pot)
            called = equity * (pot + size) - (1.0 - equity) * size
            folded = pot
            return (p_call * called + (1.0 - p_call) * folded) / bb
        return 0.0

    def best(self, table, allowed):
        """(best_action, best_ev) over the legal set."""
        scored = [(self.expected_value(table, a), a) for a in allowed]
        ev, action = max(scored)
        return action, ev


def _representative_tables(n=8, seed=3):
    """A fixed spread of equity-game states, for calibration and sparsity measurement."""
    game = EquityGame(seed=seed)
    return [game.deal() for _ in range(n)]


def _kuhn_tables(stack=100.0):
    """All 12 Kuhn information sets - the entire state space of that task."""
    return [kuhn.KuhnTable(card, history, stack=stack)
            for card, history in kuhn.INFO_SETS]


def calibration_tables(task):
    """The stimuli to calibrate the operating point on, for the task being measured.

    Sparsity has to be measured on the states the fly will actually see. Calibrating the
    Kuhn task on equity-game tables sets the operating point from a different stimulus
    distribution, which is how the network ended up effectively silent on Kuhn states at
    the old default gain.
    """
    return _kuhn_tables() if task == 'kuhn' else _representative_tables()


def _make_config(args, gain=None):
    config = FlyBrainConfig(
        mode=RunMode.offline,
        scope=Scope.whole_brain if args.whole_brain else Scope.mushroom_body,
        weight_threshold=args.weight_threshold,
        use_synthetic=args.synthetic,
        cache_dir=args.cache_dir or '',
        seed=args.seed,
    )
    if gain is None:
        gain = CALIBRATED_GAIN.get(getattr(args, 'task', 'kuhn'))
    if gain is not None:
        config.lif = LIFParams(synaptic_gain=gain)
    if getattr(args, 'no_plasticity', False):
        config.plasticity.enabled = False
    return config


def run_episodes(agent, game, n_hands, temperature=0.0, learn=True, log_every=0):
    """Play n_hands and return a metrics dict.

    Regret is measured against the analytic optimum, so 0 means the agent always picked
    the best legal action and larger is worse.
    """
    from poker.flybrain import decoding  # pylint: disable=import-outside-toplevel

    regrets, rewards, optimal_hits = [], [], 0
    action_counts = {}

    for hand in range(n_hands):
        table = game.deal()
        allowed = decoding.legal_actions(table)
        observation = agent.decide(table, allowed=allowed, temperature=temperature)

        chosen_ev = game.expected_value(table, observation.action)
        _, best_ev = game.best(table, allowed)

        regrets.append(best_ev - chosen_ev)
        rewards.append(chosen_ev)
        optimal_hits += int(abs(best_ev - chosen_ev) < 1e-9)
        action_counts[observation.action] = action_counts.get(observation.action, 0) + 1

        if learn:
            agent.reinforce(chosen_ev)
        else:
            agent.abandon_hand()

        if log_every and (hand + 1) % log_every == 0:
            window = regrets[-log_every:]
            log.info("  hand %6d  mean regret %.3f bb  optimal %.1f%%",
                     hand + 1, float(np.mean(window)),
                     100.0 * optimal_hits / (hand + 1))

    return {
        'hands': n_hands,
        'mean_regret_bb': float(np.mean(regrets)) if regrets else float('nan'),
        'mean_ev_bb': float(np.mean(rewards)) if rewards else float('nan'),
        'optimal_action_rate': optimal_hits / n_hands if n_hands else float('nan'),
        'action_counts': action_counts,
    }


def kuhn_opponents(kind, alpha=1.0 / 6.0):
    """The opponent pool to train or evaluate against.

    'nash'   equilibrium play. The most the fly can win is the game value, -1/18.
    'random' aggress half the time everywhere. Exploitable by 0.458, so there is real
             headroom and a clear learning signal.
    'mix'    half of each, which keeps the headroom without overfitting to one opponent.
    """
    if kind == 'nash':
        return [kuhn.nash_policy(alpha)]
    if kind == 'random':
        return [kuhn.uniform_policy()]
    if kind == 'mix':
        return [kuhn.nash_policy(alpha), kuhn.uniform_policy()]
    raise ValueError(f"unknown opponent {kind!r}")


def run_kuhn(agent, n_hands, opponents, seed=0, temperature=0.0, learn=True, log_every=0):
    """Play Kuhn hands, alternating seats, and return metrics.

    Seats alternate so the fly has to learn both sides; its information sets are disjoint
    between seats, so this is not two tasks competing for the same weights.
    """
    rng = np.random.default_rng(seed)
    chips, decisions = [], 0

    for hand in range(n_hands):
        opponent = opponents[hand % len(opponents)]
        fly_seat = 1 + (hand % 2)
        won, n_decisions = kuhn.play_hand(agent, opponent, fly_seat, rng,
                                          temperature=temperature)
        chips.append(won)
        decisions += n_decisions

        if learn:
            agent.reinforce(won)
        else:
            agent.abandon_hand()

        if log_every and (hand + 1) % log_every == 0:
            window = chips[-log_every:]
            log.info("  hand %6d  chips/hand %+.4f", hand + 1, float(np.mean(window)))

    return {
        'hands': n_hands,
        'fly_decisions': decisions,
        'chips_per_hand': float(np.mean(chips)) if chips else float('nan'),
    }


def evaluate_kuhn(brain, seed=0, alpha=1.0 / 6.0):
    """Evaluate the fly's Kuhn policy exactly - nothing here is sampled.

    Once the policy is read off the 12 information sets (12 LIF simulations), every number
    below follows from the game tree in closed form: exploitability, and chips per hand
    against each fixed opponent averaged over both seats. Sampling them instead would cost
    thousands of simulations and add variance to quantities that have exact values.
    """
    del seed
    pure_policy = kuhn.extract_policy(brain, temperature=0.0)
    mixed_policy = kuhn.extract_policy(brain, temperature=1.0)

    def chips_against(fly_policy, opponent):
        """Exact chips per hand to the fly, averaged over the two seats."""
        as_p1 = kuhn.expected_value(kuhn.merge_policies(fly_policy, opponent))
        as_p2 = -kuhn.expected_value(kuhn.merge_policies(opponent, fly_policy))
        return 0.5 * (as_p1 + as_p2)

    # chips_vs_nash is bounded above by 0 and reaches it only for optimal play. Against
    # an equilibrium opponent no strategy can beat the game value, so the fly gets at most
    # -1/18 in seat 1 and at most +1/18 in seat 2; averaged over the two seats that is a
    # ceiling of 0. It is a second, independent read on the same question exploitability
    # answers, and the two should move together.
    nash = kuhn.nash_policy(alpha)
    uniform = kuhn.uniform_policy()
    return {
        'exploitability_pure': kuhn.exploitability(pure_policy),
        'exploitability_mixed': kuhn.exploitability(mixed_policy),
        'chips_vs_nash': chips_against(pure_policy, nash),
        'chips_vs_random': chips_against(pure_policy, uniform),
        'chips_vs_random_mixed': chips_against(mixed_policy, uniform),
        'policy': pure_policy,
        'mixed_policy': mixed_policy,
    }


def print_kuhn_reference():
    """The fixed points every result is read against."""
    print("Kuhn poker reference points (exploitability, chips/hand):")
    print(f"  equilibrium                     {0.0:.4f}")
    print(f"  best deterministic policy       {1.0 / 6.0:.4f}   <- the bar at temperature 0")
    print(f"  aggress 50% everywhere          {kuhn.exploitability(kuhn.uniform_policy()):.4f}")
    print(f"  never bet, never call           {kuhn.exploitability(kuhn.always_pass_policy()):.4f}")
    print(f"  game value to player 1          {kuhn.GAME_VALUE_TO_P1:+.4f}")


def cmd_reference(args):
    """Print the Kuhn reference points and verify the implementation."""
    del args
    kuhn.self_test()
    print_kuhn_reference()
    print(f"\nNash policy (alpha=1/6):\n{kuhn.describe_policy(kuhn.nash_policy())}")
    return 0


def cmd_info(args):
    """Print what is loaded and how big it is."""
    brain = FlyBrain(_make_config(args))
    manifest = brain.manifest()
    print(f"connectome : {manifest['connectome']}")
    print(f"synthetic  : {manifest['synthetic']}")
    print(f"attribution: {manifest['attribution']}")
    print("populations:")
    for name, count in manifest['populations'].items():
        print(f"  {name:5s} {count:7,d}")
    print(f"  DAN appetitive (PAM)  {manifest['dan_appetitive']:7,d}")
    print(f"  DAN aversive   (PPL1) {manifest['dan_aversive']:7,d}")

    tables = calibration_tables(args.task)
    sparsity = controls_mod.measure_kc_sparsity(brain, tables)
    print(f"\nKC sparsity on {args.task} stimuli at gain "
          f"{brain.config.lif.synaptic_gain:.5f}: {sparsity:.3f}")
    return 0


def cmd_calibrate(args):
    """Find the gain that puts Kenyon cell sparsity on target."""
    tables = calibration_tables(args.task)
    gain, measured = controls_mod.calibrate_gain(
        lambda g: FlyBrain(_make_config(args, gain=g)),
        tables,
        target_sparsity=args.target_sparsity,
    )
    print(f"calibrated synaptic_gain = {gain:.6f}  (KC sparsity {measured:.3f}, "
          f"target {args.target_sparsity:.3f})")
    return 0


def cmd_train(args):
    """Train and report before/after."""
    config = _make_config(args)
    if args.task == 'kuhn':
        # Kuhn payoffs are 1 or 2 chips; the default reward scale of 10 big blinds would
        # squash every outcome into the flat part of the tanh.
        config.plasticity.reward_scale_bb = 1.0
    brain = FlyBrain(config)
    if args.load:
        brain.load(args.load)

    if args.task == 'kuhn':
        kuhn.self_test()
        print_kuhn_reference()
        opponents = kuhn_opponents(args.opponent)

        before = evaluate_kuhn(brain, seed=args.seed)
        log.info("before training: exploitability %.4f (pure) / %.4f (mixed)",
                 before['exploitability_pure'], before['exploitability_mixed'])

        run_kuhn(brain, args.hands, opponents, seed=args.seed,
                 temperature=args.temperature, learn=True,
                 log_every=max(1, args.hands // 10))

        after = evaluate_kuhn(brain, seed=args.seed)
        log.info("after training : exploitability %.4f (pure) / %.4f (mixed)",
                 after['exploitability_pure'], after['exploitability_mixed'])

        print(f"\n{'':8s} {'exploit(pure)':>14s} {'exploit(mixed)':>15s} "
              f"{'chips vs nash':>14s} {'chips vs random':>16s}")
        for label, result in (('before', before), ('after', after)):
            print(f"{label:8s} {result['exploitability_pure']:>14.4f} "
                  f"{result['exploitability_mixed']:>15.4f} "
                  f"{result['chips_vs_nash']:>+14.4f} {result['chips_vs_random']:>+16.4f}")
        print(f"\nlearned policy:\n{kuhn.describe_policy(after['policy'])}")
    else:
        game = EquityGame(seed=args.seed)
        before = run_episodes(brain, EquityGame(seed=args.seed + 9999), args.eval_hands,
                              learn=False)
        trained = run_episodes(brain, game, args.hands, temperature=args.temperature,
                               learn=True, log_every=max(1, args.hands // 10))
        after = run_episodes(brain, EquityGame(seed=args.seed + 9999), args.eval_hands,
                             learn=False)
        print(f"\n{'':14s} {'regret(bb)':>11s} {'optimal%':>9s}")
        print(f"{'before':14s} {before['mean_regret_bb']:>11.3f} "
              f"{100*before['optimal_action_rate']:>8.1f}%")
        print(f"{'after':14s} {after['mean_regret_bb']:>11.3f} "
              f"{100*after['optimal_action_rate']:>8.1f}%")
        print(f"\ntraining actions: {trained['action_counts']}")

    if args.save:
        brain.save(args.save)
    return 0


KUHN_METRICS = (
    ('exploit(pure)', 'exploitability_pure'),
    ('exploit(mixed)', 'exploitability_mixed'),
    ('chips vs nash', 'chips_vs_nash'),
    ('chips vs random', 'chips_vs_random'),
)

EQUITY_METRICS = (
    ('regret(bb)', 'mean_regret_bb'),
    ('optimal action rate', 'optimal_action_rate'),
)


def _summarise(values):
    """mean, min and max of one metric across seeds."""
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.min()), float(array.max())


def _print_spread(conditions, metrics, extra_columns=()):
    """One block per metric: mean and [min, max] across seeds, per condition."""
    for title, key in metrics:
        print(f"\n{title}")
        header = f"  {'condition':24s} {'mean':>9s} {'min':>9s} {'max':>9s}"
        for name in extra_columns:
            header += f" {name:>12s}"
        print(header)
        for label, results, extras in conditions:
            mean, low, high = _summarise([r[key] for r in results])
            row = f"  {label:24s} {mean:>+9.4f} {low:>+9.4f} {high:>+9.4f}"
            for name in extra_columns:
                row += f" {extras.get(name, ''):>12s}"
            print(row)


def cmd_record(args):
    """Print the live-play track record collected by poker/flybrain/track.py."""
    from poker.flybrain import track  # pylint: disable=import-outside-toplevel

    print(track.format_summary(track.recorder(args.path or None).summary()))
    return 0


def cmd_controls(args):
    """Compare the real connectome against shuffled, frozen and random baselines.

    Every condition is calibrated to the same Kenyon cell sparsity before it is trained.
    Without that the comparison is between activity levels rather than wiring diagrams:
    at a shared gain the real connectome and its degree-preserving shuffle sit an order of
    magnitude apart in activity, because the shuffle scatters the targets of the fly's
    inhibitory neurons. The chosen gain is reported beside each condition so the reader
    can check the conditions really were matched.
    """
    from poker.flybrain import connectome as connectome_mod  # pylint: disable=import-outside-toplevel

    base = _make_config(args)
    real = connectome_mod.load(
        scope=base.scope, weight_threshold=base.weight_threshold,
        cache_dir=base.cache_dir or None, use_synthetic=base.use_synthetic,
        seed=base.seed,
    )
    tables = calibration_tables(args.task)
    seeds = list(range(args.seed, args.seed + args.seeds))

    def config_for(seed, gain=None, plasticity_enabled=True):
        config = _make_config(args, gain=gain)
        config.seed = seed
        if args.task == 'kuhn':
            config.plasticity.reward_scale_bb = 1.0
        config.plasticity.enabled = plasticity_enabled
        return config

    if args.task == 'kuhn':
        kuhn.self_test()
        print_kuhn_reference()

    def calibrated_brain(label, seed, wiring, plasticity_enabled=True):
        """Bring this condition to the target sparsity, then build it at that gain."""
        gain, measured = controls_mod.calibrate_gain(
            lambda g: FlyBrain(config_for(seed, gain=g), connectome=wiring),
            tables, target_sparsity=args.target_sparsity,
        )
        log.info("%-22s seed %d calibrated gain %.5f -> KC sparsity %.3f",
                 label, seed, gain, measured)
        brain = FlyBrain(config_for(seed, gain=gain,
                                    plasticity_enabled=plasticity_enabled),
                         connectome=wiring)
        return brain, gain, measured

    def train_and_eval(brain, seed):
        if args.task == 'kuhn':
            run_kuhn(brain, args.hands, kuhn_opponents(args.opponent), seed=seed,
                     temperature=args.temperature, learn=True)
            return evaluate_kuhn(brain, seed=seed)
        run_episodes(brain, EquityGame(seed=seed), args.hands,
                     temperature=args.temperature, learn=True)
        return run_episodes(brain, EquityGame(seed=seed + 9999), args.eval_hands,
                            learn=False)

    def run_condition(label, wiring_for_seed, plasticity_enabled=True):
        results, gains, sparsities = [], [], []
        for seed in seeds:
            brain, gain, measured = calibrated_brain(
                label, seed, wiring_for_seed(seed), plasticity_enabled=plasticity_enabled)
            gains.append(gain)
            sparsities.append(measured)
            result = train_and_eval(brain, seed)
            results.append(result)
            headline = (f"exploitability {result['exploitability_pure']:.4f}"
                        if args.task == 'kuhn'
                        else f"regret {result['mean_regret_bb']:.3f} bb")
            log.info("%-22s seed %d gain %.5f  KC %.3f  %s",
                     label, seed, gain, measured, headline)
        extras = {
            'gain': f"{np.mean(gains):.5f}",
            'KC sparsity': f"{np.mean(sparsities):.3f}",
        }
        return (label, results, extras)

    conditions = [
        run_condition('real connectome', lambda _seed: real),
        run_condition(
            'shuffled (degree-pres)',
            lambda seed: controls_mod.shuffle_preserving_degree(real, seed=seed)),
        run_condition('frozen (no plasticity)', lambda _seed: real,
                      plasticity_enabled=False),
    ]

    if args.task == 'kuhn':
        # A uniformly random agent *is* the uniform policy, so its numbers are exact and
        # seed-independent; one entry repeated keeps the table shape uniform.
        uniform = kuhn.uniform_policy()
        nash = kuhn.nash_policy()
        random_result = {
            'exploitability_pure': kuhn.exploitability(uniform),
            'exploitability_mixed': kuhn.exploitability(uniform),
            'chips_vs_nash': 0.5 * (kuhn.expected_value(kuhn.merge_policies(uniform, nash))
                                    - kuhn.expected_value(kuhn.merge_policies(nash, uniform))),
            'chips_vs_random': 0.0,
        }
        conditions.append(('random actions', [random_result] * len(seeds),
                           {'gain': 'n/a', 'KC sparsity': 'n/a'}))
        _print_spread(conditions, KUHN_METRICS, extra_columns=('gain', 'KC sparsity'))
        print(f"\nbest deterministic policy achievable: {1.0 / 6.0:.4f}    "
              f"equilibrium: 0.0000")
    else:
        random_results = [
            run_episodes(controls_mod.RandomAgent(seed=seed),
                         EquityGame(seed=seed + 9999), args.eval_hands, learn=False)
            for seed in seeds
        ]
        conditions.append(('random actions', random_results,
                           {'gain': 'n/a', 'KC sparsity': 'n/a'}))
        _print_spread(conditions, EQUITY_METRICS, extra_columns=('gain', 'KC sparsity'))

    print(f"\n{len(seeds)} seeds per condition ({seeds[0]}..{seeds[-1]}), "
          f"{args.hands} training hands each, "
          f"all conditions calibrated to {args.target_sparsity:.2f} KC sparsity on "
          f"{args.task} stimuli.")
    print("If 'real connectome' does not beat 'shuffled' at matched sparsity, the specific "
          "wiring is not contributing and the readout is carrying the result.")
    return 0


def build_parser():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog='python -m poker.flybrain.cli',
        description='Offline training and controls for the fly-brain poker decision module.',
    )
    parser.add_argument('--synthetic', action='store_true',
                        help='use the structurally-matched stand-in instead of real wiring')
    parser.add_argument('--whole-brain', action='store_true',
                        help='load all 165,122 Traced neurons instead of the mushroom body')
    parser.add_argument('--weight-threshold', type=int, default=3,
                        help='drop connections below this synapse count (default 3)')
    parser.add_argument('--cache-dir', default='',
                        help='where connectome downloads and caches live')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', choices=('kuhn', 'equity'), default='kuhn',
                        help='kuhn (default): exact exploitability against a known '
                             'optimum. equity: the older contextual bandit, kept for '
                             'comparison only - it is a poor instrument.')
    parser.add_argument('-v', '--verbose', action='store_true')

    sub = parser.add_subparsers(dest='command', required=True)

    p_info = sub.add_parser('info', help='print the loaded circuit and its sparsity')
    p_info.set_defaults(func=cmd_info)

    p_cal = sub.add_parser('calibrate', help='find the gain hitting a target KC sparsity')
    p_cal.add_argument('--target-sparsity', type=float, default=0.09)
    p_cal.set_defaults(func=cmd_calibrate)

    p_ref = sub.add_parser('reference', help='print the Kuhn reference points')
    p_ref.set_defaults(func=cmd_reference)

    p_rec = sub.add_parser('record', help='the live-play track record against real tables')
    p_rec.add_argument('--path', default='',
                       help='sqlite file to read (default poker/data/flybrain/'
                            'live_play.sqlite)')
    p_rec.set_defaults(func=cmd_record)

    p_train = sub.add_parser('train', help='train on the equity game')
    p_train.add_argument('--hands', type=int, default=4000)
    p_train.add_argument('--eval-hands', type=int, default=2000)
    p_train.add_argument('--opponent', choices=('mix', 'nash', 'random'), default='mix')
    p_train.add_argument('--temperature', type=float, default=0.3,
                         help='softmax temperature while training; 0 is greedy')
    p_train.add_argument('--no-plasticity', action='store_true')
    p_train.add_argument('--save', default='')
    p_train.add_argument('--load', default='')
    p_train.set_defaults(func=cmd_train)

    p_ctrl = sub.add_parser('controls', help='real vs shuffled vs frozen vs random')
    p_ctrl.add_argument('--hands', type=int, default=4000)
    p_ctrl.add_argument('--eval-hands', type=int, default=2000)
    p_ctrl.add_argument('--opponent', choices=('mix', 'nash', 'random'), default='mix')
    p_ctrl.add_argument('--temperature', type=float, default=0.3)
    p_ctrl.add_argument('--seeds', type=int, default=5,
                        help='how many seeds per condition; results are reported as mean '
                             'and [min, max] because which policy training lands on is '
                             'not deterministic even though exploitability is exact')
    p_ctrl.add_argument('--target-sparsity', type=float, default=0.09,
                        help='KC sparsity every condition is calibrated to before it is '
                             'trained (default 0.09)')
    p_ctrl.set_defaults(func=cmd_controls)

    return parser


def main(argv=None):
    """Entry point."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s %(message)s',
    )
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
