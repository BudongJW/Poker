"""Offline harness: train, evaluate and run controls without touching a poker client.

Everything here runs on synthetic hands, so the fly can be developed and trained with
no account involved at all. Only once it beats its controls here is there any reason to
put it in front of a real table, and then in shadow mode first.

The training task is a contextual bandit over one betting decision. It is deliberately
small enough that the optimal action is computable in closed form, which is what makes
learning measurable rather than merely asserted - the same reason the literature tests
mushroom body models on bandits (Bennett et al. 2021) and the reason a
game-theoretically solved variant like Kuhn poker is the right next step up from here.

Usage:
    python -m poker.flybrain.cli info
    python -m poker.flybrain.cli calibrate
    python -m poker.flybrain.cli train --hands 2000 --save poker/data/flybrain/brain.npz
    python -m poker.flybrain.cli controls --hands 1000
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
import argparse
import logging
import sys

import numpy as np

from poker.flybrain import controls as controls_mod
from poker.flybrain.brain import FlyBrain
from poker.flybrain.config import FlyBrainConfig, LIFParams, RunMode, Scope

log = logging.getLogger(__name__)

# Bet sizes as a multiple of the pot, matching the decoder's action vocabulary.
BET_FRACTION = {'Bet': 0.25, 'Bet half pot': 0.5, 'Bet pot': 1.0}


class OfflineTable:
    """The attributes encoding.features_from_table() and decoding.legal_actions() read.

    A stand-in for the scraper's table object, so offline hands and live hands present
    an identical interface to the brain.
    """

    # pylint: disable=too-many-instance-attributes,invalid-name
    def __init__(self, equity, pot, to_call, stack, stage, opponents=1,
                 check_available=False, big_blind=1.0):
        self.abs_equity = equity
        self.equity = equity
        self.relative_equity = equity
        self.totalPotValue = pot
        self.round_pot_value = pot
        self.minCall = to_call
        self.minBet = max(big_blind, pot * 0.25)
        self.myFunds = stack
        self.bigBlind = big_blind
        self.smallBlind = big_blind / 2.0
        self.gameStage = stage
        self.assumedPlayers = opponents + 1
        self.playersAhead = opponents
        self.other_player_has_initiative = to_call > 0
        self.checkButton = check_available
        self.callButton = not check_available
        self.betButton = True
        self.allInCallButton = False


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
    """A fixed spread of states, for calibration and sparsity measurement."""
    game = EquityGame(seed=seed)
    return [game.deal() for _ in range(n)]


def _make_config(args, gain=None):
    config = FlyBrainConfig(
        mode=RunMode.offline,
        scope=Scope.whole_brain if args.whole_brain else Scope.mushroom_body,
        weight_threshold=args.weight_threshold,
        use_synthetic=args.synthetic,
        cache_dir=args.cache_dir or '',
        seed=args.seed,
    )
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

    tables = _representative_tables()
    sparsity = controls_mod.measure_kc_sparsity(brain, tables)
    print(f"\nKC sparsity at gain {brain.config.lif.synaptic_gain:.5f}: {sparsity:.3f}")
    return 0


def cmd_calibrate(args):
    """Find the gain that puts Kenyon cell sparsity on target."""
    tables = _representative_tables()
    gain, measured = controls_mod.calibrate_gain(
        lambda g: FlyBrain(_make_config(args, gain=g)),
        tables,
        target_sparsity=args.target_sparsity,
    )
    print(f"calibrated synaptic_gain = {gain:.6f}  (KC sparsity {measured:.3f}, "
          f"target {args.target_sparsity:.3f})")
    return 0


def cmd_train(args):
    """Train on the equity game and report before/after."""
    config = _make_config(args)
    brain = FlyBrain(config)
    if args.load:
        brain.load(args.load)

    game = EquityGame(seed=args.seed)
    before = run_episodes(brain, EquityGame(seed=args.seed + 9999), args.eval_hands,
                          learn=False)
    log.info("before training: regret %.3f bb, optimal %.1f%%",
             before['mean_regret_bb'], 100 * before['optimal_action_rate'])

    trained = run_episodes(brain, game, args.hands, temperature=args.temperature,
                           learn=True, log_every=max(1, args.hands // 10))

    after = run_episodes(brain, EquityGame(seed=args.seed + 9999), args.eval_hands,
                         learn=False)
    log.info("after training : regret %.3f bb, optimal %.1f%%",
             after['mean_regret_bb'], 100 * after['optimal_action_rate'])

    print(f"\n{'':14s} {'regret(bb)':>11s} {'optimal%':>9s}")
    print(f"{'before':14s} {before['mean_regret_bb']:>11.3f} "
          f"{100*before['optimal_action_rate']:>8.1f}%")
    print(f"{'after':14s} {after['mean_regret_bb']:>11.3f} "
          f"{100*after['optimal_action_rate']:>8.1f}%")
    print(f"\ntraining actions: {trained['action_counts']}")

    if args.save:
        brain.save(args.save)
    return 0


def cmd_controls(args):
    """Compare the real connectome against shuffled, frozen and random baselines."""
    from poker.flybrain import connectome as connectome_mod  # pylint: disable=import-outside-toplevel

    config = _make_config(args)
    real = connectome_mod.load(
        scope=config.scope, weight_threshold=config.weight_threshold,
        cache_dir=config.cache_dir or None, use_synthetic=config.use_synthetic,
        seed=config.seed,
    )

    conditions = []

    def train_and_eval(label, brain):
        game = EquityGame(seed=args.seed)
        run_episodes(brain, game, args.hands, temperature=args.temperature, learn=True)
        result = run_episodes(brain, EquityGame(seed=args.seed + 9999), args.eval_hands,
                              learn=False)
        conditions.append((label, result))
        log.info("%-22s regret %.3f bb  optimal %.1f%%", label,
                 result['mean_regret_bb'], 100 * result['optimal_action_rate'])

    train_and_eval('real connectome', FlyBrain(_make_config(args), connectome=real))

    shuffled = controls_mod.shuffle_preserving_degree(real, seed=args.seed)
    train_and_eval('shuffled (degree-pres)', FlyBrain(_make_config(args), connectome=shuffled))

    frozen_config = _make_config(args)
    frozen_config.plasticity.enabled = False
    train_and_eval('frozen (no plasticity)', FlyBrain(frozen_config, connectome=real))

    random_agent = controls_mod.RandomAgent(seed=args.seed)
    result = run_episodes(random_agent, EquityGame(seed=args.seed + 9999),
                          args.eval_hands, learn=False)
    conditions.append(('random actions', result))

    print(f"\n{'condition':24s} {'regret(bb)':>11s} {'optimal%':>9s}")
    for label, result in conditions:
        print(f"{label:24s} {result['mean_regret_bb']:>11.3f} "
              f"{100*result['optimal_action_rate']:>8.1f}%")
    print("\nIf 'real connectome' does not beat 'shuffled', the specific wiring is not "
          "contributing and no claim about the fly's circuit is supported.")
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
    parser.add_argument('-v', '--verbose', action='store_true')

    sub = parser.add_subparsers(dest='command', required=True)

    p_info = sub.add_parser('info', help='print the loaded circuit and its sparsity')
    p_info.set_defaults(func=cmd_info)

    p_cal = sub.add_parser('calibrate', help='find the gain hitting a target KC sparsity')
    p_cal.add_argument('--target-sparsity', type=float, default=0.09)
    p_cal.set_defaults(func=cmd_calibrate)

    p_train = sub.add_parser('train', help='train on the equity game')
    p_train.add_argument('--hands', type=int, default=1000)
    p_train.add_argument('--eval-hands', type=int, default=300)
    p_train.add_argument('--temperature', type=float, default=0.3,
                         help='softmax temperature while training; 0 is greedy')
    p_train.add_argument('--no-plasticity', action='store_true')
    p_train.add_argument('--save', default='')
    p_train.add_argument('--load', default='')
    p_train.set_defaults(func=cmd_train)

    p_ctrl = sub.add_parser('controls', help='real vs shuffled vs frozen vs random')
    p_ctrl.add_argument('--hands', type=int, default=1000)
    p_ctrl.add_argument('--eval-hands', type=int, default=300)
    p_ctrl.add_argument('--temperature', type=float, default=0.3)
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
