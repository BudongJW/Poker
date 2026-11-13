"""Command line: ``python -m pokerlab {equity,simulate,report,preflop-table} ...``"""

import argparse

from . import advisor, preflop, report, selfplay
from .bots import BOT_TYPES


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pokerlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("equity", help="win probability and interpretation of one spot")
    p.add_argument("hand", help="hole cards, e.g. AsKs")
    p.add_argument("--board", default="", help="0, 3, 4 or 5 board cards, e.g. 'Qs Js 2d'")
    p.add_argument("--players", type=int, default=2, help="players in the hand including you (default 2)")
    p.add_argument("--range", action="append", default=[], dest="ranges", metavar="RANGE",
                   help="range of one opponent, e.g. 'TT+,AQs+' or '15%%'; repeat for more opponents")
    p.add_argument("--pot", type=float, help="pot size before your call")
    p.add_argument("--to-call", type=float, help="amount you must call")
    p.add_argument("--trials", type=int, default=200_000)
    p.add_argument("--seed", type=int)

    p = sub.add_parser("simulate", help="bots play each other; every hand is stored in SQLite")
    p.add_argument("--db", default="selfplay.sqlite")
    p.add_argument("--hands", type=int, help="stop after this many hands")
    p.add_argument("--seconds", type=float, help="stop after this many seconds (neither = until Ctrl-C)")
    p.add_argument("--players", default="6", help="table size(s), comma separated, e.g. 2,6,9")
    p.add_argument("--bots", default="tag,lag,nit,station,maniac,random",
                   help=f"bot pool, comma separated; available: {', '.join(sorted(BOT_TYPES))}")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--processes", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stack-bb", type=int, default=100)
    p.add_argument("--no-actions", action="store_true", help="do not store the per-action log")

    p = sub.add_parser("report", help="statistics over a self-play database")
    p.add_argument("--db", default="selfplay.sqlite")

    p = sub.add_parser("preflop-table", help="recompute the 169-hand pre-flop equity table")
    p.add_argument("--trials", type=int, default=1_000_000)
    p.add_argument("--processes", type=int)

    args = parser.parse_args(argv)
    if args.command == "equity":
        print(advisor.explain(args.hand, args.board, args.players, args.pot, args.to_call,
                              [preflop.parse_range(r) for r in args.ranges], args.trials, args.seed))
    elif args.command == "simulate":
        selfplay.run(args.db, args.hands, args.seconds, [int(x) for x in args.players.split(",")],
                     args.bots.split(","), args.batch_size, args.processes, args.seed, args.stack_bb,
                     not args.no_actions)
    elif args.command == "report":
        report.print_report(args.db)
    elif args.command == "preflop-table":
        preflop.build_table(args.trials, args.processes)
        print("written", preflop.TABLE_PATH)


if __name__ == "__main__":
    main()
