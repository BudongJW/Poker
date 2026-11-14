"""Command line: ``python -m pokerlab <command> ...`` (see ``--help`` of each command)."""

import argparse
import json

from . import advisor, daily, inference, phh, preflop, report, selfplay
from .bots import BOT_TYPES
from .cards import cards_str, parse_cards
from .equity import matchup


def _player(text):
    """A matchup seat: '?' (random hand), two cards ('AsKs') or a range ('TT+,AQs+', '15%')."""
    if text in ("?", "random"):
        return None
    try:
        cards = parse_cards(text)
        if len(cards) == 2:
            return tuple(cards)
    except ValueError:
        pass
    return preflop.parse_range(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pokerlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("equity", help="win probability and interpretation of one spot")
    p.add_argument("hand", help="hole cards, e.g. AsKs")
    p.add_argument("--board", default="", help="0, 3, 4 or 5 board cards, e.g. 'Qs Js 2d'")
    p.add_argument("--players", type=int, default=2, help="players still in the hand including you (default 2)")
    p.add_argument("--range", action="append", default=[], dest="ranges", metavar="RANGE",
                   help="known range of one opponent, e.g. 'TT+,AQs+', '15%%' or exact cards 'QhQd'; repeatable")
    p.add_argument("--villain", action="append", default=[], dest="lines", metavar="LINE",
                   help="what one opponent did, streets split by '/', e.g. 'open/bet/bet'; the range is "
                        "inferred from it. Pre-flop: limp open call 3bet call3bet 4bet check. "
                        "Post-flop: check bet call raise callraise reraise. Put the opponent's seat in front "
                        "if you know it: 'btn: open/bet' (ep mp co btn sb bb). A size may follow a word, in "
                        "percent of the pot ('bet66', 'call50' = called a half-pot bet), pre-flop in big "
                        "blinds ('open3x', 'call3x'), or 'allin' ('betallin', 'callallin'). A stack may stand "
                        "next to the seat: 'btn 40bb: open/bet'. Repeatable")
    p.add_argument("--position", help="your own seat (ep mp co btn sb bb); with a villain's seat it tells "
                                      "the model who acts last after the flop")
    p.add_argument("--table", type=int, help="players dealt in at the table (default 6, or 9 with more than "
                                             "six players in the hand); --players is who is still in")
    p.add_argument("--stack", type=float, metavar="BB",
                   help="effective stack in big blinds at the start of the hand, for --villain lines without their "
                        "own; without it the model answers for the average stack depth")
    p.add_argument("--model", default=advisor.DEFAULT_MODEL,
                   help=f"action model that reads --villain lines: {', '.join(inference.MODELS)} or a model file "
                        f"(default: {advisor.DEFAULT_MODEL})")
    p.add_argument("--profile", default="all", help="opponent type within the model (default: all)")
    p.add_argument("--pot", type=float, help="pot size before your call or bet")
    p.add_argument("--to-call", type=float, help="amount you must call")
    p.add_argument("--bet", type=float, help="bet size you are considering")
    p.add_argument("--precision", type=float, metavar="PCT",
                   help="simulate until the 95%% interval is within +-PCT percentage points, e.g. 0.05")
    p.add_argument("--trials", type=int, default=200_000, help="Monte Carlo trials when --precision is not given")
    p.add_argument("--seed", type=int)
    p.add_argument("--json", action="store_true", help="print the numbers as JSON instead of a report")

    p = sub.add_parser("matchup", help="equity of every player: known hands, ranges or random hands")
    p.add_argument("players", nargs="+", metavar="PLAYER", help="'AsKs', a range such as 'TT+,AQs+' or '15%%', or '?'")
    p.add_argument("--board", default="")
    p.add_argument("--dead", default="", help="cards known to be out of the deck")
    p.add_argument("--precision", type=float, metavar="PCT")
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
    p.add_argument("--format", metavar="NAMES",
                   help="play tables of tournament formats instead (see 'formats'): a name, several with commas, or a mix")

    p = sub.add_parser("formats", help="the tournament formats self-play can use, each value with how it is known")
    p.add_argument("--depths", type=int, metavar="HANDS",
                   help="also play this many hands of each format and show how deep the stacks are")

    p = sub.add_parser("report", help="statistics over a self-play database")
    p.add_argument("--db", default="selfplay.sqlite")

    p = sub.add_parser("fit-model", help="learn how players act with each hand strength from a self-play database")
    p.add_argument("--db", default="selfplay.sqlite")
    p.add_argument("--out", default=inference.MODEL_PATH, help="where to write the model (default: the bundled one)")
    p.add_argument("--hands", type=int, default=15_000, help="hands sampled from the database (default 15000)")
    p.add_argument("--processes", type=int)

    p = sub.add_parser("score-model", help="measure equity error with and without range inference on a database")
    p.add_argument("--db", default="selfplay.sqlite")
    p.add_argument("--model", default=inference.MODEL_PATH)
    p.add_argument("--hands", type=int, default=4000)
    p.add_argument("--processes", type=int)

    p = sub.add_parser("fit-history", help="learn an action model from public hand histories (PHH files)")
    p.add_argument("paths", nargs="+", metavar="PATH", help=".phh / .phhs files or directories of them")
    p.add_argument("--out", required=True, help="where to write the model")
    p.add_argument("--labels", choices=["none", *phh.LABELS], default="none",
                   help="profiles besides 'all': 'stake' = one per stake band, 'pluribus' = the bot vs the professionals")
    p.add_argument("--init", metavar="MODEL", help="starting model for hands whose cards are hidden (a name or a file)")
    p.add_argument("--hands", type=int, help="read at most this many hands, from files in random order")
    p.add_argument("--iterations", type=int, default=30, help="most EM passes per stage when cards are hidden (default 30)")
    p.add_argument("--use-sizes", action="store_true",
                   help="let bet sizes and call prices also count as evidence for what a hidden hand was. Right for "
                        "players who act on hand strength alone; on real-money hands it read shown hands worse")
    p.add_argument("--processes", type=int)

    p = sub.add_parser("score-history", help="how well an action model explains hand histories it was not fitted on")
    p.add_argument("paths", nargs="+", metavar="PATH")
    p.add_argument("--model", default=advisor.DEFAULT_MODEL, help="a bundled model name or a model file")
    p.add_argument("--hands", type=int, default=20_000)
    p.add_argument("--processes", type=int)

    p = sub.add_parser("daily", help="one learning cycle: play, update the range model and the champion, record")
    p.add_argument("--cycles", type=int, default=1, help="cycles to run back to back; 0 = until Ctrl-C")
    p.add_argument("--hands", type=int, default=60_000, help="hands per challenger match (default 60000)")
    p.add_argument("--challengers", type=int, default=3, help="challengers tried per cycle (default 3)")
    p.add_argument("--opponents", default=",".join(daily.REFERENCES),
                   help=f"the four fixed opponents at the table (default: {','.join(daily.REFERENCES)})")
    p.add_argument("--state", default=daily.STATE_DIR, help="directory holding the learned state")
    p.add_argument("--processes", type=int)
    p.add_argument("--keep-db", action="store_true", help="keep the cycle's hand database instead of deleting it")
    p.add_argument("--formats", metavar="NAMES",
                   help="also play tables of tournament formats for the range model to learn from, e.g. kr_mtt (see 'formats')")
    p.add_argument("--format-hands", type=int, help="hands of those formats per cycle (default: as many as one match)")

    sub.add_parser("history", help="what the learning cycles have done so far").add_argument(
        "--state", default=daily.STATE_DIR)

    p = sub.add_parser("serve", help="local web page: reads the windows you pick as sources, live; also a calculator")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--record", metavar="FILE",
                   help="append every hand followed in a window to this file (JSON lines)")

    p = sub.add_parser("preflop-table", help="recompute the 169-hand pre-flop equity table (about 25 minutes)")
    p.add_argument("--trials", type=int, default=4_000_000)
    p.add_argument("--processes", type=int)

    args = parser.parse_args(argv)
    if args.command == "equity":
        spot = dict(hero=args.hand, board=args.board, n_players=args.players, pot=args.pot, to_call=args.to_call,
                    ranges=[preflop.parse_range(r) for r in args.ranges], trials=args.trials, seed=args.seed,
                    precision=args.precision and args.precision / 100, bet=args.bet, lines=args.lines,
                    profile=args.profile, model=args.model, position=args.position, table=args.table,
                    stack=args.stack)
        if args.json:
            print(json.dumps(advisor.to_json(advisor.analyze(**spot)), ensure_ascii=False, indent=1))
        else:
            print(advisor.explain(**spot))
    elif args.command == "matchup":
        results = matchup([_player(text) for text in args.players], args.board, args.dead, args.trials,
                          seed=args.seed, precision=args.precision and args.precision / 100)
        how = f"전수 계산 ({results[0].trials:,}가지)" if results[0].exact else f"몬테카를로 {results[0].trials:,}회"
        print(f"보드 {cards_str(parse_cards(args.board)) or '없음'} | {how}")
        for text, r in zip(args.players, results):
            error = "" if r.exact else f"  ±{r.ci95:.2%}p"
            print(f"  {text:<16} 승률 {r.equity:7.2%}  (승 {r.win:.2%} / 무 {r.tie:.2%}){error}")
    elif args.command == "simulate":
        selfplay.run(args.db, args.hands, args.seconds, [int(x) for x in args.players.split(",")],
                     args.bots.split(","), args.batch_size, args.processes, args.seed, args.stack_bb,
                     not args.no_actions, format=args.format)
    elif args.command == "formats":
        import random

        from . import formats
        from .bots import make_bot
        print("\n".join(formats.describe()))
        if args.depths:
            pool = [make_bot(kind, seed=index) for index, kind in enumerate(("tag", "lag", "nit", "tag_range", "human", "pro"))]
            turn = iter(range(10 ** 9))
            print(f"\n형식별 스택 깊이 ({args.depths:,}핸드, 봇: tag·lag·nit·tag_range·human·pro)")
            for name, fmt in formats.load().items():
                if fmt.playable:
                    found = formats.depths(fmt, lambda: pool[next(turn) % len(pool)], random.Random(1), args.depths)
                    share = " · ".join(f"{key} {value:.0%}" for key, value in found["share"].items())
                    print(f"  {name:<12} 테이블 {found['tables']}개 · 평균 {found['players']:.1f}명 · 스택 중앙값 {found['median_bb']:.0f}BB"
                          f" · {share} · 최고 레벨 {found['max_level']}")
    elif args.command == "report":
        report.print_report(args.db)
    elif args.command == "fit-model":
        model = inference.fit(args.db, args.hands, processes=args.processes)
        model.save(args.out)
        print(f"written {args.out}: {model.source}; profiles {', '.join(sorted(model.counts))}")
    elif args.command == "score-model":
        rows = inference.score(args.db, inference.ActionModel.load(args.model), args.hands, processes=args.processes)
        show = dict(float_format=lambda x: f"{x:.4f}", na_rep="-")
        print("실제 상대 핸드 기준 승률과의 평균 절대 오차(mae)와 편향(bias = 추정 - 실제)")
        print(inference.summarize(rows).to_string(**show))
        print("\n추정 승률 구간별 실제 승률 (said = 추정 평균, truth = 실제 평균, n = 결정 수)")
        print(inference.reliability(rows).to_string(**show))
    elif args.command == "fit-history":
        hands = phh.load(args.paths, phh.LABELS.get(args.labels), args.hands)
        print(f"{len(hands):,}핸드 읽음, 홀카드 공개 {sum(len(h['holes']) for h in hands):,}건")
        model = inference.fit_prepared(
            inference.prepare(hands, processes=args.processes), inference.load_model(args.init) if args.init else None,
            args.iterations, source=f"{len(hands)} hands from {', '.join(args.paths)}"[:200], progress=print,
            stages=(("full",) if args.init else ("coarse", "full")) if args.use_sizes else None)
        model.save(args.out)
        print(f"written {args.out}; profiles {', '.join(sorted(model.counts))}")
    elif args.command == "score-history":
        hands = phh.load(args.paths, None, args.hands, seed=1)
        model = inference.load_model(args.model)
        fit = inference.action_fit(inference.prepare(hands, processes=args.processes), model)
        print(f"{len(hands):,}핸드, 액션 {fit['actions']:,}개: 액션당 로그우도 {fit['log_likelihood']:.4f}")
        print(f"홀카드가 공개된 {fit['shown_hands']:,}건: 액션으로 추정한 레인지가 실제 핸드에 준 확률 = 균등 추측의 {fit['shown_lift']:.2f}배")
        if all(len(h["holes"]) == len(h["bots"]) for h in hands):      # every card known: equity can be checked
            rows = inference.score_hands(hands, model, processes=args.processes)
            print("실제 상대 핸드 기준 승률과의 평균 절대 오차(mae)와 편향(bias = 추정 - 실제)")
            print(inference.summarize(rows).to_string(float_format=lambda x: f"{x:.4f}"))
    elif args.command == "daily":
        daily.run(args.cycles, state_dir=args.state, hands=args.hands, challengers=args.challengers,
                  references=tuple(args.opponents.split(",")), processes=args.processes, keep_db=args.keep_db,
                  formats=args.formats, format_hands=args.format_hands)
    elif args.command == "history":
        daily.print_history(args.state)
    elif args.command == "serve":
        from . import web
        web.serve(args.host, args.port, args.record)
    elif args.command == "preflop-table":
        preflop.build_table(args.trials, args.processes, progress=print)
        print("written", preflop.TABLE_PATH)


if __name__ == "__main__":
    main()
