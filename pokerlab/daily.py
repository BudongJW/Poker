"""One learning cycle: play, learn, evaluate, record. Run it once a day (``python -m pokerlab daily``)
or back to back (``--cycles 0``).

Two things are learned, and each change must prove itself on fresh hands before it is kept:

* **What opponents hold** - the action model (:mod:`pokerlab.inference`). The day's hands are
  added to the counts learned so far (older counts fade by ``decay``). The merged model replaces
  the current one unless it estimates equity worse on hands it was not fitted on.
* **How to bet** - the champion's parameters (how strong to bet, how much to demand for a call,
  how often to bluff, how much to respect raises). Each cycle a few mutated challengers play the
  champion at the same tables, next to four fixed opponents (two of them play like real online
  players); a challenger takes over only if it wins more with 95% confidence.

With ``formats`` the day also plays tables of tournament formats (:mod:`pokerlab.formats`):
blinds that rise, stacks that get short, players who sit down late. Those hands go into what the
action model learns from and is checked on; the champion is still chosen at the usual tables.

State lives in three small files (by default next to the package data, so it can be committed):
``action_model.json``, ``champion.json`` and ``history.jsonl`` (one line per cycle).
"""

import datetime
import json
import os
import random
import sqlite3
import tempfile
import time
from contextlib import closing

from . import inference, report, selfplay

STATE_DIR = os.path.join(os.path.dirname(__file__), "data")
START = {"value": 0.15, "margin": 0.06, "bluff": 0.05, "respect": 0.2}            # the 'tag' bot
BOUNDS = {"value": (0.02, 0.45), "margin": (-0.05, 0.25), "bluff": (0.0, 0.4), "respect": (0.0, 0.5)}
STEP = {"value": 0.04, "margin": 0.03, "bluff": 0.05, "respect": 0.08}
# The four fixed opponents that fill the six-handed table: two play with the habits of real online
# players (see pokerlab.bots.PopulationBot), two are this package's equity bots.
REFERENCES = ("human", "human", "nit", "tag_range")


def load_champion(state_dir=STATE_DIR):
    path = os.path.join(state_dir, "champion.json")
    if not os.path.exists(path):
        return {"generation": 0, "params": dict(START)}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_history(state_dir=STATE_DIR):
    path = os.path.join(state_dir, "history.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mutate(params, rng):
    """Nudge one or two parameters by a random step, inside their bounds."""
    out = dict(params)
    for name in rng.sample(sorted(out), rng.choice((1, 2))):
        low, high = BOUNDS[name]
        out[name] = round(min(high, max(low, out[name] + rng.gauss(0, STEP[name]))), 3)
    return out


def _spec(name, params, model_path=None):
    """Bot spec for :func:`pokerlab.bots.make_bot`; the bot reads ranges with the model at ``model_path``."""
    model = model_path if model_path and os.path.exists(model_path) else 1
    return f"equity:name={name};model={model};" + ";".join(f"{key}={value}" for key, value in sorted(params.items()))


def _paired_difference(db_path, first_batch):
    """Challenger minus champion over the hands of one match -> (hands, bb/100, 95% half-width)."""
    with closing(sqlite3.connect(db_path)) as db:
        n, mean, squares = db.execute("""
            SELECT COUNT(*), AVG(d), AVG(d * d) FROM (
                SELECT SUM(CASE bot WHEN 'challenger' THEN net_bb ELSE -net_bb END) AS d
                FROM players JOIN hands USING (hand_id)
                WHERE bot IN ('challenger', 'champion') AND batch >= ? GROUP BY hand_id)""", (first_batch,)).fetchone()
    return n, 100 * mean, 196 * (max(squares - mean * mean, 0.0) / n) ** 0.5


def run_cycle(state_dir=STATE_DIR, hands=60_000, challengers=3, fit_hands=8000, score_hands=1500, decay=0.9,
              references=REFERENCES, processes=None, seed=None, work_dir=None, keep_db=False, progress=print,
              formats=None, format_hands=None):
    """Play, learn, evaluate and record once -> the history entry that was appended.

    formats: tournament formats or a mix ('kr_mtt', 'mtt_turbo,satellite'); ``format_hands`` hands
    of them (default: as many as one match) are played by the champion and the fixed opponents
    after the matches and join the hands the action model is fitted and checked on.
    """
    if len(references) != 4:
        raise ValueError("the table is six-handed: champion, challenger and exactly four fixed opponents")
    started = time.time()
    say = progress or (lambda message: None)
    os.makedirs(state_dir, exist_ok=True)
    history = load_history(state_dir)
    cycle = len(history) + 1
    rng = random.Random(cycle if seed is None else seed)
    champion = load_champion(state_dir)
    model_path = os.path.join(state_dir, "action_model.json")
    db_path = os.path.join(work_dir or tempfile.gettempdir(), f"pokerlab-cycle-{cycle}-{os.getpid()}.sqlite")

    try:
        # 1. play: each challenger meets the champion at six-handed tables; everything is logged
        matches, played = [], 0
        for index in range(challengers):
            first_batch = 1
            if index:                           # later matches append to the same database
                with closing(sqlite3.connect(db_path)) as db:
                    first_batch = db.execute("SELECT MAX(batch) FROM batches").fetchone()[0] + 1
            candidate = mutate(champion["params"], rng)
            lineup = [_spec("champion", champion["params"], model_path),
                      _spec("challenger", candidate, model_path), *references]
            n, _ = selfplay.run(db_path, hands=hands, players=(6,), bots=lineup, batch_size=500, processes=processes,
                                seed=rng.getrandbits(30), progress=None)
            played += n
            _, diff, ci = _paired_difference(db_path, first_batch)
            promoted = diff - ci > 0
            matches.append({"params": candidate, "diff_bb100": round(diff, 1), "ci95": round(ci, 1), "promoted": promoted})
            say(f"도전자 {index + 1}/{challengers}: {candidate} -> 챔피언 대비 {diff:+.1f} ± {ci:.1f} bb/100"
                f" ({'교체' if promoted else '유지'})")
            if promoted:
                champion = {"generation": champion["generation"] + 1, "params": candidate}
        standings = {row.bot: round(row.bb100, 1) for row in report.bot_summary(db_path).itertuples()}
        format_played = 0
        if formats:                             # tournament tables: only the action model learns from these
            lineup = [_spec("champion", champion["params"], model_path), *references]
            format_played, _ = selfplay.run(db_path, hands=format_hands or hands, bots=lineup, batch_size=500,
                                            processes=processes, seed=rng.getrandbits(30), progress=None, format=formats)
            played += format_played
            say(f"대회 형식 {formats}: {format_played:,}핸드")

        # 2. learn what opponents hold, 3. check it on the fifth of the hands that was not fitted on
        today = inference.fit(db_path, fit_hands, flop_runouts=150, seed=cycle, processes=processes, held_out=False)
        current = inference.ActionModel.load(model_path) if os.path.exists(model_path) else None
        merged = current.merged(today, decay) if current else today

        def error(model):
            rows = inference.score(db_path, model, score_hands, trials=3000, seed=cycle, processes=processes, held_out=True)
            return inference.summarize(rows).loc["all"]

        after = error(merged)
        before = error(current) if current else None
        keep_model = before is None or after["mae_model"] <= before["mae_model"] * 1.02
        say(f"레인지 모델: 평균 오차 {'-' if before is None else format(before['mae_model'], '.4f')} -> "
            f"{after['mae_model']:.4f} (랜덤 가정 {after['mae_random']:.4f}) -> {'반영' if keep_model else '기각'}")

        # 4. keep what passed, 5. record
        if keep_model:
            merged.save(model_path)
            inference.load_model.cache_clear()          # bots created from now on read the new model
        with open(os.path.join(state_dir, "champion.json"), "w", encoding="utf-8") as fh:
            json.dump(champion, fh, indent=1)
        entry = {
            "cycle": cycle, "date": datetime.date.today().isoformat(), "hands": played,
            "seconds": round(time.time() - started),
            "model": {"mae_random": round(float(after["mae_random"]), 4),
                      "mae_before": None if before is None else round(float(before["mae_model"]), 4),
                      "mae_after": round(float(after["mae_model"]), 4), "bias_after": round(float(after["bias_model"]), 4),
                      "kept": bool(keep_model)},
            "matches": matches, "champion": champion, "standings": standings, "opponents": list(references),
        }
        if formats:
            entry["formats"] = {"names": formats, "hands": format_played}
        with open(os.path.join(state_dir, "history.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    finally:
        if not keep_db:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(db_path + suffix):
                    os.remove(db_path + suffix)
    say(f"사이클 {cycle} 완료: {played:,}핸드, {entry['seconds']}초, 챔피언 {champion['generation']}세대 {champion['params']}")
    return entry


def run(cycles=1, **options):
    """``cycles`` learning cycles back to back; 0 keeps going until interrupted."""
    done = 0
    try:
        while cycles == 0 or done < cycles:
            run_cycle(**options)
            done += 1
    except KeyboardInterrupt:
        pass
    return done


def print_history(state_dir=STATE_DIR):
    history = load_history(state_dir)
    if not history:
        print("아직 기록이 없습니다. python -m pokerlab daily 로 첫 사이클을 실행하세요.")
        return
    print("사이클  날짜        핸드     오차(랜덤 가정 -> 모델 전 -> 후)   모델   챔피언 세대   도전자 결과(bb/100)")
    for e in history:
        m = e["model"]
        before = "   -  " if m["mae_before"] is None else f"{m['mae_before']:.4f}"
        duels = ", ".join(f"{x['diff_bb100']:+.0f}±{x['ci95']:.0f}{'*' if x['promoted'] else ''}" for x in e["matches"])
        print(f"{e['cycle']:>4}   {e['date']}  {e['hands']:>7,}   {m['mae_random']:.4f} -> {before} -> {m['mae_after']:.4f}"
              f"      {'반영' if m['kept'] else '기각'}   {e['champion']['generation']:>6}        {duels}")
    last = history[-1]
    print(f"\n현재 챔피언({last['champion']['generation']}세대): {last['champion']['params']}   (* = 챔피언 교체)")
    print("최근 사이클 순위(bb/100): " + ", ".join(f"{bot} {bb:+.0f}" for bot, bb in last["standings"].items()))
