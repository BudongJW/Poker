# Fly brain: picking this up locally

Everything below assumes you have pulled `claude/eager-wright-adiqrx`. Read
`doc/flybrain.md` for why the module is built the way it is; this file is only about what
to do next and what will waste your time if you don't know it.

---

## 0. Get it running (10 minutes)

```bash
pip install numpy scipy pyarrow          # pyarrow only to read the bulk connectome files

python -m poker.flybrain.cli reference   # game-theory self-test, no download needed
python -m poker.flybrain.cli info        # downloads ~1.1 GB on first run, then caches
python -m poker.flybrain.cli controls --hands 4000 --seeds 5
```

The download lands in `poker/data/flybrain/` (gitignored). Only three of the published
MaleCNS v1.0 files are needed — annotations (14 MB), neurotransmitters (42 MB), weights
(1.0 GB). The 6.8 GB and 12.7 GB files are **not** needed; the segment-to-segment weight
table already *is* the adjacency matrix.

First build scans all 151,856,684 edges (~27 s) and caches an `.npz`. Subsequent loads are
0.3 s.

No download? Everything still runs with `--synthetic` on a stand-in that matches the real
population sizes and edge count. It exercises every code path and tells you **nothing
about the fly** — `Connectome.synthetic` is `True` so you can refuse to report it.

Tests: `pytest poker/flybrain/tests/ -q` → 59 passing, no network, no download.

---

## 1. Where things stand

**The central question has been answered, in the negative: the specific wiring is not
contributing.** Task 1 is done — the operating point was re-derived, every condition is
now calibrated to the same Kenyon cell sparsity before training, and the controls were
re-run over 5 seeds. See "Kuhn controls at matched sparsity" in `doc/flybrain.md` for the
full table. Mean [min, max] over 5 seeds:

| condition | gain | KC | exploit (pure) | chips vs random |
|---|---|---|---|---|
| real connectome | 0.00499 | 8.7% | 0.4167 [0.2500, 0.8333] | +0.1750 [0.0000, 0.3750] |
| shuffled (degree-pres) | 0.00246 | 9.2% | 0.4333 [0.3333, 0.8333] | +0.1000 [0.0000, 0.3750] |
| frozen (no plasticity) | 0.00499 | 8.7% | 0.6500 [0.2500, 1.1667] | −0.0750 [−0.5833, +0.1250] |
| random actions | n/a | n/a | 0.4583 | 0.0000 |

Three things to carry forward:

- **Real does not beat shuffled.** 0.4167 vs 0.4333, ranges overlapping almost completely.
  Real is nominally ahead on all three metrics but every gap is a small fraction of the
  seed spread. This was the falsification condition stated in advance, and it fired.
- **Real barely beats a random policy** (0.4167 vs 0.4583). Everything sits far from the
  0.1667 best-deterministic bar, so this is a comparison between degenerate policies.
- **The plasticity is no longer inert — it now helps.** `frozen` is clearly worse than
  `real` at a working operating point. The earlier `frozen == real` finding was an artefact
  of 0.49% KC activity. KC→MBON depression is now the only component with a demonstrated
  effect, while the wiring has none.

Known and still standing:

- The Kuhn implementation self-tests against published facts (12 information sets, game
  value exactly −1/18, Nash exploitability exactly 0). That part is solid.
- The connectome loads correctly: 8,246 neurons, 451,855 edges, populations matching
  MaleCNS v1.0, `synthetic: False`.
- 59 tests pass; pylint 9.88/10 under pylint 4.0.8. The 9.95 recorded earlier was an older
  pylint — the newer one emits `W0012` on every `import-outside-toplevel` suppression in
  the repo, including files nobody has touched. Baseline on unmodified code is 9.87.

## 2. Task 1 — DONE. What was changed, and what it cost

The operating point is fixed and the controls have been re-run. Recorded here so nobody
re-derives it:

- **The gain is re-derived per task**, in `config.CALIBRATED_GAIN`: Kuhn 0.004994 (8.7% KC
  on the 12 information sets), equity 0.004902 (8.1% on 8 representative states).
  `LIFParams.synaptic_gain` now defaults to 0.0050 instead of 0.0026. The two tasks want
  different values, which is why they are a table and not a constant.
- **`cmd_controls` calibrates every condition separately** and prints the chosen gain and
  the achieved sparsity beside that condition's result. The gains land 2× apart — real
  0.00499, shuffled 0.00246 — which is the size of the confound that voided the first run.
- **`calibrate_gain` is fed Kuhn stimuli** via `calibration_tables(task)`, and its `bounds`
  went from `(0.0005, 0.0040)` to `(0.0002, 0.0120)`. The old upper bound could not reach
  the ≈0.0050 the real network needs, so calibration would silently have converged against
  it. It also now keeps the closest probe rather than the last, and warns if it finishes
  without reaching the target.
- **`controls --seeds N`** (default 5) reports mean and [min, max] per metric.

**The falsification condition fired.** At matched sparsity shuffled performs like real, so
the wiring is not contributing and the readout was carrying the result. Do not go looking
for a way to make this come out differently.

One thing did *not* survive the fix, in the useful direction: **the plasticity is no longer
inert.** `frozen` is now clearly worse than `real`, so the old Task 2 premise is gone. See
section 3.

**Still worth following up:** why the shuffle makes the network hyperactive at a shared
gain. The shuffle preserves in- and out-degree but scatters each neuron's targets, so the
fly's inhibition — 478 GABA/glutamate neurons out of 8,246, plus the ALLN local neurons —
stops landing on the cells it controls. Targeted inhibition is the first thing
degree-preserving rewiring destroys. It is an activity-level effect rather than a
computational one, so it is not a performance claim — but it is the one place the specific
wiring demonstrably does something, and it is now the most promising thread in the project.

## 3. Task 2 — the plasticity is no longer inert; measure what it is doing

**Rewritten after Task 1.** The premise below ("frozen == real") no longer holds. At a
working operating point `frozen` is *worse* than `real` on every metric — 0.6500 against
0.4167 exploitability, −0.0750 against +0.1750 chips vs random — with the widest seed
spread of any condition, [0.2500, 1.1667]. So KC→MBON depression is contributing, and it
is the only component that demonstrably is.

That makes the question "how much, and through what" rather than "why is it dead":

- Log `plasticity.apply()`'s L1 weight delta per hand against the readout's own update
  magnitude. Now that both matter, the ratio is the interesting number.
- `trace_decay` is 0.6 per decision and a Kuhn hand is 1–2 decisions, so every KC active in
  the hand gets near-equal credit. Test near 0 (credit only the last decision).
- The clean isolation test still applies, and is now more informative: disable the
  **readout** update instead of the plasticity and see whether KC→MBON plasticity alone
  moves exploitability. `decoder.update()` is called unconditionally in `brain.reinforce()`
  — you need a flag for it.

The historical note, kept because it explains the reversal:

`frozen == real` on all four metrics in the first run, and on a second machine they
diverged only slightly (chips vs random +0.3750 real against +0.3333 frozen). Both runs
were at 0.49% Kenyon cell activity, where there is almost nothing for a depression rule to
act on — that was the entire explanation. The lesson generalises: **a null result measured
at a broken operating point is not a null result.** Two components were written off on the
strength of those runs, and re-measuring at 9% activity reversed one of them.

Kuhn hands being 1–2 decisions long is what makes this tractable; the old equity task could
not isolate it.

---

## 4. Task 3 — free the live-play path from a third-party server

Only needed if you want to run against a real client. **Do not start here** — the fly's
value is in the decision, and that is already measurable offline.

**The blocker.** `db = https://dickreuter.com:7778/` in `config_default.ini` is a remote
REST API belonging to the upstream author, not a local MongoDB. At startup `main.py`
requires it for:

| call | what it fetches | without it |
|---|---|---|
| `updater.check_update()` | version check | startup fails |
| `get_preflop_sheet_url()` → `get_internal` | the preflop sheet **URL** | `pd.read_excel()` fails |
| `mongo.get_table(name)` | **scraper coordinates and templates** | cannot read the screen |

So the bot does not start without that host. It was unreachable from the container this was
built in, but that container's egress is proxied, so that is not evidence the server is
down — you will find out quickly on your own machine.

Also outbound during play: `increment_plays` every round, `write_log_file` every round, and
`upload_collusion_data` (hole cards, hostname, strategy, timestamp) — the last one only
when `collusion == 1`, so off by default.

**The fix, in order:**

1. **Local table-definition cache.** Same pattern as `connectome.py`: fetch once, write to
   `poker/data/`, load from there afterwards, and fall back to the cache when the remote is
   unreachable. This is the highest-value item — without it everything below is hostage to
   someone else's uptime.
2. **Screenshot replay harness.** `poker/tests/screenshots/` has 7 fixtures. Run them
   through the scraper into `FlyDecision` and log the decisions. Validates perception
   without playing a hand, and runs in CI. Note `poker/tests/__init__.py` imports
   `poker.main` (PyQt6, matplotlib, tensorflow) and reads the db URL at import time, which
   is why the flybrain tests live in `poker/flybrain/tests/` instead — put the harness
   there too.
3. **Cross-platform fixes.** `requirements.txt` does not exist, so the Dockerfile's
   `pip3 install -r requirements.txt` fails and the shipped Linux/Docker path does not
   build. And `os.environ['COMPUTERNAME']` (game_logger.py:57, 101, 129, 137) is
   Windows-only; lines 57 and 101 are in `write_log_file`, which `main.py` calls every round
   in a daemon thread, so on Linux/macOS game logging dies silently while the bot keeps
   playing.

**If you do go to a real client:** prefer one you control (the README says "Any other table
can be mapped as well"; there is a table mapper and `doc/table_mapper.mp4`). Zero terms-of-
service exposure, unlimited hands, and you get ground truth — real hole cards, a known
opponent policy — that PokerStars can never give you. For research that is strictly better.

On PokerStars specifically: its terms prohibit automated play and that **covers play-money
tables**; the exposure is the account. `mode = shadow` means the fly never touches the
mouse, so the fly adds no new risk — but the existing bot's risk is unchanged. `mode =
active` additionally requires `confirmed_play_money = true` set by hand, which is
deliberately impossible to satisfy by accident.

---

## 5. Traps that will cost you time

- **Do not switch `sensory_mode` to `'poisson'`** for anything you intend to report.
  Measured d' ≈ 1.0 at every gain and window tried, including a 5× longer one: with 451k
  recurrent edges the network is chaotic, and two runs of the *same* state diverge as much
  as two different states. `'current'` is reproducible bit-for-bit.
- **Do not turn off `quantize_features`.** Decoding a continuous equity from MBON activity
  gave in-sample R² 0.667 and out-of-sample **−6.88**. The Kenyon cell expansion separates
  patterns; it does not interpolate. Discrete cues are the regime the literature tested the
  fly in, and Kuhn's three cards are already discrete.
- **Do not use hard top-k for KC sparsity** (`kc_hard_winner_take_all`). It makes the KC
  response a step function of the input and destroys generalisation. Graded APL-style
  inhibition is the default for a reason.
- **Never `np.isin` over the whole weight table.** 151.8M rows; it OOMs. `connectome.py`
  batches at 4M rows and uses `searchsorted`. Keep that shape.
- **On the real connectome you get 3 tuning bins per feature, not 5.** There are 54 ORN type
  groups and 14 features, so `per_feature = min(5, 54 // 14) = 3`. The bin centres land on
  0.0 / 0.5 / 1.0, which happens to match J/Q/K exactly. If you add features, check you
  have not silently aliased them — `_assign_channels` asserts the banks are disjoint
  because an earlier modulo version let the last three features overwrite the first three.
- **Long runs: write straight to a file, never through `| tail`.** The pipe buffers until
  the process exits, so you see nothing and cannot tell slow from hung.
- **Never compare connectome variants at a shared gain.** This is the mistake that voided
  the first control run. Real and its own degree-preserving shuffle need gains 2× apart to
  reach the same 9% KC activity, so a shared gain compares activity levels. `cmd_controls`
  calibrates per condition now; if you add a condition, calibrate it too and print the gain.
- **A null result measured at a broken operating point is not a null result.** Both
  "shuffled cannot learn" and "the plasticity is inert" were measured at 0.49% KC activity
  and both reversed at 9%. Check the operating point before believing any negative.
- **`calibrate_gain`'s bounds must span every condition.** The old `(0.0005, 0.0040)` could
  not reach the ≈0.0050 the real network needs, and bisection converges silently against a
  bound it cannot cross. It is `(0.0002, 0.0120)` now and warns if it finishes off-target.
- **One seed tells you nothing here.** The seed spread within a condition ([0.2500, 0.8333]
  for real) is far larger than the differences between conditions. Use `--seeds 5` at
  minimum and report the range.
- **Judge the pure policy against 0.1667, not 0.** Equilibrium in Kuhn requires mixing, so
  a temperature-0 argmax readout cannot reach 0 however well it trains.

---

## 6. Quick reference

```
poker/flybrain/
  connectome.py   MaleCNS v1.0 loader, batched scan, npz cache, synthetic fallback
  circuit.py      ORN/ALPN/ALLN/KC/MBON/DAN index maps; PAM vs PPL1 split
  lif.py          sparse LIF engine, graded APL inhibition, plastic blocks
  encoding.py     features -> quantised tuning curves -> ORN rates; TableView base
  decoding.py     MBON rates -> action scores; legality masking
  plasticity.py   KC->MBON depression gated by dopaminergic RPE
  brain.py        FlyBrain: decide() / reinforce() / save() / load()
  kuhn.py         the measuring instrument - game tree, Nash family, exact exploitability
  controls.py     degree-preserving shuffle, calibrate_gain, RandomAgent
  track.py        local live-play record: per-decision rows, per-hand settlement
  decision.py     FlyDecision, a drop-in for the bot's Decision
  guard.py        the play-money declaration checkpoint
  cli.py          info / reference / calibrate / train / controls
```

Reference numbers worth memorising: equilibrium 0, best deterministic **1/6 = 0.1667**,
uniform 0.4583, never-bet 1.0, game value to player 1 **−1/18**.

Connectome attribution, required on anything published (CC-BY):

> MaleCNS v1.0 connectome, HHMI Janelia Research Campus & Google Research, CC-BY.
> https://male-cns.janelia.org/
