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
python -m poker.flybrain.cli controls --hands 4000
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

Tests: `pytest poker/flybrain/tests/ -q` → 54 passing, no network, no download.

---

## 1. Where things stand

**There is no established result yet.** The control run recorded earlier is void — see
"Kuhn controls — VOID, and why" in `doc/flybrain.md`. Short version:

At the default `synaptic_gain` of 0.0026, measured on the 12 Kuhn information sets:

| gain | real KC active | shuffled KC active | real MBON | shuffled MBON |
|---|---|---|---|---|
| **0.0026 (default)** | **0.49%** | **13.02%** | 0.34 Hz | 84.33 Hz |
| 0.0040 | 3.05% | 26.99% | 2.96 Hz | 202.19 Hz |
| **0.0050** | **8.71%** | 30.96% | 10.53 Hz | 218.34 Hz |
| 0.0080 | 35.91% | 35.97% | 90.09 Hz | 242.37 Hz |

The real network is effectively silent at the default gain; the shuffled one is 27× more
active. Every "real vs shuffled" number produced so far compares a silent network against a
hyperactive one. It also did not reproduce on a second machine, which is expected: at
0.34 Hz MBON output the action scores barely separate, so argmax decisions turn on
float-accumulation order.

**Do not run `info` and expect ~0.09.** It reports sparsity at the default gain on
equity-game states and currently gives 0.006. That is the broken operating point, not a
misconfigured install.

Known and still standing:

- The Kuhn implementation self-tests against published facts (12 information sets, game
  value exactly −1/18, Nash exploitability exactly 0). That part is solid.
- The connectome loads correctly: 8,246 neurons, 451,855 edges, populations matching
  MaleCNS v1.0, `synthetic: False`.
- 54 tests pass, pylint 9.95/10.

## 2. Task 1 — fix the operating point, then re-run the controls

Nothing else matters until this is done. There is currently no result to build on.

**Three concrete pieces:**

1. **Re-derive the default gain for the current encoder.** 0.0026 was calibrated before the
   encoder gained tuning-curve quantisation, disjoint glomerulus banks and target-mean-rate
   normalisation, and was never re-derived. On Kuhn states the real network reaches ~9% KC
   sparsity at roughly **0.0050**. Confirm that and update `LIFParams.synaptic_gain`,
   noting that the equity task may want a different value — which is an argument for
   calibrating per task rather than hardcoding one number.

2. **Make `cmd_controls` calibrate each condition separately.** It currently never calls
   `calibrate_gain` at all, so every condition is built with the same gain. Each condition
   needs its own gain chosen to hit the same target sparsity, and the chosen gain must be
   printed next to that condition's result. Expect the gains to differ a lot: real needs
   ≈0.0050, shuffled is already at 13% by 0.0026 so it needs something well below that.

3. **Feed calibration Kuhn stimuli.** `_representative_tables()` (cli.py:134) builds
   equity-game tables. Use `[kuhn.KuhnTable(c, h) for c, h in kuhn.INFO_SETS]` — exactly 12
   states, which is the whole state space of the task.

`controls.calibrate_gain(brain_factory, tables, target_sparsity=0.09)` already exists and
bisects correctly. Widen its `bounds` — the default `(0.0005, 0.0040)` cannot reach the
≈0.0050 the real network needs.

**Then** run 5+ seeds per condition and report mean and spread, not a single number.

**Also report chip metrics, not just exploitability.** In the second run three of four
conditions landed on exactly 0.3333 exploitability while differing on chips vs Nash and
chips vs random. Several distinct degenerate policies share an exploitability value, so in
this regime exploitability alone is a weak discriminator.

**Falsification, stated in advance:** if at matched sparsity shuffled performs like real,
the wiring is not contributing and the readout was carrying everything. That is a good
result. Write it down. Do not tune toward the recorded numbers — they are void.

**Worth following up separately:** why the shuffle makes the network hyperactive. The
shuffle preserves in- and out-degree but scatters each neuron's targets, so the fly's
inhibition — 478 GABA/glutamate neurons out of 8,246, plus the ALLN local neurons — stops
landing on the cells it controls. Targeted inhibition is the first thing degree-preserving
rewiring destroys. That is a real structural property, but it is an activity-level effect,
so it cannot be claimed as computation until both are compared at matched sparsity.

## 3. Task 2 — find out why the plasticity is inert (only after Task 1)

`frozen == real` on all four metrics in the first run — though on the second machine they
diverged slightly on chips vs random (+0.3750 real against +0.3333 frozen), so the premise
is weaker than recorded. And at 0.49% Kenyon cell activity there is almost nothing for the
plasticity to act on, which may be the entire explanation. **Do Task 1 first**; this may
resolve itself. Freezing the synapses the fly actually
modifies when it learns changes nothing.

Two candidate causes, in order of likelihood:

1. **The depression is too small to matter** next to the readout's delta rule. Check the
   magnitudes: `plasticity.apply()` returns the L1 of its weight delta — log it per hand
   against the readout's own update magnitude. If it is orders of magnitude smaller, that
   is the answer.
2. **The eligibility trace credits the wrong synapses.** `trace_decay` is 0.6 per decision
   and a Kuhn hand is 1–2 decisions, so the trace barely decays — every KC active in the
   hand gets equal credit. Worth testing `trace_decay` near 0 (credit only the last
   decision).

A clean isolation test: disable the **readout** update instead of the plasticity, and see
whether KC→MBON plasticity alone moves exploitability off 1.0833. If it cannot learn at all
on its own, the rule is wrong, not just weak. `decoder.update()` is called unconditionally
in `brain.reinforce()` — you will need a flag for this.

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
- **`calibrate_gain`'s default bounds `(0.0005, 0.0040)` are too low** to reach the ≈0.0050
  the real network needs on Kuhn states. Widen them before trusting a calibration result.
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
  decision.py     FlyDecision, a drop-in for the bot's Decision
  guard.py        the play-money declaration checkpoint
  cli.py          info / reference / calibrate / train / controls
```

Reference numbers worth memorising: equilibrium 0, best deterministic **1/6 = 0.1667**,
uniform 0.4583, never-bet 1.0, game value to player 1 **−1/18**.

Connectome attribution, required on anything published (CC-BY):

> MaleCNS v1.0 connectome, HHMI Janelia Research Campus & Google Research, CC-BY.
> https://male-cns.janelia.org/
