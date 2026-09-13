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

| | exploitability (pure) | chips vs random |
|---|---|---|
| real connectome | **0.3333** | **+0.3333** |
| shuffled (degree-preserving) | **1.0833** | −0.4167 |
| frozen (no KC→MBON plasticity) | **0.3333** | +0.3333 |
| random actions | 0.4583 | 0.0000 |
| *best deterministic policy* | *0.1667* | — |
| *equilibrium* | *0* | — |

Two things to carry forward:

- **Real beats shuffled 3.25×.** Suggestive that the wiring does work. Not yet established
  — see task 1.
- **`frozen` is bit-for-bit identical to `real`.** The mushroom body plasticity rule is
  contributing nothing; all the learning is in the readout delta rule. See task 2.

---

## 2. Task 1 — sparsity-matched controls (do this first, nothing else before it)

This is the one thing standing between "suggestive" and "established" on the central
question. Do not tune anything else until it is done.

**Why.** The usable `synaptic_gain` window is narrow — the network goes from silent to
saturated between 0.0015 and 0.0030, with no plateau. A rewired network does not sit at the
same operating point as the real one. The shuffle also dropped 227 self-loops, leaving
430,221 edges against 451,855 (4.8% fewer). So part of "shuffled cannot learn" may simply
be "shuffled is not at a working operating point," and the 3.25× gap would then be
measuring activity level rather than wiring.

**Two concrete gaps in the code, both small:**

1. `cmd_controls` in `poker/flybrain/cli.py` **never calls `calibrate_gain`.** Every
   condition is built with the same config, so nothing is matched. The fix is to calibrate
   each condition's gain before training it and log the gain that was chosen alongside its
   result.
2. `_representative_tables()` (cli.py:134), which `cmd_calibrate` feeds to
   `calibrate_gain`, builds **equity-game** tables. For the Kuhn task you want sparsity
   measured on Kuhn stimuli — `[kuhn.KuhnTable(card, history) for card, history in
   kuhn.INFO_SETS]` is the right set, and it is exactly 12 states.

`controls.calibrate_gain(brain_factory, tables, target_sparsity=0.09)` already exists and
bisects correctly; it just is not plumbed in.

**Then:** run 5+ seeds per condition and report the spread, not one number. Exploitability
is exact given a policy, but which policy training lands on is not.

**What would falsify the finding:** if, once both are at 9% KC sparsity, shuffled lands
near 0.3333 too, then the wiring was never the thing and the readout was carrying it.
That is a perfectly good result — write it down.

---

## 3. Task 2 — find out why the plasticity is inert

`frozen == real` on all four metrics, exactly. Freezing the synapses the fly actually
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
