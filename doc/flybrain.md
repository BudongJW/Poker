# Fly-brain poker decision module

A connectome-constrained decision module built on **MaleCNS v1.0**, the first complete
connectome of an adult male *Drosophila* central nervous system (HHMI Janelia Research
Campus + Google Research, released 2026-09-03; 166,700 neurons). It instantiates the
fly's mushroom body from the published wiring and uses it to choose a poker action.

**Connectome data is licensed CC-BY.** Any published result must carry the attribution in
`connectome.ATTRIBUTION`:

> MaleCNS v1.0 connectome, HHMI Janelia Research Campus & Google Research, CC-BY.
> https://male-cns.janelia.org/

---

## Read this first: what this can and cannot be

A fly cannot play poker. Its central nervous system has no circuit for multi-street
betting trees, opponent modelling, bluffing, or mapping 169 starting hands onto equity.
Anyone claiming otherwise is describing a demo, not a result.

What the fly *does* have is the circuit that solves the part of poker that is a
value-based choice under uncertainty:

- **Rajagopalan et al., PNAS 120:e2221415120 (2023).** Individual flies in a dynamic
  foraging task with probabilistic rewards follow Herrnstein's operant matching law and
  track changing reward probabilities over time. The mechanism is synaptic plasticity in
  the mushroom body that incorporates an *expectation* of reward; bypassing the
  expectation abolishes matching.
- **Bennett et al., Nat Commun 12:2569 (2021).** A reduced mushroom body model solves a
  multi-armed bandit, with dopaminergic neurons computing a reinforcement prediction
  error from MBON feedback rather than reporting absolute reward.

So the honest framing, and the one this module is built around:

> The fly brain is not asked to play poker. It is asked to make a value-based choice
> under uncertainty, and poker is the presentation layer.

The existing Monte Carlo computes equity. That number is handed to the fly as a sensory
channel. **The fly is not computing equity**, and the module says so in code and in logs
rather than blurring it — which is the main thing separating this from the viral
fly-brain demos, where an external learner does the work while the connectome supplies a
fixed scaffold.

---

## The circuit

Extracted from MaleCNS v1.0 by cell class. Counts are measured, not estimated:

| Population | Class annotation | Count | Role |
|---|---|---|---|
| ORN | `olfactory` | 2,639 | sensory input channels |
| ALPN | `ALPN` | 686 | antennal lobe projection neurons |
| ALLN | `ALLN` | 420 | local neurons, inhibitory gain control |
| KC | `Kenyon_Cell` | 4,064 | sparse high-dimensional recoding |
| MBON | `MBON` | 97 | action readout |
| DAN | `DAN` | 340 | teaching signal |

DANs split by cluster, which is what lets reward and punishment be wired separately:

- **PAM clusters — 316 cells.** Appetitive. Driven by chips won.
- **PPL1 clusters — 16 cells, including the PPL101 pair.** Aversive. Driven by chips
  lost. (PPL101 is the same pair DOOMFLY drove with in-game damage.)

Measured connectivity at `weight >= 3`: ORN→ALPN 20,742 edges, ALPN→KC 20,777, KC→MBON
44,042. Whole subcircuit: **8,246 neurons / 451,855 edges**.

### Why the subcircuit and not the whole brain

| Scope | Neurons | Edges (w≥3) | Practical |
|---|---|---|---|
| Mushroom body | 8,246 | 451,855 | runs in ~0.2 s/decision on CPU |
| Whole brain (`Traced`) | 165,122 | 10,511,038 | feasible, much slower |

The decision organ is small. That is why this needs no GPU, while the open-source
fly-brain projects that simulate all 165k neurons hit GPU as their bottleneck.

---

## Findings from calibration — read before trusting any number

The connectome gives **wiring, not dynamics**. Time constants, thresholds and the gain
that turns a synapse count into a voltage are all calibration choices. Four things were
measured during development, and all four constrain what this module can claim.

### 1. Naive LIF on the raw connectome saturates

At the first-guess gain, 91.8% of Kenyon cells fired and population rates hit 200–380 Hz.
Real KCs code sparsely (~5–10% active for a given odour) and fly central neurons do not
sustain those rates. Saturated KCs make every poker state look identical to the MBONs,
which makes the KC→MBON plasticity useless. Sparsity is not cosmetic here.

### 2. The usable gain window is narrow

Sweeping `synaptic_gain`, the network goes from silent to saturated between 0.0015 and
0.0030. There is no broad plateau. **Any comparison between connectome variants must
re-calibrate**, because a rewired network does not sit at the same operating point —
otherwise the comparison is between activity levels, not between wiring diagrams. Use
`controls.calibrate_gain()`.

### 3. Poisson sensory input puts the network in a chaotic regime

With Poisson spike input, discriminability measured as (between-state distance) /
(within-state distance) was **d' ≈ 1.0 at every gain and window tried**, including a
5× longer window. Two runs of the *same* state diverged as much as two different states,
and lengthening the window did not average it out — 451k recurrent edges and a spiking
nonlinearity are enough to amplify sampling noise indefinitely.

Graded **current injection** gives within-state distance 0.00 (bit-for-bit reproducible)
and between-state distance ~70. `sensory_mode='current'` is therefore the default;
`'poisson'` is kept for ablations and should not be used to produce results.

### 4. The mushroom body separates patterns; it does not interpolate

This is the most important finding and it changed the design.

Decoding a *continuous* equity from MBON activity:

| | R² |
|---|---|
| in-sample | 0.667 |
| out-of-sample | **−6.88** |

39 of 40 equities produced distinct MBON codes, but the distances between them were
non-monotone in equity. The circuit produces a **discriminative, non-generalising code —
a sparse random hash**. Equity 0.50 and 0.52 map to unrelated patterns. A readout can
memorise training states and cannot interpolate to new ones. Graded APL-style inhibition
and wider tuning curves both failed to fix it; hard top-k winner-take-all made it worse
(it turns the KC response into a step function of the input).

**That is not a bug. It is what the Kenyon cell expansion is for.** Pattern separation
exists so that similar odours can be learned about *independently*. Asking it to
interpolate a continuous variable is asking it to do the opposite of its job.

So features are **quantised** to a small number of bins before encoding
(`quantize_features=True`), which hands the fly what the literature actually tested it
on: a modest set of discrete cues, each of which it can learn a value for. Incidentally
this is closer to how human players think — hand-strength classes, not a continuous
number.

---

## Pipeline

```
table state ──> features (14)            encoding.features_from_table
            ──> quantise to bin centres  encoding.OdourEncoder
            ──> ORN firing rates         one disjoint glomerulus bank per feature
            ──> 100 ms LIF simulation    lif.LIFEngine, circuit from MaleCNS v1.0
                ORN → ALPN/ALLN → KC → MBON
            ──> MBON rates ──> action scores   decoding.ActionDecoder
            ──> legality mask ──> action       decoding.legal_actions

hand settles ──> chips won/lost
             ──> reward prediction error  plasticity.reward_prediction_error
             ──> PAM (gain) / PPL1 (loss) drive
             ──> depression of KC→MBON on the eligibility trace
```

The MBON→action readout is **learned, not assumed**. Published MBON valence assignments
do not cover every type in MaleCNS v1.0, and hard-coding a guess would bake in the
answer. Which MBONs end up driving which action is reported as a result
(`ActionDecoder.mbon_contributions`). Wiring from ORN to MBON is fixed by the connectome;
only the small final readout is fitted — which is what "connectome-constrained" means in
the literature.

---

## Run modes, and the play-money constraint

PokerStars' terms prohibit automated play, and that prohibition **covers play-money
tables too** — the exposure there is the account, not a balance. This repository's
operator plays points/play-money games only, so that constraint is enforced in code
rather than left as an intention.

| Mode | Behaviour |
|---|---|
| `shadow` | **Default.** The fly decides, its choice is logged next to the rule-based one, and the rule-based decision still drives the mouse. Nothing about the account's behaviour changes. |
| `active` | The fly drives the mouse. Refused unless `confirmed_play_money=true` is set by hand. |
| `offline` | No scraper. Self-play and replay only. |

`PlayMoneyGuard` is a **declaration checkpoint, not a safety net** — it cannot detect a
cash table on its own. It is deliberately impossible to satisfy by accident.

Start in shadow mode. It produces paired `(state, fly action, baseline action, outcome)`
records at zero risk, and those records are exactly the training data the fly needs.

---

## Controls — the bar the viral demos skip

The flychess authors stated it plainly: *"we commit to reward-conditioned experiments and
frozen/shuffled controls before making any claim about learning or chess skill."* Almost
no fly-brain demo clears it. A connectome-shaped network can perform a task for reasons
that have nothing to do with the fly — the readout alone may be carrying it.

```bash
python -m poker.flybrain.cli controls --hands 1000
```

| Control | What it breaks | What it proves |
|---|---|---|
| `shuffled` | degree-preserving edge rewiring | whether the *specific* wiring matters, or only coarse statistics |
| `frozen` | plasticity off | what the wiring computes vs what learning adds |
| `random` | policy | the floor |

**If `real connectome` does not beat `shuffled`, no claim about the fly's circuit is
supported.** Calibrate each condition to the same KC sparsity first.

---

## Setup

```bash
pip install numpy scipy pyarrow          # pyarrow only for reading the bulk files

python -m poker.flybrain.cli info        # downloads ~1.1 GB on first run, then caches
python -m poker.flybrain.cli calibrate
python -m poker.flybrain.cli train --hands 2000 --save poker/data/flybrain/brain.npz
python -m poker.flybrain.cli controls --hands 1000
```

Downloads land in `poker/data/flybrain/` (already gitignored). Only three of the published
files are needed — the 6.8 GB synaptic-partner and 12.7 GB synapse-point files are not,
because the segment-to-segment weight table already *is* the adjacency matrix:

| File | Size |
|---|---|
| `body-annotations-...feather` | 14 MB |
| `body-neurotransmitters-...feather` | 42 MB |
| `connectome-weights-...feather` | 1.0 GB |

The first build scans all 151,856,684 edges (~27 s) and caches the result as `.npz`.
Every pass over the weight table is batched: materialising a mask over all 151.8 M rows
at once exhausts a normal container.

Without the download, everything still runs on a structurally-matched **synthetic**
connectome with the real population sizes and edge count. It exercises every code path
and says nothing whatsoever about the fly; `Connectome.synthetic` is `True` so callers and
logs can refuse to report it.

### config.ini

```ini
[flybrain]
mode = shadow
scope = mushroom_body
weight_threshold = 3
plasticity = true
learning_rate = 0.10
play_money_only = true
confirmed_play_money = false
```

A missing `[flybrain]` section, a bad value, or a missing connectome all fall back to the
rule-based `Decision`. The bot must never stop playing because the fly failed.

---

## Status

Working and measured: the loader, the circuit extraction, the LIF engine, the encoding,
the legality guarantees, the plasticity wiring and sign, save/load, the controls, and the
offline harness. 37 tests pass on the synthetic connectome; pylint 9.99/10. Calibration
findings 1–4 above are reproducible.

**It has not been shown that the connectome contributes.** On the offline equity game,
900 training hands then 300 evaluation hands per condition:

| condition | regret (bb) | optimal action % |
|---|---|---|
| real connectome | 2.176 | 1.0% |
| shuffled (degree-preserving) | 2.393 | 22.3% |
| frozen (no plasticity) | **1.739** | 1.0% |
| random actions | 3.585 | 27.7% |

Three things to read off this, none of them flattering:

1. **Real barely beats shuffled** (2.176 vs 2.393, ~9%), which is inside the noise of 300
   hands. By the standard this module set itself, that means the specific wiring is not
   demonstrably doing anything.
2. **Turning plasticity off makes it better** (1.739). The learning rule as implemented
   is a net negative, so the KC→MBON depression and the readout update are fighting each
   other or the credit assignment across a hand is wrong.
3. **Regret and optimal-action-rate disagree completely.** Low-regret policies almost
   never pick the exact best action, and the high-optimal-rate policies have worse regret.
   The equity game rewards betting often enough that a bet-heavy policy scores well on
   regret while being wrong about which bet; a random policy folds sometimes and collects
   the hands where folding was optimal. **The task is not yet a good measuring
   instrument**, and fixing that comes before any further tuning.

Earlier training runs did show regret roughly halving within a condition (9.48 → 4.52),
so the machinery learns *something* — it just does not beat its own controls.

So: a working instrument and an honest negative result, not a fly that plays poker.
Saying so is the point. `flychess` reports no performance metrics at all, and `Stonkfly`
and `OpenFly` both state that no profitable edge has been demonstrated.

### What to do next, in order

1. **Replace the equity game with Kuhn poker**, then Leduc. Both are solved, so distance
   from the known optimal strategy is exactly measurable and a single number cannot be
   gamed by a degenerate policy. This is the prerequisite for trusting anything else.
2. **Fix credit assignment.** Reinforcing every decision in a hand with the hand's final
   payoff is crude; the frozen-beats-plastic result points here first.
3. **Re-calibrate each condition to equal KC sparsity** before comparing, using
   `controls.calibrate_gain()`. The real and shuffled networks do not currently sit at the
   same operating point, and the gain window is narrow.
4. Only then consider whole-brain scope, or more MBON readout capacity.

### Not verified in this environment

`FlyDecision.make_decision` has not been run against a real scraped table.
`Decision.__init__` needs ~40 scraped attributes, a strategy sheet and a curve fit, and
the repository's own `test_decision.py` is skipped (`"Fix access issues"`) because it
needs a MongoDB connection and screenshot fixtures. What is verified: `FlyDecision`
subclasses `Decision`, its mode/guard/fallback logic is tested against a stubbed base
class, and `main.build_decision` falls back to the rule-based `Decision` on any failure.
First live run should be shadow mode, watching the log for `fly_error`.
