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

## The measuring instrument: Kuhn poker

The offline task is **Kuhn poker** (Harold W. Kuhn, 1950), the minimal two-player poker
game. It is here for one reason: **its optimum is known in closed form**, so a policy's
exploitability — what an optimal opponent would win against it — can be computed exactly
rather than estimated.

    Three cards, J < Q < K. Both ante 1, so the pot starts at 2. Each is dealt one card.
    One betting round, bet size 1. Player 1 checks or bets; a check lets Player 2 check
    (showdown) or bet, and a bet can be folded to or called. High card wins a showdown.

Twelve information sets, six per player. Every decision is binary — pass or aggress — so a
policy is one aggression probability per information set.

`kuhn.self_test()` checks the implementation against the published facts and runs in
every test session:

| check | value |
|---|---|
| information sets | 12 |
| game value to player 1, across the whole Nash α family | **−1/18** exactly |
| Nash exploitability, α ∈ [0, 1/3] | **0** exactly |

Every result is read against these fixed points:

| policy | exploitability (chips/hand) |
|---|---|
| equilibrium | 0 |
| **best deterministic policy** | **1/6 ≈ 0.1667** |
| aggress 50% everywhere | 0.4583 |
| never bet, never call | 1.0 |

The 1/6 row is the one that matters for this module. **Equilibrium in Kuhn requires
mixing**, so a fly reading off an argmax at temperature 0 cannot do better than 1/6 no
matter how good its circuit is. `best_pure_exploitability()` finds it by enumerating all
4096 pure policies; the optimum is the nit strategy — bet and call only with the King.

Evaluation is **entirely analytic**. Once the policy is read off the 12 information sets
(12 LIF simulations), exploitability and chips-per-hand against each fixed opponent follow
from the game tree in closed form. Sampling them would cost thousands of simulations and
add variance to quantities that have exact values.

A second, independent read comes free: **chips per hand against a Nash opponent, averaged
over both seats, is bounded above by 0 and reaches 0 only for optimal play.** No strategy
can beat the game value against an equilibrium opponent, so the fly gets at most −1/18 in
seat 1 and at most +1/18 in seat 2. It answers the same question exploitability does, by a
different route, and the two should move together — if they do not, the evaluator is
wrong.

### Why the previous task was replaced

The first offline task was a contextual bandit over one betting decision (`--task equity`,
still available). It was a bad instrument and the numbers said so: a bet-everything policy
scored well on mean regret while being wrong about which bet, a random policy scored well
on optimal-action-rate by folding often, and **the two metrics ranked the conditions in
opposite orders**. No amount of tuning on top of a broken measurement means anything. Kuhn
has no such hole — one number, exact, and a degenerate policy cannot flatter itself on it.

It also suits the circuit. The finding above was that the mushroom body separates discrete
cues but does not interpolate continuous ones. Kuhn's cues *are* discrete — three cards
— so nothing has to be quantised by hand, and the task still demands bluffing and
bluff-catching, which the equity game never did.

---

## Controls — the bar the viral demos skip

The flychess authors stated it plainly: *"we commit to reward-conditioned experiments and
frozen/shuffled controls before making any claim about learning or chess skill."* Almost
no fly-brain demo clears it. A connectome-shaped network can perform a task for reasons
that have nothing to do with the fly — the readout alone may be carrying it.

```bash
python -m poker.flybrain.cli reference          # the fixed points, and the self-test
python -m poker.flybrain.cli controls --hands 4000
```

| Control | What it breaks | What it proves |
|---|---|---|
| `shuffled` | degree-preserving edge rewiring | whether the *specific* wiring matters, or only coarse statistics |
| `frozen` | plasticity off | what the wiring computes vs what learning adds |
| `random` | policy | the floor |

**If `real connectome` does not beat `shuffled`, no claim about the fly's circuit is
supported.** Calibrate each condition to the same KC sparsity first.

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
the legality guarantees, the plasticity wiring and sign, save/load, the controls, the
Kuhn implementation (self-tested against the published game value and zero Nash
exploitability), and the offline harness. 54 tests pass on the synthetic connectome;
pylint 9.95/10. Calibration findings 1–4 above are reproducible.

### Kuhn result: learning is real, the policy is degenerate

4000 training hands against a 50/50 mix of an equilibrium and a uniform-random opponent,
real MaleCNS wiring, evaluated exactly:

| | exploitability (pure) | exploitability (mixed) | chips vs Nash | chips vs random |
|---|---|---|---|---|
| before | 1.0833 | 0.4901 | −0.1667 | −0.4583 |
| **after** | **0.3333** | 0.3333 | **−0.1111** | **+0.3333** |
| reference | 0.1667 best pure / 0 Nash | 0 Nash | 0 = optimal | 0.4583 max |

**All four metrics moved the right way together.** That is the thing the equity game could
never show: there, regret and optimal-action-rate ranked the conditions in opposite
directions. Here exploitability, the Nash ceiling and the exploitation of a weak opponent
all agree, which is the evidence that the instrument itself is sound. Against a random
opponent the fly captures 0.3333 of the 0.4583 available — **73% of the maximum
exploitation**.

But look at what it learned:

| | J | Q | K |
|---|---|---|---|
| open | bet | bet | bet |
| vs a check | **check** | bet | bet |
| facing a bet | call | call | call |

It aggresses almost everywhere. That is a degenerate policy, and three things follow from
it:

1. **It is 2× worse than the best deterministic policy** (0.3333 vs 0.1667). It is not
   close to good Kuhn play; it is good at beating a random opponent.
2. **Mixing collapsed.** `exploitability(pure) == exploitability(mixed)` means the readout
   saturated, so the softmax is effectively deterministic and the temperature knob does
   nothing. Equilibrium in Kuhn requires mixing, so in this state equilibrium is
   unreachable by construction.
3. **The opponent mix caused it.** Half the training opponent folds and calls at random,
   which pays aggression richly; the equilibrium half does not punish it hard enough to
   counteract that.

So: the module learns, measurably and reproducibly, and what it learns is "bet always".
Whether the *connectome* contributes to even that is still open — the Kuhn control runs
(real vs shuffled vs frozen) answer that and have not been run yet.

### Kuhn controls — VOID, and why

The run below is recorded for provenance only. **It does not support any conclusion**,
and it did not reproduce on a second machine (Windows / Python 3.14 / numpy 2.4.4), where
shuffled measured 0.3333 rather than 1.0833.

| condition | exploit (pure) | exploit (mixed) | chips vs Nash | chips vs random |
|---|---|---|---|---|
| real connectome | 0.3333 | 0.3333 | −0.1111 | +0.3333 |
| shuffled (degree-preserving) | 1.0833 | 0.9116 | −0.1250 | −0.4167 |
| frozen (no KC→MBON plasticity) | 0.3333 | 0.3333 | −0.1111 | +0.3333 |
| random actions | 0.4583 | 0.4583 | −0.1389 | 0.0000 |

**Why it is void.** Kenyon cell sparsity, measured on the 12 Kuhn information sets at the
default gain of 0.0026:

| gain | real KC active | shuffled KC active | real MBON | shuffled MBON |
|---|---|---|---|---|
| **0.0026 (default)** | **0.49%** | **13.02%** | 0.34 Hz | 84.33 Hz |
| 0.0040 | 3.05% | 26.99% | 2.96 Hz | 202.19 Hz |
| **0.0050** | **8.71%** | 30.96% | 10.53 Hz | 218.34 Hz |
| 0.0080 | 35.91% | 35.97% | 90.09 Hz | 242.37 Hz |

The two conditions were never at comparable operating points. At the default gain the real
network is **effectively silent** — 0.49% of Kenyon cells active, MBONs at 0.34 Hz — while
the shuffled network is **27× more active**. The comparison was between a silent network
and a hyperactive one, not between two wiring diagrams. The earlier wording here called
that "suggestive, not established"; that was far too generous. It is void.

Two further consequences:

- **The default `synaptic_gain` of 0.0026 is stale.** It was calibrated before the encoder
  gained tuning-curve quantisation, disjoint glomerulus banks and target-mean-rate
  normalisation, and was never re-derived afterwards. On Kuhn states the real network needs
  roughly **0.0050** to reach the ~9% sparsity the fly shows. Anything measured at the
  default is measured on a near-silent network.
- **That is also why the run did not reproduce.** At 0.49% KC activity and 0.34 Hz MBON
  output, almost nothing separates the action scores, so argmax decisions turn on
  float-accumulation order. Different platform, different BLAS, different degenerate
  policy. This is not seed sensitivity; it is an unusable operating point.

**Why shuffled is hyperactive** is itself interesting and worth following up: the shuffle
preserves every neuron's in- and out-degree but scatters each presynaptic neuron's
*targets*, so the fly's inhibition — 478 GABA/glutamate neurons out of 8,246, plus the
ALLN local neurons — stops landing on the cells it is supposed to control. Targeted
inhibition is the first thing degree-preserving rewiring destroys. That is a real
structural property of the connectome, but it is an activity-level effect, so it cannot be
claimed as "the wiring computes better" until both conditions are compared at matched
sparsity.

### Historical: the equity-game controls

These were measured on the task Kuhn replaced, and are kept only to record why it was
replaced. 900 training hands then 300 evaluation hands per condition:

| condition | regret (bb) | optimal action % |
|---|---|---|
| real connectome | 2.176 | 1.0% |
| shuffled (degree-preserving) | 2.393 | 22.3% |
| frozen (no plasticity) | **1.739** | 1.0% |
| random actions | 3.585 | 27.7% |

Real barely beat shuffled (~9%, inside the noise of 300 hands); turning plasticity off
made it *better*; and the two metrics ranked the conditions in opposite orders. The third
of those is what condemned the task rather than the fly — a measurement that disagrees
with itself cannot answer the question. Earlier runs did show regret roughly halving
within a condition (9.48 → 4.52), so the machinery learns something.

So: a working instrument and an honest negative result, not a fly that plays poker.
Saying so is the point. `flychess` reports no performance metrics at all, and `Stonkfly`
and `OpenFly` both state that no profitable edge has been demonstrated.

### What to do next, in order

1. **Re-derive the operating point, then re-run the controls with per-condition
   calibration.** This is not one item among several; until it is done there is no result
   here at all. Concretely: the default gain must be re-derived for the current encoder
   (real needs ≈0.0050 on Kuhn states, not 0.0026), `cmd_controls` must call
   `controls.calibrate_gain()` for each condition separately and report the gain it chose,
   and `calibrate_gain` must be fed Kuhn stimuli
   (`[kuhn.KuhnTable(c, h) for c, h in kuhn.INFO_SETS]`) rather than equity-game ones. Then
   several seeds, reporting spread.
2. **Report chip metrics alongside exploitability.** Three of four conditions landed on
   exactly 0.3333 in the second run while differing on chips, so several distinct
   degenerate policies share an exploitability value. Exploitability alone is a weak
   discriminator in that regime.
3. **Work out why the KC→MBON plasticity is inert** — but only after 1, since at 0.49% KC
   activity there is almost nothing for it to act on, which may be the whole explanation.
4. **Stop the readout saturating**, so a mixed policy survives extraction. Equilibrium in
   Kuhn needs mixing.
5. **Re-weight the training opponent.** A 50/50 equilibrium/random mix pays blanket
   aggression too well, and blanket aggression is what it learned.
6. Only then: Leduc poker, whole-brain scope, or more MBON readout capacity.

### Not verified in this environment

`FlyDecision.make_decision` has not been run against a real scraped table.
`Decision.__init__` needs ~40 scraped attributes, a strategy sheet and a curve fit, and
the repository's own `test_decision.py` is skipped (`"Fix access issues"`) because it
needs a MongoDB connection and screenshot fixtures. What is verified: `FlyDecision`
subclasses `Decision`, its mode/guard/fallback logic is tested against a stubbed base
class, and `main.build_decision` falls back to the rule-based `Decision` on any failure.
First live run should be shadow mode, watching the log for `fly_error`.
