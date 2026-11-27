# Lab notebook

The full write-up behind the [README](../README.md): every experiment, control, and correction,
in the order they were run. Numbers are test loss (nats/char) unless marked val.

## First tournament

**What the first run says** (one seed, 5 minutes, so hold it loosely)
- **Backprop still wins**, and it gets there fast: it peaks after 1 minute, then starts memorising.
- **Sign symmetry is within 0.05 nats.** A backward pass that only knows the *signs* of
  the weights, not their size, is nearly as good as the real gradient here.
- **Local per-block losses come third (+0.11)** with no gradient ever crossing a block
  boundary, and they overfit far more slowly than backprop.
- **Feedback alignment (+0.28) is still improving at minute 5.** Random feedback works,
  just slower, because the forward weights first have to "align" with it.
- **Forward-Forward only barely beats a frozen random network** (2.698 vs 2.716): its blocks
  learn something, but almost none of it helps predict the next character.
- **The classic no-backward-pass rules all lose to the frozen random network.** ES, MeZO,
  forward-gradient and hill-climbing jiggle *weights*, and one loss number per forward pass
  is hopeless signal for 830k of them.
- **🐀 `lab_rats` is the first no-backward-pass rule to beat the floor** (2.649 mean over 3 seeds vs 2.716,
  and ahead of Forward-Forward). It jiggles *neurons* instead, independently at every one of
  the 4,096 tokens in a batch, so two forward passes run 4,096 tiny experiments at once.
  Its gradient estimate lines up with the true gradient at cosine 0.01–0.16 per layer,
  vs ~0.0003 for a MeZO-style random direction. Still +1.0 nats behind backprop, and still
  improving at minute 5: plenty of room in this bracket. Go win it.
- **📊 …and then plain counting beat all of them.** `wire_together` does no optimisation at all:
  it tallies which character follows which and writes the counts into the weights in closed
  form, turning the transformer into an exact bigram model. **2.500, in under a minute**,
  ahead of every gradient-free *learner* here, Forward-Forward, and the random reservoir.
  **This is the real bar for the 🚫 bracket:** a rule that can't beat counting letter
  pairs hasn't learned anything a census couldn't. To beat it, a rule has to make use of
  *earlier* characters, which means training attention.
- **🔎 A census can teach attention with a layer-local rule.** `census_heads` scores **2.369**
  (3 seeds) vs 2.488 for the same network with frozen heads and 2.500 for counting. It never
  calls autograd, but its Q/K update is the chain rule hand-written inside one layer, so it is
  🧱 local. It was first published here as 🚫 none; that was wrong, and counting
  is still the bar for the 🚫 bracket.
- **👵 The bar moved, by counting more.** `grandmother_cells` (🚫 none) scores **1.900** over 3
  seeds by giving each MLP neuron one string to remember. It is n-gram counting compiled into
  the weights, with hand-wired attention, so it is a better *counter*, not a better learner.
- **🧬 Neurons that die and breed work; being choosy about where they are born does not obviously
  help.** `natural_selection` (🧱 local) scores **1.896** over 3 seeds. Killing the *useless* cells
  matters a lot (random deaths: 2.068). Births at surprising places: 1.896 vs 1.902 at random
  places, which is inside the noise.
- **🪞 `mirror_neurons` beats the 5-gram table**, untuned (1.806) and tuned (1.702). No 🚫 rule beats
  either; `local_losses` (1.743) beats the untuned one and sign symmetry (1.675) both. Its backward
  weights start random and receive the same update as their forward twins, so no weight is ever
  copied. 1.641 over 3 seeds, 0.008 behind backprop (1.633; a later rerun gave 1.635), where fixed
  random feedback got 1.911.
  An earlier version called the regularised pair a tie (1.544 vs 1.551). That pair was tuned on
  mirror's screen; on one shared grid backprop wins, 1.553 vs 1.584. Details below.
- **📚 A lookup table teaching a backprop-free network beats backprop.** `taught_mirror` (🪞 modified)
  counts the batches it sees into a smoothed 9-gram table (no gradients), then `mirror_neurons`
  learns 30% from the table's predictions. **1.596** over 3 seeds vs **1.635** for backprop, rerun
  in the same session: it wins every seed pair, by 0.039 on average. **The catch:** backprop with
  the same teacher does better still (1.585). The win comes from the counting teacher, not from
  mirror credit being better than the real gradient. Details below.
- **Plain tables are a higher bar than this README used to say.** Untuned 5-gram: 1.806 on the
  arena's own windows. With smoothing tuned on val: 5-gram 1.702, 9- and 10-gram about **1.615**,
  better than unregularised backprop (1.626 to 1.641 over six runs). See the table in the mirror section.
  (An earlier version quoted 2.504 / 2.113 / 1.865 / 1.782 for 2- to 5-grams: untuned, and scored on
  continuous text instead of 128-char windows.)

### 🐀 Lab rats lab notebook: three attempted fixes

The diagnosis: lab_rats' later layers get decent signal, but **block 0 gets ~none**
(cosine ≈ 0 with its true gradient), and attention `qkv` gets ~none in every block.
Three fixes, each its own rule, each run on 3 seeds (test loss, lower is better):

| rule | the fix | seed 0 | seed 1 | seed 2 | mean | verdict |
|---|---|---:|---:|---:|---:|---|
| `lab_rats` | (original) | 2.650 | 2.651 | 2.647 | 2.649 | |
| `lab_rats_maze` | block 0 gets its own little readout and is credited mostly by that nearby loss (+30% global) | 2.610 | 2.626 | 2.623 | **2.620** | ✅ wins all 3 seeds |
| `lab_rats_cycle` | jolt one block per step, round-robin: less crosstalk, same cost | 2.611 | 2.662 | 2.610 | 2.628 | 🤷 wins 2 of 3, noisy |
| `lab_rats_future` | attention jolts credited with the loss change at all *later* tokens (returns-to-go, γ=0.9) | 2.660 | 2.652 | 2.644 | 2.652 | ❌ no gain |
| `lab_rats_solo` | every block jolted in its own pair of passes (5× cost) | 2.692 | | | | ❌ clean signal, too few steps |

What the gradient probes showed along the way (cosine of the estimate with the true gradient):
- **Jolting one block at a time** cleans up the *later* layers a lot (readout −0.01 → +0.23,
  block 3 → +0.14), but block 0 barely moves. Running all 5 blocks separately costs 5× the
  passes and loses; taking turns keeps the cost and is a coin flip.
- **Cross-token credit** does partly un-blind attention (block-2 `qkv` −0.03 → +0.06) but
  pours noise onto layers that only affect their own token (readout 0.67 → 0.14 at γ=1).
  Restricted to attention, it nets out to nothing.
- **Local readout for block 0** wins, but *not* the way intended: block 0's attention and MLP
  weights stay at cosine ≈ 0.01 even against a loss one step away. Only the embedding clearly
  improves (0.06 → 0.17). Probably the gain is a better-trained embedding. It also starts
  slower (worse at minute 1) because the readout has to learn first, then overtakes by minute 5.
- **The real wall:** every token's loss change is one number shared by thousands of jolted
  neurons, and block 0's MLP and attention have a tiny share of it. Moving the loss closer
  doesn't change that ratio. Ideas welcome: per-layer noise scaling, low-rank jolts, or
  learning a local *predictor* of the loss change (synthetic-gradient style).

### 🧪 The extreme batch: counting, instincts, and dreams

| rule | the idea | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---:|---:|---:|---|
| `wire_together` | "Fire together, wire together", literally: count letter pairs, compile them into the weights. Zero optimisation. | 2.500 | | | 📊 the bar for 🚫 rules; deterministic, so one run |
| `instinct_rats` | Born wired as a bigram, then `lab_rats_maze` keeps learning on top. Nature, then nurture. | 2.498 | 2.498 | 2.499 | 🤷 nurture adds nothing |
| `dreaming` | Awake: strengthen what fires on real text. Asleep: weaken what fires on the model's own dreamt text. | 3.331 | 3.322 | 3.323 | 💤 worse than readout-only |

- **Counting is exact.** The compiled transformer's loss matches a lookup-table bigram to 4 decimals (2.4552 both ways).
- **Instincts are fragile, then frozen.** The first version had 1-sized one-hots, and lab rats
  wrecked the wiring within seconds even at lr=1e-6: `ln_f` divides by a tiny std, so any
  trickle the blocks leaked into the stream got amplified ~11×. Bigger one-hots fix that, since
  LayerNorm erases scale, but then the blocks barely move the output, and the lr sweep
  walks the learning rate down to ~0. Either way lab rats can't get past the bigram:
  doing that needs attention, the one thing it can't train.
- **Dreaming increases loss over the readout alone.** The readout
  alone gets an honest delta-rule signal and reaches 2.73 (≈ the random reservoir). Switch on
  the wake/sleep Hebbian in the hidden layers and test loss rises to 3.32–3.33. Contrasting real text
  against the model's own early dreams (≈ random letters) apparently teaches the features
  "is this real text?" and nothing about *which* character comes next.
  (Relabelled 🚫 → 🧱: its readout is the delta rule, held to the same standard as `reservoir`.)

### 🔎 Census heads: training attention with a layer-local rule

`census_heads` trains *only* attention (plus the readout), with no error signal crossing a layer:

1. **The stream is wired, not learned:** `[one-hot current char | sinusoidal position | what the heads fetched]`.
   Each head's values are a fixed random code of the character it lands on; every other block output is zeroed.
2. **E-step:** a running census of (current char, char *d* back, next char) gives, for each query,
   the posterior over which earlier position caused the next char:
   `posterior(s) ∝ attention(s) · P(y | x_t, x_s, d) / P(y | x_t)`.
3. **M-step:** delta rule on the head's own scores, `dS = attention − posterior`, turned into Q/K
   updates through the head's bilinear form. A test pins this update to autograd of the
   equivalent per-head loss, so it is exactly a *local* attention gradient (hence 🧱, not 🚫).
4. **Readout:** the delta rule (exact last-layer gradient, computed by hand).

| variant (`census_heads`, 2 heads, 5 min) | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| heads **frozen** at random init (never trained) | 2.490 | 2.486 | | 2.488 |
| heads **trained by census EM** | 2.369 | 2.368 | 2.371 | **2.369** |
| heads **hand-wired** to look exactly 1 and 2 back (a reference, not a proven bound) | 2.288 | 2.290 | | 2.289 |
| `census_heads_4`: 4 census heads, half the stream dims each | 2.408 | 2.400 | 2.400 | 2.403 |

**Corrections to the first published version of this section (commit 0e0ded8):**
- It was labelled 🚫 none. It is 🧱 local: the guard only catches autograd, and this update is the
  chain rule hand-written inside one attention layer.
- It said the self-prior was essential. It is not: over 3 seeds and 5 minutes each, a uniform prior
  scores 2.368 / 2.367 / 2.366, the self-prior 2.369 / 2.368 / 2.371. Sharper-looking attention did
  not mean a better model.
- It said 29% of attention fell beyond the census horizon. That number counted everything more
  than 5 back; the horizon is 8.
- It explained the gap to the hand-wired heads as "blending", the 4-head result as cramped codes,
  and the early plateau as a capacity limit. Those are untested guesses.
- Hand-wired heads are a reference, not a proven upper bound.
- Value codes are fixed across seeds, so the spread between seeds understates the real variance.

**Position-only control.** How much do the census counts contribute to the attention
teacher? Set `attn=position-only` to give every allowed offset a census log-likelihood
ratio of zero. Both modes use the same attention prior, masks, counting schedule, Q/K
optimizer, and readout. The attention prior and readout can still use character content.

To compare the two modes under the leaderboard's five-minute MPS budget, use the
same seed, batch size, and learning rate for both runs.

```bash
for mode in census position-only; do
  python -m bbp.arena --rule census_heads --device mps --minutes 5 --seed 0 \
    --batch-size 32 --set attn="$mode" lr=0.001 --no-test --tag "${mode}-seed0"
done
```

Compare `best_val` in the two `results/census_heads-*-seed0-val-only.json` files.
The `steps` field shows how many updates each mode completed within the time budget.

### 🧠 Census MLP: census heads plus a hidden layer taught by random feedback

`census_heads` can only *add up* evidence from the current and fetched characters. `census_mlp` adds
one MLP trained by direct feedback alignment (the output error, broadcast through fixed random
matrices). Label: 🧱 local, and that is a judgement call. Nothing is backpropagated from block to
block, but a global error is broadcast, and it goes through `ln_f`'s hand-written Jacobian.

| variant (5 min) | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| **census_mlp** (2 census heads + 1 DFA MLP) | 1.954 | 1.965 | 1.964 | **1.961** |
| heads frozen at random init | 2.440 | 2.439 | | 2.440 |
| heads hand-wired to 1 and 2 back | 1.871 | | | |
| MLP frozen (heads and readout only) | 2.363 | | | |
| uniform prior instead of self-prior | 2.071 | | | |
| no `ln_f` correction on the feedback | 2.230 | | | |
| exact feedback instead of random (*diagnostic*, hand-written backprop) | 1.937 | | | |
| 4 heads instead of 2 | 1.976 | | | |

- The MLP is what takes `census_heads` from 2.369 to about 1.96, and *where the heads look* matters
  as much as the MLP. Both parts are needed.
- Replacing random feedback with the exact error gains only 0.017 on test (one seed), though val
  improves more, 1.837 to 1.775. So how much random feedback costs is unsettled.
- The uniform prior hurts here (2.071 vs 1.954, one seed) though it makes no difference to
  `census_heads`. Do not lean on one seed.
- 2-minute runs: 1 / 2 / 4 stacked MLPs score 1.966 / 1.984 / 2.024. Deeper did not help.
- It loses to `local_losses` (1.743). It also trails `natural_selection` (1.896), but that rule is
  given hand-wired copy heads; with its own heads hand-wired, `census_mlp` gets 1.871 (one seed).

### 👵 Grandmother cells: one neuron per memory

Head h is **hand-wired** to copy the character h+1 back. Each MLP neuron is an AND-gate for one
recent string ("the ", "ROME"). Counting decides which strings get one of 2,048 neurons (most
log-likelihood gained over the one-char-shorter string), and each neuron adds
`log P(next | string) − log P(next | shorter string)`, so the cells along a suffix chain telescope.
No gradient, no error signal. It counts for 2,000 batches, then stops learning.

| | test |
|---|---:|
| **grandmother_cells** (seeds 0 / 1 / 2) | 1.899 / 1.899 / 1.903 |
| bigram table, untuned (`wire_together` is this) | 2.508 |
| trigram table, untuned | 2.122 |
| 4-gram table, untuned / tuned (11,228 contexts) | 1.885 / 1.838 |
| 5-gram table, untuned / tuned (48,539 contexts) | 1.806 / 1.702 |

- It sits between the 3- and 4-gram tables because it has 2,048 neurons, not 11,228 contexts. This
  is compression of a table, and a table with no size limit does better.
- `kappa=12` was picked by the same 1-minute sweep every rule gets.
- The smoothing constant assumes a corpus of about 1M characters.
- Cells misfire on the first 4 positions of a window, where the copied characters do not exist yet.
  Known, not fixed, small effect on the average.
- Seed changes the batches counted, not the wiring, so the spread understates the real variance.

### 🧬 Natural selection: neurons that die and breed

Starts with 65 single-character neurons. Output weights follow the delta rule. Every 50 steps the
5% of cells that move the logits least are killed, and mutants of living cells (one more character
constraint, or one fewer) take their places. Which strings the hidden neurons detect is never set
by an error gradient. Label 🧱 local: the delta rule is the one-layer readout gradient.

| variant | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| **natural_selection** (surprise births, useless cells die) | 1.905 | 1.903 | 1.879 | **1.896** |
| births at random places | 1.910 | 1.901 | 1.895 | 1.902 |
| **deaths at random** | 2.072 | 2.063 | 2.068 | 2.068 |
| no evolution (2 min; 65 fixed cells, so about a bigram) | 2.499 | | | |
| start from a full `grandmother_cells` network | 1.931 | | | |

- **Selection is doing real work:** random deaths cost 0.17. Evolution takes it from 2.50 to 1.90.
- **The "born where it is surprised" idea is not established.** The gap to random births (0.006) is
  inside the seed-to-seed spread (0.03), and the sign flips on seed 1.
- Starting from a counted network did not help: test 1.931 (val 1.799), one seed, vs test 1.905
  (val 1.804) from 65 single characters.
- `gen_every`, `cull` and the surprise switch were chosen on val in a small 1-minute seed-0 sweep;
  `lr` in the standard sweep. The ablations reuse those settings without their own sweep.
- Like `grandmother_cells` it uses hand-wired copy heads, and the plain 4-gram table (1.885 untuned, 1.838 tuned) beats
  it. It reaches the same loss as counting by trial and error, not a better one.

### 🪞 Mirror neurons: beating the 5-gram table

**The bar.** Plain n-gram tables, scored on exactly the arena's windows (`scripts/ngram_reference.py`),
smoothing (Witten-Bell) tuned on val. They have unlimited memory, up to 780k (context, next) pairs, and
do not fit in this 0.83M-parameter model, so they are a reference, not a contender:

| order | untuned (k=4) | tuned |
|---|---:|---:|
| 4 | 1.885 | 1.838 |
| 5 | 1.806 | 1.702 |
| 6 | 1.838 | 1.649 |
| 7 | 1.899 | 1.633 |
| 8 | 1.965 | 1.616 |
| 9 | 2.019 | 1.614 |
| 10 | 2.061 | 1.615 |

**The rule.** `feedback_alignment` sends the error back through *fixed* random matrices. In
`mirror_neurons` those matrices are separate and random at birth, but plastic: each gets the same
Hebbian product (error x activity) as the forward synapse joining the same two neurons. Nothing is
copied. The idea is Kolen & Pollack (1994) and Akrout et al. (2019), not ours. It is an end-to-end
backward pass with a learned stand-in for Wᵀ, so 🪞 modified, not 🚫.

| test loss, 5 min | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| `backprop` | 1.629 | 1.630 | 1.639 | **1.633** |
| `backprop`, rerun later (same session as `taught_mirror`) | 1.638 | 1.641 | 1.626 | **1.635** |
| **`mirror_neurons`** | 1.638 | 1.645 | 1.639 | **1.641** |
| control: backward weights frozen (feedback alignment) | 1.898 | | | |
| `backprop (reg)`, shared grid | 1.567 | 1.550 | 1.543 | **1.553** |
| `mirror_neurons (reg)`, shared grid | 1.571 | 1.599 | 1.583 | **1.584** |
| ~~old (reg) pair, tuned on mirror's screen~~ | ~~1.551 / 1.540~~ | | | ~~1.551 / 1.544~~ |

- **Why it works, measured:** the cosine between each layer's backward and forward weights climbs
  0.00 → 0.48 → 0.68 → 0.82 → 0.90 at steps 0 / 500 / 1,000 / 2,000 / 4,000. With the mirror off it
  reaches only 0.18.
- **Correction: `(reg)` is a loss for mirror, not a tie.** The old pair used weight decay 0.2,
  5% input noise and lr halving, picked on a 5-point screen of mirror, with each rule's old `lr`.
  Rerun on one shared grid (lr ⅔e-3 / 1e-3 / 2e-3, noise 5% / 10%, weight decay 0.01 / 0.2, lr
  halving every 3,000 steps, then lr 1.4e-3 for backprop and 3e-3 / 4e-3 for mirror to bracket
  them; val, seed 0, 3 min), each rule picks its own best point. Backprop picks lr 1e-3, 5% noise,
  weight decay 0.01 (val 1.420). Mirror picks lr 3e-3 (val 1.424). Over 3 seeds at 5 min:
  backprop **1.553**, mirror **1.584**. At lr 2e-3 and up backprop gets erratic and mirror does
  not, but that only buys mirror a higher learning rate, and at 3e-3 it is the one that wobbles
  across seeds.
- **What beat the table was the regularisation.** Without it neither rule beats the best table
  (1.614); with it both do (1.55 and 1.58).

**`mirror_window`**: the same trick with hand-wired copy heads showing 2,048 neurons the last 6
characters, output weights by the delta rule, input weights by mirrored feedback.

| `mirror_window` (seeds 0 / 1 / 2) | test |
|---|---:|
| **mirror feedback** | 1.774 / 1.784 / 1.780 (mean **1.779**) |
| fixed random feedback (seed 0) | 1.804 |
| exact feedback: plain backprop on this net (seed 0) | 1.773 |
| input weights frozen (seed 0) | 1.961 |

It beats the untuned 5-gram (1.806), not the tuned one (1.702). Mirrored feedback matches exact
backprop within noise, so the learning rule is not what holds this net back. What does (width,
overfitting, the hand-wired window) is untested: a tuned table with the same 6 characters gets 1.633.

*Did not work:* compiling decision trees into neurons (val 1.79 vs 1.81 for suffix cells, so
counting-only is still far from 1.70), starting the neurons from those trees (diverged or val 1.69),
hidden dropout (val 1.73). Each one seed, val only, in a standalone prototype.

### 📚 Taught mirror: a lookup table teaches a network that never backprops

**The idea.** Counting already beats backprop on test: a smoothed 9-gram table scores 1.611, and
plain backprop 1.635. But the table needs about 1.4M contexts × 65 next characters, which does not
fit in a 0.83M-parameter model. So let the table *teach* the network instead.

`taught_mirror` has two parts, and neither one is backprop:
1. **A counting teacher, no gradients.** For its first 1,500 steps the rule records which character
   follows every recent string in the batches it sees, once per spot in the text (deduplicated by
   a hash of the 24 characters before it, since windows overlap). Then it builds Witten-Bell
   backoff tables for contexts of 0 to 8 characters. All of this happens inside `step()`, so the
   arena's clock pays for it (about 2 s). The teacher is **leave-one-out**: for a training
   character it subtracts that character's own count first. Otherwise a 9-character context seen
   once would just give away the answer (at beta 1, train loss 1.70 leave-one-out vs 0.52 with the
   answer left in).
2. **A `mirror_neurons` student.** Its backward weights start random and get the same update as
   their forward twins, so no weight is ever copied. The loss is 70% cross-entropy on the real next
   character and 30% on the teacher's whole distribution.

| test loss, 5 min | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| `backprop`, rerun in the same session | 1.638 | 1.641 | 1.626 | **1.635** |
| the teacher alone: a table, not in the model | | | | 1.611 |
| **`taught_mirror`** (tournament, lr 1e-3) | 1.609 | 1.580 | 1.598 | **1.596** |
| same rule, a second set of runs | 1.617 | 1.599 | 1.599 | 1.605 |
| control: `backprop (taught)`, same teacher | 1.589 | 1.586 | 1.580 | **1.585** |
| (its leaderboard row is a fourth run: 1.578) | | | | |
| control: `mirror_neurons`, no teacher (old runs) | 1.638 | 1.645 | 1.639 | 1.641 |

- **It beats backprop on every seed pair** (by 0.039 on average), and it beats its own teacher
  (1.611): the student ends up better than the table it learned from.
- **The catch, read first:** backprop with the same teacher gets 1.585, so it is still 0.011
  ahead. Mirror credit is not better than the real gradient. It is close enough that a
  backprop-free credit rule *plus* a counting teacher beats the bare gradient. The teacher
  closes 0.045 of mirror's gap and 0.050 of backprop's.
- **Settings were picked on val, not test.** beta = 16 (the smoothing strength) on the teacher's
  own val loss (8 → 1.479, 16 → 1.476, 32 → 1.520, 10 val batches), and teach = 0.3 on a 3-min
  student screen (0.3 → val 1.449, 0.6 → 1.450, 0.9 → 1.482). The tournament then swept lr as for
  every rule, and picked 1e-3.
- **Does the teacher lift the weaker rules too?** Yes, but not over the line (one seed each,
  5 min): `sign_symmetry` 1.675 → **1.642** (backprop's seeds: 1.626 to 1.641), `feedback_alignment`
  1.911 → **1.874**. The teacher is worth 0.03 to 0.05 to every rule, so the ranking does not
  change. Mirror is the only backprop-free credit rule here close enough to backprop for that to
  tip it over.
- **Can a non-gradient credit rule beat backprop when *both* are taught?** Not yet: every
  variant ties, and with regularisation backprop pulls ahead.

  | all with the same teacher; test, 5 min | runs | mean |
  |---|---|---:|
  | `backprop`, lr ⅔e-3 | 1.589 1.586 1.580 1.583 1.597 1.565 | **1.583** |
  | `backprop`, lr 1e-3 | 1.584 1.583 1.587 | 1.585 |
  | `taught_mirror` + noisy feedback (0.5 × std, fresh each step; picked on val) | 1.580 1.575 1.577 1.584 1.612 1.589 | 1.586 |
  | noisy backprop: feedback born as an exact copy, plus the same noise | 1.567 1.575 1.595 | 1.579 |
  | `sam` (rho 0.15, lr 2e-3) | 1.569 1.581 1.608 | 1.586 |
  | `backprop` + lr halving + 5% input noise | 1.553 1.532 1.534 | **1.540** |
  | `taught_mirror` + noisy feedback + the same regulariser | 1.577 1.560 1.587 | 1.575 |

  After 3 seeds, noisy mirror looked 0.008 ahead. Three more seeds took that away, and noisy
  backprop's 0.006 edge over lr-matched backprop is inside the noise too. With the regulariser
  the runs are compute-bound (both peak near minute 5). Mirror's custom layer takes about 15%
  fewer steps there, and backprop wins by 0.035. **The best number in the arena is backprop with
  the counting teacher and the regulariser: 1.540.**
- *Did not help:* a Kneser-Ney teacher (val 1.477 vs 1.474 for Witten-Bell, 50 batches; left in as
  `smoothing="kn"`).
- **📏 Teaching each layer its own counting table helps local learning a lot.** In
  `taught_layers` (🧱 local), no error ever crosses a block boundary (`local_losses`). Each block's
  own readout also learns from the table cut to a memory that suits its depth: block 0 from 2
  characters of context, then 4, 6, and the top block 8. Test, 5 min: **1.640 / 1.642 / 1.670 /
  1.649** (mean 1.650) vs 1.743 for plain `local_losses`, about 0.09 better. Giving every block the
  8-character table does worse (1.666, one seed), so the depth matching matters. The teacher is
  worth roughly twice as much to this local rule as to backprop. That is the first extra here that
  favours an alternative, but not enough. Control: the same per-block tables trained end to end by
  backprop (`taught_layers_bp`) get **1.585 / 1.587 / 1.585** (1.586), 0.07 ahead.
  Per-layer tables do not help backprop (1.586 vs 1.583 with only the top teacher). So I tried them
  on mirror credit (`taught_mirror_layers`, 🪞), whose credit is slightly off: **1.623 / 1.619 /
  1.628** (1.623), worse than mirror with only the top teacher (1.596). They help only the rule
  with no cross-block credit at all.
- *Did not help:* also training on text the table **dreams up** (sampled from it, 1M chars, scored
  against the teacher), val 1.456 / 1.463 for half / one extra dreamed batch vs 1.449 without.
  The dreams cost steps and read like garbled Shakespeare.

### 🔬 Overnight re-checks: what did not beat backprop, and a split that moves

- **The val and test splits are different plays.** The data is cut 90/5/5 in order, so val is
  mostly *The Taming of the Shrew* (which the end of train is also in), and test runs into
  *The Tempest*. Only 9 of test's 21 speakers appear in train, against 11 of val's 17. Val
  rewards knowing the play you are in, test rewards carrying over to a new one. Every rule is
  still picked on val and scored on test. But **tricks that lower val do not reliably lower test**,
  and it shows below.
- **Sharpness-aware updates (`sam`, 🪞 modified) tie backprop on test and win on val.** Each step
  takes its gradient a small step uphill (rho 0.15, two backward passes). Tournament, 3 seeds: test
  **1.633** vs backprop 1.635, val **1.463** vs 1.494. The flatter minimum helps on the play it
  trained on and not on the new one. A lagged one-pass version was worse than backprop on val.
- **Backdrop (`backdrop`, 🪞 modified)**: drop the error signal for a random 10% of tokens at
  every block boundary on the way down. With the shared-grid regularisation, 3 seeds: 1.558 vs
  backprop 1.553. At 25% or 50% it gets much worse (val 1.43 and 1.63 vs 1.42).
- **Unreliable backward synapses in `mirror_neurons`**: fresh noise on the feedback weights at
  every step (0.5 × their std) lowers val from 1.493 to 1.476 (2 min, seed 0). Twice the noise
  gives 1.600; dropping 30% or 60% of feedback synapses gives 1.503 and 1.662. Not taken further.
- **A screening split that behaves like test** (`bbp/proxy.py`). Hold out *Richard II* from the
  middle of train (24 speakers, only 4 seen elsewhere in train), train on the rest and pick
  checkpoints on it. Real val and test stay untouched. Backprop + teacher's seed spread there is
  ±0.003 (1.455 / 1.460 / 1.454), tighter than test's ±0.01.
- **More credit rules, screened on that split** (all with the counting teacher, 4 min, seed 0;
  backprop 1.456):

  | rule | proxy loss |
  |---|---:|
  | `noisy_backprop`: error jittered at every block boundary, sigma 0.5 / 1.0 | 1.456 / 1.469 |
  | low-rank feedback: error sent back only through each W's top 16 / 48 directions | 1.594 / 1.472 |
  | `winner_credit`: only the k most driven MLP neurons per token get error, k = 64 / 192 | 1.788 / 1.505 |
  | `sam`, rho 0.15 / 0.3 / 0.6 | **1.449** / 1.456 / 1.581 |
  | mirror-credit SAM (sharpness-aware, no weight transport), lr 1e-3 / 2e-3 | 1.454 / 1.453 |
  | `noisy_backprop`, noise only on the error entering attention, sigma 1 / 3 | 1.452 / 1.568 |
  | `noisy_backprop`, noise only on the error entering the MLPs, sigma 1 | 1.453 |
  | `and_mask` (move a weight only where batch slices agree on its gradient's sign), 4 slices all agree / 3 of 4 / 6 of 8 / 2 of 2 | 1.465 / 1.454 / 1.477 / 1.457 |
  | `rwp`: each gradient taken at randomly jittered weights, sigma 0.01 / 0.03 / 0.1 / **0.2** / 0.3 / 0.5 | 1.455 / 1.460 / 1.435 / **1.414** / 1.448 / 1.572 |
  | `rwp` 0.2, seed 1 | 1.418 |
  | backprop + lr halving + 5% / 10% input noise (its best regulariser) | 1.437 / 1.449 |
  | `rwp` 0.2 + that regulariser on top | 1.476 |
  | `taught_mirror` + weight jitter 0.2 / 0.1 (no weight transport) | 1.447 / 1.444 |
  | `lazy_lower`: error reaches the bottom blocks only every 4th / 2nd step | 2.259 / 1.964 |

  Structure (low rank, winners only) costs a lot. Noise and sharpness are worth at most 0.007,
  wherever the noise goes: noise on the long-range (attention) credit is no better than on the MLPs.
  Masking out what the slices of a batch disagree on (meant to drop one play's names and plot) does not
  help either: random windows are not different enough "environments".
- **The two best ones on test, with the teacher:** `sam` over 6 seeds **1.578** vs backprop
  **1.583**. It wins 4 of 6 seed pairs, but paired t ≈ −0.7, so it is not a win. Mirror-credit SAM,
  3 seeds: 1.586 / 1.579 / 1.597 (**1.587**) vs backprop's 1.589 / 1.586 / 1.580 on the same seeds.
  Resolving a 0.006 gap at this seed spread would take about 40 seed pairs.
- **So I ran 26 seed pairs of `sam` vs backprop** (both with the teacher, each at its tournament
  lr, run count fixed in advance). SAM is ahead by **0.006** on average (95% CI −0.015 to +0.002),
  paired t = −1.59 (p ≈ 0.12), and wins 18 of 26 pairs (sign test p ≈ 0.08). Probably a small real
  edge, but not established at the usual 5% level. And SAM is backprop run twice per step.
- **A pre-registered confirmation failed.** Before running, I fixed 20 fresh seed pairs (26 to 45),
  a one-sided paired t-test, no early stopping and no extension. Result: SAM ahead by **0.002**,
  t = −0.43 (needed −1.73), 11 of 20 pairs. The 0.006 edge did not replicate. Over all 46 pairs
  the gap is about 0.005, inside the noise. SAM with the teacher ties backprop with the teacher.
- **Random weight perturbation (`rwp`) was the strongest candidate, and it still ties.** On the
  *Richard II* proxy, sigma 0.2 beat backprop's best regularised run by 0.023 (two seeds each
  for `rwp`). On the real arena, 6 seeds each, both with the teacher: `rwp` test **1.542**
  (1.533 1.537 1.560 1.530 1.555 1.539) vs backprop + lr halving + 5% input noise **1.538**
  (1.553 1.532 1.534 1.545 1.528 1.538). On val `rwp` is far ahead, 1.392 vs 1.405, and on test
  it is not. Like SAM, it finds flatter minima that help on the play it trained on and on
  *Richard II*, but not on *The Tempest*. What moved test was the counting teacher and input
  noise, not where the gradient is taken.
- **Adversarial embeddings (`adv_embed`, 🪞 modified)**: learn from the gradient at input
  embeddings pushed in their worst-case direction, a continuous cousin of the input noise that
  did move test. Val, with the teacher: eps 0.1 / 0.3 / 1.0 → 1.412 / 1.480 / 1.648. Test at
  eps 0.1, 3 seeds: 1.544 / 1.564 / 1.552 (**1.553**) vs regularised backprop's 1.553 / 1.532 / 1.534
  (1.540). Backprop wins.
- **A faster mirror.** The feedback weights now get their update in place instead of from a second
  AdamW. It is the same update (a test checks it matches to 1e-6), but mirror is still about 12%
  slower per step than backprop, because of its custom linear layer.

(The shared-grid `(reg)` pair is not on the leaderboard: its seed-0 runs happen to go mirror's
way, 1.560 vs 1.567, while the 3-seed means go backprop's way, 1.584 vs 1.553.)

**How to read it**
- **test loss**: cross-entropy on held-out Shakespeare, nats per character. Lower is better.
  4.17 = uniform guessing over 65 chars, ~1.5 = writes pseudo-Shakespeare.
- **unigram baseline**: character probabilities are fitted on the training split, then scored
  on held-out targets with the same evaluator and sampling seed (1234) as the models.
  `unigram_val_loss` uses the 10 validation batches shown in the learning curves;
  `unigram_test_loss` uses the 50 test batches reported in the table.
- **best ckpt at / early stopping**: TinyShakespeare is small. Backprop rips through ~90 epochs
  in 5 minutes and starts memorising the training text after about a minute. Scoring the
  *last* weights would reward rules for being bad at overfitting, so the data is split
  90/5/5 train/val/test: every rule keeps its best checkpoint by val loss, and that
  checkpoint is scored on a test split it has never seen. "best ckpt at" shows when each rule peaked.
- **autograd**: how much each rule leans on reverse-mode autodiff.
  🚫 `none` = no backward pass anywhere (**enforced**: the arena makes `.backward()` and `torch.autograd.grad` raise) ·
  🧱 `local` = backprop only *inside* a layer/block, never end to end ·
  🪞 `modified` = end-to-end backward pass, but not the true gradient ·
  ✅ `full` = backprop.
- **beats random reservoir?**: `reservoir` freezes the random network and trains only the
  final readout. If a rule can't beat that, its deep layers aren't really learning.
- **fwd passes**: total full-model forward passes spent. Wall-clock is the budget, but this
  shows how compute-hungry each idea is (ES burns 64 per step, backprop 1).
- **tuned**: each rule's one most important knob got the same quick sweep (×⅓, ×1, ×3,
  extended if the best was on the edge) before its final run. Sweep trials evaluate validation
  only; the selected configuration's final run is scored on test. This also applies to the
  unigram reference. Selection already used validation; this avoids unnecessary test scoring
  during tuning.

## All contenders, with sources

| rule | autograd | the idea | from |
|------|----------|----------|------|
| `backprop` | ✅ full | The baseline. AdamW. | Rumelhart, Hinton & Williams 1986 |
| `feedback_alignment` | 🪞 modified | The backward pass sends errors through **fixed random matrices** instead of Wᵀ. Somehow the forward weights learn to "align" with them. | Lillicrap et al. 2016 |
| `mirror_neurons` 🪞 | 🪞 modified | Backward weights are separate and start random, but get the same Hebbian update as their forward twins, so they drift into alignment without copying. | Kolen & Pollack 1994; Akrout et al. 2019 |
| `taught_mirror` 📚 | 🪞 modified | A smoothed 9-gram table, built by counting the batches it sees (leave-one-out, no gradients), teaches a `mirror_neurons` student: 30% of the loss is the table's distribution. | Kolen & Pollack 1994; Witten & Bell 1991; Hinton et al. 2015 |
| `sam` | 🪞 modified | Sharpness-aware: each update is the gradient taken a small step uphill, which steers toward wide minima. Two backward passes. | Foret et al. 2021 |
| `backdrop` | 🪞 modified | Backprop whose error is dropped for a random fraction of tokens at every block boundary on the way down; the forward pass is untouched. | Golkar & Cranmer 2018 |
| `noisy_backprop` | 🪞 modified | Backprop whose error gets fresh noise every time it crosses a block boundary on the way down. | gradient-noise injection (Neelakantan et al. 2015) |
| `winner_credit` | 🪞 modified | Backprop where only the k most driven hidden neurons in each MLP get any error. | k-winners-take-all |
| `and_mask` | 🪞 modified | A weight moves only where most slices of the batch agree on the sign of its gradient: learn what is hard to vary. | Parascandolo et al. 2020 |
| `rwp` | 🪞 modified | Each gradient is taken at a randomly jittered copy of the weights (noise scaled to each row), then the clean weights step. | Li et al. 2023 |
| `lazy_lower` | 🪞 modified | The error reaches the bottom blocks only every k-th step. | cf. FreezeOut (Brock et al. 2017) |
| `adv_embed` | 🪞 modified | Learns from the gradient at input embeddings pushed in their worst-case direction (one extra pass). | Miyato et al. 2017; Zhu et al. 2020 |
| `taught_layers` 📏 | 🧱 local | Local per-block learning where each block is also taught by a counting table cut to a memory suited to its depth (2/4/6/8 chars). | Belilovsky et al. 2019; Witten & Bell 1991 |
| `taught_layers_bp` | ✅ full | Its control: the same per-block tables, trained end to end by backprop. | |
| `taught_mirror_layers` | 🪞 modified | Mirror-neuron credit (no weight transport anywhere) plus the depth-suited per-block tables. | |
| `mirror_window` 🪞 | 🪞 modified | Mirrored feedback on one hidden layer of 2,048 neurons that see the last 6 chars through hand-wired copy heads. | + Bengio et al. 2003 |
| `sign_symmetry` | 🪞 modified | Backward pass through `sign(W)`: the feedback knows the signs of the weights but not their size. | Liao, Leibo & Poggio 2016 |
| `local_losses` | 🧱 local | Every block gets its own tiny next-char readout and learns greedily. Gradients never cross a block boundary. | Belilovsky et al. 2019; Nøkland & Eidnes 2019 |
| `forward_forward` | 🧱 local | Hinton's FF for a causal LM: each block learns high "goodness" on real text, low on text with swapped chars. The readout is a probe on top. | Hinton 2022 |
| `reservoir` | 🧱 local | Random frozen transformer, only the readout trains. Sanity floor. | Jaeger 2001 |
| `forward_grad` | 🚫 none | One forward-mode JVP along a random direction v gives (∇L·v); use (∇L·v)v as the gradient. Unbiased, hugely noisy. | Baydin et al. 2022 |
| `mezo` | 🚫 none | SPSA: nudge all weights by ±εz, compare two losses, step along z. No autograd, no activation memory. | Spall 1992; Malladi et al. 2023 |
| `es` | 🚫 none | Evolution strategies: 64 noisy clones evaluated in parallel with `vmap`, ranked, weights drift toward the winners. | Salimans et al. 2017 |
| `lab_rats` 🐀 | 🚫 none | **Every token is an experiment.** Jolt every neuron with fresh noise at every token, run ±noise forward passes, and credit each token's loss change to that token's jolt. Each layer then updates Hebbian-style (jolt-credit ⊗ its own input). No backward pass, no weight transport, only a per-token "dopamine" loss signal. | node perturbation (Williams 1992; Fiete & Seung 2006) + per-token credit in a causal LM (the twist tried here) |
| `lab_rats_maze` 🐀 | 🚫 none | lab_rats + block 0 gets its own next-char readout ("local maze") and is credited mostly by that nearby loss. Best lab_rats so far. | + local readouts à la Nøkland & Eidnes 2019 |
| `lab_rats_cycle` / `_solo` / `_future` 🐀 | 🚫 none | The other attempted fixes: one block per step, every block separately, cross-token credit for attention. See the lab notebook above. | |
| `wire_together` 📊 | 🚫 none | No optimisation: count which char follows which, then compile the counts into the weights in closed form (one-hot embeddings, silenced blocks, pseudo-inverse readout). An exact bigram in a transformer costume. | Hebb 1949; Shannon 1948 |
| `instinct_rats` | 🚫 none | Born wired as that bigram, then `lab_rats_maze` keeps learning on top. | this repo |
| `dreaming` 💤 | 🧱 local | Wake/sleep Hebbian: strengthen what fires on real text, weaken what fires on the model's own dreams; readout learns real-vs-dreamt next char (the delta rule). | Hinton & Sejnowski 1983; Crick & Mitchison 1983; Hinton 2022 |
| `census_heads` / `_4` 🔎 | 🧱 local | **Attention taught by a census.** Running letter-triple counts say which earlier char best explains the next one (E-step); each head's scores move toward that posterior (M-step). Wired stream, fixed random value codes, delta-rule readout. Hand-written, layer-local gradients. | Dempster et al. 1977 (EM); Liu et al. 2016 (supervised attention) |
| `census_mlp` 🧠 | 🧱 local | `census_heads` plus one MLP trained by direct feedback alignment, so it can learn character conjunctions. | Nøkland 2016 |
| `grandmother_cells` 👵 | 🚫 none | One neuron per memory: each MLP cell fires for one recent string, chosen and wired by counting. Copy heads are hand-wired. | Lettvin's grandmother cell; Cleary & Witten 1984 (PPM); Willems et al. 1995 |
| `natural_selection` 🧬 | 🧱 local | Cells that do the least die, mutants of the living are born, output weights follow the delta rule. Hand-wired copy heads. | Edelman 1987; Fahlman & Lebiere 1990; Widrow & Hoff 1960 |
| `hill_climb` | 🚫 none | Jiggle every weight at random; keep it if the loss dropped. The dumbest possible learner, as a floor. | Rechenberg 1973 |
