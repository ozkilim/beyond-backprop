# beyond-backprop 🧠⚡

**Can anything other than backprop train a language model as well? An arena to find out.**

Every learning rule trains the **same 0.83M-param character-level GPT** on **TinyShakespeare** for
**the same 5 minutes** of wall-clock. Each keeps its best checkpoint by val loss and is scored on a
held-out test split. It runs on a laptop (Apple-silicon GPU, CUDA, or CPU).

## The answer so far: no

About 30 rules later, nothing beats backprop on equal terms. The fair comparison gives backprop
every extra its rival gets (teacher, regulariser, optimizer), so only the credit rule differs.

| matched comparison (test loss, mean over seeds) | backprop | best rival |
|---|---:|---:|
| plain | **1.635** | 1.641 `mirror_neurons` (no weight transport) |
| both taught by a counting table | **1.583** | 1.596 `taught_mirror` · 1.578 `sam` (tie over 46 seed pairs) |
| teacher + best regulariser | **1.538** | 1.542 `rwp` · 1.553 `adv_embed` |
| local learning, same per-block teachers | **1.586** | 1.650 `taught_layers` |
| no backward pass at all | 1.635 | 1.900 `grandmother_cells` (n-gram counting compiled into weights) |

- **The near-misses are not cheaper either.** Mirror is ~14% slower per step, SAM does two backward
  passes, and both reach val 1.50 later than backprop does.
- **Only data-level changes move test.** A counting teacher and input noise help every rule.
  Changes to the credit rule (SAM, RWP, noisy or dropped error) lower val but not test.
- **A smoothed 9-gram table alone gets 1.611**, better than plain backprop, in ~2 s of counting.
  But it needs far more memory than the model, so it is a reference, not a contender.

**Why it's so hard here.** The data runs out before the clock does: every rule peaks after
1 to 4 minutes, then memorises. So the contest is "who generalises best from 1M characters", and
any regulariser can be handed to backprop too. The exact gradient is the best local direction, and
approximating it only costs. And this setup has none of backprop's weak spots: the model is tiny,
storing activations is free, and the GPU is built for exactly this. Val (*Taming of the Shrew*) and
test (*The Tempest*) are also different plays, so val-only gains don't transfer.

Where an alternative could win: a memory cap, one pass over the data, continual learning across
plays, or hardware without weight transport. Those tracks are open.

## Leaderboard

One seed-0 run per rule; the matched comparisons above average several seeds.

<!-- LEADERBOARD:START -->
| # | rule | autograd | test loss ↓ | vs backprop | beats random reservoir? | best ckpt at | steps | fwd passes | tuned |
|---|------|----------|-----------:|------------:|:--:|------:|------:|-----------:|-------|
| 1 | `backprop (taught)` | ✅ full | 1.578 | -0.059 | yes | 2.0 min | 20,483 | 20,483 |  |
| 2 | `taught_mirror` | 🪞 modified | 1.609 | -0.029 | yes | 2.0 min | 18,312 | 18,312 | lr=0.001 |
| 3 | `mirror_neurons` | 🪞 modified | 1.638 | -0.000 | yes | 1.7 min | 20,526 | 20,526 | lr=0.001 |
| 4 | `backprop` | ✅ full | 1.638 | +0.000 | yes | 1.3 min | 23,574 | 23,574 | lr=0.00067 |
| 5 | `sam` | 🪞 modified | 1.652 | +0.014 | yes | 3.7 min | 11,718 | 23,436 | lr=0.002 |
| 6 | `sign_symmetry` | 🪞 modified | 1.675 | +0.037 | yes | 2.0 min | 20,864 | 20,864 | lr=0.001 |
| 7 | `local_losses` | 🧱 local | 1.743 | +0.105 | yes | 2.7 min | 18,457 | 18,457 | lr=0.002 |
| 8 | `mirror_window` | 🪞 modified | 1.773 | +0.136 | yes | 5.0 min | 32,860 | 32,860 | lr=0.001 |
| 9 | `grandmother_cells` | 🚫 none | 1.899 | +0.261 | yes | 0.3 min | 68,804 | 68,804 | kappa=12 |
| 10 | `natural_selection` | 🧱 local | 1.905 | +0.268 | yes | 1.2 min | 53,307 | 53,307 | lr=0.001 |
| 11 | `feedback_alignment` | 🪞 modified | 1.911 | +0.273 | yes | 4.5 min | 21,233 | 21,233 | lr=0.001 |
| 12 | `census_mlp` | 🧱 local | 1.954 | +0.317 | yes | 4.7 min | 31,602 | 31,602 | lr=0.001 |
| 13 | `census_heads` | 🧱 local | 2.369 | +0.731 | yes | 0.7 min | 31,549 | 31,549 | lr=0.001 |
| 14 | `census_heads_4` | 🧱 local | 2.408 | +0.771 | yes | 1.5 min | 34,110 | 34,110 | lr=0.001 |
| 15 | `instinct_rats` | 🚫 none | 2.498 | +0.860 | yes | 0.8 min | 16,849 | 33,598 | lr=3.7e-06 |
| 16 | `wire_together` | 🚫 none | 2.500 | +0.862 | yes | 0.8 min | 65,165 | 65,165 | smoothing=2.7 |
| 17 | `lab_rats_maze` | 🚫 none | 2.610 | +0.972 | yes | 5.0 min | 16,662 | 33,324 | lr=0.00033 |
| 18 | `lab_rats_cycle` | 🚫 none | 2.611 | +0.973 | yes | 4.3 min | 28,804 | 57,608 | lr=0.001 |
| 19 | `lab_rats` | 🚫 none | 2.650 | +1.012 | yes | 4.7 min | 16,558 | 33,116 | lr=0.00033 |
| 20 | `lab_rats_future` | 🚫 none | 2.660 | +1.023 | yes | 4.3 min | 16,840 | 33,680 | lr=0.00033 |
| 21 | `lab_rats_solo` | 🚫 none | 2.692 | +1.055 | yes | 5.0 min | 5,737 | 57,370 | lr=0.003 |
| 22 | `forward_forward` | 🧱 local | 2.698 | +1.060 | yes | 4.3 min | 10,107 | 20,214 | lr=0.009 |
| 23 | `reservoir` | 🧱 local | 2.716 | +1.079 | — | 3.8 min | 52,867 | 52,867 | lr=0.003 |
| 24 | `es` | 🚫 none | 2.788 | +1.151 | no | 5.0 min | 895 | 57,280 | lr=0.003 |
| 25 | `mezo` | 🚫 none | 2.818 | +1.181 | no | 5.0 min | 25,747 | 51,494 | lr=0.0009 |
| 26 | `hill_climb` | 🚫 none | 3.153 | +1.515 | no | 5.0 min | 24,757 | 49,514 | sigma=0.01 |
| 27 | `forward_grad` | 🚫 none | 3.167 | +1.529 | no | 5.0 min | 9,280 | 9,280 | lr=0.0018 |
| 28 | `dreaming` | 🧱 local | 3.331 | +1.693 | no | 3.0 min | 21,607 | 43,214 | lr=0.0003 |
| | *unigram (letter frequencies only)* | | 3.360 | | | | | | |
| | *untrained / uniform guess* | | 4.216 | | | | | | |

<sub>0.83M-param char GPT (4 layers, d=128, ctx 128), batch 32, TinyShakespeare, 5 min of training time per rule on `mps`, loss in nats/char, best checkpoint by val scored on test. Tagged rows are controls: (taught) is backprop with the counting teacher `taught_mirror` uses.</sub>

![loss curves](results/curves.png)

What each model writes: [results/SAMPLES.md](results/SAMPLES.md).
<!-- LEADERBOARD:END -->

**autograd:** 🚫 `none` = no backward pass (enforced: `.backward()` raises) · 🧱 `local` = backprop
only inside a layer or block · 🪞 `modified` = end-to-end backward pass, but not the true gradient ·
✅ `full` = backprop. **beats random reservoir?** = does it beat a frozen random net with a trained
readout. **tuned** = the one knob each rule got a 3-point sweep on (val only).

Every experiment, ablation and correction, plus the full rule list with sources, is in the
[lab notebook](docs/lab-notebook.md).

## Quickstart

```bash
git clone https://github.com/ozkilim/beyond-backprop && cd beyond-backprop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m bbp.arena --list                        # what's in the arena
python -m bbp.arena --rule backprop --minutes 1   # train one rule, print a sample
python -m bbp.arena --rule mezo --set lr=3e-4     # override hyperparameters
python -m bbp.tournament                          # sweeps + 5-min finals for every rule (~2 h)
python -m bbp.leaderboard                         # rebuild the table, plot and samples
pytest                                            # every rule takes a few steps on a tiny CPU model
```

TinyShakespeare downloads itself on first run. `--no-test` runs score val only and stay off the
leaderboard.

## Add your own idea

Copy [`bbp/rules/_template.py`](bbp/rules/_template.py) to `bbp/rules/your_idea.py`:

```python
from . import Rule, register

@register
class MyIdea(Rule):
    name = "my_idea"
    autograd = "none"      # be honest; "none" is enforced
    blurb = "Weights learn by vibes."
    defaults = dict(lr=1e-3)

    @torch.no_grad()
    def step(self, x, y):  # x, y: (B, T) char ids and next-char targets
        _, loss = self.model(x, y)
        self.forwards += 1
        ...                # change self.model's weights however you like
        return loss.item()
```

It's picked up automatically. Run `python -m bbp.tournament --rules my_idea` and open a PR with the
result JSON. Not tried yet: predictive coding, target propagation, equilibrium propagation, PEPITA,
synthetic gradients, NoProp, HSIC bottleneck.

## Fine print

- Seed noise is about ±0.01 on test. Claims above use 3 or more seeds, paired, and one was
  pre-registered.
- Wall-clock favours rules that suit the GPU; `fwd passes` in the table lets you argue about compute.
- Short budgets favour fast starters. If your rule suddenly beats backprop, check the curve first:
  an early version scored final weights, and local losses "won" by pure overfitting.

MIT licensed.
