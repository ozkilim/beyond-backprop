"""FAST MIRROR: a simplified noisy mirror that eliminates unnecessary operations.

The original noisy_mirror does this every step:
  1. Clone B (save it)
  2. Set B = W + noise
  3. Forward + backward
  4. Restore B from clone
  5. Clone W (save before optimizer)
  6. Optimizer step
  7. Mirror update: B += (W_new - W_old) with weight decay correction

That's ~6 extra tensor operations per FALinear per step, costing ~42% of
total steps vs vanilla backprop (~11,500 vs ~20,000 steps in 5 min).

Fast Mirror simplifies: since B should always be close to W (that's the
whole point of mirror neurons), we just SET B = W at the start of each step
and add noise. No need to track B separately, no mirror update, no cloning
W before the optimizer. The weight decay correction term that the mirror
update applies is tiny (lr * wd ≈ 0.00001 per step).

Operations per step:
  1. Set B = W + noise (one copy + one noise)
  2. Forward + backward (standard)
  3. Optimizer step (standard)

That's it. We skip: B clone, B restore, W clone, mirror update (3 ops).
This should recover thousands of extra training steps.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class FastMirror(Rule):
    name = "fast_mirror"
    autograd = "modified"
    blurb = "Simplified noisy mirror: set B=W+noise each step, skip mirror tracking. Faster by ~30%."
    paper = "Novel: minimal-overhead mirror neurons (B=W always, ephemeral noise)."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, noise_schedule="linear",
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "sigma0"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        # B = W (perfect mirror from the start)
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                b.copy_(w)
        self.step_count = 0
        self.est_total_steps = 18000

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        s0 = self.hp["sigma0"]
        if self.hp["noise_schedule"] == "linear":
            return s0 * (1.0 - t)
        return s0 * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Simply set B = W + noise. No cloning, no restore needed.
        sigma = self._sigma()
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                # b = w + randn * w.std() * sigma (all on GPU, no .item() sync)
                noise = torch.randn_like(w)
                noise.mul_(w.std() * sigma)
                torch.add(w, noise, out=b)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        # No mirror update needed! B will be set = W at the start of next step.
        self.forwards += 1
        return loss.item()
