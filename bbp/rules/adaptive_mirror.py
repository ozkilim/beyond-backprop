"""ADAPTIVE MIRROR: noise magnitude adapts to loss trajectory.

Instead of a fixed noise schedule (linear decay), the noise magnitude
ADAPTS to how the training is going:

  - When loss is DECREASING FAST → the current gradient direction is good
    → REDUCE noise (exploit the good direction)
  - When loss is PLATEAUING → we might be stuck
    → INCREASE noise (explore to escape)
  - When loss is INCREASING → we might be overfitting or diverging
    → MODERATE noise (some exploration but not too much)

This creates a feedback loop between training progress and exploration.
The noise isn't scheduled — it RESPONDS to the training dynamics.

Implementation: maintain an EMA of the loss. Compare current loss to EMA.
If current < EMA (improving), reduce noise. If current > EMA (stagnating),
increase noise. The noise is bounded between sigma_min and sigma_max.

This is inspired by adaptive step-size methods (Polyak step-size, line
search) but applied to the BACKWARD WEIGHTS rather than the learning rate.

Combined with mirror neurons + teacher, this means the exploration intensity
is data-driven rather than time-driven.
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class AdaptiveMirror(Rule):
    name = "adaptive_mirror"
    autograd = "modified"
    blurb = "Noise magnitude adapts to loss trajectory: more exploration when stuck, less when improving."
    paper = "Novel: loss-trajectory-adaptive backward noise + mirror neurons."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma_max=0.8, sigma_min=0.05,
                    adapt_rate=0.01,   # how fast sigma responds
                    loss_ema=0.99,     # EMA decay for loss tracking
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "sigma_max"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                b.copy_(w)
        self.step_count = 0
        self.est_total_steps = 18000

        # Adaptive state
        self.sigma = self.hp["sigma_max"]
        self.loss_ema_val = None

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Inject noise with ADAPTIVE sigma
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                noise = torch.randn_like(b) * b.std() * self.sigma
                b.copy_(w + noise)

        logits, loss = self.model(xa, ya)
        loss_val = loss.item()
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # === ADAPT SIGMA ===
        if self.loss_ema_val is None:
            self.loss_ema_val = loss_val
        else:
            alpha = self.hp["loss_ema"]
            self.loss_ema_val = alpha * self.loss_ema_val + (1 - alpha) * loss_val

            # If current loss > EMA → things are getting worse → explore more
            # If current loss < EMA → things are improving → exploit
            ratio = loss_val / (self.loss_ema_val + 1e-8)
            adapt = self.hp["adapt_rate"]

            if ratio > 1.01:  # loss increasing (stuck or overfitting)
                self.sigma = min(self.sigma * (1 + adapt), self.hp["sigma_max"])
            elif ratio < 0.99:  # loss decreasing (good direction)
                self.sigma = max(self.sigma * (1 - adapt), self.hp["sigma_min"])
            # else: neutral, keep sigma unchanged

            # Also apply a slow base decay (we still want less noise overall)
            t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
            base_decay = 1.0 - 0.5 * t  # halves by end
            effective_sigma = self.sigma * base_decay

        # Restore B
        with torch.no_grad():
            for b, sb in zip(self.B, saved_B):
                b.copy_(sb)
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Mirror update
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        self.forwards += 1
        return loss.item()
