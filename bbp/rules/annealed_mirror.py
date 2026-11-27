"""ANNEALED MIRROR: noisy mirror with SCHEDULED teacher influence.

The counting teacher provides smoothed n-gram targets. Early in training,
the model benefits from heavy teacher guidance (smooth targets prevent
overfitting to noise). Late in training, the model should focus more on
the actual data to capture patterns the finite n-gram table can't represent.

Annealed Mirror schedules BOTH:
  - Backward noise: σ decays from σ₀ to 0 (standard noisy_mirror)
  - Teacher influence: α decays from teach_max to teach_min

The two schedules INTERACT:
  - Early: heavy noise + heavy teacher = smooth and explorative
  - Late: low noise + low teacher = precise and data-driven

The key insight: the teacher's n-gram table captures LOCAL patterns perfectly
but can't capture long-range dependencies. Early training should learn these
local patterns from the teacher. Late training should focus on the RESIDUAL
(what the teacher can't explain) which is where the neural network adds value.

This is essentially zero overhead: just a float multiplication per step to
update the teach parameter.
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class AnnealedMirror(Rule):
    name = "annealed_mirror"
    autograd = "modified"
    blurb = "Noisy mirror with scheduled teacher decay: heavy guidance early, data-focused late."
    paper = "Novel: dual-scheduled noise + teacher influence decay on mirror neurons."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,
                    teach_max=0.5, teach_min=0.15,
                    teach=0.5,  # initial value, overridden by schedule
                    beta=16.0, order=9, count_steps=1500)
    tune = "sigma0"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        # Set initial teach to teach_max
        self.hp["teach"] = self.hp["teach_max"]
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

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def _update_teach(self):
        """Cosine schedule from teach_max to teach_min."""
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        tmax, tmin = self.hp["teach_max"], self.hp["teach_min"]
        self.hp["teach"] = tmin + (tmax - tmin) * 0.5 * (1.0 + math.cos(math.pi * t))

    def step(self, x, y):
        self.step_count += 1
        self._update_teach()  # essentially free
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        sigma = self._sigma()
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                noise = torch.randn_like(b) * b.std() * sigma
                b.copy_(w + noise)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        with torch.no_grad():
            for b, sb in zip(self.B, saved_B):
                b.copy_(sb)
            before = [w.clone() for w in self.W]

        self.opt.step()

        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        self.forwards += 1
        return loss.item()
