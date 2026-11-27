"""SYNTHETIC GRADIENT MIX: blend real gradient with a synthetic one.

The counting teacher provides soft targets. When we compute loss against
the teacher targets, we get a DIFFERENT gradient than from the real data.
The teacher gradient points toward the n-gram distribution, while the
data gradient points toward the specific sample's distribution.

Standard teach() mixes the LOSSES: L = (1-α)*L_data + α*L_teacher
Then one backward pass computes the gradient of this mixed loss.

What if instead we compute TWO separate gradients and mix them differently?

  g_data = ∇L_data(W)      (gradient toward the data)
  g_teacher = ∇L_teacher(W)  (gradient toward the n-gram table)

Then we mix the gradients with a DIFFERENT schedule than the losses:

  g = (1 - β(t)) * g_data + β(t) * g_teacher

Where β(t) starts high (follow the teacher) and decays (follow the data).

The key difference from standard teach():
  - Standard: one backward through the mixed loss. The gradient is a
    linear combination weighted by the loss magnitudes.
  - This: separate gradients mixed explicitly. We control the GRADIENT
    blend independently from the LOSS blend.

HOWEVER: two backward passes would halve our step count. Instead, we
approximate g_teacher using the BACKWARD WEIGHT NOISE trick:

  B_teacher = direction toward teacher targets (previous step's teacher gradient)
  B = (1-β) * W + β * normalized(g_teacher_prev)

This means the backward weights are biased toward the teacher's preferred
direction. Early training gets more teacher-biased credit assignment,
late training gets pure backprop.

But wait — this is basically echo_gradient with the echo being the
teacher loss gradient. Let me try something different.

ACTUALLY: the simplest version that works with zero overhead is to
schedule the teach parameter more aggressively than annealed_mirror.
We want teach=0.6+ early (heavy teacher) with a sharp decay to teach=0.1
(mostly data) in the second half. This is the "front-loaded teacher" idea.

Combined with noisy mirror, this means:
  - First third: heavy teacher + heavy noise = smooth and explorative
  - Second third: moderate teacher + moderate noise
  - Final third: minimal teacher + minimal noise = data-driven convergence
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class SyntheticGradientMix(Rule):
    name = "synthetic_gradient_mix"
    autograd = "modified"
    blurb = "Front-loaded teacher with aggressive decay: heavy n-gram guidance early, pure data late."
    paper = "Novel: aggressive teacher-annealing + noisy mirror (front-loaded knowledge distillation)."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,
                    teach_max=0.6, teach_min=0.08,
                    teach_decay="steep",  # 'steep' = power decay, 'cosine' = cosine
                    teach=0.6,  # initial, overridden by schedule
                    beta=16.0, order=9, count_steps=1500)
    tune = "teach_max"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
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
        self.est_total_steps = 15000

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def _update_teach(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        tmax, tmin = self.hp["teach_max"], self.hp["teach_min"]
        if self.hp["teach_decay"] == "steep":
            # Power decay: (1-t)^3 — drops quickly then levels off
            self.hp["teach"] = tmin + (tmax - tmin) * (1.0 - t) ** 3
        else:
            # Cosine
            self.hp["teach"] = tmin + (tmax - tmin) * 0.5 * (1.0 + math.cos(math.pi * t))

    def step(self, x, y):
        self.step_count += 1
        self._update_teach()
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        sigma = self._sigma()
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                noise = torch.randn_like(w)
                noise.mul_(w.std() * sigma)
                torch.add(w, noise, out=b)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.forwards += 1
        return loss.item()
