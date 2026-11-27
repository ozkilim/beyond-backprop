"""NOISY MIRROR: mirror neurons with temperature-scheduled noise injection.

Starts from mirror neurons (which are ≈ backprop), then adds Gaussian noise
to the backward weights whose magnitude DECAYS over training:

    B_t = W_t + σ(t) * ε,  ε ~ N(0, I), σ(t) = σ₀ * (1 - t/T)

This is the OPPOSITE of phase_transition:
  - Phase transition: starts BAD (random) and gets GOOD (tracking)
  - Noisy mirror: starts GOOD (tracking) and adds TEMPORARY noise

The noise provides exploration around the true gradient direction (not in a
completely random direction like feedback alignment). As training progresses,
the noise shrinks and the method converges to pure backprop.

The theory: early noise around the TRUE direction is more efficient than
early noise in a RANDOM direction (phase transition), because the random
phase wastes time learning to track the forward weights. Here the tracking
is built-in from step 1.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class NoisyMirror(Rule):
    name = "noisy_mirror"
    autograd = "modified"
    blurb = "Mirror neurons with decaying Gaussian noise: explores around the true gradient early, converges to exact credit assignment late."
    paper = "Novel: temperature-scheduled noise on mirror backward weights."
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
        # Immediately set B = W (start from mirror, not random)
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
        elif self.hp["noise_schedule"] == "cosine":
            import math
            return s0 * 0.5 * (1.0 + math.cos(math.pi * t))
        return s0 * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Set B = W + noise (the noise is temporary, just for this backward pass)
        sigma = self._sigma()
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                # B = W + scaled noise
                noise = torch.randn_like(b) * b.std() * sigma
                b.copy_(w + noise)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Restore B to its EMA-tracked version before optimizer step
        with torch.no_grad():
            for b, sb in zip(self.B, saved_B):
                b.copy_(sb)
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Mirror update: B tracks W
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
