"""WARM-RESTART MIRROR: noisy mirror with COSINE WARM RESTARTS on the noise.

Standard noisy_mirror decays noise monotonically: σ(t) = σ₀ * (1 - t/T).
Once the noise is gone, it's gone. But what if the model gets stuck in a
local minimum partway through? The noise is already too small to escape.

Warm-restart mirror uses a COSINE SCHEDULE WITH RESTARTS (SGDR-style):
  σ(t) = σ_min + (σ₀ - σ_min) * 0.5 * (1 + cos(π * t_i / T_i))

where t_i is the time within the current cycle. The cycle length doubles
each time (T_0, 2*T_0, 4*T_0...).

The effect: the noise SPIKES periodically, allowing the model to escape
local minima or sharp basins, then decays for convergence within each cycle.
Each restart is a "shake" that tests whether the current basin is deep enough.

This is SGDR (Loshchilov & Hutter 2017) applied to the BACKWARD NOISE
instead of the learning rate. It's essentially free — just changes the
sigma computation.

Combined with mirror neurons + teacher, this creates a training dynamic
where the model alternates between:
  - EXPLORATION phases (high noise after restart)
  - EXPLOITATION phases (low noise before next restart)

The doubling cycle length means early training explores more aggressively
while late training has longer exploitation windows.
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class WarmRestartMirror(Rule):
    name = "warm_restart_mirror"
    autograd = "modified"
    blurb = "Mirror neurons with cosine warm-restart noise: periodic exploration spikes with doubling cycle length."
    paper = "Novel: SGDR-style warm restarts on backward weight noise + mirror neurons."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, sigma_min=0.05,
                    t0_frac=0.15,  # first cycle = 15% of training
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
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                b.copy_(w)
        self.step_count = 0
        self.est_total_steps = 18000

    def _sigma(self):
        t = self.step_count / max(self.est_total_steps, 1)
        s0 = self.hp["sigma0"]
        s_min = self.hp["sigma_min"]
        t0 = self.hp["t0_frac"]

        # Find which cycle we're in and position within it
        # Cycle lengths: t0, 2*t0, 4*t0, ...
        cycle_start = 0.0
        cycle_len = t0
        while cycle_start + cycle_len < t and cycle_start + cycle_len < 1.0:
            cycle_start += cycle_len
            cycle_len *= 2.0
        # Position within current cycle [0, 1]
        t_in_cycle = min((t - cycle_start) / max(cycle_len, 1e-8), 1.0)

        # Also apply a global envelope decay so late restarts don't go as high
        global_decay = 1.0 - 0.5 * min(t, 1.0)  # halves sigma by end

        # Cosine annealing within cycle
        sigma = s_min + (s0 * global_decay - s_min) * 0.5 * (1.0 + math.cos(math.pi * t_in_cycle))
        return max(sigma, s_min)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # B = W + noise (fast_mirror style: no clone/restore needed)
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
