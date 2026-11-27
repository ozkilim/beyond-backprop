"""STRATIFIED MIRROR: layer-wise phased noise on backward weights.

The key biological insight: different layers in a neural network learn different
things at different times. Early layers learn low-level features (character
bigrams), late layers learn task-specific mappings (next-character prediction).

Instead of applying the SAME noise schedule to all layers, we give each layer
its OWN phase-shifted noise schedule:

  - Last layers (close to output): start with LOW noise, end with MORE noise.
    They should learn the output mapping accurately first, then explore.
  - First layers (close to input): start with HIGH noise, end with LOW noise.
    They should explore representational space first, then lock in features.

This creates a WAVE of learning that sweeps from output to input: last layers
converge first (they see clean error signals), giving the first layers stable
targets to learn against. Then the noise reversal helps the last layers escape
overfitting while the first layers have already found good features.

This is the opposite of the standard "features first, mapping second" assumption.
It's motivated by the observation that in backprop, error gradients flow
OUTPUT → INPUT, so the last layers always get the cleanest signal anyway.
We lean into this natural asymmetry instead of fighting it.
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class StratifiedMirror(Rule):
    name = "stratified_mirror"
    autograd = "modified"
    blurb = "Layer-wise phased backward noise: output layers get clean credit early, input layers explore first, then they swap."
    paper = "Novel: spatiotemporal noise scheduling across network depth."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, phase_spread=1.0,
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
        # B = W (start from mirror)
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                b.copy_(w)
        self.step_count = 0
        self.est_total_steps = 18000
        self.n_layers = len(self.twins)

    def _sigma_for_layer(self, layer_idx):
        """Each layer gets its own noise schedule, phase-shifted by depth.

        Layer 0 (closest to input): high noise early, low noise late
        Layer N-1 (closest to output): low noise early, high noise late
        The phase_spread controls how much the schedules diverge.
        """
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        s0 = self.hp["sigma0"]
        spread = self.hp["phase_spread"]

        # Depth ratio: 0 = input, 1 = output
        depth = layer_idx / max(self.n_layers - 1, 1)

        # Input layers: standard decay (high→low)
        # Output layers: inverted (low→high)
        # Middle layers: somewhere in between
        # Phase interpolation between decay and growth
        decay = 1.0 - t  # standard: high→low
        growth = t        # inverted: low→high

        # Mix based on depth
        mix = depth * spread  # 0=pure decay, spread=pure growth
        mix = min(mix, 1.0)
        sigma = s0 * ((1.0 - mix) * decay + mix * growth)

        # Apply a smooth envelope so noise doesn't go wild at the end
        envelope = 0.5 * (1.0 + math.cos(math.pi * t))  # cosine taper
        sigma = sigma * (0.3 + 0.7 * envelope)  # keep at least 30% of sigma

        return sigma

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Per-layer noise injection
        saved_B = []
        with torch.no_grad():
            for i, (w, b) in enumerate(zip(self.W, self.B)):
                saved_B.append(b.clone())
                sigma = self._sigma_for_layer(i)
                noise = torch.randn_like(b) * b.std() * sigma
                b.copy_(w + noise)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

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
