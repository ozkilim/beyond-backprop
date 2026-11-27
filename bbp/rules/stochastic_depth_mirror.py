"""STOCHASTIC DEPTH MIRROR: randomly skip transformer blocks during training.

Stochastic depth (Huang et al. 2016) randomly drops entire residual blocks
during training. Each block survives with probability p, and the output is
scaled by 1/p. This provides:
  1. Implicit ensemble of exponentially many sub-networks
  2. Shorter effective depth → faster training per step
  3. Regularization from the sub-network averaging
  4. Each block must learn to be useful ON ITS OWN (can't rely on others)

The survival probability follows a linear schedule: early layers (more
fundamental) survive with higher probability, late layers can be dropped
more aggressively. During evaluation, all layers are used (like dropout).

Combined with mirror neurons + teacher + noise, this attacks overfitting
from yet another angle: the network architecture itself becomes stochastic.

This is fundamentally different from dropout (which drops individual neurons).
Stochastic depth drops ENTIRE COMPUTATIONAL STAGES, so the credit assignment
structure changes at every step.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


class StochasticBlock(torch.nn.Module):
    """Wrapper that randomly skips a residual block during training."""
    def __init__(self, block, survival_prob):
        super().__init__()
        self.block = block
        self.p = survival_prob

    def forward(self, x):
        if not self.training:
            return self.block(x)  # always use at eval
        if torch.rand(1).item() > self.p:
            return x  # skip this block entirely
        # Scale output to compensate for skipping
        return x + (self.block(x) - x) / self.p


@register
class StochasticDepthMirror(Rule):
    name = "stochastic_depth_mirror"
    autograd = "modified"
    blurb = "Randomly skip transformer blocks + mirror neurons: implicit ensemble of sub-networks with noisy credit."
    paper = "Novel combination: Huang 2016 stochastic depth + Kolen-Pollack mirror + counting teacher."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5,
                    p_min=0.5,   # survival prob for deepest layer
                    p_max=1.0,   # survival prob for shallowest layer
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "sigma0"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)

        # Wrap blocks with stochastic depth BEFORE swapping linears
        n = len(model.blocks)
        p_min, p_max = self.hp["p_min"], self.hp["p_max"]
        for i in range(n):
            # Linear schedule: first block p=p_max, last block p=p_min
            p = p_max - (p_max - p_min) * i / max(n - 1, 1)
            model.blocks[i] = StochasticBlock(model.blocks[i], p)

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

    def step(self, x, y):
        self.step_count += 1
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
