"""MOMENTUM NOISE: temporally correlated noise on backward weights.

Standard noisy_mirror draws FRESH iid noise every step. This means the
exploration direction changes randomly each step — a random walk in
gradient space.

Momentum noise instead maintains a RUNNING noise vector that's partially
updated each step (like SGD momentum but on the noise):

  ε_t = β * ε_{t-1} + (1-β) * z_t,  z_t ~ N(0, I)
  B_t = W_t + σ(t) * ε_t

With β=0.9, the noise direction is 90% the same as last step. This means
the model explores in CONSISTENT directions for ~10 steps before shifting.

Why this could help:
  - iid noise causes the gradient direction to jitter wildly each step,
    partially canceling itself out over consecutive steps
  - Correlated noise means the exploration signal ACCUMULATES over several
    steps, potentially finding better directions
  - It's like momentum for exploration instead of for optimization

The analogy: SGD momentum helps because it smooths the optimization path.
Momentum noise smooths the EXPLORATION path.

Uses fast_mirror approach (no clone/restore). The EMA update on the noise
buffer is essentially free — one multiply-add per parameter.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class MomentumNoise(Rule):
    name = "momentum_noise"
    autograd = "modified"
    blurb = "Temporally correlated backward noise: explores in consistent directions, like momentum for exploration."
    paper = "Novel: EMA-smoothed noise on mirror backward weights for temporally coherent exploration."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, noise_momentum=0.9,
                    noise_schedule="linear",
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
        # Initialize noise buffers to zero
        self.noise_buffers = [torch.zeros_like(w) for w in self.W]
        self.step_count = 0
        self.est_total_steps = 15000

    def _sigma(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["sigma0"] * (1.0 - t)

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        sigma = self._sigma()
        mom = self.hp["noise_momentum"]

        # Update noise buffers with momentum and set B
        with torch.no_grad():
            for i, (w, b) in enumerate(zip(self.W, self.B)):
                # EMA update of noise direction
                z = torch.randn_like(w)
                self.noise_buffers[i].mul_(mom).add_(z, alpha=1.0 - mom)
                nb = self.noise_buffers[i]
                # B = W + nb * w.std() * sigma (all on GPU, no .item())
                torch.add(w, nb * (w.std() * sigma), out=b)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        self.forwards += 1
        return loss.item()
