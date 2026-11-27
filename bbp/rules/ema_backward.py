"""EMA BACKWARD: the backward weights are a temporal blur of the forward ones.

Mirror neurons give each backward weight the same step its forward twin gets,
so after many steps B ≈ W + (B₀ − W₀): the initial random offset persists.
Here the backward weights are an exponential moving average of the forward ones:

    B_t = β * B_{t-1} + (1−β) * W_t

For β=0, B=W (exact backprop). For β→1, B is frozen (feedback alignment).
In between, B is a smoothed version of W: it averages over the recent weight
trajectory, like Stochastic Weight Averaging applied to the backward pass.

The smoothing has two effects:
  1. The initial random B₀ decays exponentially (unlike mirror neurons).
  2. The backward weights lag behind and smooth out transient weight fluctuations,
     so the credit signal acts like an implicit temporal ensemble.

Combined with the counting teacher (optional, via teach/order/beta/count_steps hp).
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class EMABackward(Rule):
    name = "ema_backward"
    autograd = "modified"
    blurb = "Backward weights are an exponential moving average of the forward weights, decaying the random initial offset and smoothing the credit signal through time."
    paper = "Novel combination: EMA (Polyak & Juditsky 1992) applied to the backward pathway."
    defaults = dict(lr=1e-3, weight_decay=0.01, ema_beta=0.99, scale=1.0,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "ema_beta"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]

    def step(self, x, y):
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        # EMA update: B ← β·B + (1−β)·W
        with torch.no_grad():
            beta = self.hp["ema_beta"]
            for w, b in zip(self.W, self.B):
                b.mul_(beta).add_(w, alpha=1.0 - beta)
        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
