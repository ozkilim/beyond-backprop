"""BACKWARD DROPOUT: dropout applied ONLY to the backward weights.

Standard dropout zeros random activations during the FORWARD pass.
Backward dropout zeros random elements of the BACKWARD WEIGHT MATRIX.
This means:

  Forward pass: sees the full network (all weights participate)
  Backward pass: sees a random SPARSE version of the credit assignment

The effect: the network must learn representations where the credit signal
is REDUNDANT — no single pathway can be essential for gradient flow.
This is like ensemble training for the CREDIT ASSIGNMENT structure.

Key differences from noisy_mirror:
  - Noisy mirror: B = W + noise (additive perturbation around W)
  - Backward dropout: B = W * mask (multiplicative sparsification of W)

  Noisy mirror explores NEAR the true gradient (W + ε).
  Backward dropout explores SUBSETS of the true gradient (W * mask).

The dropout probability decays over training (like the noise in noisy_mirror):
  - Early: aggressive dropout (heavy exploration of credit subsets)
  - Late: no dropout (converge to full credit assignment)

This is essentially the same cost as noise injection — just masking instead
of noise addition. The mask is a Bernoulli sample, same as standard dropout.
"""
import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class BackwardDropout(Rule):
    name = "backward_dropout"
    autograd = "modified"
    blurb = "Dropout on backward weights only: each backward pass sees a random sparse credit assignment."
    paper = "Novel: backward-weight-only dropout with decaying probability + mirror neurons."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    drop_max=0.3,     # initial dropout probability
                    drop_min=0.0,     # final dropout probability
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "drop_max"

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

    def _drop_prob(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        return self.hp["drop_max"] + (self.hp["drop_min"] - self.hp["drop_max"]) * t

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Set B = W * dropout_mask (backward dropout)
        p = self._drop_prob()
        saved_B = []
        with torch.no_grad():
            for w, b in zip(self.W, self.B):
                saved_B.append(b.clone())
                if p > 0:
                    mask = torch.bernoulli(torch.full_like(b, 1.0 - p))
                    b.copy_(w * mask / (1.0 - p + 1e-8))  # scale to compensate
                else:
                    b.copy_(w)

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
