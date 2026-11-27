"""PHASE TRANSITION: scheduled annealing from exploration to exploitation in the backward pass.

The backward weights start fully random (feedback alignment — maximum exploration)
and are gradually annealed toward the forward weights over training:

    B_t = (1 − α_t) * B_random  +  α_t * W_t

where α_t ramps from 0 to 1 on a cosine schedule over the training budget.

Early training: noisy backward signals explore broadly, finding the right basin.
Late training: accurate credit assignment refines the solution precisely.

Backprop is stuck with accurate gradients from step 1. If the loss landscape
has basins that are only reachable via broad, noisy search, this method could
find wider minima.

The frozen random component B_random never changes (it's the exploration prior).
The forward component W_t changes with training (it's the exploitation signal).
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class PhaseTransition(Rule):
    name = "phase_transition"
    autograd = "modified"
    blurb = "Backward weights anneal from random (exploration) to forward-tracking (exploitation) on a cosine schedule."
    paper = "Novel: scheduled interpolation of credit-assignment accuracy."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0, warmup_frac=0.0,
                    schedule="cosine", mirror=True,
                    teach=0.0, beta=16.0, order=9, count_steps=1500)
    tune = "lr"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        # Save the initial random backward weights as the exploration prior
        self.B_random = [b.detach().clone() for b in self.B]
        self.step_count = 0
        # Estimate total steps from budget (5 min * ~20k steps/5min)
        self.est_total_steps = 20000

    def _alpha(self):
        """Current mixing coefficient: 0 = pure random, 1 = pure forward."""
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        warmup = self.hp["warmup_frac"]
        if t < warmup:
            return 0.0  # pure exploration during warmup
        t_eff = (t - warmup) / (1.0 - warmup) if warmup < 1.0 else 1.0
        if self.hp["schedule"] == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t_eff))
        elif self.hp["schedule"] == "linear":
            return t_eff
        elif self.hp["schedule"] == "sqrt":
            return math.sqrt(t_eff)
        return t_eff  # default: linear

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Set backward weights to interpolation before the backward pass
        alpha = self._alpha()
        with torch.no_grad():
            for w, b, br in zip(self.W, self.B, self.B_random):
                b.copy_((1.0 - alpha) * br + alpha * w)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        # Save forward weights before step (for mirror update on the random component)
        with torch.no_grad():
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Optional: also mirror-update the random component (keep it alive)
        if self.hp["mirror"]:
            with torch.no_grad():
                for w, br, bef in zip(self.W, self.B_random, before):
                    br.add_(w - bef)  # random component drifts with the forward weights

        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
