"""PHASE CONTRASTIVE: phase-transition annealing + contrastive gradient filtering.

Combines:
  1. Phase transition: backward weights anneal from random to forward-tracking
  2. Contrastive credit: boost gradient components where consecutive steps agree,
     attenuate where they disagree

The phase transition finds the right basin through early exploration.
The contrastive filter removes batch-specific gradient noise.
Together: explore broadly, then exploit only the CONSISTENT credit signals.
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class PhaseContrastive(Rule):
    name = "phase_contrastive"
    autograd = "modified"
    blurb = "Phase-transition annealing + contrastive gradient filtering: explore broadly with noisy credit, then exploit only the signals that are consistent across steps."
    paper = "Novel combination: phase transition + contrastive gradient filtering."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    warmup_frac=0.05, schedule="cosine", mirror=True,
                    agreement_boost=1.3,
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "agreement_boost"

    def __init__(self, model, **hp):
        super().__init__(model, **hp)
        swap_linears(model, "random", self.hp["scale"])
        self.twins = [m for m in model.modules() if isinstance(m, FALinear)]
        self.opt = torch.optim.AdamW(model.parameters(),
                                     lr=self.hp["lr"],
                                     weight_decay=self.hp["weight_decay"])
        self.W = [m.weight for m in self.twins]
        self.B = [m.feedback for m in self.twins]
        self.B_random = [b.detach().clone() for b in self.B]
        self.params = [p for p in model.parameters() if p.requires_grad]
        self._prev_grads = None
        self.step_count = 0
        self.est_total_steps = 18000

    def _alpha(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        warmup = self.hp["warmup_frac"]
        if t < warmup:
            return 0.0
        t_eff = (t - warmup) / (1.0 - warmup) if warmup < 1.0 else 1.0
        return 0.5 * (1.0 - math.cos(math.pi * t_eff))

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        # Set backward weights
        alpha = self._alpha()
        with torch.no_grad():
            for w, b, br in zip(self.W, self.B, self.B_random):
                b.copy_((1.0 - alpha) * br + alpha * w)

        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.forwards += 1

        # Contrastive gradient filtering
        with torch.no_grad():
            current_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                             for p in self.params]
            if self._prev_grads is not None:
                boost = self.hp["agreement_boost"]
                for i, (g_curr, g_prev) in enumerate(zip(current_grads, self._prev_grads)):
                    agree = (g_curr.sign() == g_prev.sign()).float()
                    mask = agree * boost + (1.0 - agree) * (2.0 - boost)
                    self.params[i].grad = g_curr * mask
            self._prev_grads = current_grads
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Mirror update for random component
        if self.hp["mirror"]:
            with torch.no_grad():
                for w, br, bef in zip(self.W, self.B_random, before):
                    br.add_(w - bef)

        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
