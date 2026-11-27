"""PHASE SAM: phase transition annealing + sharpness-aware minimisation.

Combines two orthogonal ideas:
  1. Phase transition: backward weights anneal from random to forward-tracking
     (exploration→exploitation in credit assignment)
  2. SAM: each update uses the gradient at a perturbed point, steering toward
     wide minima (exploitation of landscape geometry)

The theory: phase transition finds the right BASIN (broad search via noisy
early credit), SAM finds the WIDEST minimum within that basin (gradient at
the sharpest point). Together they should find wide minima in good basins.

This is the first time these two ideas have been combined because they operate
on different axes: phase transition modifies WHERE the gradient comes from
(the backward weights), SAM modifies WHERE the gradient is evaluated (the
forward weights). They're orthogonal.
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class PhaseSAM(Rule):
    name = "phase_sam"
    autograd = "modified"
    blurb = "Phase transition annealing (exploration→exploitation in credit) combined with SAM (wide-minimum seeking): orthogonal axes of improvement."
    paper = "Novel combination: phase transition + Foret et al. 2021 (SAM)."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    warmup_frac=0.15, schedule="cosine", mirror=True,
                    rho=0.1,
                    teach=0.3, beta=16.0, order=9, count_steps=1500)
    tune = "rho"

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
        self.step_count = 0
        self.est_total_steps = 18000

    def _alpha(self):
        t = min(self.step_count / max(self.est_total_steps, 1), 1.0)
        warmup = self.hp["warmup_frac"]
        if t < warmup:
            return 0.0
        t_eff = (t - warmup) / (1.0 - warmup) if warmup < 1.0 else 1.0
        if self.hp["schedule"] == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * t_eff))
        return t_eff

    def _set_backward(self):
        """Set backward weights to interpolation."""
        alpha = self._alpha()
        with torch.no_grad():
            for w, b, br in zip(self.W, self.B, self.B_random):
                b.copy_((1.0 - alpha) * br + alpha * w)

    def _nudge(self, grads):
        """SAM: move weights rho uphill."""
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        scale = (self.hp["rho"] / (norm + 1e-12)).item()
        offsets = [g * scale for g in grads]
        for p, e in zip(self.params, offsets):
            p.add_(e)
        return offsets

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)

        self._set_backward()

        # SAM step 1: ascent (get gradient at current point, nudge uphill)
        rho = self.hp["rho"]
        e = None
        if rho > 0:
            self.opt.zero_grad(set_to_none=True)
            logits, loss0 = self.model(xa, ya)
            T = getattr(self, "teacher", None)
            if T is not None and T.ready:
                loss0 = teach(self, x0, y, logits, loss0)
            loss0.backward()
            with torch.no_grad():
                grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                         for p in self.params]
                e = self._nudge(grads)
            self.forwards += 1
            # Re-set backward weights at the perturbed point
            self._set_backward()

        # SAM step 2: descent (gradient at the perturbed point)
        self.opt.zero_grad(set_to_none=True)
        logits, loss = self.model(xa, ya)
        loss = teach(self, x0, y, logits, loss)
        loss.backward()

        with torch.no_grad():
            if e is not None:
                for p, offset in zip(self.params, e):
                    p.sub_(offset)
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Mirror update for the random component
        if self.hp["mirror"]:
            with torch.no_grad():
                for w, br, bef in zip(self.W, self.B_random, before):
                    br.add_(w - bef)

        self.forwards += 1
        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
