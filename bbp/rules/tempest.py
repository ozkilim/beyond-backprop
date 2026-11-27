"""TEMPEST: Temperature-Scheduled Ensemble of Perturbed Backward Weights.

The name, like this problem, comes from Shakespeare.

Combines three ideas that each independently help:
  1. MIRROR: backward weights track forward weights (nearly exact credit)
  2. NOISE: decaying perturbation on backward weights (exploration→exploitation)
  3. ENSEMBLE: at each step, sample N different noise realisations and average
     the resulting gradients (denoised credit signal)

The ensemble is cheap: each realisation perturbs the backward weights differently,
does one backward pass, and the gradients are averaged. No extra forward passes.

The noise starts large (σ₀) and decays to zero. Early training averages over many
noisy credit signals → implicit SWA-like regularisation. Late training → pure
mirror (≈backprop).

Why this might beat backprop: the averaged gradient from multiple noisy backward
passes is a DENOISED version of the true gradient. Where the gradient is consistent
across perturbations (generalising direction), it's preserved. Where it depends on
the specific backward path (overfitting direction), it's washed out.

This is fundamentally different from just adding noise to the gradient
(drift_diffusion), because the BACKWARD WEIGHTS are perturbed, not the gradient
itself. The backward weights determine the CREDIT ASSIGNMENT structure, so noising
them is like averaging over different credit hypotheses.
"""
import math

import torch

from . import Rule, register
from ._regularise import regularise
from ._teacher import add_dreams, teach
from .feedback_alignment import FALinear, swap_linears


@register
class Tempest(Rule):
    name = "tempest"
    autograd = "modified"
    blurb = "Ensemble of noisy backward-weight perturbations: average N different credit hypotheses per step, with temperature-scheduled noise. Named for Shakespeare's play (and this arena's test set)."
    paper = "Novel: backward-weight ensemble with temperature scheduling."
    defaults = dict(lr=1e-3, weight_decay=0.01, scale=1.0,
                    sigma0=0.5, n_ensemble=3, noise_schedule="cosine",
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
        return s0 * 0.5 * (1.0 + math.cos(math.pi * t))

    def step(self, x, y):
        self.step_count += 1
        x0, x = x, regularise(self, [self.opt], x)
        xa, ya = add_dreams(self, x, y)
        sigma = self._sigma()
        n_ens = int(self.hp["n_ensemble"])

        self.opt.zero_grad(set_to_none=True)

        if sigma < 1e-6 or n_ens <= 1:
            # No noise → standard mirror backward
            logits, loss = self.model(xa, ya)
            loss = teach(self, x0, y, logits, loss)
            loss.backward()
            self.forwards += 1
        else:
            # Ensemble: each member sets B, does forward+backward
            saved_B = [b.clone() for b in self.B]
            loss = None

            for k in range(n_ens):
                # Set B = W + noise_k BEFORE the forward pass
                with torch.no_grad():
                    for w, b in zip(self.W, self.B):
                        noise = torch.randn_like(b) * b.std() * sigma
                        b.copy_(w + noise)

                logits_k, loss_k = self.model(xa, ya)
                loss_k = teach(self, x0, y, logits_k, loss_k)
                # Scale so accumulated gradients average out
                (loss_k / n_ens).backward()
                self.forwards += 1
                if loss is None:
                    loss = loss_k.detach()

            # Restore clean B
            with torch.no_grad():
                for b, sb in zip(self.B, saved_B):
                    b.copy_(sb)

        with torch.no_grad():
            before = [w.clone() for w in self.W]

        self.opt.step()

        # Mirror update: B tracks W
        with torch.no_grad():
            lr_wd = self.opt.param_groups[0]["lr"] * self.hp["weight_decay"]
            for w, b, bef in zip(self.W, self.B, before):
                gap = b - bef
                b.add_(w - bef)
                b.add_(gap, alpha=-lr_wd)

        return loss.item()

    @torch.no_grad()
    def alignment(self):
        return sum(torch.cosine_similarity(m.weight.flatten(), m.feedback.flatten(), 0).item()
                   for m in self.twins) / len(self.twins)
